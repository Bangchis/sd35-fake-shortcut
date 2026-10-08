"""Reference kernels for the SD3.5 experiment; integrate with the recipient runtime."""

import math
from contextlib import contextmanager
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class Condition:
    tokens: torch.Tensor
    pooled: torch.Tensor

    def take(self, selection):
        return Condition(self.tokens[selection], self.pooled[selection])


def coefficients(value, x):
    return value.float().reshape(-1, *([1] * (x.ndim - 1)))


def batch_value(value, x):
    return torch.as_tensor(value, device=x.device, dtype=torch.float32).expand(
        x.shape[0]
    )


class StepConditioning(nn.Module):
    """Add a learned duration embedding without changing the pretrained d=0 path."""

    def __init__(self, original, width):
        super().__init__()
        self.original = original
        self.duration = nn.Sequential(
            nn.Linear(256, width), nn.SiLU(), nn.Linear(width, width)
        )
        nn.init.zeros_(self.duration[-1].weight)
        nn.init.zeros_(self.duration[-1].bias)
        self.active_duration = None

    @staticmethod
    def features(d):
        frequencies = torch.exp(
            -math.log(10000)
            * torch.arange(128, device=d.device, dtype=torch.float32)
            / 128
        )
        angles = d.float()[:, None] * 1000 * frequencies[None]
        return torch.cat((angles.sin(), angles.cos()), dim=-1)

    @contextmanager
    def use(self, d):
        previous = self.active_duration
        self.active_duration = d
        try:
            yield
        finally:
            self.active_duration = previous

    def forward(self, timestep, pooled_projection):
        result = self.original(timestep, pooled_projection)
        if self.active_duration is None:
            raise RuntimeError(
                "Use the SD35Field wrapper to provide duration conditioning"
            )
        d = self.active_duration
        # The duration branch contributes zero at d=0, including learned biases.
        extra = self.duration(self.features(d)) - self.duration(
            self.features(torch.zeros_like(d))
        )
        return result + extra.to(result.dtype)


class SD35Field(nn.Module):
    """SD3 velocity convention: sigma=1 noise, sigma=0 clean; step is x-d*v."""

    def __init__(self, transformer, *, shortcut, time_scale=1000):
        super().__init__()
        self.transformer = transformer
        self.shortcut = shortcut
        self.time_scale = time_scale
        if shortcut:
            transformer.time_text_embed = StepConditioning(
                transformer.time_text_embed, transformer.inner_dim
            )

    def forward(self, x, sigma, duration, condition):
        sigma = batch_value(sigma, x)
        duration = batch_value(duration, x)
        kwargs = {
            "hidden_states": x,
            "timestep": sigma * self.time_scale,
            "encoder_hidden_states": condition.tokens,
            "pooled_projections": condition.pooled,
            "return_dict": False,
        }
        if self.shortcut:
            with self.transformer.time_text_embed.use(duration):
                return self.transformer(**kwargs)[0].float()
        return self.transformer(**kwargs)[0].float()


class LoRALinear(nn.Module):
    """Frozen BF16 base projection + FP32 low-rank optimizer parameters."""

    def __init__(self, base, rank=96, alpha=96):
        super().__init__()
        self.base = base.requires_grad_(False)
        options = {"device": base.weight.device, "dtype": torch.float32}
        self.a = nn.Parameter(torch.empty(rank, base.in_features, **options))
        self.b = nn.Parameter(torch.zeros(base.out_features, rank, **options))
        nn.init.kaiming_uniform_(self.a, a=math.sqrt(5))
        self.scale = alpha / rank

    def forward(self, x):
        original = self.base(x)
        delta = F.linear(F.linear(x.float(), self.a), self.b)
        return original + (self.scale * delta).to(original.dtype)


def configure_fake_lora(field, *, rank=96, alpha=96):
    """Load the G snapshot first; create optimizers/DDP/EMA after this function."""
    if not field.shortcut:
        raise ValueError("F needs finite-duration conditioning")
    # Preserve learned G duration weights in FP32 while quantizing only the frozen base.
    duration_module = field.transformer.time_text_embed.duration
    duration_state = {
        name: value.detach().float().clone()
        for name, value in duration_module.state_dict().items()
    }
    field.requires_grad_(False).to(dtype=torch.bfloat16)
    names = []

    def replace(root, path, prefix):
        child = root.get_submodule(path)
        if not isinstance(child, nn.Linear):
            raise TypeError(f"Expected unfused Linear at {prefix}.{path}")
        parent_path, _, name = path.rpartition(".")
        parent = root.get_submodule(parent_path) if parent_path else root
        setattr(parent, name, LoRALinear(child, rank, alpha))
        names.append(prefix + "." + path)

    for index, block in enumerate(field.transformer.transformer_blocks):
        prefix = f"transformer.transformer_blocks.{index}"
        paths = [
            "attn.to_q",
            "attn.to_k",
            "attn.to_v",
            "attn.to_out.0",
            "attn.add_k_proj",
            "attn.add_v_proj",
        ]
        # Last MMDiT block discards context outputs: its added context query has
        # no useful output gradient. Keep that projection frozen, with no LoRA.
        if not block.context_pre_only:
            paths += ["attn.add_q_proj", "attn.to_add_out"]
        if block.use_dual_attention:
            paths += ["attn2.to_q", "attn2.to_k", "attn2.to_v", "attn2.to_out.0"]
        for root_name in ("ff", "ff_context"):
            feed_forward = getattr(block, root_name, None)
            if feed_forward is not None:
                paths += [
                    root_name + "." + name
                    for name, child in feed_forward.named_modules()
                    if isinstance(child, nn.Linear)
                ]
        for path in paths:
            replace(block, path, prefix)
    # No LoRA on AdaLN modulation, timestep/text projections, patch/output heads.
    if not names:
        raise ValueError("No SD3 transformer block Linear projections found")
    duration = field.transformer.time_text_embed.duration.float().requires_grad_(True)
    duration.load_state_dict(duration_state, strict=True)
    return names


class AdapterEMA:
    """EMA F trainables; reuse its immutable frozen base for no-grad calls."""

    def __init__(self, raw_task):
        self.model = raw_task
        self.shadow = {
            name: p.detach().float().clone()
            for name, p in raw_task.named_parameters()
            if p.requires_grad
        }

    @torch.no_grad()
    def update(self, decay):
        parameters = {
            name: p for name, p in self.model.named_parameters() if p.requires_grad
        }
        if parameters.keys() != self.shadow.keys():
            raise ValueError("F EMA trainable parameter set changed")
        for name, source in parameters.items():
            if source.dtype != torch.float32:
                raise ValueError("F trainables/optimizer/master must stay FP32")
            self.shadow[name].lerp_(source.detach(), 1 - decay)

    @torch.no_grad()
    def __call__(self, *args, **kwargs):
        from torch.func import functional_call

        # No dropout, mutable normalization statistics or concurrent use of raw_task.
        # Frozen base parameters remain identical, adapters/step embedding are replaced.
        return functional_call(self.model, self.shadow, args, kwargs, strict=False)


def one_step(model, noise, condition):
    # Exactly one conditional transformer evaluation; decoder is separate.
    return noise.float() - model(noise, 1.0, 1.0, condition)


@torch.no_grad()
def teacher_cfg(teacher, x, sigma, condition, negative, guidance):
    conditional = teacher(x, sigma, 0.0, condition)
    unconditional = teacher(x, sigma, 0.0, negative)
    return unconditional + guidance * (conditional - unconditional)


@torch.no_grad()
def teacher_rollout(teacher, noise, condition, negative, *, steps=32, guidance=4.5):
    if steps < 2 or steps & (steps - 1):
        raise ValueError("Use a power-of-two uniform-sigma teacher grid")
    x = noise.float()
    states = [x.cpu()]
    for index in range(steps):
        sigma = 1.0 - index / steps
        velocity = teacher_cfg(teacher, x, sigma, condition, negative, guidance)
        x = x - velocity / steps
        states.append(x.cpu())
    # Keep states FP32: subtracting nearby BF16 states corrupts small-duration targets.
    return torch.stack(states, dim=1)


def draw_dyadic(count, device, generator, *, levels=5):
    exponent = torch.randint(levels + 1, (count,), device=device, generator=generator)
    duration = 2.0 ** (-exponent.float())
    cells = 2**exponent
    slot = (torch.rand(count, device=device, generator=generator) * cells).long()
    sigma = (slot + 1).float() * duration
    return sigma, duration


@torch.no_grad()
def shortcut_target(
    target_model, x, sigma, duration, condition, *, minimum_duration=1 / 32
):
    half = duration / 2
    # At the finest parent level, approximate its two half steps with the local field.
    child_condition = torch.where(
        duration <= minimum_duration + 1e-7, torch.zeros_like(half), half
    )
    first = target_model(x, sigma, child_condition, condition)
    midpoint = x.float() - coefficients(half, x) * first
    second = target_model(midpoint, sigma - half, child_condition, condition)
    # The second input is the MODEL endpoint, not the straight-line interpolation.
    return ((first + second) / 2).detach()


@torch.no_grad()
def local_rollout(model, x, sigma, duration, condition, *, steps=32):
    """Diagnostic reference: many small Euler steps of the SAME raw local field."""
    sigma = batch_value(sigma, x)
    step = batch_value(duration, x) / steps
    state = x.float()
    for index in range(steps):
        velocity = model(state, sigma - index * step, 0.0, condition)
        state = state - coefficients(step, state) * velocity
    return state


def cached_teacher_batch(states, generator):
    """states [B,M+1,C,H,W], generated by teacher_rollout. Return x,sigma,d,target."""
    b, points = states.shape[:2]
    steps = points - 1
    if steps < 2 or steps & (steps - 1):
        raise ValueError("Cache grid must have a power-of-two step count")
    levels = int(math.log2(steps))
    exponent = torch.randint(levels, (b,), device=states.device, generator=generator)
    span = 2**exponent
    kind = torch.rand(b, device=states.device, generator=generator)
    full = kind < 0.25
    local = (kind >= 0.25) & (kind < 0.5)
    span = torch.where(full, steps, torch.where(local, 1, span))
    slot = (
        torch.rand(b, device=states.device, generator=generator) * (steps // span)
    ).long()
    start = slot * span
    row = torch.arange(b, device=states.device)
    x = states[row, start].float()
    endpoint = states[row, start + span].float()
    physical_duration = span.float() / steps
    target = (x - endpoint) / coefficients(physical_duration, x)
    sigma = 1.0 - start.float() / steps
    duration = torch.where(local, 0.0, physical_duration)
    # For d=0 the first Euler edge is the teacher's local velocity at the saved state.
    return x, sigma, duration, target.detach()


@torch.no_grad()
def fake_targets(
    g,
    ema_f,
    teacher,
    noise,
    independent_noise,
    condition,
    negative,
    generator,
    *,
    beta,
    guidance=4.5,
    levels=5,
):
    """3/4 local student-tracking CFM + 1/4 raw-F shortcut targets."""
    b = noise.shape[0]
    if b % 4:
        raise ValueError("Physical batch must be divisible by four")
    local_count = 3 * b // 4
    y = one_step(g, noise, condition).detach()
    sigma = torch.rand(b, device=y.device, generator=generator)
    duration = torch.zeros_like(sigma)
    sigma[local_count:], duration[local_count:] = draw_dyadic(
        b - local_count, y.device, generator, levels=levels
    )
    x = (1 - coefficients(sigma, y)) * y + coefficients(sigma, y) * independent_noise
    target = independent_noise.float() - y
    if beta > 0:
        selection = slice(0, local_count)
        anchor = teacher_cfg(
            teacher,
            x[selection],
            sigma[selection],
            condition.take(selection),
            negative.take(selection),
            guidance,
        )
        target[selection] = (target[selection] + beta * anchor) / (1 + beta)
    selection = slice(local_count, b)
    target[selection] = shortcut_target(
        ema_f,
        x[selection],
        sigma[selection],
        duration[selection],
        condition.take(selection),
        minimum_duration=2.0 ** (-levels),
    )
    return x.detach(), sigma, duration, target.detach()


def fake_loss(f, training_batch, condition):
    x, sigma, duration, target = training_batch
    prediction = f(x, sigma, duration, condition)
    local_count = 3 * x.shape[0] // 4
    loss = 0.5 * F.mse_loss(prediction, target)
    return loss, {
        "local_mse": F.mse_loss(
            prediction[:local_count], target[:local_count]
        ).detach(),
        "shortcut_mse": F.mse_loss(
            prediction[local_count:], target[local_count:]
        ).detach(),
    }


def dmd_proxy(
    generated, fake_velocity, teacher_velocity, noisy, sigma, *, beta, floor=1e-3
):
    with torch.no_grad():
        corrected = (1 + beta) * fake_velocity - beta * teacher_velocity
        teacher_clean = noisy - coefficients(sigma, noisy) * teacher_velocity
        dimensions = tuple(range(1, generated.ndim))
        normalization = (
            (generated.detach() - teacher_clean).abs().mean(dimensions).clamp_min(floor)
        )
        gradient = coefficients(sigma, generated) * (teacher_velocity - corrected)
        gradient = gradient / coefficients(normalization, generated)
    # Check finiteness collectively after every rank finishes forward.
    # A rank-local exception here could deadlock later FSDP collectives.
    proxy = 0.5 * F.mse_loss(generated, (generated - gradient).detach())
    return proxy, gradient.detach()


def generator_losses(
    g,
    f,
    target_model,
    teacher,
    noise,
    independent_noise,
    condition,
    negative,
    teacher_states,
    teacher_condition,
    generator,
    *,
    beta,
    guidance=4.5,
    levels=5,
    teacher_weight=0.25,
    shortcut_weight=0.25,
    dmd_weight=1.0,
):
    b = noise.shape[0]
    if b % 4:
        raise ValueError("Physical batch must be divisible by four")
    generated = one_step(g, noise, condition)
    sigma = 0.02 + 0.96 * torch.rand(b, device=noise.device, generator=generator)
    noisy = (1 - coefficients(sigma, generated)) * generated.detach()
    noisy = noisy + coefficients(sigma, generated) * independent_noise
    with torch.no_grad():
        fv = f(noisy, sigma, 0.0, condition)
        tv = teacher_cfg(teacher, noisy, sigma, condition, negative, guidance)
    direct, direction = dmd_proxy(generated, fv, tv, noisy, sigma, beta=beta)

    n = b // 4
    # Independent cached-anchor batch: do not restrict on-policy prompts to cache ids.
    # Runtime validates n rows and matching ids in teacher_states/teacher_condition.
    tx, ts, td, target = cached_teacher_batch(teacher_states, generator)
    teacher_loss = 0.5 * F.mse_loss(g(tx, ts, td, teacher_condition), target)

    selection = slice(b - n, b)
    bs, bd = draw_dyadic(n, noise.device, generator, levels=levels)
    by = generated[selection].detach()
    bx = (1 - coefficients(bs, by)) * by + coefficients(bs, by) * independent_noise[
        selection
    ]
    bc = condition.take(selection)
    target = shortcut_target(
        target_model, bx, bs, bd, bc, minimum_duration=2.0 ** (-levels)
    )
    bootstrap = 0.5 * F.mse_loss(g(bx.detach(), bs, bd, bc), target)
    total = (
        dmd_weight * direct
        + teacher_weight * teacher_loss
        + shortcut_weight * bootstrap
    )
    return total, {
        "direct": direct.detach(),
        "teacher_trajectory": teacher_loss.detach(),
        "shortcut": bootstrap.detach(),
        "dmd_direction_rms": direction.square().mean().sqrt(),
        "dmd_direction_finite": torch.isfinite(direction).all(),
        "generated_mean": generated.detach().mean(),
        "generated_std": generated.detach().std(),
    }


class FieldTask(nn.Module):
    """One outer FSDP forward owns all differentiable calls of an optimizer update."""

    def __init__(self, field):
        super().__init__()
        self.field = field

    def forward(
        self,
        x=None,
        sigma=None,
        duration=None,
        condition=None,
        *,
        operation="field",
        **kwargs,
    ):
        if operation == "field":
            return self.field(x, sigma, duration, condition)
        if operation == "g_warmup":
            x, sigma, duration, target = cached_teacher_batch(
                kwargs["teacher_states"], kwargs["generator"]
            )
            loss = 0.5 * F.mse_loss(self.field(x, sigma, duration, condition), target)
            return loss, {"teacher_trajectory": loss.detach()}
        if operation == "f_loss":
            return fake_loss(self.field, kwargs["training_batch"], condition)
        if operation == "g_loss":
            return generator_losses(self.field, condition=condition, **kwargs)
        raise ValueError(f"Unknown operation: {operation}")


@torch.no_grad()
def update_ema_shards(online, target, decay):
    """FSDP1 wrappers with matching shards, outside all forward contexts."""
    online_parameters = dict(online.named_parameters())
    target_parameters = dict(target.named_parameters())
    if online_parameters.keys() != target_parameters.keys():
        raise ValueError("EMA parameter names/layout mismatch")
    for name, source in online_parameters.items():
        destination = target_parameters[name]
        if source.dtype != torch.float32 or destination.dtype != torch.float32:
            raise ValueError(
                "EMA must update from persistent FP32 optimizer/master shards"
            )
        if source.shape != destination.shape or source.device != destination.device:
            raise ValueError("EMA shard shape/device mismatch")
        destination.lerp_(source.detach(), 1 - decay)


def wrap_fsdp(model, device, *, forward_prefetch=False):
    from torch.distributed.fsdp import (
        FullyShardedDataParallel,
        MixedPrecision,
        ShardingStrategy,
    )

    # G/EMA-G only, one whole-field unit with a uniform requires_grad flag.
    # F has mixed frozen/trainable tensors: use DDP for its trainable adapters.
    return FullyShardedDataParallel(
        model,
        device_id=device,
        sync_module_states=True,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        mixed_precision=MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
            buffer_dtype=torch.float32,
        ),
        use_orig_params=False,
        limit_all_gathers=True,
        forward_prefetch=forward_prefetch,
    )
