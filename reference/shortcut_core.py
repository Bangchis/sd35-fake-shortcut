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
        dtype = self.duration[0].weight.dtype
        extra = self.duration(self.features(d).to(dtype)) - self.duration(
            self.features(torch.zeros_like(d)).to(dtype)
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
        # Cast model inputs only. Keep sigma, cached targets and loss arithmetic FP32.
        dtype = self.transformer.pos_embed.proj.weight.dtype
        kwargs = {
            "hidden_states": x.to(dtype),
            "timestep": sigma * self.time_scale,
            "encoder_hidden_states": condition.tokens.to(dtype),
            "pooled_projections": condition.pooled.to(dtype),
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
        self.adapter_strength = 1.0

    def forward(self, x):
        original = self.base(x)
        if self.adapter_strength == 0.0:
            return original
        delta = F.linear(F.linear(x.float(), self.a), self.b)
        return original + (self.adapter_strength * self.scale * delta).to(
            original.dtype
        )


def configure_fake_lora(field, *, rank=96, alpha=96):
    """Frozen native TEACHER backbone + trainable LoRA; not a G snapshot."""
    if field.shortcut:
        raise ValueError(
            "F must be the native teacher field, without duration embedding"
        )
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
    return names


def set_fake_strength(raw_task, strength):
    """Call at target-refresh boundaries; checkpoint the scalar separately."""
    if not 0.0 <= strength <= 1.0:
        raise ValueError("LoRA strength must lie in [0,1]")
    for module in raw_task.modules():
        if isinstance(module, LoRALinear):
            module.adapter_strength = float(strength)


def fake_strength(k_g, *, fade_started_at_g=None, fade_duration=1000):
    if fade_duration <= 0:
        raise ValueError("Fade duration must be positive")
    if fade_started_at_g is None:
        return 1.0
    return max(0.0, 1.0 - max(0, k_g - fade_started_at_g) / fade_duration)


def main_phase_schedule(k_g, *, fade_started_at_g=None):
    """Apply a validated fade event BEFORE calling this at a boundary.

    Runtime owns readiness gates and transaction/replay state: a due F update
    must not execute twice after restart. Strength changes only at target refresh.
    """
    if not isinstance(k_g, int) or k_g < 0:
        raise ValueError("k_g must count successful nonnegative G updates")
    if fade_started_at_g is not None and (
        fade_started_at_g < 0 or fade_started_at_g > k_g or fade_started_at_g % 5
    ):
        raise ValueError("Fade event must be a completed target-refresh boundary")
    frozen = fade_started_at_g is not None
    refresh_k = k_g - k_g % 5
    return {
        "fake_update_due": not frozen and k_g % 5 == 0,
        "target_refresh_due": k_g % 5 == 0,
        "fake_frozen": frozen,
        "beta": 0.0 if frozen else 0.01 * min(k_g / 100, 1.0),
        "lora_strength": fake_strength(refresh_k, fade_started_at_g=fade_started_at_g),
    }


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
        # Frozen base parameters remain identical; only LoRA tensors are replaced.
        return functional_call(self.model, self.shadow, args, kwargs, strict=False)


def one_step(model, noise, condition):
    # Exactly one conditional transformer evaluation; decoder is separate.
    return noise.float() - model(noise, 1.0, 1.0, condition)


@torch.no_grad()
def sample_shortcuts(model, noise, condition, *, steps=1):
    """Optional 2/4/8-step sampling; the primary output remains one_step."""
    if steps not in (1, 2, 4, 8):
        raise ValueError("Supported evaluation NFEs: 1, 2, 4, 8")
    if steps == 1:
        return one_step(model, noise, condition)
    state = noise.float()
    duration = 1.0 / steps
    for index in range(steps):
        velocity = model(state, 1.0 - index * duration, duration, condition)
        state = state - duration * velocity
    return state


def guided_velocity(field, x, sigma, condition, negative, guidance):
    """Supervise F's COMBINED guided field, not conditional field before CFG."""
    combined = Condition(
        torch.cat((negative.tokens, condition.tokens)),
        torch.cat((negative.pooled, condition.pooled)),
    )
    predictions = field(
        torch.cat((x, x)), batch_value(sigma, x).repeat(2), 0.0, combined
    )
    unconditional, conditional = predictions.chunk(2)
    return unconditional + guidance * (conditional - unconditional)


@torch.no_grad()
def teacher_cfg(teacher, x, sigma, condition, negative, guidance):
    return guided_velocity(teacher, x, sigma, condition, negative, guidance)


@torch.no_grad()
def native_teacher_sigmas(scheduler, transformer, device, *, resolution=512, steps=50):
    """Construct the pinned SD3 pipeline's resolution-aware native Euler schedule."""
    from diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3 import (
        calculate_shift,
    )

    config = dict(scheduler.config)
    kwargs = {}
    if config.get("use_dynamic_shifting", False):
        side = resolution // 8 // transformer.config.patch_size
        kwargs["mu"] = calculate_shift(
            side * side,
            config.get("base_image_seq_len", 256),
            config.get("max_image_seq_len", 4096),
            config.get("base_shift", 0.5),
            config.get("max_shift", 1.16),
        )
    scheduler.set_timesteps(steps, device=device, **kwargs)
    return scheduler.sigmas.detach().to(device=device, dtype=torch.float32).clone()


@torch.no_grad()
def teacher_rollout(teacher, noise, condition, negative, sigmas, *, guidance=4.5):
    """Native FlowMatch Euler grid; validate 51 physical nodes for 50 steps."""
    x = noise.float()
    states = [x.cpu()]
    for index in range(len(sigmas) - 1):
        sigma = sigmas[index]
        velocity = teacher_cfg(teacher, x, sigma, condition, negative, guidance)
        x = x - (sigma - sigmas[index + 1]) * velocity
        states.append(x.cpu())
    # Keep states FP32: subtracting nearby BF16 states corrupts small-duration targets.
    return torch.stack(states, dim=1)


def draw_intervals(count, device, generator, *, minimum_duration=1 / 32):
    """Continuous shorter durations, so arbitrary EMA-G tail queries are trained."""
    if not 0 < minimum_duration < 1:
        raise ValueError("Minimum finite duration must lie in (0,1)")
    u = torch.rand(count, device=device, generator=generator)
    duration = torch.exp(math.log(minimum_duration) * (1 - u)).clamp(
        min=minimum_duration, max=1 - 1e-6
    )
    sigma = duration + (1 - duration) * torch.rand(
        count, device=device, generator=generator
    )
    return sigma, duration


def draw_split_fraction(count, device, generator, *, lower=0.25, upper=0.75):
    """Fraction of the parent interval before the join; shared policy in A/B."""
    if not 0 < lower <= upper < 1:
        raise ValueError("Split bounds must satisfy 0 < lower <= upper < 1")
    return lower + (upper - lower) * torch.rand(
        count, device=device, generator=generator
    )


@torch.no_grad()
def shortcut_target(
    target_model,
    x,
    sigma,
    duration,
    condition,
    *,
    split_fraction,
    minimum_duration=1 / 32,
):
    prefix = split_fraction * duration
    tail = duration - prefix
    # Only queries below the trained finite-duration floor use the d=0 local field.
    prefix_condition = torch.where(prefix < minimum_duration, 0.0, prefix)
    tail_condition = torch.where(tail < minimum_duration, 0.0, tail)
    first = target_model(x, sigma, prefix_condition, condition)
    join_state = x.float() - coefficients(prefix, x) * first
    second = target_model(join_state, sigma - prefix, tail_condition, condition)
    # Unequal lengths require a duration-weighted average, not (first+second)/2.
    weight = coefficients(split_fraction, x)
    return (weight * first + (1 - weight) * second).detach()


@torch.no_grad()
def local_fake_bridge(
    ema_f,
    x,
    sigma,
    prefix_duration,
    condition,
    negative,
    *,
    guidance=4.5,
    max_step=1 / 16,
    min_substeps=2,
    max_prefix_duration=0.75,
    enabled=None,
):
    """Integrate instantaneous EMA-F up to the sampled join, not a fixed midpoint."""
    if max_step <= 0 or min_substeps < 1 or not 0 < max_prefix_duration <= 1:
        raise ValueError("Invalid bridge solver bounds")
    counts = torch.ceil(prefix_duration / max_step).long().clamp_min(min_substeps)
    if enabled is not None:
        counts = torch.where(enabled, counts, 0)
    step = prefix_duration / counts.clamp_min(1).float()
    state = x.float().clone()
    # Runtime validates parent d<=1 and split<=the configured upper bound on all ranks.
    bound = max(min_substeps, math.ceil(max_prefix_duration / max_step))
    for index in range(bound):
        active = torch.nonzero(counts > index, as_tuple=True)[0]
        if active.numel() == 0:
            continue
        velocity = guided_velocity(
            ema_f,
            state[active],
            sigma[active] - index * step[active],
            condition.take(active),
            negative.take(active),
            guidance,
        )
        state[active] = (
            state[active] - coefficients(step[active], state[active]) * velocity
        )
    # Only raw F/AdapterEMA calls here: no DDP/FSDP collectives in rank-local loops.
    return state.detach(), counts.detach()


@torch.no_grad()
def generator_bootstrap_target(
    ema_g,
    ema_f,
    x,
    sigma,
    duration,
    condition,
    negative,
    *,
    source,
    split_fraction,
    minimum_duration=1 / 32,
    bridge_max_step=1 / 16,
    bridge_min_substeps=2,
    bridge_split_upper=0.75,
    guidance=4.5,
    enabled=None,
):
    if source == "ema_g_self":
        return shortcut_target(
            ema_g,
            x,
            sigma,
            duration,
            condition,
            split_fraction=split_fraction,
            minimum_duration=minimum_duration,
        ), torch.zeros_like(duration, dtype=torch.long)
    if source != "ema_f_local_then_ema_g":
        raise ValueError(f"Unsupported bootstrap target: {source}")
    prefix = split_fraction * duration
    tail = duration - prefix
    # Runtime validates source/geometry before collective forward.
    join_state, counts = local_fake_bridge(
        ema_f,
        x,
        sigma,
        prefix,
        condition,
        negative,
        guidance=guidance,
        max_step=bridge_max_step,
        min_substeps=bridge_min_substeps,
        max_prefix_duration=bridge_split_upper,
        enabled=enabled,
    )
    child = torch.where(tail < minimum_duration, 0.0, tail)
    # Exactly one EMA-G tail call on EVERY rank, regardless of local F counts.
    endpoint = join_state - coefficients(tail, join_state) * ema_g(
        join_state, sigma - prefix, child, condition
    )
    return ((x.float() - endpoint) / coefficients(duration, x)).detach(), counts


def interpolate_cached_states(states, sigmas, query):
    """FP32 piecewise-linear interpolation of a native, nonuniform teacher cache."""
    edge = torch.searchsorted(-sigmas.contiguous(), -query.contiguous(), right=True) - 1
    edge = edge.clamp(0, len(sigmas) - 2)
    row = torch.arange(states.shape[0], device=states.device)
    fraction = (sigmas[edge] - query) / (sigmas[edge] - sigmas[edge + 1])
    start, end = states[row, edge].float(), states[row, edge + 1].float()
    return start + coefficients(fraction, start) * (end - start)


def target_kinds(
    count, device, generator, *, full_probability=0.25, local_probability=0.25
):
    """Exact row counts per microbatch; runtime validates divisibility before collectives."""
    full_count = float(count * full_probability)
    local_count = float(count * local_probability)
    if (
        full_probability < 0
        or local_probability < 0
        or full_probability + local_probability > 1
        or not full_count.is_integer()
        or not local_count.is_integer()
    ):
        raise ValueError("Target fractions must produce exact integer row counts")
    order = torch.randperm(count, device=device, generator=generator)
    full = order < int(full_count)
    local = (order >= int(full_count)) & (order < int(full_count + local_count))
    return full, local


def cached_teacher_batch(
    states,
    sigmas,
    generator,
    *,
    full_step_probability=0.25,
    local_probability=0.25,
    minimum_duration=1 / 32,
):
    """Native 50-step states + sigma nodes -> local/finite/full G labels."""
    b, points = states.shape[:2]
    sigma, duration = draw_intervals(
        b, states.device, generator, minimum_duration=minimum_duration
    )
    full, local = target_kinds(
        b,
        states.device,
        generator,
        full_probability=full_step_probability,
        local_probability=local_probability,
    )
    sigma = torch.where(full, 1.0, sigma)
    duration = torch.where(full, 1.0, duration)
    x = interpolate_cached_states(states, sigmas, sigma)
    endpoint = interpolate_cached_states(states, sigmas, sigma - duration)
    target = (x - endpoint) / coefficients(duration, x)
    # Local labels use actual native vertices/edges, avoiding interpolated local states.
    edge = torch.randint(points - 1, (b,), device=states.device, generator=generator)
    row = torch.arange(b, device=states.device)
    local_x = states[row, edge].float()
    local_target = (local_x - states[row, edge + 1].float()) / coefficients(
        sigmas[edge] - sigmas[edge + 1], local_x
    )
    mask = local.reshape(-1, *([1] * (x.ndim - 1)))
    x, target = torch.where(mask, local_x, x), torch.where(mask, local_target, target)
    sigma = torch.where(local, sigmas[edge], sigma)
    duration = torch.where(local, 0.0, duration)
    return x, sigma, duration, target.detach()


@torch.no_grad()
def fake_targets(
    g,
    teacher,
    noise,
    independent_noise,
    condition,
    negative,
    generator,
    *,
    beta,
    guidance=4.5,
):
    """All samples train instantaneous F on the current one-step G distribution."""
    b = noise.shape[0]
    y = one_step(g, noise, condition).detach()
    sigma = torch.rand(b, device=y.device, generator=generator)
    duration = torch.zeros_like(sigma)
    x = (1 - coefficients(sigma, y)) * y + coefficients(sigma, y) * independent_noise
    target = independent_noise.float() - y
    if beta > 0:
        anchor = teacher_cfg(teacher, x, sigma, condition, negative, guidance)
        target = (target + beta * anchor) / (1 + beta)
    return x.detach(), sigma, duration, target.detach()


def fake_loss(f, training_batch, condition, negative, *, guidance=4.5):
    x, sigma, duration, target = training_batch
    prediction = guided_velocity(f, x, sigma, condition, negative, guidance)
    loss = 0.5 * F.mse_loss(prediction, target)
    return loss, {
        "local_mse": F.mse_loss(prediction, target).detach(),
    }


@torch.no_grad()
def student_target_batch(
    g,
    ema_g,
    ema_f,
    noise,
    independent_noise,
    condition,
    negative,
    generator,
    *,
    source,
    minimum_duration=1 / 32,
    split_lower=0.25,
    split_upper=0.75,
    full_probability=0.25,
    local_probability=0.25,
    guidance=4.5,
    bridge_max_step=1 / 16,
    bridge_min_substeps=2,
):
    """Refresh detached labels once per five G updates; no direct teacher call."""
    b = noise.shape[0]
    generated = one_step(g, noise, condition).detach()
    sigma, duration = draw_intervals(
        b, noise.device, generator, minimum_duration=minimum_duration
    )
    split_fraction = draw_split_fraction(
        b, noise.device, generator, lower=split_lower, upper=split_upper
    )
    full, local = target_kinds(
        b,
        noise.device,
        generator,
        full_probability=full_probability,
        local_probability=local_probability,
    )
    sigma, duration = torch.where(full, 1.0, sigma), torch.where(full, 1.0, duration)
    x = (1 - coefficients(sigma, generated)) * generated
    x = x + coefficients(sigma, generated) * independent_noise
    # Collective EMA-G calls cover ALL rows on every rank. Local rows use dummy
    # finite inputs until their labels are replaced below; never branch FSDP on ids.
    target, bridge_counts = generator_bootstrap_target(
        ema_g,
        ema_f,
        x,
        sigma,
        duration,
        condition,
        negative,
        source=source,
        minimum_duration=minimum_duration,
        split_fraction=split_fraction,
        bridge_split_upper=split_upper,
        bridge_max_step=bridge_max_step,
        bridge_min_substeps=bridge_min_substeps,
        guidance=guidance,
        enabled=~local,
    )
    ids = torch.nonzero(local, as_tuple=True)[0]
    local_sigma = torch.rand(ids.numel(), device=x.device, generator=generator)
    local_x = (1 - coefficients(local_sigma, generated[ids])) * generated[ids]
    local_x = (
        local_x + coefficients(local_sigma, generated[ids]) * independent_noise[ids]
    )
    # Raw AdapterEMA has no collectives. Exactly 25% local rows at the default b=8.
    if ids.numel():
        local_target = guided_velocity(
            ema_f,
            local_x,
            local_sigma,
            condition.take(ids),
            negative.take(ids),
            guidance,
        )
        x[ids], sigma[ids], duration[ids], target[ids] = (
            local_x,
            local_sigma,
            0.0,
            local_target,
        )
    return (x.detach(), sigma.detach(), duration.detach(), target.detach()), {
        "full_rows": full.sum(),
        "local_rows": local.sum(),
        "short_rows": (~full & ~local).sum(),
        "bridge_f_sample_evaluations": bridge_counts.sum(),
        # Store full row-wise fractions alongside cache labels for exact resume.
        "split_fraction": split_fraction.detach(),
        "finite_mask": (~local).detach(),
        "join_sigma": (sigma - split_fraction * duration).detach(),
        "generated_mean": generated.mean(),
        "generated_std": generated.std(),
    }


def student_loss(g, training_batch, condition):
    """One differentiable G field call against cached, detached mixed targets."""
    x, sigma, duration, target = training_batch
    prediction = g(x, sigma, duration, condition)
    row_mse = (prediction - target).square().flatten(1).mean(1)
    full, local = duration == 1.0, duration == 0.0
    statistics = {}
    for name, mask in (("full", full), ("local", local), ("short", ~full & ~local)):
        # Runtime all-reduces sums/counts; never average an empty subset.
        statistics[name + "_mse_sum"] = row_mse[mask].detach().sum()
        statistics[name + "_rows"] = mask.sum()
    return 0.5 * row_mse.mean(), statistics


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
                kwargs["teacher_states"],
                kwargs["teacher_sigmas"],
                kwargs["generator"],
                full_step_probability=kwargs.get("full_step_probability", 0.25),
                local_probability=kwargs.get("local_probability", 0.25),
                minimum_duration=kwargs.get("minimum_duration", 1 / 32),
            )
            loss = 0.5 * F.mse_loss(self.field(x, sigma, duration, condition), target)
            return loss, {
                "teacher_trajectory": loss.detach(),
                "teacher_full_step_fraction": (duration == 1.0).float().mean(),
            }
        if operation == "f_loss":
            return fake_loss(
                self.field,
                kwargs["training_batch"],
                condition,
                kwargs["negative"],
                guidance=kwargs.get("guidance", 4.5),
            )
        if operation == "g_loss":
            return student_loss(self.field, kwargs["training_batch"], condition)
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
            cast_forward_inputs=False,
            cast_root_forward_inputs=False,
        ),
        use_orig_params=False,
        limit_all_gathers=True,
        forward_prefetch=forward_prefetch,
    )
