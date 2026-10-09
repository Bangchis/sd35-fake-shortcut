# SD3.5 Medium: student distill slow fake, shortcut và LoRA fade

Ngày: 2026-10-09. Scientific ID: `sd35_fake_shortcut_512_relaion_v3_slow_fake`.

**Một project mới độc lập**, native **512×512**, checkpoint SD3.5 Medium **base nhiều
bước** dạng Diffusers local, captions **reLAION đã có**, global batch **64**.
Đây là thiết kế và reference kernels để bên nhận triển khai runtime; chưa phải
trainer chạy được, checkpoint/result handoff hoặc kết quả đã verify trên H100.
Không copy/import bất kỳ code, launcher, config hay utilities từ project cũ.

## 1. Câu hỏi thí nghiệm và những điểm chốt

**G học từ F trong main phase. F học vận tốc tức thời trên phân phối ảnh một bước
của G, đổi chậm; sau khi ổn định, giảm LoRA dần để F trở về teacher.** Chỉ G có
conditioning độ dài shortcut; F không học shortcut và không có duration MLP.

- T: SD3.5 Medium base frozen, CFG 4.5. Sinh cache bằng **50 denoising steps**.
- G: full-weight SD3.5 + duration conditioning, LR **5e-6**, ảnh chính bằng
  **1 conditional NFE**, không external CFG. Few-step 2/4/8 chỉ diagnostic.
- F: **backbone T cố định** + LoRA rank/alpha **96**, LR **2.5e-5**;
  học **combined CFG field** với CFG 4.5, không phải học conditional velocity rồi
  áp CFG thêm sau đó. Khi LoRA strength λ=0, F bằng T_CFG 4.5 theo cùng numerical path.
- Main: không DMD, không teacher trajectory loss trực tiếp vào G. Teacher đi vào
  F qua target anchor nhỏ β≤0.01; teacher dùng riêng ở P0/cache/evaluation.
- F warm-up trên ảnh one-step của G trước. Sau đó F update **1 lần / 5 G updates**;
  EMA-F làm target. Đó là cadence chậm, chưa đảm bảo field drift nhỏ: phải đo.
- Target G refresh mỗi **5 successful G updates**, giữ detached để replay 5 lần.
  Giảm target-generation cost, không giảm số G forward/backward hoặc giấu staleness.

**Matched A/B**, chỉ đổi cách tạo target finite shortcut:

| Nhánh | Local 25% của G | Finite target 75% của G |
|---|---|---|
| A — self control | distill EMA-F velocity | EMA-G đi hai half-shortcuts |
| B — proposed bridge | distill EMA-F velocity | EMA-F đi các Euler bước nhỏ qua nửa đầu; EMA-G shortcut nửa cuối |

75% finite gồm **25% full d=1 + 50% shorter intervals**; toàn batch còn **25% d=0**.
Default microbatch chia hết cho4 ⇒ global 64 có **16 full / 16 local / 32 short**,
không chỉ xác suất kỳ vọng. A cũng có F local supervision, cùng warm-up, cadence,
β, fade, data, optimizer, EMA, target replay, successful G updates và evaluation.
A không phải reproduction Shortcut Models thuần. B dùng thêm F evaluations;
ngang updates không đồng nghĩa ngang compute, phải báo GPU-giờ/NFE thực tế.

**Fade:** giữ λ=1 ít nhất 200 successful main G updates, qua readiness gate;
sau đó **đóng băng cả online adapters và EMA-F snapshot**, giảm λ tuyến tính
1→0 trong **1.000 G updates**. Không để F tiếp tục tăng adapters bù λ giảm.
λ là scale của LoRA ở từng layer, **không phải tỷ lệ mixture teacher/student trong
output**; λ→0 bảo đảm về backbone teacher, nhưng khoảng cách field không nhất
thiết giảm đơn điệu. EMA-F cũng phải dùng đúng λ, không chỉ online F.

Pilot 300G/arm, nếu fade bắt đầu tại200, kết thúc ở λ=0.9. Pilot trả lời bridge
có tín hiệu sớm không và hệ thống có ổn không; **chưa kiểm chứng toàn bộ fade**.
Muốn test λ=0 cần tiếp tục cả A/B tới1200G với policy đã khóa. Không mặc định
200G là “đã hội tụ”; nếu gate chưa đạt, dừng screening và báo, không tự đổi lịch
chỉ một arm. Để tách riêng lợi ích fade, một thí nghiệm sau phải so B-fade với
B-hold λ=1, cùng freeze event; A/B này chỉ isolate bridge dưới cùng fade policy.

Logic cần nói rõ: khi F chỉ khớp phân phối hiện tại của G rồi G học lại F, hệ này
có thể chủ yếu giữ nguyên phân phối student. **Không có bảo đảm cải thiện ảnh chỉ
nhờ tracking/shortcut loss giảm.** Nguồn hướng về teacher là anchor nhỏ và nhất
là LoRA fade; phải đo one-step quality khi fade diễn ra. Trước fade, A/B chỉ kiểm
tra lợi ích finite target route, chưa chứng minh chất lượng vượt teacher/baseline.

Revision này thay v2/v3 draft cũ: bỏ F-shortcut, F backbone copy từ G, DMD và
teacher loss trực tiếp trong main; full target25%; teacher cache50; slow F và
fade có freeze. **Không resume state của thuật toán cũ** vào revision này.

## 2. Đã đọc setting FD-loss, nhưng không bê nguyên sang đây

Nguồn: [Representation Fréchet Loss, Appendix B.4 / Table B.3](https://arxiv.org/html/2604.28190v1#A2.SS4).
Setting SD3.5 của paper:

| Hạng mục | FD-loss SD3.5 Medium |
|---|---|
| Resolution / NFE / CFG | 256×256 / 1 / 1 |
| AdamW / betas / weight decay | `[0.9, 0.95]` / `0` |
| Peak LR / schedule | `1e-5` / cosine |
| LR warm-up / total updates | 2.500 / 15.000 |
| Global batch / precision | 1.024 / BF16 |
| Feature-space objective | SigLIP2 + Inception-v3 + MAE |
| Feature-stat EMA / initialization | `0.999` / 50k base samples |
| Weight EMA | không dùng |
| Reference data | BLIP3o-Pretrain-Long-3M hoặc BLIP3o-GPT4o-60k |

**Đó là feature-distribution post-training, không phải teacher trajectory
distillation.** Feature-stat EMA khác EMA weights; 50k samples không phải warm-up F.

Vì vậy config của ta dùng LR chủ repo chọn **G5e-6 / F2.5e-5**, AdamW `[0.9,0.999]`,
weight decay `0.01`, clip `10`, EMA weights riêng và warm-up F. Không dùng batch
1024, cosine 15k/2500, feature-stat estimator hay loss FD-SIM trong thử nghiệm này.
**512px áp dụng từ cache, warm-up, training đến generation/eval**, theo lựa chọn
chủ repo; không screening 256 rồi upscale. Đây là một khác biệt thêm so FD-loss,
phải ghi rõ khi so sánh. Chi phí thực tế cần profile trên H100.

FD-loss phần SD3.5 minh họa bằng Figure 7/G; không lấy FID ImageNet của paper làm
FID SD3.5 trên COCO. Muốn so với checkpoint FD-loss phải có đúng checkpoint đó và
đo lại cùng prompts/resolution/evaluator. Không dùng mốc SDXL 17.80 để tuyên bố
SD3.5-512 thắng paper.

## 3. Config đã chốt

Schema dưới đây cần runtime mới triển khai. Đường dẫn là placeholder phải map
trên máy bên nhận. Batch/GPU và bộ nhớ cần profile H100, chưa có số đo ở Mac.

```yaml
scientific_id: sd35_fake_shortcut_512_relaion_v3_slow_fake
backbone_path: /ABS/PATH/TO/SD35_MEDIUM_DIFFUSERS
backbone_kind: sd35_medium_base
local_files_only: true
resolution: 512
latent_channels: 16
prompts_source: local_relaion
prompts_path: /ABS/PATH/TO/RELAION_CAPTIONS
prompt_format: auto
prompt_text_column: caption  # map tên cột thật; không giả định dataset có cột này
training_prompt_limit: 8192
validation_prompt_count: 128
test_prompt_count: 512
prompt_split_seed: 10
text_max_sequence_length: 256
teacher_cfg: 4.5
fake_cfg: 4.5
teacher_grid: native_flowmatch_schedule
teacher_cache_steps: 50
teacher_cfg_arithmetic_dtype: float32
teacher_euler_state_dtype: float32
teacher_cache_train_count: 512
teacher_cache_validation_count: 128
student_external_cfg: 1.0
primary_inference_steps: 1
primary_evaluation_steps: 1
diagnostic_inference_steps: [2, 4, 8]
diagnostic_prompt_count: 16
minimum_shortcut_duration: 0.03125

generator_train_mode: full_weight
fake_train_mode: lora
fake_backbone_source: frozen_teacher
fake_field_kind: instantaneous_velocity
fake_step_embedding_trainable: false
fake_lora_rank: 96
fake_lora_alpha: 96
fake_lora_dropout: 0.0
fake_lora_targets: mmdit_joint_dual_attention_and_ff_excluding_unused_context_query
world_size: 4
per_device_batch_size: 8
gradient_accumulation_steps: 2
target_global_batch_size: 64
generator_distributed_strategy: fsdp1_whole_field_full_shard
generator_sync_each_microbatch: true
fake_distributed_strategy: ddp_trainable_adapters
cpu_offload: false
generator_persistent_parameter_dtype: float32_sharded
fake_frozen_base_dtype: bfloat16
fake_trainable_parameter_dtype: float32
forward_parameter_dtype: bfloat16
gradient_reduce_dtype: float32
optimizer_moment_dtype: float32
generator_ema_master_dtype: float32_sharded
fake_ema_master_dtype: float32_trainable_adapters_only
gradient_checkpointing: true

generator_lr: 5.0e-6
fake_lr: 2.5e-5
adam_betas: [0.9, 0.999]
weight_decay: 0.01
max_grad_norm: 10.0
lr_schedule: constant_with_linear_warmup
lr_warmup_updates: 20
# Successful optimizer updates, not microbatch/attempted-step counters:
g_bootstrap_updates: 300
f_warmup_updates: 200
f_warmup_max_updates: 400
experiment_generator_updates: 300
fake_update_interval_g_updates: 5
fake_bootstrap_fraction: 0.0
g_target_refresh_interval_g_updates: 5
g_teacher_full_step_probability: 0.25
g_teacher_local_probability: 0.25
g_bootstrap_full_step_probability: 0.25
g_main_local_probability: 0.25
g_main_teacher_loss_weight: 0.0
g_main_dmd_weight: 0.0
teacher_anchor_beta_max: 0.01
teacher_anchor_beta_ramp_g_updates: 100
generator_ema_decay: 0.99
fake_ema_decay: 0.99
fake_ema_reset_after_warmup: true
fake_lora_hold_min_g_updates: 200
fake_lora_fade_duration_g_updates: 1000
freeze_fake_optimizer_on_fade: true
freeze_fake_ema_on_fade: true
fade_requires_readiness_gate: true
fade_gate_policy: stop_if_not_ready  # không đổi lịch riêng từng arm
fade_probe_max_mse_ratio: 2.0
fade_probe_patience: 2

g_shortcut_target_source: ema_g_self  # B: ema_f_local_then_ema_g
fake_bridge_max_sigma_step: 0.0625
fake_bridge_min_substeps: 2
training_seed: 10
probe_seed: 12345
checkpoint_interval_g_updates: 150
checkpoint_keep_last: 1
probe_interval_updates: 50
fixed_sample_interval_g_updates: 150
fixed_sample_count: 8
tensorboard: true
performance_interval_updates: 10
metric_plugin: recipient_metrics
primary_metric: hps_v2_1
secondary_metrics: [clip_score, image_reward, hps_v3]
```

Batch default `8×4×2=64`; nếu profile đầy đủ cho thấy vừa, chọn chung A/B
`16×4×1=64` để bỏ accumulation. F CFG call có physical rows2B, cần đo cả F
backward/AdamW và G/EMA forwards, không chỉ inference. Nếu b8 OOM: `4×4×4=64`.
2GPU có thể profile `8×2×4=64`. Không đổi global 64. Activation checkpointing bật;
không giữ model/moments/EMA tất cả BF16 vì LR nhỏ và cần xác minh update thực.

G full-weight và F LoRA không cùng LR rule; đây là LR chủ project chọn để test.
F update thưa giảm compute và cadence; LR cao5 lần không chứng minh F chậm về
field distance. Log norm/relative delta online→EMA-F và EMA-F→T ở fixed states.
EMA-F sau warm-up reset shadow bằng online adapters đã qua gate, tránh target
bắt đầu main với EMA còn lag. Sau đó EMA decay0.99, freeze đồng thời khi fade.
P0 và P2 G dùng optimizer mới/warm20; F P1→P2 **giữ moments/counter/LR đã warmed**.
Không reset F optimizer ở fork; A/B copy nguyên F state, RNG và snapshot.

## 4. Convention và shortcut cho MMDiT

Dùng physical sigma: `sigma=1` noise, `sigma=0` clean;
`x_sigma=(1-sigma)*y + sigma*z`, local velocity target `v=z-y`;
Euler `x_next=x-d*v`. SD3 timestep được truyền là `1000*sigma` (xác nhận
num_train_timesteps local=1000); không bê DDPM indices/epsilon/scheduler SDXL.
G là **average velocity trên đoạn dài d**, không gọi instantaneous velocity d=1
là one-step shortcut. `one_step(G,z,c)=z-G(z,1,1,c)`.

Giữ nguyên SD3.5 config, dual attention, qk_norm, context stream và VAE geometry.
Thêm riêng duration MLP cho G vào `time_text_embed` trước các transformer blocks:
`temb_G=temb_native(t,c)+MLP(Fourier(d))-MLP(Fourier(0))`.
Last Linear zero-init, nên cả d>0 lúc init và d=0 luôn giữ pretrained embedding
path. Sau học d=0 vẫn dùng native time/text path với G weights đã train.
Duration khác timestep: mọi lời gọi G đều có sigma và d, không cộng d vào sigma.
MLP chạy trước block checkpointing; yêu cầu non-reentrant checkpointing theo
pinned Diffusers, kiểm tra gradient của nhánh này khi recipient smoke.

A finite target: gọi EMA-G trên hai nửa độ dài `d/2`, lần2 ở **model midpoint**,
lấy trung bình velocities. Với parent d=1/32, child sử dụng d=0 làm local boundary.
B finite target: tích phân **guided EMA-F instantaneous** qua `d/2` bằng Euler,
mỗi bước≤1/16 và ít nhất2 substeps; EMA-G dự đoán nửa còn lại từ midpoint đó.
Full d=1 cần8 F velocity evaluations +1 EMA-G tail; interval ngắn cần2..4 F calls.
Internal CFG gộp2B vào một field forward: báo cả denoising evaluations và
conditional branches, không gọi nó là một conditional NFE. Target detached.

G local25% distill trực tiếp guided EMA-F ở re-noised one-step G states; full25%
học endpoint từ noise; short50% học các d∈{1/2,1/4,1/8,1/16,1/32}. Cả A/B dùng
cùng loss `0.5*mean((G-target)^2)` trên64 rows, không còn auxiliary1/4 khác batch.
Chỉ G có finite duration head; F không bootstrap tự học và không nhận finite d.

F target: generate `y_G` với current online G, detach; lấy z' độc lập noise đã tạo
ảnh, `x=(1-sigma)*y_G+sigma*z'`. MSE supervised field là
`F_CFG(x,sigma,c)`, target `(z'-y_G + beta*T_CFG(x,sigma,c))/(1+beta)`.
Warm-up beta0. Main beta=`0.01*min(k_G/100,1)` trước freeze; teacher coefficient
max~0.99%. Đây là **coefficient**, không bảo đảm norm ảnh hưởng≤1% khi norms
khác nhau. Log teacher/student target RMS và mixing delta; không áp affine
“corrected F” khi rollout hoặc distill G. Main không chạy T trong G target/loss.

F native base clone từ T, không G-after-P0. Train combined F_CFG: nếu train
conditional F riêng rồi extrapolate CFG thì learned CFM target bị đổi nghĩa.
λ=0 đúng T_CFG chỉ khi base weights/config/dtype/negative prompt/CFG giống T.
LoRA scaling plain scalar phải checkpoint riêng và set vào **cả online và EMA-F**.
Freeze optimizer/EMA updates bằng schedule; không đổi parameter-name set giữa run.

## 5. Data và teacher cache 50 bước

Bên nhận đọc captions local, ghi format/schema/source hash, loại null/empty,
canonicalize NFKC + casefold + whitespace để deduplicate và kiểm tra split overlap.
Giữ original text cho encoding. Split deterministically bằng seed10:
train8192 unique, validation128, final test512; thiếu thì fail/ghi quyết định chung,
không fill duplicates hoặc lấy final test vào train. Cache teacher512 train prompts
và128 val prompts; các128 val dùng gates, không G optimization.

Text encoders frozen/cache rồi unload khỏi trainer: dùng native SD3 pipeline
`encode_prompt` cùng prompt cho CLIP-L/CLIP-G/T5, sequence length256; cache
positive tokens+pooled và negative empty string qua cùng encoders. Không tự nối
embeddings khác native pipeline. Hash tokenizer/text model/precision/negative policy.
Training prompts stream phân chia disjoint theo rank; lưu shuffle epoch/cursor/RNG.

Teacher cache dùng **native FlowMatchEulerDiscreteScheduler.set_timesteps(50)**,
mu theo image sequence ở512 khi `use_dynamic_shifting`, từ config local. Lưu
**51 states + actual 51 physical sigmas**, `timestep=1000*sigma`. Sigma đã shift;
không shift lần2 hoặc giả định grid uniform/dyadic. Assert starts1, ends0,
strict descending, no stochastic sampling/invert sigmas/unsupported scheduler.
Native50 là **setting teacher đã chốt**, không phải giới hạn cứng của model;
[model card chính thức](https://huggingface.co/stabilityai/stable-diffusion-3.5-medium)
minh họa40 và số bước configurable.

Code teacher rollout dùng native sigma grid và Euler update **tích lũy FP32**,
CFG arithmetic FP32 từ BF16 predictions. Cache FP32 thực, không cast mỗi Euler
state về BF16. Stock pipeline BF16 có rounding khác: GPU preflight kiểm tra
scheduler update trên cùng FP32 sample/output, rồi report endpoint difference với
stock pipeline50 từ cùng noise/CFG/encodings; không hứa bitwise parity. Không dùng clipping/rescale/skip-layer guidance hoặc
control adapters.
Cache states FP32 `[B,51,16,64,64]`, lưu chunk nhỏ. 640×51 latent states~7.97GiB,
chưa gồm text/metadata; không giữ toàn bộ cache GPU. BF16 subtraction hai states
cạnh nhau có thể phá small-d labels, không tiết kiệm bằng cách hạ states BF16.
Cache manifest lưu model/config/scheduler/sigma/noise/prompt hashes, code commit.

Full target d1 lấy exact cache noise→endpoint. Finite dyadic target nội suy
piecewise-linear **actual physical sigma** giữa native states rồi lấy average
velocity; đó là label từ cached Euler interpolant, không gọi exact continuous
teacher flow. Local d0 label lấy slope actual native edge ở cached vertex. Không
lấy binary index spans trên50-step cache. Teacher cache chỉ tạo một lần, không
50 teacher calls ở mỗi optimizer update.

## 6. Training phases, cadence và readiness

**P0 — G bootstrap:** G bắt đầu từ T native full weights + zero duration branch.
Train300 successfulG trên cache train512, loss mixed25%full/25%local/50%short.
G/EMA-G master FP32, BF16 forwards; không F/DMD. Validate cả online G và EMA-G native512 one-step trên
128 val; EMA-G phải qua gate vì dùng để eval/target: finite, no obvious collapse/blank; endpoint MSE cải thiện≥20% so initialized
G trên matched validation noise, primary image score không giảm>0.25 baseline
per-prompt std. Đây là heuristic cold-start gate, không theorem. Nếu P0 fail,
không tiếp tục giả là student one-step usable; báo và revise chung trước fork.

**P1 — F warm:** freeze current online G, F base T + fresh zero-B LoRA. Train200F
on-policy local CFM beta0, batch 64; G không update. Validate fixed held-out
renoising pairs từ one-step G: normalized F_CFG CFM MSE ≤0.8×native T_CFG MSE,
nonzero optimizer master delta, no nonfinite. Báo online+EMA F, field RMS,
relative T→F drift. Nếu chưa đạt, extend200 một lần (max400), không thay LR/target
âm thầm. Gate phải đạt với EMA-F used as target; nếu chỉ online đạt, reset EMA=online
và chạy lại val gate. Chốt shared init G/EMA-G/F/EMA-F/optimizers/RNG/probes;
reset EMA-F to validated online F, G P2 optimizer mới nhưng F optimizer giữ.

**P2 — từng arm A/B, k_G reset0:**

1. Boundary k_G=0,5,10,... trước fade: 1 F update successful theo fresh detached
   current G samples, β ramp; update EMA-F. F targets không cache/replay5 lần.
2. Set λ theo event schedule. Tạo global 64 G targets từ current G + EMA-G/EMA-F,
   detached, fixed prompt/noise IDs. A/B nguồn finite khác nhau, local giống nhau.
3. G train5 successful updates trên **chính64 cached target rows**; mỗi update
   loss64 và accumulation đúng. EMA-G update sau mỗi G optimizer success.
4. Boundary k_G=200: **trước** F update mới, audit fade gate. Nếu fail, save/stop
   arm; không claim final A/B đủ matched khi một arm thiếu steps. Nếu cả arm đạt,
   freeze online F và EMA-F; không optimizer/EMA-F update sau event; giảm λ theo
   k_G tại mỗi refresh boundary. Adapter snapshots giữ nguyên, không merge LoRA.
5. Pilot stop at 300 successfulG; lưu EMA-G export, matched final one-step test.
   Continue cả A/B tới1200 nếu cần kiểm chứng hoàn tất fade, đúng fixed policy.

Fade readiness tối thiểu: gates P0/P1 còn hiệu lực; tại150 và200 F tracking giữ
≤0.8×T CFM MSE trên matched current student probe states; current val primary
không giảm>0.25 baseline std so shared init và không tụt>0.1 baseline std từ150→200;
no nonfinite/skip trong50G gần nhất. Ngưỡng là screening heuristic phải khóa trước
fork; đo thêm target drift và G gradient/update norms. Không đủ evidence thì
**200 không được tự coi là “đủ lâu”**. Primary image metric đã khóa chưa usable thì chưa thể qua image gate.
Gates là validation-only; final test không dùng chọn lịch. Báo cả gate pass/fail
bởi khác biệt đó cũng là outcome, không lọc arm xấu khỏi báo cáo.

Sau freeze, λ(k)=max(0,1-(k-200)/1000), actual target strength giữ piecewise theo
refresh5G; k200=1,300=.9,700=.5,1200=0. Float strength ứng với target refresh k,
không recompute λ tại từng microbatch. F updates main ởk0..195:40, cộngP1;
P2 teacher anchor chỉ tại 40 F updates, không phải300G updates. Pilot G có60 target
refreshes (64×60=3840 new target rows,19200 row presentations), không gọi19200
unique training samples. P0/P1/shared init là chung overhead, báo riêng.

Mỗi refresh sau fade λ giảm0.005 cho5 G updates: target đứng yên trong block để
G thích nghi. Đây là lịch chậm đề xuất, chưa bảo đảm G theo kịp. Mỗi50G dựng fresh
validation targets theo đúng arm, đo G local/full/short MSE ngoài replay batches.
Nếu tổng MSE >2×max(mốc k200,1e-8) trong2 probes liên tiếp, hoặc primary image score giảm
>0.25 baseline std so shared init, save/stop và report adaptation failure. Không
âm thầm tăng LR, pause λ riêng một arm hay chọn checkpoint đẹp để che lag. Muốn
lịch chậm hơn thì revise chung protocol cho run mới từ saved common boundary.

## 7. Original reference kernels — cần integration trên máy bên nhận

Code sau là phương trình, model wrappers, LoRA, target builders và FSDP helper;
**không** có distributed loop, CLI, manifest/checkpointer, metric adapters hoặc
fade readiness implementation hoàn chỉnh. Hai target phases phải tách khỏi
backward phase để FSDP EMA-G không nằm trong G outer differentiable forward.
`main_phase_schedule` chứa cadence/β/λ default; runtime phải xử lý readiness event
trước khi gọi và checkpoint transaction state, không phải scheduler thay thế gates.
Mọi runtime checks trước collective phải thực hiện đồng bộ trên các ranks.

```python
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


def draw_dyadic(count, device, generator, *, levels=5, full_step_probability=None):
    if full_step_probability is None:
        # Uniform hierarchy when no full-interval probability is requested.
        exponent = torch.randint(
            levels + 1, (count,), device=device, generator=generator
        )
    else:
        # Explicit G policy: full noise-to-clean, or a shorter finite interval.
        exponent = torch.randint(
            1, levels + 1, (count,), device=device, generator=generator
        )
        full = torch.rand(count, device=device, generator=generator)
        exponent = torch.where(full < full_step_probability, 0, exponent)
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
def local_fake_bridge(
    ema_f,
    x,
    sigma,
    duration,
    condition,
    negative,
    *,
    guidance=4.5,
    max_step=1 / 16,
    min_substeps=2,
    enabled=None,
):
    """Integrate instantaneous EMA-F across the FIRST HALF of a G interval."""
    half = duration / 2
    counts = torch.ceil(half / max_step).long().clamp_min(min_substeps)
    if enabled is not None:
        counts = torch.where(enabled, counts, 0)
    step = half / counts.clamp_min(1).float()
    state = x.float().clone()
    # Native d<=1, so at most eight F evaluations per sample at max_step=1/16.
    bound = max(min_substeps, math.ceil(0.5 / max_step))
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
    minimum_duration=1 / 32,
    bridge_max_step=1 / 16,
    bridge_min_substeps=2,
    guidance=4.5,
    enabled=None,
):
    if source == "ema_g_self":
        return shortcut_target(
            ema_g, x, sigma, duration, condition, minimum_duration=minimum_duration
        ), torch.zeros_like(duration, dtype=torch.long)
    if source != "ema_f_local_then_ema_g":
        raise ValueError(f"Unsupported bootstrap target: {source}")
    # Runtime validates the source whitelist before collective forward.
    midpoint, counts = local_fake_bridge(
        ema_f,
        x,
        sigma,
        duration,
        condition,
        negative,
        guidance=guidance,
        max_step=bridge_max_step,
        min_substeps=bridge_min_substeps,
        enabled=enabled,
    )
    half = duration / 2
    child = torch.where(duration <= minimum_duration + 1e-7, 0.0, half)
    # Exactly one EMA-G tail call on EVERY rank, even with different local F counts.
    endpoint = midpoint - coefficients(half, midpoint) * ema_g(
        midpoint, sigma - half, child, condition
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
    levels=5,
):
    """Native 50-step states + sigma nodes -> local/finite/full G labels."""
    b, points = states.shape[:2]
    sigma, duration = draw_dyadic(
        b, states.device, generator, levels=levels, full_step_probability=0.0
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
    levels=5,
    full_probability=0.25,
    local_probability=0.25,
    guidance=4.5,
    bridge_max_step=1 / 16,
    bridge_min_substeps=2,
):
    """Refresh detached labels once per five G updates; no direct teacher call."""
    b = noise.shape[0]
    generated = one_step(g, noise, condition).detach()
    sigma, duration = draw_dyadic(
        b, noise.device, generator, levels=levels, full_step_probability=0.0
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
        minimum_duration=2.0 ** (-levels),
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
```

## 8. Runtime mới bên nhận cần triển khai

Tạo package `sd35_shortcut` mới từ Torch/Diffusers upstream và reference trên.
Dependency seed: Python 3.11, Torch 2.6.0 CUDA phù hợp máy, Diffusers 0.33.1,
Transformers 4.49.0, Sentencepiece 0.2.0, safetensors, TensorBoard, NumPy, PyYAML.
Lock toàn bộ resolved versions trên máy nhận; metric libraries hiện có dùng qua
adapter mới. Không cài Torch, import Torch hoặc chạy tests trên Mac authoring.

Modules cần có: config/schema, local asset inspection, prompt splits/encoding
cache, teacher cache, field adapters, distributed trainer, probes, logging,
checkpoint/export, metric adapters, paired evaluation. Các lệnh sau là **CLI
contracts cần triển khai**, hiện chưa tồn tại trainer/CLI chạy được:

```text
python -m sd35_shortcut inspect --config config.yaml
python -m sd35_shortcut prepare --config config.yaml
python -m sd35_shortcut cache-teacher --config config.yaml
torchrun --standalone --nproc_per_node=4 -m sd35_shortcut train --phase g-bootstrap --config config.yaml
torchrun --standalone --nproc_per_node=4 -m sd35_shortcut train --phase f-warmup --config config.yaml
torchrun --standalone --nproc_per_node=4 -m sd35_shortcut train --phase experiment --arm A --config config.yaml
torchrun --standalone --nproc_per_node=4 -m sd35_shortcut train --phase experiment --arm B --config config.yaml
python -m sd35_shortcut export --checkpoint RUN/last --weights ema-g --steps 1
python -m sd35_shortcut evaluate --manifest EVAL_MANIFEST.json
python -m sd35_shortcut compare --a A/scores.jsonl --b B/scores.jsonl
```

`inspect` chỉ đọc local metadata: xác nhận SD3.5 Medium base, 16 latent channels,
MMDiT/dual attention/qk_norm, patch size và time scale 1000; record model revision,
config và hashes. Require assets local, không lưu token trong config/log/manifest.
Text encoding và VAE/evaluator chạy riêng khỏi training loop. Cache 8192 T5
embeddings có thể tốn nhiều GiB; dùng disk/memmap và stream từng batch, không
load toàn bộ text cache lên GPU. Teacher states cũng đọc theo chunks.

Load T frozen BF16 eval. F là **native teacher clone độc lập**, inject FP32 LoRA.
G là native clone độc lập FP32, thêm StepConditioning; EMA-G độc lập FP32 cùng
structure. Assert không alias trainable tensors/storage G↔F↔T; F base hash bằng T;
F không có duration modules. Native unfused attention processor, LoRA discovery
fail rõ nếu khác kiến trúc pinned. Record actual target names/trainable count;
không bỏ context/dual branches âm thầm. Không adapter dropout/mutable statistics.

G và EMA-G: **FSDP1 whole FieldTask FULL_SHARD**, cùng shard layout/device;
optimizer G tạo sau wrap. Uniform requires_grad trong mỗi FSDP instance; EMA-G
all frozen. BF16 compute, FP32 persistent parameters/moments/EMA/reduction.
F không dùng FSDP helper vì mixed frozen/trainable: dùng DDP(FieldTask) và optimizer
chỉ FP32 LoRA. `find_unused_parameters=False` chỉ sau gradient audit; final unused
context query đã excluded. `broadcast_buffers=False` nếu không mutable stats.
F raw module dành no-grad calls/EMA, F DDP forward dành differentiable loss.

FSDP input casting bị tắt trong helper: **không cast target/sigma/loss inputs sang
BF16**. SD35Field tự cast hidden/text inputs theo compute weights; output/CFG/MSE
FP32. Dùng BF16 autocast cho model forward; ngoài autocast giữ target builder,
Euler state updates và loss statistics FP32. Audit actual tensor dtypes trên GPU.
Enable native non-reentrant block checkpointing; nhánh duration chạy trước block
checkpoint. Không recompute whole wrapper bằng reentrant checkpoint làm mất context.

Raw F và functional EMA-F không gọi concurrent. AdapterEMA thay chỉ LoRA tensors,
shares immutable base và scalar λ. Freeze bằng no optimizer step/no EMA update,
giữ parameter names/requires_grad set để shadow mapping ổn định. Sau fade không
backward F; λ=0 bỏ adapter branch. Set λ cho model trước cả online/EMA-F query.

## 9. Update transaction và distributed correctness

Trước phase/fork, broadcast resolved config/hash/decision. Mọi rank kiểm tra cùng
world size, batch64, microbatch chia hết4, source whitelist, cache geometry/sigmas,
parameter ownership và prompt cursors. Exception trên rank0 phải thông báo/abort
đồng bộ; không để ranks khác chờ collective.

Mỗi refresh dựng global64 targets bằng `student_target_batch` theo từng microbatch.
Builder chạy no_grad; online G và EMA-G FSDP calls **ngoài differentiable G loss**.
Teacher không truyền vào builder/loss. Cache x/sigma/d/label FP32 detached,
condition references, prompt IDs/noise hashes, generation k_G, λ, EMA revisions,
source, RNG và replay cursor. Cache64×2 latents chỉ~32MiB; text có thể stream CPU.
Replay cùng row order5 updates; A/B dùng riêng namespace nhưng matched prompt/
noise RNG streams. Không reuse target tensors giữa arms vì states khác nhau.

A cần2 EMA-G calls; B cần1 EMA-G tail và rank-local F loops. **Mọi rank gọi cùng
số FSDP collective forwards**, kể cả dummy finite inputs của local rows trước
khi overwrite bằng local F labels. Không branch FSDP call theo active IDs hoặc
số Euler loops khác nhau. F loops chỉ raw F/AdapterEMA, không collectives. Audit
min2/max8 counts và effective CFG branches; đếm discarded dummy EMA-G work vào
profiling. Local target là combined F_CFG, không conditional F hay double CFG.

Microbatch gradient gọi outer task:
`g_task(operation='g_loss', training_batch=batch, condition=cond)` hoặc
`f_task(operation='f_loss', training_batch=batch, condition=cond, negative=neg, guidance=4.5)`.
Không bypass differentiable G qua `.module`; một outer FieldTask sở hữu graph.
G loss một field call; F loss một combined2B call. Targets detached, G không có
backward qua F/T/EMA; F không có backward qua G.

G accum2: loss/2, **FSDP sync mỗi microbatch**, không G `no_sync` vì có thể giữ
full gradients gây OOM. F DDP có thể `no_sync` cả forward/backward ở microbatch
chưa cuối. Sau đủ64 rows: all-reduce finite decision, backward/skip cùng lịch;
audit gradient finite toàn ranks, G FSDP `.clip_grad_norm_` để norm shards đúng,
F replicated trainables dùng norm tương ứng. Step và EMA đúng một lần khi success.
Không rank-local raise/return trong lúc ranks khác đang forward/backward collectives.

F transaction tại k_G%5=0 trước fade. Nếu F update fail, retry/stop đồng bộ trước
khi tiến G; không coi đó là successful F update. G skip giữ target cache/replay
position, không advance counters; repeated nonfinite thì save failure và stop.
Schedules β/λ/LR/EMA/checkpoints dùng successful counters, attempted steps log riêng.
Resume phải nhớ F boundary đã hoàn tất hay chưa, tránh update F lần2.

G EMA update từ **FP32 online optimizer shards** ngoài forward, cùng layout;
không từ gathered BF16 weights. Log master nonzero delta và BF16-visible changed
fraction. F EMA FP32, reset validated online F sau P1 rồi shared fork. Scalar λ
không nằm trong state_dict mặc định: lưu/restore trước query F hoặc EMA-F.

Save giữa replay block phải lưu detached target batch, condition references và
cursor. Regenerate labels từ G/EMA hiện tại làm thay đổi experiment, không gọi
exact resume. Strict scientific ID/config/model/schema check; revision cũ fail.

## 10. Logging, TensorBoard và profile

Rank0 append/flush `metrics.jsonl` và TensorBoard từ cùng payload; rank-local
errors/cursors ghi riêng. Nonfinite JSON dùng null+finite flag, không invalid NaN.
Mỗi row có phase/arm/k_G/k_F/attempt/replay position/target revision/λ/β/freeze
state/walltime/GPU time/config hash/code commit.

Scalar groups cần có:

- Loss tổng; full/local/short MSE sums và row counts; F local MSE; LR G/F.
- Grad norm trước clip; master delta absolute/relative/nonzero; BF16-visible
  changed fraction; online→EMA drift; EMA-F→T relative field RMS.
- Latent mean/std/range/nonfinite; skips/reasons; actual64/16full/16local/32short.
- New target rows và replay row presentations; target age; F cadence;
  actual λ target strength; beta và teacher coefficient/RMS/mixing delta.
- Call counts T/F/G/EMA theo target generation/backward/probe; denoising
  evaluations và conditional branches. Teacher50 là50 CFG calls/100 branches;
  standalone G export là1 conditional transformer call.

Reduce sums/counts, không mean empty subset thành NaN. Missing metric ghi
unavailable, không 0. Không log F-shortcut loss vì F không học shortcut.

Profile every10G: target refresh (G endpoint, F bridge, EMA-G tail), transfer/cache
load, G forward/backward/reduction/AdamW, F forward/backward/AdamW, EMA, checkpoint/
probe I/O. Median/p90, peak allocated/reserved VRAM, GPU utilization và GPU-giờ.
CUDA events cho kernels, wall time cho I/O/collectives; không synchronize từng
layer để log. Báo target refresh amortized /5, F cost /5 và whole cycle cost;
không hứa tốc độ từ riêng G loss forward time.

Probes every50 updates trên validation/fixed seed: F local tracking so T, field
RMS distance, EMA-G composition residual, one-step teacher endpoint MSE và primary
image score. Fixed8 uncurated images at150/300; 2/4/8-step diagnostics16 val prompts
riêng. Final test không dùng trong training/tuning dashboards. Giữ probe scalars
và fixed hashes, không dump full activations/gradients hoặc vô số sample images.

Trước fork, kiểm tra F bridge midpoint maxstep1/16 vs1/32 trên16 val states cùng
λ: normalized RMS difference và finiteness. Nếu difference>0.05×finer-midpoint RMS,
flag coarse; refine chung trước fork hoặc stop. Ngưỡng heuristic, không proof
Euler exact. 2..8 F calls khác teacher50; không gọi bridge là exact teacher path.

## 11. Checkpoint, freeze và minimal retention

Shared P0/P1 init bảo vệ riêng. Mỗi arm giữ latest1 rolling checkpoint interval150G
và selected final EMA-G export; failure/last-good manifests bảo vệ. Atomic temp→
verify hashes→completion marker→replace; prune chỉ sau save hoàn tất. Không prune
assets/checkpoint của run khác. Không commit weights/data/logs/images/tokens lên GitHub.

Full resume lưu G/F/EMA-G/EMA-F, cả optimizer states FP32; phase/counters/boundary/
replay; per-rank CPU/CUDA RNG và prompt cursor; cached labels/references nếu replay
active; β/λ/fade event/gate history/frozen adapter hashes; architecture/duration/
LoRA targets/versions/world size. F base có thể rehydrate từ local T nếu strict
hash check, nhưng chỉ LoRA state_dict thiếu λ/freeze event thì chưa đủ resume.
A/B khởi đầu cùng shared init, G optimizer mới; F optimizer giữ nguyên. Arm resume
phải restore optimizer/counters, không warm-up/reset lại.

Recipient smoke save→restart→one update trên **4GPU strategy thật**, so continuous
run. Thêm shortened artificial fade smoke để kiểm tra freeze, λ0F=T_CFG, resume
λ/labels/EMA và standalone G export/reload. Shortened fade chỉ numerical smoke,
không trộn vào scientific A/B run. Record actual tolerances, không hứa bitwise
với nondeterministic kernels. Đổi world size tạo lineage mới, không exact resume.

Export inference G/EMA-G config+duration MLP+source hashes+weight choice;
strict loader dựng SD35Field. `one_step` rồi native VAE decode:
`vae.decode(latent/vae.config.scaling_factor + vae.config.shift_factor)`.
Native16 channels/64 spatial, không SDXL scale hoặc latent clipping. Export không
load F/T; không hidden rollout hoặc external CFG. Profile đúng1 conditional call;
metadata primary inference/eval1 và optional diagnostic2/4/8 support.

## 12. Evaluation: nhanh nhưng có ý nghĩa

Lock primary trước fork: HPSv2.1 nếu checkpoint/version/preprocessing/scale verified;
nếu ImageReward usable thì có thể chọn IR chung trước fork. Chỉ CLIP-S usable có
thể làm semantic screening, không claim aesthetic quality. Secondary missing ghi
unavailable. Dùng metric libraries đã có qua adapters mới và đầy đủ provenance.

Rows: T **50steps/CFG4.5**, shared init G **1NFE**, A300G **1NFE**, B300G **1NFE**,
native512 tất cả. T128 validation cached chỉ là val reference/gate, không so với
student test mean. Final students test512 held-out reLAION×2 fresh noise seeds =
1024 images/model, matched prompt IDs/noise hashes/decoder/resize/crop/evaluator.
Dùng **EMA-G** prechosen cho cả A/B; online G phụ, không pick theo arm/prompt.
Muốn teacher-vs-student final mean thì generate teacher trên đúng test pairs.
Không dùng final test để tune/gate/chọn checkpoint; test noise khác train/cache/probe.
Few-step báo riêng; B4-step>A1-step không chứng minh one-step bridge tốt hơn.

T reference trong project dùng CFG và Euler accumulation FP32 từ BF16 field
predictions, cùng cache policy §5. Native stock BF16 pipeline có rounding khác;
record numerical policy, optional đo stock50 baseline riêng. **Native sigma grid
không đồng nghĩa bitwise stock-pipeline parity.** Paper parity vẫn unverified cho
tới khi audit cùng sampler, precision, prompts, resolution và evaluator.

Positive early signal: B−A primary held-out metric positive paired95% CI, no clear
secondary/uncurated image regression, gates pass, actual one-step export. Loss/MSE
giảm không chứng minh ảnh đẹp. Pilot positive chưa kiểm chứng fade đến0 hoặc
seed robustness; pilot negative chưa bác bỏ toàn method. Báo GPU-giờ/target NFEs/
new target rows/skips; equal updates khác equal compute. Nếu cần compute-matched
comparison, thêm control A với budget compute tương đương, report riêng.

Paired CI theo prompt: trung bình2 noise scores rồi resample prompt IDs chung:

```python
import numpy as np


def paired_prompt_ci(scores_a, scores_b, *, seed=2026, draws=10000):
    # Each mapping: {(prompt_id, noise_seed): scalar_score}; higher is better.
    if scores_a.keys() != scores_b.keys() or not scores_a:
        raise ValueError("A/B must have exactly matched prompt/seed pairs")
    if draws < 1:
        raise ValueError("draws must be positive")
    grouped = {}
    for key in sorted(scores_a):
        delta = float(scores_b[key]) - float(scores_a[key])
        if not np.isfinite(delta):
            raise ValueError("Nonfinite scores must be resolved, not silently dropped")
        grouped.setdefault(key[0], []).append(delta)
    if any(len(values) != 2 for values in grouped.values()):
        raise ValueError("This screening protocol requires 2 noise seeds per prompt")
    delta = np.array([np.mean(values) for values in grouped.values()])
    rng = np.random.default_rng(seed)
    means = []
    for start in range(0, draws, 128):
        indices = rng.integers(0, len(delta), size=(min(128, draws-start), len(delta)))
        means.extend(delta[indices].mean(axis=1))
    low, high = np.quantile(means, [0.025, 0.975])
    return {"delta_B_minus_A": float(delta.mean()), "ci95": [float(low), float(high)],
            "prompts": len(delta), "bootstrap_seed": seed, "draws": draws}
```
CI chỉ đo uncertainty across prompts của hai checkpoints, không training seed
variance. Nếu có tín hiệu, repeat A/B main seed11 từ shared warm init với cùng
protocol; đó là replication conditional on warm init. End-to-end robustness cần
lặp P0/P1. Không chỉ test B rồi so số paper ở setting khác.

FID1024 chỉ exploratory, không so FID10K/50K. Sau screening, để so paper chạy cùng
COCO2014-val10K/native512 protocol cho A/B, T và baseline checkpoint thật nếu có.
Record reference/split/count/captions/seed/noise/resize/crop/feature checkpoint/
evaluator commit/aggregation. Paper khác resolution/data/CFG/prompts/evaluator
thì reported scores là context; re-evaluate checkpoint bằng common protocol mới
là empirical comparison. Không lấy FID SDXL17.80 hoặc FD-loss ImageNet FID làm mốc
SD3.5 COCO. Thiếu audit thì `paper_protocol_equivalence=unverified`.

## 13. Method checks phải làm trên GPU bên nhận

1. Native50 sigma grid, sign và scheduler Euler update với FP32 sample/model output;
   compare native pipeline theo declared precision/tolerance, report rounding
   differences; VAE scale/shift, actual51 nodes, local/full cache labels đúng.
2. Duration zero-init giữ pretrained prediction; d0 branch cancel sau training;
   duration gradients thật dưới non-reentrant checkpointing; G d1 usable.
3. F zero-B hoặc λ0 bằng T_CFG4.5; shrink frozen adapters về0 tiến tới T. Combined
   F_CFG backward finite, useful LoRA projections có gradients; F không duration.
   Master/moments/EMA FP32, cached targets không bị FSDP cast BF16.
4. Constant/linear velocity toy fields verify Euler sign, model midpoint, child
   local boundary, min/max substeps và B `(x-end)/d`; targets detached, independent
   re-noising noise. Không coi instantaneous F×d là finite shortcut.
5. Exact64/16/16/32; accumulation mean, sync/clip/step và gradient ownership đúng;
   same FSDP collective call order khi rank-local active masks khác nhau.
6. k0..195 có40 successful F updates; cache replay5G; β≤.01 chỉ F trước freeze;
   main G không teacher/DMD calls; F optimizer và EMA không đổi sau freeze.
7. λ tại200/300/700/1200 =1/.9/.5/0 nếu gate pass200; resume freeze/cache cursor không
   thêm F step; standalone export/reload G đúng1 conditional NFE, không load F/T.
8. Disjoint prompt/noise splits, matched per-sample metrics, unavailable metrics
   explicit, paired CI by prompt ID, không duplicate prompt leakage.

4H100 smoke ghi actual memory/timings. b8/accum2 là proposal, chưa verified fit.
Không chạy dài trước save/resume/export/cadence/freeze smoke. Mac authoring chỉ
static theo yêu cầu chủ project; không Torch imports, tests, training hoặc downloads.

## 14. Sources và report bên nhận

- [Representation Fréchet Loss v1, B.4/Table B.3](https://arxiv.org/html/2604.28190v1#A2.SS4),
  [official repo pinned](https://github.com/Jiawei-Yang/FD-Loss/tree/5c03b8112fec8b9432631e4ce053c0d918cc24bc).
  Config context; experiment này không implement FD loss.
- [Shortcut Models v3, §3](https://arxiv.org/html/2410.12557v3#S3),
  [official target code pinned](https://github.com/kvfrans/shortcut-models/blob/601004348667094e1b71f30942199759412d4432/targets_shortcut.py).
  Binary composition/duration/self-bootstrap background; slow F bridge/LoRA fade
  là thiết kế mới, không paper reproduction.
- [SD3.5 Medium official model card](https://huggingface.co/stabilityai/stable-diffusion-3.5-medium),
  Diffusers0.33.1 [transformer](https://github.com/huggingface/diffusers/blob/v0.33.1/src/diffusers/models/transformers/transformer_sd3.py),
  [attention](https://github.com/huggingface/diffusers/blob/v0.33.1/src/diffusers/models/attention.py),
  [pipeline](https://github.com/huggingface/diffusers/blob/v0.33.1/src/diffusers/pipelines/stable_diffusion_3/pipeline_stable_diffusion_3.py),
  [scheduler](https://github.com/huggingface/diffusers/blob/v0.33.1/src/diffusers/schedulers/scheduling_flow_match_euler_discrete.py),
  [checkpointing](https://github.com/huggingface/diffusers/blob/v0.33.1/src/diffusers/models/modeling_utils.py).
- PyTorch2.6 [FSDP API source](https://github.com/pytorch/pytorch/blob/v2.6.0/torch/distributed/fsdp/api.py)
  và [functional_call source](https://github.com/pytorch/pytorch/blob/v2.6.0/torch/_functorch/functional_call.py):
  kiểm tra mixed precision/input casting và EMA adapter query contract.
- Human latest decisions override old handoff: G distills F; F instantaneous,
  warm-up, slow tracking, small teacher anchor, gradual LoRA fade; native512,
  reLAION/global64, G5e-6/F2.5e-5/LoRA96/25%full, teacher50, independent new project.

Bên nhận report source/config/checkpoint hashes; fresh smoke/resume/export evidence;
P0/P1 gates; actual batch/VRAM/timing breakdown; kG/kF/λ/freeze event; F tracking;
init/A/B per-sample metrics+CI; uncurated one-step images; unique/replayed samples;
GPU-giờ; limitations và paper parity status. Commit scientific changes trước run,
ghi lineage ID, không báo convergence từ training loss riêng.

**Authoring status:** đọc primary sources, viết original code/document; Ruff/AST/
YAML arithmetic và consistency checks tĩnh. **Chưa chạy:** Torch imports, unit/toy/
GPU tests, numerical parity, distributed runtime, training, save/resume thật,
metrics hoặc benchmark. Runtime integration và GPU verification còn bên nhận làm.
