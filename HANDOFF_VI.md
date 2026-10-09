# SD3.5 Medium → 1-step: shortcut generator + student-tracking shortcut fake

Ngày: 2026-10-09. Scientific ID: `sd35_fake_shortcut_512_relaion_v2_onestep`.

Đây là **bản bàn giao thiết kế + code lõi để bên nhận triển khai**, theo yêu cầu mới
của chủ repo. **Dự án mới độc lập `sd35-fake-shortcut`, package `sd35_shortcut`**:
viết mới toàn bộ, không import/copy/reuse code, launcher, config hay checkpoint
runtime của dự án SDXL cũ. Các entrypoint ở cuối file là contract cần viết,
không phải lệnh hiện đã tồn tại. Code
Python trong file đã được rà soát tĩnh, chưa import Torch/chạy tests/GPU/training.

Yêu cầu chốt: **native 512×512**, captions từ **reLAION local bên nhận đã có**,
checkpoint SD3.5 Medium **base nhiều bước, thư mục Diffusers** bên
nhận đã có; G inference một bước; G và F đều có step-size conditioning; F phải
warm-up để theo phân phối student trước khi cung cấp trajectory targets. Batch
hiệu dụng **64**. G LR **5e-6 full-weight**; F LR **2.5e-5 LoRA rank96** theo chốt
cuối của chủ repo. Metrics dùng implementation có sẵn bên nhận qua plugin mới;
không đưa code training cũ vào project này. File là design + reference kernels,
chưa phải trainer hoàn chỉnh hoặc bằng chứng phương pháp đã thành công.

## 1. Quyết định thí nghiệm và điều nó kiểm chứng

Giả thuyết cần kiểm tra: **khi generator đang thay đổi, dùng một fake field theo
phân phối generator và đã học các shortcut hữu hạn để tạo target cho G có tốt hơn
G tự tạo shortcut targets bằng EMA-G không?** Teacher ảnh hưởng F qua một local
anchor nhỏ; F ảnh hưởng G qua cả DMD và shortcut trajectory.

Chạy hai nhánh từ **cùng một checkpoint sau G bootstrap + F warm-up**:

| Nhánh | DMD vào G | F training | Target cho shortcut của G |
|---|---|---|---|
| A — self shortcut | corrected F + teacher, giống B | giống B | hai half-shortcuts của EMA-G |
| B — fake bridge | corrected F + teacher, giống A | giống A | hai half-shortcuts của EMA-F |

**Chỉ khác `g_shortcut_target_source: ema_g / ema_f`.** Cùng architecture, tham số
trainable, khởi tạo, cache, prompts, batch 64, LR, EMA decay, beta, loss weights,
số G/F updates và cách eval. Cả A lẫn B vẫn có F và DMD; không gọi A là bản tái lập
Shortcut Models thuần hoặc Decoupled DMD. Việc giữ F ở A làm công bằng cả nhánh DMD
lẫn chi phí; không so B với một baseline đã bỏ bớt losses hoặc train ít hơn.

Đây là kiểm tra nhân quả **hẹp nhưng rõ** của nguồn trajectory target. Nó không
tự tách hiệu quả riêng của beta. Nếu B có lợi, ablation kế tiếp mới là B với
`beta=0`; không mở sweep nhiều chiều ngay từ đầu. So với checkpoint khởi tạo cũng
được báo, nhưng không dùng riêng phép so đó để kết luận fake bridge hiệu quả.

### 1.1 One-step là mục tiêu chính, few-step là khả năng phụ

G phải hỗ trợ và được train trực tiếp để sinh ảnh bằng **1 conditional transformer
evaluation**: `y_G=z-S_G(z, sigma=1, d=1, c)`. F warm-up/tracking và DMD luôn dùng
endpoint one-step này. Không thay nó bằng endpoint2/4 steps trong training chính.

G teacher bootstrap: tỷ lệ lấy mẫu kỳ vọng **50% full interval d=1**, 25% local
d=0, 25% finite intervals ngắn hơn. Full teacher label chính là `z-y_teacher`,
nên velocity MSE ởd=1 tương đương endpoint MSE của ảnh one-step trong latent space.
Common teacher anchor của P2 giữ cùng phân phối này.

G shortcut bootstrap P2: **50% full interval (sigma=1,d=1)**, 50% shorter dyadic
intervals. Với full interval, input là fresh Gaussian noise; target gồm hai
half-shortcuts d=1/2 từ EMA-G ởA hoặc EMA-F ởB. G chỉ chạy **một** full shortcut
để khớp target hai bước đã detach. Không cần thêm differentiable forwards so với
loss hiện có; đổi tỷ lệ sampling để ưu tiên output one-step.

Inference2/4/8 steps dùng cùng G với d=1/N và sigma giảm1/N mỗi lần, không train
một generator khác. Đây là diagnostic trên16 validation prompts; primary held-out
test1024 images và paper-comparable export/eval dùng **1NFE**. Ảnh4 steps tốt mà
1 step còn noise/collapse thì **chưa đạt mục tiêu**, phải ghi rõ và sửa/đào tạo tiếp.
Cho phép gọi1 step không tự chứng minh1-step quality; gates/metrics phải đo nó.

Revisionv2 đổi sampling và acceptance contract của G theo yêu cầu ưu tiên one-step;
không đổi LR, batch, F warm-up/hierarchy hay nguồn target A/B. Cả hai nhánh phải
được fork lại từ shared initialization của cùng revision/config, không trộn v1/v2.

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

## 3. Config đã chọn cho screening

Các trường bên dưới là schema của project **mới cần triển khai**. Không phụ thuộc
schema hay package ở repo cũ. Đây là cấu hình đề xuất có lý do, chưa phải
setting đã chạy thành công hoặc bảo đảm đủ để có ảnh một bước tốt.

```yaml
scientific_id: sd35_fake_shortcut_512_relaion_v2_onestep
backbone_path: /ABS/PATH/TO/SD35_MEDIUM_DIFFUSERS
backbone_kind: sd35_medium_base
local_files_only: true
generator_train_mode: full_weight
fake_train_mode: lora
fake_lora_rank: 96
fake_lora_alpha: 96
fake_lora_dropout: 0.0
fake_lora_targets: mmdit_joint_dual_attention_and_ff_excluding_unused_context_query
fake_step_embedding_trainable: true
resolution: 512
latent_channels: 16
prompts_source: local_relaion
prompts_path: /ABS/PATH/TO/RELAION_CAPTIONS
prompt_format: auto  # explicit jsonl/parquet/csv khi format không xác định được
prompt_text_column: caption  # bên nhận map tên cột thật, không mặc định có cột này
training_prompt_limit: 8192
validation_prompt_count: 128
test_prompt_count: 512
prompt_split_seed: 10
text_max_sequence_length: 256
teacher_cfg: 4.5
teacher_grid: uniform_physical_sigma
teacher_cache_steps: 32
teacher_cache_train_count: 512
teacher_cache_validation_count: 128
minimum_shortcut_duration: 0.03125  # 1/32, không phải một DDPM index
student_external_cfg: 1.0
primary_inference_steps: 1
primary_evaluation_steps: 1
diagnostic_inference_steps: [2, 4, 8]
diagnostic_prompt_count: 16

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
lr_warmup_updates: 20  # mỗi phase bắt đầu optimizer mới
g_bootstrap_updates: 300
f_warmup_updates: 200
f_warmup_max_updates: 400  # gia hạn một lần nếu gate tracking chưa đạt
experiment_generator_updates: 300  # cho MỖI nhánh A/B
fake_updates_per_generator: 1
fake_bootstrap_fraction: 0.25
g_auxiliary_batch_fraction: 0.25
g_teacher_full_step_probability: 0.5
g_teacher_local_probability: 0.25
g_bootstrap_full_step_probability: 0.5
g_teacher_trajectory_weight: 0.25
g_shortcut_weight: 0.25
g_dmd_weight: 1.0
teacher_anchor_beta_max: 0.05
teacher_anchor_beta_ramp_g_updates: 100
generator_ema_decay: 0.99
fake_ema_decay: 0.99
g_shortcut_target_source: ema_g  # B chỉ đổi thành ema_f

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

Batch default ở512: `8 × 4 × 2 = 64`. Profile `16 × 4 × 1 = 64` và chỉ chọn nó
nếu toàn bộ P2, optimizer states và checkpoint/resume vừa VRAM với headroom; đổi
chung trước fork A/B. Nếu b8 OOM, dùng `4 × 4 × 4 = 64` chung cho hai nhánh.
Hai GPU thì profile `8 × 2 × 4 = 64`; không hứa batch32/GPU vừa bộ nhớ.
Không hạ global batch theo rank hoặc tự đổi sang 65/128/1024.
Accumulation chỉ là cách chia batch 64; giảm accumulation không giảm lượng mẫu
phải tính ở cùng batch. Physical batch phải chia hết cho 4 để có subset 3/4–1/4.

Lý do config: F:G=1:1, G sinh một bước, native512, frozen
text encoders được cache, teacher trajectories sinh một lần rồi tái dùng. F LoRA96
giảm phần backward/optimizer/EMA của critic; LR F cao hơn5 lần giúp tracking nhanh
hơn nhưng không chứng minh đã theo kịp G. Không đưa VAE, T5 hoặc preference
evaluator vào training step. Batch8/16 trên512 cần profile, không phải kết
quả đo. G FSDP giữ optimizer/master/EMA shards FP32 trên GPU, BF16 cho forwards;
F frozen base BF16 + adapter/step embedding/AdamW/EMA FP32 trên GPU. Tránh vòng copy
full FP32 CPU↔GPU mỗi step. Đây là **mixed precision**, không AdamW all-BF16.

`5e-6` và300G có thể chưa đủ để G đạt chất lượng một bước tốt. Gate bên dưới đo
điều đó thay vì mặc định thành công. Không tự sweep/đổi LR đã chốt trong A/B. Nếu
P0 không có learning signal, dừng báo lỗi/đề xuất calibration chung trước fork.
Kết quả âm tính cũng có thể đến từ capacity của F LoRA96; không suy ra mọi fake
field full-weight đều thất bại. G full vs F LoRA giống hệt giữa A/B, nên không gây
confound cho phép so nguồn target, nhưng giới hạn phạm vi kết luận.

## 4. Quy ước flow và shortcut bắt buộc

SD3 dùng physical sigma: `sigma=1` là noise, `sigma=0` là clean. Network teacher
dự đoán velocity theo chiều tăng sigma. Với clean latent y và noise z:

```text
h_sigma = (1-sigma)*y + sigma*z
v_local = z-y
h_(sigma-d) = h_sigma - d*S(h_sigma, sigma, d, c)
G_1step(z,c) = z-S_G(z,1,1,c)
```

`S(x,sigma,d,c)` với `d>0` là **average velocity trên đoạn hữu hạn**, không phải
instantaneous velocity được đổi tên. `d=0` là local field dùng cho F tracking/DMD.
G và F đều nhận sigma và d; teacher chỉ có sigma, không có shortcut d hữu hạn.

Với `h=d/2`, target shortcut từ R=EMA-G hoặc EMA-F:

```text
a = R(x, sigma, h, c)
x_mid = x-h*a
b = R(x_mid, sigma-h, h, c)
target = stopgrad((a+b)/2)
```

Ở parent `d=1/32`, query hai local fields `d_condition=0` tại hai physical half
steps 1/64. Các parent còn lại query đúng duration h. Lấy sigma từ các mốc phù hợp
với d, bảo đảm `0<=d<=sigma<=1`. Midpoint phải do model tạo ra, không lấy điểm
thẳng trên đường noise–clean. Không copy clipping [-4,4] của latent space khác
sang SD3.5 mà chưa kiểm tra VAE/latent scale.

Đây là adapter dựa trên nguyên lý step-size conditioning và binary composition
của [Shortcut Models](https://arxiv.org/html/2410.12557v3#S3); paper gốc dùng một
network với local-flow và self-bootstrap, còn ta có teacher, F, DMD và warm-up riêng.
Không gọi phương pháp mới là official Shortcut Models reproduction.

F teacher anchoring **chỉ ở d=0**:

```text
target_F_local = (z-y_G + beta*T_CFG(h_sigma,sigma,c))/(1+beta)
F_local_corrected = (1+beta)*F(h_sigma,sigma,0,c)-beta*T_CFG(h_sigma,sigma,c)
```

Warm-up F dùng beta=0. DMD dùng corrected **local** F, trajectory dùng raw
**finite** EMA-F. Không dùng affine correction trên F finite shortcut: correction
instantaneous không chứng minh correction finite-time ODE. Không áp external CFG
lên G hay EMA-F sau khi target teacher có guidance đã được học vào conditional path.

## 5. Data, teacher cache và conditioning

**Dùng reLAION local sẵn có**, chỉ đọc captions, không tải ảnh hoặc thay bằng bộ
caption khác. Bên nhận map cột text thực tế của JSONL/Parquet/CSV; tên reLAION không
đủ để tự đoán split, revision hay quality filter. Ghi local source fingerprint,
schema, text/id columns, original row ids, sampling order và filter policy.

Sau lọc caption rỗng/duplicate, chọn bằng seed10: 8192 training prompts, 128
validation prompts, 512 test prompts **không giao nhau**. Nếu nguồn không đủ
8832 unique prompts thì fail và báo count, không lấy lại validation/test vào train.
Canonical duplicate key: Unicode NFKC, collapse whitespace, casefold; giữ nguyên
caption gốc cho encoders. Nếu chuẩn bị benchmark COCO, loại overlap với final
COCO prompts trước khi chốt train pool. Không dùng test để tune hoặc chọn images.

Teacher cache chỉ chọn512 prompts **từ train pool8192**, mỗi prompt một seed noise,
cộng128 validation prompts. P0 dùng cache512; P1/P2 on-policy dùng cả train
pool8192 với fresh seeds. Common teacher anchor P2 lấy batch riêng từ cache512 và
conditioning riêng đúng ids; không ép mọi on-policy sample phải nằm trong cache.
8192/512 là budget screening đề xuất, không giả là toàn bộ reLAION. Tất cả A/B
dùng cùng danh sách/hash/prompt streams, không fork rồi tự chọn lại dữ liệu.

Cache teacher uniform **physical sigma** 32 Euler steps từ noise đến clean, CFG4.5.
Đây là solver ta chọn cho các labels dyadic; không gọi nó là scheduler mặc định
SD3.5. `scheduler.shift` không được áp lại vào sigma/timestep. Lưu scheduler config
gốc và lựa chọn `uniform_physical_sigma` rõ ràng. Nếu teacher cache ở512 cho ảnh
kém ngay từ đầu, sửa solver/budget chung trước khi distill; không dùng teacher kém
làm chuẩn rồi kết luận student tốt. Reference teacher official sampler có thể đo
riêng, với NFE/scheduler/CFG thực tế được ghi lại.

Tái dùng 32-step trajectories cho G bootstrap và common teacher anchor sau fork.
Cache `[M+1,16,H/8,W/8]` **FP32** trên disk, đọc batch theo rank. G input noise và
teacher clean phải cùng seed/coupling. Từ các states x_i,x_j:
`v_bar=(x_i-x_j)/(sigma_i-sigma_j)`. Local label d=0 lấy slope của edge đầu tiên,
đúng velocity teacher đã dùng trong Euler. Không finite-difference states BF16.

Ở512, latent64×64, phần states khoảng `640×33×16×64×64×4 ≈ 5.16 GiB`; embeddings và metadata
thêm dung lượng. Đây là tính số học của tensor states, không phải disk usage đo
thật. Không lưu teacher images mọi sample, chỉ decode nhóm kiểm tra nhỏ.

Conditioning dùng đủ CLIP-L, CLIP-G và T5 qua `StableDiffusion3Pipeline.encode_prompt`;
cache positive tokens/pooled và negative empty-prompt tokens/pooled. **Negative
không phải zero embeddings kiểu SDXL.** G/F chỉ nhận positive; T_CFG nhận cả hai.
Giữ T5 maximum length256, truncation policy và tokenizer/encoder hashes giống nhau.
Chuẩn bị cache text trong job riêng rồi giải phóng cả ba encoders; training chỉ
load transformer T/G/F/EMA và conditioning cache. VAE chỉ decode khi probe/eval.

Decode SD3: `vae.decode(latent/scaling_factor + shift_factor)`; lấy coefficients
từ VAE checkpoint. Cả teacher/G/F dùng latent `[B,16,64,64]` và decode native512.
Cache text cho cả train/val/test, còn teacher trajectories chỉ640 prompts. Negative
empty-prompt embeddings có thể lưu một bản rồi expand, không copy hàng nghìn lần.

Nguồn adapter: [SD3 Diffusers v0.33.1](https://github.com/huggingface/diffusers/blob/v0.33.1/src/diffusers/pipelines/stable_diffusion_3/pipeline_stable_diffusion_3.py),
[SD3 transformer v0.33.1](https://github.com/huggingface/diffusers/blob/v0.33.1/src/diffusers/models/transformers/transformer_sd3.py).
CFG4.5 là lựa chọn từ [ví dụ model card SD3.5 Medium](https://huggingface.co/stabilityai/stable-diffusion-3.5-medium),
không lấy CFG1 của FD-loss gán thành guidance teacher.

## 6. Ba phase và warm-up F theo student

### P0 — G học shortcut teacher, trước khi tạo đối chứng

Từ base pretrained T, tạo G độc lập và thêm embedding duration có zero output ở
khởi tạo. Optimizer G mới. Train300 successful G updates bằng labels teacher cache:
kỳ vọng25% d=0, 50% d=1, 25% dyadic finite khác; chọn sigma đúng grid mỗi sample. T/F không
train; P0 chưa dùng DMD hay cross-fake targets. Local teacher CFG và finite average
velocity được học vào G conditional, nên inference không cần external CFG. EMA-G
khởi tạo từ G và update từ FP32 shards sau từng successful G step trong P0.

Validation cố định: endpoint MSE so teacher ở 1-step trên128 prompts;
endpoint errors ở2/4/8 steps chỉ trên16 validation prompts để giảm chi phí,
ảnh 1-step/teacher, latent mean/std, số nonfinite, global grad norm và master delta.
G phải có learning signal trên validation, không chỉ train loss. Nếu 300 updates
vẫn không cải thiện **1-step** endpoint error và ảnh1-step còn noise/collapse,
chưa chạy A/B dù ảnhfew-step tốt. Dừng
để debug dấu/sigma/duration/grad/conditioning hoặc calibration LR chung. Không
chuyển sang DMD2/distilled checkpoint để che lỗi cold start.

### P1 — F warm-up trên phân phối G đã bootstrap

Snapshot online G sau P0 để khởi tạo **F độc lập**, gồm backbone và duration
embedding; không chia sẻ Parameter/storage với G. Freeze backbone F ở BF16, thêm
LoRA96 zero-B lên joint/dual attention và FF projections đã chọn; duration MLP
train FP32. Không LoRA hóa mọi Linear một cách mù quáng; xem §7/§8.
Copy weights chỉ là initialization. F chỉ update LoRA và duration MLP, không update
frozen backbone, original time/text projection, patch Conv2d hay output projection.
Giữ G cố định về weights, không step optimizer G và không update EMA-G. Sinh fresh
noise, lấy `y_G=G_1step(z,c).detach()`, rồi dùng **noise độc lập** để tạo h_sigma.

Train200 successful F updates: 3/4 batch local CFM target `z_new-y_G`; 1/4 batch
raw EMA-F binary shortcut target. beta=0 toàn bộ phase; không teacher-output bias,
không DMD vào G. EMA-F chỉ lưu FP32 trainable LoRA/duration tensors và dùng lại
frozen base bất biến của F; update sau mỗi successful F step. Với checkpointing,
frozen G forward vẫn ở no_grad; không cần đổi requires_grad của FlatParameter
FSDP sau khi wrap. Ở parent nhỏ nhất EMA-F local field là base case.

Probe tracking trước/sau warm-up trên 128 **held-out** prompts của G và noise cố
định: local velocity MSE / target energy, buckets sigma0.1/0.3/0.5/0.7/0.9, và
finite shortcut composition residual. CFM có variance không triệt tiêu nên không
đặt mục tiêu loss=0. Gate đề xuất: normalized held-out local MSE giảm ít nhất10%
so F vừa copy từ G, không nonfinite/collapse, finite targets có RMS hợp lý. 10% là
heuristic screening, không phải định lý F đã khớp phân phối. Nếu chưa đạt, gia hạn
warm-up thêm200F một lần; vẫn không đạt thì dừng và báo thiếu tracking trước A/B.

Kiểm tra finite field trên16 validation prompts: cùng start state, so một
EMA-F shortcut d=1/1⁄2/1⁄4 với32 local Euler substeps của chính EMA-F(d=0),
ghi normalized endpoint error và latent RMS trước/sau warm-up. Không ép endpoint
F từ một noise phải trùng `G(noise)`: F local học marginal bằng independent
re-noising, không học đúng pairing của generator. Self-composition residual nhỏ
một mình không chứng minh F đã theo student; cần cả local tracking và images.

Gate không được chỉ nhìn warm-up train loss; không dùng beta>0 để làm target dễ hơn.
F vừa học student local distribution vừa học finite shortcuts; không chỉ thêm MSE
teacher local rồi gọi là trajectory distillation.

### P2 — fork checkpoint chung, A/B mỗi nhánh300G

Lưu G/F, EMA-G/EMA-F, config, cache/provenance và tracking report. Đây là **shared
initialization**. Cả A/B load đúng cùng hashes, tạo optimizer G/F mới, reset
experiment counters k_G=k_F=0, cùng seed và prompt/noise streams. Lưu EMA-G P0
không đổi xuyên P1; giữ EMA-F đã warm-up, không reset EMA-F về F/teacher khác.

Trong mỗi cycle: một successful F update trước, một successful G update sau.
`beta=0.05*min(k_G/100,1)`, giữ nguyên beta xuyên F và G của cycle. F cập nhật
3/4 mixed local tracking + 1/4 EMA-F shortcut. EMA-F update sau F success.

G loss cho cả A/B:

```text
L_G = L_DMD + 0.25*L_teacher_trajectory + 0.25*L_bootstrap
```

L_DMD: sinh ảnh một bước từ fresh noise (giữ graph), re-noise tại sigma trong
[0.02,0.98], direction detached từ frozen teacher CFG và corrected F local.
L_teacher_trajectory: batch phụ bằng1/4 physical batch, lấy riêng từ cache512
cùng conditioning đúng cache ids, giống P0 (50% full d=1).
L_bootstrap: subset1/4 batch, 50% full(sigma=1,d=1) và50% shorter intervals,
trên h_sigma re-noised từ current
G endpoint; target bằng hai raw half-shortcuts của **EMA-G ở A / EMA-F ở B**.
Target, midpoint và generated endpoint dùng làm input phụ đều detach. Chỉ G nhận
gradient từ G loss; chỉ F nhận gradient từ F loss.

G direct/DMD dùng **100% batch endpoints từ one-step G**. Teacher anchor và
bootstrap ưu tiên full interval như trên; không chỉ train local/few-step rồi hy
vọng G tự suy ra một bước. Các tỷ lệ là sampling probabilities, không bảo đảm
đúng50% trong từng microbatch nhỏ; runtime log counts rồi aggregate qua ranks/windows.

G direct dùng guided DMD trên một noisy state/time, không bê lịch 4 anchors SDXL.
Ở one-step này dùng coupled direct estimator để giảm teacher calls; không claim
đã reproduce Decoupled CA/DM independent schedules. Correction chỉ phục vụ F local.

Flow DMD surrogate cụ thể: `x0_T=x-sigma*T_CFG`,
`x0_F=x-sigma*F_corrected`, direction `(x0_F-x0_T)` chia theo per-sample
`mean(abs(y_G-x0_T)).clamp_min(1e-3)`. G proxy là MSE tới
`stopgrad(y_G-direction)`. Đây là lựa chọn clean-prediction DMD-style weighting
chung A/B, không khẳng định là unbiased KL gradient với mọi flow time weighting.
Theo dõi direction RMS/clip rate, không diễn giải số proxy loss là quality metric.

EMA-G update từ FP32 optimizer/master shards sau G success; EMA-F update từ FP32
LoRA/duration optimizer tensors sau F success, không từ full BF16 outputs/weights.
EMA-F và EMA-G cùng
decay0.99, cùng1:1 update ratio trong P2 để nguồn target không bị confound bởi decay
khác. Update counters là successful optimizer steps, không microbatches; cả A/B
phải ghi số skipped attempts và tổng samples để nhận biết budget thực tế khác.

## 7. Code lõi mới để tách thành `src/sd35_shortcut/shortcut_core.py`

Code dưới đây viết mới cho design này, không lấy code từ dự án training cũ.
Torch/Diffusers upstream là dependencies; paper/official implementations là nguồn
đối chiếu nguyên lý. Bên nhận viết data loader, trainer, checkpoint, collective
finite checks và logger mới trong project độc lập. **Đây chưa phải trainer CLI
hoàn chỉnh.** Các inputs/kwargs phải được runtime validate trên mọi rank trước
forward; không dùng code mẫu như một vòng training đã được kiểm nghiệm.

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


def draw_dyadic(count, device, generator, *, levels=5, full_step_probability=None):
    if full_step_probability is None:
        # F retains its uniform hierarchy, including d=1.
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
def local_rollout(model, x, sigma, duration, condition, *, steps=32):
    """Diagnostic reference: many small Euler steps of the SAME raw local field."""
    sigma = batch_value(sigma, x)
    step = batch_value(duration, x) / steps
    state = x.float()
    for index in range(steps):
        velocity = model(state, sigma - index * step, 0.0, condition)
        state = state - coefficients(step, state) * velocity
    return state


def cached_teacher_batch(
    states, generator, *, full_step_probability=0.5, local_probability=0.25
):
    """states [B,M+1,C,H,W], generated by teacher_rollout. Return x,sigma,d,target."""
    b, points = states.shape[:2]
    steps = points - 1
    if steps < 2 or steps & (steps - 1):
        raise ValueError("Cache grid must have a power-of-two step count")
    levels = int(math.log2(steps))
    exponent = torch.randint(levels, (b,), device=states.device, generator=generator)
    span = 2**exponent
    kind = torch.rand(b, device=states.device, generator=generator)
    full = kind < full_step_probability
    local = (kind >= full_step_probability) & (
        kind < full_step_probability + local_probability
    )
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
    teacher_full_step_probability=0.5,
    teacher_local_probability=0.25,
    bootstrap_full_step_probability=0.5,
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
    tx, ts, td, target = cached_teacher_batch(
        teacher_states,
        generator,
        full_step_probability=teacher_full_step_probability,
        local_probability=teacher_local_probability,
    )
    teacher_loss = 0.5 * F.mse_loss(g(tx, ts, td, teacher_condition), target)

    selection = slice(b - n, b)
    bs, bd = draw_dyadic(
        n,
        noise.device,
        generator,
        levels=levels,
        full_step_probability=bootstrap_full_step_probability,
    )
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
        "teacher_full_step_fraction": (td == 1.0).float().mean().detach(),
        "bootstrap_full_step_fraction": (bd == 1.0).float().mean().detach(),
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
                kwargs["teacher_states"],
                kwargs["generator"],
                full_step_probability=kwargs.get("full_step_probability", 0.5),
                local_probability=kwargs.get("local_probability", 0.25),
            )
            loss = 0.5 * F.mse_loss(self.field(x, sigma, duration, condition), target)
            return loss, {
                "teacher_trajectory": loss.detach(),
                "teacher_full_step_fraction": (duration == 1.0).float().mean(),
            }
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
```

## 8. Nối model, FSDP/DDP và optimizer đúng ownership

Tạo **repo/project độc lập** `sd35-fake-shortcut`, package `src/sd35_shortcut/`;
viết mới `config.py`, `backend.py`, `prompts.py`, `cache.py`, `trainer.py`,
`checkpoint.py`, `logging.py`, `evaluation.py`, `cli.py`. Không import/copy bất kỳ
code, utilities, tests, launcher hoặc configs của dự án training cũ. Metrics
implementation có sẵn bên nhận có thể dùng qua interface plugin mới, ghi provenance.
Metadata phải ghi model config thật, đặc biệt channels/dual attention; không tự
downgrade thành SD3 Medium cũ nếu checkpoint SD3.5 không load strict.

Pin Torch2.6.0, Diffusers0.33.1, Transformers4.49.0 cho API code mẫu này; text tokenizer T5
cần `sentencepiece==0.2.0`. Ghi môi trường thực tế bên nhận và xác nhận API adapter.
Không upgrade Diffusers tùy tiện trước khi kiểm tra `time_text_embed` signature,
activation checkpoint và weight layout. Dùng weights local sẵn có, không tải lại.

Pseudo initialization (cần nối IO, checkpoint và process group; không chạy nguyên
block này vì `load_*` là contract của backend mới):

```text
set_same_model_initialization_seed_on_every_rank(10)
T = load_local_SD35_transformer(BF16, frozen, eval)
G_task = FieldTask(SD35Field(load_same_transformer(FP32), shortcut=True))
G_task.field.transformer.enable_gradient_checkpointing()  # verify non-reentrant
EMA_G_task = independent_copy(G_task, FP32, frozen, eval, checkpointing=False)
check_identical_logical_parameter_names_shapes_order(G_task, EMA_G_task)
G = wrap_fsdp(G_task, local_cuda_device)
EMA_G = wrap_fsdp(EMA_G_task, local_cuda_device)
optimizer_G = AdamW(G.parameters(), lr=5e-6, betas=(0.9,0.999), weight_decay=0.01)

# Sau P0: gather/save G snapshot chung rồi construct F trước khi fork.
F_field = load_strict_unwrapped_G_snapshot_into_independent_SD35Field()
target_module_names = configure_fake_lora(F_field, rank=96, alpha=96)
F_raw = FieldTask(F_field).to(local_cuda_device)
F_raw.field.transformer.enable_gradient_checkpointing()  # non-reentrant
F_raw.train()  # dropout phải bằng0, không running-stat modules
F_ddp = DDP(F_raw, broadcast_buffers=False,
            gradient_as_bucket_view=True, find_unused_parameters=False)
optimizer_F = AdamW([p for p in F_raw.parameters() if p.requires_grad],
                   lr=2.5e-5, betas=(0.9,0.999), weight_decay=0.01)
EMA_F = AdapterEMA(F_raw)  # tạo SAU DDP initialization synchronization
```

`configure_fake_lora` là implementation LoRA riêng trong code mẫu, không phải PEFT
file format. Save danh sách module names và shapes thực tế. Không dùng standard
PEFT loader để bỏ qua keys custom; nếu bên nhận dùng PEFT thay code mẫu, phải
preserve target modules/rank/alpha/dropout và duration MLP, rồi kiểm tra numerical
identity. F frozen base bắt nguồn **G sau P0**, không SD3.5 T gốc và không checkpoint
distill có sẵn. Lưu base này một lần, reuse theo hash cho tất cả adapter checkpoints.

### Adapter shortcut phù hợp MMDiT/SD3.5

Giữ nguyên transformer từ config local: joint attention giữa image/text, dual
attention ở những block mà checkpoint khai báo, QK normalization, patching và
positional embeddings. Không hardcode số layers/width thành một bản SD3 khác.
Ở512, VAE downsample8 → latent64×64; nếu patch_size=2 thì1024 image tokens.
Kiểm tra config thật và positional crop support; không resize pretrained weights
cho vừa input. Token sequence conditioning theo pipeline thật, không tự concatenate
CLIP/T5 kiểu UNet cross-attention.

Embedding sigma pretrained vẫn nhận `1000*sigma`. Thêm
`E_d(d)=MLP(Fourier(1000*d))-MLP(Fourier(0))` vào output `time_text_embed`, trước
các AdaLN modulation và final norm. MLP output layer khởi tạo0: toàn transformer
có hành vi base tại khởi tạo cho mọi d, sau đó mới học finite-duration field.
`E_d(0)=0` luôn: **duration branch** không đổi local path; backbone/LoRA vẫn có
thể đổi local field khi train. Không thay sigma bằng sigma-d, không thêm d vào
text embeddings hoặc viết lại SD35AdaLayerNormZeroX.

Zero init khiến layer đầu duration MLP có thể grad0 ở first step; layer cuối phải
có gradient trên d>0. LoRA B phải có gradient khi A random/B0; LoRA A có thể grad0
lúc đầu. Log riêng hai trường hợp này, không suy “model không học” từ một tensor.

F LoRA96 áp q/k/v/out của image attention, context k/v và q/out khi context output
được dùng; có cả **attn2** của dual-attention blocks và image/context FF Linear.
Block cuối `context_pre_only=True` bỏ context output: không LoRA hóa `add_q_proj`
của block đó. Không LoRA hóa AdaLN modulation, input/output heads hoặc original
time/text projections. F giữ duration MLP trainable riêng. Danh sách actual module
paths/counts/shapes phải được export; không diễn giải rank96 là96 parameters.

Không bật QKV projection fusion sau LoRA injection: code mẫu dựa trên các Linear
độc lập, không phải fused/custom attention processor adapter. DDP `find_unused=False`
chỉ dùng sau audit mọi local/finite/dual path và backward không có trainable
`grad=None`; nếu local checkpoint/processor khác source pin, fail compatibility
hoặc dùng detection trong smoke rồi sửa danh sách modules. Không âm thầm tắt các
dual layers để vừa VRAM.

Nguồn: [SD3 transformer](https://github.com/huggingface/diffusers/blob/v0.33.1/src/diffusers/models/transformers/transformer_sd3.py),
[joint block](https://github.com/huggingface/diffusers/blob/v0.33.1/src/diffusers/models/attention.py),
[joint attention processor](https://github.com/huggingface/diffusers/blob/v0.33.1/src/diffusers/models/attention_processor.py).
Vị trí duration injection và lựa chọn LoRA là thiết kế mới dựa trên các API đó,
chưa được kiểm chứng bằng GPU.

G/EMA-G: FSDP1 `FULL_SHARD`, một whole `FieldTask` mỗi wrapper, không nested auto-wrap
trong bản đầu. G master weights/AdamW moments/EMA-G logical shards FP32, forward
BF16. F: frozen BF16 base replica + trainable FP32 LoRA/duration, DDP all-reduce chỉ
trainables. Không wrap F bằng FSDP `use_orig_params=False` với mixed requires_grad;
không thêm manual BF16 optimizer/master copy vào G FSDP vì sẽ nhân đôi state.

EMA-G và G phải cùng flattening/order/world size. Update EMA-G chỉ ngoài FSDP
forward/backward/full-param contexts khi persistent shards đã được trả về. Dùng
`update_ema_shards` sau successful optimizer_G.step. Với F, `AdapterEMA.update`
sau optimizer_F.step; checkpoint shadow dictionary FP32. Frozen F base không cần
EMA vì bất biến. Functional target calls bỏ qua DDP, không optimizer/hook gradients.

**Một outer FSDP forward cho toàn bộ G loss**, gọi:

```python
with torch.autocast("cuda", dtype=torch.bfloat16):
    loss, diagnostics = G(
        operation="g_loss",
        condition=condition,
        f=F_raw,
        target_model=EMA_G if arm == "A" else EMA_F,
        teacher=T,
        noise=noise,
        independent_noise=independent_noise,
        negative=negative_condition,
        teacher_states=teacher_states,
        teacher_condition=teacher_condition,  # independent cached-anchor ids
        generator=g_aux_rng,
        beta=beta,
        teacher_full_step_probability=0.5,
        teacher_local_probability=0.25,
        bootstrap_full_step_probability=0.5,
    )
```

`FieldTask` giữ ba differentiable G queries bên trong một FSDP lifetime, tránh
ba outer G forwards trước một backward. F training tương tự:

```python
with torch.autocast("cuda", dtype=torch.bfloat16):
    training_batch = fake_targets(
        G, EMA_F, T, noise, independent_noise, condition, negative_condition,
        f_aux_rng, beta=beta,
    )
    loss, diagnostics = F_ddp(
        operation="f_loss", condition=condition, training_batch=training_batch
    )
```

Activation checkpoint phải **non-reentrant**; F frozen input không có requires_grad,
reentrant checkpoint có thể làm mất LoRA gradients. Diffusers0.33.1 default dùng
non-reentrant nhưng phải xác nhận smoke. Code duration context phù hợp bản này vì
checkpoint nằm ở transformer blocks, sau khi time/duration embedding đã được tính.
Không checkpoint toàn bộ SD35Field bằng một wrapper khác khiến duration context
đã reset khi recompute. Không multi-thread/concurrent-call cùng một FieldTask.

### Optimizer step và accumulation

Mỗi optimizer window giữ beta/loss weights, target EMA và sampled data policy cố
định. Mỗi microbatch loss mean chia cho `accumulation_steps`. Với G FSDP, mặc định
**sync/reduce-scatter mỗi microbatch**, giữ gradients sharded và chỉ step một lần
khi đủ batch64. `FSDP.no_sync()` có thể giữ full unsharded gradients qua accumulation,
tốn nhiều VRAM; không dùng mặc định ở512. F DDP no_sync bao cả forward/backward
microbatch chưa cuối; trainables nhỏ hơn. BF16 không cần GradScaler.

Sau **tất cả ranks** hoàn thành forward, all-reduce finite flag của loss và DMD
direction trước backward. Không rank-local raise/skip trước collective tiếp theo.
Sau backward, G dùng `G.clip_grad_norm_(10)` cho norm sharded đúng toàn model; F
dùng `clip_grad_norm_` trên trainables đã DDP sync. Kiểm tra finite gradient norm
collectively rồi mới step/EMA/counter. Skip đồng bộ cả window khi nonfinite; clear
grads, ghi failure reason, không nan_to_num hoặc advance optimizer moments/counters.
Sau3 failed attempts liên tiếp save failure state của cả ranks và abort.

Beta của cycle lấy từ k_G trước F, giữ nguyên khi G retry. Nếu G fail sau F success,
retry G, không chạy F thêm trong cùng slot; checkpoint lưu `next_actor`/cycle phase.
F update thất bại retry đúng slot F. LR warm-up đếm successful updates của actor ở
phase hiện tại; trong P2 `lr = peak*min(1,(k_actor+1)/20)`. Không scale LR tuyến
tính theo world size/batch.

## 9. RNG, matched fork và noise coupling

Tách generators: model initialization, data shuffle, G noise, F noise, G auxiliary,
F auxiliary, probe/eval. Per-rank data shards disjoint; global shuffle/cursor chung.
Sau fork tạo các streams A/B từ cùng seed, rank, actor và attempted-update index;
**không đưa tên arm vào seed**. EMA target calls deterministic, dropout0, không
tiêu thụ training RNG. Probe/sample giữ và restore mọi training RNG.

Teacher batch riêng có n=B/4 states và `teacher_condition` đúng ids, không lấy
positive condition từ on-policy batch khác của reLAION. Runtime validate shapes,
caption ids và cache hashes trước forward; validate failure phải đồng bộ mọi rank.
Noise cho re-noising
F phải độc lập initial noise đã tạo y_G. DMD và bootstrap cùng dùng fresh on-policy
G endpoints, không dùng teacher endpoint giả làm current generated sample.

P0 gọi `G(operation="g_warmup", teacher_states=..., condition=teacher_condition,
generator=..., full_step_probability=0.5, local_probability=0.25)`; P2 map config
`g_teacher_*`/`g_bootstrap_*` tới keyword args tương ứng của `generator_losses`.
Validate probabilities trong[0,1], full+local<=1 và dyadic levels>=1 trước collective
forward; ghi actual full/local/short counts. F giữ uniform dyadic hierarchy;
không truyền G full-step bias vào F sampler rồi làm lệch mục tiêu tracking đã chốt.

Khóa A/B bằng một shared-init manifest và một config diff chỉ có source target và
run directory/name. Cache/prompts/world-size/batch/precision/LR/checkpoint weight
choice/optimizer initialization/EMA shadows đều phải bằng nhau. F sẽ khác giữa A/B
sau training vì G endpoints khác; đó là hậu quả của treatment, không ép F tiếp tục
giống nhau để làm mất on-policy tracking.

## 10. Logging/TensorBoard và profile để kiểm tra tốc độ thật

JSONL mỗi rank, tổng hợp rank0; TensorBoard trên rank0, không upload logs tự động.
Ghi phase/P0_G/P1_F/P2_G/P2_F, arm, k_G/k_F, attempted updates, effective batch64,
samples seen, LR, beta, target source và successful/skipped events. Ghi F local
MSE, F shortcut MSE, held-out normalized errors theo sigma; G direct/teacher/shortcut
losses, DMD-direction RMS, latent mean/std, full G global grad norm, F trainable
grad norm, clip rate, số trainables và d/sigma histograms.

Ghi riêng full-interval teacher/shortcut losses bằng sum/count qua ranks, cùng
`teacher_full_step_fraction`/`bootstrap_full_step_fraction`; empty subset không
được mean thành NaN rồi skip training. Đếm mọi differentiable G endpoint ởDMD là
1NFE; log1/2/4/8-step diagnostic results ởtags khác nhau. Không gộp metric của nhiều
NFEs thành một score hoặc tự chọn steps theo từng image để cải thiện kết quả.

Master update diagnostics: RMS delta, relative delta norm và nonzero delta fraction
cho G **FP32 master shards**, F LoRA và duration MLP; không log BF16 shadow delta rồi
kết luận optimizer đứng. Đo mỗi10 updates bằng fixed sampled parameter elements
trước/sau optimizer step; ghi sample ids/counts và gọi là sampled statistics.
Không clone cả model FP32 mỗi update hoặc copy nó về CPU chỉ để log delta.
Phân tích gradient interference chỉ thêm ở sparse probe
khi có budget và gọi rõ output-space hay parameter-space; không dùng cosine giữa
hai vector ở khác state/time và diễn giải là parameter gradient alignment.

Timing riêng: data/cache IO, G endpoint generation, F target build, F forward/backward,
teacher CFG, G teacher anchor, G bootstrap target, G forward/backward, optimizer,
EMA, probe, image decode và checkpoint. Dùng CUDA events hoặc synchronized sparse
profiling; ghi profiling overhead. All-reduce max seconds/peak VRAM qua ranks.
Logical forward counts ghi actor, batch size và CFG conditional/unconditional
evaluations; tách activation-checkpoint recomputations khi profiler đo được.

Warm smoke phải đi qua **P2 có beta>0, d=1 và cả nguồn EMA-G/EMA-F**, không chỉ P0.
Profile b8/accum2 rồi b16/accum1 ở512 khi tất cả T/G/F/EMA và AdamW moments đã allocate; peak
VRAM trước optimizer state allocation không chứng minh cả run vừa. Run A/B thật
300G với ramp100 để có200G ở beta_max, không chỉ train đến cuối ramp rồi kết luận.

Ước lượng ETA sau10 finite P2 cycles, báo median/p90 F/G và GPU-giờ
`world_size*wall_seconds/3600`; cộng chi phí cache/P0/P1, eval và checkpoint. Không
hứa thời gian cụ thể trước profile. Cache/warm-up là chi phí dùng chung, báo riêng
và phân bổ rõ nếu so cost per arm. Baseline A có G-EMA FSDP target all-gather còn B
target F LoRA; throughput có thể khác dù logical target NFEs giống nhau.

TensorBoard images chỉ8 fixed prompts G/EMA-G/teacher ở milestone150/300; latent
metrics/probes mỗi50. Không lưu full image batch mỗi step. Viết logger mới trong
project này, không reuse logger code của repo cũ; không giả `k_G` cho warm-up F.

## 11. Checkpoint/resume/export và retention

Giữ shared-init sau P1 cố định, một latest atomic checkpoint mỗi arm, một selected
final G export. Checkpoint mốc150 cũ chỉ prune sau khi checkpoint mới complete và
manifest hash đã xác nhận; protect `.pin`, failures, source assets và selected export.
Không nhân bản frozen F base hoặc teacher mỗi checkpoint. Không ghi weights/cache/
metrics output vào Git.

Schema mới cần chứa: model config/sigma convention/duration encoding; G/EMA-G FP32
shards; F trainable tensors, target module list, rank96/alpha96, frozen-base hash;
EMA-F shadows; optimizer_G/F states, rank/world size and partition layout; phase,
actor slot/counters/ramp/LR schedule; RNG từng stream; prompt order/cursor; cache
hashes, source base and G-P0 snapshot hashes; environment/commit và validation gates.

Use Torch2.6 FSDP sharded state-dict + optimizer-state API hoặc DistributedCheckpoint;
không `torch.save(G.state_dict())` trên rank0 rồi cho rằng đó là full portable weights.
Save/resume là collective, phải có đủ shard files/checksums/complete marker. Resume
đúng world size/strategy; đổi topology là conversion riêng, không silent resume.
Test resume sau P1 và một complete P2 cycle cho cả A/B. Kiểm tra tiếp theo cùng
batch/noise/LR/beta/loss trong tolerance BF16; preserve tensor dtypes và optimizer
ownership. Frozen base hash mismatch hoặc missing adapter key/shape phải fail.

Shared-init fork **khác exact resume**: tạo optimizers mới, reset experiment clocks,
giữ G/F/EMA đã warm. Mỗi arm đều cùng policy này. Không resume optimizer cũ của A
vào B. Khôi phục exact checkpoint phải restore moments/counters/RNG/cycle.

Export G chỉ inference weights/config/duration embedding, G/EMA-G weight choice và
nguồn train. Loader mới dựng SD35Field trước khi load strict; pipeline SD3 thường
không tự nhận duration embedding. Inference chỉ `one_step(G,z,c)` rồi VAE decode;
F/T không cần load. “One step” phải đo **1 conditional transformer evaluation**,
không thêm unconditional CFG call, hidden F rollout hay nhiều G calls.

Export metadata ghi `primary_inference_steps=1`, externalCFG1 và duration support.
`sample_shortcuts(...,steps=2/4/8)` là sampler diagnostic tùy chọn của cùng model;
1-step loader vẫn phải hoạt động độc lập. Reload kiểm tra1-step output/NFE thật,
không chỉ kiểm tra một pipeline fallback nhiều bước.

## 12. Evaluation để biết có work và cách so paper

Chốt primary metric **trước fork** dựa trên metric bên nhận đã có: default
**HPSv2.1**, kiểm tra checkpoint/version/scale trước khi lock. Nếu HPSv2.1 chưa
usable nhưng ImageReward có provenance hợp lệ, đổi chung sang ImageReward trước
fork và ghi rõ choice;
CLIP-S và metric còn lại là secondary. Nếu chỉ có CLIP-S thì ghi đó là primary
semantic score và không claim aesthetic quality từ CLIP-S riêng. Không cài/tải lại
ImageReward/HPS chỉ vì tên metric xuất hiện trong docs; dùng adapters với version,
checkpoint revision, preprocessing, score scale và aggregation thực tế.

Reference rows: base teacher32 uniform sigma, base teacher official sampler nếu
có, G sau P0/P1 (1NFE), A300G (1NFE), B300G (1NFE), **native512 tất cả**.
Teacher reference ở128 validation prompts có sẵn; không bắt generate1024 teacher
images trước khi có screening. Nếu báo teacher vs student trên final test, phải
generate teacher cùng test pairs, không so validation mean với test mean.
Dùng **EMA-G** cho cả A/B theo
policy định trước; online G báo phụ, không pick G/EMA theo từng prompt hoặc theo arm.
Test512 **reLAION held-out** prompts ×2 noise seeds mới =1024 ảnh mỗi student model;
cùng prompt ids/noise hashes,
resize/crop/decoder/evaluator. Lưu per-sample primary + secondary scores, không chỉ
mean. Noise test khác cache/probe/training. Validation128 dùng gates; final test
không dùng tune beta/LR/loss weights/chọn milestone sau khi xem kết quả.

Primary test và so A/B luôn **1NFE**; few-step dùng16 validation prompts chung,
báo riêng2/4/8NFE và đúng cost. B4step>A1step không chứng minh B tốt hơn ởone-step.
Không chọn final export chỉ vì few-step metric cao trong khi one-step thất bại.

Điều kiện tín hiệu dương: B tốt hơn A trên primary **held-out image** metric với
paired CI và không có degradation rõ trên secondary/ảnh uncurated; tracking F đạt
gate, không collapse, NFE1 thật. Teacher endpoint error và shortcut residual là
diagnostics riêng, không dùng latent MSE giảm để chứng minh ảnh đẹp hơn. Chi phí
ngang số successful updates nhưng báo GPU-giờ/samples/skips để không giấu khác cost.

Paired bootstrap **theo prompt**: trung bình2 noise scores trong mỗi prompt rồi
resample prompt ids chung cho A/B. Code tham khảo (thực thi trên máy bên nhận):

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

CI này đo uncertainty trên prompts của hai checkpoint cụ thể; nó không đo variance
giữa training seeds. Nếu có tín hiệu, lặp P2 A/B với training seed11 từ cùng shared
init, giữ test/metric policy. Báo đó là replication conditional on shared warm init,
không giả là hai independent end-to-end runs. Nếu cần claim end-to-end robustness,
lặp cả P0/P1 sau. Screening300G âm tính không đủ bác bỏ toàn bộ phương pháp.

FID trên1024 ảnh chỉ exploratory, không so trực tiếp FID-10k hay FID-50k paper.
Khi B qua screening, chạy A/B final **cùng COCO2014-val10k, native512**
với cùng protocol, full metrics/provenance. COCO chỉ eval, train vẫn reLAION. Reference
real data/split/hash, image resize, feature checkpoint và count phải matched. Việc
so một SD3.5-512 score với SDXL native1024→512 score không cho attribution phương pháp.
Nếu muốn so FD-loss/Decoupled trên SD3.5, cần checkpoint/baseline **cùng backbone**
và đo lại; không tự điền số baseline từ ImageNet hoặc bảng SDXL.

Nếu muốn kiểm tra trực tiếp fake bridge vs **frozen-teacher** trajectory, thêm arm C
sau screening: cùng on-policy states và losses, teacher integrate mỗi half interval
bằng đủ Euler substeps, thay target source; ghi extra NFEs. Không dùng hai local
teacher calls cho một d=1/2 rồi gọi đó là accurate teacher finite shortcut. Arm A
hiện tại kiểm tra self-vs-fake, chưa chứng minh superiority so teacher trajectory.

## 13. Lộ trình triển khai và contract CLI

Bên nhận viết theo thứ tự, không chạy dài trước khi có save/resume:

1. Project mới + backend flow SD3.5 + reLAION reader/conditioning cache + duration embedding.
2. Teacher cache512px, disjoint split/hash và teacher-quality gate.
3. G warm-up, F LoRA96 warm-up và tracking gates.
4. G FSDP / F DDP / two EMA modes, finite/skip/accumulation, checkpoint+export.
5. Shared-init fork lock, A/B300G, TensorBoard/profile, paired eval/plugins.

Những entrypoint sau **cần được implement mới** trong `sd35_shortcut.cli`. Sau
khi viết xong parser, command contract trên máy GPU của bên nhận là:

```bash
# Các configs/local*.yaml phải được tạo từ schema ở §3 với local asset paths thật.
torchrun --standalone --nproc_per_node=4 -m sd35_shortcut.cli prepare-cache \
  --config configs/local-sd35-common.yaml

torchrun --standalone --nproc_per_node=4 -m sd35_shortcut.cli warmup \
  --config configs/local-sd35-common.yaml

# Fork lấy cùng complete shared-init manifest; tạo optimizer mới cho cả hai.
torchrun --standalone --nproc_per_node=4 -m sd35_shortcut.cli train \
  --config configs/local-sd35-A.yaml --fork-from runs/sd35/shared-init

torchrun --standalone --nproc_per_node=4 -m sd35_shortcut.cli train \
  --config configs/local-sd35-B.yaml --fork-from runs/sd35/shared-init

python -m sd35_shortcut.cli compare \
  --run-a runs/sd35/A --run-b runs/sd35/B --metric-plugin recipient_metrics
```

`compare` phải đọc finalized generation/evaluator reports, không tự generate scores
hoặc coi thiếu plugin là0. `prepare-cache` multi-rank chia ids, rank0 finalize sau
barrier/hash/count checks; chỉ skip existing cache khi manifest matches, không
overwrite teacher artifacts có scientific settings khác. `warmup` chỉ chốt
shared-init khi cả hai gates đạt hoặc có exception reason rõ trong metadata.
`train` chỉ cho diff A/B whitelist; fail fast nếu config khác LR/batch/seed/EMA.

Không import/copy launcher/trainer/config/utility cũ; không override unknown fields
để bỏ qua schema. Dùng venv/package/run directories riêng, log commit của repo mới.
Không reset tiến trình bên nhận đang chạy chỉ vì clone/pull bản handoff này.

## 14. Checks bắt buộc trên máy bên nhận và gói trả về

Các checks có giá trị phương pháp: analytical constant/linear field xác nhận dấu
`x-d*v` và binary composition; duration branch contributes0 ởd=0 kể cả sau training;
smallest-duration base case; native512 shapes/dual layers; G one-step vs multistep NFE; gradient ownership/detach;
F frozen-base hash bất biến, chỉ adapter/duration đổi; EMA masters FP32 và update
đúng actor; accumulation batch64; all-rank skip không deadlock; FSDP G EMA shard
layout đúng; strict resume/export/reload; A/B config diff chỉ source; metric id/seed
pairing và prompt disjointness. Pure toy tests không thay H100 smoke thực tế.

Kiểm tra G sampler fractions full/local/short, full interval luôn(sigma=1,d=1),
full G teacher label luônnoise–teacher_clean, và gradient từ DMD đi qua one-step
G graph. `sample_shortcuts(steps=1)` phải bằng `one_step` và gọi model1 lần;
2/4/8 gọi đúngN lần cùng wrapper và d=1/N. Few-step pass không bỏ qua one-step gate.

Smoke4GPU2–4 cycles có AdamW states allocated, checkpoint/resume cycle thật, d=1,
beta>0 và cảEMA targets, rồi mới P0/P1/P2 thật. Kiểm tra generated noise seed,
teacher labels, sigmas và decoded images đúng SD3.5 scale. F LoRA có nonzero B
gradients/update sau first step; A có thể grad0 ngay initial zero-B, không coi đó
là lỗi. Không checkpoint reentrant làm mất cả B gradients rồi report LoRA “đã train”.

Gói trả về: commit/config hashes, model/cache/frozen-F-base hashes, gates before/after,
resolved batch64, successful/skipped counts, G/F/duration master deltas, profile
timings/VRAM/GPU-hours bao gồm setup, shared-init và latest A/B checkpoints, G
selected export+loader metadata, uncurated same-noise samples, metric provenance,
per-prompt scores/paired CI, seed replication nếu có. Không claim paper win hoặc
convergence khi chỉ có losses/logs.

## 15. Sources và trạng thái bàn giao

- FD-loss: [paper v1, B.4/Table B.3](https://arxiv.org/html/2604.28190v1#A2.SS4);
  [official repository](https://github.com/Jiawei-Yang/FD-Loss/tree/5c03b8112fec8b9432631e4ce053c0d918cc24bc),
  HEAD đọc ngày2026-10-09. Setting được đọc từ paper, không suy từ ImageNet scripts.
- Shortcut Models: [paper v3, §3](https://arxiv.org/html/2410.12557v3#S3),
  [official code pinned](https://github.com/kvfrans/shortcut-models/blob/601004348667094e1b71f30942199759412d4432/targets_shortcut.py).
- SD3.5: [official model card](https://huggingface.co/stabilityai/stable-diffusion-3.5-medium);
  Diffusers [transformer](https://github.com/huggingface/diffusers/blob/v0.33.1/src/diffusers/models/transformers/transformer_sd3.py),
  [pipeline](https://github.com/huggingface/diffusers/blob/v0.33.1/src/diffusers/pipelines/stable_diffusion_3/pipeline_stable_diffusion_3.py)
  và [checkpointing implementation](https://github.com/huggingface/diffusers/blob/v0.33.1/src/diffusers/models/modeling_utils.py).
- Yêu cầu chủ project: repo mới độc lập, native512, reLAION local, SD3.5 Medium base
  Diffusers, one-step, cảG/F shortcut, F LoRA96 + warm-up trước bridge,
  G5e-6/F2.5e-5, batch64, log chi tiết + TensorBoard, retention ít.

**Đã làm ở phiên authoring:** đọc primary sources, viết thiết kế/code reference,
static Ruff/AST và kiểm tra YAML/số học config. **Chưa làm:** import Torch, unit
tests, H100 smoke, FSDP/DDP runtime, numerical parity, save/resume thật, training,
metric execution hoặc benchmark. Runtime integration còn do bên nhận thực hiện;
không dùng trạng thái static-only này để báo “coding/training thành công”.
