# SD3.5 Fake Shortcut

Project mới độc lập: **G distill slow instantaneous F**, học one-step shortcut;
sau warm-up/tracking, giảm LoRA của F dần để trở về SD3.5 Medium **base teacher**.
Native **512×512**, prompts **reLAION local**, global batch **64**.

**Bàn giao chính: [HANDOFF_VI.md](HANDOFF_VI.md)** — thiết kế, config, code reference,
hướng dẫn runtime, gates, TensorBoard/logging, checkpoint/resume và matched A/B.
[reference/shortcut_core.py](reference/shortcut_core.py) giống block code §7.

| Chốt | Setting |
|---|---|
| Teacher | Local Diffusers SD3.5 Medium base frozen; 50 steps, CFG4.5 |
| G | Full-weight + duration conditioning; LR `5e-6`; inference1 conditional NFE |
| F | Frozen **teacher** backbone + LoRA rank/alpha96; **instantaneous**, không shortcut |
| F LR / supervision | `2.5e-5`; combined CFG4.5 CFM trên one-step G samples |
| Main G targets | **25% full d1 / 25% local F / 50% shorter finite** |
| Main G teacher/DMD loss | Cả hai bằng0; teacher anchor chỉ trong F target, β≤0.01 |
| Slow F / target replay | 1 F update / 5 G updates; refresh G labels mỗi5 G updates |
| Fade | Hold≥200G + readiness gate; freeze adapters/EMA-F, LoRA1→0 trong1000G |
| Precision | BF16 model forwards; FP32 master/moments/EMA/targets/CFG/Euler/loss |
| Batch | `8 × 4 GPU × accum2 = 64`; profile `16 × 4 × 1` trước fork |
| Phases | G teacher bootstrap300G → F warm200F → A/B300G mỗi arm |
| Random join | rho uniform `[0.25,0.75]`; prefix=rho×d, tail=(1-rho)×d; same A/B policy |
| Duration training | 50% continuous random-short d in `[1/32,1)` từ P0; below-floor child dùng d0 |
| Goal / checkpoint policy | One-step primary; few-step chất lượng tốt không thay thế one-step acceptance |
| A/B difference | Prefix EMA-G / local EMA-F Euler; cùng EMA-G tail tới cuối đoạn |

Main G không nhận loss teacher trực tiếp. F phải warm trên phân phối student và
qua tracking gate trước. F base không copy từ G: **λ0 phải đúng T_CFG4.5**, nên
F guided field dùng cùng CFG/negative policy với teacher. Khi fade bắt đầu, F
optimizer và EMA-F ngừng cập nhật để adapters không bù lại scale giảm.

Pilot300G chỉ tới LoRA strength0.9 nếu gate pass200; chưa chứng minh fade hoàn tất.
Để kiểm chứng tới teacher, tiếp tục cả arms tới1200G theo policy đã khóa. One-step
là evaluation chính; optional2/4/8-step chỉ diagnostic. Equal updates không equal
compute: B có thêm F bridge calls. Báo matched held-out scores/paired CI và GPU-giờ;
training loss riêng không chứng minh quality hoặc paper parity.

Revision: `sd35_fake_shortcut_512_relaion_v4_random_splice`. Không exact-resume v3/v2/older training
states. v4 thay fixed-half bằng random join và train continuous durations ngay P0;
G không nhận split condition, chỉ `(x,sigma,d,c)`. Row-wise rho phải lưu trong target
cache/resume; maxstep1/16,min2 và rho≤.75 ⇒ tối đa12 F bridge calls/row. Teacher/text
caches có thể reuse khi hashes/policy khớp.
Teacher cache dùng native50 sigma grid nhưng FP32 CFG/Euler accumulation; stock
BF16 pipeline rounding khác, chưa claim bitwise sampler/paper parity.

Repo là **thiết kế + reference kernels**, chưa có distributed trainer/CLI hoàn
chỉnh. Static Ruff/AST/config checks đã làm; chưa import Torch, chạy tests, GPU
smoke, training, save/resume thật hoặc metrics. Các CLI trong tài liệu là contracts
bên nhận cần implement. Batch/throughput/quality chưa verified trên H100.

Toàn bộ code reference viết mới. Không dùng code/config/utilities của training
project cũ. Bên nhận tạo package/runtime mới, dùng upstream Torch/Diffusers và
metric libraries sẵn có qua adapters có provenance. Không commit weights, data,
tokens, logs hoặc generated images.
