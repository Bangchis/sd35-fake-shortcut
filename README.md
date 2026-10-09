# SD3.5 Fake Shortcut

Dự án mới độc lập để kiểm tra **fake trajectory shortcut có giúp one-step generator
tốt hơn self shortcut không**, từ SD3.5 Medium **base**. Native **512×512**, prompts
từ **reLAION local**, global batch **64**.

**Bàn giao chính: [HANDOFF_VI.md](HANDOFF_VI.md)** — có thiết kế, YAML config đầy đủ,
code lõi, cách triển khai runtime, warm-up, TensorBoard/logging, checkpoint/resume
và matched A/B evaluation. Bản code riêng để dễ đọc:
[reference/shortcut_core.py](reference/shortcut_core.py), giống block code §7.

| Chốt | Setting |
|---|---|
| Teacher | SD3.5 Medium base local Diffusers, frozen; CFG 4.5 |
| Generator | Full-weight, finite-duration conditioning, inference 1 conditional NFE |
| One-step priority | G teacher targets 50% full d=1; G shortcut targets 50% full d=1 |
| Generator LR | `5e-6` |
| Fake | Frozen G-after-bootstrap backbone + LoRA rank/alpha 96 + duration MLP |
| Fake LR | `2.5e-5` |
| Precision | BF16 forwards; FP32 optimizer trainables/moments/EMA |
| Batch | `8 × 4 GPU × accumulation 2 = 64`; profile `16 × 4 × 1` before fork |
| Phases | G teacher bootstrap 300G → F warm-up 200F → A/B 300G mỗi nhánh |
| F:G in A/B | `1:1` |
| A/B difference | G shortcut target: EMA-G / EMA-F; các settings khác giống nhau |

F phải warm-up theo samples của G và qua tracking gate trước khi làm target cho G.
One-step là output và evaluation chính; DMD luôn nhận endpoint one-step G.
Cùng checkpoint có thể sample2/4/8 steps để diagnostic, nhưng few-step tốt không
thay thế yêu cầu1-step quality. Revision hiện tại: `v2_onestep`.
300 updates là screening budget, chưa phải setting đã chứng minh convergence.
So A/B bằng held-out image metrics với paired confidence interval; benchmark paper
cần cùng prompts/resolution/evaluator. Training loss riêng không chứng minh chất lượng.

Repo hiện là **thiết kế + reference kernels**, chưa có trainer/CLI hoàn chỉnh.
Không chạy được các CLI contract trong tài liệu cho tới khi bên nhận triển khai
runtime mới. Đã kiểm tra tĩnh Ruff/AST/config; chưa import Torch, chạy tests,
GPU smoke, training hoặc metrics trên máy authoring.

Toàn bộ code reference viết mới; không dùng code của project training cũ. Bên nhận
tạo package `sd35_shortcut` và runtime riêng, dùng Torch/Diffusers upstream, local
base checkpoint/reLAION và metric libraries sẵn có. Không commit weights, data,
tokens, logs hoặc generated images vào repo này.
