# 2026-01-04 — LOSO 4-class — EA-STACK-MULTI-SAFE：把 TSA 当高风险 arm 做 family-specific gate（neg_transfer≈0）

目标：沿“**证书有效性 + 安全选择/回退**”主线，优先把 `neg_transfer_rate` 压到接近 0，并在严格可比协议下观察 mean / worst 是否同步改善。  
本轮只动一个杠杆：**对 TSA arm 增加更严格的 family-specific gate**（更高 guard 阈值 / 最小 ridge 预测提升 / drift hard gate），避免 S9 这类崩盘。

---

## 1) 上一次失败分析（post-mortem）

上一次（`docs/experiments/20260104_loso4_actionset_stack_fbcsp_gate_v1.md`）在“FBCSP gate”之后：
- `ea-stack-multi-safe-csp-lda` mean acc **0.5243**（比 EA 低 0.77% abs）
- `neg_transfer_rate_vs_ea = 0.3333`
- 主要负迁移来源不再是 `fbcsp`，而是 `tsa`：
  - **S9**：最终选到 `tsa`，EA `0.6840 → 0.5694`（Δ = **−0.1146**，主导掉点）

因此本轮假设：**把 TSA 视为高风险族并加严格 gate，可以消掉大幅负迁移（把 neg_transfer 压到 0 附近）**；代价可能是 accept_rate 下降、mean 提升变小，但这是“先稳住再追涨”的必要步骤。

---

## 2) 本次改动（只动一个杠杆）

唯一杠杆：在 `ea-stack-multi-safe-csp-lda` 的 selection 阶段加入 **TSA 专属 gate**：
- 更高的 guard 阈值（`--stack-safe-tsa-guard-threshold`）
- 最小 ridge 预测提升（`--stack-safe-tsa-min-pred-improve`）
- 额外 drift hard gate（`--stack-safe-tsa-drift-delta`，mean KL(p_anchor||p_tsa)）

实现提交：
- `git commit 73735bf`（`Add TSA high-risk gate for stacked selector`）

---

## 3) 怎么做的（命令 + 输出目录）

命令（严格可比协议不变）：
```bash
_MNE_FAKE_HOME_DIR=$PWD/.mne_home MNE_DATA=$PWD/.mne_data \
conda run -n eeg python run_csp_lda_loso.py \
  --dataset BNCI2014_001 --preprocess paper_fir \
  --events left_hand,right_hand,feet,tongue --sessions 0train \
  --n-components 6 \
  --methods ea-csp-lda,ea-stack-multi-safe-csp-lda \
  --si-proj-dim 21 --si-subject-lambda 1 --si-ridge 1e-6 \
  --si-chan-ranks 21 --si-chan-lambdas 0.5,1,2 \
  --oea-zo-selector calibrated_stack_ridge_guard \
  --oea-zo-calib-max-subjects 0 --oea-zo-calib-seed 0 \
  --oea-zo-calib-guard-threshold 0.5 \
  --stack-safe-fbcsp-guard-threshold 0.95 \
  --stack-safe-fbcsp-min-pred-improve 0.05 \
  --stack-safe-fbcsp-drift-delta 0.15 \
  --stack-safe-tsa-guard-threshold 0.95 \
  --stack-safe-tsa-min-pred-improve 0.05 \
  --stack-safe-tsa-drift-delta 0.15 \
  --no-plots \
  --run-name loso4_actionset_stack_fbcsp_tsa_gate_v1
```

Outputs：
- `outputs/20260104/4class/loso4_actionset_stack_fbcsp_tsa_gate_v1/20260104_method_comparison.csv`
- `outputs/20260104/4class/loso4_actionset_stack_fbcsp_tsa_gate_v1/20260104_results.txt`
- `outputs/20260104/4class/loso4_actionset_stack_fbcsp_tsa_gate_v1/20260104_predictions_all_methods.csv`

---

## 4) 结果（主表指标 + 关键被试）

来自 `20260104_method_comparison.csv`：
- `ea-csp-lda` mean acc **0.5320**, worst **0.2569**
- `ea-stack-multi-safe-csp-lda`（FBCSP+TSA gate）mean acc **0.5405**（**+0.85% abs**）, worst **0.2569**
  - `accept_rate = 3/9 = 0.3333`
  - `neg_transfer_rate_vs_ea = 0/9 = 0.0000` ✅
  - `cert_improve_spearman = 0.4333`（相比上一轮 0.3000 继续改善）
  - `guard_improve_spearman = 0.2000`

关键现象（见 `20260104_results.txt` 的 per-subject 表）：
- **S9 不再崩盘**：本轮 `tsa` 被 gate 阻止（`stack_multi_tsa_blocked=1`），最终回退到 EA（`stack_multi_family=ea`），Δ=0。
- S7/S8 同样出现 `tsa` 被阻止并回退到更安全的选择（S7→rpa，S8→ea），从而把 `neg_transfer_rate` 压到 0。
- 本轮 mean 的正增益主要来自：
  - S3：选到 `chan`，EA 0.7500 → 0.7951（+4.51%）
  - S5：选到 `rpa`，EA 0.3090 → 0.3403（+3.13%）

结论：**TSA gate 达成了“先稳住（neg_transfer≈0）”的目标，并在不掉 worst 的情况下带来小幅 mean 提升。**

---

## 5) 图（用于汇报/诊断）

使用脚本 `scripts/plot_stack_multi_safe_summary.py` 从 `*_results.txt` 生成：
- per-subject Δacc vs EA：`docs/experiments/figures/20260104_loso4_actionset_stack_fbcsp_tsa_gate_v1_delta.png`
- ridge 预测提升 vs 真实提升：`docs/experiments/figures/20260104_loso4_actionset_stack_fbcsp_tsa_gate_v1_ridge_vs_true.png`
- guard 概率 vs 真实提升：`docs/experiments/figures/20260104_loso4_actionset_stack_fbcsp_tsa_gate_v1_guard_vs_true.png`
- pre vs final family（标注被 gate 阻止的 arm）：`docs/experiments/figures/20260104_loso4_actionset_stack_fbcsp_tsa_gate_v1_family.png`

---

## 6) 下一步（在“neg_transfer≈0”基础上追 mean）

本轮把“灾难负迁移”压住后，下一轮才能更合理地追提升（仍建议一次只动一个杠杆）：
- 方向 1：**更精细地放开 TSA**（例如只调 `tsa_min_pred_improve` 或只调 `tsa_guard_threshold`），观察 mean 的甜点区，同时监控 `neg_transfer_rate`。
- 方向 2：提升 `chan` / `rpa` 族的覆盖与收益（例如更密的 `(rank, λ)` 网格），在不引入高风险崩盘的前提下抬高 mean。

