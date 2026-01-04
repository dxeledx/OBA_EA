# 2026-01-04 — LOSO 4-class — EA-STACK-MULTI-SAFE：把 FBCSP 当高风险 arm 做 family-specific gate（仍未达成 neg_transfer≈0）

目标：按“**证书有效性 + 安全选择/回退**”主线，先把 `neg_transfer_rate` 压到接近 0，再追 mean 提升。  
本轮只动一个杠杆：**对 FBCSP arm 增加更严格的 family-specific gate（更高阈值/最小预测提升/drift hard gate）**。

---

## 1) 上一次失败分析（post-mortem）

上一次（`docs/experiments/20260103_loso4_actionset_stack_probe_guard_v1.md`）：
- `ea-stack-multi-safe-csp-lda` mean acc **0.5081**（比 EA 低 2.39% abs）
- `neg_transfer_rate_vs_ea = 0.4444`
- 失败模式：选择器系统性偏向 `fbcsp`，并在 S5/S7/S8/S9 出现明显负迁移。

因此本轮假设：**先把 FBCSP 视为高风险族，用更严格 gate 禁止“看起来很好但实际很差”的误接受**，可以显著降低负迁移并抬升 mean。

---

## 2) 本次假设与改动（只动一个杠杆）

唯一杠杆：在 `ea-stack-multi-safe-csp-lda` 的 selection 阶段加入 **FBCSP 专属 gate**：
- 更高的 guard 阈值（`--stack-safe-fbcsp-guard-threshold`）
- 最小 ridge 预测提升（`--stack-safe-fbcsp-min-pred-improve`）
- 额外 drift hard gate（`--stack-safe-fbcsp-drift-delta`，mean KL(p_anchor||p_fbcsp)）

实现提交：
- `git commit 87f2fd7`（`Add FBCSP high-risk gate for stacked selector`）

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
  --no-plots \
  --run-name loso4_actionset_stack_fbcsp_gate_v1
```

Outputs：
- `outputs/20260104/4class/loso4_actionset_stack_fbcsp_gate_v1/20260104_method_comparison.csv`
- `outputs/20260104/4class/loso4_actionset_stack_fbcsp_gate_v1/20260104_results.txt`
- `outputs/20260104/4class/loso4_actionset_stack_fbcsp_gate_v1/20260104_predictions_all_methods.csv`

---

## 4) 结果（主表指标 + 关键被试）

来自 `20260104_method_comparison.csv`：
- `ea-csp-lda` mean acc **0.5320**, worst **0.2569**
- `ea-stack-multi-safe-csp-lda`（FBCSP gate 后）mean acc **0.5243**（**-0.77% abs**）, worst **0.2569**
  - `accept_rate = 5/9 = 0.5556`
  - `neg_transfer_rate_vs_ea = 3/9 = 0.3333`
  - `cert_improve_spearman = 0.3000`（相比上一轮 -0.40 有改善）
  - `guard_improve_spearman = 0.1667`

FBCSP gate 的直接效果（见 `20260104_results.txt` 的 per-subject 表）：
- S5/S7/S8/S9：`stack_multi_pre_family = fbcsp`，但 `stack_multi_fbcsp_blocked = 1`，最终被改选为 `rpa` 或 `tsa`。
  - S5：改选 `rpa` 后反而 **+0.0313**（EA 0.3090 → 0.3403），说明“阻止 fbcsp”确实能救一部分被试。

但仍然存在明显负迁移（不是来自 fbcsp，而是来自 `tsa`）：
- **S9**：最终选到 `tsa`，EA `0.6840 → 0.5694`（Δ = **−0.1146**，主导 neg_transfer）
- S8：EA `0.7188 → 0.6979`（Δ = −0.0208）
- S7：EA `0.5729 → 0.5625`（Δ = −0.0104）

结论：**FBCSP gate 有效地抑制了“fbcsp 导致的负迁移”，但 neg_transfer 仍未压到 0，因为 TSA 仍是高风险 arm。**

---

## 5) 图（用于汇报/诊断）

- per-subject Δacc vs EA：`docs/experiments/figures/20260104_loso4_actionset_stack_fbcsp_gate_v1_delta.png`
- ridge 预测提升 vs 真实提升：`docs/experiments/figures/20260104_loso4_actionset_stack_fbcsp_gate_v1_ridge_vs_true.png`
- guard 概率 vs 真实提升：`docs/experiments/figures/20260104_loso4_actionset_stack_fbcsp_gate_v1_guard_vs_true.png`
- pre vs final family（红 x 表示 FBCSP 被 gate 阻止）：`docs/experiments/figures/20260104_loso4_actionset_stack_fbcsp_gate_v1_family.png`

---

## 6) 下一步（仍然“先压 neg_transfer≈0”）

本轮证明：**高风险并不只来自 FBCSP**。下一轮若继续按“先压 neg_transfer”策略，只动一个杠杆即可：
- 把 `tsa` 也视作高风险 arm（加 family-specific gate 或直接在动作集里暂时禁用 TSA 作为消融），优先消掉 S9 这类大幅负迁移。

完成 neg_transfer≈0 之后，再讨论如何扩大 mean（例如再逐步放开高风险族、或做更强的证书校准）。

