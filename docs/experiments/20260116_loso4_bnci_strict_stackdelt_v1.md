# 20260116 — BNCI2014_001 (4-class) LOSO — strict (pseudo-iters=0) + stacked_delta

## 0) Motivation / post-mortem (previous best)

最近一轮在 `BNCI2014_001` 4-class LOSO 上，`ea-stack-multi-safe-csp-lda` 能做到 **+1.77pp mean acc** 且 **0 负迁移**（见 `docs/experiments/results_registry.csv`，run=20260107）。

但有两个需要立即修正/验证的问题：

1) **严格协议口径**：主表严格 LOSO 需要 `--oea-pseudo-iters 0`（不在目标被试上做伪标签迭代/闭式旋转）。因此本轮把 pseudo-iters 显式设为 0。
2) **命名可比性**：历史名字 `rpa-csp-lda/tsa-csp-lda` 实际是 LEA 视图（log-Euclidean whitening）及其上的闭式 target rotation 变体。本轮起使用 **paper-faithful naming**：`lea-csp-lda/lea-rot-csp-lda`（旧名保留为 alias）。

同时，我们想验证一个核心假设：

> **证书/校准特征改为 EA-anchor 相对化（`stacked_delta`）**，能提高证书-真实收益一致性（rho 变正、oracle gap 缩小），并在严格设置下依然保持提升。

本轮只动一个主杠杆：`--stack-feature-set stacked_delta`（证书特征从绝对值 → 相对 EA 的变化）。

---

## 1) Protocol (strict LOSO)

- Dataset: `MOABB BNCI2014_001` (BCI-IV 2a)
- Task: 4-class (`left_hand,right_hand,feet,tongue`)
- Sessions: `0train`
- Preprocess: `paper_fir` (8–30 Hz, causal FIR, resample 250 Hz, tmin=0.5, tmax=3.5)
- CSP components: `6`
- Strict setting: `--oea-pseudo-iters 0`

---

## 2) Command & artifacts

Command (exact):

```bash
conda run -n eeg python run_csp_lda_loso.py \
  --dataset BNCI2014_001 --preprocess paper_fir \
  --events left_hand,right_hand,feet,tongue --sessions 0train \
  --n-components 6 \
  --methods csp-lda,fbcsp-lda,ea-csp-lda,ea-fbcsp-lda,lea-csp-lda,riemann-mdm,ts-lr,rpa-mdm,rpa-ts-lr,ea-stack-multi-safe-csp-lda \
  --oea-pseudo-iters 0 \
  --stack-feature-set stacked_delta \
  --stack-candidate-families ea,fbcsp,rpa,chan \
  --si-proj-dim 21 --si-subject-lambda 1 --si-ridge 1e-6 \
  --si-chan-ranks 21 --si-chan-lambdas 0.25,0.35,0.5,0.7,1,1.4,2 \
  --oea-zo-selector calibrated_stack_ridge_guard_borda \
  --oea-zo-calib-guard-threshold 0.5 \
  --stack-safe-anchor-guard-delta 0.05 \
  --stack-safe-anchor-probe-hard-worsen -0.01 \
  --stack-safe-fbcsp-guard-threshold 0.95 \
  --stack-safe-fbcsp-min-pred-improve 0.05 \
  --stack-safe-fbcsp-drift-delta 0.15 \
  --stack-calib-per-family --stack-calib-per-family-mode blend --stack-calib-per-family-shrinkage 20 \
  --no-plots \
  --run-name loso4_bnci_strict_stackdelt_v1
```

Outputs:

- `outputs/20260116/4class/loso4_bnci_strict_stackdelt_v1/20260116_results.txt`
- `outputs/20260116/4class/loso4_bnci_strict_stackdelt_v1/20260116_method_comparison.csv`
- `outputs/20260116/4class/loso4_bnci_strict_stackdelt_v1/20260116_predictions_all_methods.csv`

Figures (paper-ready):

- `docs/experiments/figures/20260116_loso4_bnci_strict_stackdelt_v1/bnci4_strict_stackdelt_v1_delta.png`
- `docs/experiments/figures/20260116_loso4_bnci_strict_stackdelt_v1/bnci4_strict_stackdelt_v1_ridge_vs_true.png`
- `docs/experiments/figures/20260116_loso4_bnci_strict_stackdelt_v1/bnci4_strict_stackdelt_v1_guard_vs_true.png`
- `docs/experiments/figures/20260116_loso4_bnci_strict_stackdelt_v1/bnci4_strict_stackdelt_v1_family.png`

---

## 3) Main results (strict LOSO, macro avg)

From `20260116_method_comparison.csv`:

- `ea-csp-lda`: mean acc **0.5320**, worst **0.2569**
- `ea-stack-multi-safe-csp-lda` (`stacked_delta`): mean acc **0.5486** (**+1.66pp**), worst **0.2535**
  - neg_transfer_rate_vs_ea: **0.1111** (1/9)
  - accept_rate: **0.6667**
  - cert_improve_spearman: **0.6167**
  - guard_improve_spearman: **0.4500**

结论（本轮）：在 **严格 pseudo-iters=0** 的设置下，`stacked_delta` 仍能带来 **~+1.7pp** 的均值提升，但 **仍存在 1/9 的小幅负迁移**（worst-subject 也略差于 EA）。

---

## 4) Diagnosis (证书从“更有效”到“仍会选错”的证据)

已观察到：

- `ridge_pred_improve` 与真实 `Δacc` 的相关性为正（见 `*_ridge_vs_true.png`），说明 **证书有效性整体在改善**；
- 但仍有个别 subject（例如 S2）出现 **predicted improve > 0 但真实 Δacc < 0**，说明仍有 **假阳性**，需要更强的“可拒绝负迁移” gate。

---

## 5) Next step (只动一个杠杆)

下一轮优先只加一个全局 safety gate（不改训练/预处理/候选集）：

- **Global min-pred-improve gate**：对所有非 EA 候选要求 `ridge_pred_improve >= τ`，否则回退 EA（目标：把 S2 这类小幅负迁移压到 0）
  - 对应开关：`--stack-safe-min-pred-improve ...`（从小 τ 扫到能做到 neg_transfer≈0 的最小值）

完成后再考虑叠加更强的 family-specific gate（例如 FBCSP 的 `pred_disagree` gate）。

