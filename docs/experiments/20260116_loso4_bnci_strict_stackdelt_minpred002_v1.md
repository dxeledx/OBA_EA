# 20260116 — BNCI2014_001 (4-class) LOSO — strict + stacked_delta + global min_pred gate

## 0) Post-mortem (why this change)

上一次严格设置（`--oea-pseudo-iters 0`）下的 `stacked_delta` 版本：

- 均值 **+1.66pp**，但仍有 **1/9 负迁移**（`neg_transfer_rate_vs_ea=0.1111`，典型为 S2 小幅掉点）。
- 根因是“无标签证书/校准的假阳性”：`guard/probe` 认为是正提升，但真实 `Δacc<0`。

因此本轮只动一个杠杆：加 **全候选通用的 min_pred_improve gate**（多候选下的轻量多重检验校正）。

直觉：当候选集变大时，偶然的“看起来更好”会变多；我们要求校准器输出 **至少达到 τ 的预测收益** 才允许离开 EA(anchor)。

---

## 1) Protocol (strict LOSO)

- Dataset: `MOABB BNCI2014_001`
- Task: 4-class (`left_hand,right_hand,feet,tongue`)
- Sessions: `0train`
- Preprocess: `paper_fir`
- CSP components: `6`
- Strict: `--oea-pseudo-iters 0`

---

## 2) Change (one lever)

- Added: `--stack-safe-min-pred-improve 0.02`

其余协议与开关保持不变（同 preproc / 同指标 / 同候选集 / 同 selector）。

---

## 3) Command & artifacts

```bash
conda run -n eeg python run_csp_lda_loso.py \
  --dataset BNCI2014_001 --preprocess paper_fir \
  --events left_hand,right_hand,feet,tongue --sessions 0train \
  --n-components 6 \
  --methods ea-csp-lda,ea-stack-multi-safe-csp-lda \
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
  --stack-safe-min-pred-improve 0.02 \
  --diagnose-subjects 1,2,3,4,5,6,7,8,9 \
  --no-plots \
  --run-name loso4_bnci_strict_stackdelt_minpred002_v1
```

Outputs:

- `outputs/20260116/4class/loso4_bnci_strict_stackdelt_minpred002_v1/20260116_results.txt`
- `outputs/20260116/4class/loso4_bnci_strict_stackdelt_minpred002_v1/20260116_method_comparison.csv`
- `outputs/20260116/4class/loso4_bnci_strict_stackdelt_minpred002_v1/20260116_predictions_all_methods.csv`
- Candidate diagnostics: `outputs/20260116/4class/loso4_bnci_strict_stackdelt_minpred002_v1/diagnostics/`

Figures:

- `docs/experiments/figures/20260116_loso4_bnci_strict_stackdelt_minpred002_v1/bnci4_strict_stackdelt_minpred002_v1_delta.png`
- `docs/experiments/figures/20260116_loso4_bnci_strict_stackdelt_minpred002_v1/bnci4_strict_stackdelt_minpred002_v1_ridge_vs_true.png`
- `docs/experiments/figures/20260116_loso4_bnci_strict_stackdelt_minpred002_v1/bnci4_strict_stackdelt_minpred002_v1_guard_vs_true.png`
- `docs/experiments/figures/20260116_loso4_bnci_strict_stackdelt_minpred002_v1/bnci4_strict_stackdelt_minpred002_v1_family.png`

---

## 4) Results (main table items)

From `20260116_method_comparison.csv`:

- `ea-csp-lda`: mean acc **0.5320**, worst **0.2569**
- `ea-stack-multi-safe-csp-lda` (+ stacked_delta + min_pred=0.02):
  - mean acc **0.5471** (**+1.50pp**)
  - worst acc **0.2569** (回到 EA 的 worst)
  - **neg_transfer_rate_vs_ea = 0.0 (0/9)**
  - accept_rate **0.4444 (4/9)**
  - cert_improve_spearman **0.8333**（证书-真实收益一致性显著为正）

结论：该 gate **如预期消掉了 S2 类的小幅负迁移**，代价是 accept_rate 降低、mean 提升略变小（1.66pp → 1.50pp）。

---

## 5) Notes / next step

如果下一步希望在保持 `neg_transfer≈0` 的同时恢复更多 accept（提升 mean），优先路线不是再扫大量超参，而是：

- 把这个全局 gate 从“只看 ridge_pred”升级为与 `borda_ridge_probe` 一致的 **joint predicted gain**（例如 `max(ridge_pred_improve, probe_improve)` 的阈值），避免误杀“ridge 负但 probe 强”的正样本（如 S1）。

（下一轮只动一个杠杆：gate 的定义；其余不变。）

