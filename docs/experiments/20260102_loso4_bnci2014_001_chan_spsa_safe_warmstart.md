# 2026-01-02 — LOSO 4-class — EA‑SI‑CHAN‑SPSA‑SAFE（v2：网格 warm‑start + λ trust‑region，仍有少量负迁移）

目标：针对上一轮 `EA‑SI‑CHAN‑SPSA‑SAFE` 的负迁移（证书外推/探索过宽），在**不更换上层证书**的前提下，把连续搜索“安全化”，优先把 `neg_transfer_rate` 压下去。

协议固定（可比）：MOABB `BNCI2014_001`，4‑class（`left_hand,right_hand,feet,tongue`），LOSO，`sessions=0train`，`paper_fir`（8–30 Hz, causal FIR order=50），epoch 0.5–3.5s，resample 250 Hz，`CSP n_components=6`，LDA。

---

## 1) 上一次失败分析（post‑mortem）

上一轮 `EA‑SI‑CHAN‑SPSA‑SAFE`（连续 λ）的问题是：
- mean 提升存在（约 +1.16% abs），但 **`neg_transfer_rate_vs_ea = 0.3333`**，worst‑subject 也下降；
- 典型失效机制：SPSA 会把 λ 推到离散网格之外（甚至极小值），上层 ridge/guard 在训练时只见过 `{0.5,1,2}`，对 off‑grid 候选的外推不可靠，导致误接受。

对应记录：`docs/experiments/20260102_loso4_bnci2014_001_chan_spsa_safe.md:1`

---

## 2) 本次假设与准备怎么做（只动一个杠杆）

**假设**：连续搜索的负迁移主要来自“候选分布偏离证书训练分布”，因此只要把 λ 搜索限制在“网格附近”，并从网格最优点 warm‑start，就能显著降低负迁移。

**唯一改动杠杆（下层搜索策略）**：
- 先在离散网格 `λ∈{0.5,1,2}` 上评估（使用同一套 calibrated ridge/guard 打分），取 predicted‑best 作为 **warm‑start**；
- 连续 SPSA 的 `λ` 搜索改为 **trust‑region**（相对网格做 modest 扩展，例如 `[min/2, max*2]`），避免极端外推；
- 上层证书/guard 不变，仍然 `calibrated_ridge_guard + fallback(EA)`。

---

## 3) 怎么做的（命令 + 输出目录）

命令：
```bash
_MNE_FAKE_HOME_DIR=$PWD/.mne_home MNE_DATA=$PWD/.mne_data \
conda run -n eeg python run_csp_lda_loso.py \
  --dataset BNCI2014_001 --preprocess paper_fir \
  --events left_hand,right_hand,feet,tongue --sessions 0train \
  --n-components 6 \
  --methods ea-csp-lda,ea-si-chan-multi-safe-csp-lda,ea-si-chan-spsa-safe-csp-lda \
  --si-proj-dim 21 --si-subject-lambda 1 --si-ridge 1e-6 \
  --si-chan-ranks 21 --si-chan-lambdas 0.5,1,2 \
  --oea-zo-selector calibrated_ridge_guard \
  --oea-zo-calib-max-subjects 0 --oea-zo-calib-seed 0 \
  --oea-zo-calib-guard-threshold 0.5 \
  --oea-zo-iters 30 --oea-zo-lr 0.5 --oea-zo-mu 0.1 --oea-zo-seed 0 \
  --no-plots \
  --run-name loso4_bnci2014_001_chan_spsa_safe_warmstart
```

Outputs：
- `outputs/20260102/4class/loso4_bnci2014_001_chan_spsa_safe_warmstart/20260102_method_comparison.csv`
- `outputs/20260102/4class/loso4_bnci2014_001_chan_spsa_safe_warmstart/20260102_results.txt`
- `outputs/20260102/4class/loso4_bnci2014_001_chan_spsa_safe_warmstart/20260102_predictions_all_methods.csv`

---

## 4) 结果（主表 + 掉点被试）

来自 `20260102_method_comparison.csv`：
- `ea-csp-lda` mean acc **0.5320**, worst **0.2569**
- `ea-si-chan-multi-safe-csp-lda` mean acc **0.5463**（**+1.43% abs**）, worst **0.2604**, `neg_transfer_rate_vs_ea=0.0`
- `ea-si-chan-spsa-safe-csp-lda`（v2）mean acc **0.5440**（**+1.20% abs**）, worst **0.2326**, `neg_transfer_rate_vs_ea=0.1111`

本轮 `EA‑SI‑CHAN‑SPSA‑SAFE` 的 accept/掉点（来自 `20260102_results.txt`）：
- `accept_rate = 6/9 = 0.6667`
- 负迁移被试：**S2**（EA 0.2569 → 0.2326，Δ=−0.0243），但 guard 仍给出 `p_pos≈0.96`、ridge 预测 `+0.054`。

**解释**：这说明“限制外推”能减少崩盘，但仍无法彻底保证安全；在 S2 这种 case，证书对 `λ≈0.47` 的局部外推仍会误判（并把它排在网格点 `λ=0.5` 之前）。

---

## 5) 图（证书有效性与失败定位）

- Δacc 对比（multi‑safe vs spsa‑safe v2）：`docs/experiments/figures/20260102_loso4_chan_spsa_safe_warmstart_delta_compare.png`
- SPSA 选到的 λ vs 真实 Δacc：`docs/experiments/figures/20260102_loso4_chan_spsa_safe_warmstart_lambda_vs_true.png`
- ridge 预测提升 vs 真实提升：`docs/experiments/figures/20260102_loso4_chan_spsa_safe_warmstart_pred_vs_true.png`

---

## 6) 下一步（围绕“证书有效性”，最可能解决 S2）

当前最关键的事实：**连续 λ 搜索会诱导“off‑grid 局部最优”，而 ridge/guard 没有在这些 off‑grid 候选上被监督过**。

因此下一轮建议（仍然一次只动一个杠杆）优先做其一：

1) **把连续搜索改为“密网格”**（最省事且最稳）：把 `--si-chan-lambdas` 扩展为更密的集合（如 `0.25,0.35,0.5,0.7,1,1.4,2`），让证书训练与选择都在“见过的候选分布”里完成；先追 `neg_transfer_rate≈0` 再追 mean。

2) **证书训练分布对齐连续候选**（更像方法创新，但更耗时）：在 pseudo‑target 校准阶段，为每个 inner fold 额外采样一小批 off‑grid λ（例如在 `[0.25,2]` 内 log‑uniform 采样 5 个），把这些样本也加入 ridge/guard 训练，让证书在连续 λ 上不再外推。

