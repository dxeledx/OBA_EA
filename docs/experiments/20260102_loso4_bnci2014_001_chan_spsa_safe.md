# 2026-01-02 — LOSO 4-class — EA‑SI‑CHAN‑SPSA‑SAFE（半联动：连续 λ 搜索 + calibrated ridge/guard + fallback）

目标：验证“半联动”是否能在不破坏安全性的前提下，突破 `EA‑SI‑CHAN‑MULTI‑SAFE` 的离散网格上限。

本轮协议固定（可比）：MOABB `BNCI2014_001`，4‑class（`left_hand,right_hand,feet,tongue`），LOSO，`sessions=0train`，`paper_fir`（8–30 Hz, causal FIR order=50），epoch 0.5–3.5s，resample 250 Hz，`CSP n_components=6`，LDA。

---

## 1) 上一次结果/问题（post‑mortem）

上一个“可写进汇报主线”的结果是：
- `ea-si-chan-multi-safe-csp-lda` mean acc **0.5463**（vs `ea-csp-lda` 0.5320，**+1.43% abs**）
- `neg_transfer_rate_vs_ea = 0.0`（安全）
- `accept_rate = 5/9 = 0.5556`（较保守）

瓶颈：下层候选集来自离散网格（`λ ∈ {0.5,1,2}`），可能错过更优的连续超参；均值提升虽稳定但幅度偏小。

---

## 2) 本次假设与准备怎么做（只动一个杠杆）

**主假设**：把下层从离散网格改为连续参数搜索（连续 `λ`），能够扩大候选覆盖（减少 “search gap”），进而提升 mean accuracy。

**唯一改动杠杆**：下层候选生成方式  
- 从：`EA‑SI‑CHAN‑MULTI‑SAFE` 的离散候选 `{A(I), A(r,λ)}`（网格枚举）  
- 到：`EA‑SI‑CHAN‑SPSA‑SAFE` 的连续候选（在目标被试上用 SPSA 在 `φ=log λ` 上迭代，生成轨迹候选 `A(λ_t)`；上层仍用同一套 calibrated ridge/guard 打分并做 fallback）

上层（不变）：`selector=calibrated_ridge_guard`，并保持 **EA anchor fallback** 以避免崩盘。

---

## 3) 怎么做的（命令 + 输出目录）

> 注：本轮为了观察“半联动”本身的效果，guard 阈值仍沿用之前的 `0.5`（较松）；这一点在 5) 的失败分析里会解释其后果。

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
  --run-name loso4_bnci2014_001_chan_spsa_safe
```

Outputs：
- `outputs/20260102/4class/loso4_bnci2014_001_chan_spsa_safe/20260102_method_comparison.csv`
- `outputs/20260102/4class/loso4_bnci2014_001_chan_spsa_safe/20260102_results.txt`
- `outputs/20260102/4class/loso4_bnci2014_001_chan_spsa_safe/20260102_predictions_all_methods.csv`

---

## 4) 结果（主表指标 + 证书有效性 + 关键被试）

来自 `20260102_method_comparison.csv`：
- `ea-csp-lda` mean acc **0.5320**, worst **0.2569**
- `ea-si-chan-multi-safe-csp-lda` mean acc **0.5463**（**+1.43% abs**）, worst **0.2604**, `neg_transfer_rate_vs_ea=0.0`
- `ea-si-chan-spsa-safe-csp-lda` mean acc **0.5436**（**+1.16% abs**）, worst **0.2396**, `neg_transfer_rate_vs_ea=0.3333`

本轮 `EA‑SI‑CHAN‑SPSA‑SAFE` 额外统计（由 `20260102_results.txt` 的 per-subject 表计算）：
- `accept_rate = 7/9 = 0.7778`
- accepted 中出现负迁移（相对 EA 的真实 Δacc < 0）：**S1/S2/S8**（每个 −0.01736）

证书有效性（本轮 `EA‑SI‑CHAN‑SPSA‑SAFE`，跨被试 Spearman）：
- Spearman(`ridge_pred_improve`, `true_improve`) ≈ **0.270**
- 但在 **pred_improve 很小**（例如 0.007–0.008）时仍会误接受，导致负迁移。

---

## 5) 这次说明了什么（失败原因 + 直接结论）

结论非常清晰：**半联动（连续 λ 搜索）能提高 mean，但破坏了 “safe” 的关键性质**（出现负迁移、worst-subject 下降）。

机制性原因（结合本轮数据）：
1) 连续 SPSA 会探索到网格外（例如 `λ≈0.006`），证书的外推不稳定；即使 `ridge_pred_improve>0`，也可能对应真实负迁移。  
2) 当前接收条件偏松（`guard_threshold=0.5` 且只要求 `pred_improve>0`），当 `pred_improve` 接近 0 时，误接受概率高；S1/S2/S8 就属于“微弱正预测→真实负提升”。

因此：本轮不建议把 `EA‑SI‑CHAN‑SPSA‑SAFE` 作为主方法；当前主线仍应以 `EA‑SI‑CHAN‑MULTI‑SAFE`（neg_transfer≈0）为主。

---

## 6) 图（用于汇报/诊断）

本轮新增三张诊断图（都在 `docs/experiments/figures/`）：
- per-subject Δacc 对比（multi-safe vs spsa-safe）：`docs/experiments/figures/20260102_loso4_chan_spsa_safe_delta_compare.png`
- 选择的 λ vs 真实 Δacc：`docs/experiments/figures/20260102_loso4_chan_spsa_safe_lambda_vs_true.png`
- 证书预测提升 vs 真实提升：`docs/experiments/figures/20260102_loso4_chan_spsa_safe_pred_vs_true.png`

---

## 7) 下一步怎么做（把连续搜索重新“安全化”）

下一轮仍然只动一个杠杆：**更严格的接收/回退**（不改下层连续搜索本身），目标是把 `neg_transfer_rate` 压回接近 0：
- 把接收条件从 `pred_improve>0` 提升为 `pred_improve ≥ ε`（例如 ε=0.01）
- 提高 `guard_threshold`（例如 0.85），并启用 drift hard gate（若已实现）
- 对 λ 加 trust region（限制在训练时见过的范围附近，例如 `[0.3, 3]` 或对 `log λ` 限幅）

