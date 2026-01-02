# 2026-01-02 — LOSO 4-class — EA‑SI‑CHAN‑MULTI‑SAFE：密 λ 网格（尝试降低外推，但出现 S8 负迁移）

目标：按“证书有效性 + 安全选择/回退”主线，验证一个直觉改动：**放弃连续 SPSA**，把下层候选从稀疏 λ 网格扩展为密网格，使证书不需要外推；优先把 `neg_transfer_rate≈0` 保住，再追 mean 提升。

协议固定（可比）：MOABB `BNCI2014_001`，4‑class（`left_hand,right_hand,feet,tongue`），LOSO，`sessions=0train`，`paper_fir`（8–30 Hz, causal FIR order=50），epoch 0.5–3.5s，resample 250 Hz，`CSP n_components=6`，LDA。

---

## 1) 上一次失败分析（post‑mortem）

上一次我们尝试“半联动连续 λ”（`EA‑SI‑CHAN‑SPSA‑SAFE`），mean 有提升但出现明显负迁移（worst-subject 下降），原因是连续搜索会跑到训练未覆盖的 off‑grid λ，证书/guard 外推失效。

对应记录：
- `docs/experiments/20260102_loso4_bnci2014_001_chan_spsa_safe.md:1`
- `docs/experiments/20260102_loso4_bnci2014_001_chan_spsa_safe_warmstart.md:1`

---

## 2) 本次假设与准备怎么做（只动一个杠杆）

**假设**：让候选 λ 的分布与证书训练分布一致（密网格枚举），可以减少外推误判，从而压低负迁移。

**唯一改动杠杆**：只改候选集（下层）  
- 从：`λ ∈ {0.5,1,2}`  
- 到：`λ ∈ {0.25,0.35,0.5,0.7,1,1.4,2}`（rank 固定 `r=21`，其余配置不变）

上层不变：`selector=calibrated_ridge_guard`，fallback 到 EA anchor。

---

## 3) 怎么做的（命令 + 输出目录）

命令：
```bash
_MNE_FAKE_HOME_DIR=$PWD/.mne_home MNE_DATA=$PWD/.mne_data \
conda run -n eeg python run_csp_lda_loso.py \
  --dataset BNCI2014_001 --preprocess paper_fir \
  --events left_hand,right_hand,feet,tongue --sessions 0train \
  --n-components 6 \
  --methods ea-csp-lda,ea-si-chan-multi-safe-csp-lda \
  --si-proj-dim 21 --si-subject-lambda 1 --si-ridge 1e-6 \
  --si-chan-ranks 21 --si-chan-lambdas 0.25,0.35,0.5,0.7,1,1.4,2 \
  --oea-zo-selector calibrated_ridge_guard \
  --oea-zo-calib-max-subjects 0 --oea-zo-calib-seed 0 \
  --oea-zo-calib-guard-threshold 0.5 \
  --no-plots \
  --run-name loso4_bnci2014_001_chan_multi_safe_lam_dense
```

Outputs：
- `outputs/20260102/4class/loso4_bnci2014_001_chan_multi_safe_lam_dense/20260102_method_comparison.csv`
- `outputs/20260102/4class/loso4_bnci2014_001_chan_multi_safe_lam_dense/20260102_results.txt`

---

## 4) 最终结果（主表指标 + 关键被试）

来自 `20260102_method_comparison.csv`：
- `ea-csp-lda` mean acc **0.5320**, worst **0.2569**
- `ea-si-chan-multi-safe-csp-lda`（密网格）mean acc **0.5336**（**+0.15% abs**）, worst **0.2604**
  - `accept_rate = 4/9 = 0.4444`
  - `neg_transfer_rate_vs_ea = 0.1111`

关键失败：**S8** 被误接受后出现显著负迁移（`20260102_results.txt`）：
- EA anchor: **0.71875**
- selected: **0.67014**（Δ = **−0.04861**）
- 但当时证书给出 `guard_p_pos≈0.896` 且 `ridge_pred_improve≈+0.0298`，并选择 `λ=0.35`。

对照（旧网格 `λ={0.5,1,2}` 的 multi-safe，来自 `outputs/20260102/4class/loso4_bnci2014_001_chan_spsa_safe/`）：
- mean acc **0.5463**（+1.43% abs），`neg_transfer_rate_vs_ea = 0.0`

---

## 5) 这次说明了什么（为什么密网格反而更差）

直观上“密网格避免外推”是对的，但本轮结果说明：**候选数变多会放大证书/guard 的校准难度**，出现“相关性更高但仍会误接受”的现象。

证据：
- 本轮 `cert_improve_spearman = 0.4333`（看起来更相关），但仍存在 S8 的大幅负迁移；
- `guard_train_auc_mean` 从旧网格的 ≈0.95 降到 **0.915**，说明 guard 训练稳定性下降。

结论：当前证书并不是“对所有 λ 一视同仁”的稳定风险估计；加入高风险候选（如较小 λ）会触发误接受，从而拉低 mean 并破坏 `neg_transfer≈0` 的目标。

---

## 6) 图（用于定位 S8）

- per-subject Δacc 对比（旧网格 vs 密网格）：`docs/experiments/figures/20260102_loso4_chan_multi_safe_lam_dense_delta_compare.png`
- 密网格：λ vs 真实 Δacc（可见 S8 在 λ=0.35 掉点）：`docs/experiments/figures/20260102_loso4_chan_multi_safe_lam_dense_lambda_vs_true.png`
- 密网格：ridge 预测 vs 真实 Δacc（展示误判）：`docs/experiments/figures/20260102_loso4_chan_multi_safe_lam_dense_pred_vs_true.png`

---

## 7) 下一步（优先把 neg_transfer 压回 0）

在不引入目标标签的前提下，要避免 S8 这类误接受，下一轮应只动一个杠杆做“更严格的安全门控”，例如：
- 对 **低 λ 候选** 设更严格的接受条件（更高 guard 阈值 / 更高最小预测提升 / drift hard gate）
- 或者直接 **移除高风险 λ（如 0.35/0.25）**，先恢复 `neg_transfer≈0` 再做更细的 grid search（分阶段扩大候选集）

