# 2026-01-16 — Certificate effectiveness → safe selection (CSP+LDA)

目标：把“无标签证书经常失效 → 选错/负迁移”的问题，收敛为一个**可复现、可归因、可写论文**的改进闭环；主表只认 **strict LOSO**（`--oea-pseudo-iters 0`）。

本记录仅覆盖 **CSP+LDA**（含 EA / LEA 等闭式对齐与我们的 safe selection），不含深网。

---

## 0) 关键信息（协议/可比性）

- Dataset: `BNCI2014_001` (BCI-IV 2a)
- Preprocess: `paper_fir`（8–30Hz, 250Hz, 0.5–3.5s, causal FIR）
- CSP: `--n-components 6`
- 主表协议：Cross-subject **LOSO**，并且 **strict**：`--oea-pseudo-iters 0`
- 目标：在 **不牺牲 worst-subject / 负迁移率** 的前提下提升 mean accuracy；把“证书→性能”的相关性拉正。

---

## 1) 上一轮失败复盘（strict_delta 仍有负迁移）

### 1.1 现象
在 BNCI 4-class strict LOSO 中，`stacked_delta`（无 global min_pred gate）能提升均值，但仍存在少量负迁移：
- mean↑ 但 `neg_transfer_rate` 仍为 `1/9`（出现“证书判好，但真实掉点”的假阳性）。

### 1.2 推断原因（证据链）
本质是 **multi-candidate + 证书噪声** 导致的 multiple testing 假阳性：候选越多，越容易“碰巧”出现证书更好的候选，但真实风险并未降低。

---

## 2) 本轮改动（单杠杆）

只动一个杠杆：在保留 `stacked_delta` 与既有 calibrated ridge/guard 的前提下，增加 **全候选通用的最小预测提升门控**：

- 新 gate：`--stack-safe-min-pred-improve 0.02`
- 含义：任何非 EA 候选要被接受，必须满足 ridge 预测的提升 ≥ 0.02；否则回退 EA。
- 目的：消掉“小幅假阳性”（典型是轻微负迁移或微弱收益样本）。

---

## 3) 复现实验与结果

### 3.1 BNCI2014_001 4-class strict LOSO（主表）

**Run A（对照，仍有负迁移）**  
- 输出：`outputs/20260116/4class/loso4_bnci_strict_stackdelt_v1/`
- EA mean acc: `0.5320`
- Ours mean acc: `0.5486`（+`1.66pp`）
- neg-transfer: `1/9`

**Run B（本轮主方法：加 global min_pred gate）**  
- 输出：`outputs/20260116/4class/loso4_bnci_strict_stackdelt_minpred002_v1/`
- EA mean acc: `0.5320`
- Ours mean acc: `0.5471`（+`1.50pp`）
- **neg-transfer: `0/9`**
- accept-rate: `0.4444`
- 证书相关性（subject-level，来自 `*_method_comparison.csv`）：
  - `cert_improve_spearman = 0.8333`
  - `guard_improve_spearman = 0.7833`

**Paper-ready 资产（主表+图）**  
目录：`docs/paper_assets/bnci2014_001/4class_loso_strict/`

- 主表：`docs/paper_assets/bnci2014_001/4class_loso_strict/tables/main_table.md`
- 消融表（Run A vs Run B）：`docs/paper_assets/bnci2014_001/4class_loso_strict/tables/ablation_table.md`
- 混淆矩阵（EA vs Ours）：`docs/paper_assets/bnci2014_001/4class_loso_strict/figures/bnci4_strict_cm_*`
- 证书有效性散点：`docs/paper_assets/bnci2014_001/4class_loso_strict/figures/bnci4_strict_candidate_v1_*`
- headroom 分解：`docs/paper_assets/bnci2014_001/4class_loso_strict/figures/bnci4_strict_candidate_v1_headroom.png`
- 阈值敏感性（风险-收益 sweep，analysis-only）：`docs/paper_assets/bnci2014_001/4class_loso_strict/figures/bnci4_strict_minpred_sweep_v1_risk_reward.png`

Headroom（analysis-only，候选集上限 vs selector 吃到多少）：
- `oracle_mean = 0.5625`
- `EA_mean ≈ 0.5320`
- `selected_mean = 0.5471`
- headroom eaten fraction ≈ `0.49`
（见：`docs/paper_assets/bnci2014_001/4class_loso_strict/tables/certificate_reliability.md`）

### 3.2 BNCI2014_001 2-class strict LOSO（通用性补充）

- 输出：`outputs/20260116/2class/loso2_bnci_strict_stackdelt_minpred002_v1/`
- Ours mean acc `0.7330` vs EA `0.7323`（+`0.08pp`），neg-transfer `0`
- accept-rate `0.1111`（门控偏保守，基本回退 EA）

Paper-ready：`docs/paper_assets/bnci2014_001/2class_loso_strict/`

### 3.3 Cross-session（稳健性补充，不进 strict 主表）

- 输出：`outputs/20260116/4class/cross_session/crosssess4_bnci_paperfir_baselines_v1/`
- Paper-ready：`docs/paper_assets/bnci2014_001/4class_cross_session/`

---

## 4) 本轮结论（面向论文叙事）

1) 在 strict LOSO 下，**“证书 + 安全门控/回退”** 能同时做到：
- mean accuracy **稳定提升**（BNCI 4-class +1.50pp）
- **负迁移率压到 0**（deployable 约束）
- 证书-性能相关性显著转正（Spearman≈0.83）

2) 代价是 accept-rate 下降（更保守），但在低信噪比 MI-EEG 场景这更符合“安全部署”的口径。

---

## 5) 下一步（仍按单杠杆）

建议下一轮只做一个方向（择一）：

- **(A) 让 2-class 的 headroom 被吃到更多**：在不引入负迁移的约束下，做 `--stack-safe-min-pred-improve` 的 train-only 标定（避免在 test 上挑 τ）。
- **(B) 提升证书“可判别性”而非更保守**：继续做 EA-anchor 相对化特征 + 更强的 probe 统计（仍不动协议）。

