# 20260108 — BNCI2014\_001 4-class LOSO — stacked\_delta + 全候选通用 min\_pred\_improve gate（结论：压掉负迁移，均值略降）

## 0) 上一轮（对照）与本轮目标
**协议固定**：BNCI2014\_001（BCI-IV 2a），4-class（left/right/feet/tongue），严格 LOSO，同预处理 `paper_fir`，同模型 CSP(6)+LDA，同指标（macro）。

**上一轮（delta features，无全局 min\_pred gate）**：
- 输出：`outputs/20260108/4class/loso4_actionset_stack_familyblend_k20_probe_minimp001_borda_deltafeat_v1/`
- `ea-csp-lda` mean acc = **0.5320**
- `ea-stack-multi-safe-csp-lda` mean acc = **0.5490**（+1.70pp）
- `accept_rate = 0.6667`
- `neg_transfer_rate_vs_ea = 0.1111`（S2：0.2569 → 0.2535，-0.35pp）

**本轮目标（one lever）**：在 **保留 `--stack-feature-set stacked_delta`** 的前提下，引入一个“全候选通用的 min\_pred\_improve gate”，把 S2 这种 **小幅负迁移** 直接压掉（目标：`neg_transfer_rate≈0`），同时尽量保住 mean。

## 1) 单杠杆改动（one lever）
新增 `--stack-safe-min-pred-improve`（全候选 gate）：
- 对所有 **非 identity** 候选，要求其（blended）`ridge_pred_improve >= min_pred`，否则 **强制回退到 EA(anchor)**。
- 直觉：候选数变多时（multiple testing），会出现“证书/guard 假阳性”。这个 gate 等价于一个轻量的“抗多候选修正”：当模型只能预测很小的提升时，宁可不动（回退 EA）。

代码位置：
- `run_csp_lda_loso.py`：新增 CLI 参数 `--stack-safe-min-pred-improve`
- `csp_lda/evaluation.py`：在 stack-multi-safe 选择后增加全局 gate + 记录 `stack_multi_min_pred_blocked`

## 2) 复现实验（严格 LOSO）
- Code commit（运行时）：`ad516f4`
- Command：见 `outputs/20260108/4class/loso4_actionset_stack_familyblend_k20_probe_minimp001_borda_deltafeat_minpred002_v1_01/20260108_results.txt` 顶部 `Command:` 行
- Outputs：`outputs/20260108/4class/loso4_actionset_stack_familyblend_k20_probe_minimp001_borda_deltafeat_minpred002_v1_01/`

本轮关键新增参数：
- `--stack-safe-min-pred-improve 0.02`
- `--stack-feature-set stacked_delta`（保持不变）

## 3) 主表结果（同协议可比）
来自：`outputs/20260108/4class/loso4_actionset_stack_familyblend_k20_probe_minimp001_borda_deltafeat_minpred002_v1_01/20260108_method_comparison.csv`

- `ea-csp-lda`：mean acc **0.5320**, worst **0.2569**
- `ea-stack-multi-safe-csp-lda`（delta features + global min\_pred gate）：
  - mean acc **0.5475**（+1.54pp vs EA）
  - worst **0.2569**（与 EA 相同）
  - accept_rate **0.4444**
  - neg\_transfer\_rate\_vs\_ea **0.0**

与上一轮（无全局 min\_pred gate，仍是 stacked\_delta）对比：
- mean：**0.5490 → 0.5475**（-0.15pp）
- `neg_transfer_rate`：**0.1111 → 0.0**（S2 负迁移被压掉）
- accept\_rate：**0.6667 → 0.4444**（更保守，回退更多被试）

## 4) 逐被试现象（压掉负迁移的代价）
从 `diagnostics/.../subject_XX/candidates.csv` 统计：
- 本轮只接受 4/9：S3、S5、S7、S8（其余回退 EA）
- 对 S2：直接回退 EA（从上一轮的 -0.35pp 变为 0）

这说明：**全局 min\_pred gate 确实能“止血”负迁移**，但会以 **accept rate 下降** 为代价，从而 mean 略降。

## 5) 候选集诊断：oracle gap 与证书相关性
来自：`python3 scripts/analyze_candidate_certificates.py ...`

- `oracle_mean = 0.5633`
- `gap_sel_mean = 0.0158`（比上一轮 0.0143 略大，原因：回退更多导致离 oracle 更远）
- `rho_ridge_mean = 0.3313`, `rho_guard_mean = 0.2283`（与上一轮 stacked\_delta 一致：本轮没动证书特征/校准，只加了 gate）

结论：本轮改动的作用是 **把“容易翻车的微小 predicted gain”拒掉**，而不是提升证书相关性/缩小 oracle gap。

## 6) 图表（Paper-grade 证据）
输出目录：`docs/experiments/figures/20260108_loso4_borda_deltafeat_minpred002_v1/`
- `loso4_borda_deltafeat_minpred002_v1_ridge_vs_true.png`
- `loso4_borda_deltafeat_minpred002_v1_guard_vs_true.png`
- `loso4_borda_deltafeat_minpred002_v1_oracle_gap.png`
- `loso4_borda_deltafeat_minpred002_v1_delta.png`
- `loso4_borda_deltafeat_minpred002_v1_family.png`

## 7) 本轮结论与下一步（仍按 one lever）
**本轮结论**：全候选通用 `min_pred_improve` gate 在 4-class LOSO 下实现了 **0 负迁移**，但 **mean 略降、accept rate 明显下降**。

**下一步（建议作为下一轮 one lever）**：把 `--stack-safe-min-pred-improve` 做一个小范围 sweep（例如 0.00/0.01/0.015/0.02），寻找“neg\_transfer≈0 且 mean 最大”的甜点区，再决定是否需要引入更强的“证书校准/学习”（而不是继续加 gate）。

