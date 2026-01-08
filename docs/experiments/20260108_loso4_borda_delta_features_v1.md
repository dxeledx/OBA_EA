# 20260108 — BNCI2014\_001 4-class LOSO — EA-anchor 相对化（delta features）尝试（结论：未带来最终性能增益）

## 0) 上一轮（对照）与本轮目标
**协议固定**：BNCI2014\_001（BCI-IV 2a），4-class（left/right/feet/tongue），严格 LOSO，同预处理 `paper_fir`，同模型 CSP(6)+LDA，同指标（macro）。

**上一轮最佳对照（Borda, absolute features）**：
- 输出：`outputs/20260107/4class/loso4_actionset_stack_familyblend_k20_probe_minimp001_borda_fix_v1/`
- `ea-csp-lda` mean acc = **0.5320**
- `ea-stack-multi-safe-csp-lda` mean acc = **0.5498**（+1.77pp）
- `neg_transfer_rate_vs_ea = 0.0`, `accept_rate = 0.6667`
- `oracle_mean = 0.5633`, `gap_sel_mean = 0.0135`

**本轮目标**：只动一个杠杆，把 calibrated ridge/guard 的输入特征从“绝对无标签特征”替换为“相对 EA(anchor) 的变化（delta features）”，期待：
1) 提升候选集合内证书相关性（`rho_ridge_mean`, `rho_guard_mean`），
2) 缩小 `oracle gap`（`gap_sel_mean`），
3) 保持低负迁移率。

## 1) 单杠杆改动（one lever）
新增并启用 `--stack-feature-set stacked_delta`：
- 对每个候选 record \(c\) 与 anchor(EA) record \(c_0\)，构造
  - **非 meta 特征**：\(\phi_\Delta(c)=\phi(c)-\phi(c_0)\)
  - **meta 特征**（family one-hot / rank / lambda）：仍保留绝对值，不做差分
- 直觉：我们训练的目标本来就是 \(\Delta\)acc over EA（伪目标域上），用 anchor-relative 特征可降低 subject-specific 的“绝对尺度漂移”。

代码位置（核心实现）：
- `csp_lda/certificate.py`：`stacked_candidate_features_delta_from_records` + selector 支持 `feature_set="stacked_delta"`
- `csp_lda/evaluation.py`：stack multi-safe 的校准与选择阶段用 `stack_feature_set`
- `run_csp_lda_loso.py`：新增 CLI 参数 `--stack-feature-set`

## 2) 复现实验（严格 LOSO）
- Commit：`4d38b38772422a8d2cb88bd9084e187f67ae803d`
- Command：见 `outputs/20260108/4class/loso4_actionset_stack_familyblend_k20_probe_minimp001_borda_deltafeat_v1/20260108_results.txt` 顶部 `Command:` 行
- Outputs：`outputs/20260108/4class/loso4_actionset_stack_familyblend_k20_probe_minimp001_borda_deltafeat_v1/`

## 3) 主表结果（同协议可比）
来自：`outputs/20260108/4class/loso4_actionset_stack_familyblend_k20_probe_minimp001_borda_deltafeat_v1/20260108_method_comparison.csv`

- `ea-csp-lda`：mean acc **0.5320**, worst **0.2569**
- `ea-stack-multi-safe-csp-lda`（delta features）：
  - mean acc **0.5490**（+1.70pp vs EA）
  - worst **0.2535**（略低于上一轮 0.2569）
  - accept_rate **0.6667**（与上一轮相同）
  - neg_transfer_rate_vs_ea **0.1111**（S2：0.2569 → 0.2535，-0.35pp）

与上一轮（absolute features）对比：
- mean acc：**0.5498 → 0.5490**（-0.08pp）
- `gap_sel_mean`：**0.0135 → 0.0143**（略变差）
- `neg_transfer_rate_vs_ea`：**0.0 → 0.1111**（出现轻微负迁移）

## 4) 诊断：证书相关性 vs 最终选择
来自：`scripts/analyze_candidate_certificates.py`

本轮（delta features）：
- `rho_ridge_mean = 0.3313`（显著高于上一轮 ~0.05）
- `rho_guard_mean = 0.2283`（略高于上一轮 ~0.18）
- `oracle_mean = 0.5633`, `gap_sel_mean = 0.0143`

结论：**证书在候选集合内的相关性确实变强了**，但仍未把最终选择推近 oracle，且对 S2 出现轻微负迁移。

图与表：
- `docs/experiments/figures/20260108_loso4_borda_deltafeat_v1/loso4_borda_deltafeat_v1_ridge_vs_true.png`
- `docs/experiments/figures/20260108_loso4_borda_deltafeat_v1/loso4_borda_deltafeat_v1_guard_vs_true.png`
- `docs/experiments/figures/20260108_loso4_borda_deltafeat_v1/loso4_borda_deltafeat_v1_oracle_gap.png`
- `docs/experiments/figures/20260108_loso4_borda_deltafeat_v1/candidate_certificate_table.csv`

## 5) 本轮结论（为什么“证书更有效”但“性能没涨”）
1) **证书相关性提升 ≠ 选择一定更好**：Borda/guard gate 的组合仍可能在边界样本（例如 S2）上放行“预测提升很小但真实无益/略负”的候选。
2) **oracle gap 仍存在**：`oracle_mean` 不变但 `gap_sel_mean` 未缩小，说明“更好的候选仍在集合里，但选择器仍会错过一部分”。

## 6) 下一步（仍按 one lever 原则）
本轮已经证明“delta features 能提升证书相关性”，但要把它转化为更高 mean 且 0 负迁移，下一轮更划算的单杠杆是：
- 在不改候选集的情况下，把“允许小幅 predicted improvement 的候选被选中”变得更保守（例如统一的 `min_pred_improve`/margin 机制，而不只对 FBCSP/TSA），目标是把 S2 这种微小负迁移压掉，再看 mean 是否回升并逼近 oracle。

