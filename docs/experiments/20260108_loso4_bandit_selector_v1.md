# 20260108 — BNCI2014\_001（BCI-IV 2a）4-class LOSO — 验证 RL-like Bandit Selector（负结果）

## 0) 上一轮结论（作为本轮对照）
**目标任务**：BNCI2014\_001 四分类（left/right/feet/tongue），严格 LOSO，同预处理 `paper_fir`、同指标（macro）。

**上一轮最佳（主线）**：`ea-stack-multi-safe-csp-lda` + `calibrated_stack_ridge_guard_borda`（Borda 合成 `ridge_pred_improve` 与 `probe_mixup_hard`），在同协议下：
- mean acc = **0.5498**（EA=0.5320，**+1.77pp**）
- neg_transfer_rate_vs_ea = **0.0**
- accept_rate = **0.6667**

输出目录：`outputs/20260107/4class/loso4_actionset_stack_familyblend_k20_probe_minimp001_borda_fix_v1/`

## 1) 本轮假设（单杠杆）
由于候选集合较大（multi-family action-set），存在“多重比较/假阳性”风险；我们验证一个**更接近 contextual bandit（RL-like）**的选择器是否能在不改候选集与安全 gate 的前提下，提升选择稳定性与均值：

- **唯一改动（one lever）**：`--oea-zo-selector calibrated_stack_bandit_guard`
- 其余保持不变：候选 families、per-family calib、anchor guard δ、anchor probe gate、高风险 family gate（FBCSP/TSA）等。

## 2) 复现实验（严格 LOSO）
- Commit：`c9dcc63cf4f35b3f291e1b7878f767eb5f5979b1`
- Command：见 `outputs/20260107/4class/loso4_actionset_stack_familyblend_k20_probe_minimp001_bandit_v1_01/20260107_results.txt` 顶部 `Command:` 行
- Outputs：`outputs/20260107/4class/loso4_actionset_stack_familyblend_k20_probe_minimp001_bandit_v1_01/`

## 3) 结果（主表口径）
来自：`outputs/20260107/4class/loso4_actionset_stack_familyblend_k20_probe_minimp001_bandit_v1_01/20260107_method_comparison.csv`

- `ea-csp-lda`: mean acc **0.5320**, worst **0.2569**
- `ea-stack-multi-safe-csp-lda`（bandit）：mean acc **0.5363**（**+0.42pp vs EA**）, worst **0.2569**
  - accept_rate **0.4444**
  - neg_transfer_rate_vs_ea **0.1111**（1/9 被试负迁移）

**对比上一轮（Borda）**：mean acc **0.5498**（+1.77pp），accept **0.6667**，neg_transfer **0.0** —— bandit 选择器明显更差。

## 4) 诊断（证书/选择器有效性）
来自：`scripts/analyze_candidate_certificates.py`（已保存表格到图目录）

- candidate oracle mean = **0.5633**
- gap\_sel\_mean：
  - bandit：**0.0270**
  - Borda：**0.0135**
  => bandit 的“选中候选”离 oracle 更远（选错更严重）。
- 证书-acc Spearman（候选集合内）：
  - `rho_bandit_mean = -0.041`（bandit_score 与真实 acc **近似无关/偏反向**）
  - 作为参照：`rho_probe_mean = 0.288`, `rho_probe_hard_mean = 0.197`
- 负迁移集中在 **S8**：`0.7188 → 0.6910`（详见候选表 `neg_transfer` 列）。

**图（论文级证据）**：
- `docs/experiments/figures/20260107_loso4_bandit_selector_v1/loso4_bandit_selector_v1_bandit_vs_true.png`
- `docs/experiments/figures/20260107_loso4_bandit_selector_v1/loso4_bandit_selector_v1_guard_vs_true.png`
- `docs/experiments/figures/20260107_loso4_bandit_selector_v1/loso4_bandit_selector_v1_ridge_vs_true.png`
- `docs/experiments/figures/20260107_loso4_bandit_selector_v1/loso4_bandit_selector_v1_oracle_gap.png`

## 5) 结论与下一步
**结论**：在当前“候选动作集合 + 安全 gate”框架下，直接用 `calibrated_stack_bandit_guard`（线性 softmax policy）替换 Borda，并不能提升证书有效性，反而降低 mean/accept，并引入负迁移。

**下一步建议（仍保持单杠杆）**：保留 Borda 作为主线选择器；若继续走“RL 叙事”，需要先解决“policy score 与 Δacc 不一致”的根因（例如 reward/训练目标/特征集合/分组归一化），否则只会把“证书无效”问题换个名字。

