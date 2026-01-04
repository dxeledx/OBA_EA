# 2026-01-03/04 — LOSO 4-class — EA-STACK-MULTI-SAFE + stacked ridge/guard（probe/evidence 特征，失败记录）

目标：沿“**证书有效性 + 安全选择/回退**”主线，验证一个明确假设：  
在跨动作集合（EA/RPA/TSA/CHAN/FBCSP）选择中，**仅用 entropy/drift/p̄ 的 base 特征会证书失效**（上一轮 S9 误选已证明），因此把候选记录补齐 **evidence_nll + probe_mixup(+hard)** 并做 **stacked ridge + guard**，应能显著降低误选与负迁移。

> 重要：本次 run 的输出目录在 `outputs/20260103/...`，但 `*_results.txt` 头部记录的 Date 为 `20260104`，属于跨天运行的正常现象（开始/结束日期不一致）。

---

## 1) 上一次失败分析（post-mortem）

来自 `docs/experiments/20260103_loso4_actionset_selector_calibrated.md`：
- cross-family selector（base 特征）出现 **Subject 9 灾难性误选**：EA `0.6840 → 0.5347`（Δ = −0.1493）
- 说明问题不是“动作集没有 headroom”，而是 **证书/选择器无法识别高风险误选**。

---

## 2) 本次假设与改动（只动一个杠杆）

**唯一杠杆：证书特征升级（从 base → stacked probes）**  
- 在 `ea-stack-multi-safe-csp-lda` 分支里，为每个候选动作记录补齐：
  - `evidence_nll_*`（LDA 生成式证据）
  - `probe_mixup_*` 与 `probe_mixup_hard_*`（MixUp probe，hard-major 对应 MixVal 的 λ>0.5 思想）
- 并新增 `selector=calibrated_stack_ridge_guard`：用 stacked 特征训练 ridge(预测提升) + guard(拒绝负迁移)。

实现提交：
- `git commit 93162c7`（`Add probe/evidence signals to stacked selector`）

---

## 3) 怎么做的（命令 + 输出目录）

命令（严格可比协议）：
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
  --no-plots \
  --run-name loso4_actionset_stack_probe_guard_v1
```

Outputs：
- `outputs/20260103/4class/loso4_actionset_stack_probe_guard_v1/20260103_method_comparison.csv`
- `outputs/20260103/4class/loso4_actionset_stack_probe_guard_v1/20260103_results.txt`
- `outputs/20260103/4class/loso4_actionset_stack_probe_guard_v1/20260103_predictions_all_methods.csv`

---

## 4) 结果（主表指标 + 证书有效性）

来自 `20260103_method_comparison.csv`：
- `ea-csp-lda` mean acc **0.5320**, worst **0.2569**
- `ea-stack-multi-safe-csp-lda`（stacked ridge/guard）mean acc **0.5081**（**−2.39% abs**）, worst **0.2569**
  - `accept_rate = 5/9 = 0.5556`
  - `neg_transfer_rate_vs_ea = 4/9 = 0.4444`
  - `cert_improve_spearman = -0.4000`
  - `guard_improve_spearman = -0.4167`

关键负迁移（来自 `20260103_results.txt`）：
- S8：EA `0.71875 → 0.61806`（Δ = **−0.10069**，选择 `fbcsp`）
- S9：EA `0.68403 → 0.61111`（Δ = **−0.07292**，选择 `fbcsp`）
- S5：EA `0.30903 → 0.25694`（Δ = **−0.05208**，选择 `fbcsp`）
- S7：EA `0.57292 → 0.53819`（Δ = **−0.03472**，选择 `fbcsp`）

直观结论：**stacked 特征没有修复证书失效，反而系统性“过度相信” FBCSP 分支**，导致多被试负迁移。

---

## 5) 图（用于汇报/诊断）

- per-subject Δacc vs EA：`docs/experiments/figures/20260103_loso4_actionset_stack_probe_guard_v1_delta.png`
- ridge 预测提升 vs 真实提升：`docs/experiments/figures/20260103_loso4_actionset_stack_probe_guard_v1_ridge_vs_true.png`
- guard 概率 vs 真实提升：`docs/experiments/figures/20260103_loso4_actionset_stack_probe_guard_v1_guard_vs_true.png`
- 选择到的 family（橙=accept，灰=EA fallback）：`docs/experiments/figures/20260103_loso4_actionset_stack_probe_guard_v1_family.png`

---

## 6) 这次说明了什么（失败原因的“机制性”解释）

证书相关性变成负值，说明不是“噪声导致不显著”，而是 **选择器学到了错误的偏好**。结合本次 per-subject 选择结果，最可疑的机制是：

1) **跨 family 的证书不可比 / 分布不稳**  
`evidence_nll` 与 `probe_mixup` 都依赖于 *各自 family 的特征空间与 LDA*；其数值尺度、维度、分布差异很大。  
即使加入 `cand_family_*` one-hot，训练样本很少时 ridge/guard 也容易学出“错误的 family 偏置”。

2) **FBCSP 在 pseudo-target 上“看起来经常有利”，但在真实 target 上高方差**  
导致 guard/ridge 对 fbcsp 候选的接受概率偏高；一旦遇到坏被试（S5/S7/S8/S9），就会出现成片负迁移。

3) **安全门控强度不足**  
当前 `guard_threshold=0.5` 且 `guard_margin=0`，对高风险 family 过松；本轮 accept 的 5 个里 4 个为负迁移。

---

## 7) 下一步（仍然只动一个杠杆）

把 **FBCSP 当“高风险 arm”**，做更严格的 *family-specific gate*（保持 EA anchor，目标：先把 `neg_transfer≈0` 压住）：
- 对 `fbcsp` 单独提高接受条件（更高 `guard_threshold` / 更高 `min_pred_improve` / drift hard gate）
- 或者先把 `fbcsp` 从动作集移除作为消融，验证 “负迁移主要来自 fbcsp” 是否成立

这一步不改训练协议、不用目标标签，且可以直接回答：**证书失败是否由某个高风险 family 主导**。

