# 20260107 — BNCI2014_001 4-class LOSO: 扩大 SI-CHAN ranks(19–21) **失败复盘**（负迁移回来了）

## Post-mortem（上一轮成功基线）
上一轮（Borda 选择器修复）在严格同协议下达到当前最佳：
- `outputs/20260107/4class/loso4_actionset_stack_familyblend_k20_probe_minimp001_borda_fix_v1/`
- EA mean acc: **0.5320**
- Stack mean acc: **0.5498**（**+1.77% abs**）
- `neg_transfer_rate_vs_ea = 0.0`
- `accept_rate = 0.6667`
- oracle gap（平均）：`gap_sel_mean = 0.0135`

结论：证书/选择器在当前 action-set 下是“可用”的，并且能保证 0 负迁移。

---

## 本轮假设与目标（只动一个杠杆）
**假设**：当前还存在 oracle gap，可能是因为 SI‑CHAN 的候选覆盖不足（只用 rank=21）。

**单杠杆改动**：把 SI‑CHAN 候选从 `rank=21` 扩展为 `rank∈{19,20,21}`，其他全部保持不变（同 selector / 同 gate / 同预处理 / 同指标）。

---

## Protocol（固定可比）
- Dataset: `BNCI2014_001`
- Task: 4-class（`left_hand,right_hand,feet,tongue`）
- Protocol: LOSO，`sessions=0train`
- Preprocess: `paper_fir`（8–30 Hz, causal FIR order=50），epoch 0.5–3.5s，resample 250 Hz
- Model: `CSP(n_components=6) + LDA`
- Methods: `ea-csp-lda` vs `ea-stack-multi-safe-csp-lda`
- Selector: `calibrated_stack_ridge_guard_borda`
- Safety（保持不变）：
  - anchor‑δ：`--stack-safe-anchor-guard-delta 0.05`
  - probe_hard min‑improve：`--stack-safe-anchor-probe-hard-worsen -0.01`
  - FBCSP/TSA high‑risk gates（guard/min_pred/drift）
  - per‑family calibration blend：`K=20`

---

## Run
Output dir：
- `outputs/20260107/4class/loso4_actionset_stack_familyblend_k20_chanr19-21_lamdense_anchor_delta005_probe_minimp001_borda_fix_v2/`

命令见 `20260107_results.txt`。

---

## 结果（失败）
来自 `20260107_method_comparison.csv`：
- EA mean acc: **0.5320**
- Stack mean acc: **0.5417**（**+0.96% abs**，显著低于 v1 的 +1.77%）
- worst-subject: **0.2292**（低于 EA 的 0.2569）
- `accept_rate = 0.8889`（8/9 接受候选）
- `neg_transfer_rate_vs_ea = 0.2222`（2/9 出现负迁移）

按被试 Δacc（stack − EA）：
- 掉点：S2 **−2.78%**、S9 **−2.78%**
- 其余：5 个提升、2 个持平

图：
- `docs/experiments/figures/20260107_loso4_chanr19-21_borda_fix_v2/loso4_chanr19_21_borda_fix_v2_delta.png`
- `docs/experiments/figures/20260107_loso4_chanr19-21_borda_fix_v2/loso4_chanr19_21_borda_fix_v2_oracle_gap.png`

---

## 失败原因（证据链）
### 1) 候选扩张 → “证书假阳性”变多
本轮 SI‑CHAN 候选从 7 个扩到 21 个（rank×λ），accept_rate 上升到 8/9。
但证书并不能保证“被选候选更接近真实 acc 最优”，导致负迁移复现（S2、S9）。

### 2) probe_hard gate 在扩大 action-set 时会“错杀真提升 + 放行坏提升”
典型例子：S2
- 真实最优（oracle）是 `rpa`：acc **0.28125**（比 EA +2.43%）
- 但它的 `probe_mixup_hard_best` 几乎不变，无法通过 **min‑improve** gate
- 最终被选的是 `chan(rank=19, λ=0.25)`：acc **0.22917**（比 EA −2.78%）

这说明：当 action-set 变大时，“强制 probe_hard 提升”不再等价于“更可能涨 acc”，反而可能把真正的提升路径（例如 rpa）拒掉。

---

## 决策（按 Q1 规则）
该单杠杆改动 **不保留**：
- mean 明显下降
- worst-subject 下降
- 负迁移回归

**当前 best 仍然是 v1**：
- `outputs/20260107/4class/loso4_actionset_stack_familyblend_k20_probe_minimp001_borda_fix_v1/`

---

## 下一步建议（还没执行；需另起一轮、仍坚持“一次只动一个杠杆”）
想继续扩大 action-set（更多 rank / 更多 family）而不引入负迁移，必须先把“抗多候选假阳性”的机制增强，例如：
1) 让 δ/ε 与候选规模 K 挂钩（multiple-testing 风格的 δ(K)），或
2) 引入额外的安全证书（例如更强的 drift guard / coverage gate），专门拒绝 S9 类崩盘。

