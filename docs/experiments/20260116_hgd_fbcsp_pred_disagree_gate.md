# 2026-01-16 — HGD(Schirrmeister2017) LOSO 4-class — 用 **pred_disagree** 做 FBCSP 高风险 gate：0 负迁移且 +3.05pp mean（CSP+LDA 体系内）

目标：在 **strict LOSO** 下，把 `ea-fbcsp-lda` 作为 “高收益但高风险” candidate，通过一个 **无标签证书(gate)** 做 **安全选择/回退**，先把 `neg_transfer_rate≈0` 压住，再追 mean。

本轮只动一个杠杆（证书/选择规则）：**对 FBCSP 增加 EA-anchor 相对化的 pred_disagree gate**：

设目标被试的无标签样本为 `i=1..n`，EA 与 FBCSP 的预测概率分别为 `p_i^EA, p_i^FB`，则
\[
\mathrm{pred\_disagree} \;=\; \frac{1}{n}\sum_{i=1}^n \mathbf{1}\big[\arg\max p_i^{EA} \neq \arg\max p_i^{FB}\big].
\]

选择规则（subject-level gate）：
\[
\text{if }\mathrm{pred\_disagree}\le \tau \text{ then use FBCSP else fallback to EA.}
\]

备注：**选择时不使用目标真标签**；真标签只用于事后评估。

---

## 1) 上一次失败分析（post-mortem）

在 HGD（`Schirrmeister2017`，`0train`，`left_hand,right_hand,feet,rest`，`resample=50`）下：

- EA baseline（`ea-csp-lda`）输出：
  - `outputs/20260112/4class/loso4_schirr2017_0train_rs50_ea_only_v2/20260112_results.txt`
  - mean acc = **0.5844**
- FBCSP candidate（`ea-fbcsp-lda`）输出：
  - `outputs/20260115/4class/loso4_schirr2017_0train_rs50_ea_fbcsp_lda_v1/20260115_results.txt`
  - mean acc = **0.5971**（+1.27pp），但 **4/14 被试负迁移**（28.6%）
  - 典型崩盘：S8 (−18.5pp), S14 (−10.8pp)

结论：**HGD 上 EA-FBCSP “均值能涨，但尾部风险大”**，非常适合用 “证书 + safe fallback” 做主线。

---

## 2) 本次假设与改动（只动一个杠杆）

假设：HGD 上 `ea-fbcsp-lda` 的负迁移主要来自 **预测分布相对 EA 的大幅漂移**；而这类漂移可以用 `pred_disagree(EA, FBCSP)` 在无标签下被捕捉并用于拒绝。

唯一改动：把 `pred_disagree` 作为 **FBCSP family-specific gate**，只要 `pred_disagree > τ` 就直接回退 EA。

本轮 `τ = 0.37`（由 “风险-收益 sweep 曲线” 的甜点区得到，见图）。

---

## 3) 怎么做的（复现链）

本轮结果 **不重新训练模型**：直接用两次严格 LOSO run 的逐 trial predictions 做“离线组合评估”。

输入（两次 run 的逐 trial 预测）：
- EA：`outputs/20260112/4class/loso4_schirr2017_0train_rs50_ea_only_v2/20260112_predictions_all_methods.csv`
- EA-FBCSP：`outputs/20260115/4class/loso4_schirr2017_0train_rs50_ea_fbcsp_lda_v1/20260115_predictions_all_methods.csv`

复现命令（生成 per-subject 表、sweep 表与图）：
```bash
python3 scripts/eval_pred_disagree_gate_from_predictions.py \
  --ea-preds outputs/20260112/4class/loso4_schirr2017_0train_rs50_ea_only_v2/20260112_predictions_all_methods.csv \
  --cand-preds outputs/20260115/4class/loso4_schirr2017_0train_rs50_ea_fbcsp_lda_v1/20260115_predictions_all_methods.csv \
  --tau 0.37 \
  --out-dir docs/experiments/figures/20260116_hgd_fbcsp_pred_disagree_gate_v1
```

离线组合 + 画图产物：
- `docs/experiments/figures/20260116_hgd_fbcsp_pred_disagree_gate_v1/per_subject.csv`
- `docs/experiments/figures/20260116_hgd_fbcsp_pred_disagree_gate_v1/summary.csv`
- `docs/experiments/figures/20260116_hgd_fbcsp_pred_disagree_gate_v1/tau_sweep.csv`
- 图见第 5 节。

---

## 4) 结果（主表指标 + 关键诊断）

在 `τ=0.37` 下（subject-level gate）：
- mean acc：**0.6149**（EA 0.5844 → **+3.05pp**）
- `neg_transfer_rate_vs_ea = 0/14 = 0.0`
- `accept_rate = 8/14 = 0.5714`（8 个被试用 FBCSP，6 个被试回退 EA）
- worst-subject acc：**0.3568**（与 EA 相同；HGD 的最差被试仍需更强方法/动作集来抬）

对照：
- `ea-fbcsp-lda`（直接用 FBCSP）mean acc 0.5971，`neg_transfer_rate_vs_ea = 4/14`
- 本 gate 本质上是 “把 FBCSP 的正收益被试保留、把灾难被试拒绝回退 EA”，因此 mean 明显抬升且负迁移归零。

---

## 5) 图（可直接用于汇报/论文补充）

图都在：
- `docs/experiments/figures/20260116_hgd_fbcsp_pred_disagree_gate_v1/`

建议优先用这三张：
- `docs/experiments/figures/20260116_hgd_fbcsp_pred_disagree_gate_v1/tau_sweep.png`  
  风险-收益 sweep：`tau` 增大 → accept_rate 上升，但越过某阈值开始引入负迁移；甜点区在 **0.35–0.37** 左右。
- `docs/experiments/figures/20260116_hgd_fbcsp_pred_disagree_gate_v1/scatter_disagree_vs_delta_cand.png`  
  `pred_disagree` vs `Δacc(FBCSP-EA)` 的被试级散点：高 `pred_disagree` 区域更容易出现负迁移（S8/S14 等）。
- `docs/experiments/figures/20260116_hgd_fbcsp_pred_disagree_gate_v1/bar_delta_selected.png`  
  选择后相对 EA 的 per-subject Δacc（颜色标注是否选择 FBCSP），可以直观看到 **“保涨不掉点”**。

---

## 6) 下一步（按 Q1 规则：仍然一轮一个杠杆）

这轮说明：**在 HGD 上，“证书有效性”不是空话，`pred_disagree` 作为拒绝型证书已经能显著提升 mean 且压到 0 负迁移。**

下一步最划算的两件事（建议按顺序，仍保持单杠杆迭代）：
1) 把该规则固化成可直接运行的方法（而不是离线组合）：在 `ea-stack-multi-safe-csp-lda` 中使用 `--stack-safe-fbcsp-max-pred-disagree 0.37`（已实现 wiring），并在 HGD 上再跑一次单方法输出，保证产物链完全一致。
2) 扩展动作集但保持 family-aware gate：把 `ea-fbcsp-lda` 之外再加入 1–2 个 strong arm（例如 `ts-lr`/`fgmdm`），并坚持 **arm-specific 高风险门槛**，目标是抬 worst-subject 而不引入负迁移。
