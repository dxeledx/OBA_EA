# Paper draft — Experiments section (CSP+LDA line only)

本草稿用于你写 JBHI 初稿时直接“填空式”替换数值/图表引用；图表全部来自 `docs/paper_assets/`。

---

## E.1 Datasets & protocols

**BNCI2014_001 (BCI-IV 2a)**  
- Task: 4-class MI (`left_hand,right_hand,feet,tongue`)（主表），以及 2-class（补充）  
- Protocol: cross-subject LOSO, strict（不做目标域伪标签迭代/不更新分类器）  
- Preprocess: `paper_fir`（8–30 Hz, 250 Hz, 0.5–3.5 s, causal FIR）  
- CSP: `n_components=6`

**Schirrmeister2017 (HGD)**  
- Task: 4-class MI (`left_hand,right_hand,feet,rest`)  
- Protocol: cross-subject LOSO, strict（同上），`sessions=0train`  
- Preprocess: `moabb`（8–30 Hz）, `resample=50`（效率设置）, 0.5–3.5 s  
- CSP: `n_components=4`

> 统一口径：所有“选择/门控/回退”都只用 **目标被试无标签统计**；目标真标签只用于最终评估与作图。

---

## E.2 Compared methods (CSP+LDA family)

**EA-CSP-LDA**（anchor / safety baseline）  
**EA-FBCSP-LDA**（强但高风险 candidate；HGD 上尤甚）  
**Ours: certificate-calibrated safe selection**  
- 以 EA 为 anchor，候选为若干对齐/特征 family；用无标签证书 + 校准器做选择，并允许回退到 EA 以控制负迁移风险。

---

## E.3 Metrics & statistical tests

主指标：subject-averaged **accuracy**（宏平均）与 **Cohen’s κ**。  
稳健性指标：**worst-subject** accuracy、**negative transfer rate**（相对 EA）。  
统计检验：对每个 subject 的 paired Δacc 做 Wilcoxon / sign test（见主表 csv）。

---

## E.4 Results on BNCI2014_001 (main table)

主表与图（直接引用）：  
- Table (main): `docs/paper_assets/bnci2014_001/4class_loso_strict/tables/main_table.md`  
- Fig (mean acc/kappa + neg-transfer): `docs/paper_assets/bnci2014_001/4class_loso_strict/figures/bnci4_strict_main_*`  
- Fig (confusion matrices): `docs/paper_assets/bnci2014_001/4class_loso_strict/figures/bnci4_strict_cm_*`  
- Fig (certificate effectiveness / oracle gap): `docs/paper_assets/bnci2014_001/4class_loso_strict/figures/bnci4_strict_candidate_v1_*`

写法建议（2–3 句话模板）：  
1) 在 strict LOSO 下，我们的方法在 mean accuracy 上相对 EA 获得稳定提升，同时将负迁移率压到 0。  
2) 逐被试散点/柱状图表明提升并非由少数被试驱动，且最差被试不恶化（或不显著恶化）。  
3) 证书-真实性能相关性显著为正，且能解释/拒绝之前的灾难 case。

---

## E.5 Results on Schirrmeister2017 (HGD)

主表与图（直接引用）：  
- Table (main): `docs/paper_assets/schirrmeister2017/4class_loso_strict_calib_pred_disagree/tables/main_table.md`  
- Fig (mean acc/kappa + neg-transfer): `docs/paper_assets/schirrmeister2017/4class_loso_strict_calib_pred_disagree/figures/hgd4_strict_main_*`  
- Fig (confusion matrices): `docs/paper_assets/schirrmeister2017/4class_loso_strict_calib_pred_disagree/figures/hgd4_strict_cm_*`  
- Fig (tau sensitivity, analysis-only): `docs/paper_assets/schirrmeister2017/4class_loso_strict_calib_pred_disagree/figures/hgd4_strict_pred_disagree_tau037/tau_sweep.png`

写法建议：强调 **高风险 candidate + 安全化选择** 的必要性：  
- EA-FBCSP 在均值上可能更高，但存在明显负迁移；我们的证书校准 + safe fallback 在保持 0 负迁移的同时吃到大部分 headroom。

---

## E.6 Certificate effectiveness (the “science question” evidence)

你需要把“证书从失效到有效”的证据写成 3 件事：
1) **Correlation**：证书分数与真实 Δacc 的 Spearman（正相关）。  
2) **Error analysis**：被拒绝/被接受样本的假阳性分析（哪些 subject 曾经误接受导致负迁移）。  
3) **Headroom decomposition**：候选集 oracle 与 selector 之间的 gap（吃到多少 headroom）。

对应图表入口：
- BNCI：`docs/paper_assets/bnci2014_001/4class_loso_strict/figures/bnci4_strict_candidate_v1_*`
- HGD：`docs/paper_assets/schirrmeister2017/4class_loso_strict_calib_pred_disagree/figures/hgd4_strict_pred_disagree_tau037/*`

---

## E.7 Runtime / cost

建议报告：候选数、每被试额外 wall-time、以及是否需要额外训练（我们这条 CSP+LDA 线基本是 “不更新分类器 + 轻量选择/门控”）。

