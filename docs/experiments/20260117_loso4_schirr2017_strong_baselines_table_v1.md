# 2026-01-17 — HGD(Schirrmeister2017) 4-class strict LOSO — 补齐 strong baselines 主表（CSP+LDA）

目标：把 HGD 的主表补齐到与 BNCI 主表同一等级的“强基线同协议对照”，并把我们的主方法（safe selection）放进同一张表里，避免论文叙事被质疑“不同数据集换了一套比较体系”。

约束：**strict LOSO**（选择/门控只用目标无标签统计；目标真标签仅用于最终评估）。

---

## 0) 为什么要补齐（failure-first）

之前 HGD 侧的 `docs/paper_assets/schirrmeister2017/...` 只覆盖了少量方法（或仅覆盖某一次 run），与 BNCI 的 strong-baselines 主表口径不一致；这会带来论文风险：
- 审稿人容易质疑：不同数据集上是否“换了方法/换了动作库/换了评价标准”。

因此本轮只做一件事：**把 HGD strong baselines 主表补齐**，并导出 paper-ready 图表。

---

## 1) 本轮做了什么（单杠杆：补齐主表，不改算法）

我们不重新训练、不改代码逻辑，只利用已完成 run 的逐 trial 预测，做一次严格对齐的合并：

1) 选择 HGD 上同协议的 baselines run（CSP+LDA family）
2) 把我们的 `ea-fbcsp-pred-disagree-safe`（train-only 标定 τ）也放进同一张表
3) 用 `merge_loso_runs.py` 合并成一个“可复现 run 目录”
4) 导出 main_table + per-subject + figures 到 `docs/paper_assets/`

---

## 2) 复现链（源 runs → 合并 run）

### 2.1 源 runs（同 protocol / 同 trials）

- EA / RPA / TSA（CSP+LDA family table）：
  - `outputs/20260114/4class/loso4_schirr2017_0train_rs50_csp_family_table_v3_merged/`
- FBCSP baseline（no EA）：
  - `outputs/20260114/4class/loso4_schirr2017_0train_rs50_fbcsp_lda_v1/`
- Ours（EA↔EA-FBCSP pred_disagree gate，train-only 标定 τ）：
  - `outputs/20260117/4class/loso4_schirr2017_0train_rs50_pred_disagree_calib_tau_v1_merged/`

### 2.2 合并后的“强基线 + 我们” run 目录

输出：
- `outputs/20260117/4class/loso4_schirr2017_0train_rs50_strong_baselines_plus_ours_v1_merged/`

合并命令：
```bash
python3 scripts/merge_loso_runs.py \
  --out-run-dir outputs/20260117/4class/loso4_schirr2017_0train_rs50_strong_baselines_plus_ours_v1_merged \
  --run-dirs \
    outputs/20260114/4class/loso4_schirr2017_0train_rs50_csp_family_table_v3_merged \
    outputs/20260114/4class/loso4_schirr2017_0train_rs50_fbcsp_lda_v1 \
    outputs/20260117/4class/loso4_schirr2017_0train_rs50_pred_disagree_calib_tau_v1_merged \
  --prefer-date-prefix 20260117
```

---

## 3) 主表结果（HGD strict LOSO）

主表文件：
- `docs/paper_assets/schirrmeister2017/4class_loso_strict/tables/main_table.md`

其中我们的主方法（`ea-fbcsp-pred-disagree-safe`）：
- mean acc：`0.6112` vs EA `0.5844`（+`2.68pp`）
- `neg_transfer_rate_vs_ea = 0/14`

---

## 4) Paper-ready 图表输出

目录：
- `docs/paper_assets/schirrmeister2017/4class_loso_strict/`

包含：
- `tables/main_table.md`（主表）
- `figures/hgd4_strict_main_*`（mean/worst/neg-transfer、per-subject Δacc/Δkappa 等）
- `figures/hgd4_strict_cm_*`（EA / EA-FBCSP / Ours 混淆矩阵）

数据与来源说明：
- `docs/paper_assets/schirrmeister2017/4class_loso_strict/SOURCES.md`

---

## 5) 下一步（写论文口径）

现在 HGD 与 BNCI 都有同级别的 strong baselines 主表与图表资产，可以开始写 JBHI/TBME 初稿的实验章节：
- 主表：BNCI + HGD（同协议）
- 证书有效性：BNCI 侧用 candidate diagnostics + oracle gap；HGD 侧用“高风险 candidate + 安全门控”的风险-收益曲线（补充材料）

