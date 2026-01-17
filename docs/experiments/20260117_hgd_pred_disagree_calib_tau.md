# 2026-01-17 — HGD(Schirrmeister2017) 4-class strict LOSO — pred_disagree gate：**train-only 标定 τ**（CSP+LDA）

目标：把 `ea-fbcsp-lda` 在 HGD 上的“高收益但高风险”现象，做成 **strict LOSO** 下可写论文的 **安全选择/回退**闭环：
- mean ≥ EA
- `neg_transfer_rate = 0`
- 选择时不使用目标真标签（真标签只用于事后评估）

本记录只覆盖 **CSP+LDA 体系**，不涉及深网。

---

## 0) 上一轮复盘（failure-first）

我们已经验证过：在 HGD strict LOSO 下，`ea-fbcsp-lda` 的均值往往能涨，但尾部风险明显（会出现多个 subject 负迁移）。

同时，若直接用全数据的 label 做 `tau` sweep 选甜点区，会引入 **test-label 调参泄露**，不能作为 strict 主表口径。

因此需要把 `tau` 变成 **训练折可标定**的参数。

---

## 1) 本轮唯一改动（单杠杆）

仍用同一个无标签证书：
\[
\mathrm{pred\_disagree}=\frac{1}{n}\sum_i \mathbf{1}[\arg\max p^{EA}(x_i)\ne \arg\max p^{FBCSP}(x_i)].
\]

但将阈值从“手动固定/全数据 sweep”升级为：

**LOSO 风格 train-only 标定 τ（per-subject）**  
对每个 test subject，把其它 subject 当作训练折，只在训练折上选 `tau`，并要求训练折达到 0 负迁移：
- 在训练折上：最大化 mean accuracy
- 约束：训练折 `neg_transfer_rate = 0`
然后把该 `tau` 应用到当前 test subject（依旧不看目标标签）。

实现脚本：`scripts/build_pred_disagree_calib_run.py --mode calib_loso_safe0`

---

## 2) 复现链（可追溯）

### 2.1 依赖的两次 strict LOSO run（只需要逐 trial 预测）

- EA anchor：
  - `outputs/20260112/4class/loso4_schirr2017_0train_rs50_ea_only_v2/`
- EA-FBCSP candidate：
  - `outputs/20260115/4class/loso4_schirr2017_0train_rs50_ea_fbcsp_lda_v1/`

### 2.2 生成 “合并后的可复现 run 目录”（含主表文件）

输出目录：
- `outputs/20260117/4class/loso4_schirr2017_0train_rs50_pred_disagree_calib_tau_v1_merged/`

命令：
```bash
python3 scripts/build_pred_disagree_calib_run.py \
  --ea-run-dir outputs/20260112/4class/loso4_schirr2017_0train_rs50_ea_only_v2 \
  --cand-run-dir outputs/20260115/4class/loso4_schirr2017_0train_rs50_ea_fbcsp_lda_v1 \
  --ea-method ea-csp-lda \
  --cand-method ea-fbcsp-lda \
  --mode calib_loso_safe0 \
  --sweep-max 0.6 --sweep-steps 61 \
  --selected-method-name ea-fbcsp-pred-disagree-safe \
  --out-run-dir outputs/20260117/4class/loso4_schirr2017_0train_rs50_pred_disagree_calib_tau_v1_merged \
  --date-prefix 20260117
```

---

## 3) 结果（strict 主表口径）

从 `outputs/20260117/4class/loso4_schirr2017_0train_rs50_pred_disagree_calib_tau_v1_merged/20260117_pred_disagree_gate_summary.csv`：

- EA mean acc：`0.5844`
- EA-FBCSP mean acc：`0.5971`（+`1.27pp`，但不安全）
- **Ours mean acc：`0.6112`（+`2.68pp`）**
- **neg-transfer vs EA：`0/14`**
- accept-rate：`0.50`
- worst-subject：`0.3568`（与 EA 相同；后续要靠更强动作集/更强分类器抬）
- `tau`（train-only 标定）分布：约 `0.35–0.37`（见 `*_pred_disagree_gate_per_subject.csv`）

---

## 4) Paper-ready 图表输出

目录：
- `docs/paper_assets/schirrmeister2017/4class_loso_strict_calib_pred_disagree/`

包含：
- `tables/main_table.md`（含 mean±std、worst、neg-transfer、统计检验）
- `figures/hgd4_strict_main_*`（mean/worst、neg-transfer、per-subject Δacc/Δkappa 等）
- `figures/hgd4_strict_cm_*`（EA / EA-FBCSP / Ours 的混淆矩阵）
- `figures/hgd4_strict_pred_disagree_tau037/`（analysis-only：固定 τ 的 sweep 与散点图，用于敏感性/补充材料）

---

## 5) 结论与下一步

结论：在 HGD 4-class strict LOSO 下，`pred_disagree` 作为 **EA-anchor 相对化证书**，配合 **train-only 标定阈值 + safe fallback**，
可以实现 “mean 明显提升 + 0 负迁移”。

下一步（仍遵循一次只动一个杠杆）：
- 若要进一步抬 mean/worst：扩展动作集（加入更强/更稳的候选 family），但必须保持 **family-aware high-risk gate**；或在更强 CSP 系 baseline（如 FBCSP）上复用同样的“证书有效性 + safe selection”框架。

