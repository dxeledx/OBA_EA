# 20260108 — BNCI2014\_001 4-class LOSO — 强 baselines 主表 + 统计检验（JBHI 论文主实验补齐）

## 0) 目标与协议（严格可比）
**目标**：把当前最佳配置（`ea-stack-multi-safe-csp-lda`：+1.77pp 且 0 负迁移）固化为“当前主方法”，并在 **同协议/同预处理/同指标** 下补齐：
1) **强 baselines 主表**（包含经典 CSP/FBCSP 与 Riemannian baselines）；
2) **统计检验**（paired test + CI），用于 JBHI 写作。

**协议固定**：
- Dataset：BNCI2014\_001（BCI-IV 2a）
- Task：4-class（left/right/feet/tongue）
- Split：严格 LOSO（cross-subject）
- Preprocess：`paper_fir`（8–30 Hz, causal FIR）
- Model（CSP-family）：CSP(n=6) + LDA（以及 FBCSP 的 LDA shrinkage\_auto）
- Metrics：macro（脚本默认以 per-subject accuracy 做 paired test）

## 1) 主方法（固化配置）
**主方法**：`ea-stack-multi-safe-csp-lda`（Borda selector + calibrated ridge/guard + safe fallback to EA）。

该配置与 20260107 “best run”一致，并在本次“强 baselines 主表 run”中再次复现相同均值表现：
- mean acc：**0.5498**
- vs `ea-csp-lda`：**+1.77pp**
- `neg_transfer_rate_vs_ea = 0.0`
- `accept_rate = 0.6667`

## 2) 复现实验（主表 run）
输出目录：
- `outputs/20260108/4class/loso4_strong_baselines_table_v1/`

复现命令：见
- `outputs/20260108/4class/loso4_strong_baselines_table_v1/20260108_results.txt` 顶部 `Command:` 行。

包含方法（同一 run 同协议）：
- `csp-lda`
- `fbcsp-lda`
- `ea-csp-lda`
- `ea-fbcsp-lda`
- `rpa-csp-lda`
- `tsa-csp-lda`
- `riemann-mdm`
- `ts-lr`
- `rpa-mdm`
- `rpa-ts-lr`
- `ea-stack-multi-safe-csp-lda`（主方法）

## 3) 主表（强 baselines）
主表（CSV + Markdown）由脚本从 `*_predictions_all_methods.csv` 生成：
- CSV：`docs/experiments/figures/20260108_loso4_strong_baselines_table_v1/main_table.csv`
- Markdown：`docs/experiments/figures/20260108_loso4_strong_baselines_table_v1/main_table.md`

脚本（可复用）：
- `scripts/make_main_table_and_stats.py`

## 4) 统计检验与稳定性（针对主方法 vs EA）
来自 `main_table.csv` 中 `ea-stack-multi-safe-csp-lda` 行（baseline=`ea-csp-lda`）：
- mean Δacc：**+0.01775**
- 95% bootstrap CI（subject-level mean Δacc）：**[+0.00694, +0.02894]**
- per-subject 变化：**5/9 提升，4/9 持平，0/9 下降**（neg\_transfer=0）
- Wilcoxon signed-rank：
  - two-sided：**p = 0.0625**
  - one-sided（greater）：**p = 0.03125**

解释：由于 BNCI2014\_001 只有 9 个 subject，two-sided 检验在 n 小且存在多 tie 的情况下会偏保守；但 “0 负迁移 + 多数提升” 给出稳定性证据。论文主表建议报 two-sided p，并在补充材料报 one-sided 与 sign test/CI。

## 5) 下一步（面向 JBHI 的最小补齐）
当前主表与统计脚本已经具备“可复现 + 可写进论文”的基本形态。下一步最关键的是补齐 **跨数据集/跨 setting 的稳定性**（否则 JBHI 容易质疑泛化性），同时保持 strict LOSO 可比性。

