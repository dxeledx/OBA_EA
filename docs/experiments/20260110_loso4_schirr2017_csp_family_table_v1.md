# 20260110 — Schirrmeister2017 4-class LOSO（CSP+LDA 家族对比；rs=50Hz；0train-only）

## 0) 目的（扩展到第二个 MOABB 数据集）
在 BNCI2014_001（BCI-IV 2a）4-class LOSO 上，我们已经把主方法 `ea-stack-multi-safe-csp-lda` 固化为“+1.77pp 且 0 负迁移”的可复现结果与主表/统计脚本。

本条实验的目的不是再调参，而是**扩展到第二个公开 MI 数据集**，检验：
- EA 在不同数据集上的强度；
- 我们的“安全选择/回退”在 **更高密度通道** 数据集上是否仍可运行、是否仍稳定。

> 注意：跨数据集数值不做“直接可比”主张；我们只保证**同一数据集内**同协议/同预处理/同指标可比。

---

## 1) 协议与工程约束
**Dataset**：MOABB `Schirrmeister2017`（HGD；4-class：`left_hand/right_hand/feet/rest`）  
**Task**：4-class  
**Split**：严格 LOSO（cross-subject）  
**预处理**：`--preprocess moabb`，bandpass 8–30 Hz，epoch `tmin=0.5s` 到 `tmax=3.5s`  
**为什么用 rs=50Hz？**
- 在 rs=100Hz 且包含 multi-safe 的校准/候选族时，出现多次系统 OOM（RSS > 20GB，kernel oom-kill）。
- 为了在不改算法主逻辑的前提下得到可复现的跨数据集对比，本次把 `--resample` 下调到 **50Hz**（同一表内所有方法一致），以保证完整跑完并产出 paper-grade 表格/图。

**为什么用 `--sessions 0train`？**
- Schirrmeister2017 的 MOABB `meta` 中 `session` 常量为 `'0'`，而 train/test 分割在 `run` 列（`0train/1test`）。
- 为与 BNCI2014_001 主线一致（均为 `0train`），本次选择 `--sessions 0train`（实现为对 `session/run` 任一列匹配）。

---

## 2) 运行与合并方式（避免多方法同进程 OOM）
在该数据集上，多方法同进程容易触发内存峰值/碎片化导致 OOM，因此采用 **“单方法 run → 合并 predictions → 生成主表”** 的可复现流程：

### 2.1 单方法/小方法集 runs（输出目录）
- `outputs/20260110/4class/loso4_schirr2017_0train_rs50_sanity_v0/`（`csp-lda`, `ea-csp-lda`）
- `outputs/20260110/4class/loso4_schirr2017_0train_rs50_rpa_csp_lda_v1/`（`rpa-csp-lda`）
- `outputs/20260110/4class/loso4_schirr2017_0train_rs50_tsa_csp_lda_v1/`（`tsa-csp-lda`）
- `outputs/20260110/4class/loso4_schirr2017_0train_rs50_ea_stack_multi_safe_rpa_tsa_calib3_v1/`（`ea-stack-multi-safe-csp-lda`，候选族限定为 `ea,rpa,tsa`，并使用 `--oea-zo-calib-max-subjects 3` 的 holdout-calibration）

### 2.2 合并（生成一个“merged run dir”）
合并脚本：
- `scripts/merge_loso_runs.py`

合并后目录：
- `outputs/20260110/4class/loso4_schirr2017_0train_rs50_csp_family_table_v1_merged/`
  - `20260110_predictions_all_methods.csv`
  - `20260110_method_comparison.csv`
  - `20260110_MERGED_FROM.txt`（记录来源 runs 与各自 command）

---

## 3) 主表与统计（同协议）
表格与统计由脚本从 merged run 生成：
- 表格脚本：`scripts/make_main_table_and_stats.py`
- 输出目录：`docs/experiments/figures/20260110_loso4_schirr2017_0train_rs50_csp_family_table_v1/`
  - `main_table.csv`
  - `main_table.md`
  - `per_subject_metrics.csv`

图（论文级）：
- 图脚本：`scripts/plot_strong_baselines_table_figures.py`
- 生成到同一 figure 目录（PNG+PDF）

---

## 4) 结果摘要（现象先行）
来自 `main_table.csv`（baseline=`ea-csp-lda`）：
- `ea-csp-lda`：mean acc **0.5844**
- `ea-stack-multi-safe-csp-lda`：mean acc **0.5768**（Δmean = **-0.76pp**）
  - `accept_rate = 0.0714`（14 个被试里仅 1 个被试选择了非 EA 候选）
  - `neg_transfer_rate_vs_baseline = 0.0714`（该被试发生负迁移）
- `rpa-csp-lda`：mean acc **0.5049**（明显低于 EA）
- `tsa-csp-lda`：mean acc **0.4104**（与 `csp-lda` 接近，显著低于 EA）

**结论（对该数据集）**：
1) EA 是强基线；RPA/TSA 在该协议下整体不如 EA。  
2) multi-safe 在候选族 `{EA,RPA,TSA}` 下几乎总是回退 EA（accept_rate 很低），因此整体非常接近 EA；但仍存在“偶发误选”导致小幅负迁移（1/14）。

---

## 5) 下一步（Paper 口径）
这条数据集的价值更偏向 **“泛化与安全性证据”**：
- 对 HGD（高密度通道、更多 trial），我们的方法不会系统性崩盘（大部分被试回退 EA）。
- 但“能否显著提升 mean”并未成立，提示：在该数据集上 **候选族需要重新设计**（例如引入更强但可计算的候选族，或更严格/更对齐 LDA 风险的一致性证书），否则多候选会带来假阳性选择风险。

