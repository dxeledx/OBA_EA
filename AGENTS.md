# AGENTS.md — EA Project (OBA_EA) — Q1 Research Agent Rules

你是 Codex，在本仓库中扮演 **科研研究者 + 工程实现者**。
目标不是“跑通”，而是为 **SCI Q1 论文**打造可复用的、可复现的、证据链完整的实验与方法创新。

本项目是 MI-EEG 的经典管线与改进线：**CSP + LDA**（MOABB MotorImagery），默认数据集为 `BNCI2014_001`（BCI-IV 2a），支持 2 类与 4 类任务，评估默认 LOSO:contentReference[oaicite:0]{index=0}。

---

## 0) 不可违反的硬约束（Q1 生死线）

### 0.1 协议与可比性
- 任何新方法必须在**相同协议/相同预处理/相同指标**下与基线对比（否则不可宣称提升）。
- 主表必须是**严格协议**；若使用 TTA（目标域无标签适配），只能作为附表/补充，并且要写清楚边界。

### 0.2 严格 LOSO vs TTA 的论文口径（必须遵循）
- 主表（严格 LOSO 建议）：`--oea-pseudo-iters 0`（目标域不做伪标签迭代；分类器不更新）:contentReference[oaicite:1]{index=1}  
- 附表（TTA-Q）：`--oea-pseudo-iters 1~2`（仍不更新分类器，仅用无标签目标数据对 Q_t 迭代）:contentReference[oaicite:2]{index=2}

### 0.3 先诊断再改动（强制流程）
任何改动前必须先对“上一次结果”做 post-mortem：逐被试、逐方法、负迁移率、最差被试、混淆矩阵/预测分布等（见第 3 节）。

---

## 1) 本仓库的“真实入口”与产物（必须按它做）

### 1.1 标准 LOSO（默认入口）
- 运行入口（最常用）：
  - `conda run -n eeg python run_csp_lda_loso.py`:contentReference[oaicite:3]{index=3}
- 默认输出在：`outputs/YYYYMMDD/<N>class/HHMMSS/`，且包含：
  - `*_results.txt`
  - `*_predictions_all_methods.csv`（逐 trial 的 y_true/y_pred/proba_*，用于逐被试诊断）
  - 各方法独立 predictions、CSP patterns、confusion matrix、模型对比图等:contentReference[oaicite:4]{index=4}

### 1.2 单被试跨 session（0train→1test，debug/稳健性补充）
当“跨被试证书失效/负迁移”太严重时，可先做单被试跨 session：
- `conda run -n eeg python run_csp_lda_cross_session.py`，输出到 `outputs/YYYYMMDD/<N>class/cross_session/HHMMSS/`:contentReference[oaicite:5]{index=5}

### 1.3 实验笔记与结果登记（必须维护）
本仓库已经有“实验笔记（lab notebook）+ 结果 registry”机制：
- 笔记放在 `docs/experiments/`，按 `YYYYMMDD_<topic>.md` 命名，记录命令、输出目录、观察/诊断:contentReference[oaicite:6]{index=6}  
- `results_registry.csv` 由脚本扫描 `outputs/**/**/**/**_method_comparison.csv` 等产物生成；刷新命令：  
  `python3 scripts/update_results_registry.py --outputs-dir outputs --out docs/experiments/results_registry.csv`:contentReference[oaicite:7]{index=7}

---

## 2) 方法线与“创新点”必须数学化表达（Q1 必需）

本仓库的核心创新应围绕 **EA 的正交不确定性**与“无标签可选/可优化”的 Q：
- EA 白化矩阵可写为：`W_s = Q_s C_s^{-1/2},  Q_s ∈ O(C)`:contentReference[oaicite:8]{index=8}  
- OEA/OEA-ZO/EA-ZO 等方法族围绕如何选择/优化 `Q_s` 与 `Q_t` 展开（并明确“冻结分类器、不用目标真标签”）:contentReference[oaicite:9]{index=9}  

### 2.1 你的“创新点”不得只写成工程描述
任何新贡献必须能写成至少一种数学形式：
- 明确变量（Q 的参数化、候选集合、证书/guard 评分函数等）
- 明确目标（例如 entropy/infomax/pCE/confidence 等无标签目标，或证书的风险上界/估计器）
- 明确约束（信任域、漂移约束、类边际先验约束、回退策略等）

---

## 3) 强制 “失败优先”诊断闭环（每轮必须做）

每次提出新改动前，必须先对最近一次 run 做以下诊断，并写入 `docs/experiments/YYYYMMDD_<topic>.md`：

### 3.1 读哪些文件
- `*_results.txt`：总体指标/运行信息
- `*_method_comparison.csv`（如存在）：方法级汇总（mean、worst-subject、neg-transfer 等）
- `*_predictions_all_methods.csv`：逐被试、逐 trial 误差结构、confusion matrix（你也可以重画）

### 3.2 诊断必须包含的最小内容（Paper-grade）
- mean accuracy +（若实现/可得）kappa
- worst-subject（最差被试）
- negative transfer rate（相对 EA 或强 anchor 的掉点比例）
- 哪些被试“贡献了掉点”（Top-k bad subjects）
- 失败类型判别（至少归因到一类）：
  - 目标无标签目标不可靠（误选 Q/候选集）
  - 负迁移主导（需要更强的 safe gate / fallback）
  - 4 类多类设置下源域 Q_s “乐观选择”反而伤（见 4 类建议）

---

## 4) 4 类任务的特别规则（避免走弯路）

仓库 README 已明确警告：在 4 类任务上，多类 OEA 的“源域选解集 Q_s”经常导致整体性能低于 `ea-csp-lda`，因此建议优先用 **EA 训练 + 仅测试时优化 Q_t** 的 `ea-zo-*`:contentReference[oaicite:10]{index=10}。

**因此：**
- 4 类主线优先做 `ea-zo-*`，并把 `oea-zo-*` 作为对照/消融。
- 如果做 `oea-zo-*`，必须解释为什么需要源域“乐观选择 Q_s”，以及如何避免它在多类下的系统性负效应。

---

## 5) 研发规则：一次只动一个“杠杆”（保证可解释）

每个 PR/实验批次只能有一个主改动点（其余保持不变），并配套：
- Baseline：`ea-csp-lda`（至少）:contentReference[oaicite:11]{index=11}  
- Ablation：逐个关闭新机制（holdout、trust、marginal、drift、selector 变化等）
- 需要安全化时，优先使用 repo 已有的开关（不要重复造轮子）：
  - `--oea-zo-holdout-fraction`（holdout 选 best-iterate）
  - `--oea-zo-trust-*`（信任域防负迁移）
  - `--oea-zo-reliable-*`（可靠样本连续加权）
  - `--oea-zo-marginal-*`（类边际/先验约束）
  - `--oea-zo-drift-*`（预测漂移守门员）
  - `--oea-zo-selector ...`（候选集选择/证书/校准/stacking）
  - `--oea-zo-min-improvement`（不够好就回退 Q=I）:contentReference[oaicite:12]{index=12}

---

## 6) “漂亮实验”最低门槛（不满足 = 不算完成）

一条“方法有效”的结论，至少要交付：

### 6.1 主表（必须）
- 2 类与 4 类（至少其中一个作为主任务）在 LOSO 下的：
  - mean、std（若多 seed）
  - worst-subject
  - neg-transfer rate（相对 anchor）

### 6.2 图与诊断（必须）
- 混淆矩阵（至少主任务）
- 逐被试柱状/散点图（baseline vs proposed）
- 证书/selector 的有效性图（如 Spearman、AUC、accept_rate 等）

### 6.3 预算与复杂度（尽量提供，Q1 加分）
- 新方法相对 baseline 的额外计算（每被试 ZO iters、候选数、wall-time）
- 若涉及更复杂的候选族/stacking，需要声明“为什么值得这份预算”

---

## 7) 允许并鼓励联网检索（Related Work / SOTA）

当你提出“创新点/新 baseline/新证书”时，必须：
- 联网检索近年 Q1/IF>5 的 MI-EEG：alignment / test-time adaptation / uncertainty / safe model selection / Riemannian 方法
- 在 `docs/SOTA.md`（或同等位置）维护对比表：
  - paper、venue、year、dataset、protocol、metric、是否严格可比、关键 trick
- 不可比的协议必须明确标注 “non-comparable”。

---

## 8) DONE 定义（Q1 标准）

只有同时满足以下条件，任务才算完成：
- 复现命令 + 输出目录可追溯（写入实验笔记）
- 严格可比的 baseline 对照 + 消融齐全
- 提升不仅体现在 mean，也在 worst-subject/neg-transfer 上有合理解释
- 创新点能写成数学目标/约束，并能用实验直接支撑
- 图表可直接进入论文（“漂亮”）
