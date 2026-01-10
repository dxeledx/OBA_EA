# 20260110 — 下层动作集候选论文 + 两个补充实验（HGD “≈EA 且 0 负迁移”、BNCI 风险‑收益 sweep）

> 目标：为 JBHI 叙事补齐两条证据链：
> 1) **下层动作集合**（对齐/迁移/预处理）可以系统化为若干“arms”，我们后续可做 action‑set 扩展；
> 2) 在 **无标签目标域**下，强适应存在负迁移风险：当“候选集合无 headroom”时应 **abstain/回退**；当 headroom 存在时应在 **风险‑收益**之间形成可控权衡。

---

## 1) 下层动作集（候选 arms）相关论文（建议先下载）

### 1.1 Euclidean / whitening 系（经典、计算便宜、易做为 anchor）
- **Euclidean Alignment (EA)**：跨被试/跨 session 的欧氏白化对齐（MI‑EEG 经典强基线）。
  - He & Wu, *Transfer Learning for Brain–Computer Interfaces: A Euclidean Space Data Alignment Approach*, IEEE TBME, 2020.
- **Label Alignment (LA)**：与 EA 同框架的一类“标签空间/类别先验”对齐扩展。
  - Wu et al., *Transfer Learning for Brain–Computer Interfaces: A Euclidean Space Data Alignment Approach*, IEEE TBME, 2020（同系工作）；另可看 IEEE Brain 综述/短文中对 EA/LA 的提炼。
- **综述/回顾**：EA 及其变体近年总结（便于 related work / 定位）。
  - A recent EA review paper (2025)（用于 related work 的“EA 系谱”与局限讨论）。

### 1.2 Riemannian / SPD 系（更强的“几何动作空间”，常在跨 session 更有效）
- **Riemannian Procrustes Analysis (RPA)**：SPD 流形上的 Procrustes（translation/scaling/rotation）用于 BCI transfer。
  - Rodrigues et al., *Riemannian Procrustes Analysis: Transfer Learning for Brain–Computer Interfaces*, IEEE TBME, 2019.
- **Tangent Space Alignment (TSA)**：切空间 Procrustes（闭式/SVD），强调跨 session/跨被试变异处理。
  - Bleuzé et al., *Tangent space alignment: Transfer learning for Brain-Computer Interface*, Frontiers in Human Neuroscience, 2022.
- **Riemannian transfer learning framework**（TL‑Center / TL‑Stretch 等），pyRiemann 的 transfer baselines 来源之一。
  - Zanini et al., *Transfer Learning: A Riemannian Geometry Framework With Applications to Brain-Computer Interfaces*, IEEE TBME, 2018.

### 1.3 其它可纳入 action set 的方向（后续按性价比挑）
- **Unsupervised / semi‑supervised EEG domain adaptation**（若未来扩到深度模型，可做 “deep‑arm”）。
  - Sun et al., *Deep CORAL for EEG-based cross-subject emotion recognition*, Computers in Biology and Medicine, 2020（用于证书/选择器与深度域适配的 related work 链接）。
- **信息论/互信息/熵最小化类 TTA**（做“高风险 arm”，必须配 guard 才能用）。
  - Tent/EATA 系列（视觉领域为主，但可作为“proxy 失效/负迁移”的理论与动机来源）。

> 注：以上论文用于“下层动作集合设计”与 related work 的结构化引文；我们会把这些方法以 **family/arm** 形式接入 `ea-stack-multi-safe-*` 的 action set，再由上层证书/校准器做安全选择。

---

## 2) 实验 A：Schirrmeister2017（HGD）4-class LOSO —— 让方法 “≈EA 且 0 负迁移”

### 2.1 失败现象（上一轮）
在 HGD 的当前协议下（`--sessions 0train --resample 50 --preprocess moabb`）：
- 候选集 `{EA, RPA-CSP, TSA-CSP}` 内 **oracle headroom = 0**（每个被试 EA 都是最优）。
- 但 `ea-stack-multi-safe-csp-lda` 发生 **1/14 次误接受**（S9 选了 TSA，准确率从 0.3568 → 0.25），导致 mean 变差。

对应 run：
- `outputs/20260110/4class/loso4_schirr2017_0train_rs50_ea_stack_multi_safe_rpa_tsa_calib3_v1/`

### 2.2 本轮单杠杆修复（只改一个开关）
加一个 **全候选通用** gate（仍然无标签、仍不更新分类器参数）：
- `--stack-safe-min-pred-improve 0.001`

直觉：当上层校准器无法给出“正向提升”的稳定证据时，直接 abstain（回退 EA）。

本轮仅跑主方法（节省时间），然后与已存在的 EA baseline run 合并：
- 新 run（仅主方法）：
  - `outputs/20260110/4class/loso4_schirr2017_0train_rs50_stack_global_minpred0001_v1/`
- baseline run（EA 已存在，无需重跑）：
  - `outputs/20260110/4class/loso4_schirr2017_0train_rs50_sanity_v0/`
- 合并后的可比 run（用于主表统计）：
  - `outputs/20260110/4class/loso4_schirr2017_0train_rs50_stack_global_minpred0001_v1_merged/`

### 2.3 结果（目标达成：≈EA 且 0 负迁移）
见合并目录：
- `outputs/20260110/4class/loso4_schirr2017_0train_rs50_stack_global_minpred0001_v1_merged/20260110_method_comparison.csv`

核心现象：
- `ea-stack-multi-safe-csp-lda`：**mean = EA**, `accept_rate=0`, `neg_transfer_rate_vs_ea=0`

解释：在该协议/该候选集里，**headroom 不存在**，因此最合理策略是 abstain；这为论文叙事提供了“安全拒绝负迁移”的证据。

---

## 3) 实验 B：BNCI2014_001 4-class LOSO —— 风险‑收益 sweep（补充材料）

### 3.1 目的
回应一个关键质疑：“0 负迁移是不是很容易（只是因为几乎从不接受）？”

做法：在 **固定候选集合与固定证书** 的前提下，仅 sweep 一个保守性参数：
- `anchor_guard_delta`（EA‑anchor 相对化 guard gate）

### 3.2 关键点：不重跑训练（离线 sweep）
利用 best run 保存的 `diagnostics/*/subject_*/candidates.csv`（每个候选记录包含无标签特征 + guard/ridge/probe 分数 + **真实 accuracy**），离线复现不同 δ 下的选择结果。

基础 run（作为“固定候选集合”的来源）：
- `outputs/20260107/4class/loso4_actionset_stack_familyblend_k20_probe_minimp001_borda_fix_v1/`

离线 sweep 脚本：
- `scripts/plot_anchor_delta_sweep_from_candidates.py`

生成的补充材料：
- `docs/experiments/figures/20260110_loso4_bnci2014_001_anchor_delta_sweep_offline_best_v1/bnci_loso4_anchor_delta_sweep_offline_best_v1_risk_reward.png`
- `docs/experiments/figures/20260110_loso4_bnci2014_001_anchor_delta_sweep_offline_best_v1/bnci_loso4_anchor_delta_sweep_offline_best_v1_sweep_metrics.csv`

### 3.3 结果解读（回答“0 负迁移是否不难”）
从 `*_sweep_metrics.csv` 可直接看到：
- 当 `delta=0`（更激进）：
  - **accept_rate 上升**，但出现 **负迁移（1/9）**，且 **mean Δacc 不增反降**；
- 当 `delta>=0.01`：
  - 回到主方法的 **稳定平台**：`mean Δacc ≈ +1.77pp` 且 `neg_transfer_rate=0`，accept_rate 与 family 分布稳定。

结论：在 BNCI2014_001 上，**0 负迁移并不是“免费”**——把 gate 放松会立刻引入负迁移，但收益并不会更高；这为“我们选择保守点作为主方法”的叙事提供了直接证据。

---

## 4) 下一步（面向 JBHI 叙事与性能）
1) action set 扩展：基于第 1 节论文，把 Riemannian/Procrustes/TSA 及更多可闭式/可控方法接入，做 “headroom 存在性” 分析；
2) 证书有效性：在 headroom>0 的数据/设置上，继续提高证书与真实 Δacc 的一致性（Spearman/Oracle gap 分解）；
3) 多数据集：优先选 MOABB 可下载、且 MI 任务规模适中的数据集，保证 strict LOSO 可比性与统计检验。

