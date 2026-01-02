# 2026-01-01 — 汇报用：CSP+LDA 证书从“失效”到“有效”的迭代记录（BNCI2014_001 4-class LOSO）

> 范围：**只统计 CSP+LDA family**（EA / EA‑ZO / EA‑SI‑CHAN*），不包含深度模型等；cross‑model 的 MDM 仅作为“扩展结果”附带一句。

---

## 0) 一页结论（可直接贴 PPT）

**协议（固定可比）**：MOABB `BNCI2014_001`，4‑class（`left_hand,right_hand,feet,tongue`），LOSO，`sessions=0train`，`paper_fir`（8–30 Hz, causal FIR order=50），epoch 0.5–3.5s，resample 250 Hz，`CSP n_components=6`。

**关键结论**：
- Anchor `ea-csp-lda`：mean acc **0.5320**，worst‑subject **0.2569**。
- 当前最好的 **CSP+LDA‑only** 方法：`ea-si-chan-multi-safe-csp-lda`
  - mean acc **0.5463**（**+1.43% abs** vs EA）
  - worst‑subject **0.2604**（略高于 EA）
  - `neg_transfer_rate_vs_ea = 0.0`
  - `accept_rate = 5/9 = 0.5556`
  - **证书有效性（跨被试）**：Spearman(`pred_improve`, `true_improve`) = **0.3667**
- （扩展，不算“CSP+LDA‑only”）`ea-mm-safe`：mean acc **0.5475**（+1.54% abs），但 candidate 中包含一次 MDM 选择。

---

## 1) Round‑0（失败）— 证书失效：无标签 objective 与真实精度不同向（S4 诊断）

### 1.1 上一次实验“失败现象”
我们希望用无标签 objective（InfoMax / entropy + marginal 约束等）在目标被试上挑选/优化 (Q)：
\[
Q^\star=\arg\min_{Q\in\mathcal C}\;J_{\text{unlabeled}}(Q),
\]
但在 **S4（4-class）** 上出现典型的 **certificate failure**：最小 objective 的候选并不是最高 accuracy 的候选（甚至更差）。

对应诊断实验见：`docs/experiments/20251223_S4_certificate_diagnostics.md:1`

关键事实（S4）：
- best by **true acc**：`identity`（EA）acc = **0.451389**
- best by **unlabeled objective**：`q_delta` acc = **0.368056**
- Spearman(objective, acc) ≈ **−0.046**（接近 0 / 反号）

**图（证书失效证据）**：
- objective vs acc（S4）：`docs/experiments/figures/20251223_S4_objective_vs_accuracy.png`
- `p̄` 轨迹（S4）：`docs/experiments/figures/20251223_S4_pbar_trajectory.png`

### 1.2 失败原因分析（为什么“证书会失效”）
在 4‑class 下，很多无标签 surrogate（熵最小化 / InfoMax / 边际均衡约束）会被两类因素主导而与真实风险脱钩：
- **marginal/均衡项主导**：模型可以通过“把预测推向更均匀/更高熵的边际”来降低 penalty，但并不保证分类边界更对。
- **负迁移放大**：当 anchor 预测本身不可靠时，无标签自证循环会把错的推得更自信，导致“objective 更好但 acc 更差”的反直觉候选被选中。

### 1.3 下一轮“准备怎么做”（从 Round‑0 失败导出的策略）
核心转向：把问题从“再造一个 objective”转为 **无标签模型选择/证书学习**，并引入“可拒绝负迁移”的安全化：
- 下层负责 **生成候选 (A / Q)**（对齐/投影等）。
- 上层负责 **学一个证书/守门员**，只在“有把握提升”时接受候选，否则回退到 EA anchor。

---

## 2) Round‑1（成功）— 双层分开改进：证书学习/校准 + safe fallback（CSP+LDA‑only）

> 对应完整实验记录：`docs/experiments/20251227_loso_4class_ea_si_chan_multi_safe.md:1`

### 2.1 这次准备怎么做（方法定义）

**下层（候选集 \(\mathcal A\)）：EA‑SI‑CHAN 投影族**
\[
\mathcal A=\{A_0=I\}\cup\{A(r,\lambda)\},
\]
其中 \(A(r,\lambda)\) 是由训练被试学到的 rank‑deficient channel projector（rank \(r\)，超参 \(\lambda\)）。

**上层（证书/guard + 选择规则 + 回退）**
在 source fold 内构造 pseudo‑target 被试，得到每个候选的“真实提升”监督信号：
\[
\Delta\mathrm{acc}(A)=\mathrm{acc}_{\text{pseudoT}}(A)-\mathrm{acc}_{\text{pseudoT}}(A_0).
\]
然后学习两类上层模型（都不看真正 test subject 标签）：
- 证书回归器（ridge）：\(\hat\Delta(A)\approx\Delta\mathrm{acc}(A)\)
- 守门员（logistic guard）：\(g(A)=\Pr(\Delta\mathrm{acc}(A)\ge m)\)

目标被试选择规则（无标签）：
\[
\text{pick }A^\star=\arg\max_{A\in\mathcal A}\hat\Delta(A)\quad
\text{s.t. }g(A)\ge \tau;\;\text{且 }\hat\Delta(A)>0,
\]
否则回退到 \(A_0=I\)（EA anchor）。

### 2.2 怎么做的（命令 + 输出）
```bash
conda run -n eeg python run_csp_lda_loso.py \
  --preprocess paper_fir --n-components 6 \
  --events left_hand,right_hand,feet,tongue --sessions 0train \
  --methods ea-csp-lda,ea-si-chan-csp-lda,ea-si-chan-multi-safe-csp-lda \
  --si-proj-dim 21 --si-subject-lambda 1 --si-ridge 1e-6 \
  --si-chan-ranks 21 --si-chan-lambdas 0.5,1,2 \
  --oea-zo-selector calibrated_ridge_guard \
  --oea-zo-calib-max-subjects 0 --oea-zo-calib-seed 0 \
  --oea-zo-calib-guard-threshold 0.5 \
  --no-plots \
  --run-name loso4_easichan_multi_safe_r21_l05-2_calAll_thr05
```
Outputs：
- `outputs/20251227/4class/loso4_easichan_multi_safe_r21_l05-2_calAll_thr05/20251227_method_comparison.csv`
- `outputs/20251227/4class/loso4_easichan_multi_safe_r21_l05-2_calAll_thr05/20251227_results.txt`
- `outputs/20251227/4class/loso4_easichan_multi_safe_r21_l05-2_calAll_thr05/20251227_predictions_all_methods.csv`

### 2.3 最终结果（主表指标 + 图）
主表（来自 `20251227_method_comparison.csv`）：
- `ea-csp-lda` mean acc: **0.5320**
- `ea-si-chan-csp-lda` mean acc: **0.5394**
- `ea-si-chan-multi-safe-csp-lda` mean acc: **0.5463**（**+1.43% abs vs EA**）
- `neg_transfer_rate_vs_ea`: **0.0**
- worst‑subject acc: **0.2604**

证书有效性（来自 `20251227_results.txt`）：`cert_improve_spearman` = **0.3667**

**图（稳定性 + 证书有效性）**：
- per‑subject Δacc（橙色=接受候选；蓝色=回退到 EA）：`docs/experiments/figures/20251227_loso4_easichan_multi_safe_delta.png`
- 证书预测提升 vs 真实提升：`docs/experiments/figures/20251227_loso4_easichan_multi_safe_cert_vs_true.png`

### 2.4 这次说明了什么（可汇报表述）
1) Round‑0 的“无标签 objective 失效”可以通过 **证书学习/校准 + guard + fallback** 显著缓解：我们能在 **不引入负迁移** 的前提下获得 **≈+1.5%** 的 mean 提升。  
2) 证书不需要完美相关，但需要 **“比随机更相关 + 能拒绝坏候选”**：Spearman ≈ 0.37 与 `neg_transfer_rate=0` 同时出现，说明证书开始“可用”。

---

## 3) Round‑2（尝试，未达标）— 半联动：连续 λ 搜索（SPSA）会引入负迁移

> 完整记录：`docs/experiments/20260102_loso4_bnci2014_001_chan_spsa_safe.md:1`

动机：希望把下层从离散网格（`λ∈{0.5,1,2}`）升级到连续 λ 搜索，以扩大候选覆盖并进一步提高 mean。

方法：`ea-si-chan-spsa-safe-csp-lda`（目标被试上用 SPSA 优化 `φ=log λ`，上层仍用 calibrated ridge/guard + fallback）。

结果（同协议 BNCI2014_001 4‑class LOSO）：
- mean acc **0.5436**（**+1.16% abs** vs EA），但
- worst‑subject **0.2396**（低于 EA 的 0.2569）
- `neg_transfer_rate_vs_ea = 0.3333`（负迁移回来了）
- `accept_rate = 7/9` 且 S1/S2/S8 被误接受后真实掉点（每个 −0.01736）

诊断图：
- `docs/experiments/figures/20260102_loso4_chan_spsa_safe_delta_compare.png`
- `docs/experiments/figures/20260102_loso4_chan_spsa_safe_lambda_vs_true.png`
- `docs/experiments/figures/20260102_loso4_chan_spsa_safe_pred_vs_true.png`

结论：连续搜索“有增益但不稳”，当前不适合作为主方法；若继续，需要先把接收规则/λ‑trust‑region 做到 **neg_transfer≈0** 再谈提升。

---

## 4) Round‑3（改进，仍未达标）— 连续 λ 加入 warm‑start + trust‑region：负迁移显著减少但未清零

> 完整记录：`docs/experiments/20260102_loso4_bnci2014_001_chan_spsa_safe_warmstart.md:1`

动机：上一轮负迁移主要来自“off‑grid 外推”，因此本轮只动下层搜索策略：
- 在 `{0.5,1,2}` 网格上先选 predicted‑best 做 warm‑start；
- 把 SPSA 的 λ 搜索限制在网格附近（trust‑region）。

结果（同协议 BNCI2014_001 4‑class LOSO）：
- `ea-si-chan-spsa-safe-csp-lda` mean acc **0.5440**（+**1.20%** abs vs EA）
- `neg_transfer_rate_vs_ea = 0.1111`（只剩 **S2** 一例负迁移，但仍无法保证 safety）
- worst‑subject **0.2326**（仍低于 EA 的 0.2569）

结论：限制外推能“止血”，但连续 λ 仍会诱导证书外推误判（S2：ridge/guard 仍高置信误接受）。若继续沿连续方向，需要把证书训练分布扩展到 off‑grid 候选，或改成“密网格”替代连续搜索。

---

## 5) Round‑4（扩展，非 CSP+LDA‑only）— 加入跨 family 候选（EA‑MM‑SAFE）

> 仅用于补充：`docs/experiments/20251231_loso4_bnci2014_001_ea_chan_mdm_mmsafe.md:1`

在保持同协议下，`ea-mm-safe` 达到 mean acc **0.5475**（+1.54% abs），并且 `neg_transfer_rate_vs_ea=0.0`；但其 candidate set 含一次 MDM 选择，因此不计入“CSP+LDA‑only 主线”。

---

## 6) 下一次实验记录模板（从现在开始按这个写）

每次新实验笔记建议固定为四段（对应你要求的“先分析失败→再计划→再执行→再结果”）：

1) **上一次失败分析（post‑mortem）**  
   - mean / worst‑subject / neg‑transfer‑rate  
   - Top‑k 掉点被试  
   - confusion matrix / 预测边际（如有）  
   - 证书有效性：Spearman(证书, acc) 或 Spearman(预测提升, 真实提升)、oracle gap（若有 candidates）

2) **本次假设与准备做什么（只动一个杠杆）**  
   - 明确改动点（例如：特征集/候选集/证书特征/guard 阈值/漂移约束）  
   - 预期影响：提升均值 or 降低掉点 or 提升证书相关性（至少写一个）

3) **怎么做的（命令 + 输出目录）**  
   - 贴完整命令  
   - 写明输出目录（`outputs/...`）

4) **最终结果（主表 + 图）**  
   - 主表：mean / worst / neg‑transfer  
   - 至少 1 张：per‑subject Δacc 图  
   - 若主打“证书有效性”：附证书‑acc 图（散点 + 相关系数）
