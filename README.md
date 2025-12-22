# CSP+LDA (MOABB BNCI2014-001 / BCI Competition IV 2a)

本项目实现脑电运动意图识别中的 **CSP + LDA** 基线方法：

- 数据集：MOABB `BNCI2014_001`（BCI Competition IV 2a，4 类：left/right/feet/tongue）
- 默认对齐 He & Wu (EA) 论文的 2 类设置：left vs right，仅使用训练会话 `0train`
- 预处理（MOABB paradigm）：8–30 Hz 带通、重采样 250 Hz、分段 `tmin=0.5s` 到 `tmax=3.5s`（相对 cue 出现后的时间窗）
- 模型：CSP（`n_components=4`）+ LDA（默认参数）
- 评估：Leave-One-Subject-Out（LOSO）交叉验证，默认同时运行 `csp-lda` 与 `ea-csp-lda`

## 环境

建议在 `conda` 环境 `eeg` 中运行（用户需求）。

如需从 GitHub 安装 MOABB（示例）：

```bash
conda activate eeg
pip install -U "git+https://github.com/NeuroTechX/moabb.git"
```

## 运行

```bash
conda run -n eeg python run_csp_lda_loso.py
```

默认会在 `outputs/YYYYMMDD/HHMMSS/` 目录生成（同一天多次运行不会覆盖）：

- `YYYYMMDD_results.txt`
- `YYYYMMDD_csp-lda_csp_patterns.png`
- `YYYYMMDD_csp-lda_confusion_matrix.png`
- `YYYYMMDD_ea-csp-lda_csp_patterns.png`
- `YYYYMMDD_ea-csp-lda_confusion_matrix.png`
- `YYYYMMDD_oea-cov-csp-lda_csp_patterns.png`（如启用）
- `YYYYMMDD_oea-cov-csp-lda_confusion_matrix.png`（如启用）
- `YYYYMMDD_oea-csp-lda_csp_patterns.png`（如启用）
- `YYYYMMDD_oea-csp-lda_confusion_matrix.png`（如启用）
- `YYYYMMDD_oea-zo-csp-lda_csp_patterns.png`（如启用）
- `YYYYMMDD_oea-zo-csp-lda_confusion_matrix.png`（如启用）
- `YYYYMMDD_model_compare_accuracy.png`

可选参数：

```bash
conda run -n eeg python run_csp_lda_loso.py --out-dir ./outputs --run-name exp1
```

启用论文中的因果 FIR(Hamming)（50 阶）带通滤波（可随时开关）：

```bash
conda run -n eeg python run_csp_lda_loso.py --preprocess paper_fir
```

复现论文 Table I（约 67% vs 73%）的常用设置（经验上需要更多 CSP 分量，例如 6）：

```bash
conda run -n eeg python run_csp_lda_loso.py --preprocess paper_fir --n-components 6
```

## OEA：从 EA 解集选择 Q（你的改进点落地）

在 EA 中每个被试的白化矩阵可写成解集：

`W_s = Q_s C_s^{-1/2},  Q_s ∈ O(C)`。

这里提供两个可运行的选择策略（都不使用目标被试真实标签、也不更新分类器参数）：

- `oea-cov-csp-lda`：**无监督**策略，按训练被试的平均协方差构造参考特征基 `U_ref`，并选择 `Q_s=U_ref U_s^T`（`U_s` 为每个被试 `C_s` 的特征向量）。
- `oea-csp-lda`：**判别式（更“乐观”）**策略，使用 `Δ=Cov(class1)-Cov(class0)` 做二阶判别签名，对训练被试用真标签构造 `Δ_ref` 并选 `Q_s`；对目标被试用模型预测的伪标签迭代 `--oea-pseudo-iters` 次估计 `Δ_t` 从而选 `Q_t`。
- `oea-zo-csp-lda`：**零阶优化（主创新候选）**：源域同 `oea-csp-lda`（用 `Δ_ref` 选 `Q_s`），目标域仅用无标签数据，冻结分类器，直接在正交群上（Givens 低维参数化）用 SPSA 零阶方法优化 `Q_t` 的分类相关目标（默认 `entropy`；也支持 `infomax` / `pseudo_ce` / `confidence`）。

为方便一次运行对比不同 ZO 目标，额外提供方法别名（仅覆盖 `--oea-zo-objective`，其余参数不变）：

- `oea-zo-ent-csp-lda`（OEA-ZO，entropy，主推）
- `oea-zo-im-csp-lda`（OEA-ZO-IM，infomax，防止 entropy 塌缩的稳健变体/主推候选）
- `oea-zo-pce-csp-lda`（OEA-ZO-pCE，非光滑变体/消融）
- `oea-zo-conf-csp-lda`（OEA-ZO-conf，置信度目标/消融）

运行示例：

```bash
conda run -n eeg python run_csp_lda_loso.py --preprocess paper_fir --n-components 6 --methods csp-lda,ea-csp-lda,oea-cov-csp-lda,oea-csp-lda,oea-zo-csp-lda
```

可调参数：

```bash
conda run -n eeg python run_csp_lda_loso.py --oea-pseudo-iters 2 --oea-eps 1e-10 --oea-shrinkage 0.0
```

如需让 `Q` 更“保守”（更接近 EA 的 `Q=I`），可调：

```bash
conda run -n eeg python run_csp_lda_loso.py --oea-q-blend 0.3
```

### 建议的论文口径（严格 LOSO vs TTA）

- **严格 LOSO（主表建议）**：`--oea-pseudo-iters 0`（目标域不做伪标签迭代，只做 EA 的 `C_t^{-1/2}`；分类器不更新）。  
- **TTA-Q（附表/补充）**：`--oea-pseudo-iters 1~2`（仍不更新分类器，仅用无标签目标数据做 Q_t 迭代）。

TTA-Q 稳健化参数（可选）：

```bash
conda run -n eeg python run_csp_lda_loso.py --oea-pseudo-mode hard --oea-pseudo-confidence 0.7 --oea-pseudo-topk-per-class 40 --oea-pseudo-balance
```

OEA-ZO（零阶）常用参数（可选）：

```bash
conda run -n eeg python run_csp_lda_loso.py --methods oea-zo-ent-csp-lda --oea-zo-iters 30 --oea-zo-k 50 --oea-zo-lr 0.5 --oea-zo-mu 0.1
```

对比 entropy vs infomax vs pCE（同一次运行输出到同一份 `YYYYMMDD_results.txt`）：

```bash
conda run -n eeg python run_csp_lda_loso.py --methods oea-zo-ent-csp-lda,oea-zo-im-csp-lda,oea-zo-pce-csp-lda
```
