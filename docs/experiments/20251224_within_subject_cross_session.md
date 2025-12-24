# 2025-12-24 — Within-subject cross-session (0train→1test)

## Goal

Switch from **cross-subject LOSO** to **within-subject cross-session** evaluation to:
- reduce domain shift (session shift < subject shift),
- potentially mitigate “unlabeled certificate failure” in TTA-Q selection,
- check whether EA-ZO / OEA-ZO can consistently beat EA in a cleaner setting.

## Protocol

- Dataset: MOABB `BNCI2014_001` (BCI IV 2a)
- Train sessions: `0train`
- Test sessions: `1test`
- For each subject `s`: train a subject-specific CSP+LDA on `0train`, evaluate on `1test`.

## Runner

- Script: `run_csp_lda_cross_session.py`
- Outputs: `outputs/YYYYMMDD/<N>class/cross_session/<run-name>/`

## Commands (to reproduce)

### 4-class

```bash
conda run -n eeg python run_csp_lda_cross_session.py \
  --preprocess paper_fir --n-components 6 \
  --events left_hand,right_hand,feet,tongue \
  --train-sessions 0train --test-sessions 1test \
  --methods ea-csp-lda,ea-zo-imr-csp-lda
```

### 2-class (left vs right)

```bash
conda run -n eeg python run_csp_lda_cross_session.py \
  --preprocess paper_fir --n-components 6 \
  --events left_hand,right_hand \
  --train-sessions 0train --test-sessions 1test \
  --methods ea-csp-lda,ea-zo-imr-csp-lda
```

## Results

### 4-class (paper_fir, CSP=6)

- Output: `outputs/20251224/4class/cross_session/4c_fir6_basic/20251224_results.txt`
- `ea-csp-lda` mean acc: **0.680170**
- `ea-zo-imr-csp-lda` (objective=`infomax_bilevel`) mean acc: **0.677469** (slightly worse)

Per-subject deltas (EA-ZO-IMR − EA):
- S1: 0.802083 → 0.802083 (0)
- S2: 0.517361 → 0.513889 (-0.003472)
- S3: 0.850694 → 0.850694 (0)
- S4: 0.631944 → 0.621528 (-0.010416)
- S5: 0.416667 → 0.406250 (-0.010417)
- S6: 0.541667 → 0.538194 (-0.003473)
- S7: 0.840278 → 0.840278 (0)
- S8: 0.833333 → 0.833333 (0)
- S9: 0.687500 → 0.690972 (+0.003472)

结论：跨 session 场景下，4 类任务的 EA 已经较强；当前 EA-ZO-IMR 并未带来整体提升。

补充：`oea-csp-lda`（伪标签对齐到训练 session signature，`--oea-pseudo-iters 2`）在本设置下更差：
- Output: `outputs/20251224/4class/cross_session/4c_fir6_oea/20251224_results.txt`
- `oea-csp-lda` mean acc: **0.674383**

#### Oracle headroom (analysis-only)

为了判断“方法还有没有上限空间”（是优化/证书问题，还是变换族太弱），我们做了 oracle 选解：
在同一组候选变换中，用 **真实标签 accuracy** 选出最优候选（只用于上限分析，不作为方法设置）。

- Output: `outputs/20251224/4class/cross_session/4c_fir6_oracle_Q/20251224_results.txt`
  - `ea-zo-imr-csp-lda` + oracle selector (orthogonal `Q`) mean acc: **0.681327**（几乎无 headroom）
- Output: `outputs/20251224/4class/cross_session/4c_fir6_oracle_rot_scale/20251224_results.txt`
  - `ea-zo-imr-csp-lda` + oracle selector (rot+scale `A=diag(exp(s))·Q`) mean acc: **0.693673**（对比 EA +1.3503%）

对应地，在同样的 rot+scale 变换族下，用当前无标签目标直接选（objective selector）会选错：
- Output: `outputs/20251224/4class/cross_session/4c_fir6_obj_rot_scale/20251224_results.txt`
  - `ea-zo-imr-csp-lda` (objective selector, rot+scale) mean acc: **0.677083**（低于 EA）

结论：在跨 session 的 4 类任务里，**“只旋转”很可能变换族太弱**；而加入“旋转+缩放”后出现显著 oracle headroom，
说明主要瓶颈转移到了 **无标签证书/选择规则**（objective 与真实 accuracy 不一致）。

### 2-class (paper_fir, CSP=6)

- Output: `outputs/20251224/2class/cross_session/2c_fir6_basic/20251224_results.txt`
- `ea-csp-lda` mean acc: **0.803241**
- `ea-zo-imr-csp-lda` mean acc: **0.805556**（小幅提升 +0.2315%）

结论：2 类跨 session 下，EA-ZO-IMR 有轻微增益，但幅度较小。

## LDA evidence certificate attempt (Dec 24)

动机：既然分类器是 CSP+LDA（近似高斯生成式），尝试用 **LDA 的无标签边缘似然**（evidence, `-log p(z)`）作为 candidate selection 的“证书”，看能否缓解无标签证书失效。

### 4-class: IMR optimization + evidence selector (rot+scale)

Command:

```bash
conda run -n eeg python run_csp_lda_cross_session.py \
  --preprocess paper_fir --n-components 6 \
  --events left_hand,right_hand,feet,tongue \
  --train-sessions 0train --test-sessions 1test \
  --methods ea-csp-lda,ea-zo-imr-csp-lda \
  --oea-zo-transform rot_scale \
  --oea-zo-selector evidence \
  --run-name 4c_fir6_imr_opt_evidence_sel_rot_scale \
  --diagnose-subjects 4
```

Results:
- Output: `outputs/20251224/4class/cross_session/4c_fir6_imr_opt_evidence_sel_rot_scale/20251224_results.txt`
- `ea-csp-lda` mean acc: **0.680170**
- `ea-zo-imr-csp-lda` + evidence selector mean acc: **0.680556**（+0.0386%）

Diagnostics (S4):
- `outputs/20251224/4class/cross_session/4c_fir6_imr_opt_evidence_sel_rot_scale/diagnostics/ea-zo-imr-csp-lda/subject_04/summary.txt`
  - best_by_accuracy ≠ best_by_evidence（evidence 仍会选回 identity）
  - 证书相关性仍不稳定（`pearson(evidence, accuracy)` 为正）

结论：evidence 作为 selector 在 4 类跨 session 上并没有显著解决“选不到 oracle candidate”的问题。

### 4-class: evidence as the optimization objective (lda_nll)

Command:

```bash
conda run -n eeg python run_csp_lda_cross_session.py \
  --preprocess paper_fir --n-components 6 \
  --events left_hand,right_hand,feet,tongue \
  --train-sessions 0train --test-sessions 1test \
  --methods ea-csp-lda,ea-zo-csp-lda \
  --oea-zo-objective lda_nll \
  --oea-zo-transform rot_scale \
  --run-name 4c_fir6_lda_nll_rot_scale \
  --diagnose-subjects 4
```

Results:
- Output: `outputs/20251224/4class/cross_session/4c_fir6_lda_nll_rot_scale/20251224_results.txt`
- `ea-zo-csp-lda` mean acc: **0.680170**（与 EA 完全一致；几乎总回退 identity）

结论：直接用 evidence 做优化目标会导致 SPSA 在 rot+scale 下更容易跑飞（产生极差 candidates），最终策略倾向选择 identity。

### 2-class: evidence selector

Orthogonal Q (more comparable to earlier 2-class basic):

```bash
conda run -n eeg python run_csp_lda_cross_session.py \
  --preprocess paper_fir --n-components 6 \
  --events left_hand,right_hand \
  --train-sessions 0train --test-sessions 1test \
  --methods ea-csp-lda,ea-zo-imr-csp-lda \
  --oea-zo-transform orthogonal \
  --oea-zo-selector evidence \
  --run-name 2c_fir6_imr_evidence_sel_Q
```

- Output: `outputs/20251224/2class/cross_session/2c_fir6_imr_evidence_sel_Q/20251224_results.txt`
- `ea-csp-lda` overall acc: **0.803241**
- `ea-zo-imr-csp-lda` + evidence selector overall acc: **0.804012**（+0.0771%）

Rot+scale:

- Output: `outputs/20251224/2class/cross_session/2c_fir6_imr_opt_evidence_sel_rot_scale/20251224_results.txt`
- `ea-zo-imr-csp-lda` + evidence selector mean acc: **0.801698**（低于 EA）

结论：2 类上 evidence selector 也没有优于已有 IMR/objective 选择（且 rot+scale 可能更不稳）。

## MixUp probe selector attempt (Dec 24)

动机：既然 oracle headroom 在 rot+scale 里存在，但常规无标签证书（objective / evidence）难以选到好解，
尝试一个更“判别式”的无标签证书：在 CSP 特征空间做 MixUp-style probe，一致性越好分数越低（越优）。

Command:

```bash
conda run -n eeg python run_csp_lda_cross_session.py \
  --preprocess paper_fir --n-components 6 \
  --events left_hand,right_hand,feet,tongue \
  --train-sessions 0train --test-sessions 1test \
  --methods ea-csp-lda,ea-zo-imr-csp-lda \
  --oea-zo-transform rot_scale \
  --oea-zo-selector probe_mixup \
  --run-name 4c_fir6_probe_mixup_sel_rot_scale
```

Results:
- Output: `outputs/20251224/4class/cross_session/4c_fir6_probe_mixup_sel_rot_scale/20251224_results.txt`
- `ea-csp-lda` mean acc: **0.680170**
- `ea-zo-imr-csp-lda` + probe_mixup selector mean acc: **0.680556**（+0.0386%）

结论：probe_mixup 相比 “objective selector (rot+scale)” 的 **明显掉点**（0.677083）更稳（接近 EA），
但仍未能逼近 oracle headroom（0.693673），说明证书/选择问题仍然是主瓶颈。
