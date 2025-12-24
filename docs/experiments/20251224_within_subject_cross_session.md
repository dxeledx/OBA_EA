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

### 2-class (paper_fir, CSP=6)

- Output: `outputs/20251224/2class/cross_session/2c_fir6_basic/20251224_results.txt`
- `ea-csp-lda` mean acc: **0.803241**
- `ea-zo-imr-csp-lda` mean acc: **0.805556**（小幅提升 +0.2315%）

结论：2 类跨 session 下，EA-ZO-IMR 有轻微增益，但幅度较小。
