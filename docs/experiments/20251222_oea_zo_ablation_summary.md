# 2025-12-22 — OEA / OEA-ZO Ablations (LOSO, BNCI2014_001, 2-class)

This note summarizes the ablations requested in the discussion:

- **Priority 0 (no code change):** sweep `--oea-q-blend` back to `0.3/0.2`, and add `oea-csp-lda` with `--oea-pseudo-iters 0` as an intermediate point to separate “source Δ-alignment” vs “target Q_t selection”.
- **Priority 1/2 (code change):** add (a) “reliable trial keep” for entropy/infomax/confidence objectives, (b) unlabeled holdout for best-iterate selection, (c) warm-start from pseudo-Δ alignment, (d) an unlabeled safety fallback (class-marginal entropy).

## Environment / common setup
- Dataset: MOABB `BNCI2014_001` (BCI Competition IV 2a)
- Classes: `left_hand` vs `right_hand` (2-class)
- Session(s): `0train`
- Evaluation: LOSO (9 subjects)
- Metrics: mean accuracy reported below (full metrics saved in each run folder)

## Key takeaway (what the numbers show)
1. **`--oea-q-blend` is a critical knob** because it affects *both* (a) source-subject alignment used to train CSP+LDA and (b) target-subject alignment at test time.  
   In these runs, `q_blend=0.5` consistently hurts compared to `0.2–0.3`.
2. Adding `oea-csp-lda` with `--oea-pseudo-iters 0` makes it clear that **a lot of the gain comes from source-side Δ-alignment alone**, while **target-side ZO adds only small extra gains** (sometimes none).
3. The “stability options” (keep/holdout/warm-start/fallback) were implemented and tested, but under these tested settings they were **often conservative** and did **not reliably improve the mean** over the simpler `q_blend=0.2–0.3` baseline.

---

## A) “CSP=4 + non-causal preprocessing (MOABB)” run (the 69% vs 72% comparison)
This is the setup you mentioned: **EA ≈ 69%**, **ours ≈ 72%**.

Command:
```bash
conda run -n eeg python run_csp_lda_loso.py \
  --preprocess moabb --n-components 4 \
  --oea-pseudo-iters 0 \
  --methods csp-lda,ea-csp-lda,oea-csp-lda,oea-zo-ent-csp-lda,oea-zo-pce-csp-lda \
  --run-name moabb_csp4_ablation
```

Mean accuracy (from `outputs/20251222/moabb_csp4_ablation/20251222_results.txt`):
- `csp-lda`: **0.690586**
- `ea-csp-lda`: **0.698302**
- `oea-csp-lda` (`pseudo-iters=0`, source Δ only): **0.723765**
- `oea-zo-ent-csp-lda`: **0.722222**
- `oea-zo-pce-csp-lda`: **0.723765**

---

## B) Priority 0 ablation on paper-aligned preprocessing (paper_fir, CSP=6)

### B1) `q_blend=0.3` + include `oea-csp-lda --oea-pseudo-iters 0`
Command:
```bash
conda run -n eeg python run_csp_lda_loso.py \
  --preprocess paper_fir --n-components 6 \
  --oea-q-blend 0.3 --oea-pseudo-iters 0 \
  --methods ea-csp-lda,oea-csp-lda,oea-zo-ent-csp-lda,oea-zo-im-csp-lda,oea-zo-pce-csp-lda \
  --run-name p0_blend03
```

Mean accuracy (from `outputs/20251222/p0_blend03/20251222_results.txt`):
- `ea-csp-lda`: **0.732253**
- `oea-csp-lda` (`pseudo-iters=0`): **0.734568**
- `oea-zo-ent-csp-lda`: **0.733796**
- `oea-zo-im-csp-lda`: **0.733796**
- `oea-zo-pce-csp-lda`: **0.735340**

### B2) `q_blend=0.2` + include `oea-csp-lda --oea-pseudo-iters 0`
Command:
```bash
conda run -n eeg python run_csp_lda_loso.py \
  --preprocess paper_fir --n-components 6 \
  --oea-q-blend 0.2 --oea-pseudo-iters 0 \
  --methods ea-csp-lda,oea-csp-lda,oea-zo-ent-csp-lda,oea-zo-im-csp-lda,oea-zo-pce-csp-lda \
  --run-name p0_blend02
```

Mean accuracy (from `outputs/20251222/p0_blend02/20251222_results.txt`):
- `ea-csp-lda`: **0.732253**
- `oea-csp-lda` (`pseudo-iters=0`): **0.733025**
- `oea-zo-ent-csp-lda`: **0.734568**
- `oea-zo-im-csp-lda`: **0.734568**
- `oea-zo-pce-csp-lda`: **0.733796**

---

## C) “Unstable regime” example (`q_blend=0.5`)

### C1) Without holdout/warm-start/fallback (keep selection is active via pseudo args)
Command:
```bash
conda run -n eeg python run_csp_lda_loso.py \
  --preprocess paper_fir --n-components 6 \
  --oea-pseudo-iters 0 --oea-q-blend 0.5 \
  --oea-zo-iters 100 --oea-zo-k 20 --oea-zo-lr 0.3 --oea-zo-mu 0.05 --oea-zo-l2 0.001 \
  --oea-pseudo-confidence 0.7 --oea-pseudo-topk-per-class 40 --oea-pseudo-balance \
  --methods ea-csp-lda,oea-csp-lda,oea-zo-ent-csp-lda,oea-zo-im-csp-lda,oea-zo-pce-csp-lda \
  --run-name stab_qb05_keep
```

Mean accuracy (from `outputs/20251222/stab_qb05_keep/20251222_results.txt`):
- `ea-csp-lda`: **0.732253**
- `oea-csp-lda` (`pseudo-iters=0`): **0.724537**
- `oea-zo-ent-csp-lda`: **0.726080**
- `oea-zo-im-csp-lda`: **0.726080**
- `oea-zo-pce-csp-lda`: **0.726080**

### C2) With holdout + warm-start(Δ) + fallback enabled
Command:
```bash
conda run -n eeg python run_csp_lda_loso.py \
  --preprocess paper_fir --n-components 6 \
  --oea-pseudo-iters 0 --oea-q-blend 0.5 \
  --oea-zo-holdout-fraction 0.5 \
  --oea-zo-warm-start delta --oea-zo-warm-iters 2 \
  --oea-zo-fallback-min-marginal-entropy 0.2 \
  --oea-zo-iters 100 --oea-zo-k 20 --oea-zo-lr 0.3 --oea-zo-mu 0.05 --oea-zo-l2 0.001 \
  --oea-pseudo-confidence 0.7 --oea-pseudo-topk-per-class 40 --oea-pseudo-balance \
  --methods ea-csp-lda,oea-csp-lda,oea-zo-ent-csp-lda,oea-zo-im-csp-lda,oea-zo-pce-csp-lda \
  --run-name improved_qb05
```

Mean accuracy (from `outputs/20251222/improved_qb05/20251222_results.txt`):
- `ea-csp-lda`: **0.732253**
- `oea-csp-lda` (`pseudo-iters=0`): **0.724537**
- `oea-zo-ent-csp-lda`: **0.716049**
- `oea-zo-im-csp-lda`: **0.716821**
- `oea-zo-pce-csp-lda`: **0.719136**

---

## D) Priority 1/2 options tested in a “good” blend regime (`q_blend=0.3`)

### D1) Aggressive ZO hyperparameters + stability options
Command:
```bash
conda run -n eeg python run_csp_lda_loso.py \
  --preprocess paper_fir --n-components 6 \
  --oea-pseudo-iters 0 --oea-q-blend 0.3 \
  --oea-zo-holdout-fraction 0.5 \
  --oea-zo-warm-start delta --oea-zo-warm-iters 2 \
  --oea-zo-fallback-min-marginal-entropy 0.2 \
  --oea-zo-iters 100 --oea-zo-k 20 --oea-zo-lr 0.3 --oea-zo-mu 0.05 --oea-zo-l2 0.001 \
  --oea-pseudo-confidence 0.7 --oea-pseudo-topk-per-class 40 --oea-pseudo-balance \
  --methods ea-csp-lda,oea-csp-lda,oea-zo-ent-csp-lda,oea-zo-im-csp-lda,oea-zo-pce-csp-lda \
  --run-name improved_qb03
```

Mean accuracy (from `outputs/20251222/improved_qb03/20251222_results.txt`):
- `ea-csp-lda`: **0.732253**
- `oea-csp-lda` (`pseudo-iters=0`): **0.734568**
- `oea-zo-ent-csp-lda`: **0.734568**
- `oea-zo-im-csp-lda`: **0.734568**
- `oea-zo-pce-csp-lda`: **0.732253**

### D2) Default ZO hyperparameters + stability options
Command:
```bash
conda run -n eeg python run_csp_lda_loso.py \
  --preprocess paper_fir --n-components 6 \
  --oea-pseudo-iters 0 --oea-q-blend 0.3 \
  --oea-zo-holdout-fraction 0.5 \
  --oea-zo-warm-start delta --oea-zo-warm-iters 2 \
  --oea-zo-fallback-min-marginal-entropy 0.2 \
  --oea-pseudo-confidence 0.7 --oea-pseudo-topk-per-class 40 --oea-pseudo-balance \
  --methods ea-csp-lda,oea-csp-lda,oea-zo-ent-csp-lda,oea-zo-im-csp-lda,oea-zo-pce-csp-lda \
  --run-name improved_defaultzo
```

Mean accuracy (from `outputs/20251222/improved_defaultzo/20251222_results.txt`):
- `ea-csp-lda`: **0.732253**
- `oea-csp-lda` (`pseudo-iters=0`): **0.734568**
- `oea-zo-ent-csp-lda`: **0.732253**
- `oea-zo-im-csp-lda`: **0.732253**
- `oea-zo-pce-csp-lda`: **0.733025**

