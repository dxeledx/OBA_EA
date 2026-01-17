# SAFE‑TTA (Safety‑Aware Fallback‑Enabled Test‑Time Adaptation)
## Methods draft (for advisor report / paper draft)

**Working title**: *SAFE‑TTA: Certificate‑Guided Risk‑Controlled Test‑Time Adaptation for Cross‑Subject EEG Decoding*  
**Scope of this draft**: the **CSP+LDA** line (including EA / closed‑form alignment baselines + our safe test‑time action selection).  
**Strict protocol**: selection/adaptation uses **target unlabeled data only**; target labels are used **only for evaluation**.

---

## 1. Problem setting (cross‑subject strict LOSO TTA)

We consider cross‑subject EEG decoding under a strict leave‑one‑subject‑out protocol (LOSO). Let
- subjects \(s \in \{1,\dots,S\}\),
- target/test subject \(t\),
- trials \( (X_i^{(t)}, y_i^{(t)})_{i=1}^{n_t}\) for the target, where \(X_i \in \mathbb{R}^{C\times T}\) is an EEG epoch and \(y_i\in\{1,\dots,K\}\).

We train a decoder on source subjects \(\{s\neq t\}\) and **do not update classifier parameters at test time**.  
At test time, we allow **label‑free** test‑time adaptation/model selection to mitigate subject shift, but must control the risk of **negative transfer**.

We denote the (unknown) target risk of a test‑time strategy \(a\) by
\[
R_t(a) \;=\; \mathbb{E}_{(X,y)\sim \mathcal{D}_t}\big[\mathbb{1}(\hat y_a(X)\neq y)\big].
.\]
The scientific difficulty is that \(R_t(a)\) is unobservable at test time (no labels), while common unsupervised proxies can be **misaligned** with true risk in low‑SNR MI‑EEG.

---

## 2. Key idea: TTA as **risk‑controlled action selection** (with a safe anchor)

SAFE‑TTA casts test‑time adaptation as selecting an “action” from an **action set** \(\mathcal{A}\) using only unlabeled target data:

- **Anchor action** \(a_0\): a strong, stable default strategy (in this repo: `ea-csp-lda`).  
- **Candidate actions** \(a\in\mathcal{A}\setminus\{a_0\}\): alternative alignments / feature extractors / classifiers that may improve accuracy but can also harm some subjects (high variance / high risk).

At test time, SAFE‑TTA outputs either a candidate action (if certified safe) or **falls back** to the anchor:
\[
a^\star \;=\; \textsf{Select}(\mathcal{A}, X^{(t)};\ \psi)\quad\Rightarrow\quad
\hat y = \hat y_{a^\star}(X).
\]

The core novelty is **certificate‑guided risk control**: we do not assume any single unlabeled proxy is perfect; instead we (i) compute multiple certificate signals, (ii) calibrate them on source subjects, and (iii) add explicit gates and a fallback‑to‑anchor rule to control negative transfer.

---

## 3. Base decoder family (CSP+LDA)

All actions in this draft share the same base classifier family:

1) (Optional) **Alignment / spatial transform** applied to epochs \(X\) (label‑free).  
2) **CSP feature extractor** (fitted on source training data).  
3) **LDA classifier** (fitted on source training data; frozen at test time).

We denote the resulting predicted probabilities for a target trial by
\[
p_a(y\mid X_i) \in \Delta^{K-1}.
\]

---

## 4. Action set \(\mathcal{A}\): what can be selected at test time?

SAFE‑TTA supports a **mixed action set** (discrete candidates), e.g.:

### 4.1 Anchor (always included)
- **EA‑CSP‑LDA** (`ea-csp-lda`): per‑subject label‑free whitening (EA) + CSP + LDA.

### 4.2 Closed‑form alignment candidates (low cost)
- **LEA‑CSP‑LDA** (`lea-csp-lda`, historically `rpa-csp-lda`): closed‑form alignment baseline.
- **LEA‑ROT‑CSP‑LDA** (`lea-rot-csp-lda`, historically `tsa-csp-lda`): closed‑form rotation baseline.

These are deterministic actions; the only decision is whether to adopt them for a target subject.

### 4.3 High‑reward / high‑risk candidates (require safety gating)
- **EA‑FBCSP‑LDA** (`ea-fbcsp-lda`): EA + filterbank CSP + feature selection + LDA (frozen).  
  Empirically: can improve mean, but can catastrophically hurt some subjects (especially on HGD), so SAFE‑TTA treats it as “high risk”.

### 4.4 Parametric candidate families (grid → multiple candidates)
- **Channel‑invariant candidates** (`chan` family): a grid over \((\text{rank},\lambda)\) produces multiple candidate actions.  
  This increases headroom but also increases **multiple‑testing false positives**, motivating stricter risk control.

> In implementation, each candidate action \(a\) produces a record with `cand_family`, `cand_rank`, `cand_lambda`, and target unlabeled prediction statistics.

---

## 5. Certificates: unlabeled signals computed on target data

For each candidate \(a\), SAFE‑TTA computes a certificate feature vector
\[
\phi(a;X^{(t)}) \in \mathbb{R}^d
\]
from unlabeled target predictions \(p_a(y\mid X_i)\) (and optionally from the anchor predictions \(p_{a_0}\)).

### 5.1 Base statistics (examples)
Typical statistics include:
- mean entropy \(\frac{1}{n}\sum_i H(p_a(\cdot|X_i))\),
- marginal distribution entropy \(H(\bar p_a)\), \(\bar p_a=\frac1n\sum_i p_a(\cdot|X_i)\),
- confidence statistics \(\mathbb{E}[\max_k p_{a,k}]\),
- drift statistics relative to anchor \(D_{\mathrm{KL}}(p_{a_0}\,\|\,p_a)\) and its quantiles,
- sample‑selection coverage (if a reliability filter is used),
- multi‑certificate stacking signals (e.g., evidence‑NLL, mixup probes).

### 5.2 Stacked certificates (certificate stacking)
SAFE‑TTA uses a **stacked feature vector** that concatenates:
- the base candidate features, plus
- multiple certificate signals (e.g., evidence‑NLL, probe mixup scores, hard‑probe scores).

### 5.3 Anchor‑relative (delta) features (recommended)
To improve cross‑dataset stability, SAFE‑TTA can use **anchor‑relative features**
\[
\phi_\Delta(a;X^{(t)}) \;=\; \phi(a;X^{(t)}) - \phi(a_0;X^{(t)})
\]
for all non‑meta features, while keeping candidate meta info (family one‑hots / rank / lambda) as absolute.

This matches the learning target: **predict improvement over the anchor**.

---

## 6. Calibration on source subjects (learning a reliable selector)

SAFE‑TTA learns a selector using only **training subjects** in each LOSO fold.

### 6.1 Pseudo‑target construction (within a fold)
For a LOSO fold with test subject \(t\), we treat each source subject \(s\neq t\) as a *pseudo‑target*:
- train the base models on remaining training subjects,
- evaluate candidate actions on pseudo‑target \(s\),
- compute unlabeled features \(\phi(\cdot;X^{(s)})\),
- compute the **true improvement** (using pseudo‑target labels, allowed because \(s\) is not the test subject):
\[
\Delta \mathrm{acc}_s(a) \;=\; \mathrm{acc}_s(a) - \mathrm{acc}_s(a_0).
\]

### 6.2 Two learned components
SAFE‑TTA learns:

**(i) Calibrated certificate regressor (ridge)**  
Predict expected improvement:
\[
\widehat{\Delta}(a) \;=\; g_\psi(\phi_\Delta(a;X^{(t)})).
\]

**(ii) Safety guard classifier (logistic)**  
Predict probability that improvement is positive (or exceeds a margin):
\[
\widehat{p}_{\mathrm{pos}}(a) \;=\; h_\psi(\phi_\Delta(a;X^{(t)})) \approx \mathbb{P}(\Delta \mathrm{acc}>0).
\]

Optionally we use **per‑family calibration** (separate \(g_{\psi,f},h_{\psi,f}\) per candidate family) with shrinkage/blending to stabilize families with few training samples.

---

## 7. Risk‑controlled test‑time selection with fallback (SAFE‑TTA rule)

At test time for subject \(t\), we compute candidate records for all \(a\in\mathcal{A}\) and then apply:

### 7.1 Primary guard gate (reject likely negative transfer)
Keep only candidates with:
\[
\widehat{p}_{\mathrm{pos}}(a) \;\ge\; \tau_{\mathrm{guard}}.
\]

### 7.2 Anchor‑relative guard tightening (multiple testing correction)
To counter “candidate explosion”, require:
\[
\widehat{p}_{\mathrm{pos}}(a) \;\ge\; \widehat{p}_{\mathrm{pos}}(a_0) + \delta_{\mathrm{anchor}}.
\]

### 7.3 Optional certificate gate (anti‑collapse)
For some runs we also require the hard‑probe certificate not to be worse than anchor by more than \(\epsilon\):
\[
\mathrm{probe\_hard}(a) \;\le\; \mathrm{probe\_hard}(a_0) + \epsilon_{\mathrm{worsen}}.
\]

### 7.4 Global minimum predicted improvement gate (risk‑control lever)
Require the predicted gain exceed a minimum:
\[
\widehat{\Delta}(a) \;\ge\; \epsilon_{\min}.
\]
If the selected candidate fails this, SAFE‑TTA searches for the best alternative satisfying all gates; otherwise it **falls back** to \(a_0\).

### 7.5 Family‑specific high‑risk gates (examples)
Certain families are treated as high risk and get additional gates. Example: **EA‑FBCSP** can be gated by a simple unlabeled disagreement certificate:
\[
\mathrm{pred\_disagree}(a) \;=\; \frac{1}{n}\sum_i \mathbf{1}[\arg\max p_{a_0}(X_i)\neq \arg\max p_a(X_i)].
\]
Accept the candidate only if \(\mathrm{pred\_disagree}(a)\le \tau_f\).  
Crucially, \(\tau_f\) can be **trained/calibrated on training subjects only** (LOSO‑style), avoiding test‑label leakage.

### 7.6 Final selection
Among the remaining candidates, select by maximum \(\widehat{\Delta}(a)\) (or a Borda‑style aggregation that also favors probe improvements).  
If no candidate passes gates, output the anchor \(a_0\).

This yields a deployment‑friendly guarantee: **never worse than the anchor under the same decision rule** (fallback), and empirically drives negative transfer toward 0.

---

## 8. Figure descriptions (what to draw in the paper)

### Fig. 1 — SAFE‑TTA overview (train vs test)
**Layout**: two columns.

**Left column (Training / calibration within a LOSO fold)**:
1) Source subjects \(\{s\neq t\}\) → train anchor model \(a_0\) and candidate families.
2) Pseudo‑target loop: each pseudo‑target subject produces candidate records:
   - compute unlabeled features \(\phi_\Delta\),
   - compute true \(\Delta \mathrm{acc}\) (labels available in training fold),
   - train \(g_\psi\) (ridge) and \(h_\psi\) (logistic guard).

**Right column (Test / deployment on target subject \(t\))**:
1) Target unlabeled trials \(X^{(t)}\) → run anchor and all candidates → candidate records.
2) Compute certificates/features \(\phi_\Delta\).
3) Apply guard gates (global + family‑specific).
4) Select best passing candidate; else fallback to anchor.
5) Output predictions.

### Fig. 2 — Action set (candidate families)
**Layout**: a tree or table:
- Anchor: EA‑CSP‑LDA
- Low‑risk candidates: LEA‑CSP‑LDA, LEA‑ROT‑CSP‑LDA
- High‑risk candidates: EA‑FBCSP‑LDA (needs extra gate)
- Parametric family: Chan‑candidates (grid over rank/lambda)

Use colored boxes to indicate “high risk” families and the extra gates attached.

### Fig. 3 — Certificates and delta‑feature stacking
**Layout**: a feature pipeline.
- Inputs: \(p_{a_0}(X)\) and \(p_a(X)\).
- Blocks: entropy/marginal/drift, evidence/probe blocks, keep/coverage blocks.
- Stack: concatenate into \(\phi(a)\); subtract anchor → \(\phi_\Delta(a)\).
- Outputs: ridge prediction \(\widehat{\Delta}\) and guard probability \(\widehat{p}_{pos}\).

### Fig. 4 — Risk‑control gates and fallback
**Layout**: a funnel diagram:
Candidates → guard threshold → anchor‑delta threshold → minPred threshold → family‑specific gate → selected / fallback.

---

## 9. Implementation notes (to keep protocols comparable across datasets)

To avoid reviewer complaints, SAFE‑TTA should be presented as a **single algorithm** with:
- fixed protocol: strict LOSO; no target labels used for adaptation/selection,
- fixed evaluation metrics: mean/worst/neg‑transfer, kappa + statistical tests,
- train‑only calibration for any thresholds (including family‑specific gates) to avoid leakage,
- consistent baselines within each dataset (EA anchor always included).

Dataset‑specific preprocessing differences are acceptable, but must be:
1) fixed per dataset, and  
2) shared across all compared methods in that dataset.

