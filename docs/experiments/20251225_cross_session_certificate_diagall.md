# 2025-12-25 — Cross-session (0train→1test) certificate diagnostics (diag-all)

本记录用于回答一个非常具体的问题：

> 在 within-subject cross-session（session shift）下，candidate-set 上的无标签证书（objective / evidence / probe）是否能更稳定地选到更好的变换？

## 设置

- Dataset: MOABB `BNCI2014_001` (BCI IV 2a)
- Protocol: within-subject cross-session, `0train → 1test`
- Preprocess: `paper_fir`（causal FIR, order=50）
- Model: `EA → CSP(n=6) → LDA`
- ZO transform family: `rot_scale`（`A = diag(exp(s)) · Q`）
- 目标域优化：`ea-zo-imr-csp-lda`（objective=`infomax_bilevel`，ZO 默认超参）
- Diagnostics：对全部 subject 生成 `diagnostics/.../subject_XX/candidates.csv`

## 4-class（left/right/feet/tongue）

### Run（evidence selector, diag-all）

```bash
conda run -n eeg python run_csp_lda_cross_session.py \
  --preprocess paper_fir --n-components 6 \
  --events left_hand,right_hand,feet,tongue \
  --train-sessions 0train --test-sessions 1test \
  --methods ea-csp-lda,ea-zo-imr-csp-lda \
  --oea-zo-transform rot_scale \
  --oea-zo-selector evidence \
  --run-name 4c_fir6_evidence_sel_rot_scale_diagall_v2 \
  --diagnose-subjects 1,2,3,4,5,6,7,8,9
```

- Results: `outputs/20251225/4class/cross_session/4c_fir6_evidence_sel_rot_scale_diagall_v2/20251225_results.txt`
- Diagnostics: `outputs/20251225/4class/cross_session/4c_fir6_evidence_sel_rot_scale_diagall_v2/diagnostics/ea-zo-imr-csp-lda/subject_XX/candidates.csv`

### Candidate 证书有效性分析（label-only analysis）

```bash
conda run -n eeg python scripts/analyze_candidate_certificates.py \
  --run-dir outputs/20251225/4class/cross_session/4c_fir6_evidence_sel_rot_scale_diagall_v2 \
  --method ea-zo-imr-csp-lda
```

关键汇总（mean over 9 subjects）：
- EA mean acc: `0.680170`
- Selected mean acc (evidence selector): `0.680556`
- Oracle mean acc (max over candidates, label-only): `0.691358`（headroom ≈ +1.12% abs）
- Oracle gap（oracle - selected）mean: `0.010802`
- Negative transfer rate（selected < identity/EA）: `0.444444`
- Spearman mean（within-candidate, per-subject 再平均）：
  - `rho_score_mean ≈ 0.299`（较弱）
  - `rho_ev_mean ≈ -0.075`（接近 0 且偏负）
  - `rho_probe_mean ≈ 0.040`（接近 0）

解释：rot+scale 的 candidate-set 里确实存在提升空间（oracle>EA），但当前无标签证书仍难以可靠选到这些 oracle candidates；提升更多来自“更保守地选回 identity”的安全化，而非真正学会选到更优变换。

### Run（probe_mixup_hard selector, diag-all）

动机：借鉴 MixVal 的一个常见做法——当 mix 系数 λ>0.5 时，用 dominant 样本的 **hard pseudo label** 来构造 probe（我们在 probe 内部做了 λ folding，并固定 Beta(0.4,0.4) 采样）。

```bash
conda run -n eeg python run_csp_lda_cross_session.py \
  --preprocess paper_fir --n-components 6 \
  --events left_hand,right_hand,feet,tongue \
  --train-sessions 0train --test-sessions 1test \
  --methods ea-csp-lda,ea-zo-imr-csp-lda \
  --oea-zo-transform rot_scale \
  --oea-zo-selector probe_mixup_hard \
  --run-name 4c_fir6_probe_mixup_hard_sel_rot_scale_diagall \
  --diagnose-subjects 1,2,3,4,5,6,7,8,9
```

- Results: `outputs/20251225/4class/cross_session/4c_fir6_probe_mixup_hard_sel_rot_scale_diagall/20251225_results.txt`

结论（本次实现下）：`probe_mixup_hard` 效果更差（selected mean acc `0.676698` < EA `0.680170`），且 `rho_probe_hard_mean≈0.023`（几乎无相关性）。在当前 CSP 特征空间的 mix probe 设定里，“hard-major”并没有改善证书失效，反而更容易被伪标签噪声牵引。

### Run（iwcv selector, diag-all）

动机：把“证书”从纯无标签 surrogate（entropy/IM/evidence/probe）切到一个 **risk-consistent** 的替代：
用有标签源域（train session）上的 NLL 作为风险估计，并用目标域无标签分布做 **covariate-shift importance weighting**（domain classifier density ratio）。

```bash
conda run -n eeg python run_csp_lda_cross_session.py \
  --preprocess paper_fir --n-components 6 \
  --events left_hand,right_hand,feet,tongue \
  --train-sessions 0train --test-sessions 1test \
  --methods ea-csp-lda,ea-zo-imr-csp-lda \
  --oea-zo-transform rot_scale \
  --oea-zo-selector iwcv \
  --run-name 4c_fir6_iwcv_sel_rot_scale_diagall_v2 \
  --diagnose-subjects 1,2,3,4,5,6,7,8,9
```

- Results: `outputs/20251225/4class/cross_session/4c_fir6_iwcv_sel_rot_scale_diagall_v2/20251225_results.txt`
- Diagnostics: `outputs/20251225/4class/cross_session/4c_fir6_iwcv_sel_rot_scale_diagall_v2/diagnostics/ea-zo-imr-csp-lda/subject_XX/candidates.csv`

Candidate 证书有效性分析（label-only）：

```bash
conda run -n eeg python scripts/analyze_candidate_certificates.py \
  --run-dir outputs/20251225/4class/cross_session/4c_fir6_iwcv_sel_rot_scale_diagall_v2 \
  --method ea-zo-imr-csp-lda
```

关键汇总（mean over 9 subjects）：
- EA mean acc: `0.680170`
- Selected mean acc (iwcv selector): `0.681713`
- Oracle mean acc: `0.691358`（headroom ≈ +1.12% abs）
- Oracle gap（oracle - selected）mean: `0.009645`
- Negative transfer rate（selected < identity/EA）: `0.000000`
- Spearman mean（within-candidate）：
  - `rho_ev_mean ≈ -0.075`（evidence 仍偏负）
  - `rho_probe_mean ≈ 0.040`（probe 仍接近 0）
  - `rho_iwcv_mean ≈ 0.033`（IWCV 对“排序”仍弱，但更像一个 safety gate：能稳定避免掉点）

## 2-class（left vs right）

### Run（evidence selector, diag-all）

```bash
conda run -n eeg python run_csp_lda_cross_session.py \
  --preprocess paper_fir --n-components 6 \
  --events left_hand,right_hand \
  --train-sessions 0train --test-sessions 1test \
  --methods ea-csp-lda,ea-zo-imr-csp-lda \
  --oea-zo-transform rot_scale \
  --oea-zo-selector evidence \
  --run-name 2c_fir6_evidence_sel_rot_scale_diagall_v2 \
  --diagnose-subjects 1,2,3,4,5,6,7,8,9
```

- Results: `outputs/20251225/2class/cross_session/2c_fir6_evidence_sel_rot_scale_diagall_v2/20251225_results.txt`

### Candidate 证书有效性分析（label-only analysis）

```bash
conda run -n eeg python scripts/analyze_candidate_certificates.py \
  --run-dir outputs/20251225/2class/cross_session/2c_fir6_evidence_sel_rot_scale_diagall_v2 \
  --method ea-zo-imr-csp-lda
```

关键汇总：
- EA mean acc: `0.803241`
- Selected mean acc (evidence selector): `0.801698`（略差）
- Oracle mean acc: `0.814815`（headroom ≈ +1.16% abs）
- Spearman mean：
  - `rho_score_mean ≈ 0.548`（明显更正相关）
  - `rho_ev_mean ≈ -0.543`（明显负相关）
  - `rho_probe_mean ≈ -0.037`（接近 0）

解释：2 类下 objective(score) 与真实 acc 的相关性更强，但 evidence 反而显著失效；rot+scale 的 headroom 存在，关键仍是“如何选到好 candidate”。

### Run（iwcv selector, diag-all）

```bash
conda run -n eeg python run_csp_lda_cross_session.py \
  --preprocess paper_fir --n-components 6 \
  --events left_hand,right_hand \
  --train-sessions 0train --test-sessions 1test \
  --methods ea-csp-lda,ea-zo-imr-csp-lda \
  --oea-zo-transform rot_scale \
  --oea-zo-selector iwcv \
  --run-name 2c_fir6_iwcv_sel_rot_scale_diagall_v2 \
  --diagnose-subjects 1,2,3,4,5,6,7,8,9
```

Candidate 证书有效性分析（label-only）：

```bash
conda run -n eeg python scripts/analyze_candidate_certificates.py \
  --run-dir outputs/20251225/2class/cross_session/2c_fir6_iwcv_sel_rot_scale_diagall_v2 \
  --method ea-zo-imr-csp-lda
```

关键汇总：
- EA mean acc: `0.803241`
- Selected mean acc (iwcv selector): `0.808642`
- Oracle mean acc: `0.814815`（headroom ≈ +1.16% abs）
- Oracle gap mean: `0.006173`
- Negative transfer rate: `0.111111`
- Spearman mean：`rho_iwcv_mean ≈ 0.007`（仍接近 0；收益更像来自“避免选到坏 candidate”而不是“精确挑到 oracle”）
