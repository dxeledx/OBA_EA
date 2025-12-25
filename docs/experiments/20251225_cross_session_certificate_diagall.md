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

### Run（iwcv_ucb selector, diag-all）

动机：把 IWCV 从点估计升级为带不确定性惩罚的 UCB 证书：
\( \text{IWCV-UCB}(Q)=\widehat{R}(Q)+\kappa\sqrt{\widehat{\mathrm{Var}}(Q)/n_{\text{eff}}(Q)} \)。

```bash
conda run -n eeg python run_csp_lda_cross_session.py \
  --preprocess paper_fir --n-components 6 \
  --events left_hand,right_hand,feet,tongue \
  --train-sessions 0train --test-sessions 1test \
  --methods ea-csp-lda,ea-zo-imr-csp-lda \
  --oea-zo-transform rot_scale \
  --oea-zo-selector iwcv_ucb \
  --oea-zo-iwcv-kappa 1.0 \
  --run-name 4c_fir6_iwcv_ucb_k1_rot_scale_diagall_v1 \
  --diagnose-subjects 1,2,3,4,5,6,7,8,9
```

Candidate 证书有效性分析（label-only）：

```bash
conda run -n eeg python scripts/analyze_candidate_certificates.py \
  --run-dir outputs/20251225/4class/cross_session/4c_fir6_iwcv_ucb_k1_rot_scale_diagall_v1 \
  --method ea-zo-imr-csp-lda
```

关键汇总：
- Selected mean acc (iwcv_ucb, kappa=1): `0.681713`（本次与 iwcv 选择几乎一致）
- `rho_iwcv_ucb_mean ≈ 0.037`（相比 iwcv 略增，但仍很弱）

### Run（dev selector, diag-all）

动机：实现 DEV（ICML 2019）的 control variate 重要性加权证书：
\( \widehat R_{\text{DEV}}=\overline{w\ell} + \eta(\overline w - 1) \)，其中 \(\eta=-\mathrm{Cov}(w\ell,w)/\mathrm{Var}(w)\)。

```bash
conda run -n eeg python run_csp_lda_cross_session.py \
  --preprocess paper_fir --n-components 6 \
  --events left_hand,right_hand,feet,tongue \
  --train-sessions 0train --test-sessions 1test \
  --methods ea-csp-lda,ea-zo-imr-csp-lda \
  --oea-zo-transform rot_scale \
  --oea-zo-selector dev \
  --run-name 4c_fir6_dev_sel_rot_scale_diagall_v1 \
  --diagnose-subjects 1,2,3,4,5,6,7,8,9
```

Candidate 证书有效性分析（label-only）：

```bash
conda run -n eeg python scripts/analyze_candidate_certificates.py \
  --run-dir outputs/20251225/4class/cross_session/4c_fir6_dev_sel_rot_scale_diagall_v1 \
  --method ea-zo-imr-csp-lda
```

关键汇总：
- Selected mean acc (dev selector): `0.681713`（本次与 iwcv 选择几乎一致）
- Oracle gap mean: `0.009645`
- Negative transfer rate: `0.000000`
- `rho_dev_mean ≈ 0.039`（略高于 iwcv，但仍很弱）

### Run（calibrated_ridge selector, diag-all）

动机：把“候选选择”显式当作无标签模型选择问题，在其他被试上用 **label-only** 的 improvement 数据把证书校准成一个回归器：
`(candidate features) -> (expected improvement over identity/EA)`。

```bash
conda run -n eeg python run_csp_lda_cross_session.py \
  --preprocess paper_fir --n-components 6 \
  --events left_hand,right_hand,feet,tongue \
  --train-sessions 0train --test-sessions 1test \
  --methods ea-csp-lda,ea-zo-imr-csp-lda \
  --oea-zo-transform rot_scale \
  --oea-zo-selector calibrated_ridge \
  --oea-zo-calib-ridge-alpha 1.0 \
  --oea-zo-calib-max-subjects 0 \
  --oea-zo-calib-seed 0 \
  --run-name 4c_fir6_calib_ridge_all_rot_scale_diagall_v3 \
  --diagnose-subjects 1,2,3,4,5,6,7,8,9
```

Candidate 证书有效性分析（label-only）：

```bash
conda run -n eeg python scripts/analyze_candidate_certificates.py \
  --run-dir outputs/20251225/4class/cross_session/4c_fir6_calib_ridge_all_rot_scale_diagall_v3 \
  --method ea-zo-imr-csp-lda
```

关键汇总：
- EA mean acc: `0.680170`
- Selected mean acc (calibrated_ridge): `0.684414`
- Oracle mean acc: `0.691358`（headroom ≈ +1.12% abs）
- Oracle gap mean: `0.006944`
- Negative transfer rate: `0.111111`
- `rho_ridge_mean ≈ 0.319`（相比 evidence/probe 更“对齐”，但仍不算强相关）

### Run（calibrated_stack_ridge selector, diag-all）

动机：把候选选择当作“无标签证书有效性/模型选择”问题：在校准阶段对每个 candidate 计算多种证书信号（objective+drift+evidence+probe），再用 ridge 回归拟合“相对 identity 的提升”，用于目标域选择。

```bash
conda run -n eeg python run_csp_lda_cross_session.py \
  --preprocess paper_fir --n-components 6 \
  --events left_hand,right_hand,feet,tongue \
  --train-sessions 0train --test-sessions 1test \
  --methods ea-csp-lda,ea-zo-imr-csp-lda \
  --oea-zo-transform rot_scale \
  --oea-zo-selector calibrated_stack_ridge \
  --oea-zo-calib-ridge-alpha 1.0 \
  --oea-zo-calib-max-subjects 0 --oea-zo-calib-seed 0 \
  --run-name 4c_fir6_calib_stack_ridge_all_rot_scale_diagall_v1 \
  --diagnose-subjects 1,2,3,4,5,6,7,8,9
```

Candidate 证书有效性分析（label-only）：

```bash
conda run -n eeg python scripts/analyze_candidate_certificates.py \
  --run-dir outputs/20251225/4class/cross_session/4c_fir6_calib_stack_ridge_all_rot_scale_diagall_v1 \
  --method ea-zo-imr-csp-lda
```

关键汇总：
- EA mean acc: `0.680170`
- Selected mean acc (calibrated_stack_ridge): `0.687114`
- Oracle mean acc: `0.691358`
- Oracle gap mean: `0.004244`
- Negative transfer rate: `0.000000`

### Seed 稳定性检查（4-class）

说明：`calibrated_stack_ridge` 的主要随机性来自 `--oea-zo-seed`（影响候选生成/证书 probe 计算/校准阶段的 ZO 轨迹）。

固定参数同上，仅改变 `--oea-zo-seed`，不输出 diagnostics（更快）。

seed=1：

```bash
conda run -n eeg python run_csp_lda_cross_session.py \
  --preprocess paper_fir --n-components 6 \
  --events left_hand,right_hand,feet,tongue \
  --train-sessions 0train --test-sessions 1test \
  --methods ea-csp-lda,ea-zo-imr-csp-lda \
  --oea-zo-transform rot_scale \
  --oea-zo-selector calibrated_stack_ridge \
  --oea-zo-calib-ridge-alpha 1.0 \
  --oea-zo-calib-max-subjects 0 --oea-zo-calib-seed 0 \
  --oea-zo-seed 1 \
  --run-name 4c_fir6_stack_ridge_seed1
```

- Results: `outputs/20251225/4class/cross_session/4c_fir6_stack_ridge_seed1/20251225_results.txt`

seed=2：

```bash
conda run -n eeg python run_csp_lda_cross_session.py \
  --preprocess paper_fir --n-components 6 \
  --events left_hand,right_hand,feet,tongue \
  --train-sessions 0train --test-sessions 1test \
  --methods ea-csp-lda,ea-zo-imr-csp-lda \
  --oea-zo-transform rot_scale \
  --oea-zo-selector calibrated_stack_ridge \
  --oea-zo-calib-ridge-alpha 1.0 \
  --oea-zo-calib-max-subjects 0 --oea-zo-calib-seed 0 \
  --oea-zo-seed 2 \
  --run-name 4c_fir6_stack_ridge_seed2
```

- Results: `outputs/20251225/4class/cross_session/4c_fir6_stack_ridge_seed2/20251225_results.txt`

seed=3：

```bash
conda run -n eeg python run_csp_lda_cross_session.py \
  --preprocess paper_fir --n-components 6 \
  --events left_hand,right_hand,feet,tongue \
  --train-sessions 0train --test-sessions 1test \
  --methods ea-csp-lda,ea-zo-imr-csp-lda \
  --oea-zo-transform rot_scale \
  --oea-zo-selector calibrated_stack_ridge \
  --oea-zo-calib-ridge-alpha 1.0 \
  --oea-zo-calib-max-subjects 0 --oea-zo-calib-seed 0 \
  --oea-zo-seed 3 \
  --run-name 4c_fir6_stack_ridge_seed3
```

- Results: `outputs/20251225/4class/cross_session/4c_fir6_stack_ridge_seed3/20251225_results.txt`

seed=4：

```bash
conda run -n eeg python run_csp_lda_cross_session.py \
  --preprocess paper_fir --n-components 6 \
  --events left_hand,right_hand,feet,tongue \
  --train-sessions 0train --test-sessions 1test \
  --methods ea-csp-lda,ea-zo-imr-csp-lda \
  --oea-zo-transform rot_scale \
  --oea-zo-selector calibrated_stack_ridge \
  --oea-zo-calib-ridge-alpha 1.0 \
  --oea-zo-calib-max-subjects 0 --oea-zo-calib-seed 0 \
  --oea-zo-seed 4 \
  --run-name 4c_fir6_stack_ridge_seed4
```

- Results: `outputs/20251225/4class/cross_session/4c_fir6_stack_ridge_seed4/20251225_results.txt`

汇总（mean acc across subjects）：

| oea_zo_seed | Selected mean acc | Δ vs EA mean | #neg subjects |
|---:|---:|---:|---:|
| 0 | 0.687114 | +0.006944 | 0 |
| 1 | 0.685957 | +0.005787 | 0 |
| 2 | 0.684799 | +0.004630 | 2 |
| 3 | 0.684028 | +0.003858 | 1 |
| 4 | 0.685571 | +0.005401 | 1 |

Across seeds（0–4）：`mean=0.685494`, `std=0.001170`（对应 Δ vs EA：`mean=0.005324`, `std=0.001170`）。

注意：平均提升相对稳定，但仍存在少数 seed 下个别 subject 负迁移（总计 4/45 的 subject×seed 为负迁移），说明证书仍未“完全可靠”。

### 下层切换：RPA-center / TSA（4-class）

动机：把“下层对齐”从 EA（Euclidean mean whitening）切到更几何的强基线：

- `rpa-*`：log-Euclidean mean whitening（RPA-center）
- `tsa-*`：RPA-center + pseudo-label Procrustes rotation（TSA 风格闭式旋转）

然后把现有的证书 stacking（`calibrated_stack_ridge`）原样移植到该下层上，观察：

1) headroom（oracle mean - identity mean）是否变大  
2) 证书有效性（certificate-acc Spearman、oracle gap、负迁移率）是否改善

#### Baselines（无 ZO）

```bash
conda run -n eeg python run_csp_lda_cross_session.py \
  --preprocess paper_fir --n-components 6 \
  --events left_hand,right_hand,feet,tongue \
  --train-sessions 0train --test-sessions 1test \
  --methods ea-csp-lda,rpa-csp-lda,tsa-csp-lda \
  --run-name 4c_fir6_rpa_tsa_baselines_v1
```

- Results: `outputs/20251225/4class/cross_session/4c_fir6_rpa_tsa_baselines_v1/20251225_results.txt`
- Mean acc：EA `0.680170`，RPA `0.682870`（+0.27% abs），TSA `0.681713`（+0.15% abs）

#### EA-ZO（stacking，作为对照）

```bash
conda run -n eeg python run_csp_lda_cross_session.py \
  --preprocess paper_fir --n-components 6 \
  --events left_hand,right_hand,feet,tongue \
  --train-sessions 0train --test-sessions 1test \
  --methods ea-csp-lda,ea-zo-imr-csp-lda \
  --oea-zo-transform rot_scale --oea-zo-selector calibrated_stack_ridge \
  --oea-zo-calib-ridge-alpha 1.0 --oea-zo-calib-max-subjects 0 --oea-zo-calib-seed 0 \
  --run-name 4c_fir6_ea_zo_stack_diagall_v2 \
  --diagnose-subjects 1,2,3,4,5,6,7,8,9
```

```bash
conda run -n eeg python scripts/analyze_candidate_certificates.py \
  --run-dir outputs/20251225/4class/cross_session/4c_fir6_ea_zo_stack_diagall_v2 \
  --method ea-zo-imr-csp-lda
```

关键汇总：

- Selected mean acc: `0.687114`
- Identity mean acc: `0.680170`
- Oracle mean acc: `0.691358`
- Oracle gap mean: `0.004244`
- Negative transfer rate: `0.000000`
- `rho_ridge_mean ≈ 0.293459`

#### RPA-ZO（stacking）

```bash
conda run -n eeg python run_csp_lda_cross_session.py \
  --preprocess paper_fir --n-components 6 \
  --events left_hand,right_hand,feet,tongue \
  --train-sessions 0train --test-sessions 1test \
  --methods rpa-csp-lda,rpa-zo-imr-csp-lda \
  --oea-zo-transform rot_scale --oea-zo-selector calibrated_stack_ridge \
  --oea-zo-calib-ridge-alpha 1.0 --oea-zo-calib-max-subjects 0 --oea-zo-calib-seed 0 \
  --run-name 4c_fir6_rpa_zo_stack_diagall_v1 \
  --diagnose-subjects 1,2,3,4,5,6,7,8,9
```

```bash
conda run -n eeg python scripts/analyze_candidate_certificates.py \
  --run-dir outputs/20251225/4class/cross_session/4c_fir6_rpa_zo_stack_diagall_v1 \
  --method rpa-zo-imr-csp-lda
```

关键汇总：

- Selected mean acc: `0.681327`（< identity `0.682870`）
- Oracle mean acc: `0.689043`
- Oracle gap mean: `0.007716`
- Negative transfer rate: `0.222222`
- `rho_ridge_mean ≈ 0.201658`

结论：RPA-center 作为 baseline 有小幅提升，但在当前候选集/证书设定下，stacking 更容易选错（证书有效性反而变差）。

#### TSA-ZO（stacking）

```bash
conda run -n eeg python run_csp_lda_cross_session.py \
  --preprocess paper_fir --n-components 6 \
  --events left_hand,right_hand,feet,tongue \
  --train-sessions 0train --test-sessions 1test \
  --methods tsa-csp-lda,tsa-zo-imr-csp-lda \
  --oea-zo-transform rot_scale --oea-zo-selector calibrated_stack_ridge \
  --oea-zo-calib-ridge-alpha 1.0 --oea-zo-calib-max-subjects 0 --oea-zo-calib-seed 0 \
  --run-name 4c_fir6_tsa_zo_stack_diagall_v1 \
  --diagnose-subjects 1,2,3,4,5,6,7,8,9
```

```bash
conda run -n eeg python scripts/analyze_candidate_certificates.py \
  --run-dir outputs/20251225/4class/cross_session/4c_fir6_tsa_zo_stack_diagall_v1 \
  --method tsa-zo-imr-csp-lda
```

关键汇总：

- Selected mean acc: `0.683256`
- Identity mean acc: `0.681713`
- Oracle mean acc: `0.686343`
- Oracle gap mean: `0.003086`（gap 变小）
- Negative transfer rate: `0.111111`
- `rho_ridge_mean ≈ 0.329794`（Spearman 变好）

结论：TSA-base 在当前实现下更像“把 headroom 吃掉了”（oracle-headroom 变小），证书相关性略改善，但最终平均准确率仍低于 EA-ZO 的最佳结果。

### Run（calibrated_guard selector, diag-all）

动机：学习一个二分类守门员 `P(improve ≥ margin | features)`，先拒绝更可能负迁移的 candidates，再在保留集合里按 objective/score 选最优（identity 总是允许）。

```bash
conda run -n eeg python run_csp_lda_cross_session.py \
  --preprocess paper_fir --n-components 6 \
  --events left_hand,right_hand,feet,tongue \
  --train-sessions 0train --test-sessions 1test \
  --methods ea-csp-lda,ea-zo-imr-csp-lda \
  --oea-zo-transform rot_scale \
  --oea-zo-selector calibrated_guard \
  --oea-zo-calib-guard-c 1.0 \
  --oea-zo-calib-guard-threshold 0.5 \
  --oea-zo-calib-guard-margin 0.0 \
  --oea-zo-calib-max-subjects 0 \
  --oea-zo-calib-seed 0 \
  --run-name 4c_fir6_calib_guard_all_rot_scale_diagall_v3 \
  --diagnose-subjects 1,2,3,4,5,6,7,8,9
```

Candidate 证书有效性分析（label-only）：

```bash
conda run -n eeg python scripts/analyze_candidate_certificates.py \
  --run-dir outputs/20251225/4class/cross_session/4c_fir6_calib_guard_all_rot_scale_diagall_v3 \
  --method ea-zo-imr-csp-lda
```

关键汇总：
- EA mean acc: `0.680170`
- Selected mean acc (calibrated_guard): `0.685185`
- Oracle mean acc: `0.691358`
- Oracle gap mean: `0.006173`
- Negative transfer rate: `0.222222`
- `rho_guard_mean ≈ 0.338`（guard 概率与 acc 的相关性比 ridge 略高，但更容易出现少数被试掉点）

### Run（calibrated_ridge_guard selector, diag-all）

动机：把 `calibrated_ridge` 和 `calibrated_guard` 组合成一个更“安全”的选择器：
先用 guard 过滤，再在保留集合里按 ridge 预测提升选最优（若预测提升≤0 则回退 identity）。

```bash
conda run -n eeg python run_csp_lda_cross_session.py \
  --preprocess paper_fir --n-components 6 \
  --events left_hand,right_hand,feet,tongue \
  --train-sessions 0train --test-sessions 1test \
  --methods ea-csp-lda,ea-zo-imr-csp-lda \
  --oea-zo-transform rot_scale \
  --oea-zo-selector calibrated_ridge_guard \
  --oea-zo-calib-ridge-alpha 1.0 \
  --oea-zo-calib-guard-c 1.0 --oea-zo-calib-guard-threshold 0.5 --oea-zo-calib-guard-margin 0.0 \
  --oea-zo-calib-max-subjects 0 --oea-zo-calib-seed 0 \
  --run-name 4c_fir6_calib_ridge_guard_all_rot_scale_diagall_v1 \
  --diagnose-subjects 1,2,3,4,5,6,7,8,9
```

关键汇总（本次实现下）：`calibrated_ridge_guard` 与 `calibrated_ridge` 基本等价（Selected mean acc `0.684414`，Neg transfer `0.111111`），未带来额外收益。

### 4-class 对比表（可复现）

固定：EA mean `0.680170`，Oracle mean `0.691358`（headroom ≈ `+0.011188`）。

| Selector | Selected mean acc | Oracle gap mean | Neg transfer rate | Certificate Spearman mean |
|---|---:|---:|---:|---:|
| `objective` | 0.676697 | 0.014661 | 0.555556 | `rho_score_mean=0.299418` |
| `evidence` | 0.680556 | 0.010802 | 0.000000 | `rho_ev_mean=-0.075179` |
| `dev` | 0.681713 | 0.009645 | 0.000000 | `rho_dev_mean=0.038799` |
| `iwcv` | 0.681713 | 0.009645 | 0.000000 | `rho_iwcv_mean=0.033065` |
| `iwcv_ucb (k=1.0)` | 0.681713 | 0.009645 | 0.000000 | `rho_iwcv_ucb_mean=0.037097` |
| `calibrated_ridge` | 0.684414 | 0.006944 | 0.111111 | `rho_ridge_mean=0.319355` |
| `calibrated_guard` | 0.685185 | 0.006173 | 0.222222 | `rho_guard_mean=0.337858` |
| `calibrated_ridge_guard` | 0.684414 | 0.006944 | 0.111111 | `rho_ridge_mean=0.319355` |
| `calibrated_stack_ridge` | 0.687114 | 0.004244 | 0.000000 | `rho_ridge_mean=0.293459` |

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

### Run（iwcv_ucb selector, diag-all）

```bash
conda run -n eeg python run_csp_lda_cross_session.py \
  --preprocess paper_fir --n-components 6 \
  --events left_hand,right_hand \
  --train-sessions 0train --test-sessions 1test \
  --methods ea-csp-lda,ea-zo-imr-csp-lda \
  --oea-zo-transform rot_scale \
  --oea-zo-selector iwcv_ucb \
  --oea-zo-iwcv-kappa 1.0 \
  --run-name 2c_fir6_iwcv_ucb_k1_rot_scale_diagall_v1 \
  --diagnose-subjects 1,2,3,4,5,6,7,8,9
```

关键汇总：
- Selected mean acc: `0.807099`（略低于 iwcv `0.808642`）
- `rho_iwcv_ucb_mean ≈ 0.033`（比 iwcv 的 `0.007` 大，但仍不强）

### Run（dev selector, diag-all）

```bash
conda run -n eeg python run_csp_lda_cross_session.py \
  --preprocess paper_fir --n-components 6 \
  --events left_hand,right_hand \
  --train-sessions 0train --test-sessions 1test \
  --methods ea-csp-lda,ea-zo-imr-csp-lda \
  --oea-zo-transform rot_scale \
  --oea-zo-selector dev \
  --run-name 2c_fir6_dev_sel_rot_scale_diagall_v1 \
  --diagnose-subjects 1,2,3,4,5,6,7,8,9
```

Candidate 证书有效性分析（label-only）：

```bash
conda run -n eeg python scripts/analyze_candidate_certificates.py \
  --run-dir outputs/20251225/2class/cross_session/2c_fir6_dev_sel_rot_scale_diagall_v1 \
  --method ea-zo-imr-csp-lda
```

关键汇总：
- EA mean acc: `0.803241`
- Selected mean acc (dev selector): `0.808642`（本次与 iwcv 选择几乎一致）
- Oracle mean acc: `0.814815`
- Oracle gap mean: `0.006173`
- Negative transfer rate: `0.111111`
- `rho_dev_mean ≈ 0.013`（仍接近 0）

### Run（calibrated_ridge selector, diag-all）

```bash
conda run -n eeg python run_csp_lda_cross_session.py \
  --preprocess paper_fir --n-components 6 \
  --events left_hand,right_hand \
  --train-sessions 0train --test-sessions 1test \
  --methods ea-csp-lda,ea-zo-imr-csp-lda \
  --oea-zo-transform rot_scale \
  --oea-zo-selector calibrated_ridge \
  --oea-zo-calib-ridge-alpha 1.0 \
  --oea-zo-calib-max-subjects 0 \
  --oea-zo-calib-seed 0 \
  --run-name 2c_fir6_calib_ridge_all_rot_scale_diagall_v3 \
  --diagnose-subjects 1,2,3,4,5,6,7,8,9
```

关键汇总：
- EA mean acc: `0.803241`
- Selected mean acc: `0.799383`（更差）
- `rho_ridge_mean ≈ 0.023`（几乎无相关性）

### Run（calibrated_guard selector, diag-all）

```bash
conda run -n eeg python run_csp_lda_cross_session.py \
  --preprocess paper_fir --n-components 6 \
  --events left_hand,right_hand \
  --train-sessions 0train --test-sessions 1test \
  --methods ea-csp-lda,ea-zo-imr-csp-lda \
  --oea-zo-transform rot_scale \
  --oea-zo-selector calibrated_guard \
  --oea-zo-calib-guard-c 1.0 \
  --oea-zo-calib-guard-threshold 0.5 \
  --oea-zo-calib-guard-margin 0.0 \
  --oea-zo-calib-max-subjects 0 \
  --oea-zo-calib-seed 0 \
  --run-name 2c_fir6_calib_guard_all_rot_scale_diagall_v3 \
  --diagnose-subjects 1,2,3,4,5,6,7,8,9
```

关键汇总：
- EA mean acc: `0.803241`
- Selected mean acc: `0.803241`（几乎等同 EA）
- `rho_guard_mean ≈ -0.291`（相关性为负，说明该 guard 在 2 类下并不可靠）

### Run（calibrated_ridge_guard selector, diag-all）

```bash
conda run -n eeg python run_csp_lda_cross_session.py \
  --preprocess paper_fir --n-components 6 \
  --events left_hand,right_hand \
  --train-sessions 0train --test-sessions 1test \
  --methods ea-csp-lda,ea-zo-imr-csp-lda \
  --oea-zo-transform rot_scale \
  --oea-zo-selector calibrated_ridge_guard \
  --oea-zo-calib-ridge-alpha 1.0 \
  --oea-zo-calib-guard-c 1.0 --oea-zo-calib-guard-threshold 0.5 --oea-zo-calib-guard-margin 0.0 \
  --oea-zo-calib-max-subjects 0 --oea-zo-calib-seed 0 \
  --run-name 2c_fir6_calib_ridge_guard_all_rot_scale_diagall_v1 \
  --diagnose-subjects 1,2,3,4,5,6,7,8,9
```

结论：2 类下该组合选择器更差（Selected mean acc `0.802469` < EA `0.803241`，Neg transfer `0.222222`），不推荐用于 2 类主结果。

### 2-class 对比表（可复现）

固定：EA mean `0.803241`，Oracle mean `0.814815`（headroom ≈ `+0.011574`）。

| Selector | Selected mean acc | Oracle gap mean | Neg transfer rate | Certificate Spearman mean |
|---|---:|---:|---:|---:|
| `evidence` | 0.801698 | 0.013117 | 0.333333 | `rho_ev_mean=-0.542966` |
| `dev` | 0.808642 | 0.006173 | 0.111111 | `rho_dev_mean=0.013306` |
| `iwcv` | 0.808642 | 0.006173 | 0.111111 | `rho_iwcv_mean=0.007034` |
| `iwcv_ucb (k=1.0)` | 0.807099 | 0.007716 | 0.111111 | `rho_iwcv_ucb_mean=0.033423` |
| `calibrated_ridge` | 0.799383 | 0.015432 | 0.222222 | `rho_ridge_mean=0.023163` |
| `calibrated_guard` | 0.803241 | 0.011574 | 0.222222 | `rho_guard_mean=-0.290860` |
| `calibrated_ridge_guard` | 0.802469 | 0.012346 | 0.222222 | `rho_ridge_mean=0.023163` |
