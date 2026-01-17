# 20260112 — BNCI2014_001 4-class LOSO：扩展/替换下层动作集（TS-SVC / TSA-TS-SVC / FgMDM）验证能否超过 EA

## 0) 上一次结论（failure-first）
在 BNCI2014_001 4-class 严格 LOSO 上，我们此前最好的 `ea-stack-multi-safe-csp-lda` 结果约为 **+1.77pp 且 0 负迁移**（见 `docs/experiments/results_registry.csv` 中 `loso4_actionset_stack_familyblend_k20_probe_minimp001_borda_fix_v1`）。

但在 HGD (`Schirrmeister2017`) 上，候选 `{EA,RPA,TSA}` 基本没有 headroom，selector 常回退 EA（accept≈0），提示：**要想再提升 mean，关键可能不在再加 gate，而在于引入更强的“下层动作/候选族”**。

---

## 1) 本轮科学问题（单杠杆）
**问题**：如果我们把下层候选集从“旧的 CSP-family action set”替换成更接近文献的 Riemannian / tangent-space baselines，能否在同协议下 **超过 EA**，并让 safe selector 有可选的正收益候选？

**本轮唯一主杠杆**：在 `ea-stack-multi-safe-csp-lda` 中把候选族改为：
`{EA(anchor), FBCSP, TS-SVC, TSA-TS-SVC, FgMDM}`  
（即 `--stack-candidate-families ea,fbcsp,ts_svc,tsa_ts_svc,fgmdm`；不再包含 `rpa/tsa(chan-space)/chan` 这些旧 family）

---

## 2) 协议（严格可比）
- Dataset：MOABB `BNCI2014_001`
- Task：4-class（`left_hand,right_hand,feet,tongue`）
- Split：严格 LOSO（cross-subject）
- Preprocess：`--preprocess paper_fir`（8–30 Hz，`tmin=0.5s` 到 `tmax=3.5s`）
- 指标：macro（accuracy/kappa 等）

---

## 3) 运行命令与输出
- 输出：`outputs/20260112/4class/loso4_bnci2014_001_actionset2_ts_tsa_fgmdm_v1/`
- 结果：`outputs/20260112/4class/loso4_bnci2014_001_actionset2_ts_tsa_fgmdm_v1/20260112_results.txt`
- 方法汇总：`outputs/20260112/4class/loso4_bnci2014_001_actionset2_ts_tsa_fgmdm_v1/20260112_method_comparison.csv`

图与主表（从 `*_predictions_all_methods.csv` 生成）：
- `docs/experiments/figures/20260112_loso4_bnci_actionset2_ts_tsa_fgmdm_v1/main_table.md`
- `docs/experiments/figures/20260112_loso4_bnci_actionset2_ts_tsa_fgmdm_v1/20260112_actionset2_scatter_subject_acc_ea-csp-lda_vs_ea-stack-multi-safe-csp-lda.png`
- `docs/experiments/figures/20260112_loso4_bnci_actionset2_ts_tsa_fgmdm_v1/20260112_actionset2_stack_family.png`

---

## 4) 结果摘要（结论先行）
来自 `20260112_method_comparison.csv`：
- `ea-csp-lda`：mean acc **0.5320**
- `ea-stack-multi-safe-csp-lda`（新 action set）：mean acc **0.5320**（Δ=**0.00pp**），`accept_rate=0.0`
- `tsa-ts-svc`：mean acc **0.4892**（明显低于 EA）
- `ts-svc`：mean acc **0.3364**（接近 chance）
- `fgmdm`：mean acc **0.3360**（接近 chance）
- `ea-fbcsp-lda`：mean acc **0.5058**（低于 EA）

**核心现象**：
1) 新加入的 `TS-SVC / FgMDM` 在该协议下表现很差（≈0.336），并不具备作为“强 action”的 headroom。  
2) `TSA-TS-SVC` 虽然在部分被试上较高（例如 S1/S8/S9），但整体 mean 仍显著低于 EA。  
3) 因为候选族整体不强，`ea-stack-multi-safe-csp-lda` 的 ridge/guard 选择最终 **全部回退到 EA**（accept=0），因此 mean 等于 EA。

---

## 5) 失败原因（基于本轮证据）
### 5.1 “动作集 headroom 不足”是主因
这轮并不是 selector/证书误选导致的负迁移；相反，安全选择器几乎拒绝所有非 EA（accept=0），说明：
- 在该协议下这些新 baselines 没有提供稳定的可选正收益；
- 即使没有证书失效，**也选不出能超过 EA 的候选**。

### 5.2 现实含义：不能把“文献里强”直接当成“我们协议里强”
这些方法在文献中通常伴随特定的实现细节（更强的特征/频带、SVC 参数、对齐细节、协方差构造等）。在我们当前的 *paper_fir 8–30 + epoch 0.5–3.5 + LOSO* 设定下，直接加入并不能带来 headroom。

---

## 6) 下一步（仍然一次只动一个杠杆）
本轮证明“替换为 TS-SVC / TSA-TS-SVC / FgMDM 这一套并不能超过 EA”。下一步要想继续提升 mean，有两条更现实的路线：
1) **回到已验证有 headroom 的动作族**（例如 `chan` family），并在其上做更系统的收益-风险曲线（主表+附表）；  
2) 引入真正能提升绝对精度的强基线（例如更强的 FBCSP 设定或深度模型 family），再把 safe selector/证书框架移植过去。

