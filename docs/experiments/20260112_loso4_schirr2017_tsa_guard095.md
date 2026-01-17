# 20260112 — Schirrmeister2017 4-class LOSO：给 TSA 加高风险 gate（thr=0.95）以消除 S9 崩盘

## 0) 背景（failure-first）
在 HGD（MOABB `Schirrmeister2017`）4-class 严格 LOSO 上，我们的 `ea-stack-multi-safe-csp-lda`（候选族 `{ea,rpa,tsa}`）曾出现 **1/14 负迁移**，且完全由 **S9 误选 TSA** 导致（`ea → tsa`，accuracy 从 0.3568 掉到 0.25）。

该现象见上一轮 run：
- 输出：`outputs/20260110/4class/loso4_schirr2017_0train_rs50_ea_stack_multi_safe_rpa_tsa_calib3_v1/`

本轮目标：**只动一个杠杆**，把这类“高风险 family 的偶发误选”用 gate 直接压掉，验证能否把 `neg_transfer_rate` 压到 0。

---

## 1) 协议（严格可比）
- Dataset：MOABB `Schirrmeister2017`（HGD）
- Task：4-class（`left_hand,right_hand,feet,rest`）
- Split：严格 LOSO（cross-subject）
- Preprocess：`--preprocess moabb`（8–30 Hz，`tmin=0.5s` 到 `tmax=3.5s`）
- Resample：`--resample 50`（为避免 multi-safe 组合导致的 OOM；同表内一致）
- Model：CSP(`n_components=4`) + LDA

---

## 2) 本轮唯一改动（one lever）
仅新增一个 high-risk gate（其余参数保持与 20260110 的 stack run 一致）：
- `--stack-safe-tsa-guard-threshold 0.95`

直觉：把 TSA 当作高风险 family，只有在 guard 置信度足够高时才允许被选中；否则回退到 EA anchor。

---

## 3) 运行命令与输出
- 输出：`outputs/20260112/4class/loso4_schirr2017_0train_rs50_ea_stack_multi_safe_rpa_tsa_calib3_tsa_guard095_v1/`
- 结果：`outputs/20260112/4class/loso4_schirr2017_0train_rs50_ea_stack_multi_safe_rpa_tsa_calib3_tsa_guard095_v1/20260112_results.txt`
- 汇总：`outputs/20260112/4class/loso4_schirr2017_0train_rs50_ea_stack_multi_safe_rpa_tsa_calib3_tsa_guard095_v1/20260112_method_comparison.csv`

---

## 4) 关键结果（现象与证据链）

### 4.1 主结论
- 本轮 `ea-stack-multi-safe-csp-lda`：
  - mean acc：**0.5844**
  - worst-subject acc：**0.3568**
  - accept_rate：**0.0**（全部回退 EA）
  - `neg_transfer_rate_vs_ea = 0`（因为所有 subject 的 `stack_multi_improve=0`）

对比上一轮（无 TSA gate）的 stack run：
- mean acc：**0.5768**
- worst-subject acc：**0.25**（S9）
- accept_rate：**0.0714**
- `neg_transfer_rate_vs_ea = 0.0714`（1/14）

**解释**：TSA gate 成功把 “S9 误选 TSA” 这一唯一负迁移 case 关掉，使整体回到 EA anchor（同时不引入新的掉点）。

### 4.2 S9 的“因果证据”（pre-family vs blocked）
上一轮（无 TSA gate）S9 被选 TSA 并崩盘：
- `stack_multi_family=tsa`, `acc_anchor=0.356818`, `acc_selected=0.250000`, `improve=-0.106818`

本轮（TSA gate=0.95）S9：
- `stack_multi_pre_family=tsa`（说明 selector 仍倾向 TSA）
- `stack_multi_tsa_blocked=1`, `stack_multi_tsa_block_reason=guard`
- 最终回退 `stack_multi_family=ea`，且 `acc_anchor=acc_selected=0.356818`

---

## 5) 下一步（保持单杠杆）
在 HGD 上当前瓶颈不是“安全性”（已做到 0 负迁移），而是 “有益候选族稀缺 / 证书有效性不足导致 accept_rate≈0”。
下一轮如果要追提升，需要单独设计/引入更强但可控的候选族（例如更贴近 HGD 的频带/特征族），并在相同协议下复跑验证是否能在保持低负迁移的同时提升 mean。

