from __future__ import annotations

import argparse
from pathlib import Path
import re

import numpy as np
import pandas as pd


def _rankdata(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(len(x), dtype=float)
    return ranks


def _spearman_pos(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman correlation between x and y (no tie correction; diagnostic use)."""

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if x.size < 2:
        return float("nan")
    rx = _rankdata(x)
    ry = _rankdata(y)
    return float(np.corrcoef(rx, ry)[0, 1])


def _load_method_acc(results_txt: Path, method: str) -> pd.Series:
    txt = results_txt.read_text(encoding="utf-8")
    pat = re.compile(rf"=== Method: {re.escape(method)} ===\n(.*?)\n\nSummary", re.S)
    m = pat.search(txt)
    if not m:
        raise RuntimeError(f"Method block not found: {method} in {results_txt}")
    block = m.group(1).strip().splitlines()
    header = block[0].split()
    rows = []
    for line in block[1:]:
        line = line.strip()
        if not line:
            continue
        rows.append(line.split())
    df = pd.DataFrame(rows, columns=header)
    df["subject"] = df["subject"].astype(int)
    df["accuracy"] = df["accuracy"].astype(float)
    return df.set_index("subject")["accuracy"]


def analyze_run(*, run_dir: Path, method: str) -> tuple[pd.DataFrame, dict]:
    run_dir = Path(run_dir)
    results_txts = sorted(run_dir.glob("*_results.txt"))
    if not results_txts:
        raise RuntimeError(f"No *_results.txt found in {run_dir}")
    results_txt = results_txts[0]
    method_acc = _load_method_acc(results_txt, method=method)

    diag_root = run_dir / "diagnostics" / method
    cand_paths = sorted(diag_root.glob("subject_*/candidates.csv"))
    if not cand_paths:
        raise RuntimeError(f"No candidates.csv under {diag_root}")

    rows = []
    for cand_path in cand_paths:
        m = re.search(r"subject_(\d+)", str(cand_path))
        if not m:
            continue
        subj = int(m.group(1))
        df = pd.read_csv(cand_path)

        id_df = df[df["kind"] == "identity"]
        id_acc = float(id_df["accuracy"].iloc[0]) if len(id_df) else float("nan")
        oracle_acc = float(df["accuracy"].max())

        score = df["score"].astype(float)
        best_score_acc = float(df.loc[score.idxmin(), "accuracy"])

        best_ev_acc = float("nan")
        if "evidence_nll_best" in df.columns:
            ev = df["evidence_nll_best"].astype(float)
            ev_mask = np.isfinite(ev.to_numpy())
            if ev_mask.any():
                best_ev_acc = float(df.loc[ev[ev_mask].idxmin(), "accuracy"])

        best_pm_acc = float("nan")
        if "probe_mixup_best" in df.columns:
            pm = df["probe_mixup_best"].astype(float)
            pm_mask = np.isfinite(pm.to_numpy())
            if pm_mask.any():
                best_pm_acc = float(df.loc[pm[pm_mask].idxmin(), "accuracy"])

        best_pmh_acc = float("nan")
        if "probe_mixup_hard_best" in df.columns:
            pmh = df["probe_mixup_hard_best"].astype(float)
            pmh_mask = np.isfinite(pmh.to_numpy())
            if pmh_mask.any():
                best_pmh_acc = float(df.loc[pmh[pmh_mask].idxmin(), "accuracy"])

        best_iwcv_acc = float("nan")
        if "iwcv_nll" in df.columns:
            iw = df["iwcv_nll"].astype(float)
            iw_mask = np.isfinite(iw.to_numpy())
            if iw_mask.any():
                best_iwcv_acc = float(df.loc[iw[iw_mask].idxmin(), "accuracy"])

        best_ridge_acc = float("nan")
        if "ridge_pred_improve" in df.columns:
            rp = df["ridge_pred_improve"].astype(float)
            rp_mask = np.isfinite(rp.to_numpy())
            if rp_mask.any():
                best_ridge_acc = float(df.loc[rp[rp_mask].idxmax(), "accuracy"])

        best_guard_acc = float("nan")
        if "guard_p_pos" in df.columns:
            gp = df["guard_p_pos"].astype(float)
            gp_mask = np.isfinite(gp.to_numpy())
            if gp_mask.any():
                best_guard_acc = float(df.loc[gp[gp_mask].idxmax(), "accuracy"])

        acc = df["accuracy"].astype(float).to_numpy()
        rho_score = _spearman_pos(-score.to_numpy(), acc)

        rho_ev = float("nan")
        if "evidence_nll_best" in df.columns:
            ev = df["evidence_nll_best"].astype(float).to_numpy()
            rho_ev = _spearman_pos(-ev, acc)

        rho_pm = float("nan")
        if "probe_mixup_best" in df.columns:
            pm = df["probe_mixup_best"].astype(float).to_numpy()
            rho_pm = _spearman_pos(-pm, acc)

        rho_pmh = float("nan")
        if "probe_mixup_hard_best" in df.columns:
            pmh = df["probe_mixup_hard_best"].astype(float).to_numpy()
            rho_pmh = _spearman_pos(-pmh, acc)

        rho_iwcv = float("nan")
        if "iwcv_nll" in df.columns:
            iw = df["iwcv_nll"].astype(float).to_numpy()
            rho_iwcv = _spearman_pos(-iw, acc)

        rho_ridge = float("nan")
        if "ridge_pred_improve" in df.columns:
            rp = df["ridge_pred_improve"].astype(float).to_numpy()
            rho_ridge = _spearman_pos(rp, acc)

        rho_guard = float("nan")
        if "guard_p_pos" in df.columns:
            gp = df["guard_p_pos"].astype(float).to_numpy()
            rho_guard = _spearman_pos(gp, acc)

        sel_acc = float(method_acc.get(subj, float("nan")))
        # Results in *_results.txt are printed with limited decimals, so use a small tolerance.
        tol = 1e-6
        rows.append(
            {
                "subject": subj,
                "id_acc": id_acc,
                "sel_acc": sel_acc,
                "oracle_acc": oracle_acc,
                "gap_sel": oracle_acc - sel_acc,
                "best_score_acc": best_score_acc,
                "gap_score": oracle_acc - best_score_acc,
                "best_ev_acc": best_ev_acc,
                "gap_ev": oracle_acc - best_ev_acc,
                "best_probe_acc": best_pm_acc,
                "gap_probe": oracle_acc - best_pm_acc,
                "best_probe_hard_acc": best_pmh_acc,
                "gap_probe_hard": oracle_acc - best_pmh_acc,
                "best_iwcv_acc": best_iwcv_acc,
                "gap_iwcv": oracle_acc - best_iwcv_acc,
                "best_ridge_acc": best_ridge_acc,
                "gap_ridge": oracle_acc - best_ridge_acc,
                "best_guard_acc": best_guard_acc,
                "gap_guard": oracle_acc - best_guard_acc,
                "rho_score": rho_score,
                "rho_ev": rho_ev,
                "rho_probe": rho_pm,
                "rho_probe_hard": rho_pmh,
                "rho_iwcv": rho_iwcv,
                "rho_ridge": rho_ridge,
                "rho_guard": rho_guard,
                "neg_transfer": float((sel_acc + tol) < id_acc),
            }
        )

    table = pd.DataFrame(rows).sort_values("subject")
    summary = {
        "run_dir": str(run_dir),
        "method": str(method),
        "sel_mean": float(table["sel_acc"].mean()),
        "id_mean": float(table["id_acc"].mean()),
        "oracle_mean": float(table["oracle_acc"].mean()),
        "gap_sel_mean": float(table["gap_sel"].mean()),
        "neg_transfer_rate": float(table["neg_transfer"].mean()),
        "rho_score_mean": float(table["rho_score"].mean()),
        "rho_ev_mean": float(table["rho_ev"].mean()),
        "rho_probe_mean": float(table["rho_probe"].mean()),
        "rho_probe_hard_mean": float(table["rho_probe_hard"].mean()),
        "rho_iwcv_mean": float(table["rho_iwcv"].mean()) if "rho_iwcv" in table.columns else float("nan"),
        "rho_ridge_mean": float(table["rho_ridge"].mean()) if "rho_ridge" in table.columns else float("nan"),
        "rho_guard_mean": float(table["rho_guard"].mean()) if "rho_guard" in table.columns else float("nan"),
        "best_score_mean": float(table["best_score_acc"].mean()),
        "best_ev_mean": float(table["best_ev_acc"].mean()),
        "best_probe_mean": float(table["best_probe_acc"].mean()),
        "best_probe_hard_mean": float(table["best_probe_hard_acc"].mean()),
        "best_iwcv_mean": float(table["best_iwcv_acc"].mean()) if "best_iwcv_acc" in table.columns else float("nan"),
        "best_ridge_mean": float(table["best_ridge_acc"].mean()) if "best_ridge_acc" in table.columns else float("nan"),
        "best_guard_mean": float(table["best_guard_acc"].mean()) if "best_guard_acc" in table.columns else float("nan"),
    }
    return table, summary


def main() -> None:
    ap = argparse.ArgumentParser(description="Analyze per-subject candidate-set certificate diagnostics.")
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--method", type=str, required=True, help="Method tag used under diagnostics/ (e.g., ea-zo-imr-csp-lda).")
    ap.add_argument("--save-csv", type=Path, default=None, help="Optional path to save the per-subject table CSV.")
    args = ap.parse_args()

    table, summary = analyze_run(run_dir=args.run_dir, method=args.method)
    if args.save_csv is not None:
        args.save_csv.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(args.save_csv, index=False)

    print("=== Summary ===")
    for k in [
        "sel_mean",
        "id_mean",
        "oracle_mean",
        "gap_sel_mean",
        "neg_transfer_rate",
        "rho_score_mean",
        "rho_ev_mean",
        "rho_probe_mean",
        "rho_probe_hard_mean",
        "rho_iwcv_mean",
        "rho_ridge_mean",
        "rho_guard_mean",
        "best_score_mean",
        "best_ev_mean",
        "best_probe_mean",
        "best_probe_hard_mean",
        "best_iwcv_mean",
        "best_ridge_mean",
        "best_guard_mean",
    ]:
        v = summary.get(k, float("nan"))
        print(f"{k}: {v:.6f}")

    print("\n=== Per-subject ===")
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
