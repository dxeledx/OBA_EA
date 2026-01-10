from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def _find_single(run_dir: Path, pattern: str) -> Path:
    paths = sorted(Path(run_dir).glob(pattern))
    if not paths:
        raise RuntimeError(f"No files match {pattern} under {run_dir}")
    if len(paths) > 1:
        raise RuntimeError(f"Expected 1 file match for {pattern} under {run_dir}, got {len(paths)}: {paths}")
    return paths[0]


def _infer_date_prefix(pred_path: Path) -> str:
    name = pred_path.name
    # Expect YYYYMMDD_predictions_all_methods.csv
    parts = name.split("_", 1)
    if parts and len(parts[0]) == 8 and parts[0].isdigit():
        return parts[0]
    return "merged"


def _load_accept_rates(run_dir: Path) -> dict[str, float]:
    try:
        path = _find_single(run_dir, "*_method_comparison.csv")
    except Exception:
        return {}
    df = pd.read_csv(path)
    if "method" not in df.columns or "accept_rate" not in df.columns:
        return {}
    out: dict[str, float] = {}
    for _, row in df.iterrows():
        m = str(row["method"])
        v = float(row["accept_rate"]) if np.isfinite(row["accept_rate"]) else float("nan")
        out[m] = v
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Merge multiple LOSO run folders into a single predictions_all_methods.csv.")
    p.add_argument("--out-run-dir", type=Path, required=True)
    p.add_argument(
        "--run-dirs",
        type=Path,
        nargs="+",
        required=True,
        help="One or more run directories containing *_predictions_all_methods.csv.",
    )
    p.add_argument(
        "--prefer-date-prefix",
        type=str,
        default="",
        help="Optional YYYYMMDD prefix to use for output filenames (default: infer from first run).",
    )
    args = p.parse_args()

    run_dirs = [Path(d) for d in args.run_dirs]
    if len(run_dirs) < 1:
        raise SystemExit("Need at least one --run-dirs entry.")

    out_dir = Path(args.out_run_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    method_to_df: dict[str, pd.DataFrame] = {}
    base_key: pd.DataFrame | None = None
    accept_rate: dict[str, float] = {}

    date_prefix = str(args.prefer_date_prefix).strip() or ""

    sources: list[tuple[str, str]] = []
    for run_dir in run_dirs:
        pred_path = _find_single(run_dir, "*_predictions_all_methods.csv")
        if not date_prefix:
            date_prefix = _infer_date_prefix(pred_path)
        df = pd.read_csv(pred_path)
        required = {"method", "subject", "trial", "y_true", "y_pred"}
        missing = required - set(df.columns)
        if missing:
            raise RuntimeError(f"Missing columns in {pred_path}: {sorted(missing)}")

        # Keep only needed columns; preserve proba_* if present.
        keep_cols = [c for c in df.columns if c in required or c.startswith("proba_")]
        df = df[keep_cols].copy()
        df["method"] = df["method"].astype(str)

        # Build/validate a base key (subject,trial,y_true) to ensure consistent trial indexing across runs.
        key_cols = ["subject", "trial", "y_true"]
        cur_key = df[key_cols].drop_duplicates().sort_values(key_cols).reset_index(drop=True)
        if base_key is None:
            base_key = cur_key
        else:
            if not cur_key.equals(base_key):
                raise RuntimeError(
                    f"Trial index mismatch when merging {run_dir}. "
                    "Ensure all runs use identical dataset/events/sessions/preprocess settings."
                )

        for m, g in df.groupby("method", sort=True):
            m = str(m)
            g2 = g.sort_values(["subject", "trial"]).reset_index(drop=True)
            if m in method_to_df:
                # If duplicated method is provided, require exact match.
                if not g2.equals(method_to_df[m]):
                    raise RuntimeError(f"Method '{m}' appears multiple times with different predictions.")
            else:
                method_to_df[m] = g2

        accept_rate.update({k: v for k, v in _load_accept_rates(run_dir).items() if k not in accept_rate})
        # Record source command if results.txt exists.
        try:
            results_path = _find_single(run_dir, "*_results.txt")
            cmd_line = ""
            for line in results_path.read_text(encoding="utf-8").splitlines():
                if line.startswith("Command:"):
                    cmd_line = line[len("Command:") :].strip()
                    break
            sources.append((str(run_dir), cmd_line))
        except Exception:
            sources.append((str(run_dir), ""))

    if base_key is None or not method_to_df:
        raise RuntimeError("No methods found to merge.")

    # Write merged per-method predictions + combined file.
    all_parts: list[pd.DataFrame] = []
    for m in sorted(method_to_df.keys()):
        df_m = method_to_df[m].copy()
        out_path = out_dir / f"{date_prefix}_{m}_predictions.csv"
        df_m.drop(columns=["method"]).to_csv(out_path, index=False)
        all_parts.append(df_m)
    merged = pd.concat(all_parts, axis=0, ignore_index=True)
    merged.to_csv(out_dir / f"{date_prefix}_predictions_all_methods.csv", index=False)

    # Minimal method comparison table (mean/worst acc + optional accept_rate).
    merged["correct"] = (merged["y_true"].astype(str) == merged["y_pred"].astype(str)).astype(float)
    acc = merged.groupby(["method", "subject"], sort=True)["correct"].mean().reset_index()
    rows = []
    base_method = "ea-csp-lda"
    base_acc = None
    if base_method in merged["method"].unique().tolist():
        base_acc = acc[acc["method"] == base_method].set_index("subject")["correct"]
    for m in sorted(acc["method"].unique().tolist()):
        g = acc[acc["method"] == m]
        row = {
            "method": m,
            "n_subjects": int(g.shape[0]),
            "mean_accuracy": float(g["correct"].mean()),
            "worst_accuracy": float(g["correct"].min()),
            "accept_rate": float(accept_rate.get(m, float("nan"))),
        }
        if base_acc is not None and m != base_method:
            m_acc = g.set_index("subject")["correct"]
            common = m_acc.index.intersection(base_acc.index)
            if len(common) > 0:
                delta = m_acc.loc[common] - base_acc.loc[common]
                row["mean_delta_vs_ea"] = float(delta.mean())
                row["neg_transfer_rate_vs_ea"] = float(np.mean(delta < -1e-12))
            else:
                row["mean_delta_vs_ea"] = float("nan")
                row["neg_transfer_rate_vs_ea"] = float("nan")
        rows.append(row)
    pd.DataFrame(rows).to_csv(out_dir / f"{date_prefix}_method_comparison.csv", index=False)

    # Source provenance.
    prov_lines = ["Merged from:"]
    for d, cmd in sources:
        prov_lines.append(f"- {d}")
        if cmd:
            prov_lines.append(f"  Command: {cmd}")
    (out_dir / f"{date_prefix}_MERGED_FROM.txt").write_text("\n".join(prov_lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

