#!/usr/bin/env python3
"""
Experiment 2: multi-point off-grid validation of bilinear ZND profile reconstruction.

Reuses the existing cached T-P manifold produced by
znd_tp_bilinear_timeout_safe.py and validates bilinear reconstruction against
direct SDToolbox truth at nine off-grid states (one existing + eight new).

Run from ~/Desktop/h2_detonation:
    conda activate sdtoolbox
    python validate_bilinear_multipoint.py

Required beside this file:
    znd_tp_bilinear_timeout_safe.py
    ffcm2_h2.yaml
    sdtoolbox/
    znd_tp_profiles/   (existing manifold cache)

Outputs:
    results/bilinear_multipoint_validation.csv
    results/bilinear_multipoint_summary.csv
    results/bilinear_multipoint_case_summary.csv
    results/bilinear_multipoint_weights.csv
    figures/bilinear_multipoint/<case>/*.png
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import znd_tp_bilinear_timeout_safe as base


# ---------------------------------------------------------------------------
# 50 x 50 manifold configuration
# ---------------------------------------------------------------------------

base.FULL_T_VALUES = np.linspace(300.0, 1000.0, 50)
base.FULL_P_VALUES = np.linspace(0.1, 5.0, 50)
base.PROFILE_DIR = Path("znd_tp_profiles")


# ---------------------------------------------------------------------------
# Experiment design
# ---------------------------------------------------------------------------

VALIDATION_POINTS = [
    (351.5, 0.84),  # existing Experiment-1 truth
    (351.5, 2.54),
    (351.5, 4.64),
    (651.5, 0.84),
    (651.5, 2.54),
    (651.5, 4.64),
    (951.5, 0.84),
    (951.5, 2.54),
    (951.5, 4.64),
]

RESULT_DIR = Path("results_50x50")
FIGURE_ROOT = Path("figures_50x50") / "bilinear_multipoint"

LONG_CSV = RESULT_DIR / "bilinear_multipoint_validation.csv"
SUMMARY_CSV = RESULT_DIR / "bilinear_multipoint_summary.csv"
CASE_SUMMARY_CSV = RESULT_DIR / "bilinear_multipoint_case_summary.csv"
WEIGHTS_CSV = RESULT_DIR / "bilinear_multipoint_weights.csv"

# Direct SDToolbox truth stays independent of the interpolation manifold.
TRUTH_PROFILE_DIR = Path("znd_tp_profiles")

TRUTH_TIMEOUT_S = 60.0


def case_id(T_K: float, P_atm: float) -> str:
    return f"T{T_K:06.1f}_P{P_atm:05.2f}".replace(".", "p")


def truth_path(T_K: float, P_atm: float) -> Path:
    old_profile_dir = base.PROFILE_DIR
    try:
        base.PROFILE_DIR = TRUTH_PROFILE_DIR
        return base.target_filename(T_K, P_atm)
    finally:
        base.PROFILE_DIR = old_profile_dir


def ensure_truth_timeout_safe(T_K: float, P_atm: float, timeout_s: float) -> Path:
    """Generate direct truth in an isolated subprocess so one CJ hang cannot freeze the sweep."""
    path = truth_path(T_K, P_atm)
    if path.exists():
        print(f"  Reusing truth: {path}")
        return path

    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--truth-only",
        str(T_K),
        str(P_atm),
    ]
    print(f"  Calculating direct truth (timeout {timeout_s:.0f} s)...")
    try:
        subprocess.run(cmd, check=True, timeout=timeout_s)
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            f"Direct truth timed out at T={T_K} K, P={P_atm} atm"
        ) from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f"Direct truth failed at T={T_K} K, P={P_atm} atm"
        ) from exc

    if not path.exists():
        raise RuntimeError(f"Truth subprocess returned but did not create {path}")
    return path


def truth_worker(T_K: float, P_atm: float) -> None:
    """Hidden worker used by ensure_truth_timeout_safe()."""
    TRUTH_PROFILE_DIR.mkdir(exist_ok=True)
    profile = base.run_znd_profile(T_K, P_atm, base.PHI)
    base.save_profile(profile, truth_path(T_K, P_atm))


def verify_required_corners(T_K: float, P_atm: float) -> None:
    T0, T1 = base.bracket(T_K, base.FULL_T_VALUES)
    P0, P1 = base.bracket(P_atm, base.FULL_P_VALUES)
    missing = []
    for Tc, Pc in [(T0, P0), (T1, P0), (T0, P1), (T1, P1)]:
        f = base.case_filename(Tc, Pc)
        if not f.exists():
            missing.append((Tc, Pc, str(f)))
    if missing:
        text = "; ".join(f"{T:g} K, {P:g} atm ({f})" for T, P, f in missing)
        raise RuntimeError(f"Missing required bilinear corner(s): {text}")


def save_case_plots(comparison: dict, out_dir: Path) -> None:
    """Reuse the baseline plotting routine, then move its files into a case-specific folder."""
    out_dir.mkdir(parents=True, exist_ok=True)
    old_plot_dir = base.PLOT_DIR
    temp_dir = FIGURE_ROOT / "_current"
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    base.PLOT_DIR = temp_dir
    try:
        base.make_plots(comparison)
        for p in temp_dir.glob("*.png"):
            shutil.move(str(p), out_dir / p.name)
    finally:
        base.PLOT_DIR = old_plot_dir
        if temp_dir.exists():
            shutil.rmtree(temp_dir)


def aggregate_summary(long_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for quantity, g in long_df.groupby("quantity", sort=False):
        vals = g["NRMSE_pct"].dropna().to_numpy(float)
        rows.append({
            "quantity": quantity,
            "n_cases": len(vals),
            "mean_NRMSE_pct": np.mean(vals) if len(vals) else np.nan,
            "median_NRMSE_pct": np.median(vals) if len(vals) else np.nan,
            "max_NRMSE_pct": np.max(vals) if len(vals) else np.nan,
            "worst_case_T_K": (
                g.loc[g["NRMSE_pct"].idxmax(), "target_T_K"]
                if g["NRMSE_pct"].notna().any() else np.nan
            ),
            "worst_case_P_atm": (
                g.loc[g["NRMSE_pct"].idxmax(), "target_P_atm"]
                if g["NRMSE_pct"].notna().any() else np.nan
            ),
        })
    return pd.DataFrame(rows)


def main(timeout_s: float) -> None:
    RESULT_DIR.mkdir(exist_ok=True)
    FIGURE_ROOT.mkdir(parents=True, exist_ok=True)

    all_errors = []
    all_weights = []
    case_rows = []

    print("=" * 78)
    print("EXPERIMENT 2: MULTI-POINT BILINEAR ZND PROFILE VALIDATION")
    print("=" * 78)
    print(f"Validation states: {len(VALIDATION_POINTS)}")
    print("50 x 50 manifold is reused; no full 2500-state regeneration is performed.")
    print()

    for i, (T_K, P_atm) in enumerate(VALIDATION_POINTS, 1):
        cid = case_id(T_K, P_atm)
        print("\n" + "-" * 78)
        print(f"CASE {i}/{len(VALIDATION_POINTS)}: T={T_K:.1f} K, P={P_atm:.2f} atm")
        print("-" * 78)

        try:
            verify_required_corners(T_K, P_atm)

            pred, info = base.bilinear_interpolate_profiles(
                T_K, P_atm, base.FULL_T_VALUES, base.FULL_P_VALUES
            )

            tpath = ensure_truth_timeout_safe(T_K, P_atm, timeout_s)
            truth = base.load_profile(tpath)

            err_df, comparison = base.validate_prediction(pred, truth)
            err_df.insert(0, "case_id", cid)
            err_df.insert(1, "target_T_K", T_K)
            err_df.insert(2, "target_P_atm", P_atm)
            all_errors.append(err_df)

            weights = [
                ("T0_P0", info["T0"], info["P0"], info["w_T0_P0"]),
                ("T1_P0", info["T1"], info["P0"], info["w_T1_P0"]),
                ("T0_P1", info["T0"], info["P1"], info["w_T0_P1"]),
                ("T1_P1", info["T1"], info["P1"], info["w_T1_P1"]),
            ]
            for corner, Tc, Pc, w in weights:
                all_weights.append({
                    "case_id": cid,
                    "target_T_K": T_K,
                    "target_P_atm": P_atm,
                    "corner": corner,
                    "corner_T_K": Tc,
                    "corner_P_atm": Pc,
                    "weight": w,
                })

            save_case_plots(comparison, FIGURE_ROOT / cid)

            non_n2 = err_df.loc[err_df["quantity"] != "Y_N2", "NRMSE_pct"].dropna()
            case_rows.append({
                "case_id": cid,
                "target_T_K": T_K,
                "target_P_atm": P_atm,
                "status": "ok",
                "mean_NRMSE_pct_non_N2": non_n2.mean(),
                "max_NRMSE_pct_non_N2": non_n2.max(),
                "worst_quantity": (
                    err_df.loc[
                        err_df.loc[err_df["quantity"] != "Y_N2", "NRMSE_pct"].idxmax(),
                        "quantity"
                    ] if len(non_n2) else ""
                ),
            })

            print(err_df[["quantity", "NRMSE_pct"]].to_string(index=False))

        except Exception as exc:
            print(f"  FAILED: {exc}")
            case_rows.append({
                "case_id": cid,
                "target_T_K": T_K,
                "target_P_atm": P_atm,
                "status": "failed",
                "mean_NRMSE_pct_non_N2": np.nan,
                "max_NRMSE_pct_non_N2": np.nan,
                "worst_quantity": "",
                "error_message": str(exc),
            })

        # Checkpoint after every case.
        if all_errors:
            pd.concat(all_errors, ignore_index=True).to_csv(LONG_CSV, index=False)
        if all_weights:
            pd.DataFrame(all_weights).to_csv(WEIGHTS_CSV, index=False)
        pd.DataFrame(case_rows).to_csv(CASE_SUMMARY_CSV, index=False)

    if not all_errors:
        raise RuntimeError("No validation cases completed successfully.")

    long_df = pd.concat(all_errors, ignore_index=True)
    summary_df = aggregate_summary(long_df)
    summary_df.to_csv(SUMMARY_CSV, index=False)

    print("\n" + "=" * 78)
    print("AGGREGATE BILINEAR NRMSE SUMMARY")
    print("=" * 78)
    print(summary_df.to_string(index=False))

    print("\n" + "=" * 78)
    print("FILES SAVED")
    print("=" * 78)
    print(f"Per-case validation: {LONG_CSV.resolve()}")
    print(f"Aggregate summary:   {SUMMARY_CSV.resolve()}")
    print(f"Case summary:        {CASE_SUMMARY_CSV.resolve()}")
    print(f"Bilinear weights:    {WEIGHTS_CSV.resolve()}")
    print(f"Profile plots:       {FIGURE_ROOT.resolve()}")
    print("\nUse the aggregate table to identify the worst operating state before")
    print("drawing conclusions about where bilinear interpolation succeeds or fails.")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--truth-only",
        nargs=2,
        metavar=("T_K", "P_ATM"),
        type=float,
        help=argparse.SUPPRESS,
    )
    p.add_argument(
        "--truth-timeout",
        type=float,
        default=TRUTH_TIMEOUT_S,
        help=f"Timeout for each new direct-truth solve [s] (default {TRUTH_TIMEOUT_S:g}).",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.truth_only is not None:
        truth_worker(*args.truth_only)
    else:
        main(args.truth_timeout)