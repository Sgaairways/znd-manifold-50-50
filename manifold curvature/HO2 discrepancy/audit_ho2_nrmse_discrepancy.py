#!/usr/bin/env python3
"""
audit_ho2_nrmse_discrepancy.py

Audit the discrepancy between the previously reported HO2 NRMSE (~1.0483%)
and the targeted worst-cell result (~0.1375%) at:

    T1 = 351.5 K
    P1 = 4.64 atm
    phi = 1

This script:
  1. Loads the four surrounding 50x50 manifold corner profiles.
  2. Loads the direct-truth profile at the query state.
  3. Reconstructs HO2 with bilinear interpolation on common physical-x grids.
  4. Sweeps common-grid resolution to test NRMSE sensitivity.
  5. Reports several normalization conventions.
  6. Computes an x-integrated (trapezoidal) NRMSE.
  7. Reports peak magnitude/location/FWHM errors.
  8. Saves CSVs and diagnostic figures.

NO SDToolbox calculations are run.

Expected project layout:
    h2_detonation/
      znd_tp_profiles/
      audit_ho2_nrmse_discrepancy.py

If automatic truth-file discovery cannot find the 351.5 K / 4.64 atm file,
set TRUTH_FILE manually below.
"""

from pathlib import Path
import re
import csv
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PROFILE_DIR = Path("znd_tp_profiles")
OUTPUT_DIR = Path("ho2_nrmse_discrepancy_audit")

QUERY_T = 351.5
QUERY_P = 4.64

T_LO = 342.857143
T_HI = 357.142857
P_LO = 4.6
P_HI = 4.7

GRID_SIZES = [500, 1200, 2500, 5000, 10000]

# Set to Path("...npz") if automatic discovery fails.
TRUTH_FILE = None

# Optional: if you know the exact old validator used a particular number
# of common-x points, add it here.
OLD_VALIDATOR_NX = 1200

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def scalar(v):
    a = np.asarray(v)
    return float(a.reshape(-1)[0])

def decode_species_names(arr):
    names = []
    for s in np.asarray(arr).reshape(-1):
        if isinstance(s, bytes):
            names.append(s.decode("utf-8"))
        else:
            names.append(str(s))
    return names

def load_npz(path):
    with np.load(path, allow_pickle=True) as d:
        keys = set(d.files)

        def first(candidates):
            for k in candidates:
                if k in keys:
                    return k
            return None

        xk = first(["x_m", "x", "distance_m", "distance"])
        yk = first(["Y", "mass_fractions"])
        sk = first(["species_names", "species"])
        tk = first(["T1_K", "T1", "temperature_initial_K"])
        pk = first(["P1_atm", "P1", "pressure_initial_atm"])

        if xk is None or yk is None or sk is None:
            raise KeyError(
                f"{path}: required x/Y/species keys not found. Keys={sorted(keys)}"
            )

        x = np.asarray(d[xk], dtype=float).squeeze()
        Y = np.asarray(d[yk], dtype=float)
        species = decode_species_names(d[sk])

        # Orient Y as (n_x, n_species)
        if Y.ndim != 2:
            raise ValueError(f"{path}: expected 2-D Y, got shape {Y.shape}")
        if Y.shape[0] == len(species) and Y.shape[1] == len(x):
            Y = Y.T
        elif Y.shape[0] == len(x) and Y.shape[1] == len(species):
            pass
        else:
            raise ValueError(
                f"{path}: cannot reconcile x={len(x)}, species={len(species)}, Y={Y.shape}"
            )

        T1 = scalar(d[tk]) if tk else np.nan
        P1 = scalar(d[pk]) if pk else np.nan

    if "HO2" not in species:
        raise ValueError(f"{path}: HO2 not found in species list {species}")

    ho2 = Y[:, species.index("HO2")]

    good = np.isfinite(x) & np.isfinite(ho2)
    x = x[good]
    ho2 = ho2[good]

    order = np.argsort(x)
    x = x[order]
    ho2 = ho2[order]

    # np.interp requires increasing coordinates; remove duplicate x values.
    x_unique, idx = np.unique(x, return_index=True)
    ho2 = ho2[idx]
    x = x_unique

    return {
        "path": path,
        "T1": T1,
        "P1": P1,
        "x": x,
        "ho2": ho2,
    }

def scan_profiles():
    records = []
    for path in PROFILE_DIR.rglob("*.npz"):
        try:
            rec = load_npz(path)
            records.append(rec)
        except Exception:
            continue
    if not records:
        raise RuntimeError(f"No readable ZND NPZ profiles found in {PROFILE_DIR.resolve()}")
    return records

def nearest_record(records, T, P, tol_T=0.05, tol_P=0.005, exclude=None):
    candidates = []
    for r in records:
        if exclude is not None and r["path"] == exclude:
            continue
        dT = abs(r["T1"] - T)
        dP = abs(r["P1"] - P)
        if dT <= tol_T and dP <= tol_P:
            candidates.append((dT + 10*dP, r))
    if not candidates:
        return None
    candidates.sort(key=lambda z: z[0])
    return candidates[0][1]

def bilinear_weights(T, P):
    a = (T - T_LO) / (T_HI - T_LO)
    b = (P - P_LO) / (P_HI - P_LO)
    return {
        (T_LO, P_LO): (1-a)*(1-b),
        (T_HI, P_LO): a*(1-b),
        (T_LO, P_HI): (1-a)*b,
        (T_HI, P_HI): a*b,
    }

def interp_profile(rec, x_common):
    return np.interp(x_common, rec["x"], rec["ho2"])

def rmse(a, b):
    return float(np.sqrt(np.mean((a-b)**2)))

def trapz_rms(error, x):
    L = float(x[-1] - x[0])
    if L <= 0:
        return np.nan
    # NumPy 2.x: trapezoid. Fall back for older NumPy.
    integ = np.trapezoid(error**2, x) if hasattr(np, "trapezoid") else np.trapz(error**2, x)
    return float(np.sqrt(integ / L))

def feature_metrics(x, y):
    i = int(np.nanargmax(y))
    peak = float(y[i])
    xpeak = float(x[i])

    half = 0.5 * peak
    above = np.where(y >= half)[0]
    if len(above) >= 2:
        fwhm = float(x[above[-1]] - x[above[0]])
    else:
        fwhm = np.nan

    return peak, xpeak, fwhm

def pct(err, denom):
    return 100.0 * err / denom if denom != 0 else np.nan

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    print("=" * 86)
    print("HO2 NRMSE DISCREPANCY AUDIT")
    print("=" * 86)
    print(f"Query: T1={QUERY_T:.3f} K, P1={QUERY_P:.3f} atm")
    print("No SDToolbox calculations will be run.\n")

    records = scan_profiles()
    print(f"Readable cached NPZ profiles discovered: {len(records)}")

    # Four corners
    corner_targets = [
        (T_LO, P_LO),
        (T_HI, P_LO),
        (T_LO, P_HI),
        (T_HI, P_HI),
    ]

    corners = {}
    for T, P in corner_targets:
        rec = nearest_record(records, T, P)
        if rec is None:
            raise FileNotFoundError(
                f"Could not find corner profile near T={T}, P={P} in {PROFILE_DIR}"
            )
        corners[(T, P)] = rec

    # Truth
    if TRUTH_FILE is not None:
        truth = load_npz(Path(TRUTH_FILE))
    else:
        truth = nearest_record(records, QUERY_T, QUERY_P)

    if truth is None:
        # Give useful inventory of near-query files.
        ranked = sorted(
            records,
            key=lambda r: abs(r["T1"]-QUERY_T)/10 + abs(r["P1"]-QUERY_P)
        )[:10]
        print("\nAutomatic truth discovery failed.")
        print("Closest cached profiles were:")
        for r in ranked:
            print(f"  T={r['T1']:.6f} K, P={r['P1']:.6f} atm : {r['path']}")
        print("\nSet TRUTH_FILE at the top of this script to the direct-truth NPZ.")
        return

    print("\nFiles used:")
    for key, rec in corners.items():
        print(f"  corner {key[0]:.6f} K, {key[1]:.6f} atm -> {rec['path']}")
    print(f"  truth  {truth['T1']:.6f} K, {truth['P1']:.6f} atm -> {truth['path']}")

    weights = bilinear_weights(QUERY_T, QUERY_P)
    print("\nBilinear weights:")
    for key, w in weights.items():
        print(f"  ({key[0]:.6f} K, {key[1]:.6f} atm): {w:.8f}")
    print(f"  sum = {sum(weights.values()):.12f}")

    # Common domain must be supported by truth and all corners.
    x0 = max([truth["x"][0]] + [r["x"][0] for r in corners.values()])
    xmax = min([truth["x"][-1]] + [r["x"][-1] for r in corners.values()])
    if xmax <= x0:
        raise RuntimeError("No overlapping physical-x interval among truth and corners.")

    print(f"\nCommon physical-x interval: {x0*1e3:.9f} to {xmax*1e3:.9f} mm")
    print(f"Common extent: {(xmax-x0)*1e3:.9f} mm")

    rows = []
    saved_profiles = {}

    for nx in GRID_SIZES:
        x = np.linspace(x0, xmax, nx)
        yt = interp_profile(truth, x)

        yi = np.zeros_like(x)
        for key, rec in corners.items():
            yi += weights[key] * interp_profile(rec, x)

        e = yi - yt
        r = rmse(yi, yt)
        r_int = trapz_rms(e, x)

        peak_abs = float(np.max(np.abs(yt)))
        yrange = float(np.max(yt) - np.min(yt))
        mean_abs = float(np.mean(np.abs(yt)))
        rms_truth = float(np.sqrt(np.mean(yt**2)))

        p_t, xp_t, fw_t = feature_metrics(x, yt)
        p_i, xp_i, fw_i = feature_metrics(x, yi)

        row = {
            "nx": nx,
            "dx_um": (x[1]-x[0])*1e6,
            "xmax_mm": xmax*1e3,
            "rmse": r,
            "trapz_rms": r_int,
            "truth_peak": p_t,
            "interp_peak": p_i,
            "peak_error_pct": 100*(p_i-p_t)/p_t,
            "truth_xpeak_mm": xp_t*1e3,
            "interp_xpeak_mm": xp_i*1e3,
            "xpeak_error_um": (xp_i-xp_t)*1e6,
            "truth_fwhm_mm": fw_t*1e3,
            "interp_fwhm_mm": fw_i*1e3,
            "fwhm_error_pct": 100*(fw_i-fw_t)/fw_t if fw_t else np.nan,
            "nrmse_peak_pct": pct(r, peak_abs),
            "nrmse_range_pct": pct(r, yrange),
            "nrmse_meanabs_pct": pct(r, mean_abs),
            "nrmse_truthrms_pct": pct(r, rms_truth),
            "integrated_nrmse_peak_pct": pct(r_int, peak_abs),
            "integrated_nrmse_range_pct": pct(r_int, yrange),
        }
        rows.append(row)
        saved_profiles[nx] = (x, yt, yi)

    # CSV
    csv_path = OUTPUT_DIR / "ho2_nrmse_grid_sensitivity.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    # Console table
    print("\n" + "-" * 86)
    print("GRID-RESOLUTION / NORMALIZATION AUDIT")
    print("-" * 86)
    print(
        f"{'Nx':>7} {'dx [um]':>10} {'RMSE':>13} "
        f"{'peak-norm %':>13} {'range-norm %':>14} {'trapz peak %':>14}"
    )
    for r in rows:
        print(
            f"{r['nx']:7d} {r['dx_um']:10.4f} {r['rmse']:13.6e} "
            f"{r['nrmse_peak_pct']:13.6f} {r['nrmse_range_pct']:14.6f} "
            f"{r['integrated_nrmse_peak_pct']:14.6f}"
        )

    # Highlight 1200-point result because old validator used 1200 common points.
    match = next((r for r in rows if r["nx"] == OLD_VALIDATOR_NX), None)
    if match:
        print("\n1200-point common-grid result:")
        for k in [
            "rmse", "nrmse_peak_pct", "nrmse_range_pct",
            "nrmse_meanabs_pct", "nrmse_truthrms_pct",
            "integrated_nrmse_peak_pct"
        ]:
            print(f"  {k:30s}: {match[k]:.10g}")

    # Feature metrics at finest grid
    fine = rows[-1]
    print("\n" + "-" * 86)
    print(f"HO2 FEATURE METRICS (Nx={fine['nx']})")
    print("-" * 86)
    print(f"Truth peak Y               : {fine['truth_peak']:.9e}")
    print(f"Bilinear peak Y            : {fine['interp_peak']:.9e}")
    print(f"Peak magnitude error       : {fine['peak_error_pct']:+.6f} %")
    print(f"Truth peak x               : {fine['truth_xpeak_mm']:.9f} mm")
    print(f"Bilinear peak x            : {fine['interp_xpeak_mm']:.9f} mm")
    print(f"Peak-location error        : {fine['xpeak_error_um']:+.6f} um")
    print(f"Truth FWHM                 : {fine['truth_fwhm_mm']:.9f} mm")
    print(f"Bilinear FWHM              : {fine['interp_fwhm_mm']:.9f} mm")
    print(f"FWHM error                 : {fine['fwhm_error_pct']:+.6f} %")

    # Figure 1: NRMSE vs grid size
    fig, ax = plt.subplots(figsize=(9, 5.5))
    nxv = np.array([r["nx"] for r in rows])
    ax.plot(nxv, [r["nrmse_peak_pct"] for r in rows], marker="o",
            label="pointwise RMSE / truth peak")
    ax.plot(nxv, [r["integrated_nrmse_peak_pct"] for r in rows], marker="s",
            label="x-integrated RMS / truth peak")
    ax.axhline(1.048301, linestyle="--", linewidth=1.5,
               label="previous reported 1.048301%")
    ax.axhline(0.137488, linestyle=":", linewidth=1.5,
               label="targeted reported 0.137488%")
    ax.set_xscale("log")
    ax.set_xlabel("Common physical-x grid points")
    ax.set_ylabel("HO2 normalized error [%]")
    ax.set_title("HO2 NRMSE discrepancy audit: grid-resolution sensitivity")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "01_nrmse_vs_grid_resolution.png", dpi=220)
    plt.close(fig)

    # Figure 2: finest-grid profile comparison
    x, yt, yi = saved_profiles[GRID_SIZES[-1]]
    fig, ax = plt.subplots(figsize=(10, 5.8))
    ax.plot(x*1e3, yt, linewidth=3, label="direct truth")
    ax.plot(x*1e3, yi, linestyle="--", linewidth=2.5, label="bilinear reconstruction")
    ax.set_xlim(0, min(0.20, xmax*1e3))
    ax.set_xlabel("Distance behind shock x [mm]")
    ax.set_ylabel("HO2 mass fraction")
    ax.set_title("HO2 truth vs bilinear reconstruction on audit grid")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "02_ho2_truth_vs_bilinear_audit.png", dpi=220)
    plt.close(fig)

    # Figure 3: squared-error density near reaction zone
    fig, ax = plt.subplots(figsize=(10, 5.8))
    ax.plot(x*1e3, (yi-yt)**2)
    ax.set_xlim(0, min(0.20, xmax*1e3))
    ax.set_xlabel("Distance behind shock x [mm]")
    ax.set_ylabel("(bilinear - truth)^2")
    ax.set_title("Where the HO2 profile error is accumulated")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "03_ho2_squared_error_vs_x.png", dpi=220)
    plt.close(fig)

    # Save exact file/weight audit.
    audit_txt = OUTPUT_DIR / "audit_inputs.txt"
    with audit_txt.open("w") as f:
        f.write(f"Query T1={QUERY_T} K, P1={QUERY_P} atm\n")
        f.write(f"Truth: {truth['path']}\n")
        f.write(f"Truth metadata: T1={truth['T1']}, P1={truth['P1']}\n")
        f.write(f"Common x: {x0} to {xmax} m\n\n")
        for key, rec in corners.items():
            f.write(
                f"Corner {key}: weight={weights[key]:.12f}, "
                f"file={rec['path']}, metadata T={rec['T1']}, P={rec['P1']}\n"
            )

    print("\nSaved:")
    print(f"  {csv_path}")
    print(f"  {OUTPUT_DIR / 'audit_inputs.txt'}")
    print(f"  {OUTPUT_DIR / '01_nrmse_vs_grid_resolution.png'}")
    print(f"  {OUTPUT_DIR / '02_ho2_truth_vs_bilinear_audit.png'}")
    print(f"  {OUTPUT_DIR / '03_ho2_squared_error_vs_x.png'}")

    print("\nInterpretation rule:")
    print("  • If the error changes strongly with Nx, the metric is grid-resolution sensitive.")
    print("  • If it converges near one value, compare the old/new normalization denominator.")
    print("  • If neither explains the discrepancy, compare audit_inputs.txt against the old")
    print("    validator's exact truth file, x-domain, corner files, and resampling sequence.")
    print("=" * 86)

if __name__ == "__main__":
    main()
