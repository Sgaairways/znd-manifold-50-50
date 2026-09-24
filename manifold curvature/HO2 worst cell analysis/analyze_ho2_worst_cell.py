#!/usr/bin/env python3
"""
analyze_ho2_worst_cell.py

Targeted follow-up to the 50x50 manifold curvature audit.

For the worst independent validation state:
    T1 = 351.5 K
    P1 = 4.64 atm

compare:
  1) the four surrounding cached 50x50 manifold profiles,
  2) the direct off-grid SDToolbox truth profile already cached, and
  3) a bilinear reconstruction made from the four corners.

No new SDToolbox calculations are run.

The script reports HO2 peak magnitude, peak x-location, and FWHM, then
generates a zoomed overlay plus a normalized-shape overlay.
"""

from pathlib import Path
import csv
import numpy as np
import matplotlib.pyplot as plt

PROFILE_DIR = Path("znd_tp_profiles")
OUT = Path("ho2_worst_cell_analysis")

TQ = 351.5
PQ = 4.64
TGRID = np.linspace(300.0, 1000.0, 50)
PGRID = np.linspace(0.1, 5.0, 50)

NODE_TOL_T = 1e-5
NODE_TOL_P = 1e-7
TRUTH_TOL_T = 1e-4
TRUTH_TOL_P = 1e-5

N_COMMON = 5000


def scalar(d, key):
    return float(np.asarray(d[key]).squeeze())


def decode_names(a):
    return [v.decode() if isinstance(v, bytes) else str(v)
            for v in np.asarray(a).ravel()]


def load_ho2(path):
    with np.load(path, allow_pickle=True) as d:
        needed = ["T1_K", "P1_atm", "x_m", "Y", "species_names"]
        if any(k not in d.files for k in needed):
            raise KeyError("missing expected keys")
        T1 = scalar(d, "T1_K")
        P1 = scalar(d, "P1_atm")
        x = np.asarray(d["x_m"], float).squeeze()
        Y = np.asarray(d["Y"], float)
        sp = decode_names(d["species_names"])

    if Y.shape == (len(sp), len(x)):
        Y = Y.T
    if Y.shape != (len(x), len(sp)):
        raise ValueError(f"unexpected Y shape {Y.shape}")
    if "HO2" not in sp:
        raise ValueError("HO2 not present")

    y = np.asarray(Y[:, sp.index("HO2")], float)
    good = np.isfinite(x) & np.isfinite(y)
    x, y = x[good], y[good]
    order = np.argsort(x)
    return {"T": T1, "P": P1, "x": x[order], "y": y[order],
            "file": path.name}


def scan_profiles():
    profiles = []
    for p in sorted(PROFILE_DIR.glob("*.npz")):
        try:
            profiles.append(load_ho2(p))
        except Exception:
            pass
    return profiles


def find_profile(profiles, T, P, tolT, tolP):
    hits = [q for q in profiles
            if abs(q["T"] - T) <= tolT and abs(q["P"] - P) <= tolP]
    if not hits:
        raise RuntimeError(f"No cached profile found near T={T}, P={P}")
    if len(hits) > 1:
        # Prefer the closest exact metadata match.
        hits.sort(key=lambda q: abs(q["T"]-T) + abs(q["P"]-P))
    return hits[0]


def bracket(grid, q):
    hi = int(np.searchsorted(grid, q, side="right"))
    lo = hi - 1
    if lo < 0 or hi >= len(grid):
        raise ValueError("query outside grid")
    return float(grid[lo]), float(grid[hi])


def peak_metrics(x, y):
    k = int(np.nanargmax(y))
    peak = float(y[k])
    xp = float(x[k])
    half = 0.5 * peak

    above = np.where(y >= half)[0]
    if len(above) < 2:
        fwhm = np.nan
    else:
        left_i, right_i = int(above[0]), int(above[-1])

        # Refine half-height crossings linearly where possible.
        xl = float(x[left_i])
        if left_i > 0 and y[left_i] != y[left_i-1]:
            xl = float(x[left_i-1] + (half-y[left_i-1]) *
                       (x[left_i]-x[left_i-1])/(y[left_i]-y[left_i-1]))

        xr = float(x[right_i])
        if right_i < len(x)-1 and y[right_i+1] != y[right_i]:
            xr = float(x[right_i] + (half-y[right_i]) *
                       (x[right_i+1]-x[right_i])/(y[right_i+1]-y[right_i]))

        fwhm = xr - xl

    return peak, xp, fwhm


def bilinear_weights(T0, T1, P0, P1, Tq, Pq):
    a = (Tq - T0) / (T1 - T0)
    b = (Pq - P0) / (P1 - P0)
    return {
        (T0, P0): (1-a)*(1-b),
        (T1, P0): a*(1-b),
        (T0, P1): (1-a)*b,
        (T1, P1): a*b,
    }


def nrmse(yhat, y):
    den = np.nanmax(np.abs(y))
    if den == 0:
        return np.nan
    return 100.0 * np.sqrt(np.nanmean((yhat-y)**2)) / den


def main():
    OUT.mkdir(exist_ok=True)
    if not PROFILE_DIR.exists():
        raise FileNotFoundError(f"Missing {PROFILE_DIR.resolve()}")

    profiles = scan_profiles()
    T0, T1 = bracket(TGRID, TQ)
    P0, P1 = bracket(PGRID, PQ)

    corners = {}
    for T in (T0, T1):
        for P in (P0, P1):
            corners[(T, P)] = find_profile(
                profiles, T, P, NODE_TOL_T, NODE_TOL_P
            )

    truth = find_profile(
        profiles, TQ, PQ, TRUTH_TOL_T, TRUTH_TOL_P
    )

    # Match the validated interpolation policy: common physical x extent is
    # limited by the shortest of the four corner profiles.
    xmax = min(float(q["x"][-1]) for q in corners.values())
    x = np.linspace(0.0, xmax, N_COMMON)

    weights = bilinear_weights(T0, T1, P0, P1, TQ, PQ)

    corner_y = {}
    for key, q in corners.items():
        corner_y[key] = np.interp(x, q["x"], q["y"])

    ybil = sum(weights[k] * corner_y[k] for k in corners)

    # Compare truth on exactly the same common physical x grid.
    valid = x <= truth["x"][-1]
    xv = x[valid]
    ytruth = np.interp(xv, truth["x"], truth["y"])
    ybil_v = ybil[valid]

    rows = []
    for key, q in corners.items():
        peak, xp, width = peak_metrics(q["x"], q["y"])
        rows.append({
            "profile": f"corner T={key[0]:.6f} K, P={key[1]:.6f} atm",
            "T1_K": key[0], "P1_atm": key[1],
            "weight": weights[key],
            "peak_HO2": peak, "peak_x_m": xp, "FWHM_m": width,
            "source_file": q["file"]
        })

    peak, xp, width = peak_metrics(truth["x"], truth["y"])
    rows.append({
        "profile": "direct truth",
        "T1_K": TQ, "P1_atm": PQ, "weight": np.nan,
        "peak_HO2": peak, "peak_x_m": xp, "FWHM_m": width,
        "source_file": truth["file"]
    })

    peak_b, xp_b, width_b = peak_metrics(xv, ybil_v)
    rows.append({
        "profile": "bilinear reconstruction",
        "T1_K": TQ, "P1_atm": PQ, "weight": np.nan,
        "peak_HO2": peak_b, "peak_x_m": xp_b, "FWHM_m": width_b,
        "source_file": "constructed from four cached corners"
    })

    with (OUT/"ho2_peak_metrics.csv").open("w", newline="") as f:
        fields = ["profile","T1_K","P1_atm","weight",
                  "peak_HO2","peak_x_m","FWHM_m","source_file"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    err = nrmse(ybil_v, ytruth)

    # Focus plot around the entire set of corner/truth peaks.
    all_peak_x = [r["peak_x_m"] for r in rows if np.isfinite(r["peak_x_m"])]
    all_width = [r["FWHM_m"] for r in rows if np.isfinite(r["FWHM_m"])]
    right = max(all_peak_x) + 4.0 * (max(all_width) if all_width else 1e-4)
    right = min(right, xmax)
    if right <= 0:
        right = min(0.003, xmax)

    fig, ax = plt.subplots(figsize=(10,6))
    for key in [(T0,P0),(T1,P0),(T0,P1),(T1,P1)]:
        q = corners[key]
        mask = q["x"] <= right
        ax.plot(q["x"][mask]*1000, q["y"][mask],
                linewidth=1.3, alpha=0.72,
                label=f"corner {key[0]:.3f} K, {key[1]:.1f} atm")
    mask = truth["x"] <= right
    ax.plot(truth["x"][mask]*1000, truth["y"][mask],
            linewidth=2.8, label="direct truth 351.5 K, 4.64 atm")
    mask = xv <= right
    ax.plot(xv[mask]*1000, ybil_v[mask],
            linestyle="--", linewidth=2.4,
            label="bilinear reconstruction")
    ax.set_xlabel("Distance behind shock x [mm]")
    ax.set_ylabel("HO2 mass fraction")
    ax.set_title("Worst validation cell: HO2 profile alignment")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(OUT/"01_ho2_four_corners_truth_bilinear.png",
                dpi=300,bbox_inches="tight")
    plt.close(fig)

    # Normalize each curve by its own peak: isolates spatial translation/shape.
    fig, ax = plt.subplots(figsize=(10,6))
    for key in [(T0,P0),(T1,P0),(T0,P1),(T1,P1)]:
        q = corners[key]
        peak = np.nanmax(q["y"])
        mask = q["x"] <= right
        ax.plot(q["x"][mask]*1000, q["y"][mask]/peak,
                linewidth=1.3, alpha=0.72,
                label=f"corner {key[0]:.3f} K, {key[1]:.1f} atm")
    peak = np.nanmax(truth["y"])
    mask = truth["x"] <= right
    ax.plot(truth["x"][mask]*1000, truth["y"][mask]/peak,
            linewidth=2.8, label="direct truth")
    peak = np.nanmax(ybil_v)
    mask = xv <= right
    ax.plot(xv[mask]*1000, ybil_v[mask]/peak,
            linestyle="--", linewidth=2.4,
            label="bilinear reconstruction")
    ax.set_xlabel("Distance behind shock x [mm]")
    ax.set_ylabel("HO2 / own peak")
    ax.set_title("HO2 spatial alignment with amplitude normalized out")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(OUT/"02_ho2_normalized_shape_alignment.png",
                dpi=300,bbox_inches="tight")
    plt.close(fig)

    # Peak x versus corner/query state: compact diagnostic.
    fig, ax = plt.subplots(figsize=(8,5.5))
    for key, q in corners.items():
        _, px, _ = peak_metrics(q["x"], q["y"])
        ax.scatter(key[0], px*1000, s=80,
                   label=f"P1={key[1]:.1f} atm")
    ax.scatter(TQ, rows[-2]["peak_x_m"]*1000, marker="*", s=180,
               label="direct truth")
    ax.scatter(TQ, rows[-1]["peak_x_m"]*1000, marker="x", s=100,
               label="bilinear")
    ax.set_xlabel("Initial temperature T1 [K]")
    ax.set_ylabel("HO2 peak location [mm]")
    ax.set_title("HO2 peak location inside worst interpolation cell")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(OUT/"03_ho2_peak_location_cell.png",
                dpi=300,bbox_inches="tight")
    plt.close(fig)

    print("="*78)
    print("TARGETED HO2 WORST-CELL EXPERIMENT")
    print("="*78)
    print(f"Query: T1={TQ:.3f} K, P1={PQ:.3f} atm")
    print(f"Cell: T=[{T0:.6f}, {T1:.6f}] K, P=[{P0:.6f}, {P1:.6f}] atm")
    print(f"Common physical-x extent: {xmax*1000:.6f} mm")
    print("\nBilinear weights:")
    for key in [(T0,P0),(T1,P0),(T0,P1),(T1,P1)]:
        print(f"  ({key[0]:.6f} K, {key[1]:.6f} atm): {weights[key]:.8f}")

    print("\nHO2 feature metrics:")
    print(f"{'profile':46s} {'peak Y':>12s} {'xpeak mm':>12s} {'FWHM mm':>12s}")
    for r in rows:
        print(f"{r['profile'][:46]:46s} "
              f"{r['peak_HO2']:12.6e} "
              f"{r['peak_x_m']*1000:12.6f} "
              f"{r['FWHM_m']*1000:12.6f}")

    print(f"\nFull-profile HO2 NRMSE on common x grid: {err:.6f}%")
    print("\nSaved:")
    for name in [
        "ho2_peak_metrics.csv",
        "01_ho2_four_corners_truth_bilinear.png",
        "02_ho2_normalized_shape_alignment.png",
        "03_ho2_peak_location_cell.png",
    ]:
        print(f"  {OUT/name}")
    print("\nNo SDToolbox calculations were run.")


if __name__ == "__main__":
    main()
