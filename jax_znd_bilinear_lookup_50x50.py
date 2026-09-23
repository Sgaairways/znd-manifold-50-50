#!/usr/bin/env python3
"""
jax_znd_bilinear_lookup.py

Controlled JAX port of the validated SciPy bilinear ZND profile lookup.

Purpose
-------
Keep the 50 x 50 SDToolbox-generated manifold and physical-x preprocessing,
but replace the T-P bilinear profile reconstruction with JAX.

For each off-grid query:
1. Load the same four cached ZND corner profiles used by the SciPy baseline.
2. Resample those four profiles onto the same common physical-x grid.
3. Compute a SciPy RegularGridInterpolator reference prediction.
4. Compute the same bilinear weighted sum with JAX.
5. Compare JAX vs SciPy (implementation regression test).
6. Compare JAX vs the existing direct SDToolbox truth, if cached.

This script DOES NOT regenerate the ZND manifold. Put it in the same project
directory as znd_tp_profiles_50x50/ and the original znd_tp_profiles/ truth cache.

Examples
--------
Single original validation state:
    python jax_znd_bilinear_lookup.py --target-T 351.5 --target-P 0.84

Run the established 9-point validation set:
    python jax_znd_bilinear_lookup.py --nine-point

Optional benchmark:
    python jax_znd_bilinear_lookup.py --nine-point --benchmark
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp


FULL_T_VALUES = np.linspace(300.0, 1000.0, 50)
FULL_P_VALUES = np.linspace(0.1, 5.0, 50)

N_COMMON_X = 1200
PROFILE_DIR = Path("znd_tp_profiles")
TRUTH_DIR = Path("znd_tp_profiles")
OUTPUT_DIR = Path("jax_validation")

PREFERRED_SPECIES = ["H2", "O2", "H2O", "OH", "H", "O", "HO2", "H2O2", "N2"]

NINE_POINT_CASES = [
    (351.5, 0.84),
    (351.5, 2.54),
    (351.5, 4.64),
    (651.5, 0.84),
    (651.5, 2.54),
    (651.5, 4.64),
    (951.5, 0.84),
    (951.5, 2.54),
    (951.5, 4.64),
]


def case_filename(T_K: float, P_atm: float) -> Path:
    return PROFILE_DIR / f"znd_T{T_K:07.2f}_P{P_atm:05.2f}.npz"


def target_filename(T_K: float, P_atm: float) -> Path:
    return TRUTH_DIR / f"truth_target_T{T_K:07.2f}_P{P_atm:06.3f}.npz"


def load_profile(filename: Path) -> dict:
    if not filename.exists():
        raise FileNotFoundError(f"Missing required profile: {filename}")
    with np.load(filename, allow_pickle=False) as d:
        return {key: d[key] for key in d.files}


def as_1d(a) -> np.ndarray:
    return np.asarray(a, dtype=float).reshape(-1)


def bracket(value: float, grid: np.ndarray) -> tuple[float, float]:
    grid = np.asarray(grid, dtype=float)
    if not (grid[0] < value < grid[-1]):
        raise ValueError(f"Target {value} must lie strictly inside [{grid[0]}, {grid[-1]}].")
    hi = int(np.searchsorted(grid, value, side="right"))
    lo = hi - 1
    if np.isclose(value, grid[lo]) or np.isclose(value, grid[hi]):
        raise ValueError(f"Target {value} is explicitly tabulated; choose an off-grid point.")
    return float(grid[lo]), float(grid[hi])


def resample_profile(profile: dict, x_common: np.ndarray) -> dict:
    x = as_1d(profile["x_m"])
    T_prof = as_1d(profile["T_K"])
    P_prof = as_1d(profile["P_Pa"])
    Y = np.asarray(profile["Y"], dtype=float)
    return {
        "T_K": np.interp(x_common, x, T_prof),
        "P_Pa": np.interp(x_common, x, P_prof),
        "Y": np.vstack([np.interp(x_common, x, Y[k, :]) for k in range(Y.shape[0])]),
    }


@jax.jit
def jax_bilinear_combine(
    target_T: jax.Array,
    target_P: jax.Array,
    T0: jax.Array,
    T1: jax.Array,
    P0: jax.Array,
    P1: jax.Array,
    cube: jax.Array,
) -> jax.Array:
    """Bilinear interpolation of a [2,2,...] corner tensor."""
    a = (target_T - T0) / (T1 - T0)
    b = (target_P - P0) / (P1 - P0)

    w00 = (1.0 - a) * (1.0 - b)
    w10 = a * (1.0 - b)
    w01 = (1.0 - a) * b
    w11 = a * b

    return (
        w00 * cube[0, 0]
        + w10 * cube[1, 0]
        + w01 * cube[0, 1]
        + w11 * cube[1, 1]
    )


def prepare_corner_cubes(target_T: float, target_P: float):
    T0, T1 = bracket(target_T, FULL_T_VALUES)
    P0, P1 = bracket(target_P, FULL_P_VALUES)

    corner_files = {
        (T0, P0): case_filename(T0, P0),
        (T0, P1): case_filename(T0, P1),
        (T1, P0): case_filename(T1, P0),
        (T1, P1): case_filename(T1, P1),
    }

    missing = [k for k, f in corner_files.items() if not f.exists()]
    if missing:
        raise RuntimeError(
            "Cannot interpolate because required manifold corner(s) are missing: "
            + ", ".join(f"({T:.1f} K, {P:.2f} atm)" for T, P in missing)
        )

    corners = {k: load_profile(f) for k, f in corner_files.items()}
    x_max_common = min(float(np.max(c["x_m"])) for c in corners.values())
    x_common = np.linspace(0.0, x_max_common, N_COMMON_X)

    species_names = list(corners[(T0, P0)]["species_names"].astype(str))
    for c in corners.values():
        if list(c["species_names"].astype(str)) != species_names:
            raise ValueError("Species ordering differs between cached profiles.")

    sampled = {k: resample_profile(c, x_common) for k, c in corners.items()}

    T_cube = np.empty((2, 2, N_COMMON_X), dtype=np.float64)
    P_cube = np.empty((2, 2, N_COMMON_X), dtype=np.float64)
    Y_cube = np.empty((2, 2, len(species_names), N_COMMON_X), dtype=np.float64)

    for i, Tv in enumerate([T0, T1]):
        for j, Pv in enumerate([P0, P1]):
            s = sampled[(Tv, Pv)]
            T_cube[i, j] = s["T_K"]
            P_cube[i, j] = s["P_Pa"]
            Y_cube[i, j] = s["Y"]

    return x_common, np.asarray(species_names, dtype="U"), (T0, T1, P0, P1), T_cube, P_cube, Y_cube


def scipy_reference(target_T, target_P, bounds, T_cube, P_cube, Y_cube):
    T0, T1, P0, P1 = bounds
    axes = (np.array([T0, T1]), np.array([P0, P1]))
    q = np.array([target_T, target_P], dtype=float)

    return {
        "T_K": np.asarray(RegularGridInterpolator(axes, T_cube)(q)).reshape(-1),
        "P_Pa": np.asarray(RegularGridInterpolator(axes, P_cube)(q)).reshape(-1),
        "Y": np.asarray(RegularGridInterpolator(axes, Y_cube)(q)).reshape(Y_cube.shape[2], N_COMMON_X),
    }


def jax_prediction(target_T, target_P, bounds, T_cube, P_cube, Y_cube):
    T0, T1, P0, P1 = bounds
    scalars = tuple(jnp.asarray(v, dtype=jnp.float64) for v in (target_T, target_P, T0, T1, P0, P1))

    pred_T = jax_bilinear_combine(*scalars, jnp.asarray(T_cube))
    pred_P = jax_bilinear_combine(*scalars, jnp.asarray(P_cube))
    pred_Y = jax_bilinear_combine(*scalars, jnp.asarray(Y_cube))

    # block_until_ready makes timing and completion explicit.
    pred_Y.block_until_ready()
    return {
        "T_K": np.asarray(pred_T),
        "P_Pa": np.asarray(pred_P),
        "Y": np.asarray(pred_Y),
    }


def field_metrics(pred: np.ndarray, truth: np.ndarray, normalization: str) -> dict:
    pred = np.asarray(pred, dtype=float)
    truth = np.asarray(truth, dtype=float)
    err = pred - truth
    rmse = float(np.sqrt(np.mean(err**2)))
    mae = float(np.mean(np.abs(err)))
    max_abs = float(np.max(np.abs(err)))

    if normalization == "range":
        denom = float(np.max(truth) - np.min(truth))
    elif normalization == "peak":
        denom = float(np.max(np.abs(truth)))
    else:
        raise ValueError(normalization)

    nrmse = np.nan if denom <= 1e-30 else 100.0 * rmse / denom
    return {"RMSE": rmse, "MAE": mae, "max_abs_error": max_abs, "NRMSE_pct": nrmse}


def compare_jax_scipy(jax_pred: dict, scipy_pred: dict, species_names: np.ndarray) -> pd.DataFrame:
    rows = []
    for label, a, b in [
        ("Temperature", jax_pred["T_K"], scipy_pred["T_K"]),
        ("Pressure", jax_pred["P_Pa"], scipy_pred["P_Pa"]),
    ]:
        d = np.asarray(a) - np.asarray(b)
        rows.append({
            "quantity": label,
            "max_abs_JAX_minus_SciPy": float(np.max(np.abs(d))),
            "RMSE_JAX_minus_SciPy": float(np.sqrt(np.mean(d**2))),
        })

    names = list(species_names.astype(str))
    for sp in PREFERRED_SPECIES:
        if sp in names:
            k = names.index(sp)
            d = jax_pred["Y"][k] - scipy_pred["Y"][k]
            rows.append({
                "quantity": f"Y_{sp}",
                "max_abs_JAX_minus_SciPy": float(np.max(np.abs(d))),
                "RMSE_JAX_minus_SciPy": float(np.sqrt(np.mean(d**2))),
            })
    return pd.DataFrame(rows)


def compare_to_truth(jax_pred: dict, truth_raw: dict, x_common: np.ndarray, species_names: np.ndarray) -> pd.DataFrame:
    mask = x_common <= float(np.max(truth_raw["x_m"]))
    x = x_common[mask]
    truth = resample_profile(truth_raw, x)

    rows = [
        {"quantity": "Temperature", **field_metrics(jax_pred["T_K"][mask], truth["T_K"], "range")},
        {"quantity": "Pressure", **field_metrics(jax_pred["P_Pa"][mask], truth["P_Pa"], "range")},
    ]

    names = list(species_names.astype(str))
    for sp in PREFERRED_SPECIES:
        if sp in names:
            k = names.index(sp)
            rows.append({
                "quantity": f"Y_{sp}",
                **field_metrics(jax_pred["Y"][k, mask], truth["Y"][k], "peak"),
            })
    return pd.DataFrame(rows)


def run_case(target_T: float, target_P: float, benchmark: bool = False):
    x, species_names, bounds, T_cube, P_cube, Y_cube = prepare_corner_cubes(target_T, target_P)

    scipy_pred = scipy_reference(target_T, target_P, bounds, T_cube, P_cube, Y_cube)

    # First call includes JIT compilation.
    t0 = time.perf_counter()
    jax_pred = jax_prediction(target_T, target_P, bounds, T_cube, P_cube, Y_cube)
    first_jax_s = time.perf_counter() - t0

    regression = compare_jax_scipy(jax_pred, scipy_pred, species_names)

    truth_file = target_filename(target_T, target_P)
    truth_errors = None
    if truth_file.exists():
        truth = load_profile(truth_file)
        truth_errors = compare_to_truth(jax_pred, truth, x, species_names)

    bench = {}
    if benchmark:
        # Warmed JAX timing: interpolation only; excludes disk I/O and x-resampling.
        T0, T1, P0, P1 = bounds
        scalars = tuple(jnp.asarray(v, dtype=jnp.float64) for v in (target_T, target_P, T0, T1, P0, P1))
        Yj = jnp.asarray(Y_cube)

        _ = jax_bilinear_combine(*scalars, Yj).block_until_ready()
        n = 1000
        t0 = time.perf_counter()
        for _ in range(n):
            out = jax_bilinear_combine(*scalars, Yj)
        out.block_until_ready()
        bench["jax_species_lookup_us"] = 1e6 * (time.perf_counter() - t0) / n

        axes = (np.array([T0, T1]), np.array([P0, P1]))
        rgi = RegularGridInterpolator(axes, Y_cube)
        q = np.array([target_T, target_P])
        t0 = time.perf_counter()
        for _ in range(n):
            _ = rgi(q)
        bench["scipy_species_lookup_us"] = 1e6 * (time.perf_counter() - t0) / n

    return {
        "target_T": target_T,
        "target_P": target_P,
        "bounds": bounds,
        "species_names": species_names,
        "first_jax_s": first_jax_s,
        "regression": regression,
        "truth_errors": truth_errors,
        "benchmark": bench,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--target-T", type=float, default=351.5)
    p.add_argument("--target-P", type=float, default=0.84)
    p.add_argument("--nine-point", action="store_true")
    p.add_argument("--benchmark", action="store_true")
    args = p.parse_args()

    if not PROFILE_DIR.exists():
        raise FileNotFoundError(
            f"{PROFILE_DIR} not found. Put this script in the existing h2_detonation project "
            "directory containing the cached ZND manifold."
        )

    OUTPUT_DIR.mkdir(exist_ok=True)

    print("=" * 78)
    print("JAX ZND BILINEAR PROFILE LOOKUP")
    print("=" * 78)
    print(f"JAX version:       {jax.__version__}")
    print(f"JAX backend:       {jax.default_backend()}")
    print(f"JAX x64 enabled:   {jax.config.jax_enable_x64}")
    print("Physics/database:  cached 50 x 50 SDToolbox manifold")
    print("Profile coord.:    same physical-x preprocessing as SciPy baseline")
    print("Interpolation:     JAX-jitted bilinear weighted sum")

    cases = NINE_POINT_CASES if args.nine_point else [(args.target_T, args.target_P)]

    all_truth = []
    all_reg = []

    for i, (Tq, Pq) in enumerate(cases, 1):
        print("\n" + "-" * 78)
        print(f"CASE {i}/{len(cases)}: T={Tq:.1f} K, P={Pq:.2f} atm")
        print("-" * 78)

        result = run_case(Tq, Pq, benchmark=args.benchmark)
        T0, T1, P0, P1 = result["bounds"]
        print(f"Corners: T=({T0:.1f},{T1:.1f}) K, P=({P0:.2f},{P1:.2f}) atm")
        print(f"First JAX call (includes compile): {1e3*result['first_jax_s']:.3f} ms")

        reg = result["regression"].copy()
        reg.insert(0, "P_atm", Pq)
        reg.insert(0, "T_K", Tq)
        all_reg.append(reg)

        max_reg = reg["max_abs_JAX_minus_SciPy"].max()
        print(f"Max absolute JAX-SciPy discrepancy across reported fields: {max_reg:.6e}")

        if result["truth_errors"] is not None:
            truth_df = result["truth_errors"].copy()
            truth_df.insert(0, "P_atm", Pq)
            truth_df.insert(0, "T_K", Tq)
            all_truth.append(truth_df)
            print(truth_df[["quantity", "NRMSE_pct"]].to_string(index=False))
        else:
            print(f"Direct truth cache not found: {target_filename(Tq, Pq)}")
            print("JAX-vs-SciPy regression was still completed.")

        if result["benchmark"]:
            print("Warmed interpolation-only benchmark:")
            for k, v in result["benchmark"].items():
                print(f"  {k}: {v:.3f}")

    reg_all = pd.concat(all_reg, ignore_index=True)
    reg_file = OUTPUT_DIR / "jax_vs_scipy_regression.csv"
    reg_all.to_csv(reg_file, index=False)

    if all_truth:
        truth_all = pd.concat(all_truth, ignore_index=True)
        truth_file = OUTPUT_DIR / "jax_vs_sdtoolbox_validation.csv"
        truth_all.to_csv(truth_file, index=False)

        aggregate = (
            truth_all.groupby("quantity")["NRMSE_pct"]
            .agg(["count", "mean", "median", "max"])
            .reset_index()
            .rename(columns={
                "count": "n_cases",
                "mean": "mean_NRMSE_pct",
                "median": "median_NRMSE_pct",
                "max": "max_NRMSE_pct",
            })
        )
        aggregate_file = OUTPUT_DIR / "jax_aggregate_nrmse.csv"
        aggregate.to_csv(aggregate_file, index=False)

        print("\n" + "=" * 78)
        print("AGGREGATE JAX VS DIRECT SDToolbox")
        print("=" * 78)
        print(aggregate.to_string(index=False))

    print("\n" + "=" * 78)
    print("REGRESSION CHECK: JAX VS SCIPY")
    print("=" * 78)
    reg_summary = (
        reg_all.groupby("quantity")[["max_abs_JAX_minus_SciPy", "RMSE_JAX_minus_SciPy"]]
        .max()
        .reset_index()
    )
    print(reg_summary.to_string(index=False))

    print("\nSaved:")
    print(f"  {reg_file.resolve()}")
    if all_truth:
        print(f"  {truth_file.resolve()}")
        print(f"  {aggregate_file.resolve()}")

    print("\nSuccess criterion:")
    print(
        "JAX should agree with the established SciPy bilinear prediction to numerical precision, "
        "and the nine-point direct-truth NRMSE should reproduce the 50 x 50 SciPy validation baseline."
    )


if __name__ == "__main__":
    main()
