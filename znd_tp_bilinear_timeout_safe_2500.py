#!/usr/bin/env python3
"""
znd_tp_manifold_baseline.py

First full-profile ZND manifold / interpolation baseline for Ryan + Chris.

At fixed stoichiometric H2-air (phi = 1):
    ZND(T1, P1) -> Y_i(x), T(x), p(x)

Workflow:
1. Build/cache a regular T-P table of SDToolbox ZND profiles.
2. Choose an off-grid state (default 351.5 K, 0.85 atm).
3. Use SciPy RegularGridInterpolator for bilinear interpolation in T-P.
4. Independently calculate the SDToolbox truth at the target state.
5. Compare temperature, pressure, and species profiles.
6. Save plots, error tables, interpolation weights, and NPZ data.

Expected project layout:
    ffcm2_h2.yaml
    sdtoolbox/
    znd_tp_manifold_baseline.py

First run a smoke test:
    conda activate sdtoolbox
    cd ~/Desktop/h2_detonation
    python znd_tp_manifold_baseline.py --quick

Then run the full requested table:
    python znd_tp_manifold_baseline.py

Full default grid:
    T = 300, 400, ..., 1000 K
    P = 0.1, 0.2, ..., 5.0 atm
    phi = 1.0
Total = 400 tabulated ZND solutions.

Each case is cached individually, so interrupted runs can be restarted.\nGrid cases use timeout/retry protection so a hanging CJ solve is logged and\nskipped instead of freezing the full sweep.\n"""

from __future__ import annotations

import argparse
import time
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterable, Tuple

import cantera as ct
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator

from sdtoolbox.postshock import CJspeed, PostShock_fr
from sdtoolbox.znd import zndsolve


# =============================================================================
# SETTINGS
# =============================================================================

MECH = "ffcm2_h2.yaml"
PHI = 1.0

FULL_T_VALUES = np.linspace(300.0, 1000.0, 50)
FULL_P_VALUES = np.linspace(0.1, 5.0, 50)

DEFAULT_TARGET_T = 351.5
DEFAULT_TARGET_P = 0.85

# Keep the same convention used in the earlier project scripts.
OVERDRIVE_FACTOR = 1.005
INITIAL_T_END = 5.0e-5
MAX_T_END = 2.0e-3
MAX_STEP = 1.0e-7
REL_TOL = 1.0e-6
ABS_TOL = 1.0e-10
ODE_METHOD = "BDF"

# Run each expensive SDToolbox grid case in an isolated child process.
# This prevents a single hanging CJspeed()/CJ_calc solve from freezing
# the entire manifold construction.
CASE_TIMEOUT_S = 20.0
MAX_CASE_RETRIES = 1

N_COMMON_X = 1200

PREFERRED_SPECIES = [
    "H2", "O2", "H2O", "OH", "H", "O", "HO2", "H2O2", "N2"
]

PROFILE_DIR = Path("znd_tp_profiles")
PLOT_DIR = Path("znd_tp_validation_plots")
MANIFEST_FILE = Path("znd_tp_manifest.csv")
WEIGHTS_FILE = Path("znd_bilinear_weights.csv")
ERROR_FILE = Path("znd_profile_error_summary.csv")
VALIDATION_NPZ = Path("znd_profile_validation.npz")


# =============================================================================
# HELPERS
# =============================================================================

def mixture_string(phi: float = PHI) -> str:
    """Stoichiometric H2-air at phi=1 is H2:2 O2:1 N2:3.76."""
    return f"H2:{2.0 * phi:.12g} O2:1 N2:3.76"


def case_filename(T_K: float, P_atm: float) -> Path:
    return PROFILE_DIR / f"znd_T{T_K:07.2f}_P{P_atm:05.2f}.npz"


def target_filename(T_K: float, P_atm: float) -> Path:
    return PROFILE_DIR / f"truth_target_T{T_K:07.2f}_P{P_atm:06.3f}.npz"


def _as_1d(a) -> np.ndarray:
    return np.asarray(a, dtype=float).reshape(-1)


def _clean_profile_x(x: np.ndarray, *arrays: np.ndarray):
    """Sort by x and remove duplicate x locations."""
    x = _as_1d(x)
    order = np.argsort(x)
    x = x[order]

    ordered = []
    for arr in arrays:
        arr = np.asarray(arr)
        if arr.ndim == 1:
            ordered.append(arr[order])
        elif arr.ndim == 2:
            ordered.append(arr[:, order])
        else:
            raise ValueError(f"Unsupported profile shape: {arr.shape}")

    x_unique, idx = np.unique(x, return_index=True)
    cleaned = [
        arr[idx] if arr.ndim == 1 else arr[:, idx]
        for arr in ordered
    ]
    return (x_unique, *cleaned)


def extract_species_matrix(znd: dict, n_species: int, n_points: int) -> np.ndarray:
    """Read znd['species'] while tolerating either common orientation."""
    if "species" not in znd:
        raise KeyError(
            "zndsolve output has no 'species' field. "
            f"Available keys: {sorted(znd.keys())}"
        )

    Y = np.asarray(znd["species"], dtype=float)
    if Y.shape == (n_species, n_points):
        return Y
    if Y.shape == (n_points, n_species):
        return Y.T

    raise ValueError(
        f"Unexpected znd['species'] shape {Y.shape}; expected "
        f"({n_species}, {n_points}) or ({n_points}, {n_species})."
    )


def reaction_peak_near_end(znd: dict) -> bool:
    """Detect a likely truncated reaction zone using the thermicity peak."""
    if "thermicity" not in znd:
        return False

    th = np.abs(_as_1d(znd["thermicity"]))
    if len(th) < 10 or not np.any(np.isfinite(th)):
        return False

    i_peak = int(np.nanargmax(th))
    return i_peak >= int(0.85 * (len(th) - 1))


# =============================================================================
# SDToolbox PROFILE GENERATION
# =============================================================================

def run_znd_profile(T1: float, P_atm: float, phi: float = PHI) -> Dict:
    """Run one CJ + ZND case and return the complete profile."""
    P1 = float(P_atm) * ct.one_atm
    q = mixture_string(phi)

    gas0 = ct.Solution(MECH)
    gas0.TPX = float(T1), P1, q
    species_names = list(gas0.species_names)

    cj_speed = float(CJspeed(P1, float(T1), q, MECH))
    U = OVERDRIVE_FACTOR * cj_speed

    t_end = INITIAL_T_END
    last_error = None

    while t_end <= MAX_T_END * (1.0 + 1e-12):
        try:
            gas_shock = PostShock_fr(U, P1, float(T1), q, MECH)
            znd = zndsolve(
                gas_shock,
                gas0,
                U,
                t_end=t_end,
                max_step=MAX_STEP,
                relTol=REL_TOL,
                absTol=ABS_TOL,
                advanced_output=True,
                Method=ODE_METHOD,
            )

            x = _as_1d(znd["distance"])
            T_prof = _as_1d(znd["T"])
            P_prof = _as_1d(znd["P"])
            Y = extract_species_matrix(znd, len(species_names), len(x))
            x, T_prof, P_prof, Y = _clean_profile_x(x, T_prof, P_prof, Y)

            if reaction_peak_near_end(znd) and t_end < MAX_T_END:
                t_end = min(2.0 * t_end, MAX_T_END)
                continue

            return {
                "T1_K": float(T1),
                "P1_atm": float(P_atm),
                "phi": float(phi),
                "cj_speed_m_s": cj_speed,
                "integration_speed_m_s": U,
                "t_end_s": float(t_end),
                "x_m": x,
                "T_K": T_prof,
                "P_Pa": P_prof,
                "Y": Y,
                "species_names": np.asarray(species_names, dtype="U"),
            }

        except Exception as exc:
            last_error = exc
            if t_end >= MAX_T_END:
                break
            t_end = min(2.0 * t_end, MAX_T_END)

    raise RuntimeError(
        f"ZND failed for T={T1} K, P={P_atm} atm. "
        f"Last error: {last_error!r}"
    )


def save_profile(profile: dict, filename: Path) -> None:
    filename.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        filename,
        T1_K=profile["T1_K"],
        P1_atm=profile["P1_atm"],
        phi=profile["phi"],
        cj_speed_m_s=profile["cj_speed_m_s"],
        integration_speed_m_s=profile["integration_speed_m_s"],
        t_end_s=profile["t_end_s"],
        x_m=profile["x_m"],
        T_K=profile["T_K"],
        P_Pa=profile["P_Pa"],
        Y=profile["Y"],
        species_names=profile["species_names"],
    )


def load_profile(filename: Path) -> dict:
    with np.load(filename, allow_pickle=False) as d:
        return {key: d[key] for key in d.files}


def run_case_timeout_safe(
    T1: float,
    P_atm: float,
    timeout_s: float = CASE_TIMEOUT_S,
    max_retries: int = MAX_CASE_RETRIES,
) -> tuple[str, str]:
    """Run one tabulated SDToolbox ZND case in an isolated subprocess."""
    outfile = case_filename(T1, P_atm)
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker-case",
        f"{float(T1):.12g}",
        f"{float(P_atm):.12g}",
    ]

    last_message = ""
    for attempt in range(max_retries + 1):
        try:
            completed = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=timeout_s,
            )
            if completed.returncode == 0 and outfile.exists():
                return "ok", completed.stdout.strip()

            last_message = (
                f"returncode={completed.returncode}; "
                f"stderr={completed.stderr[-2000:]!r}"
            )
        except subprocess.TimeoutExpired:
            last_message = (
                f"Timed out after {timeout_s:.1f} s "
                f"(attempt {attempt + 1}/{max_retries + 1})."
            )

        if attempt < max_retries:
            print(f"             retrying: {last_message}")

    return ("timeout" if "Timed out" in last_message else "failed"), last_message


def build_database(T_values: Iterable[float], P_values: Iterable[float]) -> pd.DataFrame:
    PROFILE_DIR.mkdir(exist_ok=True)
    T_values = np.asarray(list(T_values), dtype=float)
    P_values = np.asarray(list(P_values), dtype=float)
    total = len(T_values) * len(P_values)

    print("\n" + "=" * 78)
    print("BUILDING / REUSING STOICHIOMETRIC H2-AIR ZND T-P MANIFOLD")
    print("=" * 78)
    print(f"phi:             {PHI}")
    print(f"T nodes:         {len(T_values)}")
    print(f"P nodes:         {len(P_values)}")
    print(f"Total profiles:  {total}")

    rows = []
    case_id = 0

    for T1 in T_values:
        for P_atm in P_values:
            case_id += 1
            f = case_filename(T1, P_atm)

            if f.exists():
                d = load_profile(f)
                runtime = 0.0
                status = "cached"
                cj = float(d["cj_speed_m_s"])
                t_end = float(d["t_end_s"])
                npts = len(d["x_m"])
                xmax = float(np.max(d["x_m"]))
                print(f"[{case_id:3d}/{total}] cached  T={T1:7.2f} K  P={P_atm:5.2f} atm")
            else:
                print(f"[{case_id:3d}/{total}] running T={T1:7.2f} K  P={P_atm:5.2f} atm ...")
                t0 = time.perf_counter()
                status, message = run_case_timeout_safe(T1, P_atm)
                runtime = time.perf_counter() - t0

                if status == "ok" and f.exists():
                    d = load_profile(f)
                    cj = float(d["cj_speed_m_s"])
                    t_end = float(d["t_end_s"])
                    npts = len(d["x_m"])
                    xmax = float(np.max(d["x_m"]))
                    print(
                        f"             CJ={cj:.2f} m/s, xmax={1e3*xmax:.3f} mm, "
                        f"t_end={1e6*t_end:.1f} us, runtime={runtime:.2f} s"
                    )
                else:
                    cj = np.nan
                    t_end = np.nan
                    npts = 0
                    xmax = np.nan
                    print(f"             {status.upper()}: {message} Continuing.")

            rows.append({
                "case_id": case_id,
                "T_K": T1,
                "P_atm": P_atm,
                "phi": PHI,
                "status": status,
                "cj_speed_m_s": cj,
                "t_end_s": t_end,
                "n_profile_points": npts,
                "x_max_m": xmax,
                "profile_file": str(f),
                "runtime_s": runtime,
                "error_message": "" if status in ("ok", "cached") else message,
            })
            pd.DataFrame(rows).to_csv(MANIFEST_FILE, index=False)

    out = pd.DataFrame(rows)
    out.to_csv(MANIFEST_FILE, index=False)
    return out


# =============================================================================
# BILINEAR INTERPOLATION
# =============================================================================

def bracket(value: float, grid: np.ndarray) -> Tuple[float, float]:
    grid = np.asarray(grid, dtype=float)
    if not (grid[0] < value < grid[-1]):
        raise ValueError(
            f"Target {value} must lie strictly inside [{grid[0]}, {grid[-1]}]."
        )

    hi = int(np.searchsorted(grid, value, side="right"))
    lo = hi - 1
    if np.isclose(value, grid[lo]) or np.isclose(value, grid[hi]):
        raise ValueError(f"Target {value} is explicitly tabulated; choose an off-grid point.")
    return float(grid[lo]), float(grid[hi])


def bilinear_weights(T, P, T0, T1, P0, P1) -> Dict[str, float]:
    a = (T - T0) / (T1 - T0)
    b = (P - P0) / (P1 - P0)
    return {
        "T_fraction_a": a,
        "P_fraction_b": b,
        "w_T0_P0": (1.0 - a) * (1.0 - b),
        "w_T1_P0": a * (1.0 - b),
        "w_T0_P1": (1.0 - a) * b,
        "w_T1_P1": a * b,
    }


def resample_profile(profile: dict, x_common: np.ndarray) -> dict:
    x = _as_1d(profile["x_m"])
    T_prof = _as_1d(profile["T_K"])
    P_prof = _as_1d(profile["P_Pa"])
    Y = np.asarray(profile["Y"], dtype=float)

    return {
        "T_K": np.interp(x_common, x, T_prof),
        "P_Pa": np.interp(x_common, x, P_prof),
        "Y": np.vstack([
            np.interp(x_common, x, Y[k, :])
            for k in range(Y.shape[0])
        ]),
    }


def bilinear_interpolate_profiles(target_T, target_P, T_values, P_values):
    T0, T1 = bracket(target_T, T_values)
    P0, P1 = bracket(target_P, P_values)

    corner_files = {
        (T0, P0): case_filename(T0, P0),
        (T0, P1): case_filename(T0, P1),
        (T1, P0): case_filename(T1, P0),
        (T1, P1): case_filename(T1, P1),
    }

    missing = [key for key, filename in corner_files.items() if not filename.exists()]
    if missing:
        missing_text = ", ".join(
            f"(T={Tv:.1f} K, P={Pv:.2f} atm)" for Tv, Pv in missing
        )
        raise RuntimeError(
            "Cannot perform bilinear interpolation because required corner "
            f"profile(s) are missing: {missing_text}. "
            "Choose a different target cell or resolve the missing SDToolbox case."
        )

    corners = {
        key: load_profile(filename)
        for key, filename in corner_files.items()
    }

    x_max_common = min(float(np.max(c["x_m"])) for c in corners.values())
    x_common = np.linspace(0.0, x_max_common, N_COMMON_X)

    species_names = list(corners[(T0, P0)]["species_names"].astype(str))
    for c in corners.values():
        if list(c["species_names"].astype(str)) != species_names:
            raise ValueError("Species ordering differs between cached profiles.")

    sampled = {key: resample_profile(c, x_common) for key, c in corners.items()}

    T_cube = np.empty((2, 2, N_COMMON_X))
    P_cube = np.empty((2, 2, N_COMMON_X))
    Y_cube = np.empty((2, 2, len(species_names), N_COMMON_X))

    Ts = [T0, T1]
    Ps = [P0, P1]
    for i, Tv in enumerate(Ts):
        for j, Pv in enumerate(Ps):
            s = sampled[(Tv, Pv)]
            T_cube[i, j, :] = s["T_K"]
            P_cube[i, j, :] = s["P_Pa"]
            Y_cube[i, j, :, :] = s["Y"]

    interp_T = RegularGridInterpolator((np.array(Ts), np.array(Ps)), T_cube)
    interp_P = RegularGridInterpolator((np.array(Ts), np.array(Ps)), P_cube)
    interp_Y = RegularGridInterpolator((np.array(Ts), np.array(Ps)), Y_cube)

    query = np.array([target_T, target_P], dtype=float)
    prediction = {
        "x_m": x_common,
        "T_K": np.asarray(interp_T(query)).reshape(-1),
        "P_Pa": np.asarray(interp_P(query)).reshape(-1),
        "Y": np.asarray(interp_Y(query)).reshape(len(species_names), N_COMMON_X),
        "species_names": np.asarray(species_names, dtype="U"),
    }

    info = {
        "T0": T0,
        "T1": T1,
        "P0": P0,
        "P1": P1,
        **bilinear_weights(target_T, target_P, T0, T1, P0, P1),
    }
    return prediction, info


# =============================================================================
# VALIDATION
# =============================================================================

def get_or_run_target_truth(target_T: float, target_P: float) -> dict:
    f = target_filename(target_T, target_P)
    if f.exists():
        print(f"\nReusing independent target truth: {f}")
        return load_profile(f)

    print(f"\nRunning direct SDToolbox truth at T={target_T:.3f} K, P={target_P:.4f} atm ...")
    t0 = time.perf_counter()
    truth = run_znd_profile(target_T, target_P)
    save_profile(truth, f)
    print(f"Target truth complete in {time.perf_counter() - t0:.2f} s")
    return truth


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

    nrmse_pct = np.nan if denom <= 1e-30 else 100.0 * rmse / denom
    return {
        "RMSE": rmse,
        "MAE": mae,
        "max_abs_error": max_abs,
        "NRMSE_pct": nrmse_pct,
    }


def validate_prediction(pred: dict, truth_raw: dict):
    x = pred["x_m"]
    mask = x <= float(np.max(truth_raw["x_m"]))
    x = x[mask]

    truth = resample_profile(truth_raw, x)
    pred_T = pred["T_K"][mask]
    pred_P = pred["P_Pa"][mask]
    pred_Y = pred["Y"][:, mask]

    rows = []
    rows.append({"quantity": "Temperature", "units": "K", **field_metrics(pred_T, truth["T_K"], "range")})
    rows.append({"quantity": "Pressure", "units": "Pa", **field_metrics(pred_P, truth["P_Pa"], "range")})

    names = list(pred["species_names"].astype(str))
    for sp in PREFERRED_SPECIES:
        if sp not in names:
            continue
        k = names.index(sp)
        rows.append({
            "quantity": f"Y_{sp}",
            "units": "mass fraction",
            **field_metrics(pred_Y[k], truth["Y"][k], "peak"),
        })

    comparison = {
        "x_m": x,
        "pred_T_K": pred_T,
        "truth_T_K": truth["T_K"],
        "pred_P_Pa": pred_P,
        "truth_P_Pa": truth["P_Pa"],
        "pred_Y": pred_Y,
        "truth_Y": truth["Y"],
        "species_names": pred["species_names"],
    }
    return pd.DataFrame(rows), comparison


# =============================================================================
# OUTPUTS
# =============================================================================

def save_weight_table(target_T, target_P, info):
    rows = [
        {"corner": "(T0, P0)", "T_K": info["T0"], "P_atm": info["P0"], "weight": info["w_T0_P0"]},
        {"corner": "(T1, P0)", "T_K": info["T1"], "P_atm": info["P0"], "weight": info["w_T1_P0"]},
        {"corner": "(T0, P1)", "T_K": info["T0"], "P_atm": info["P1"], "weight": info["w_T0_P1"]},
        {"corner": "(T1, P1)", "T_K": info["T1"], "P_atm": info["P1"], "weight": info["w_T1_P1"]},
    ]
    df = pd.DataFrame(rows)
    df.insert(0, "target_T_K", target_T)
    df.insert(1, "target_P_atm", target_P)
    df.to_csv(WEIGHTS_FILE, index=False)
    return df


def make_plots(comparison: dict) -> None:
    PLOT_DIR.mkdir(exist_ok=True)
    x_mm = comparison["x_m"] * 1e3

    plt.figure(figsize=(7.2, 4.8))
    plt.plot(x_mm, comparison["truth_T_K"], label="Direct SDToolbox")
    plt.plot(x_mm, comparison["pred_T_K"], "--", label="Bilinear interpolation")
    plt.xlabel("Distance behind shock [mm]")
    plt.ylabel("Temperature [K]")
    plt.title("ZND temperature profile")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "temperature_profile.png", dpi=240)
    plt.close()

    plt.figure(figsize=(7.2, 4.8))
    plt.plot(x_mm, comparison["truth_P_Pa"] / 1e5, label="Direct SDToolbox")
    plt.plot(x_mm, comparison["pred_P_Pa"] / 1e5, "--", label="Bilinear interpolation")
    plt.xlabel("Distance behind shock [mm]")
    plt.ylabel("Pressure [bar]")
    plt.title("ZND pressure profile")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "pressure_profile.png", dpi=240)
    plt.close()

    names = list(comparison["species_names"].astype(str))
    for sp in PREFERRED_SPECIES:
        if sp not in names:
            continue
        k = names.index(sp)
        plt.figure(figsize=(7.2, 4.8))
        plt.plot(x_mm, comparison["truth_Y"][k], label="Direct SDToolbox")
        plt.plot(x_mm, comparison["pred_Y"][k], "--", label="Bilinear interpolation")
        plt.xlabel("Distance behind shock [mm]")
        plt.ylabel(f"{sp} mass fraction")
        plt.title(f"ZND {sp} profile")
        plt.legend()
        plt.grid(alpha=0.25)
        plt.tight_layout()
        plt.savefig(PLOT_DIR / f"species_{sp}_profile.png", dpi=240)
        plt.close()


def jax_smoke_test(comparison: dict) -> None:
    print("\n" + "=" * 78)
    print("JAX COMPATIBILITY CHECK")
    print("=" * 78)
    try:
        import jax
        import jax.numpy as jnp

        T_jax = jnp.asarray(comparison["pred_T_K"])
        P_jax = jnp.asarray(comparison["pred_P_Pa"])
        Y_jax = jnp.asarray(comparison["pred_Y"])

        print("JAX available:       yes")
        print(f"JAX backend:         {jax.default_backend()}")
        print(f"Temperature shape:   {T_jax.shape}")
        print(f"Pressure shape:      {P_jax.shape}")
        print(f"Species array shape: {Y_jax.shape}")
        print("SDToolbox-generated arrays can move into JAX successfully.")
    except ImportError:
        print("JAX is not installed in this environment yet.")
        print("That is fine for the SciPy bilinear baseline.")
        print("The saved NumPy arrays are already ready for later jnp.asarray(...).")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--quick",
        action="store_true",
        help="Only build the four surrounding nodes T={300,400}, P={0.8,0.9} atm.",
    )
    p.add_argument("--target-T", type=float, default=DEFAULT_TARGET_T)
    p.add_argument("--target-P", type=float, default=DEFAULT_TARGET_P)
    p.add_argument(
        "--worker-case",
        nargs=2,
        type=float,
        metavar=("T_K", "P_ATM"),
        help=argparse.SUPPRESS,
    )
    return p.parse_args()


def main():
    args = parse_args()

    if args.worker_case is not None:
        T1, P_atm = args.worker_case
        profile = run_znd_profile(T1, P_atm)
        save_profile(profile, case_filename(T1, P_atm))
        return

    print("=" * 78)
    print("STOICHIOMETRIC H2-AIR FULL-PROFILE ZND MANIFOLD BASELINE")
    print("=" * 78)
    print(f"Mechanism:       {MECH}")
    print(f"phi:             {PHI}")
    print(f"Target T:        {args.target_T:.3f} K")
    print(f"Target P:        {args.target_P:.4f} atm")
    print(f"Overdrive:       {OVERDRIVE_FACTOR:.5f} x CJ")
    print("Interpolation:   SciPy RegularGridInterpolator")
    print("Profile coord.:  physical distance x")

    if not Path(MECH).exists():
        raise FileNotFoundError(
            f"Could not find '{MECH}'. Put this script next to ffcm2_h2.yaml and sdtoolbox/."
        )

    gas_check = ct.Solution(MECH)
    print(f"Mechanism size:  {gas_check.n_species} species, {gas_check.n_reactions} reactions")

    if args.quick:
        T_values = np.array([300.0, 400.0])
        P_values = np.array([0.8, 0.9])
        if not (300.0 < args.target_T < 400.0 and 0.8 < args.target_P < 0.9):
            raise ValueError("--quick target must lie inside T=(300,400) K and P=(0.8,0.9) atm.")
        print("\nQUICK MODE: four surrounding tabulated states only.")
    else:
        T_values = FULL_T_VALUES.copy()
        P_values = FULL_P_VALUES.copy()
        print(f"\nFULL MODE: {len(T_values)} x {len(P_values)} = {len(T_values)*len(P_values)} profiles.")

    manifest = build_database(T_values, P_values)

    bad = manifest[manifest["status"].isin(["timeout", "failed"])]
    if not bad.empty:
        print("\n" + "=" * 78)
        print("MANIFOLD COMPLETION WARNING")
        print("=" * 78)
        print(f"Timed out: {(bad['status'] == 'timeout').sum()}")
        print(f"Failed:    {(bad['status'] == 'failed').sum()}")
        print(bad[["T_K", "P_atm", "status"]].to_string(index=False))
        print(
            "\nThese holes were skipped so the sweep could continue. "
            "Bilinear interpolation is valid only when all four corners "
            "of the requested T-P cell exist."
        )

    print("\n" + "=" * 78)
    print("BILINEAR PROFILE INTERPOLATION")
    print("=" * 78)

    pred, info = bilinear_interpolate_profiles(
        args.target_T, args.target_P, T_values, P_values
    )
    weights = save_weight_table(args.target_T, args.target_P, info)

    print(f"Target is bracketed by T={info['T0']:.1f}, {info['T1']:.1f} K and P={info['P0']:.2f}, {info['P1']:.2f} atm")
    print("\nBilinear weights:")
    with pd.option_context("display.float_format", "{:.6f}".format):
        print(weights[["corner", "T_K", "P_atm", "weight"]].to_string(index=False))
    print(f"Weight sum = {weights['weight'].sum():.12f}")

    truth = get_or_run_target_truth(args.target_T, args.target_P)
    errors, comparison = validate_prediction(pred, truth)
    errors.to_csv(ERROR_FILE, index=False)

    np.savez_compressed(
        VALIDATION_NPZ,
        x_m=comparison["x_m"],
        pred_T_K=comparison["pred_T_K"],
        truth_T_K=comparison["truth_T_K"],
        pred_P_Pa=comparison["pred_P_Pa"],
        truth_P_Pa=comparison["truth_P_Pa"],
        pred_Y=comparison["pred_Y"],
        truth_Y=comparison["truth_Y"],
        species_names=comparison["species_names"],
        target_T_K=args.target_T,
        target_P_atm=args.target_P,
    )

    make_plots(comparison)
    jax_smoke_test(comparison)

    print("\n" + "=" * 78)
    print("PROFILE ERROR SUMMARY")
    print("=" * 78)
    with pd.option_context("display.width", 140, "display.float_format", "{:.6g}".format):
        print(errors.to_string(index=False))

    print("\n" + "=" * 78)
    print("FILES SAVED")
    print("=" * 78)
    print(f"Manifold manifest:       {MANIFEST_FILE.resolve()}")
    print(f"Cached ZND profiles:     {PROFILE_DIR.resolve()}")
    print(f"Bilinear weights:        {WEIGHTS_FILE.resolve()}")
    print(f"Profile error summary:   {ERROR_FILE.resolve()}")
    print(f"Validation arrays:       {VALIDATION_NPZ.resolve()}")
    print(f"Validation plots:        {PLOT_DIR.resolve()}")

    print("\nInterpretation:")
    print(
        "This is the deliberately simple bilinear baseline on physical x. "
        "If error concentrates around a shifted induction/reaction front, "
        "that directly motivates profile alignment, compression, or a later ML/JAX representation."
    )


if __name__ == "__main__":
    main()