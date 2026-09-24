#!/usr/bin/env python3
"""
audit_low_T_low_P.py

Audit the low-temperature / low-pressure edge of a precomputed ZND manifold.

Purpose
-------
This script addresses the question:
    "What do the ZND solutions at the lower T1/P1 limits look like, and do
     the computed profiles behave smoothly and internally consistently?"

It DOES NOT determine experimental detonability. A converged ideal CJ/ZND
solution is not, by itself, proof that a self-sustaining detonation is
physically realizable in a particular geometry/experiment.

Expected NPZ schema
-------------------
T1_K, P1_atm, phi, cj_speed_m_s, integration_speed_m_s,
t_end_s, x_m, T_K, P_Pa, Y, species_names

The script only reads cached NPZ files. It does NOT run SDToolbox.

Default manifold directory:
    znd_tp_profiles/

Outputs:
    low_T_low_P_audit/
        low_pressure_summary.csv
        01_cj_speed_vs_pressure.png
        02_cj_speed_low_pressure_zoom.png
        03_temperature_profiles_300K.png
        04_pressure_profiles_300K.png
        05_major_species_profiles_300K.png
        06_radical_intermediate_profiles_300K.png
        07_characteristic_lengths_vs_pressure.png

Usage
-----
    conda activate sdtoolbox
    cd ~/Desktop/h2_detonation
    python audit_low_T_low_P.py
"""

from pathlib import Path
import csv
import warnings

import numpy as np
import matplotlib.pyplot as plt


PROFILE_DIR = Path("znd_tp_profiles")
OUTPUT_DIR = Path("low_T_low_P_audit")

TARGET_T_K = 300.0
SELECTED_P_ATM = (0.1, 0.2, 0.5, 1.0)

# Plot windows. None means use the full common domain.
TEMPERATURE_XMAX_M = None
PRESSURE_XMAX_M = None
SPECIES_XMAX_M = None

# Fraction-of-total-temperature-rise locations used as simple,
# reproducible profile-length diagnostics.
RISE_FRACTIONS = (0.10, 0.50, 0.90)

T_MATCH_TOL_K = 1.0e-6
P_MATCH_TOL_ATM = 1.0e-6


def scalar(d, key):
    """Return a scalar NPZ entry as float."""
    return float(np.asarray(d[key]).squeeze())


def decode_species_names(raw):
    """Convert stored species names to ordinary Python strings."""
    names = []
    for item in np.asarray(raw).ravel():
        if isinstance(item, bytes):
            names.append(item.decode("utf-8"))
        else:
            names.append(str(item))
    return names


def load_profile(path):
    """Load one cached ZND profile and perform basic consistency checks."""
    with np.load(path, allow_pickle=True) as d:
        required = [
            "T1_K", "P1_atm", "phi", "cj_speed_m_s",
            "integration_speed_m_s", "x_m", "T_K",
            "P_Pa", "Y", "species_names"
        ]
        missing = [k for k in required if k not in d.files]
        if missing:
            raise KeyError(f"{path.name}: missing keys {missing}")

        x = np.asarray(d["x_m"], dtype=float).squeeze()
        T = np.asarray(d["T_K"], dtype=float).squeeze()
        P = np.asarray(d["P_Pa"], dtype=float).squeeze()
        Y = np.asarray(d["Y"], dtype=float)
        species = decode_species_names(d["species_names"])

        if x.ndim != 1 or T.ndim != 1 or P.ndim != 1:
            raise ValueError(f"{path.name}: x/T/P must be 1-D")
        if not (len(x) == len(T) == len(P)):
            raise ValueError(f"{path.name}: x/T/P lengths do not match")

        # Accept either [n_x, n_species] or [n_species, n_x].
        if Y.ndim != 2:
            raise ValueError(f"{path.name}: Y must be 2-D")
        if Y.shape == (len(x), len(species)):
            pass
        elif Y.shape == (len(species), len(x)):
            Y = Y.T
        else:
            raise ValueError(
                f"{path.name}: Y shape {Y.shape} incompatible with "
                f"{len(x)} x-points and {len(species)} species"
            )

        out = {
            "path": path,
            "T1_K": scalar(d, "T1_K"),
            "P1_atm": scalar(d, "P1_atm"),
            "phi": scalar(d, "phi"),
            "cj_speed_m_s": scalar(d, "cj_speed_m_s"),
            "integration_speed_m_s": scalar(d, "integration_speed_m_s"),
            "t_end_s": scalar(d, "t_end_s") if "t_end_s" in d.files else np.nan,
            "x_m": x,
            "T_K": T,
            "P_Pa": P,
            "Y": Y,
            "species_names": species,
        }

    # Basic numerical sanity flags.
    out["x_monotonic"] = bool(np.all(np.diff(x) >= 0.0))
    out["finite_all"] = bool(
        np.all(np.isfinite(x))
        and np.all(np.isfinite(T))
        and np.all(np.isfinite(P))
        and np.all(np.isfinite(Y))
    )
    out["min_Y"] = float(np.nanmin(Y))
    out["max_species_sum_error"] = float(
        np.nanmax(np.abs(np.sum(Y, axis=1) - 1.0))
    )
    return out


def discover_temperature_row():
    """Find all cached profiles whose T1 matches TARGET_T_K."""
    if not PROFILE_DIR.exists():
        raise FileNotFoundError(
            f"Could not find {PROFILE_DIR.resolve()}\n"
            "Run this script from the project directory containing "
            "'znd_tp_profiles/'."
        )

    profiles = []
    skipped = []

    for path in sorted(PROFILE_DIR.glob("*.npz")):
        try:
            p = load_profile(path)
        except Exception as exc:
            skipped.append((path.name, str(exc)))
            continue

        if np.isclose(p["T1_K"], TARGET_T_K, atol=T_MATCH_TOL_K, rtol=0.0):
            profiles.append(p)

    profiles.sort(key=lambda p: p["P1_atm"])

    if not profiles:
        raise RuntimeError(
            f"No cached profiles found at T1 = {TARGET_T_K:g} K "
            f"in {PROFILE_DIR.resolve()}."
        )

    return profiles, skipped


def nearest_profile(profiles, target_p):
    """Return exact pressure match when available; otherwise None."""
    for p in profiles:
        if np.isclose(
            p["P1_atm"], target_p, atol=P_MATCH_TOL_ATM, rtol=0.0
        ):
            return p
    return None


def crossing_location(x, y, target):
    """
    First x where y crosses target, with linear interpolation.

    Works for either increasing or decreasing local crossings.
    Returns NaN if no crossing occurs.
    """
    y0 = y[:-1] - target
    y1 = y[1:] - target
    idx = np.where((y0 == 0.0) | (y0 * y1 <= 0.0))[0]

    if len(idx) == 0:
        return np.nan

    i = int(idx[0])
    if y[i] == target:
        return float(x[i])

    dy = y[i + 1] - y[i]
    if dy == 0.0:
        return float(x[i])

    frac = (target - y[i]) / dy
    return float(x[i] + frac * (x[i + 1] - x[i]))


def temperature_rise_locations(p):
    """
    Compute x10/x50/x90 based on the rise from T(x=0) to T(x=end).

    These are descriptive profile metrics, not formal induction lengths.
    """
    x = p["x_m"]
    T = p["T_K"]
    T_start = float(T[0])
    T_end = float(T[-1])
    delta = T_end - T_start

    result = {}
    for frac in RISE_FRACTIONS:
        target = T_start + frac * delta
        result[f"x{int(100*frac):02d}_T_m"] = crossing_location(x, T, target)

    result["T_start_K"] = T_start
    result["T_end_K"] = T_end
    result["delta_T_K"] = delta
    return result


def summarize_profile(p):
    rise = temperature_rise_locations(p)
    row = {
        "T1_K": p["T1_K"],
        "P1_atm": p["P1_atm"],
        "phi": p["phi"],
        "cj_speed_m_s": p["cj_speed_m_s"],
        "integration_speed_m_s": p["integration_speed_m_s"],
        "x_end_m": float(p["x_m"][-1]),
        "T_min_K": float(np.min(p["T_K"])),
        "T_max_K": float(np.max(p["T_K"])),
        "P_min_Pa": float(np.min(p["P_Pa"])),
        "P_max_Pa": float(np.max(p["P_Pa"])),
        "x_monotonic": p["x_monotonic"],
        "finite_all": p["finite_all"],
        "min_mass_fraction": p["min_Y"],
        "max_species_sum_error": p["max_species_sum_error"],
    }
    row.update(rise)
    return row


def write_summary(rows):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / "low_pressure_summary.csv"
    fields = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return path


def savefig(name):
    path = OUTPUT_DIR / name
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    return path


def plot_cj_all(profiles):
    P = np.array([p["P1_atm"] for p in profiles])
    U = np.array([p["cj_speed_m_s"] for p in profiles])

    plt.figure(figsize=(8, 5))
    plt.plot(P, U, marker="o", markersize=3)
    plt.xlabel(r"Initial pressure $P_1$ [atm]")
    plt.ylabel("CJ speed [m/s]")
    plt.title(rf"CJ speed along the $T_1={TARGET_T_K:g}$ K manifold row")
    plt.grid(alpha=0.25)
    return savefig("01_cj_speed_vs_pressure.png")


def plot_cj_zoom(profiles):
    low = [p for p in profiles if p["P1_atm"] <= 1.0 + P_MATCH_TOL_ATM]
    P = np.array([p["P1_atm"] for p in low])
    U = np.array([p["cj_speed_m_s"] for p in low])

    plt.figure(figsize=(8, 5))
    plt.plot(P, U, marker="o")
    plt.xlabel(r"Initial pressure $P_1$ [atm]")
    plt.ylabel("CJ speed [m/s]")
    plt.title(
        rf"Low-pressure CJ-speed behavior at $T_1={TARGET_T_K:g}$ K"
    )
    plt.grid(alpha=0.25)
    return savefig("02_cj_speed_low_pressure_zoom.png")


def plot_thermo(selected, quantity):
    plt.figure(figsize=(8, 5.5))

    for p in selected:
        x_mm = 1000.0 * p["x_m"]
        if quantity == "T":
            y = p["T_K"]
            ylabel = "Temperature [K]"
            title = rf"ZND temperature profiles at $T_1={TARGET_T_K:g}$ K"
            xmax = TEMPERATURE_XMAX_M
            filename = "03_temperature_profiles_300K.png"
        else:
            y = p["P_Pa"] / 1.0e5
            ylabel = "Pressure [bar]"
            title = rf"ZND pressure profiles at $T_1={TARGET_T_K:g}$ K"
            xmax = PRESSURE_XMAX_M
            filename = "04_pressure_profiles_300K.png"

        plt.plot(x_mm, y, label=rf"$P_1={p['P1_atm']:g}$ atm")

    if xmax is not None:
        plt.xlim(0.0, 1000.0 * xmax)
    plt.xlabel("Distance behind shock [mm]")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.25)
    return savefig(filename)


def species_index(p, name):
    try:
        return p["species_names"].index(name)
    except ValueError:
        return None


def plot_species_groups(selected, species_group, filename, title):
    """
    One figure with one panel per species.

    Different pressures are overlaid within each panel.
    """
    available = [
        s for s in species_group
        if any(species_index(p, s) is not None for p in selected)
    ]
    if not available:
        warnings.warn(f"No requested species available for {filename}")
        return None

    n = len(available)
    fig, axes = plt.subplots(n, 1, figsize=(8, 2.8*n), sharex=True)
    if n == 1:
        axes = [axes]

    for ax, s in zip(axes, available):
        for p in selected:
            idx = species_index(p, s)
            if idx is None:
                continue
            x_mm = 1000.0 * p["x_m"]
            ax.plot(
                x_mm,
                p["Y"][:, idx],
                label=rf"$P_1={p['P1_atm']:g}$ atm"
            )
        ax.set_ylabel(rf"$Y_{{{s}}}$")
        ax.grid(alpha=0.25)

    if SPECIES_XMAX_M is not None:
        axes[-1].set_xlim(0.0, 1000.0 * SPECIES_XMAX_M)

    axes[-1].set_xlabel("Distance behind shock [mm]")
    axes[0].legend(ncol=2)
    fig.suptitle(title, y=1.002)
    path = OUTPUT_DIR / filename
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_characteristic_lengths(rows):
    P = np.array([r["P1_atm"] for r in rows])
    x10 = 1000.0 * np.array([r["x10_T_m"] for r in rows])
    x50 = 1000.0 * np.array([r["x50_T_m"] for r in rows])
    x90 = 1000.0 * np.array([r["x90_T_m"] for r in rows])

    plt.figure(figsize=(8, 5.5))
    plt.plot(P, x10, marker="o", markersize=3, label="10% temperature rise")
    plt.plot(P, x50, marker="o", markersize=3, label="50% temperature rise")
    plt.plot(P, x90, marker="o", markersize=3, label="90% temperature rise")
    plt.xlabel(r"Initial pressure $P_1$ [atm]")
    plt.ylabel("Distance behind shock [mm]")
    plt.title(
        rf"Temperature-rise length scales along $T_1={TARGET_T_K:g}$ K row"
    )
    plt.legend()
    plt.grid(alpha=0.25)
    return savefig("07_characteristic_lengths_vs_pressure.png")


def print_selected_summary(selected):
    print("\nSelected low-pressure states")
    print("-" * 88)
    print(
        f"{'P1 [atm]':>9} {'CJ [m/s]':>11} {'T(0) [K]':>11} "
        f"{'T(end) [K]':>12} {'Pmax [bar]':>12} {'x90 [mm]':>10}"
    )
    print("-" * 88)
    for p in selected:
        r = summarize_profile(p)
        print(
            f"{r['P1_atm']:9.3f} "
            f"{r['cj_speed_m_s']:11.2f} "
            f"{r['T_start_K']:11.2f} "
            f"{r['T_end_K']:12.2f} "
            f"{r['P_max_Pa']/1e5:12.3f} "
            f"{1000*r['x90_T_m']:10.4f}"
        )


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("LOW-T / LOW-P ZND MANIFOLD AUDIT")
    print("=" * 72)
    print(f"Profile directory : {PROFILE_DIR.resolve()}")
    print(f"Target T1         : {TARGET_T_K:g} K")
    print("No SDToolbox calculations will be run.\n")

    profiles, skipped = discover_temperature_row()
    rows = [summarize_profile(p) for p in profiles]

    print(
        f"Loaded {len(profiles)} cached profiles at "
        f"T1 = {TARGET_T_K:g} K."
    )
    print(
        f"Pressure range: {profiles[0]['P1_atm']:g} to "
        f"{profiles[-1]['P1_atm']:g} atm"
    )

    if skipped:
        print(
            f"Note: {len(skipped)} NPZ file(s) elsewhere in the directory "
            "could not be parsed with the expected schema."
        )

    selected = []
    for target_p in SELECTED_P_ATM:
        p = nearest_profile(profiles, target_p)
        if p is None:
            warnings.warn(
                f"No exact cached state found at T1={TARGET_T_K:g} K, "
                f"P1={target_p:g} atm."
            )
        else:
            selected.append(p)

    if not selected:
        raise RuntimeError("None of the selected low-pressure states were found.")

    print_selected_summary(selected)

    summary_path = write_summary(rows)

    outputs = [
        plot_cj_all(profiles),
        plot_cj_zoom(profiles),
        plot_thermo(selected, "T"),
        plot_thermo(selected, "P"),
        plot_species_groups(
            selected,
            ["H2", "O2", "H2O"],
            "05_major_species_profiles_300K.png",
            rf"Major-species ZND profiles at $T_1={TARGET_T_K:g}$ K",
        ),
        plot_species_groups(
            selected,
            ["H", "O", "OH", "HO2", "H2O2"],
            "06_radical_intermediate_profiles_300K.png",
            rf"Radical/intermediate profiles at $T_1={TARGET_T_K:g}$ K",
        ),
        plot_characteristic_lengths(rows),
    ]

    print("\nBasic numerical sanity checks across the 300 K row")
    print("-" * 72)
    print(
        "All x grids monotonic :",
        all(p["x_monotonic"] for p in profiles)
    )
    print(
        "All values finite     :",
        all(p["finite_all"] for p in profiles)
    )
    print(
        "Minimum Y             :",
        f"{min(p['min_Y'] for p in profiles):.6e}"
    )
    print(
        "Max |sum(Y)-1|        :",
        f"{max(p['max_species_sum_error'] for p in profiles):.6e}"
    )

    print("\nSaved:")
    print(f"  {summary_path}")
    for path in outputs:
        if path is not None:
            print(f"  {path}")

    print("\nInterpretation reminder:")
    print(
        "  These diagnostics test numerical/profile consistency of the cached "
        "ideal CJ/ZND solutions."
    )
    print(
        "  They do NOT, by themselves, establish experimental or practical "
        "detonability."
    )


if __name__ == "__main__":
    main()
