#!/usr/bin/env python3
"""
Plot ZND profile-shape maps from the existing 50 x 50 cached manifold.

Creates four figures requested by Ryan:
  1) T1 = 300 K: x vs P1, color = T(x)
  2) T1 = 300 K: x vs P1, color = p(x)
  3) T1 = 500 K: x vs P1, color = T(x)
  4) T1 = 500 K: x vs P1, color = p(x)

No SDToolbox calculations are performed. This script only reads cached .npz files.

For each fixed T1, all available pressure profiles are resampled onto a common
physical-distance grid:
    x_common in [0, min(x_max over available P1 profiles)]
This preserves physical distance behind the shock rather than normalizing x.

Expected project layout:
    h2_detonation/
      plot_znd_profile_surfaces.py
      znd_tp_profiles/
        *.npz

Output:
    profile_shape_figures/
"""

from pathlib import Path
import re
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
PROFILE_DIR = ROOT / "znd_tp_profiles"
OUT_DIR = ROOT / "profile_shape_figures"
OUT_DIR.mkdir(exist_ok=True)

TARGET_TEMPS = [300.0, 500.0]
N_COMMON_X = 1200

# Pressure grid used by the 50 x 50 manifold.
PGRID = np.linspace(0.1, 5.0, 50)


def save(fig, name):
    fig.savefig(OUT_DIR / f"{name}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def first_key(npz, candidates):
    """Return the first matching array key from a cached NPZ file."""
    keys = list(npz.files)

    # Exact match first.
    for candidate in candidates:
        if candidate in keys:
            return candidate

    # Case-insensitive exact match.
    lower = {k.lower(): k for k in keys}
    for candidate in candidates:
        if candidate.lower() in lower:
            return lower[candidate.lower()]

    return None


def scalar_from_npz(npz, candidates):
    key = first_key(npz, candidates)
    if key is None:
        return None
    arr = np.asarray(npz[key])
    if arr.size != 1:
        return None
    return float(arr.reshape(-1)[0])


def parse_TP_from_filename(path):
    """
    Fallback parser for filenames containing patterns such as:
      T0300.00_P00.100
      T300_P0.1
      T0300.000_P05.000
    """
    name = path.stem

    mt = re.search(r"(?:^|[_-])T(?:1)?[_-]?0*([0-9]+(?:\.[0-9]+)?)", name, re.I)
    mp = re.search(r"(?:^|[_-])P(?:1)?[_-]?0*([0-9]+(?:\.[0-9]+)?)", name, re.I)

    T = float(mt.group(1)) if mt else None
    P = float(mp.group(1)) if mp else None
    return T, P


def identify_state(path):
    """Read T1/P1 metadata when present; otherwise infer it from filename."""
    try:
        with np.load(path, allow_pickle=True) as d:
            T = scalar_from_npz(
                d, ["T1", "T1_K", "T_K", "initial_temperature",
                    "initial_temperature_K", "T_initial"]
            )
            P = scalar_from_npz(
                d, ["P1_atm", "P_atm", "initial_pressure_atm",
                    "P1", "initial_pressure"]
            )

            # If a stored pressure scalar looks like Pa, convert to atm.
            if P is not None and P > 100.0:
                P = P / 101325.0
    except Exception:
        T = P = None

    Tf, Pf = parse_TP_from_filename(path)
    if T is None:
        T = Tf
    if P is None:
        P = Pf

    return T, P


def load_profile(path):
    """
    Load physical x, temperature profile, and pressure profile.

    Supports several likely key names so it can work with the existing cache
    without requiring a specific NPZ schema.
    """
    with np.load(path, allow_pickle=True) as d: 
        xkey = first_key(d, ["x_m", "x", "distance", "distance_m"])
        tkey = first_key(d, ["T_K", "T", "temperature", "temperature_K", "T_profile"])
        pkey = first_key(d, ["P_Pa", "P", "pressure", "pressure_Pa", "P_profile"])

        if xkey is None or tkey is None or pkey is None:
            raise KeyError(
                f"{path.name}: could not find x/T/P arrays. "
                f"Available keys: {list(d.files)}"
            )

        x = np.asarray(d[xkey], dtype=float).squeeze()
        T = np.asarray(d[tkey], dtype=float).squeeze()
        P = np.asarray(d[pkey], dtype=float).squeeze()

    if x.ndim != 1 or T.ndim != 1 or P.ndim != 1:
        raise ValueError(f"{path.name}: x, T, and P must be 1-D arrays.")
    if not (len(x) == len(T) == len(P)):
        raise ValueError(f"{path.name}: x/T/P lengths do not match.")

    good = np.isfinite(x) & np.isfinite(T) & np.isfinite(P)
    x, T, P = x[good], T[good], P[good]

    order = np.argsort(x)
    x, T, P = x[order], T[order], P[order]

    # np.interp requires increasing coordinates; remove duplicate x values.
    x_unique, idx = np.unique(x, return_index=True)
    T = T[idx]
    P = P[idx]
    x = x_unique

    return x, T, P


def discover_profiles(target_T, tol=1e-3):
    """Find all cached profiles corresponding to one fixed initial temperature."""
    if not PROFILE_DIR.exists():
        raise FileNotFoundError(f"Profile directory not found: {PROFILE_DIR}")

    matches = []
    skipped = []

    for path in sorted(PROFILE_DIR.glob("*.npz")):
        # Ignore independent validation/truth files.
        if "truth" in path.name.lower() or "target" in path.name.lower():
            continue

        T1, P1 = identify_state(path)
        if T1 is None or P1 is None:
            continue

        if np.isclose(T1, target_T, atol=tol, rtol=0.0):
            try:
                x, T, P = load_profile(path)
                if len(x) < 2:
                    skipped.append((path.name, "too few x points"))
                    continue
                matches.append({
                    "path": path,
                    "T1": T1,
                    "P1": P1,
                    "x": x,
                    "T": T,
                    "P": P,
                })
            except Exception as exc:
                skipped.append((path.name, str(exc)))

    matches.sort(key=lambda r: r["P1"])
    return matches, skipped


def build_common_fields(profiles):
    """
    Resample all profiles at a fixed T1 onto one common physical-x grid.

    The common extent ends at the shortest available profile:
        x_max_common = min(max(x_i))
    """
    if not profiles:
        raise ValueError("No profiles supplied.")

    x_start = max(float(p["x"][0]) for p in profiles)
    x_end = min(float(p["x"][-1]) for p in profiles)

    if x_end <= x_start:
        raise ValueError("Profiles do not share a usable physical-x interval.")

    x_common = np.linspace(x_start, x_end, N_COMMON_X)
    pressures_atm = np.array([p["P1"] for p in profiles], dtype=float)

    T_field = np.vstack([
        np.interp(x_common, p["x"], p["T"]) for p in profiles
    ])
    P_field = np.vstack([
        np.interp(x_common, p["x"], p["P"]) for p in profiles
    ])

    return x_common, pressures_atm, T_field, P_field


def plot_field(x, p1, field, fixed_T, quantity):
    """
    Plot x vs initial pressure with profile value represented by color.
    """
    fig, ax = plt.subplots(figsize=(10.5, 6.5))

    if quantity == "temperature":
        Z = field
        label = r"ZND temperature, $T(x)$ [K]"
        title = rf"ZND temperature-profile shape at $T_1={fixed_T:.0f}$ K"
        suffix = "temperature"
    elif quantity == "pressure":
        # Display ZND pressure in atm for an immediately readable color scale.
        Z = field / 101325.0
        label = r"ZND pressure, $p(x)$ [atm]"
        title = rf"ZND pressure-profile shape at $T_1={fixed_T:.0f}$ K"
        suffix = "pressure"
    else:
        raise ValueError(quantity)

    mesh = ax.pcolormesh(x, p1, Z, shading="auto")
    cb = fig.colorbar(mesh, ax=ax)
    cb.set_label(label)

    ax.set_xlabel(r"Distance behind shock, $x$ [m]")
    ax.set_ylabel(r"Initial pressure, $P_1$ [atm]")
    ax.set_title(title)

    if quantity == "temperature":
        ax.set_xlim(0.0, 0.005)
    elif quantity == "pressure":
        ax.set_xlim(0.0, 0.001)

    ax.text(
        0.02, 0.98,
        f"{len(p1)} cached pressure profiles\n"
        f"common physical-x extent: {x[-1]:.3e} m",
        transform=ax.transAxes,
        va="top",
        fontsize=10,
        bbox=dict(facecolor="white", alpha=0.75, edgecolor="none")
    )

    fig.tight_layout()
    save(fig, f"T1_{int(round(fixed_T)):04d}K_{suffix}_profile_map")


def process_temperature(target_T):
    profiles, skipped = discover_profiles(target_T)

    print("\n" + "=" * 78)
    print(f"T1 = {target_T:.1f} K")
    print("=" * 78)
    print(f"Loaded profiles: {len(profiles)}")

    if skipped:
        print(f"Skipped files:   {len(skipped)}")
        for name, reason in skipped:
            print(f"  {name}: {reason}")

    if not profiles:
        raise RuntimeError(
            f"No cached profiles found for T1={target_T:.1f} K in {PROFILE_DIR}"
        )

    found_p = np.array([p["P1"] for p in profiles])
    print(f"P1 range:        {found_p.min():.3f} to {found_p.max():.3f} atm")

    # Helpful warning if any expected pressure states are absent.
    missing_p = [
        p for p in PGRID
        if not np.any(np.isclose(found_p, p, atol=1e-3, rtol=0.0))
    ]
    if missing_p:
        print("Missing expected P1 states:")
        print("  " + ", ".join(f"{p:.2f}" for p in missing_p))
    else:
        print("Pressure sweep:  all 50 expected states available")

    x, p1, T_field, P_field = build_common_fields(profiles)

    print(f"Common x range:  {x[0]:.6e} to {x[-1]:.6e} m")
    print(f"Field shape:     {T_field.shape}")

    plot_field(x, p1, T_field, target_T, "temperature")
    plot_field(x, p1, P_field, target_T, "pressure")


def main():
    print("=" * 78)
    print("ZND PROFILE-SHAPE MAPS FROM CACHED 50 x 50 MANIFOLD")
    print("=" * 78)
    print(f"Profile directory: {PROFILE_DIR}")
    print(f"Output directory:  {OUT_DIR}")
    print("No SDToolbox calculations will be run.")

    for T1 in TARGET_TEMPS:
        process_temperature(T1)

    print("\n" + "=" * 78)
    print("DONE")
    print("=" * 78)
    print("Created four Ryan/Chris profile-shape figures:")
    print("  T1=300 K: temperature and pressure")
    print("  T1=500 K: temperature and pressure")
    print(f"Saved as PNG + SVG in: {OUT_DIR}")


if __name__ == "__main__":
    main()
