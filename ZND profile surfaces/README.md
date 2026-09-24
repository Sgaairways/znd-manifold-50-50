# ZND Profile Surfaces

This directory visualizes **slices through the precomputed 50 × 50 ZND manifold**. The figures show how complete temperature and pressure profiles vary with initial pressure at fixed initial temperature.

No new SDToolbox calculations are performed to create these plots. The plotting script reads the ZND profiles that were already computed and stored as part of the manifold.

## Requested manifold slices

Four profile maps are included:

| Fixed initial temperature | Quantity shown | Displayed distance behind shock |
|---:|---|---:|
| 300 K | Temperature, T(x) | 0–5 mm |
| 300 K | Pressure, p(x) | 0–1 mm |
| 500 K | Temperature, T(x) | 0–5 mm |
| 500 K | Pressure, p(x) | 0–1 mm |

For each map:

- **x-axis:** physical distance behind the shock
- **y-axis:** initial pressure P₁ from 0.1 to 5.0 atm
- **color:** local ZND temperature or pressure
- **data source:** the precomputed 50 × 50 manifold

Both 300 K and 500 K are exact temperature nodes in the 50-point temperature grid, so these slices do not require interpolation in T₁.

## Physical-x treatment

Individual ZND solutions have different physical distance extents. For each fixed-temperature slice, the cached profiles are therefore resampled onto a common physical-`x` grid.

The common plotting extent is determined from the shortest available profile in that slice:

**x_max, common = minimum x_max across the pressure profiles at the selected T₁**

This preserves **physical distance behind the shock** rather than normalizing each profile by its individual length.

The displayed figures intentionally zoom into the region where the profile structure is most visible:

- Temperature maps: **0–5 mm**
- Pressure maps: **0–1 mm**

The pressure window is shorter because the strongest pressure variation is concentrated much closer to the shock in these manifold slices.

## What the plots show

The temperature maps show a clear curved transition across the `(x, P₁)` plane. Visually, the transition extends farther downstream at lower initial pressure and moves closer to the shock as initial pressure increases.

The pressure maps show that the strongest pressure evolution is concentrated very close to the shock compared with the temperature evolution.

The 300 K and 500 K slices also show visibly different profile fields, demonstrating that the full manifold contains structure in both the initial-temperature and initial-pressure dimensions.

These observations are descriptions of the plotted manifold structure. The figures alone are not intended to establish a specific physical mechanism or prove that a particular nonuniform sampling strategy is optimal.

## Files

- `plot_znd_profile_surfaces.py` — reads the cached manifold and generates the profile maps
- `T1_0300K_temperature_profile_map.png` — temperature field at T₁ = 300 K
- `T1_0300K_pressure_profile_map.png` — pressure field at T₁ = 300 K
- `T1_0500K_temperature_profile_map.png` — temperature field at T₁ = 500 K
- `T1_0500K_pressure_profile_map.png` — pressure field at T₁ = 500 K

## Why these surfaces matter

The scalar interpolation-error metrics quantify reconstruction accuracy, while these surfaces expose the **shape of the underlying full-profile manifold**.

Together, they help separate two questions:

1. **Can an off-grid ZND profile be reconstructed accurately from nearby precomputed states?**
2. **How does the profile itself vary across the operating parameter space?**

The first is addressed by the validation results. These profile surfaces provide a direct visual view of the second and motivate the next stage of studying how efficiently the parameter space should be sampled.
