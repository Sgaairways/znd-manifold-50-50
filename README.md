# ZND Manifold — 50 × 50

This repository contains the refined **50 × 50 precomputed ZND profile manifold** for stoichiometric H₂–air, together with interpolation validation, JAX lookup work, and full-profile visualization.

The manifold is parameterized by initial temperature and pressure:

[
ZND(T_1, P_1) ightarrow {T(x), p(x), Y_i(x)}
]

where (x) is distance behind the shock.

## Current manifold

- **Temperature grid:** 50 points from 300 to 1000 K
- **Pressure grid:** 50 points from 0.1 to 5.0 atm
- **Requested states:** 2500
- **Successful precomputed profiles:** 2497
- **Timed-out states:** 3
- **Mixture:** stoichiometric H₂–air, (phi = 1)

The three timed-out states are treated as numerical/tool timeouts rather than physically invalid states.

## Why 50 × 50?

The earlier baseline used an **8 × 50** grid. Refining the temperature sampling to 50 points while keeping the same 50 pressure points substantially reduced full-profile bilinear interpolation error across the nine tested off-grid validation states.

### Aggregate validation

| Quantity | 8 × 50 mean NRMSE | 50 × 50 mean NRMSE |
|---|---:|---:|
| Temperature | 0.113727% | 0.028877% |
| Pressure | 1.669653% | 0.043397% |
| H₂ | 0.145745% | 0.052173% |
| O₂ | 0.139578% | 0.049045% |
| H₂O | 0.136586% | 0.046162% |
| OH | 0.158660% | 0.063558% |
| H | 0.409724% | 0.110893% |
| O | 0.291064% | 0.091815% |
| HO₂ | 0.395153% | 0.170375% |
| H₂O₂ | 0.266157% | 0.098059% |

Pressure mean NRMSE decreased from **1.67% to 0.043%**, a reduction of about **97.4%**.

These nine validation states show a strong improvement, but they do **not** establish that 50 × 50 is an optimal sampling density or that uniform sampling is the most efficient strategy.

## Validation approach

For each off-grid query point:

1. The four surrounding manifold states are identified.
2. Their full ZND profiles are resampled onto a common physical-(x) grid.
3. Bilinear interpolation is performed in (T_1) and (P_1).
4. The reconstructed profiles are compared against a direct SDToolbox calculation.

Validation includes:

- Temperature (T(x))
- Pressure (p(x))
- Species mass-fraction profiles (Y_i(x))

The full-profile overlays show that the bilinear reconstruction closely follows the direct SDToolbox solution, including the rapid near-shock structure of radical and intermediate species.

## JAX implementation

A JAX version of the bilinear interpolation was also validated against the SciPy implementation.

Across the tested cases, the JAX and SciPy interpolated profiles agreed to floating-point precision.

For the warmed interpolation-only kernel on CPU:

- **JAX:** ~7.36 µs/query
- **SciPy:** ~49.09 µs/query
- **Measured speedup:** ~6.67× 

This timing excludes disk I/O, manifold generation, corner loading, and physical-(x) resampling.

The purpose of the JAX path is runtime architecture, JIT compilation, batching/vectorization, and eventual integration with downstream thermal and optimization workflows.

## Profile-shape visualization

The precomputed manifold is also used directly to visualize how complete ZND profiles vary across pressure at fixed initial temperature.

Current slices include:

- (T_1 = 300) K: (T(x)) across all pressure states
- (T_1 = 300) K: (p(x)) across all pressure states
- (T_1 = 500) K: (T(x)) across all pressure states
- (T_1 = 500) K: (p(x)) across all pressure states

These figures are generated entirely from the **precomputed manifold**; no new SDToolbox calculations are required.

The temperature maps are displayed over a 0–5 mm window and the pressure maps over a 0–1 mm window to expose the near-shock and reaction-zone structure more clearly.

## Research direction

The current progression is:

[
8 	imes 50 	ext{ baseline}
ightarrow
50 	imes 50 	ext{ refinement}
ightarrow
	ext{sampling-strategy study}
ightarrow
	ext{fast JAX lookup}
]

The next question is whether **irregular, random, nonuniform, or adaptive sampling** can reconstruct the same full ZND profile manifold with fewer expensive SDToolbox solutions.

Longer term, the goal is to provide fast profile lookup for downstream heat-transfer and cooling optimization:

[
(T_1,P_1,phi)
ightarrow
	ext{ZND profile lookup}
ightarrow
T(x),p(x),Y_i(x)
ightarrow
	ext{thermal boundary conditions}
ightarrow
	ext{heat-transfer / cooling model}
ightarrow
	ext{optimization}
]

## Related repository

The original baseline study is preserved separately in:

[**Sgaairways/znd-manifold-8-50**](https://github.com/Sgaairways/znd-manifold-8-50)
