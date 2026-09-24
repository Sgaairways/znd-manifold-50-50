# JAX Bilinear ZND Lookup — 50 × 50

This directory contains the **JAX implementation and validation** of bilinear full-profile lookup from the 50 × 50 precomputed ZND manifold.

The purpose of the JAX implementation is not to change the interpolation model or improve its physical accuracy. It reproduces the same bilinear interpolation used in the SciPy validation while providing a path toward **JIT compilation, batching/vectorization, accelerator compatibility, and downstream optimization workflows**.

## What is being interpolated?

For a query initial state `(T1, P1)`, the lookup reconstructs the complete ZND profiles:

**(T₁, P₁) → T(x), p(x), Yᵢ(x)**

The four surrounding precomputed manifold states are used as interpolation corners. Their profiles are first represented on a common physical distance-behind-shock grid, and bilinear interpolation is then performed across the initial-temperature and initial-pressure dimensions.

## JAX vs. SciPy regression

The JAX implementation was checked directly against the existing SciPy bilinear implementation at the same nine off-grid validation states.

The two implementations agree to floating-point precision. Representative maximum absolute differences across the regression include:

- **Temperature:** approximately 9.09 × 10⁻¹³ K
- **Pressure:** up to approximately 1.86 × 10⁻⁹ Pa
- **Species mass fractions:** generally on the order of 10⁻¹⁷ or smaller
- **N₂:** differences up to approximately 1.11 × 10⁻¹⁶ from numerical roundoff

These differences are numerical precision effects, not differences in physical or interpolation accuracy.

## JAX vs. direct SDToolbox

The JAX-interpolated profiles were also compared against direct SDToolbox solutions at the nine off-grid validation states.

| Quantity | Mean NRMSE | Median NRMSE | Maximum NRMSE |
|---|---:|---:|---:|
| Temperature | 0.028877% | 0.003300% | 0.095866% |
| Pressure | 0.043397% | 0.043367% | 0.097126% |
| H₂ | 0.052173% | 0.007593% | 0.155266% |
| O₂ | 0.049045% | 0.005790% | 0.152601% |
| H₂O | 0.046162% | 0.006145% | 0.140419% |
| OH | 0.063558% | 0.005700% | 0.260256% |
| H | 0.110893% | 0.021923% | 0.354613% |
| O | 0.091815% | 0.011381% | 0.330610% |
| HO₂ | 0.170375% | 0.011810% | 1.048301% |
| H₂O₂ | 0.098059% | 0.036494% | 0.258931% |
| N₂ | ~0% | 0% | ~0% |

As expected, these errors reproduce the SciPy bilinear validation because JAX is implementing the same interpolation calculation.

## Performance

For the warmed interpolation-only kernel on CPU:

- **JAX:** approximately 7.36 µs/query
- **SciPy:** approximately 49.09 µs/query
- **Measured mean speedup:** approximately 6.67×

The first JAX call includes compilation overhead and should not be compared directly with warmed calls.

The benchmark is specifically for the **warmed interpolation kernel**. It does not include:

- SDToolbox manifold generation
- Disk I/O
- Loading the four corner files
- Physical-`x` resampling
- Other end-to-end workflow overhead

Therefore, the 6.67× value should not be interpreted as a speedup for the complete ZND workflow.

## Files

- `jax_znd_bilinear_lookup_50x50.py` — JAX bilinear lookup and validation script
- `jax_vs_scipy_regression.csv` — direct numerical comparison between JAX and SciPy interpolation
- `jax_vs_sdtoolbox_validation.csv` — JAX interpolation compared with direct SDToolbox truth
- `jax_aggregate_nrmse.csv` — aggregate NRMSE statistics across the nine validation states

## Role in the larger workflow

The current computational path is:

**Precomputed ZND manifold → JAX profile lookup → T(x), p(x), Yᵢ(x) → thermal boundary conditions → heat-transfer / cooling model → optimization**

The next architectural step is to move toward an **in-memory JAX manifold representation** and batched queries, so many operating states can be evaluated efficiently without treating individual NPZ files as the runtime lookup mechanism.

A key issue for that step is the profile coordinate: the stored ZND solutions do not all share the same physical `x_max`. Any global tensor representation therefore requires an explicit coordinate policy rather than silently forcing all profiles onto one global physical-distance grid.

The current implementation establishes the important baseline first: **JAX reproduces the validated SciPy bilinear interpolation to numerical precision while providing a faster, JAX-compatible lookup kernel.**
