# 50 × 50 Bilinear Multipoint Validation

This directory contains the independent off-grid validation of the **50 × 50 precomputed ZND manifold** using bilinear interpolation in initial temperature and pressure.

The objective is to test whether full ZND profiles reconstructed from the precomputed manifold reproduce direct SDToolbox solutions at states that are not manifold grid points.

## Validation procedure

For each validation state:

1. Identify the four surrounding states in the 50 × 50 manifold.
2. Load the four precomputed ZND profiles.
3. Resample the profiles onto a common physical distance-behind-shock grid.
4. Apply bilinear interpolation in initial temperature and pressure.
5. Compute a direct SDToolbox ZND solution at the validation state.
6. Compare the interpolated and direct profiles for temperature, pressure, and species mass fractions.

The nine validation states are combinations of:

- **Temperature:** 351.5, 651.5, and 951.5 K
- **Pressure:** 0.84, 2.54, and 4.64 atm

These states are off-grid relative to the 50 × 50 manifold.

## Aggregate NRMSE results

| Quantity | Mean NRMSE | Median NRMSE | Maximum NRMSE | Worst validation state |
|---|---:|---:|---:|---|
| Temperature | 0.028877% | 0.003300% | 0.095866% | 351.5 K, 4.64 atm |
| Pressure | 0.043397% | 0.043367% | 0.097126% | 351.5 K, 4.64 atm |
| H₂ | 0.052173% | 0.007593% | 0.155266% | 351.5 K, 4.64 atm |
| O₂ | 0.049045% | 0.005790% | 0.152601% | 351.5 K, 4.64 atm |
| H₂O | 0.046162% | 0.006145% | 0.140419% | 351.5 K, 4.64 atm |
| OH | 0.063558% | 0.005700% | 0.260256% | 351.5 K, 4.64 atm |
| H | 0.110893% | 0.021923% | 0.354613% | 351.5 K, 4.64 atm |
| O | 0.091815% | 0.011381% | 0.330610% | 351.5 K, 4.64 atm |
| HO₂ | 0.170375% | 0.011810% | 1.048301% | 351.5 K, 4.64 atm |
| H₂O₂ | 0.098059% | 0.036494% | 0.258931% | 351.5 K, 0.84 atm |
| N₂ | ~0% | 0% | ~0% | Numerical roundoff only |

The largest observed error is the **1.0483% HO₂ NRMSE** at 351.5 K and 4.64 atm. Temperature and pressure remain below **0.1% maximum NRMSE** across all nine validation states.

## Case-level summary

| T₁ [K] | P₁ [atm] | Mean NRMSE excluding N₂ | Maximum NRMSE | Worst quantity |
|---:|---:|---:|---:|---|
| 351.5 | 0.84 | 0.150443% | 0.266203% | H |
| 351.5 | 2.54 | 0.016195% | 0.064913% | Pressure |
| 351.5 | 4.64 | 0.282393% | 1.048301% | HO₂ |
| 651.5 | 0.84 | 0.114515% | 0.208404% | H₂O₂ |
| 651.5 | 2.54 | 0.010411% | 0.021864% | H |
| 651.5 | 4.64 | 0.003007% | 0.020043% | Pressure |
| 951.5 | 0.84 | 0.093032% | 0.161570% | H₂O₂ |
| 951.5 | 2.54 | 0.006426% | 0.014907% | H |
| 951.5 | 4.64 | 0.002497% | 0.010499% | Pressure |

## Files

- `validate_bilinear_multipoint_50x50.py` — validation script
- `bilinear_multipoint_validation.csv` — detailed quantity-by-case validation results
- `bilinear_multipoint_summary.csv` — aggregate statistics by quantity
- `bilinear_multipoint_case_summary.csv` — aggregate statistics by validation state
- `bilinear_multipoint_weights.csv` — interpolation corner and weight information

## Interpretation

The 50 × 50 manifold reproduces the tested off-grid direct SDToolbox profiles closely using bilinear interpolation. The major thermodynamic quantities and major species have small profile errors across the nine tested states, while the largest discrepancy occurs for HO₂ at the low-temperature, high-pressure validation state.

These results demonstrate strong performance at the **nine tested off-grid states**. They should not be interpreted as proving that the 50 × 50 grid is globally optimal or that uniform sampling is the most efficient possible manifold design.
