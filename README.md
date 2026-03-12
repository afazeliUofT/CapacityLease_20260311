# CapacityLease simulation + verification package

This package reproduces the simulation figures and raw outputs for the CapacityLease paper and adds numerical verification modules for sensitive root-finding / equilibrium solving.

## Included configs

- `configs/paper_figures_consistent.json`
  - Best config to reproduce the paper's narrative and figure discussion.
  - Uses `delta=0.01` because the published monopoly benchmark / figure discussion align with that value much better.
- `configs/paper_strict_text.json`
  - Uses the simulation table exactly as written in `CpctLeaseFinal.tex`.
  - This is included to help diagnose the paper's internal parameter inconsistency.
- `configs/smoke_test.json`
  - Quick end-to-end test.

## Main outputs

Outputs are written under:

- `outputs/<config_name>/figures`
- `outputs/<config_name>/data`
- `outputs/<config_name>/reports`

Figure files are written in `png`, `pdf`, and `eps` formats.

## What is generated

### Reproduction run

- `monopoly_n_R`
- `monopoly_Ag_r`
- `MNO_MVNO_CapacityBlocks`
- `Optimal_Prices_vs_Capacity`
- `MVNO_MNO_Revenue`
- `MVNO_MNO_Prices`
- Raw CSV sweeps for monopoly, market-clearing, and flexible-participation cases
- `summary.json`

### Diagnostics run

- `claim_check.json`
- `root_stability.json`
- `delta_consistency.json`
- `parameter_sensitivity.json`

## Recommended cluster run order

1. Create the venv with `scripts/setup_venv.sh`
2. Run `scripts/slurm_smoke_test.sh` once
3. Run `scripts/slurm_reproduce_figures.sh`
4. Run `scripts/slurm_diagnostics.sh`
5. Optionally run `scripts/slurm_reproduce_text.sh`

## Core numerical choices

- Monotone equations are solved with bracketed bisection.
- Flexible-participation equilibrium prices are solved with multi-start `scipy.optimize.root(method="hybr")` and `least_squares` fallback.
- Diagnostics re-run the core solvers under multiple tolerances and parameter perturbations.
- Parallel execution uses Python process pools and is intended for 64 CPU jobs.
