# CapacityLease simulation + verification package

This package reproduces the simulation figures and raw outputs for the CapacityLease paper and adds numerical verification modules for sensitive root-finding / equilibrium solving.

## Included configs

- `configs/paper_figures_consistent.json`
  - Best config to reproduce the paper's narrative and figure discussion.
  - Uses `delta=0.01` because the published monopoly benchmark / figure discussion align with that value much better.
- `configs/paper_strict_text.json`
  - Uses the simulation table exactly as written in `CpctLeaseFinal.tex`.
  - Included to diagnose the paper's internal parameter inconsistency.
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
- `timing.json`

### Diagnostics run

- `claim_check.json`
- `root_stability.json`
- `delta_consistency.json`
- `parameter_sensitivity.json`
- `solver_certification.json`
- `paper_table_alignment.json`

### Cross-config audit

- `outputs/paper_audit/paper_config_comparison.json`

## Numerical changes in this patch

- Monopoly optimum is now computed with a bounded continuous optimizer instead of only taking the best point from a plotting grid.
- Market-clearing and flexible capacity sweeps now use coarse parallel sweeps plus local integer-capacity refinement near the best coarse candidates.
- Flexible-participation price solving now uses a stronger fallback: default multistart, bounded least-squares, and a deterministic seed-grid fallback for hard roots.
- Flexible best-response search now refines around multiple promising `n_V` candidates instead of refining around only one coarse candidate.
- Diagnostics now add explicit certification reports and a strict-table vs narrative cross-config audit.

## Recommended cluster run order

1. Create the venv with `scripts/setup_venv.sh`
2. Run `scripts/slurm_smoke_test.sh` once
3. Run `scripts/slurm_reproduce_figures.sh`
4. Run `scripts/slurm_diagnostics.sh`
5. Run `scripts/slurm_reproduce_text.sh`
6. Run `scripts/slurm_paper_audit.sh`

## Core numerical choices

- Monotone equations are solved with bracketed bisection.
- Monopoly revenue is optimized with bounded scalar optimization.
- Flexible-participation equilibrium prices are solved with multistart `root.hybr`, bounded `least_squares`, and deterministic seed-grid fallback.
- Diagnostics re-run the core solvers under multiple tolerances, start values, and parameter perturbations.
- Parallel execution uses Python process pools and is intended for 64 CPU jobs.
