from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np

from .config import ensure_dirs, load_json, save_json
from .diagnostics import (
    DiagnosticBundle,
    claim_check,
    delta_consistency_check,
    parameter_sensitivity_report,
    root_stability_report,
)
from .flexible import FlexibleSolver
from .io_utils import write_rows
from .market_clearing import MarketClearingSolver
from .models import ModelSpec
from .monopoly import MonopolySolver
from .plotting import (
    plot_capacity_prices,
    plot_capacity_revenue,
    plot_monopoly_acceptance_rate,
    plot_monopoly_nr,
    plot_prices_vs_nM,
    plot_revenue_vs_nM,
)


def _market_eval_worker(config_path: str, monopoly_optimum: dict[str, float], C_V: float) -> dict[str, Any]:
    payload = load_json(config_path)
    spec = ModelSpec(payload)
    solver = MarketClearingSolver(spec, monopoly_optimum)
    return solver.solve_for_capacity(float(C_V))


def _flex_eval_worker(config_path: str, monopoly_optimum: dict[str, float], C_V: float) -> dict[str, Any]:
    payload = load_json(config_path)
    spec = ModelSpec(payload)
    solver = FlexibleSolver(spec, monopoly_optimum)
    return solver.solve_for_capacity(float(C_V))


def _parallel_capacity_eval(
    worker_kind: str,
    config_path: str,
    monopoly_optimum: dict[str, float],
    values: list[float],
    n_jobs: int,
) -> list[dict[str, Any]]:
    worker = _market_eval_worker if worker_kind == "market" else _flex_eval_worker
    if n_jobs <= 1:
        return [worker(config_path, monopoly_optimum, value) for value in values]

    rows: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=n_jobs) as pool:
        futures = {
            pool.submit(worker, config_path, monopoly_optimum, value): value for value in values
        }
        for future in as_completed(futures):
            rows.append(future.result())
    rows.sort(key=lambda row: row["C_V"])
    return rows


def _capacity_grid(step: int, C: float) -> list[float]:
    if step <= 0:
        raise ValueError("capacity step must be positive")
    values = list(np.arange(0.0, max(C - 1e-9, 0.0), step, dtype=float))
    if 0.0 not in values:
        values = [0.0] + values
    if values and values[-1] >= C:
        values = [value for value in values if value < C]
    return values


def reproduce(config_path: str | Path, project_root: str | Path, n_jobs: int) -> dict[str, Any]:
    payload = load_json(config_path)
    spec = ModelSpec(payload)

    project_root = Path(project_root)
    output_root = project_root / "outputs" / spec.name
    figure_dir = output_root / "figures"
    data_dir = output_root / "data"
    report_dir = output_root / "reports"
    ensure_dirs([figure_dir, data_dir, report_dir])

    monopoly_solver = MonopolySolver(spec)
    monopoly_rows = monopoly_solver.sweep()
    monopoly_optimum = monopoly_solver.optimum()
    write_rows(data_dir / "monopoly_curve.csv", monopoly_rows)
    save_json(report_dir / "monopoly_optimum.json", monopoly_optimum)

    market_step = int(spec.search.get("market_capacity_step", 5))
    market_capacities = _capacity_grid(market_step, spec.C)
    market_solver = MarketClearingSolver(spec, monopoly_optimum)
    market_rows = _parallel_capacity_eval("market", str(config_path), monopoly_optimum, market_capacities, n_jobs=n_jobs)
    write_rows(data_dir / "market_clearing_capacity_sweep.csv", market_rows)

    flex_step = int(spec.search.get("flex_capacity_step", 5))
    flex_capacities = _capacity_grid(flex_step, spec.C)
    flex_rows = _parallel_capacity_eval("flex", str(config_path), monopoly_optimum, flex_capacities, n_jobs=n_jobs)
    write_rows(data_dir / "flexible_capacity_sweep.csv", flex_rows)

    market_valid = [row for row in market_rows if np.isfinite(row["best_R_MV"])]
    market_best = max(market_valid, key=lambda row: row["best_R_MV"])
    flex_valid = [row for row in flex_rows if np.isfinite(row["best_R_MV"])]
    flex_best = max(flex_valid, key=lambda row: row["best_R_MV"]) if flex_valid else None

    market_curve_rows = market_solver.curve_at_capacity(float(market_best["C_V"]))
    write_rows(data_dir / "market_curve_at_optimal_capacity.csv", market_curve_rows)

    if flex_best is not None:
        flex_solver = FlexibleSolver(spec, monopoly_optimum)
        flex_curve_rows = flex_solver.curve_at_capacity(float(flex_best["C_V"]))
    else:
        flex_curve_rows = []
    write_rows(data_dir / "flex_curve_at_optimal_capacity.csv", flex_curve_rows)

    plot_monopoly_nr(monopoly_rows, figure_dir / "monopoly_n_R")
    plot_monopoly_acceptance_rate(monopoly_rows, figure_dir / "monopoly_Ag_r", spec.group_names)
    plot_capacity_revenue(market_rows, flex_rows, float(monopoly_optimum["optimal_revenue"]), figure_dir / "MNO_MVNO_CapacityBlocks")
    plot_capacity_prices(market_rows, flex_rows, figure_dir / "Optimal_Prices_vs_Capacity")
    plot_revenue_vs_nM(market_curve_rows, flex_curve_rows, figure_dir / "MVNO_MNO_Revenue")
    plot_prices_vs_nM(market_curve_rows, flex_curve_rows, figure_dir / "MVNO_MNO_Prices")

    summary = {
        "config_name": spec.name,
        "monopoly_optimum": monopoly_optimum,
        "market_best": market_best,
        "flex_best": flex_best,
        "output_root": str(output_root),
    }
    save_json(report_dir / "summary.json", summary)
    return summary


def diagnostics(config_path: str | Path, project_root: str | Path, n_jobs: int) -> dict[str, Any]:
    payload = load_json(config_path)
    spec = ModelSpec(payload)
    project_root = Path(project_root)
    output_root = project_root / "outputs" / spec.name
    report_dir = output_root / "reports"
    data_dir = output_root / "data"
    ensure_dirs([report_dir, data_dir])

    summary_path = report_dir / "summary.json"
    if not summary_path.exists():
        reproduce(config_path=config_path, project_root=project_root, n_jobs=n_jobs)

    monopoly_optimum = load_json(report_dir / "monopoly_optimum.json")
    market_rows = []
    flex_rows = []

    import csv

    with (data_dir / "market_clearing_capacity_sweep.csv").open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            market_rows.append({k: float(v) if v not in ("", "nan", "NaN") else float("nan") for k, v in row.items()})
    with (data_dir / "flexible_capacity_sweep.csv").open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            converted: dict[str, Any] = {}
            for k, v in row.items():
                if v in ("", "nan", "NaN"):
                    converted[k] = float("nan")
                else:
                    try:
                        converted[k] = float(v)
                    except ValueError:
                        converted[k] = v
            flex_rows.append(converted)

    bundle = DiagnosticBundle(
        monopoly_optimum=monopoly_optimum,
        market_rows=market_rows,
        flex_rows=flex_rows,
    )

    claim_payload = claim_check(spec, bundle)
    root_payload = root_stability_report(spec, bundle)
    delta_payload = delta_consistency_check(payload)
    sensitivity_payload = parameter_sensitivity_report(spec, claim_payload, n_jobs=n_jobs)

    save_json(report_dir / "claim_check.json", claim_payload)
    save_json(report_dir / "root_stability.json", root_payload)
    save_json(report_dir / "delta_consistency.json", delta_payload)
    save_json(report_dir / "parameter_sensitivity.json", sensitivity_payload)

    return {
        "claim_check": str(report_dir / "claim_check.json"),
        "root_stability": str(report_dir / "root_stability.json"),
        "delta_consistency": str(report_dir / "delta_consistency.json"),
        "parameter_sensitivity": str(report_dir / "parameter_sensitivity.json"),
    }
