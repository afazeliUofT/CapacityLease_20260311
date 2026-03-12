from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any

import numpy as np

from .config import deep_copy_dict, save_json, update_nested
from .market_clearing import MarketClearingSolver
from .models import ModelSpec
from .monopoly import MonopolySolver
from .flexible import FlexibleSolver


@dataclass
class DiagnosticBundle:
    monopoly_optimum: dict[str, float]
    market_rows: list[dict[str, float]]
    flex_rows: list[dict[str, float]]


def _best_valid(rows: list[dict[str, float]], field: str) -> dict[str, float] | None:
    valid = [row for row in rows if np.isfinite(row.get(field, np.nan))]
    if not valid:
        return None
    return max(valid, key=lambda row: row[field])


def _get_nested(payload: dict[str, Any], path: list[str | int]) -> Any:
    cur: Any = payload
    for key in path:
        cur = cur[key]
    return cur


def _sensitivity_eval_worker(
    base_payload: dict[str, Any],
    label: str,
    path: list[str | int],
    base_value: float,
    trial_value: float,
    direction: str,
) -> dict[str, Any]:
    payload = deep_copy_dict(base_payload)
    cursor: Any = payload
    for key in path[:-1]:
        cursor = cursor[key]
    cursor[path[-1]] = trial_value

    trial_spec = ModelSpec(payload)
    monopoly_solver = MonopolySolver(trial_spec)
    monopoly_opt = monopoly_solver.optimum()

    market_solver = MarketClearingSolver(trial_spec, monopoly_opt)
    market_capacity = int(round(trial_spec.claims.get("market_optimal_C_V", trial_spec.search.get("market_reference_capacity", 100))))
    market_row = market_solver.solve_for_capacity(float(market_capacity))

    flex_capacity = int(round(trial_spec.claims.get("flex_optimal_C_V", trial_spec.search.get("flex_reference_capacity", 350))))
    flex_solver = FlexibleSolver(trial_spec, monopoly_opt)
    flex_row = flex_solver.solve_for_capacity(float(flex_capacity))

    return {
        "parameter": label,
        "direction": direction,
        "base_value": float(base_value),
        "trial_value": float(trial_value),
        "monopoly_revenue": float(monopoly_opt["optimal_revenue"]),
        "market_R_MV_at_ref_CV": float(market_row["best_R_MV"]),
        "flex_R_MV_at_ref_CV": float(flex_row["best_R_MV"]) if np.isfinite(flex_row["best_R_MV"]) else float("nan"),
    }


def claim_check(spec: ModelSpec, bundle: DiagnosticBundle) -> dict[str, Any]:
    claims = spec.claims
    market_best = _best_valid(bundle.market_rows, "best_R_MV")
    flex_best = _best_valid(bundle.flex_rows, "best_R_MV")

    actual = {
        "monopoly_revenue": float(bundle.monopoly_optimum["optimal_revenue"]),
        "monopoly_price": float(bundle.monopoly_optimum["optimal_price"]),
        "market_optimal_C_V": None if market_best is None else float(market_best["C_V"]),
        "market_optimal_p_M": None if market_best is None else float(market_best["best_p_M"]),
        "market_optimal_p_V": None if market_best is None else float(market_best["best_p_V"]),
        "market_optimal_n_M": None if market_best is None else float(market_best["best_n_M"]),
        "market_optimal_n_V": None if market_best is None else float(market_best["best_n_V"]),
        "market_optimal_industry_revenue": None if market_best is None else float(market_best["best_R_industry_net"]),
        "flex_optimal_C_V": None if flex_best is None else float(flex_best["C_V"]),
        "flex_optimal_p_M": None if flex_best is None else float(flex_best["best_p_M"]),
        "flex_optimal_p_V": None if flex_best is None else float(flex_best["best_p_V"]),
        "flex_optimal_n_M": None if flex_best is None else float(flex_best["best_n_M"]),
        "flex_optimal_n_V": None if flex_best is None else float(flex_best["best_n_V"]),
        "flex_optimal_served": None if flex_best is None else float(flex_best["best_n_M"] + flex_best["best_n_V"]),
        "flex_optimal_unserved_share": None if flex_best is None else float(1.0 - (flex_best["best_n_M"] + flex_best["best_n_V"]) / spec.N_total),
        "flex_optimal_industry_revenue": None if flex_best is None else float(flex_best["best_R_industry_net"]),
    }

    comparison: dict[str, Any] = {"claims": claims, "actual": actual, "errors": {}}
    for key, target in claims.items():
        if key not in actual:
            continue
        value = actual[key]
        if value is None:
            comparison["errors"][key] = None
        else:
            comparison["errors"][key] = float(value) - float(target)
    return comparison


def delta_consistency_check(base_payload: dict[str, Any]) -> dict[str, Any]:
    delta_values = [0.1, 0.01]
    results: dict[str, Any] = {}
    for delta in delta_values:
        payload = update_nested(base_payload, ["parameters", "delta"], delta)
        spec = ModelSpec(payload)
        solver = MonopolySolver(spec)
        optimum = solver.optimum()
        results[str(delta)] = optimum
    return results


def root_stability_report(spec: ModelSpec, bundle: DiagnosticBundle) -> dict[str, Any]:
    market_best = _best_valid(bundle.market_rows, "best_R_MV")
    flex_best = _best_valid(bundle.flex_rows, "best_R_MV")
    base_payload = deep_copy_dict(spec.to_dict())

    report: dict[str, Any] = {"monopoly": [], "market_clearing": [], "flexible": []}
    tolerances = [1e-6, 1e-8, 1e-10, 1e-12]

    for tol in tolerances:
        payload = deep_copy_dict(base_payload)
        payload.setdefault("search", {})
        payload["search"]["root_tol"] = tol
        payload["search"]["flex_root_tol"] = tol
        trial_spec = ModelSpec(payload)

        monopoly_solver = MonopolySolver(trial_spec)
        monopoly_opt = monopoly_solver.optimum()
        report["monopoly"].append(
            {
                "tol": tol,
                "optimal_price": float(monopoly_opt["optimal_price"]),
                "optimal_subscribers": float(monopoly_opt["optimal_subscribers"]),
                "optimal_revenue": float(monopoly_opt["optimal_revenue"]),
                "critical_price": float(monopoly_opt["critical_price"]),
            }
        )

        if market_best is not None and float(market_best["C_V"]) > 0.0 and float(market_best["best_n_V"]) > 0.0:
            market_solver = MarketClearingSolver(trial_spec, monopoly_opt)
            C_V = float(market_best["C_V"])
            n_V = int(round(market_best["best_n_V"]))
            p_M, residual = market_solver.solve_p_M(C_V, n_V)
            cand = market_solver.evaluate_candidate(C_V, n_V)
            report["market_clearing"].append(
                {
                    "tol": tol,
                    "C_V": C_V,
                    "n_V": n_V,
                    "p_M": float(p_M),
                    "p_V": float(market_solver.p_V(C_V, n_V)),
                    "R_MV": None if cand is None else float(cand.R_MV),
                    "residual": float(residual),
                }
            )

        if flex_best is not None and float(flex_best["C_V"]) > 0.0 and float(flex_best["best_n_M"]) > 0.0 and float(flex_best["best_n_V"]) > 0.0:
            flex_solver = FlexibleSolver(trial_spec, monopoly_opt)
            C_V = float(flex_best["C_V"])
            n_M = int(round(flex_best["best_n_M"]))
            n_V = int(round(flex_best["best_n_V"]))
            starts = [
                (float(flex_best["best_p_M"]), float(flex_best["best_p_V"])),
                (max(0.01, float(flex_best["best_p_M"]) * 0.8), float(flex_best["best_p_V"])),
                (float(flex_best["best_p_M"]), max(0.01, float(flex_best["best_p_V"]) * 1.2)),
                (20.0, 5.0),
                (60.0, 10.0),
            ]
            for start in starts:
                prices, residual, method = flex_solver.solve_prices(C_V, n_M, n_V, warm_start=start, root_tol=tol)
                report["flexible"].append(
                    {
                        "tol": tol,
                        "start_p_M": float(start[0]),
                        "start_p_V": float(start[1]),
                        "C_V": C_V,
                        "n_M": n_M,
                        "n_V": n_V,
                        "p_M": None if prices is None else float(prices[0]),
                        "p_V": None if prices is None else float(prices[1]),
                        "residual_inf": float(residual),
                        "method": method,
                    }
                )
    return report


def parameter_sensitivity_report(spec: ModelSpec, baseline_claims: dict[str, Any], n_jobs: int = 1) -> dict[str, Any]:
    perturb_pct = float(spec.diagnostics.get("parameter_perturbation_pct", 0.01))
    selected_paths: list[tuple[str, list[str | int]]] = [
        ("delta", ["parameters", "delta"]),
        ("zeta", ["parameters", "zeta"]),
        ("lambda", ["parameters", "lambda"]),
        ("pi_V", ["parameters", "pi_V"]),
        ("C", ["parameters", "C"]),
    ]
    for g in range(spec.G):
        selected_paths.extend(
            [
                (f"group_{g+1}_N", ["parameters", "groups", g, "N"]),
                (f"group_{g+1}_alpha", ["parameters", "groups", g, "alpha"]),
                (f"group_{g+1}_beta", ["parameters", "groups", g, "beta"]),
                (f"group_{g+1}_eps_mu", ["parameters", "groups", g, "epsilon", "mu"]),
                (f"group_{g+1}_eps_sigma", ["parameters", "groups", g, "epsilon", "sigma"]),
                (f"group_{g+1}_mu_M", ["parameters", "groups", g, "epsilon_M", "mu"]),
                (f"group_{g+1}_sigma_M", ["parameters", "groups", g, "epsilon_M", "sigma"]),
                (f"group_{g+1}_mu_V", ["parameters", "groups", g, "epsilon_V", "mu"]),
                (f"group_{g+1}_sigma_V", ["parameters", "groups", g, "epsilon_V", "sigma"]),
            ]
        )

    base_payload = deep_copy_dict(spec.to_dict())
    jobs: list[tuple[str, list[str | int], float, float, str]] = []
    for label, path in selected_paths:
        base_value = float(_get_nested(base_payload, path))
        for sign, direction in [(-1.0, "down"), (1.0, "up")]:
            trial_value = base_value * (1.0 + sign * perturb_pct)
            jobs.append((label, path, base_value, trial_value, direction))

    rows: list[dict[str, Any]] = []
    if n_jobs <= 1:
        for label, path, base_value, trial_value, direction in jobs:
            rows.append(_sensitivity_eval_worker(base_payload, label, path, base_value, trial_value, direction))
    else:
        with ProcessPoolExecutor(max_workers=n_jobs) as pool:
            futures = {
                pool.submit(_sensitivity_eval_worker, base_payload, label, path, base_value, trial_value, direction): (label, direction)
                for label, path, base_value, trial_value, direction in jobs
            }
            for future in as_completed(futures):
                rows.append(future.result())

    rows.sort(key=lambda row: (row["parameter"], row["direction"]))
    return {"perturbation_pct": perturb_pct, "rows": rows, "baseline_claims": baseline_claims}


def save_diagnostics(output_dir: str, payload: dict[str, Any]) -> None:
    save_json(output_dir, payload)
