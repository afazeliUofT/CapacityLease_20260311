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


def _row_close(a: dict[str, Any] | None, b: dict[str, Any] | None, *, value_key: str, tol: float = 1e-9) -> bool:
    if a is None or b is None:
        return False
    return abs(float(a[value_key]) - float(b[value_key])) <= tol


def _paper_table_reference() -> dict[str, Any]:
    return {
        "C": 1000.0,
        "delta": 0.1,
        "zeta": 0.01,
        "lambda": 0.1,
        "pi_V": 500.0,
        "groups": [
            {
                "N": 200.0,
                "alpha": 3.0,
                "beta": 0.2,
                "epsilon": {"mu": 3.0, "sigma": 0.1},
                "epsilon_M": {"mu": 2.5, "sigma": 0.1},
                "epsilon_V": {"mu": 0.5, "sigma": 0.1},
            },
            {
                "N": 200.0,
                "alpha": 3.0,
                "beta": 0.6,
                "epsilon": {"mu": 2.0, "sigma": 1.0},
                "epsilon_M": {"mu": 1.5, "sigma": 1.0},
                "epsilon_V": {"mu": 2.5, "sigma": 1.0},
            },
            {
                "N": 900.0,
                "alpha": 2.0,
                "beta": 0.6,
                "epsilon": {"mu": 2.0, "sigma": 0.8},
                "epsilon_M": {"mu": 1.5, "sigma": 0.8},
                "epsilon_V": {"mu": 1.5, "sigma": 0.8},
            },
            {
                "N": 700.0,
                "alpha": 1.0,
                "beta": 1.0,
                "epsilon": {"mu": 0.5, "sigma": 0.1},
                "epsilon_M": {"mu": 0.2, "sigma": 0.1},
                "epsilon_V": {"mu": 3.0, "sigma": 0.1},
            },
        ],
    }


def _paper_table_alignment_report(spec: ModelSpec) -> dict[str, Any]:
    ref = _paper_table_reference()
    diffs: dict[str, Any] = {}
    matches = True

    for key in ["C", "delta", "zeta", "lambda", "pi_V"]:
        actual = float(spec.raw["parameters"][key])
        target = float(ref[key])
        diffs[key] = actual - target
        if abs(diffs[key]) > 1e-12:
            matches = False

    group_diffs: list[dict[str, Any]] = []
    for idx, group in enumerate(spec.raw["parameters"]["groups"]):
        ref_group = ref["groups"][idx]
        diff_row: dict[str, Any] = {"group_index": idx + 1, "differences": {}}
        for key in ["N", "alpha", "beta"]:
            value = float(group[key])
            target = float(ref_group[key])
            diff_row["differences"][key] = value - target
            if abs(diff_row["differences"][key]) > 1e-12:
                matches = False
        for block in ["epsilon", "epsilon_M", "epsilon_V"]:
            for key in ["mu", "sigma"]:
                value = float(group[block][key])
                target = float(ref_group[block][key])
                diff_row["differences"][f"{block}.{key}"] = value - target
                if abs(diff_row["differences"][f"{block}.{key}"]) > 1e-12:
                    matches = False
        group_diffs.append(diff_row)

    return {
        "matches_paper_simulation_table": matches,
        "scalar_parameter_differences": diffs,
        "group_parameter_differences": group_diffs,
    }


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
        monopoly_solver = MonopolySolver(spec)
        monopoly_opt = monopoly_solver.optimum()

        market_solver = MarketClearingSolver(spec, monopoly_opt)
        market_capacity = float(spec.search.get("market_reference_capacity", spec.claims.get("market_optimal_C_V", 100)))
        market_row = market_solver.solve_for_capacity(market_capacity)

        flex_solver = FlexibleSolver(spec, monopoly_opt)
        flex_capacity = float(spec.search.get("flex_reference_capacity", spec.claims.get("flex_optimal_C_V", 350)))
        flex_row = flex_solver.solve_for_capacity(flex_capacity)

        results[str(delta)] = {
            "monopoly": monopoly_opt,
            "market_at_reference_capacity": market_row,
            "flex_at_reference_capacity": flex_row,
        }
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
                (1.0, 1.0),
            ]
            for start in starts:
                prices, residual, method = flex_solver.solve_prices(
                    C_V,
                    n_M,
                    n_V,
                    warm_start=start,
                    root_tol=tol,
                    force_global_seed_fallback=True,
                )
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


def _cluster_solution_rows(rows: list[dict[str, Any]], *, residual_key: str, tol: float = 1e-6) -> list[dict[str, Any]]:
    successful = [row for row in rows if row.get("p_M") is not None and row.get("p_V") is not None and float(row[residual_key]) <= tol]
    clusters: list[dict[str, Any]] = []
    for row in successful:
        p_M = float(row["p_M"])
        p_V = float(row["p_V"])
        placed = False
        for cluster in clusters:
            if abs(p_M - cluster["representative_p_M"]) <= 1e-6 and abs(p_V - cluster["representative_p_V"]) <= 1e-6:
                cluster["count"] += 1
                cluster["rows"].append(row)
                placed = True
                break
        if not placed:
            clusters.append(
                {
                    "representative_p_M": p_M,
                    "representative_p_V": p_V,
                    "count": 1,
                    "rows": [row],
                }
            )
    return clusters


def solver_certification_report(spec: ModelSpec, bundle: DiagnosticBundle) -> dict[str, Any]:
    market_best = _best_valid(bundle.market_rows, "best_R_MV")
    flex_best = _best_valid(bundle.flex_rows, "best_R_MV")
    report: dict[str, Any] = {
        "paper_table_alignment": _paper_table_alignment_report(spec),
        "monopoly": {},
        "market_clearing": {},
        "flexible": {},
        "verdict": {},
    }

    monopoly = bundle.monopoly_optimum
    grid_revenue = monopoly.get("grid_optimal_revenue")
    report["monopoly"] = {
        "exact_optimum": monopoly,
        "grid_vs_exact_revenue_gap": None if grid_revenue is None else float(monopoly["optimal_revenue"]) - float(grid_revenue),
        "grid_vs_exact_price_gap": None if monopoly.get("grid_optimal_price") is None else float(monopoly["optimal_price"]) - float(monopoly["grid_optimal_price"]),
    }

    if market_best is not None:
        monopoly_solver = MonopolySolver(spec)
        market_solver = MarketClearingSolver(spec, monopoly_solver.optimum())
        half_window = int(spec.search.get("market_certify_capacity_half_window", 10))
        center = int(round(float(market_best["C_V"])))
        lo = max(0, center - half_window)
        hi = min(int(spec.C) - 1, center + half_window)
        local_rows = [market_solver.solve_for_capacity(float(C_V)) for C_V in range(lo, hi + 1)]
        local_best = _best_valid(local_rows, "best_R_MV")
        report["market_clearing"] = {
            "reported_best": market_best,
            "local_capacity_window": [lo, hi],
            "local_best": local_best,
            "reported_matches_local_best": _row_close(market_best, local_best, value_key="best_R_MV"),
        }

    if flex_best is not None:
        monopoly_solver = MonopolySolver(spec)
        monopoly_opt = monopoly_solver.optimum()
        flex_solver = FlexibleSolver(spec, monopoly_opt)
        C_V = float(flex_best["C_V"])
        n_M = int(round(flex_best["best_n_M"]))
        n_V = int(round(flex_best["best_n_V"]))

        starts = [
            (float(flex_best["best_p_M"]), float(flex_best["best_p_V"])),
            (max(0.01, 0.8 * float(flex_best["best_p_M"])), float(flex_best["best_p_V"])),
            (float(flex_best["best_p_M"]), max(0.01, 1.2 * float(flex_best["best_p_V"]))),
            (20.0, 5.0),
            (60.0, 10.0),
            (1.0, 1.0),
            (100.0, 20.0),
            (5.0, 1.0),
        ]
        root_rows: list[dict[str, Any]] = []
        for start in starts:
            prices, residual, method = flex_solver.solve_prices(
                C_V,
                n_M,
                n_V,
                warm_start=start,
                root_tol=float(spec.search.get("flex_root_tol", 1e-9)),
                force_global_seed_fallback=True,
            )
            root_rows.append(
                {
                    "start_p_M": float(start[0]),
                    "start_p_V": float(start[1]),
                    "p_M": None if prices is None else float(prices[0]),
                    "p_V": None if prices is None else float(prices[1]),
                    "residual_inf": float(residual),
                    "method": method,
                }
            )
        clusters = _cluster_solution_rows(root_rows, residual_key="residual_inf", tol=float(spec.search.get("flex_accept_residual_inf", 1e-6)))

        n_V_half_window = int(spec.search.get("flex_certify_nv_half_window", 20))
        exact_nV_best = flex_solver.exact_best_response_for_n_M(
            C_V,
            n_M,
            n_V_lo=max(spec.min_subscribers, n_V - n_V_half_window),
            n_V_hi=min(int(spec.N_total - n_M), n_V + n_V_half_window),
            force_global_seed_fallback=True,
        )

        n_M_half_window = int(spec.search.get("flex_certify_nm_half_window", 8))
        local_nm_rows: list[dict[str, Any]] = []
        for trial_n_M in range(max(spec.min_subscribers, n_M - n_M_half_window), min(int(spec.N_total - spec.min_subscribers), n_M + n_M_half_window) + 1):
            cand = flex_solver.best_response_for_n_M(C_V, trial_n_M)
            if cand is None:
                continue
            local_nm_rows.append(
                {
                    "n_M": float(cand.n_M),
                    "n_V": float(cand.n_V),
                    "best_R_MV": float(cand.R_MV),
                    "best_R_V_subscriber": float(cand.R_V_subscriber),
                    "best_p_M": float(cand.p_M),
                    "best_p_V": float(cand.p_V),
                    "best_residual_inf": float(cand.residual_inf),
                }
            )
        local_nm_best = _best_valid(local_nm_rows, "best_R_MV")

        capacity_half_window = int(spec.search.get("flex_certify_capacity_half_window", 10))
        local_capacity_rows = []
        for trial_C_V in range(max(0, int(round(C_V)) - capacity_half_window), min(int(spec.C) - 1, int(round(C_V)) + capacity_half_window) + 1):
            local_capacity_rows.append(flex_solver.solve_for_capacity(float(trial_C_V)))
        local_capacity_best = _best_valid(local_capacity_rows, "best_R_MV")

        report["flexible"] = {
            "reported_best": flex_best,
            "root_restart_rows": root_rows,
            "root_success_count": int(sum(float(row["residual_inf"]) <= float(spec.search.get("flex_accept_residual_inf", 1e-6)) for row in root_rows)),
            "root_attempt_count": len(root_rows),
            "root_success_rate": float(
                sum(float(row["residual_inf"]) <= float(spec.search.get("flex_accept_residual_inf", 1e-6)) for row in root_rows) / max(len(root_rows), 1)
            ),
            "root_solution_clusters": [
                {
                    "representative_p_M": float(cluster["representative_p_M"]),
                    "representative_p_V": float(cluster["representative_p_V"]),
                    "count": int(cluster["count"]),
                }
                for cluster in clusters
            ],
            "local_exact_nV_window": [max(spec.min_subscribers, n_V - n_V_half_window), min(int(spec.N_total - n_M), n_V + n_V_half_window)],
            "local_exact_nV_best": None if exact_nV_best is None else {
                "n_M": float(exact_nV_best.n_M),
                "n_V": float(exact_nV_best.n_V),
                "p_M": float(exact_nV_best.p_M),
                "p_V": float(exact_nV_best.p_V),
                "R_MV": float(exact_nV_best.R_MV),
                "R_V_subscriber": float(exact_nV_best.R_V_subscriber),
                "residual_inf": float(exact_nV_best.residual_inf),
            },
            "local_nM_best": local_nm_best,
            "local_capacity_best": local_capacity_best,
            "local_capacity_window": [max(0, int(round(C_V)) - capacity_half_window), min(int(spec.C) - 1, int(round(C_V)) + capacity_half_window)],
        }

    min_success_rate = float(spec.search.get("flex_certify_min_root_success_rate", 0.75))
    market_report = report.get("market_clearing", {})
    flexible_report = report.get("flexible", {})
    report["verdict"] = {
        "matches_paper_simulation_table": bool(report["paper_table_alignment"]["matches_paper_simulation_table"]),
        "monopoly_exact_optimizer_used": bool(monopoly.get("optimizer_success", False)),
        "market_local_capacity_certified": bool(market_report.get("reported_matches_local_best", False)),
        "flex_root_restarts_certified": bool(flexible_report.get("root_success_rate", 0.0) >= min_success_rate),
        "flex_local_nV_certified": bool(
            flexible_report.get("local_exact_nV_best") is not None
            and abs(float(flexible_report["local_exact_nV_best"]["R_MV"]) - float(flex_best["best_R_MV"])) <= 1e-6
        ) if flex_best is not None and flexible_report else False,
        "flex_local_nM_certified": bool(
            flexible_report.get("local_nM_best") is not None
            and abs(float(flexible_report["local_nM_best"]["best_R_MV"]) - float(flex_best["best_R_MV"])) <= 1e-6
        ) if flex_best is not None and flexible_report else False,
        "flex_local_capacity_certified": bool(
            flexible_report.get("local_capacity_best") is not None
            and abs(float(flexible_report["local_capacity_best"]["best_R_MV"]) - float(flex_best["best_R_MV"])) <= 1e-6
        ) if flex_best is not None and flexible_report else False,
    }
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
