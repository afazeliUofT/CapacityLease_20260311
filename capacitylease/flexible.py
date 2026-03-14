from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Iterable

import numpy as np
from scipy.optimize import least_squares, root

from .distributions import standard_bivariate_rect_prob
from .models import ModelSpec


@dataclass
class FlexibleCandidate:
    C_V: float
    n_M: int
    n_V: int
    p_M: float
    p_V: float
    r_M: float
    r_V: float
    R_MV: float
    R_V_subscriber: float
    R_V_retained: float
    R_industry_net: float
    residual_inf: float
    method: str


class FlexibleSolver:
    def __init__(self, spec: ModelSpec, monopoly_optimum: dict[str, float]) -> None:
        self.spec = spec
        self.monopoly_optimum = monopoly_optimum
        self._cache: dict[tuple[float, int, int], FlexibleCandidate | None] = {}
        self._best_response_cache: dict[tuple[float, int], FlexibleCandidate | None] = {}

    def _rates(self, C_V: float, n_M: int, n_V: int) -> tuple[float, float]:
        r_M = (self.spec.C - C_V) / (self.spec.delta * float(n_M))
        r_V = C_V / (self.spec.delta * float(n_V))
        return r_M, r_V

    def _prob_join_M(self, group_index: int, p_M: float, p_V: float, C_V: float, n_M: int, n_V: int) -> float:
        r_M, r_V = self._rates(C_V, n_M, n_V)
        a = self.spec.beta[group_index] * p_M - self.spec.alpha[group_index] * math.log(r_M)
        d = self.spec.beta[group_index] * (p_V - p_M) + self.spec.alpha[group_index] * math.log(r_M / r_V)

        sigma_Z = math.sqrt(self.spec.sigma_M[group_index] ** 2 + self.spec.sigma_V[group_index] ** 2)
        u_a = (a - self.spec.mu_M[group_index]) / self.spec.sigma_M[group_index]
        v_d = (d - (self.spec.mu_V[group_index] - self.spec.mu_M[group_index])) / sigma_Z
        rho = -self.spec.sigma_M[group_index] / sigma_Z
        return float(standard_bivariate_rect_prob(u_a, float("inf"), float("-inf"), v_d, rho))

    def _prob_join_V(self, group_index: int, p_M: float, p_V: float, C_V: float, n_M: int, n_V: int) -> float:
        r_M, r_V = self._rates(C_V, n_M, n_V)
        a = self.spec.beta[group_index] * p_V - self.spec.alpha[group_index] * math.log(r_V)
        d = self.spec.beta[group_index] * (p_M - p_V) + self.spec.alpha[group_index] * math.log(r_V / r_M)

        sigma_Z = math.sqrt(self.spec.sigma_M[group_index] ** 2 + self.spec.sigma_V[group_index] ** 2)
        u_a = (a - self.spec.mu_V[group_index]) / self.spec.sigma_V[group_index]
        v_d = (d - (self.spec.mu_M[group_index] - self.spec.mu_V[group_index])) / sigma_Z
        rho = -self.spec.sigma_V[group_index] / sigma_Z
        return float(standard_bivariate_rect_prob(u_a, float("inf"), float("-inf"), v_d, rho))

    def theta(self, p_M: float, p_V: float, C_V: float, n_M: int, n_V: int) -> tuple[float, float]:
        probs_M = np.array([self._prob_join_M(g, p_M, p_V, C_V, n_M, n_V) for g in range(self.spec.G)], dtype=float)
        probs_V = np.array([self._prob_join_V(g, p_M, p_V, C_V, n_M, n_V) for g in range(self.spec.G)], dtype=float)
        theta_M = float(np.dot(self.spec.N_g, probs_M))
        theta_V = float(np.dot(self.spec.N_g, probs_V))
        return theta_M, theta_V

    def residual_vector(self, prices: np.ndarray, C_V: float, n_M: int, n_V: int) -> np.ndarray:
        p_M = float(prices[0])
        p_V = float(prices[1])
        if p_M < 0.0 or p_V < 0.0:
            return np.array([1e6 + abs(p_M), 1e6 + abs(p_V)], dtype=float)
        theta_M, theta_V = self.theta(p_M, p_V, C_V, n_M, n_V)
        return np.array([theta_M - float(n_M), theta_V - float(n_V)], dtype=float)

    def _dedupe_starts(self, starts: Iterable[Iterable[float]]) -> list[list[float]]:
        seen: set[tuple[float, float]] = set()
        out: list[list[float]] = []
        for start in starts:
            pair = [float(start[0]), float(start[1])]
            key = (round(pair[0], 10), round(pair[1], 10))
            if key not in seen and pair[0] >= 0.0 and pair[1] >= 0.0:
                out.append(pair)
                seen.add(key)
        return out

    def _default_starts(self, warm_start: tuple[float, float] | None = None) -> list[list[float]]:
        starts: list[list[float]] = []
        if warm_start is not None:
            w0, w1 = float(warm_start[0]), float(warm_start[1])
            starts.extend(
                [
                    [w0, w1],
                    [max(0.01, 0.9 * w0), w1],
                    [min(max(0.01, 1.1 * w0), float(self.spec.search.get("flex_price_bound", 500.0))), w1],
                    [w0, max(0.01, 0.9 * w1)],
                    [w0, min(max(0.01, 1.1 * w1), float(self.spec.search.get("flex_price_bound", 500.0)))],
                ]
            )
        monopoly_ref = float(self.monopoly_optimum["optimal_price"])
        starts.extend(
            [
                [monopoly_ref, 1.0],
                [0.9 * monopoly_ref, 2.0],
                [0.8 * monopoly_ref, 5.0],
                [60.0, 10.0],
                [55.0, 12.0],
                [40.0, 10.0],
                [20.0, 5.0],
                [5.0, 5.0],
                [20.0, 10.0],
                [50.0, 5.0],
                [50.0, 10.0],
                [80.0, 10.0],
                [100.0, 20.0],
                [20.0, 20.0],
                [10.0, 2.0],
                [30.0, 15.0],
                [1.0, 1.0],
            ]
        )
        return self._dedupe_starts(starts)

    def _price_bounds(
        self,
        C_V: float,
        n_M: int,
        n_V: int,
        warm_start: tuple[float, float] | None = None,
    ) -> tuple[float, float, float, float]:
        r_M, r_V = self._rates(C_V, n_M, n_V)
        bound_cap = float(self.spec.search.get("flex_price_bound", self.spec.search.get("market_price_bound", 500.0)))
        monopoly_ref = float(self.monopoly_optimum["optimal_price"])

        mean_p_M = float(np.max((self.spec.mu_M + self.spec.alpha * np.log(r_M)) / self.spec.beta))
        mean_p_V = float(np.max((self.spec.mu_V + self.spec.alpha * np.log(r_V)) / self.spec.beta))

        p_M_hi = min(bound_cap, max(20.0, 1.35 * monopoly_ref, 1.15 * mean_p_M))
        p_V_hi = min(bound_cap, max(10.0, 1.25 * mean_p_V, 0.2 * p_M_hi))

        if warm_start is not None:
            p_M_hi = min(bound_cap, max(p_M_hi, 1.25 * float(warm_start[0])))
            p_V_hi = min(bound_cap, max(p_V_hi, 1.25 * float(warm_start[1])))

        return 0.0, p_M_hi, 0.0, p_V_hi

    def _seed_grid(self, C_V: float, n_M: int, n_V: int, warm_start: tuple[float, float] | None = None) -> list[list[float]]:
        if not bool(self.spec.search.get("flex_use_global_seed_fallback", True)):
            return []

        p_M_lo, p_M_hi, p_V_lo, p_V_hi = self._price_bounds(C_V, n_M, n_V, warm_start=warm_start)
        p_M_points = int(self.spec.search.get("flex_seed_grid_p_M_points", 7))
        p_V_points = int(self.spec.search.get("flex_seed_grid_p_V_points", 7))
        keep = int(self.spec.search.get("flex_seed_grid_keep", 12))

        starts: list[list[float]] = []
        starts.extend(self._default_starts(warm_start=warm_start))

        monopoly_ref = float(self.monopoly_optimum["optimal_price"])
        p_M_values = list(np.linspace(p_M_lo, p_M_hi, p_M_points)) + [monopoly_ref, 0.75 * monopoly_ref, 0.5 * monopoly_ref]
        p_V_values = list(np.linspace(p_V_lo, p_V_hi, p_V_points)) + [0.5, 1.0, 2.0, 5.0, 10.0]
        if warm_start is not None:
            p_M_values.extend([float(warm_start[0]), 0.9 * float(warm_start[0]), 1.1 * float(warm_start[0])])
            p_V_values.extend([float(warm_start[1]), 0.9 * float(warm_start[1]), 1.1 * float(warm_start[1])])

        deduped_grid = self._dedupe_starts([[p_M, p_V] for p_M in p_M_values for p_V in p_V_values])

        def score(start: list[float]) -> tuple[float, float, float]:
            res = self.residual_vector(np.asarray(start, dtype=float), C_V, n_M, n_V)
            return (float(np.linalg.norm(res, ord=np.inf)), start[0], start[1])

        ranked = sorted(deduped_grid, key=score)
        starts.extend(ranked[:keep])
        return self._dedupe_starts(starts)

    def _register_candidate(
        self,
        x: np.ndarray | None,
        C_V: float,
        n_M: int,
        n_V: int,
        method: str,
        current_best_x: np.ndarray | None,
        current_best_res: float,
        current_best_method: str,
    ) -> tuple[np.ndarray | None, float, str]:
        if x is None or not np.all(np.isfinite(x)):
            return current_best_x, current_best_res, current_best_method
        if np.min(x) < 0.0:
            return current_best_x, current_best_res, current_best_method
        res = float(np.linalg.norm(self.residual_vector(np.asarray(x, dtype=float), C_V, n_M, n_V), ord=np.inf))
        if res < current_best_res:
            return np.asarray(x, dtype=float), res, method
        return current_best_x, current_best_res, current_best_method

    def solve_prices(
        self,
        C_V: float,
        n_M: int,
        n_V: int,
        warm_start: tuple[float, float] | None = None,
        *,
        root_tol: float = 1e-9,
        force_global_seed_fallback: bool = False,
    ) -> tuple[np.ndarray | None, float, str]:
        best_x: np.ndarray | None = None
        best_res = float("inf")
        best_method = "none"

        def fun(x: np.ndarray) -> np.ndarray:
            return self.residual_vector(x, C_V, n_M, n_V)

        def run_stage(starts: list[list[float]], stage_name: str) -> tuple[np.ndarray | None, float, str]:
            nonlocal best_x, best_res, best_method
            for start in starts:
                x0 = np.asarray(start, dtype=float)
                try:
                    sol = root(fun, x0=x0, method="hybr", options={"maxfev": int(self.spec.search.get("flex_root_maxfev", 180))})
                except Exception:
                    sol = None
                if sol is not None:
                    best_x, best_res, best_method = self._register_candidate(
                        np.asarray(sol.x, dtype=float),
                        C_V,
                        n_M,
                        n_V,
                        f"{stage_name}:root.hybr",
                        best_x,
                        best_res,
                        best_method,
                    )
                    if best_res <= root_tol:
                        return best_x, best_res, best_method

                try:
                    sol_ls = least_squares(
                        fun,
                        x0=x0,
                        bounds=(0.0, np.inf),
                        xtol=float(self.spec.search.get("flex_ls_xtol", 1e-12)),
                        ftol=float(self.spec.search.get("flex_ls_ftol", 1e-12)),
                        gtol=float(self.spec.search.get("flex_ls_gtol", 1e-12)),
                        max_nfev=int(self.spec.search.get("flex_ls_max_nfev", 300)),
                    )
                except Exception:
                    sol_ls = None
                if sol_ls is not None:
                    best_x, best_res, best_method = self._register_candidate(
                        np.asarray(sol_ls.x, dtype=float),
                        C_V,
                        n_M,
                        n_V,
                        f"{stage_name}:least_squares",
                        best_x,
                        best_res,
                        best_method,
                    )
                    if best_res <= root_tol:
                        return best_x, best_res, best_method

                    try:
                        polish = root(
                            fun,
                            x0=np.asarray(sol_ls.x, dtype=float),
                            method="hybr",
                            options={"maxfev": int(self.spec.search.get("flex_root_maxfev", 180))},
                        )
                    except Exception:
                        polish = None
                    if polish is not None:
                        best_x, best_res, best_method = self._register_candidate(
                            np.asarray(polish.x, dtype=float),
                            C_V,
                            n_M,
                            n_V,
                            f"{stage_name}:least_squares->root.hybr",
                            best_x,
                            best_res,
                            best_method,
                        )
                        if best_res <= root_tol:
                            return best_x, best_res, best_method
            return best_x, best_res, best_method

        initial_starts = self._default_starts(warm_start=warm_start)
        run_stage(initial_starts, stage_name="default")

        if best_res > root_tol and (force_global_seed_fallback or bool(self.spec.search.get("flex_use_global_seed_fallback", True))):
            fallback_starts = self._seed_grid(C_V, n_M, n_V, warm_start=warm_start)
            run_stage(fallback_starts, stage_name="seedgrid")

        return best_x, best_res, best_method

    def evaluate_candidate(
        self,
        C_V: float,
        n_M: int,
        n_V: int,
        warm_start: tuple[float, float] | None = None,
        *,
        force_global_seed_fallback: bool = False,
    ) -> FlexibleCandidate | None:
        cache_key = (float(C_V), int(n_M), int(n_V))
        if cache_key in self._cache and not force_global_seed_fallback:
            return self._cache[cache_key]

        prices, residual_inf, method = self.solve_prices(
            C_V,
            n_M,
            n_V,
            warm_start=warm_start,
            root_tol=float(self.spec.search.get("flex_root_tol", 1e-9)),
            force_global_seed_fallback=force_global_seed_fallback,
        )
        if prices is None or residual_inf > float(self.spec.search.get("flex_accept_residual_inf", 1e-6)):
            if not force_global_seed_fallback:
                self._cache[cache_key] = None
            return None

        p_M = float(prices[0])
        p_V = float(prices[1])
        if p_M < 0.0 or p_V < 0.0:
            if not force_global_seed_fallback:
                self._cache[cache_key] = None
            return None

        r_M, r_V = self._rates(C_V, n_M, n_V)
        R_V_subscriber = float(n_V) * p_V
        R_MV = float(n_M) * p_M + (1.0 - self.spec.lam) * R_V_subscriber - self.spec.pi_V
        R_V_retained = self.spec.lam * R_V_subscriber + self.spec.pi_V
        R_industry_net = float(n_M) * p_M + R_V_subscriber

        cand = FlexibleCandidate(
            C_V=float(C_V),
            n_M=int(n_M),
            n_V=int(n_V),
            p_M=p_M,
            p_V=p_V,
            r_M=r_M,
            r_V=r_V,
            R_MV=R_MV,
            R_V_subscriber=R_V_subscriber,
            R_V_retained=R_V_retained,
            R_industry_net=R_industry_net,
            residual_inf=float(residual_inf),
            method=method,
        )
        if not force_global_seed_fallback:
            self._cache[cache_key] = cand
        return cand

    def _better_mvno_response(self, cand: FlexibleCandidate, best: FlexibleCandidate | None) -> bool:
        if best is None:
            return True
        cand_key = (cand.R_V_subscriber, cand.R_MV, -cand.residual_inf)
        best_key = (best.R_V_subscriber, best.R_MV, -best.residual_inf)
        return cand_key > best_key

    def exact_best_response_for_n_M(
        self,
        C_V: float,
        n_M: int,
        *,
        n_V_lo: int | None = None,
        n_V_hi: int | None = None,
        force_global_seed_fallback: bool = True,
    ) -> FlexibleCandidate | None:
        n_V_min = self.spec.min_subscribers if n_V_lo is None else max(self.spec.min_subscribers, int(n_V_lo))
        n_V_max = int(self.spec.N_total - n_M) if n_V_hi is None else min(int(self.spec.N_total - n_M), int(n_V_hi))
        if n_V_max < n_V_min:
            return None

        benchmark = float(self.monopoly_optimum["optimal_revenue"])
        best: FlexibleCandidate | None = None
        warm: tuple[float, float] | None = None
        for n_V in range(n_V_min, n_V_max + 1):
            cand = self.evaluate_candidate(C_V, n_M, n_V, warm_start=warm, force_global_seed_fallback=force_global_seed_fallback)
            if cand is None:
                continue
            warm = (cand.p_M, cand.p_V)
            if cand.R_MV + 1e-9 < benchmark:
                continue
            if self._better_mvno_response(cand, best):
                best = cand
        return best

    def best_response_for_n_M(self, C_V: float, n_M: int) -> FlexibleCandidate | None:
        cache_key = (float(C_V), int(n_M))
        if cache_key in self._best_response_cache:
            return self._best_response_cache[cache_key]

        n_V_min = self.spec.min_subscribers
        n_V_max = int(self.spec.N_total - n_M)
        if n_V_max < n_V_min:
            self._best_response_cache[cache_key] = None
            return None

        coarse_step = int(self.spec.search.get("flex_nv_coarse_step", 10))
        refine_top_k = int(self.spec.search.get("flex_nv_refine_top_k", 5))
        refine_half_window = int(self.spec.search.get("flex_nv_refine_half_window", max(coarse_step, 10)))
        benchmark = float(self.monopoly_optimum["optimal_revenue"])

        coarse_candidates: list[FlexibleCandidate] = []
        warm: tuple[float, float] | None = None
        for n_V in range(n_V_min, n_V_max + 1, max(coarse_step, 1)):
            cand = self.evaluate_candidate(C_V, n_M, n_V, warm_start=warm)
            if cand is None:
                continue
            warm = (cand.p_M, cand.p_V)
            if cand.R_MV + 1e-9 < benchmark:
                continue
            coarse_candidates.append(cand)

        if not coarse_candidates:
            self._best_response_cache[cache_key] = None
            return None

        best = max(coarse_candidates, key=lambda cand: (cand.R_V_subscriber, cand.R_MV, -cand.residual_inf))

        full_scan_threshold = int(self.spec.search.get("flex_nv_exact_full_scan_threshold", 0))
        total_width = n_V_max - n_V_min + 1
        if full_scan_threshold > 0 and total_width <= full_scan_threshold:
            exact_values = list(range(n_V_min, n_V_max + 1))
        else:
            selected = sorted(
                coarse_candidates,
                key=lambda cand: (cand.R_V_subscriber, cand.R_MV, -cand.residual_inf),
                reverse=True,
            )[: max(refine_top_k, 1)]
            exact_set: set[int] = {int(cand.n_V) for cand in selected}
            for cand in selected:
                lo = max(n_V_min, cand.n_V - refine_half_window)
                hi = min(n_V_max, cand.n_V + refine_half_window)
                exact_set.update(range(lo, hi + 1))
            exact_values = sorted(exact_set)

        warm = None
        for n_V in exact_values:
            cand = self.evaluate_candidate(C_V, n_M, n_V, warm_start=warm)
            if cand is None:
                continue
            warm = (cand.p_M, cand.p_V)
            if cand.R_MV + 1e-9 < benchmark:
                continue
            if self._better_mvno_response(cand, best):
                best = cand

        self._best_response_cache[cache_key] = best
        return best

    def solve_for_capacity(self, C_V: float) -> dict[str, Any]:
        if C_V <= 0.0:
            return {
                "C_V": 0.0,
                "best_n_M": float(self.monopoly_optimum["optimal_subscribers"]),
                "best_n_V": 0.0,
                "best_p_M": float(self.monopoly_optimum["optimal_price"]),
                "best_p_V": float("nan"),
                "best_r_M": float(self.monopoly_optimum["optimal_target_rate"]),
                "best_r_V": float("nan"),
                "best_R_MV": float(self.monopoly_optimum["optimal_revenue"]),
                "best_R_V_subscriber": 0.0,
                "best_R_V_retained": 0.0,
                "best_R_industry_net": float(self.monopoly_optimum["optimal_revenue"]),
                "best_residual_inf": 0.0,
                "best_method": "monopoly",
            }

        n_M_min = max(self.spec.min_subscribers, int(self.spec.search.get("flex_nm_min", self.spec.min_subscribers)))
        n_M_max = min(
            int(self.spec.N_total - self.spec.min_subscribers),
            int(self.spec.search.get("flex_nm_max", self.spec.N_total - self.spec.min_subscribers)),
        )
        n_M_step = max(1, int(self.spec.search.get("flex_nm_step", 1)))
        none_run_stop = int(self.spec.search.get("flex_nm_none_run_stop", 0))
        refine_top_k = int(self.spec.search.get("flex_nm_refine_top_k", 5))
        refine_half_window = int(self.spec.search.get("flex_nm_refine_half_window", max(n_M_step, 4)))

        coarse_candidates: list[FlexibleCandidate] = []
        none_run = 0
        for n_M in range(n_M_min, n_M_max + 1, n_M_step):
            br = self.best_response_for_n_M(C_V, n_M)
            if br is None:
                none_run += 1
                if none_run_stop > 0 and coarse_candidates and none_run >= none_run_stop:
                    break
                continue
            coarse_candidates.append(br)
            none_run = 0

        if not coarse_candidates:
            return {
                "C_V": float(C_V),
                "best_n_M": float("nan"),
                "best_n_V": float("nan"),
                "best_p_M": float("nan"),
                "best_p_V": float("nan"),
                "best_r_M": float("nan"),
                "best_r_V": float("nan"),
                "best_R_MV": float("nan"),
                "best_R_V_subscriber": float("nan"),
                "best_R_V_retained": float("nan"),
                "best_R_industry_net": float("nan"),
                "best_residual_inf": float("nan"),
                "best_method": "none",
            }

        best = max(coarse_candidates, key=lambda cand: (cand.R_MV, cand.R_V_subscriber, -cand.residual_inf))

        if n_M_step > 1:
            selected = sorted(
                coarse_candidates,
                key=lambda cand: (cand.R_MV, cand.R_V_subscriber, -cand.residual_inf),
                reverse=True,
            )[: max(refine_top_k, 1)]
            exact_n_M: set[int] = {int(cand.n_M) for cand in selected}
            for cand in selected:
                lo = max(n_M_min, cand.n_M - refine_half_window)
                hi = min(n_M_max, cand.n_M + refine_half_window)
                exact_n_M.update(range(lo, hi + 1))
            for n_M in sorted(exact_n_M):
                br = self.best_response_for_n_M(C_V, n_M)
                if br is None:
                    continue
                if (br.R_MV, br.R_V_subscriber, -br.residual_inf) > (best.R_MV, best.R_V_subscriber, -best.residual_inf):
                    best = br

        return {
            "C_V": float(best.C_V),
            "best_n_M": float(best.n_M),
            "best_n_V": float(best.n_V),
            "best_p_M": float(best.p_M),
            "best_p_V": float(best.p_V),
            "best_r_M": float(best.r_M),
            "best_r_V": float(best.r_V),
            "best_R_MV": float(best.R_MV),
            "best_R_V_subscriber": float(best.R_V_subscriber),
            "best_R_V_retained": float(best.R_V_retained),
            "best_R_industry_net": float(best.R_industry_net),
            "best_residual_inf": float(best.residual_inf),
            "best_method": best.method,
        }

    def curve_at_capacity(self, C_V: float) -> list[dict[str, float]]:
        if C_V <= 0.0:
            return []
        n_M_min = max(
            self.spec.min_subscribers,
            int(self.spec.search.get("flex_curve_nm_min", self.spec.search.get("flex_nm_min", self.spec.min_subscribers))),
        )
        n_M_max = min(
            int(self.spec.N_total - self.spec.min_subscribers),
            int(self.spec.search.get("flex_curve_nm_max", self.spec.search.get("flex_nm_max", self.spec.N_total - self.spec.min_subscribers))),
        )
        rows: list[dict[str, float]] = []
        for n_M in range(n_M_min, n_M_max + 1, 1):
            br = self.best_response_for_n_M(C_V, n_M)
            if br is None:
                continue
            rows.append(
                {
                    "C_V": float(C_V),
                    "n_M": float(br.n_M),
                    "n_V": float(br.n_V),
                    "p_M": float(br.p_M),
                    "p_V": float(br.p_V),
                    "r_M": float(br.r_M),
                    "r_V": float(br.r_V),
                    "R_MV": float(br.R_MV),
                    "R_V_subscriber": float(br.R_V_subscriber),
                    "R_V_retained": float(br.R_V_retained),
                    "R_industry_net": float(br.R_industry_net),
                    "residual_inf": float(br.residual_inf),
                    "method": br.method,
                }
            )
        return rows
