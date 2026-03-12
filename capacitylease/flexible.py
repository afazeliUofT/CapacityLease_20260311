from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

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

    def _default_starts(self, warm_start: tuple[float, float] | None = None) -> list[list[float]]:
        starts: list[list[float]] = []
        if warm_start is not None:
            starts.append([float(warm_start[0]), float(warm_start[1])])
            starts.append([max(0.01, float(warm_start[0]) * 0.95), float(warm_start[1])])
            starts.append([float(warm_start[0]), max(0.01, float(warm_start[1]) * 0.95)])
        starts.extend([
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
        ])
        seen: set[tuple[float, float]] = set()
        out: list[list[float]] = []
        for start in starts:
            key = (round(start[0], 10), round(start[1], 10))
            if key not in seen:
                out.append(start)
                seen.add(key)
        return out

    def solve_prices(
        self,
        C_V: float,
        n_M: int,
        n_V: int,
        warm_start: tuple[float, float] | None = None,
        *,
        root_tol: float = 1e-9,
    ) -> tuple[np.ndarray | None, float, str]:
        starts = self._default_starts(warm_start=warm_start)
        best_x: np.ndarray | None = None
        best_res = float("inf")
        best_method = "none"

        def fun(x: np.ndarray) -> np.ndarray:
            return self.residual_vector(x, C_V, n_M, n_V)

        for start in starts:
            try:
                sol = root(fun, x0=np.asarray(start, dtype=float), method="hybr", options={"maxfev": 120})
            except Exception:
                sol = None
            if sol is not None and np.all(np.isfinite(sol.x)):
                res = float(np.linalg.norm(fun(np.asarray(sol.x, dtype=float)), ord=np.inf))
                if res < best_res and np.min(sol.x) >= 0.0:
                    best_x = np.asarray(sol.x, dtype=float)
                    best_res = res
                    best_method = "root.hybr"
                if res <= root_tol and np.min(sol.x) >= 0.0:
                    return np.asarray(sol.x, dtype=float), res, "root.hybr"

        for start in starts[:4]:
            try:
                sol_ls = least_squares(
                    fun,
                    x0=np.asarray(start, dtype=float),
                    bounds=(0.0, np.inf),
                    xtol=1e-12,
                    ftol=1e-12,
                    gtol=1e-12,
                    max_nfev=200,
                )
            except Exception:
                sol_ls = None
            if sol_ls is not None and np.all(np.isfinite(sol_ls.x)):
                res = float(np.linalg.norm(fun(np.asarray(sol_ls.x, dtype=float)), ord=np.inf))
                if res < best_res and np.min(sol_ls.x) >= 0.0:
                    best_x = np.asarray(sol_ls.x, dtype=float)
                    best_res = res
                    best_method = "least_squares"
                if res <= root_tol and np.min(sol_ls.x) >= 0.0:
                    return np.asarray(sol_ls.x, dtype=float), res, "least_squares"

        return best_x, best_res, best_method

    def evaluate_candidate(
        self,
        C_V: float,
        n_M: int,
        n_V: int,
        warm_start: tuple[float, float] | None = None,
    ) -> FlexibleCandidate | None:
        cache_key = (float(C_V), int(n_M), int(n_V))
        if cache_key in self._cache:
            return self._cache[cache_key]

        prices, residual_inf, method = self.solve_prices(
            C_V,
            n_M,
            n_V,
            warm_start=warm_start,
            root_tol=float(self.spec.search.get("flex_root_tol", 1e-9)),
        )
        if prices is None or residual_inf > float(self.spec.search.get("flex_accept_residual_inf", 1e-6)):
            self._cache[cache_key] = None
            return None

        p_M = float(prices[0])
        p_V = float(prices[1])
        if p_M < 0.0 or p_V < 0.0:
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
        self._cache[cache_key] = cand
        return cand

    def best_response_for_n_M(self, C_V: float, n_M: int) -> FlexibleCandidate | None:
        n_V_min = self.spec.min_subscribers
        n_V_max = int(self.spec.N_total - n_M)
        if n_V_max < n_V_min:
            return None

        coarse_step = int(self.spec.search.get("flex_nv_coarse_step", 10))
        refine_steps = list(self.spec.search.get("flex_nv_refine_steps", [2, 1]))
        benchmark = float(self.monopoly_optimum["optimal_revenue"])

        best: FlexibleCandidate | None = None
        warm: tuple[float, float] | None = None

        for n_V in range(n_V_min, n_V_max + 1, coarse_step):
            cand = self.evaluate_candidate(C_V, n_M, n_V, warm_start=warm)
            if cand is None:
                continue
            warm = (cand.p_M, cand.p_V)
            if cand.R_MV + 1e-9 < benchmark:
                continue
            if best is None or cand.R_V_subscriber > best.R_V_subscriber:
                best = cand

        if best is None:
            return None

        span = coarse_step
        for step in refine_steps:
            lo = max(n_V_min, best.n_V - span)
            hi = min(n_V_max, best.n_V + span)
            local_best = best
            warm = (best.p_M, best.p_V)
            for n_V in range(lo, hi + 1, int(step)):
                cand = self.evaluate_candidate(C_V, n_M, n_V, warm_start=warm)
                if cand is None:
                    continue
                warm = (cand.p_M, cand.p_V)
                if cand.R_MV + 1e-9 < benchmark:
                    continue
                if cand.R_V_subscriber > local_best.R_V_subscriber + 1e-12:
                    local_best = cand
            best = local_best
            span = max(int(step) * 2, 2)
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
        n_M_max = min(int(self.spec.N_total - self.spec.min_subscribers), int(self.spec.search.get("flex_nm_max", self.spec.N_total - self.spec.min_subscribers)))
        n_M_step = int(self.spec.search.get("flex_nm_step", 1))
        none_run_stop = int(self.spec.search.get("flex_nm_none_run_stop", 40))

        best: FlexibleCandidate | None = None
        none_run = 0
        for n_M in range(n_M_min, n_M_max + 1, n_M_step):
            br = self.best_response_for_n_M(C_V, n_M)
            if br is None:
                none_run += 1
                if best is not None and none_run >= none_run_stop:
                    break
                continue
            none_run = 0
            if best is None or br.R_MV > best.R_MV:
                best = br

        if best is None:
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
        n_M_min = max(self.spec.min_subscribers, int(self.spec.search.get("flex_curve_nm_min", self.spec.search.get("flex_nm_min", self.spec.min_subscribers))))
        n_M_max = min(int(self.spec.N_total - self.spec.min_subscribers), int(self.spec.search.get("flex_curve_nm_max", self.spec.search.get("flex_nm_max", self.spec.N_total - self.spec.min_subscribers))))
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
