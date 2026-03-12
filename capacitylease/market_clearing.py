from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .distributions import normal_cdf, normal_ppf
from .models import ModelSpec
from .numerical import monotone_bisection


@dataclass
class MarketCandidate:
    C_V: float
    n_V: int
    n_M: float
    p_M: float
    p_V: float
    r_M: float
    r_V: float
    R_MV: float
    R_V_subscriber: float
    R_V_retained: float
    R_industry_net: float
    residual: float


class MarketClearingSolver:
    def __init__(self, spec: ModelSpec, monopoly_optimum: dict[str, float]) -> None:
        self.spec = spec
        self.monopoly_optimum = monopoly_optimum

        self.kappa = normal_ppf(self.spec.zeta, self.spec.mu_V, self.spec.sigma_V) / self.spec.beta
        self.tau = self.spec.alpha / self.spec.beta
        self.diff_mu = self.spec.mu_M - self.spec.mu_V
        self.diff_sigma = np.sqrt(self.spec.sigma_M**2 + self.spec.sigma_V**2)
        self.threshold = self.spec.pi_V / (1.0 - self.spec.lam)

    def p_V(self, C_V: float, n_V: float) -> float:
        return float(np.min(self.kappa + self.tau * np.log(C_V / (self.spec.delta * n_V))))

    def feasibility_interval(self, C_V: float) -> tuple[float, float] | None:
        lo = 1.0 / self.spec.delta
        hi = self.spec.N_total - 1.0 / self.spec.delta
        T_V = float(np.min(self.kappa / self.tau))
        if T_V < -math.log(C_V):
            return None
        upper_threshold = math.log((self.spec.delta * self.spec.N_total - 1.0) / C_V)
        if T_V >= upper_threshold:
            return (lo, hi)
        upper = (C_V / self.spec.delta) * math.exp(T_V)
        upper = min(hi, upper)
        if upper < lo:
            return None
        return (lo, upper)

    def profitability_interval_for_group(self, C_V: float, group_index: int, base_interval: tuple[float, float]) -> tuple[float, float] | None:
        lo, hi = base_interval
        kappa = float(self.kappa[group_index])
        tau = float(self.tau[group_index])

        def f(n_value: float) -> float:
            return n_value * (kappa + tau * math.log(C_V / (self.spec.delta * n_value))) - self.threshold

        n_peak = (C_V / self.spec.delta) * math.exp(kappa / tau - 1.0)
        n_peak = max(lo, min(hi, n_peak))
        f_lo = f(lo)
        f_hi = f(hi)
        f_peak = f(n_peak)

        if max(f_lo, f_hi, f_peak) < 0:
            return None

        left = lo
        if f_lo < 0:
            left = monotone_bisection(lambda x: f(x), lo, n_peak, tol=float(self.spec.search.get("root_tol", 1e-12)))

        right = hi
        if f_hi < 0:
            right = monotone_bisection(lambda x: f(x), n_peak, hi, tol=float(self.spec.search.get("root_tol", 1e-12)))

        if right < left:
            return None
        return (left, right)

    def profitability_interval(self, C_V: float) -> tuple[float, float] | None:
        feasibility = self.feasibility_interval(C_V)
        if feasibility is None:
            return None
        current = feasibility
        for group_index in range(self.spec.G):
            interval = self.profitability_interval_for_group(C_V, group_index, current)
            if interval is None:
                return None
            current = (max(current[0], interval[0]), min(current[1], interval[1]))
            if current[1] < current[0]:
                return None
        return current

    def upsilon(self, p_M: float, n_V: float, C_V: float) -> float:
        p_V = self.p_V(C_V, n_V)
        arg = (
            self.spec.beta * (p_M - p_V)
            + self.spec.alpha * np.log(((self.spec.N_total - n_V) * C_V) / ((self.spec.C - C_V) * n_V))
        )
        value = np.sum(self.spec.N_g * normal_cdf(arg, self.diff_mu, self.diff_sigma)) - n_V
        return float(value)

    def solve_p_M(self, C_V: float, n_V: int) -> tuple[float, float]:
        bound = float(self.spec.search.get("market_price_bound", 500.0))

        def f(price: float) -> float:
            return self.upsilon(price, float(n_V), C_V)

        lo = -bound
        hi = bound
        f_lo = f(lo)
        f_hi = f(hi)
        while f_lo > 0:
            lo *= 2.0
            f_lo = f(lo)
        while f_hi < 0:
            hi *= 2.0
            f_hi = f(hi)
        root = monotone_bisection(f, lo, hi, tol=float(self.spec.search.get("root_tol", 1e-12)))
        residual = abs(f(root))
        return root, residual

    def evaluate_candidate(self, C_V: float, n_V: int) -> MarketCandidate | None:
        p_M, residual = self.solve_p_M(C_V, n_V)
        if p_M < 0:
            return None
        p_V = self.p_V(C_V, float(n_V))
        n_M = self.spec.N_total - float(n_V)
        r_M = (self.spec.C - C_V) / (self.spec.delta * n_M)
        r_V = C_V / (self.spec.delta * float(n_V))
        R_V_subscriber = float(n_V) * p_V
        R_MV = n_M * p_M + (1.0 - self.spec.lam) * R_V_subscriber - self.spec.pi_V
        R_V_retained = self.spec.lam * R_V_subscriber + self.spec.pi_V
        R_industry_net = n_M * p_M + R_V_subscriber
        return MarketCandidate(
            C_V=C_V,
            n_V=int(n_V),
            n_M=n_M,
            p_M=p_M,
            p_V=p_V,
            r_M=r_M,
            r_V=r_V,
            R_MV=R_MV,
            R_V_subscriber=R_V_subscriber,
            R_V_retained=R_V_retained,
            R_industry_net=R_industry_net,
            residual=residual,
        )

    def iter_n_V_candidates(self, C_V: float) -> list[int]:
        interval = self.profitability_interval(C_V)
        if interval is None:
            return []
        lo, hi = interval
        step = int(self.spec.search.get("market_nv_step", 1))
        start = int(math.ceil(lo))
        end = int(math.floor(hi))
        if end < start:
            return []
        return list(range(start, end + 1, step))

    def solve_for_capacity(self, C_V: float) -> dict[str, float]:
        if C_V <= 0.0:
            return {
                "C_V": 0.0,
                "best_n_V": 0.0,
                "best_n_M": float(self.monopoly_optimum["optimal_subscribers"]),
                "best_p_M": float(self.monopoly_optimum["optimal_price"]),
                "best_p_V": float("nan"),
                "best_r_M": float(self.monopoly_optimum["optimal_target_rate"]),
                "best_r_V": float("nan"),
                "best_R_MV": float(self.monopoly_optimum["optimal_revenue"]),
                "best_R_V_subscriber": 0.0,
                "best_R_V_retained": 0.0,
                "best_R_industry_net": float(self.monopoly_optimum["optimal_revenue"]),
                "best_residual": 0.0,
            }

        candidates = self.iter_n_V_candidates(C_V)
        best: MarketCandidate | None = None
        for n_V in candidates:
            cand = self.evaluate_candidate(C_V, n_V)
            if cand is None:
                continue
            if best is None or cand.R_MV > best.R_MV:
                best = cand

        if best is None:
            return {
                "C_V": float(C_V),
                "best_n_V": float("nan"),
                "best_n_M": float("nan"),
                "best_p_M": float("nan"),
                "best_p_V": float("nan"),
                "best_r_M": float("nan"),
                "best_r_V": float("nan"),
                "best_R_MV": float("nan"),
                "best_R_V_subscriber": float("nan"),
                "best_R_V_retained": float("nan"),
                "best_R_industry_net": float("nan"),
                "best_residual": float("nan"),
            }

        return {
            "C_V": float(best.C_V),
            "best_n_V": float(best.n_V),
            "best_n_M": float(best.n_M),
            "best_p_M": float(best.p_M),
            "best_p_V": float(best.p_V),
            "best_r_M": float(best.r_M),
            "best_r_V": float(best.r_V),
            "best_R_MV": float(best.R_MV),
            "best_R_V_subscriber": float(best.R_V_subscriber),
            "best_R_V_retained": float(best.R_V_retained),
            "best_R_industry_net": float(best.R_industry_net),
            "best_residual": float(best.residual),
        }

    def curve_at_capacity(self, C_V: float) -> list[dict[str, float]]:
        if C_V <= 0.0:
            return []
        rows: list[dict[str, float]] = []
        for n_V in self.iter_n_V_candidates(C_V):
            cand = self.evaluate_candidate(C_V, n_V)
            if cand is None:
                continue
            rows.append(
                {
                    "C_V": float(C_V),
                    "n_M": float(cand.n_M),
                    "n_V": float(cand.n_V),
                    "p_M": float(cand.p_M),
                    "p_V": float(cand.p_V),
                    "r_M": float(cand.r_M),
                    "r_V": float(cand.r_V),
                    "R_MV": float(cand.R_MV),
                    "R_V_subscriber": float(cand.R_V_subscriber),
                    "R_V_retained": float(cand.R_V_retained),
                    "R_industry_net": float(cand.R_industry_net),
                    "residual": float(cand.residual),
                }
            )
        return rows
