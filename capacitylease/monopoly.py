from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize_scalar

from .distributions import normal_cdf
from .models import ModelSpec
from .numerical import monotone_bisection


@dataclass
class MonopolyPoint:
    price: float
    subscribers: float
    revenue: float
    target_rate: float
    acceptance_probs: np.ndarray


class MonopolySolver:
    def __init__(self, spec: ModelSpec) -> None:
        self.spec = spec
        self._critical_price_cache: float | None = None

    def critical_price(self) -> float:
        if self._critical_price_cache is not None:
            return self._critical_price_cache

        rhs = self.spec.N_total - 1.0 / self.spec.delta

        def g(price: float) -> float:
            arg = self.spec.beta * price - self.spec.alpha * np.log(self.spec.C)
            return float(np.sum(self.spec.N_g * normal_cdf(arg, self.spec.eps_mu, self.spec.eps_sigma)) - rhs)

        lo, hi = 0.0, float(self.spec.search.get("monopoly_price_search_hi", 300.0))
        g_hi = g(hi)
        while g_hi < 0:
            hi *= 2.0
            g_hi = g(hi)
            if hi > 1e6:
                raise RuntimeError("Failed to bracket critical monopoly price.")

        self._critical_price_cache = float(monotone_bisection(g, lo, hi, tol=float(self.spec.search.get("root_tol", 1e-12))))
        return self._critical_price_cache

    def subscriber_count_at_price(self, price: float, critical_price: float | None = None) -> float:
        if critical_price is None:
            critical_price = self.critical_price()
        if price < 0 or price > critical_price:
            return float("nan")

        lo = 1.0 / self.spec.delta
        hi = self.spec.N_total

        def h(subscribers: float) -> float:
            arg = (
                self.spec.beta * price
                - self.spec.alpha * np.log(self.spec.C / self.spec.delta)
                + self.spec.alpha * np.log(subscribers)
            )
            return float(
                self.spec.N_total
                - np.sum(self.spec.N_g * normal_cdf(arg, self.spec.eps_mu, self.spec.eps_sigma))
                - subscribers
            )

        endpoint_tol = max(1e-10, 50.0 * float(self.spec.search.get("root_tol", 1e-12)))
        h_lo = h(lo)
        if abs(h_lo) <= endpoint_tol:
            return lo
        h_hi = h(hi)
        if abs(h_hi) <= endpoint_tol:
            return hi
        if h_lo * h_hi > 0:
            return lo if abs(h_lo) <= abs(h_hi) else hi
        return float(monotone_bisection(h, lo, hi, tol=float(self.spec.search.get("root_tol", 1e-12))))

    def point_at_price(self, price: float, critical_price: float | None = None) -> MonopolyPoint:
        if critical_price is None:
            critical_price = self.critical_price()
        subscribers = self.subscriber_count_at_price(price, critical_price=critical_price)
        if np.isnan(subscribers):
            return MonopolyPoint(
                price=price,
                subscribers=float("nan"),
                revenue=float("nan"),
                target_rate=float("nan"),
                acceptance_probs=np.full(self.spec.G, np.nan),
            )
        rate = self.spec.C / (self.spec.delta * subscribers)
        arg = self.spec.beta * price - self.spec.alpha * np.log(rate)
        acc = 1.0 - normal_cdf(arg, self.spec.eps_mu, self.spec.eps_sigma)
        revenue = price * subscribers
        return MonopolyPoint(
            price=price,
            subscribers=subscribers,
            revenue=revenue,
            target_rate=rate,
            acceptance_probs=np.asarray(acc, dtype=float),
        )

    def revenue_at_price(self, price: float, critical_price: float | None = None) -> float:
        point = self.point_at_price(price, critical_price=critical_price)
        if np.isnan(point.revenue):
            return float("-inf")
        return float(point.revenue)

    def sweep(self, critical_price: float | None = None) -> list[dict[str, float]]:
        if critical_price is None:
            critical_price = self.critical_price()
        plot_max = float(self.spec.search.get("monopoly_plot_price_max", critical_price))
        n_points = int(self.spec.search.get("monopoly_price_points", 3001))
        grid = np.linspace(0.0, plot_max, n_points)
        rows: list[dict[str, float]] = []
        for price in grid:
            point = self.point_at_price(float(price), critical_price=critical_price)
            row = {
                "price": float(price),
                "subscribers": float(point.subscribers),
                "revenue": float(point.revenue),
                "target_rate": float(point.target_rate),
            }
            for idx, prob in enumerate(point.acceptance_probs, start=1):
                row[f"A_group_{idx}"] = float(prob)
            rows.append(row)
        return rows

    def _grid_best(self, rows: list[dict[str, float]]) -> dict[str, float]:
        feasible = [row for row in rows if np.isfinite(row["revenue"])]
        best = max(feasible, key=lambda row: row["revenue"])
        return {
            "grid_optimal_price": float(best["price"]),
            "grid_optimal_subscribers": float(best["subscribers"]),
            "grid_optimal_revenue": float(best["revenue"]),
            "grid_optimal_target_rate": float(best["target_rate"]),
        }

    def exact_optimum(self, critical_price: float | None = None) -> dict[str, float]:
        if critical_price is None:
            critical_price = self.critical_price()

        xatol = float(self.spec.search.get("monopoly_opt_xatol", 1e-8))
        result = minimize_scalar(
            lambda p: -self.revenue_at_price(float(p), critical_price=critical_price),
            bounds=(0.0, critical_price),
            method="bounded",
            options={"xatol": xatol, "maxiter": 500},
        )

        candidate_prices = [0.0, critical_price, float(result.x)]
        best_point: MonopolyPoint | None = None
        for price in candidate_prices:
            point = self.point_at_price(price, critical_price=critical_price)
            if np.isnan(point.revenue):
                continue
            if best_point is None or point.revenue > best_point.revenue:
                best_point = point

        if best_point is None:
            raise RuntimeError("Failed to compute a feasible monopoly optimum.")

        return {
            "critical_price": float(critical_price),
            "optimal_price": float(best_point.price),
            "optimal_subscribers": float(best_point.subscribers),
            "optimal_revenue": float(best_point.revenue),
            "optimal_target_rate": float(best_point.target_rate),
            "optimizer_success": bool(result.success),
            "optimizer_nfev": int(getattr(result, "nfev", -1)),
            "optimizer_method": "scipy.optimize.minimize_scalar(method='bounded')",
        }

    def optimum(
        self,
        *,
        rows: list[dict[str, float]] | None = None,
        critical_price: float | None = None,
    ) -> dict[str, float]:
        if critical_price is None:
            critical_price = self.critical_price()
        optimum = self.exact_optimum(critical_price=critical_price)
        if rows is not None:
            optimum.update(self._grid_best(rows))
        return optimum
