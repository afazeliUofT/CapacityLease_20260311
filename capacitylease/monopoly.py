from __future__ import annotations

from dataclasses import dataclass

import numpy as np

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

    def critical_price(self) -> float:
        rhs = self.spec.N_total - 1.0 / self.spec.delta

        def g(price: float) -> float:
            arg = self.spec.beta * price - self.spec.alpha * np.log(self.spec.C)
            return float(np.sum(self.spec.N_g * normal_cdf(arg, self.spec.eps_mu, self.spec.eps_sigma)) - rhs)

        lo, hi = 0.0, float(self.spec.search.get("monopoly_price_search_hi", 300.0))
        g_lo = g(lo)
        g_hi = g(hi)
        while g_hi < 0:
            hi *= 2.0
            g_hi = g(hi)
            if hi > 1e6:
                raise RuntimeError("Failed to bracket critical monopoly price.")
        return monotone_bisection(g, lo, hi, tol=float(self.spec.search.get("root_tol", 1e-12)))

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
            return float(self.spec.N_total - np.sum(self.spec.N_g * normal_cdf(arg, self.spec.eps_mu, self.spec.eps_sigma)) - subscribers)

        h_lo = h(lo)
        if abs(h_lo) <= 1e-12:
            return lo
        h_hi = h(hi)
        if abs(h_hi) <= 1e-12:
            return hi
        return monotone_bisection(h, lo, hi, tol=float(self.spec.search.get("root_tol", 1e-12)))

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
        return MonopolyPoint(price=price, subscribers=subscribers, revenue=revenue, target_rate=rate, acceptance_probs=np.asarray(acc, dtype=float))

    def sweep(self) -> list[dict[str, float]]:
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

    def optimum(self) -> dict[str, float]:
        rows = self.sweep()
        feasible = [row for row in rows if np.isfinite(row["revenue"])]
        best = max(feasible, key=lambda row: row["revenue"])
        return {
            "critical_price": float(self.critical_price()),
            "optimal_price": float(best["price"]),
            "optimal_subscribers": float(best["subscribers"]),
            "optimal_revenue": float(best["revenue"]),
            "optimal_target_rate": float(best["target_rate"]),
        }
