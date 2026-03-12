from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _save_all(fig: plt.Figure, base_path: str | Path) -> None:
    base_path = Path(base_path)
    base_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(base_path.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(base_path.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(base_path.with_suffix(".eps"), bbox_inches="tight")
    plt.close(fig)


def plot_monopoly_nr(rows: list[dict[str, float]], base_path: str | Path) -> None:
    prices = np.array([row["price"] for row in rows], dtype=float)
    subscribers = np.array([row["subscribers"] for row in rows], dtype=float)
    revenue = np.array([row["revenue"] for row in rows], dtype=float)

    fig, ax1 = plt.subplots(figsize=(7.0, 4.5))
    ax1.plot(prices, subscribers, linewidth=2.0, label="Subscribers")
    ax1.set_xlabel("Price")
    ax1.set_ylabel("Subscribers")

    ax2 = ax1.twinx()
    ax2.plot(prices, revenue, linewidth=2.0, linestyle="--", label="Revenue")
    ax2.set_ylabel("Revenue")

    ax1.grid(True, alpha=0.3)
    _save_all(fig, base_path)


def plot_monopoly_acceptance_rate(rows: list[dict[str, float]], base_path: str | Path, group_names: Iterable[str]) -> None:
    prices = np.array([row["price"] for row in rows], dtype=float)
    target_rate = np.array([row["target_rate"] for row in rows], dtype=float)

    fig, ax1 = plt.subplots(figsize=(7.0, 4.5))
    for idx, _group_name in enumerate(group_names, start=1):
        probs = np.array([row.get(f"A_group_{idx}", np.nan) for row in rows], dtype=float)
        ax1.plot(prices, probs, linewidth=1.8)

    ax1.set_xlabel("Price")
    ax1.set_ylabel("Acceptance probability")

    ax2 = ax1.twinx()
    ax2.plot(prices, target_rate, linewidth=2.0, linestyle="--")
    ax2.set_ylabel("Target rate")

    ax1.grid(True, alpha=0.3)
    _save_all(fig, base_path)


def plot_capacity_revenue(
    market_rows: list[dict[str, float]],
    flex_rows: list[dict[str, float]],
    monopoly_revenue: float,
    base_path: str | Path,
) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    ax.plot([row["C_V"] for row in market_rows], [row["best_R_MV"] for row in market_rows], linewidth=2.0)
    ax.plot([row["C_V"] for row in flex_rows], [row["best_R_MV"] for row in flex_rows], linewidth=2.0, linestyle="--")
    ax.axhline(monopoly_revenue, linewidth=1.5, linestyle=":")
    ax.set_xlabel("Leased capacity C_V (Mbps)")
    ax.set_ylabel("MNO cooperative revenue")
    ax.grid(True, alpha=0.3)
    _save_all(fig, base_path)


def plot_capacity_prices(market_rows: list[dict[str, float]], flex_rows: list[dict[str, float]], base_path: str | Path) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    x_market = [row["C_V"] for row in market_rows]
    x_flex = [row["C_V"] for row in flex_rows]
    ax.plot(x_market, [row["best_p_M"] for row in market_rows], linewidth=2.0)
    ax.plot(x_market, [row["best_p_V"] for row in market_rows], linewidth=2.0)
    ax.plot(x_flex, [row["best_p_M"] for row in flex_rows], linewidth=2.0, linestyle="--")
    ax.plot(x_flex, [row["best_p_V"] for row in flex_rows], linewidth=2.0, linestyle="--")
    ax.set_xlabel("Leased capacity C_V (Mbps)")
    ax.set_ylabel("Price")
    ax.grid(True, alpha=0.3)
    _save_all(fig, base_path)


def plot_revenue_vs_nM(market_curve_rows: list[dict[str, float]], flex_curve_rows: list[dict[str, float]], base_path: str | Path) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    ax.plot([row["n_M"] for row in market_curve_rows], [row["R_MV"] for row in market_curve_rows], linewidth=2.0)
    ax.plot([row["n_M"] for row in market_curve_rows], [row["R_V_retained"] for row in market_curve_rows], linewidth=2.0, linestyle=":")
    ax.plot([row["n_M"] for row in flex_curve_rows], [row["R_MV"] for row in flex_curve_rows], linewidth=2.0, linestyle="--")
    ax.plot([row["n_M"] for row in flex_curve_rows], [row["R_V_retained"] for row in flex_curve_rows], linewidth=2.0, linestyle="-.")
    ax.set_xlabel("MNO subscribers n_M")
    ax.set_ylabel("Revenue")
    ax.grid(True, alpha=0.3)
    _save_all(fig, base_path)


def plot_prices_vs_nM(market_curve_rows: list[dict[str, float]], flex_curve_rows: list[dict[str, float]], base_path: str | Path) -> None:
    fig, ax1 = plt.subplots(figsize=(7.0, 4.5))
    ax1.plot([row["n_M"] for row in market_curve_rows], [row["p_M"] for row in market_curve_rows], linewidth=2.0)
    ax1.plot([row["n_M"] for row in flex_curve_rows], [row["p_M"] for row in flex_curve_rows], linewidth=2.0, linestyle="--")
    ax1.set_xlabel("MNO subscribers n_M")
    ax1.set_ylabel("MNO price p_M")

    ax2 = ax1.twinx()
    ax2.plot([row["n_M"] for row in market_curve_rows], [row["p_V"] for row in market_curve_rows], linewidth=2.0, linestyle=":")
    ax2.plot([row["n_M"] for row in flex_curve_rows], [row["p_V"] for row in flex_curve_rows], linewidth=2.0, linestyle="-.")
    ax2.set_ylabel("MVNO price p_V")
    ax1.grid(True, alpha=0.3)
    _save_all(fig, base_path)
