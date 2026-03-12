from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ModelSpec:
    raw: dict[str, Any]

    def __post_init__(self) -> None:
        params = self.raw["parameters"]
        groups = params["groups"]
        object.__setattr__(self, "name", self.raw["name"])
        object.__setattr__(self, "description", self.raw.get("description", ""))
        object.__setattr__(self, "search", self.raw.get("search", {}))
        object.__setattr__(self, "diagnostics", self.raw.get("diagnostics", {}))
        object.__setattr__(self, "claims", self.raw.get("reference_claims", {}))
        object.__setattr__(self, "C", float(params["C"]))
        object.__setattr__(self, "delta", float(params["delta"]))
        object.__setattr__(self, "zeta", float(params["zeta"]))
        object.__setattr__(self, "lam", float(params["lambda"]))
        object.__setattr__(self, "pi_V", float(params["pi_V"]))

        names = [str(g["name"]) for g in groups]
        N_g = np.array([float(g["N"]) for g in groups], dtype=float)
        alpha = np.array([float(g["alpha"]) for g in groups], dtype=float)
        beta = np.array([float(g["beta"]) for g in groups], dtype=float)

        eps_mu = np.array([float(g["epsilon"]["mu"]) for g in groups], dtype=float)
        eps_sigma = np.array([float(g["epsilon"]["sigma"]) for g in groups], dtype=float)

        mu_M = np.array([float(g["epsilon_M"]["mu"]) for g in groups], dtype=float)
        sigma_M = np.array([float(g["epsilon_M"]["sigma"]) for g in groups], dtype=float)

        mu_V = np.array([float(g["epsilon_V"]["mu"]) for g in groups], dtype=float)
        sigma_V = np.array([float(g["epsilon_V"]["sigma"]) for g in groups], dtype=float)

        object.__setattr__(self, "group_names", tuple(names))
        object.__setattr__(self, "N_g", N_g)
        object.__setattr__(self, "alpha", alpha)
        object.__setattr__(self, "beta", beta)
        object.__setattr__(self, "eps_mu", eps_mu)
        object.__setattr__(self, "eps_sigma", eps_sigma)
        object.__setattr__(self, "mu_M", mu_M)
        object.__setattr__(self, "sigma_M", sigma_M)
        object.__setattr__(self, "mu_V", mu_V)
        object.__setattr__(self, "sigma_V", sigma_V)
        object.__setattr__(self, "N_total", float(N_g.sum()))
        object.__setattr__(self, "G", int(len(groups)))

    @property
    def min_subscribers(self) -> int:
        return int(np.ceil(1.0 / self.delta))

    @property
    def max_market_clearing_mvno_subscribers(self) -> int:
        return int(np.floor(self.N_total - 1.0 / self.delta))

    def to_dict(self) -> dict[str, Any]:
        return self.raw
