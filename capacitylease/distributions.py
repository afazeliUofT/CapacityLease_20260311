from __future__ import annotations

from functools import lru_cache
from typing import Iterable

import numpy as np
from scipy.special import ndtr, ndtri
from scipy.stats import multivariate_normal

try:
    from scipy.stats._qmvnt import _bvn as _fast_bvn  # type: ignore[attr-defined]
except Exception:  # pragma: no cover
    _fast_bvn = None


def normal_cdf(x: np.ndarray | float, mu: np.ndarray | float, sigma: np.ndarray | float) -> np.ndarray | float:
    return ndtr((np.asarray(x) - mu) / sigma)


def normal_ppf(q: np.ndarray | float, mu: np.ndarray | float, sigma: np.ndarray | float) -> np.ndarray | float:
    return mu + sigma * ndtri(q)


@lru_cache(maxsize=128)
def _corr_cov(rho: float) -> np.ndarray:
    return np.array([[1.0, rho], [rho, 1.0]], dtype=float)


def bivariate_rect_prob(
    lower: Iterable[float],
    upper: Iterable[float],
    mean: Iterable[float],
    cov: np.ndarray,
) -> float:
    lower_vec = np.asarray(list(lower), dtype=float)
    upper_vec = np.asarray(list(upper), dtype=float)
    mean_vec = np.asarray(list(mean), dtype=float)

    if _fast_bvn is not None:
        shifted_lower = lower_vec - mean_vec
        shifted_upper = upper_vec - mean_vec
        return float(_fast_bvn(shifted_lower, shifted_upper, np.asarray(cov, dtype=float)))

    # Inclusion-exclusion fallback
    c00 = multivariate_normal.cdf(upper_vec, mean=mean_vec, cov=cov)
    c10 = multivariate_normal.cdf(np.array([lower_vec[0], upper_vec[1]]), mean=mean_vec, cov=cov)
    c01 = multivariate_normal.cdf(np.array([upper_vec[0], lower_vec[1]]), mean=mean_vec, cov=cov)
    c11 = multivariate_normal.cdf(lower_vec, mean=mean_vec, cov=cov)
    return float(c00 - c10 - c01 + c11)


def standard_bivariate_rect_prob(lower1: float, upper1: float, lower2: float, upper2: float, rho: float) -> float:
    cov = _corr_cov(float(rho))
    return bivariate_rect_prob(
        lower=[lower1, lower2],
        upper=[upper1, upper2],
        mean=[0.0, 0.0],
        cov=cov,
    )
