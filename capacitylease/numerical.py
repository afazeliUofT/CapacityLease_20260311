from __future__ import annotations

import math
from typing import Callable


def monotone_bisection(
    func: Callable[[float], float],
    lo: float,
    hi: float,
    *,
    tol: float = 1e-12,
    max_iter: int = 200,
) -> float:
    f_lo = func(lo)
    f_hi = func(hi)

    if abs(f_lo) <= tol:
        return lo
    if abs(f_hi) <= tol:
        return hi
    if f_lo * f_hi > 0:
        raise ValueError(f"Root not bracketed: f(lo)={f_lo}, f(hi)={f_hi}, lo={lo}, hi={hi}")

    a, b = lo, hi
    fa, fb = f_lo, f_hi
    for _ in range(max_iter):
        mid = 0.5 * (a + b)
        fm = func(mid)
        if abs(fm) <= tol or abs(b - a) <= tol:
            return mid
        if fa * fm <= 0:
            b, fb = mid, fm
        else:
            a, fa = mid, fm
    return 0.5 * (a + b)


def expand_bracket(
    func: Callable[[float], float],
    start_lo: float,
    start_hi: float,
    *,
    expansion: float = 2.0,
    max_steps: int = 60,
) -> tuple[float, float]:
    lo, hi = start_lo, start_hi
    f_lo = func(lo)
    f_hi = func(hi)
    if f_lo == 0:
        return lo, lo
    if f_hi == 0:
        return hi, hi

    for _ in range(max_steps):
        if f_lo * f_hi <= 0:
            return lo, hi
        if abs(f_lo) < abs(f_hi):
            hi *= expansion
            f_hi = func(hi)
        else:
            lo *= expansion
            f_lo = func(lo)
    raise ValueError("Could not bracket root after expansion.")


def finite_difference_slope(func: Callable[[float], float], x0: float, *, h: float = 1e-6) -> float:
    return (func(x0 + h) - func(x0 - h)) / (2.0 * h)


def safe_log(x: float) -> float:
    if x <= 0:
        raise ValueError(f"log argument must be positive, got {x}")
    return math.log(x)
