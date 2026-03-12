from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Callable, Iterable


def parallel_map(
    func: Callable[[float], dict],
    values: Iterable[float],
    *,
    max_workers: int,
) -> list[dict]:
    values = list(values)
    if max_workers <= 1:
        return [func(value) for value in values]

    out: list[dict] = []
    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(func, value): value for value in values}
        for future in as_completed(futures):
            out.append(future.result())
    return out
