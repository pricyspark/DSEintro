import time

import numpy as np
from cv2.typing import MatLike
from scipy.optimize import nnls

from morph import dual_resize, morph


def calc_elapsed(
    source: MatLike,
    target: MatLike,
    crop_edge: int | None = None,
    downsample: int | float | None = None,
) -> float:
    if crop_edge is None and downsample is None:
        raise ValueError("Either crop_edge or downsample is required.")

    if crop_edge is not None:
        source_crop, target_crop = dual_resize(
            source, target, edge_length=crop_edge
        )
    else:
        source_crop, target_crop = dual_resize(source, target, downsample)
    start = time.perf_counter()
    morph(source_crop, target_crop)
    return time.perf_counter() - start


def benchmark_until(
    source: MatLike,
    target: MatLike,
    n_start: int = 1,
    total_budget: float = 10,
    single_budget: float = 5,
    max_edge: int | None = None,
) -> list[tuple[int, float]]:
    results = []
    total_time = 0.0
    edge = n_start

    while total_time < total_budget and (max_edge is None or edge <= max_edge):
        elapsed = calc_elapsed(source, target, crop_edge=edge)
        results.append((edge, elapsed))
        total_time += elapsed
        if elapsed >= single_budget:
            break
        edge += 1

    return results


def fit_poly(results: list[tuple[int, float]]) -> tuple[float, float, float]:
    edges = np.array([edge for edge, _ in results], dtype=float)
    elapsed = np.array([elapsed for _, elapsed in results], dtype=float)
    scale = edges.max()
    design = np.column_stack(
        [np.ones_like(edges), (edges / scale) ** 4, (edges / scale) ** 6]
    )
    a, b, c = nnls(design, elapsed)[0]
    return a, b / scale**4, c / scale**6


def estimate_elapsed(results: list[tuple[int, float]], edge: int) -> float:
    for measured_edge, elapsed in results:
        if measured_edge == edge:
            return elapsed

    a, b, c = fit_poly(results)
    return a + b * edge**4 + c * edge**6


def edge_for_elapsed(results: list[tuple[int, float]], target: float) -> int:
    a, b, c = fit_poly(results)
    if b == 0 and c == 0:
        return min(results, key=lambda result: abs(result[1] - target))[0]

    low = 1
    high = max(edge for edge, _ in results)
    while a + b * high**4 + c * high**6 < target:
        high *= 2

    while low + 1 < high:
        middle = (low + high) // 2
        if a + b * middle**4 + c * middle**6 <= target:
            low = middle
        else:
            high = middle

    if a + b * high**4 + c * high**6 <= target:
        return high
    return low
