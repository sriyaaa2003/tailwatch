"""Self-similarity: aggregated-variance Hurst exponent.

For a self-similar series Var(X^(m)) ~ m^(2H-2), where X^(m) is the series averaged over blocks of m samples.
H = 0.5 for short-memory noise, 0.5 < H < 1 for long-range dependence.
"""
from __future__ import annotations

import numpy as np


def hurst_aggvar(x: np.ndarray, min_scale: int, max_scale_div: int, n_scales: int = 12) -> float:
    x = np.asarray(x, dtype=np.float64)
    max_scale = len(x) // max_scale_div
    scales = np.unique(np.geomspace(min_scale, max_scale, n_scales).astype(int))
    logv = []
    for m in scales:
        k = len(x) // m
        blocks = x[: k * m].reshape(k, m).mean(axis=1)
        logv.append(np.log(blocks.var(ddof=1)))
    slope = np.polyfit(np.log(scales), logv, 1)[0]
    return float(1.0 + slope / 2.0)


