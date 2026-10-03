"""Self-similarity check for the generator: aggregated-variance Hurst exponent.

For a self-similar series Var(X^(m)) ~ m^(2H-2), where X^(m) is the series averaged over blocks of m samples.
H = 0.5 for short-memory noise, 0.5 < H < 1 for long-range dependence.
"""
from __future__ import annotations

import dataclasses

import numpy as np

from .config import Config
from .sim import simulate


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


def theoretical_hurst(alpha_on: float, alpha_off: float) -> float:
    return (3.0 - min(alpha_on, alpha_off)) / 2.0


def validate_generator(cfg: Config) -> dict:
    """Simulate with the diurnal pattern and events switched off, so only the ON/OFF burst structure remains."""
    plain = dataclasses.replace(cfg.sim, events=[], diurnal_amplitude=0.0)
    # spread over districts: rows are cycled so every district is represented
    per_district = max(1, cfg.hurst.validate_cells // len(cfg.city.districts))
    cells = [r * cfg.city.grid_cols + d.col_start for r in range(per_district) for d in cfg.city.districts]
    df = simulate(cfg, sim=plain, cells=cells)
    rows = []
    for cid in cells:
        load = df.filter(df["cell_id"] == cid)["load_mbps"].to_numpy()
        dist = next(d for d in cfg.city.districts if d.name == df.filter(df["cell_id"] == cid)["district"][0])
        rows.append({
            "cell_id": cid, "district": dist.name,
            "H_measured": hurst_aggvar(load, cfg.hurst.min_scale_s, cfg.hurst.max_scale_div),
            "H_theory": theoretical_hurst(dist.on_alpha, dist.off_alpha),
        })
    # reference: iid noise with the same marginal variance should give H ~ 0.5
    rng = np.random.default_rng(cfg.seed)
    iid = rng.normal(size=cfg.sim.duration_s)
    return {"cells": rows, "H_iid_reference": hurst_aggvar(iid, cfg.hurst.min_scale_s, cfg.hurst.max_scale_div)}
