"""City digital twin: a hex-grid of cells whose load is a superposition of heavy-tailed ON/OFF sources.

Why this generator: aggregating many ON/OFF sources whose ON and OFF durations are Pareto with shape
1 < alpha < 2 gives long-range-dependent, self-similar traffic (Willinger et al. 1997), with theoretical
Hurst exponent H = (3 - alpha_min) / 2. This is a synthetic stand-in for real measurements and is
labelled as such everywhere; hurst.py checks the generator against its own theory.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

from .config import Config, District, Sim

_ODD_R = {
    0: [(0, -1), (0, 1), (-1, -1), (-1, 0), (1, -1), (1, 0)],  # even rows
    1: [(0, -1), (0, 1), (-1, 0), (-1, 1), (1, 0), (1, 1)],    # odd rows
}


def pareto_mean(alpha: float, minimum: float) -> float:
    return alpha * minimum / (alpha - 1.0)


def hex_neighbors(row: int, col: int, rows: int, cols: int) -> list[tuple[int, int]]:
    out = []
    for dr, dc in _ODD_R[row % 2]:
        r, c = row + dr, col + dc
        if 0 <= r < rows and 0 <= c < cols:
            out.append((r, c))
    return out


@dataclass(frozen=True)
class CityLayout:
    rows: int
    cols: int
    district_of_cell: list[str]
    neighbors: list[list[int]]

    @property
    def n_cells(self) -> int:
        return self.rows * self.cols

    def cell_id(self, row: int, col: int) -> int:
        return row * self.cols + col


def build_layout(cfg: Config) -> CityLayout:
    rows, cols = cfg.city.grid_rows, cfg.city.grid_cols
    by_col = {}
    for d in cfg.city.districts:
        for c in range(d.col_start, d.col_end):
            by_col[c] = d.name
    district_of_cell, neighbors = [], []
    for r in range(rows):
        for c in range(cols):
            district_of_cell.append(by_col[c])
            neighbors.append([nr * cols + nc for nr, nc in hex_neighbors(r, c, rows, cols)])
    return CityLayout(rows, cols, district_of_cell, neighbors)


def onoff_state(rng: np.random.Generator, n: int, alpha_on: float, alpha_off: float,
                min_on: float, min_off: float) -> np.ndarray:
    """Boolean ON/OFF series of length n with Pareto-distributed sojourn times."""
    m_on, m_off = pareto_mean(alpha_on, min_on), pareto_mean(alpha_off, min_off)
    duty = m_on / (m_on + m_off)
    pairs = int(1.5 * n / (m_on + m_off)) + 16
    while True:
        on = min_on * (1.0 + rng.pareto(alpha_on, pairs))
        off = min_off * (1.0 + rng.pareto(alpha_off, pairs))
        start_on = rng.random() < duty
        seg = np.empty(2 * pairs)
        seg[0::2], seg[1::2] = (on, off) if start_on else (off, on)
        bounds = np.cumsum(seg)
        if bounds[-1] >= n:
            break
        pairs *= 2
    # random phase so that the series does not always begin at a renewal point
    offset = rng.uniform(0.0, min(bounds[-1] - n, 10.0 * (m_on + m_off)))
    idx = np.searchsorted(bounds, np.arange(n) + offset, side="right")
    return ((idx + (0 if start_on else 1)) % 2) == 0


def event_mask(sim: Sim, district: str, n: int) -> np.ndarray:
    """Piecewise-linear 0..1 surge intensity for a district, summed over its scripted events (clipped to 1)."""
    t = np.arange(n, dtype=np.float64)
    total = np.zeros(n)
    for e in sim.events:
        if e.district != district:
            continue
        up = np.clip((t - e.start_s) / max(e.ramp_s, 1), 0.0, 1.0)
        down = np.clip((e.start_s + e.duration_s - t) / max(e.ramp_s, 1), 0.0, 1.0)
        total += np.minimum(up, down)
    return np.minimum(total, 1.0)


def _district(cfg: Config, name: str) -> District:
    return next(d for d in cfg.city.districts if d.name == name)


def simulate(cfg: Config, sim: Sim | None = None, cells: list[int] | None = None) -> pl.DataFrame:
    """Return a long frame sorted by (cell_id, t) at 1 s resolution.

    `sim` overrides cfg.sim (used by the Hurst validation to switch off diurnal pattern and events).
    `cells` restricts the simulation to a subset of cell ids.
    """
    sim = sim or cfg.sim
    layout = build_layout(cfg)
    n = sim.duration_s
    t = np.arange(n)
    diurnal = 1.0 + sim.diurnal_amplitude * np.sin(2 * np.pi * t / sim.diurnal_period_s)
    chosen = list(range(layout.n_cells)) if cells is None else cells
    frames = []
    for cid in chosen:
        # per-cell RNG stream: results do not depend on which other cells are simulated
        rng = np.random.default_rng([cfg.seed, cid])
        prof = _district(cfg, layout.district_of_cell[cid])
        n_src = max(1, int(round(prof.n_sources * rng.uniform(0.85, 1.15))))
        mu = np.log(prof.rate_mean_mbps) - 0.5 * prof.rate_sigma ** 2
        rates = rng.lognormal(mu, prof.rate_sigma, n_src)
        on_total = np.zeros(n)
        load = np.zeros(n)
        for r in rates:
            s = onoff_state(rng, n, prof.on_alpha, prof.off_alpha, sim.on_min_s, sim.off_min_s)
            on_total += s
            load += r * s
        # surge sources: gated by the event envelope
        mask = event_mask(sim, layout.district_of_cell[cid], n)
        extra_factor = max((e.extra_source_factor for e in sim.events
                            if e.district == layout.district_of_cell[cid]), default=0.0)
        n_extra = int(round(extra_factor * prof.n_sources))
        if n_extra and mask.any():
            for r in rng.lognormal(mu, prof.rate_sigma, n_extra):
                s = onoff_state(rng, n, prof.on_alpha, prof.off_alpha, sim.on_min_s, sim.off_min_s) * mask
                on_total += s
                load += r * s
        load = load * diurnal
        duty = pareto_mean(prof.on_alpha, sim.on_min_s) / (
            pareto_mean(prof.on_alpha, sim.on_min_s) + pareto_mean(prof.off_alpha, sim.off_min_s))
        expected_mean = n_src * prof.rate_mean_mbps * duty
        capacity = expected_mean / prof.target_mean_util * rng.lognormal(0.0, sim.capacity_jitter_sigma)
        util = load / capacity
        rsrp = (sim.rsrp_base_dbm + rng.normal(0, 2.0)
                + sim.rsrp_util_slope_db * np.clip(util, 0, 1.5) + rng.normal(0, sim.rsrp_noise_db, n))
        row, col = divmod(cid, layout.cols)
        frames.append(pl.DataFrame({
            "cell_id": np.full(n, cid, dtype=np.int32),
            "t": t.astype(np.int32),
            "district": np.full(n, layout.district_of_cell[cid]),
            "row": np.full(n, row, dtype=np.int16),
            "col": np.full(n, col, dtype=np.int16),
            "users": on_total.astype(np.float32),
            "load_mbps": load.astype(np.float32),
            "throughput_mbps": np.minimum(load, capacity).astype(np.float32),
            "capacity_mbps": np.full(n, capacity, dtype=np.float32),
            "rsrp_dbm": rsrp.astype(np.float32),
            "util": util.astype(np.float32),
        }))
    return pl.concat(frames)
