"""Estimating the impact of a cell outage from monitoring data alone.

What the estimator may use: per-second cell load and active users, cell capacities, the topology (neighbour lists),
the failed cell ids and the alarm window (which starts and ends late), and optionally handover statistics. It never
sees which users were lost or where they went.

Method, in four steps
1. Counterfactual: what would each cell have carried without the outage? Its mean load in a pre-period, optionally
   scaled over time by a robust (median) index built from *control* cells that are far from the outage.
2. Excess: observed minus counterfactual load over the alarm window. In a neighbour that absorbed users the mean
   excess is positive.
3. Placebo-in-space test: the same statistic computed for every control cell over the same window is a null
   distribution that has the same window length, time of day and burstiness. A neighbour is an absorber when its
   standardised excess beats the (1 - alpha) quantile of the controls' (leave-one-out).
4. Indicators: displaced traffic = counterfactual load of the failed cells; reconnected traffic = excess in the
   neighbours; loss fraction = 1 - reconnected / displaced; degradation = extra user-seconds below the served-fraction
   threshold, relative to the pre-period rate. Traffic is split between neighbours from the data, from handover
   statistics, or a blend.
"""
from __future__ import annotations

from collections import deque

import numpy as np

from .config import Outage
from .sim import CityLayout


def hop_distance(layout: CityLayout, sources: list[int]) -> np.ndarray:
    dist = np.full(layout.n_cells, 10**6, dtype=int)
    q: deque[int] = deque()
    for s in sources:
        dist[s] = 0
        q.append(s)
    while q:
        c = q.popleft()
        for n in layout.neighbors[c]:
            if dist[n] > dist[c] + 1:
                dist[n] = dist[c] + 1
                q.append(n)
    return dist


def moving_average(x: np.ndarray, width: int) -> np.ndarray:
    """Centred moving average along the last axis; windows shrink at the edges instead of padding."""
    T = x.shape[-1]
    cs = np.concatenate([np.zeros(x.shape[:-1] + (1,)), np.cumsum(x, axis=-1)], axis=-1)
    idx = np.arange(T)
    lo = np.maximum(0, idx - width // 2)
    hi = np.minimum(T, idx + width // 2 + 1)
    return (cs[..., hi] - cs[..., lo]) / (hi - lo)


def robust_level(x: np.ndarray, block: int) -> np.ndarray:
    """Median over non-overlapping block means, per row: a level estimate a surge covering under half the
    pre-period cannot drag upwards."""
    n = x.shape[-1] // block
    return np.median(x[..., : n * block].reshape(x.shape[:-1] + (n, block)).mean(axis=-1), axis=-1)


def prior_shares(layout: CityLayout, failed: list[int], neighbours: list[int], cf_rate: dict[int, float],
                 handover: list[np.ndarray]) -> dict[int, float]:
    """Neighbour shares implied by handover statistics, weighted by how much traffic each failed cell displaced."""
    acc = {j: 0.0 for j in neighbours}
    for c in failed:
        alive = [(j, handover[c][layout.neighbors[c].index(j)]) for j in layout.neighbors[c] if j not in failed]
        tot = sum(h for _, h in alive)
        if tot <= 0:
            continue
        for j, h in alive:
            acc[j] += max(cf_rate[c], 0.0) * h / tot
    s = sum(acc.values())
    return {j: (v / s if s > 0 else 1.0 / len(neighbours)) for j, v in acc.items()}


def blend(data_share: dict[int, float], prior: dict[int, float], lam: float) -> dict[int, float]:
    return {j: (1 - lam) * data_share[j] + lam * prior[j] for j in data_share}


def estimate(load: np.ndarray, users: np.ndarray, capacity: np.ndarray, layout: CityLayout, failed: list[int],
             a0: int, a1: int, o: Outage, method: str, with_series: bool = False) -> dict:
    """`method` is "level" (robust pre-period level as counterfactual) or "control" (level scaled by a control index)."""
    if method not in ("level", "control"):
        raise ValueError(method)
    p1 = a0 - o.guard_s
    p0 = p1 - o.pre_window_s
    if p0 < 0:
        raise ValueError("pre-period starts before the series does")
    dist = hop_distance(layout, failed)
    neighbours = [c for c in range(layout.n_cells) if dist[c] == 1]
    controls = [c for c in range(layout.n_cells) if dist[c] >= o.control_min_hops]
    if len(controls) < 5:
        raise ValueError("too few control cells")
    Wn = a1 - a0
    sl = slice(p0, a1)                                           # local frame: [pre | guard | window]
    L = load[:, sl]
    pre = slice(0, p1 - p0)
    win = slice(a0 - p0, a1 - p0)
    pre_mean = robust_level(L[:, pre], o.level_block_s)

    ratio = None
    if method == "control":
        smooth = moving_average(L[controls], o.control_smooth_s)
        ratio = smooth / pre_mean[controls, None]               # control cells' load relative to their own pre-period

    def index_excluding(k: int | None) -> np.ndarray:
        if method == "level":
            return np.ones(L.shape[1])
        rows = ratio if k is None else np.delete(ratio, controls.index(k), axis=0)
        g = np.median(rows, axis=0)
        return g / robust_level(g[pre][None, :], o.level_block_s)[0]

    g_all = index_excluding(None)

    def stats(cell: int, g: np.ndarray) -> tuple[float, float, float]:
        cf = pre_mean[cell] * g
        resid_pre = L[cell, pre] - cf[pre]
        sigma = float(1.4826 * np.median(np.abs(resid_pre - np.median(resid_pre)))) or 1e-9
        ebar = float((L[cell, win] - cf[win]).mean())
        return ebar, sigma, ebar / sigma

    ebar, sigma, u = {}, {}, {}
    for j in neighbours:
        ebar[j], sigma[j], u[j] = stats(j, g_all)
    u_ctrl, e_ctrl = [], []
    for k in controls:
        e, _, uk = stats(k, index_excluding(k))
        u_ctrl.append(uk)
        e_ctrl.append(e)
    u_ctrl = np.array(u_ctrl)
    q = float(np.quantile(u_ctrl, 1 - o.alpha, method="higher"))
    detected = [j for j in neighbours if u[j] > q and ebar[j] > 0]
    # leave-one-out false-positive rate on the controls: should sit near alpha if the null is honest
    loo = []
    for i in range(len(controls)):
        qi = float(np.quantile(np.delete(u_ctrl, i), 1 - o.alpha, method="higher"))
        loo.append(bool(u_ctrl[i] > qi and e_ctrl[i] > 0))

    # robust spread of the placebo statistic: turns each neighbour's sigma into a standard error of its window mean
    u_scale = float(max(1.4826 * np.median(np.abs(u_ctrl - np.median(u_ctrl))), 1e-9))
    se = {j: u_scale * sigma[j] for j in neighbours}

    cf_failed_rate = {c: float((pre_mean[c] * g_all[win]).mean()) for c in failed}
    pos = {j: max(ebar[j], 0.0) for j in neighbours}
    tot_pos = sum(pos.values())
    data_share = {j: (pos[j] / tot_pos if tot_pos > 0 else 1.0 / len(neighbours)) for j in neighbours}

    served = np.minimum(1.0, capacity[:, None] / np.maximum(L, 1e-9))
    deg_us = users[:, sl] * (served < o.degraded_served_fraction)
    deg_rate = float(sum(deg_us[j, win].mean() - deg_us[j, pre].mean() for j in neighbours))

    series = None
    if with_series:   # counterfactual curves for plotting, in the local frame that starts at p0
        series = {"t0": p0, "pre_end": p1 - p0, "win": (a0 - p0, a1 - p0),
                  "cf": {c: pre_mean[c] * g_all for c in list(failed) + neighbours}}
    return {
        "series": series, "method": method, "neighbours": neighbours, "n_controls": len(controls), "window_s": Wn,
        "displaced_rate_mbps": float(sum(cf_failed_rate.values())),
        "reconnected_rate_all_mbps": float(sum(ebar.values())),
        "reconnected_rate_detected_mbps": float(sum(ebar[j] for j in detected)),
        "ebar": ebar, "sigma": sigma, "se": se, "u": u, "q": q, "detected": detected,
        "data_share": data_share, "cf_failed_rate": cf_failed_rate,
        "degraded_excess_rate": deg_rate, "loo_false_positive_rate": float(np.mean(loo)),
    }


def gls_total(ebar: dict[int, float], se: dict[int, float], pi: dict[int, float], cap: float) -> float:
    """Total reconnected rate R from  ebar_j = R * pi_j + noise_j  (noise sd se_j): the generalised-least-squares
    estimate. Neighbours the prior says take little traffic contribute little weight, so their burst noise is not summed
    in. Clipped to [0, cap] (cap = displaced traffic: you cannot reconnect more than was displaced)."""
    num = sum(pi[j] * ebar[j] / se[j] ** 2 for j in ebar)
    den = sum(pi[j] ** 2 / se[j] ** 2 for j in ebar)
    return float(np.clip(num / den, 0.0, cap)) if den > 0 else 0.0


def loss_fraction(displaced: float, reconnected: float) -> float:
    return float(np.clip(1.0 - reconnected / displaced, 0.0, 1.0)) if displaced > 0 else 0.0
