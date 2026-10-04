"""Cell-outage scenarios on the digital twin, with exact ground truth.

A *world* is one simulated city without an outage (the counterfactual). A *scenario* takes one cell (or two
adjacent cells, a site-level fault) down for a while. Every affected user is either

* lost for the whole outage (no neighbouring cell takes them), or
* reconnected to one of the alive neighbouring cells after a reselection delay, drawn from fixed per-cell
  "reselection weights" (the part of the network a monitoring system does not see),

and returns to the repaired cell a little after it is back. Because the world is regenerated from seeds, the
counterfactual load of every cell is known exactly, so the traffic that was lost, rerouted (and to whom) and the
degradation it caused can be scored against truth. The estimator in `outage_est.py` sees only what a monitoring
system would: per-second cell load, active users, capacity, the alarm window and (optionally) noisy handover
statistics.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import numpy as np

from .config import Config
from .sim import CityLayout, build_layout, cell_capacity, draw_cell


@dataclass
class World:
    index: int
    cfg: Config                      # config carrying this world's seed (and the surge setting)
    layout: CityLayout
    diurnal: np.ndarray
    load: np.ndarray                 # [cells, T] Mbit/s with no outage (the counterfactual)
    users: np.ndarray                # [cells, T] active users with no outage
    capacity: np.ndarray             # [cells] Mbit/s
    routing: list[np.ndarray]        # true reselection weights per cell, aligned with layout.neighbors[c]


def build_world(cfg: Config, index: int) -> World:
    o = cfg.outage
    wcfg = dataclasses.replace(cfg, seed=cfg.seed + index * o.world_seed_stride)
    sim = wcfg.sim if o.use_surge_events else dataclasses.replace(wcfg.sim, events=[])
    wcfg = dataclasses.replace(wcfg, sim=sim)
    layout = build_layout(wcfg)
    T = sim.duration_s
    diurnal = 1.0 + sim.diurnal_amplitude * np.sin(2 * np.pi * np.arange(T) / sim.diurnal_period_s)
    C = layout.n_cells
    load, users, capacity = np.zeros((C, T)), np.zeros((C, T)), np.zeros(C)
    for cid in range(C):
        d = draw_cell(wcfg, sim, layout, cid)
        ld, on = np.zeros(T), np.zeros(T)
        for r, s in list(zip(d.rates, d.states)) + list(zip(d.extra_rates, d.extra_states)):
            on += s
            ld += r * s
        load[cid], users[cid] = ld * diurnal, on
        capacity[cid] = cell_capacity(sim, d.prof, d.n_src, d.rng)
    rrng = np.random.default_rng([wcfg.seed, 999])
    routing = [rrng.dirichlet(np.full(len(nb), o.routing_concentration)) if nb else np.zeros(0)
               for nb in layout.neighbors]
    return World(index, wcfg, layout, diurnal, load, users, capacity, routing)


def cell_sources(world: World, cid: int) -> tuple[np.ndarray, np.ndarray]:
    """Rates [n] and ON/OFF states [n, T] of every user of a cell, regenerated from the seeds."""
    d = draw_cell(world.cfg, world.cfg.sim, world.layout, cid)
    rates = np.concatenate([d.rates, d.extra_rates])
    states = np.vstack([np.asarray(s, dtype=np.float64) for s in d.states + d.extra_states])
    return rates, states


def handover_prior(world: World, sigma: float) -> list[np.ndarray]:
    """Historical handover shares per cell: the true reselection weights seen through log-normal noise.

    This stands in for the mobility statistics an operator has. It is correlated with, but not equal to, what users do
    when a cell dies (sigma controls how good a proxy it is)."""
    rng = np.random.default_rng([world.cfg.seed, 555, int(round(sigma * 1000))])
    out = []
    for w in world.routing:
        if len(w) == 0:
            out.append(w)
            continue
        h = w * rng.lognormal(0.0, sigma, len(w)) if sigma > 0 else w.copy()
        out.append(h / h.sum())
    return out


@dataclass
class Scenario:
    world: int
    index: int
    failed: list[int]
    t0: int                      # true outage window [t0, t1)
    t1: int
    a0: int                      # alarm window [a0, a1): what monitoring reports
    a1: int
    reconnect_prob: float
    no_coverage: bool

    @property
    def kind(self) -> str:
        return "no coverage" if self.no_coverage else ("pair" if len(self.failed) == 2 else "single")


def sample_scenarios(cfg: Config, world: World) -> list[Scenario]:
    o = cfg.outage
    T = world.cfg.sim.duration_s
    rng = np.random.default_rng([world.cfg.seed, 777])
    out = []
    for k in range(o.scenarios_per_world):
        no_cov = rng.random() < o.no_coverage_fraction
        pair = (not no_cov) and rng.random() < o.pair_fraction
        c = int(rng.integers(0, world.layout.n_cells))
        failed = [c]
        if pair:
            failed.append(int(rng.choice(world.layout.neighbors[c])))
        dur = int(rng.integers(o.duration_min_s, o.duration_max_s + 1))
        t0 = int(rng.integers(o.start_min_s, T - o.end_margin_s - dur + 1))
        t1 = t0 + dur
        a0 = t0 + int(rng.integers(0, o.alarm_start_delay_max_s + 1))
        a1 = t1 + int(rng.integers(0, o.alarm_clear_delay_max_s + 1))
        rp = 0.0 if no_cov else float(rng.uniform(o.reconnect_prob_min, o.reconnect_prob_max))
        out.append(Scenario(world.index, k, failed, t0, t1, a0, a1, rp, no_cov))
    return out


@dataclass
class Outcome:
    scenario: Scenario
    load: np.ndarray                 # [cells, T] what monitoring observes during the scenario
    users: np.ndarray
    neighbours: list[int]            # alive hex neighbours of the failed cells (candidate absorbers)
    truth: dict


def run_scenario(world: World, sc: Scenario) -> Outcome:
    o = world.cfg.outage
    lay = world.layout
    rng = np.random.default_rng([world.cfg.seed, 4242, sc.index])
    failed = set(sc.failed)
    T = world.load.shape[1]
    t_end = min(T, sc.t1 + o.return_delay_max_s + 1)
    tt = np.arange(sc.t0, t_end)
    sl = slice(sc.t0, t_end)
    inwin = (tt < sc.t1)[None, :]
    load, users = world.load.copy(), world.users.copy()
    neighbours = sorted({j for c in sc.failed for j in lay.neighbors[c] if j not in failed})

    displaced = 0.0
    rerouted: dict[int, float] = {j: 0.0 for j in neighbours}
    affected_users = reconnected_users = 0
    service_loss_user_s = 0.0
    for c in sc.failed:
        rates, S = cell_sources(world, c)
        S = S[:, sl]
        contrib = rates[:, None] * S * world.diurnal[sl][None, :]
        n = len(rates)
        alive = [j for j in lay.neighbors[c] if j not in failed]
        reconnect = np.zeros(n, dtype=bool)
        target = np.full(n, -1)
        if alive and sc.reconnect_prob > 0:
            w = np.array([world.routing[c][lay.neighbors[c].index(j)] for j in alive])
            w = w / w.sum()
            reconnect = rng.random(n) < sc.reconnect_prob
            target[reconnect] = rng.choice(alive, size=int(reconnect.sum()), p=w)
        delay = rng.integers(o.reconnect_delay_min_s, o.reconnect_delay_max_s + 1, n)
        back = sc.t1 + rng.integers(0, o.return_delay_max_s + 1, n)          # when each user re-attaches to c
        affected = tt[None, :] < back[:, None]                                # off cell c from t0 until `back`
        on_target = (tt[None, :] >= (sc.t0 + delay)[:, None]) & affected & (target[:, None] >= 0)

        load[c, sl] -= (contrib * affected).sum(0)
        users[c, sl] -= (S * affected).sum(0)
        for j in np.unique(target[target >= 0]):
            m = target == j
            load[j, sl] += (contrib[m] * on_target[m]).sum(0)
            users[j, sl] += (S[m] * on_target[m]).sum(0)
            rerouted[int(j)] += float((contrib[m] * on_target[m] * inwin).sum())
        displaced += float((contrib * inwin).sum())
        affected_users += n
        reconnected_users += int(reconnect.sum())
        service_loss_user_s += float((S * (~on_target) * inwin).sum())
    load = np.maximum(load, 0.0)
    users = np.maximum(users, 0.0)

    rerouted_total = sum(rerouted.values())
    shares = {j: (v / rerouted_total if rerouted_total > 0 else 0.0) for j, v in rerouted.items()}
    win = slice(sc.t0, sc.t1)
    tau = o.degraded_served_fraction

    def degraded(ld, us, ids):
        served = np.minimum(1.0, world.capacity[ids, None] / np.maximum(ld[ids][:, win], 1e-9))
        return (us[ids][:, win] * (served < tau)).sum()

    nb = np.array(neighbours, dtype=int)
    degraded_excess = float(degraded(load, users, nb) - degraded(world.load, world.users, nb)) if len(nb) else 0.0
    lost = displaced - rerouted_total
    dur = sc.t1 - sc.t0
    events = world.cfg.sim.events if o.use_surge_events else []
    groups = {lay.district_of_cell[c] for c in list(failed) + neighbours}
    confounded = any(e.district in groups and e.start_s < sc.t1 and e.start_s + e.duration_s > sc.t0 for e in events)
    truth = {
        "displaced_mbit": displaced, "reconnected_mbit": rerouted_total, "lost_mbit": lost,
        "displaced_rate_mbps": displaced / dur, "reconnected_rate_mbps": rerouted_total / dur,
        "loss_fraction": lost / displaced if displaced > 0 else 0.0,
        "rerouted_mbit": {int(j): v for j, v in rerouted.items()}, "shares": {int(j): v for j, v in shares.items()},
        "absorbers": sorted(int(j) for j, s in shares.items() if s >= o.absorber_min_share),
        "affected_users": affected_users, "reconnected_users": reconnected_users,
        "lost_user_share": 1.0 - reconnected_users / max(affected_users, 1),
        "service_loss_user_seconds": service_loss_user_s,
        "degraded_excess_user_seconds": degraded_excess, "degraded_excess_rate": degraded_excess / dur,
        "confounded": bool(confounded),
    }
    return Outcome(sc, load, users, neighbours, truth)
