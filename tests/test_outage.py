import dataclasses

import numpy as np
import pytest

from tailwatch.config import load_config
from tailwatch.outage_est import (blend, estimate, gls_total, hop_distance, loss_fraction, moving_average,
                                  prior_shares, robust_level)
from tailwatch.outage_sim import (Scenario, build_world, cell_sources, handover_prior, run_scenario,
                                  sample_scenarios)
from tailwatch.sim import build_layout


@pytest.fixture(scope="module")
def cfg():
    return load_config()


@pytest.fixture(scope="module")
def world(cfg):
    return build_world(cfg, 0)


def scenario(world, failed, *, prob=0.7, t0=5000, dur=600, no_cov=False):
    return Scenario(0, 99, failed, t0, t0 + dur, t0 + 10, t0 + dur + 10, prob, no_cov)


def test_regenerated_users_reproduce_the_world_baseline(world):
    rates, S = cell_sources(world, 12)
    load = (rates @ S) * world.diurnal
    assert np.allclose(load, world.load[12], rtol=1e-9, atol=1e-9)


def test_displaced_traffic_is_exactly_lost_plus_rerouted(world):
    for failed in ([9], [19, 20]):
        sc = scenario(world, failed)
        tr = run_scenario(world, sc).truth
        baseline = sum(world.load[c, sc.t0:sc.t1].sum() for c in failed)
        assert tr["displaced_mbit"] == pytest.approx(baseline, rel=1e-9)
        assert tr["displaced_mbit"] == pytest.approx(tr["lost_mbit"] + tr["reconnected_mbit"], rel=1e-9)
        assert 0 <= tr["loss_fraction"] <= 1


def test_nothing_changes_outside_the_outage_and_load_stays_non_negative(world):
    sc = scenario(world, [9])
    out = run_scenario(world, sc)
    assert (out.load >= 0).all() and (out.users >= 0).all()
    end = sc.t1 + world.cfg.outage.return_delay_max_s + 1
    assert np.array_equal(out.load[:, :sc.t0], world.load[:, :sc.t0])
    assert np.array_equal(out.load[:, end:], world.load[:, end:])
    far = [c for c in range(world.layout.n_cells) if c not in [9] + out.neighbours]
    assert np.allclose(out.load[far], world.load[far])          # cells that are not neighbours are untouched


def test_failed_cell_is_dark_during_the_outage_and_only_neighbours_gain(world):
    sc = scenario(world, [9])
    out = run_scenario(world, sc)
    assert out.load[9, sc.t0:sc.t1].max() == pytest.approx(0.0, abs=1e-6)
    gain = out.load - world.load
    assert set(np.flatnonzero(gain[:, sc.t0:sc.t1].max(axis=1) > 1e-9)) <= set(out.neighbours)


def test_users_never_reconnect_to_a_failed_cell_in_a_site_outage(world):
    c = 19
    pair = [c, world.layout.neighbors[c][0]]
    out = run_scenario(world, scenario(world, pair, prob=1.0))
    assert not (set(pair) & set(out.neighbours))
    assert set(map(int, out.truth["rerouted_mbit"])) == set(out.neighbours)
    assert (out.load[pair, 5100:5500] < 1e-6).all()


def test_no_coverage_means_everything_is_lost(world):
    tr = run_scenario(world, scenario(world, [9], prob=0.0, no_cov=True)).truth
    assert tr["reconnected_mbit"] == 0 and tr["loss_fraction"] == pytest.approx(1.0) and tr["absorbers"] == []


def test_instant_full_reconnection_conserves_total_traffic(world, cfg):
    o = dataclasses.replace(cfg.outage, reconnect_delay_min_s=0, reconnect_delay_max_s=0, return_delay_max_s=0)
    w = dataclasses.replace(world, cfg=dataclasses.replace(world.cfg, outage=o))
    sc = scenario(w, [9], prob=1.0)
    out = run_scenario(w, sc)
    assert np.allclose(out.load.sum(axis=0), w.load.sum(axis=0), rtol=1e-9)
    assert out.truth["loss_fraction"] == pytest.approx(0.0, abs=1e-9)


def test_scenarios_and_worlds_are_deterministic_and_distinct(world, cfg):
    a = run_scenario(world, scenario(world, [9]))
    b = run_scenario(world, scenario(world, [9]))
    assert np.array_equal(a.load, b.load) and a.truth == b.truth
    other = build_world(cfg, 1)
    assert not np.allclose(other.load[0], world.load[0])
    s1, s2 = sample_scenarios(cfg, world), sample_scenarios(cfg, world)
    assert [(s.failed, s.t0, s.t1) for s in s1] == [(s.failed, s.t0, s.t1) for s in s2]


def test_sampled_scenarios_respect_the_configured_bounds(world, cfg):
    o = cfg.outage
    for s in sample_scenarios(cfg, world):
        assert o.start_min_s <= s.t0 and s.t1 <= world.load.shape[1] - o.end_margin_s
        assert o.duration_min_s <= s.t1 - s.t0 <= o.duration_max_s
        assert s.t0 <= s.a0 <= s.t0 + o.alarm_start_delay_max_s and s.t1 <= s.a1 <= s.t1 + o.alarm_clear_delay_max_s
        if len(s.failed) == 2:
            assert s.failed[1] in world.layout.neighbors[s.failed[0]]


def test_handover_prior_is_a_noisy_view_of_the_truth(world):
    exact = handover_prior(world, 0.0)
    assert all(np.allclose(e, r) for e, r in zip(exact, world.routing))
    noisy = handover_prior(world, 1.0)
    assert all(abs(n.sum() - 1) < 1e-9 for n in noisy if len(n))
    assert not np.allclose(noisy[10], world.routing[10])


# ------------------------------------------------------------------------------------------------ estimator
def test_hop_distance_and_moving_average():
    lay = build_layout(load_config())
    d = hop_distance(lay, [0])
    assert d[0] == 0 and all(d[n] == 1 for n in lay.neighbors[0])
    assert np.allclose(moving_average(np.arange(10.0)[None, :], 3)[0][1:-1], np.arange(1.0, 9.0))


def test_robust_level_ignores_a_surge_under_half_the_pre_period():
    x = np.ones(1200)
    x[200:500] = 3.0                                           # a surge over a quarter of the series
    assert robust_level(x[None, :], 60)[0] == pytest.approx(1.0)
    assert x.mean() > 1.4                                       # the plain mean would have been dragged up


def test_gls_total_recovers_the_total_when_the_prior_is_exact_and_clips_to_the_cap():
    pi = {1: 0.5, 2: 0.3, 3: 0.2}
    R = 12.0
    ebar = {j: R * s for j, s in pi.items()}
    se = {1: 1.0, 2: 2.0, 3: 0.5}
    assert gls_total(ebar, se, pi, cap=100) == pytest.approx(R)
    assert gls_total(ebar, se, pi, cap=5.0) == 5.0
    assert gls_total({1: -3.0}, {1: 1.0}, {1: 1.0}, cap=10) == 0.0


def test_blend_and_loss_fraction_edges():
    assert blend({1: 1.0, 2: 0.0}, {1: 0.0, 2: 1.0}, 0.25) == {1: 0.75, 2: 0.25}
    assert loss_fraction(10.0, 4.0) == pytest.approx(0.6)
    assert loss_fraction(10.0, 25.0) == 0.0 and loss_fraction(0.0, 1.0) == 0.0


def synthetic(cfg, extra=None, seed=0):
    """Constant-ish loads with small noise on the real layout; a failed cell goes dark in the window and
    neighbours gain `extra[j]` Mbit/s there."""
    o = cfg.outage
    lay = build_layout(cfg)
    T = 6000
    rng = np.random.default_rng(seed)
    load = 30.0 + rng.normal(0, 1.0, (lay.n_cells, T))
    failed, a0, a1 = [20], 4200, 5000
    load[failed[0], a0:a1] = 0.0
    for j, e in (extra or {}).items():
        load[j, a0:a1] += e
    users = np.maximum(load / 2, 0)
    return lay, load, users, np.full(lay.n_cells, 60.0), failed, a0, a1, o


@pytest.mark.parametrize("method", ["level", "control"])
def test_estimator_recovers_displaced_and_reconnected_traffic_on_clean_data(cfg, method):
    lay0 = build_layout(cfg)
    nb = lay0.neighbors[20]
    extra = {nb[0]: 8.0, nb[1]: 4.0}
    lay, load, users, cap, failed, a0, a1, o = synthetic(cfg, extra)
    e = estimate(load, users, cap, lay, failed, a0, a1, o, method)
    assert e["displaced_rate_mbps"] == pytest.approx(30.0, abs=0.5)
    assert e["reconnected_rate_all_mbps"] == pytest.approx(12.0, abs=1.0)
    found = set(e["detected"])
    assert {nb[0], nb[1]} <= found                              # real absorbers are always found
    assert len(found - {nb[0], nb[1]}) <= 1                     # a 5% test may flag the odd non-absorber by chance


def test_estimator_flags_nothing_when_nobody_reconnects_and_placebo_fpr_is_near_alpha(cfg):
    fpr, flagged = [], 0
    for seed in range(40):
        lay, load, users, cap, failed, a0, a1, o = synthetic(cfg, None, seed)
        e = estimate(load, users, cap, lay, failed, a0, a1, o, "control")
        flagged += bool(e["detected"])
        fpr.append(e["loo_false_positive_rate"])
    assert np.mean(fpr) == pytest.approx(cfg.outage.alpha, abs=0.03)
    assert flagged <= 14                                        # 6 neighbours at alpha=0.05: a few chance flags, not many


def test_estimator_never_reads_the_alarm_guard_or_future_data_for_its_baseline(cfg):
    lay, load, users, cap, failed, a0, a1, o = synthetic(cfg)
    base = estimate(load, users, cap, lay, failed, a0, a1, o, "level")
    poisoned = load.copy()
    poisoned[:, a1:] = 1e6                                      # data after the alarm window must not matter
    poisoned[:, a0 - o.guard_s:a0] = 1e6                        # nor the guard interval before the alarm
    assert estimate(poisoned, users, cap, lay, failed, a0, a1, o, "level")["displaced_rate_mbps"] == pytest.approx(
        base["displaced_rate_mbps"])


def test_prior_shares_are_a_distribution_over_the_neighbours(cfg, world):
    c = 19
    nb = sorted(world.layout.neighbors[c])
    H = handover_prior(world, 0.5)
    pi = prior_shares(world.layout, [c], nb, {c: 25.0}, H)
    assert set(pi) == set(nb) and sum(pi.values()) == pytest.approx(1.0)
