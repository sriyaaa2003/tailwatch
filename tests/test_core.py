import dataclasses

import numpy as np
import polars as pl
import pytest

from tailwatch.config import ConfigError, load_config, validate
from tailwatch.core import Calibrator, best_f1_threshold, expected_calibration_error, lodo_folds
from tailwatch.features import build_features, feature_names
from tailwatch.hurst import hurst_aggvar
from tailwatch.probabilistic import conformal_margin
from tailwatch.sim import build_layout, hex_neighbors, simulate


@pytest.fixture(scope="module")
def cfg():
    c = load_config()
    sim = dataclasses.replace(c.sim, duration_s=1500, events=[], diurnal_period_s=1500)
    return dataclasses.replace(c, sim=sim)


@pytest.fixture(scope="module")
def raw(cfg):
    return simulate(cfg, cells=[0, 1, 2, 9])


def test_simulation_is_deterministic_and_cell_independent(cfg, raw):
    again = simulate(cfg, cells=[2])
    a = raw.filter(pl.col("cell_id") == 2)["load_mbps"].to_numpy()
    assert np.array_equal(a, again["load_mbps"].to_numpy())


def test_utilisation_is_load_over_capacity(raw):
    r = raw.row(100, named=True)
    assert r["util"] == pytest.approx(r["load_mbps"] / r["capacity_mbps"], rel=1e-4)


def test_hex_neighbors_are_symmetric(cfg):
    layout = build_layout(cfg)
    for a, nbrs in enumerate(layout.neighbors):
        for b in nbrs:
            assert a in layout.neighbors[b]
    assert len(hex_neighbors(2, 3, 6, 8)) == 6


def test_labels_match_naive_forward_window_and_features_do_not_peek(cfg, raw):
    feats = build_features(raw, cfg)
    H, thr = cfg.storm.horizon_s, cfg.storm.utilisation_threshold
    u = raw.filter(pl.col("cell_id") == 1).sort("t")["util"].to_numpy()
    sub = feats.filter(pl.col("cell_id") == 1)
    assert sub.height > 0
    for t, y, fut in zip(sub["t"].to_list()[:60], sub["y"].to_list()[:60], sub["y_fut_max"].to_list()[:60]):
        window = u[t + 1: t + H + 1]
        assert fut == pytest.approx(window.max(), rel=1e-5)
        assert y == int(window.max() >= thr)
        assert u[t] < thr                      # already-congested rows are excluded
    # trailing-window feature uses only the past
    w = cfg.features.windows_s[0]
    t0 = sub["t"].to_list()[10]
    assert sub.filter(pl.col("t") == t0)[f"util_mean_{w}"][0] == pytest.approx(u[t0 - w + 1: t0 + 1].mean(), rel=1e-4)


def test_features_have_no_nulls_and_expected_columns(cfg, raw):
    feats = build_features(raw, cfg)
    assert feats.select(feature_names(cfg)).null_count().sum_horizontal()[0] == 0


def test_lodo_folds_never_leak_district_or_future(cfg, raw):
    feats = build_features(raw, cfg)
    for f in lodo_folds(feats, cfg):
        d = feats["district"].to_numpy()
        assert not np.any(d[f.train] == f.name) and not np.any(d[f.val] == f.name)
        t = feats["t"].to_numpy()
        assert t[f.train].max() + cfg.model.purge_s <= t[f.val].min()
        assert not np.any(f.train & f.val)


def test_conformal_margin_gives_marginal_coverage():
    rng = np.random.default_rng(0)
    y_cal, y_new = rng.normal(size=4000), rng.normal(size=20000)
    lo, hi = -0.5, 0.5   # deliberately miscalibrated base interval
    q = conformal_margin(np.full(4000, lo), np.full(4000, hi), y_cal, 0.2)
    cov = np.mean((y_new >= lo - q) & (y_new <= hi + q))
    assert cov == pytest.approx(0.8, abs=0.02)


def test_hurst_estimator_separates_iid_from_long_memory():
    rng = np.random.default_rng(1)
    iid = rng.normal(size=20000)
    assert hurst_aggvar(iid, 8, 8) == pytest.approx(0.5, abs=0.08)
    lrd = np.cumsum(rng.normal(size=20000))   # strongly persistent
    assert hurst_aggvar(lrd - np.linspace(lrd[0], lrd[-1], len(lrd)), 8, 8) > 0.8


def test_calibrators_fix_a_distorted_score():
    rng = np.random.default_rng(2)
    true_p = rng.beta(1, 15, 30000)
    y = (rng.random(30000) < true_p).astype(int)
    distorted = 1 / (1 + np.exp(-(np.log(true_p / (1 - true_p)) + 2.0)))   # over-confident by +2 logits
    before = expected_calibration_error(distorted, y, 10)
    for variant in ("sigmoid", "isotonic"):
        after = expected_calibration_error(Calibrator(variant).fit(distorted, y).predict(distorted), y, 10)
        assert after < before / 3


def test_best_f1_threshold_picks_separating_value():
    p = np.array([0.1, 0.2, 0.3, 0.8, 0.9]); y = np.array([0, 0, 0, 1, 1])
    assert 0.3 < best_f1_threshold(p, y) <= 0.8


def test_config_validation_rejects_bad_values(cfg):
    bad = dataclasses.replace(cfg, storm=dataclasses.replace(cfg.storm, utilisation_threshold=-1.0))
    with pytest.raises(ConfigError):
        validate(bad)
    d0 = dataclasses.replace(cfg.city.districts[0], on_alpha=0.9)
    with pytest.raises(ConfigError):
        validate(dataclasses.replace(cfg, city=dataclasses.replace(cfg.city, districts=[d0, *cfg.city.districts[1:]])))
