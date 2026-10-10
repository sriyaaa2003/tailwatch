import dataclasses

import numpy as np
import pytest

from tailwatch.config import ConfigError, load_config, validate
from tailwatch.core import Calibrator, best_f1_threshold, expected_calibration_error
from tailwatch.hurst import hurst_aggvar
from tailwatch.probabilistic import conformal_margin


@pytest.fixture(scope="module")
def cfg():
    return load_config()


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
    bad_fiveg = dataclasses.replace(cfg, fiveg=dataclasses.replace(cfg.fiveg, cqi_event_max=cfg.fiveg.cqi_now_min))
    with pytest.raises(ConfigError):
        validate(bad_fiveg)
