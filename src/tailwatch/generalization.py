"""How much does the evaluation protocol flatter the model? Same model, four ways of splitting the data."""
from __future__ import annotations

import numpy as np
import polars as pl

from .config import Config
from .core import (Calibrator, best_f1_threshold, fit_strategy, groups_for_bootstrap, lodo_folds, metrics_with_ci, xy)


def _fit_eval(df: pl.DataFrame, cfg: Config, tr: np.ndarray, va: np.ndarray, te: np.ndarray) -> dict:
    strategy = cfg.model.default_strategy
    Xtr, ytr = xy(df.filter(pl.Series(tr)), cfg)
    Xva, yva = xy(df.filter(pl.Series(va)), cfg)
    Xte, yte = xy(df.filter(pl.Series(te)), cfg)
    model = fit_strategy(strategy, Xtr, ytr, cfg)
    cal = Calibrator("isotonic").fit(model.predict_proba(Xva)[:, 1], yva)
    thr = best_f1_threshold(cal.predict(model.predict_proba(Xva)[:, 1]), yva)
    pte = cal.predict(model.predict_proba(Xte)[:, 1])
    return metrics_with_ci(yte, pte, thr, groups_for_bootstrap(df, cfg)[te], cfg)


def run(df: pl.DataFrame, oof: pl.DataFrame, meta: dict, cfg: Config) -> list[dict]:
    n = df.height
    t = df["t"].to_numpy()
    T = cfg.sim.duration_s
    rows = []

    # 1. random row split: neighbouring seconds of the same cell land on both sides (leakage)
    rng = np.random.default_rng(cfg.seed)
    u = rng.random(n)
    f = cfg.generalization.random_test_fraction
    te = u < f
    va = (u >= f) & (u < f + (1 - f) * cfg.model.val_fraction)
    tr = u >= f + (1 - f) * cfg.model.val_fraction
    rows.append({"protocol": "random rows", **_fit_eval(df, cfg, tr, va, te)})

    # 2. temporal split on the same cells: future vs past, but the same places
    ft = cfg.generalization.temporal_test_fraction
    t_test = T * (1 - ft)
    t_val = t_test - T * (1 - ft) * cfg.model.val_fraction
    te = t >= t_test
    va = (t >= t_val) & (t < t_test - cfg.model.purge_s)
    tr = t < t_val - cfg.model.purge_s
    rows.append({"protocol": "temporal (same cells)", **_fit_eval(df, cfg, tr, va, te)})

    # 3./4. leave-one-district-out, reusing the out-of-district predictions (no refit)
    from .core import metrics_with_ci as mwc
    strategy = cfg.model.default_strategy
    test = oof.filter((pl.col("strategy") == strategy) & (pl.col("split") == "test")).sort(["fold", "row"])
    rid = test["row"].to_numpy()
    thr = np.array([meta[f"{fo}|{strategy}|isotonic|thr_f1"] for fo in test["fold"].to_list()])
    y, p = test["y"].to_numpy(), test["p_isotonic"].to_numpy()
    groups = groups_for_bootstrap(df, cfg)[rid]
    rows.append({"protocol": "unseen district", **mwc(y, p, thr, groups, cfg)})
    late = t[rid] >= T * (1.0 - cfg.model.val_fraction)
    rows.append({"protocol": "unseen district + future", **mwc(y[late], p[late], thr[late], groups[late], cfg)})
    return rows
