"""How much does the evaluation protocol flatter the model? Same model, four ways of splitting the data."""
from __future__ import annotations

import numpy as np
import polars as pl

from .config import Config
from .core import Calibrator, best_f1_threshold, fit_strategy, groups_for_bootstrap, metrics_with_ci, xy


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
    T = cfg.duration_s
    label = cfg.report.group_label
    wanted = cfg.generalization.protocols
    rows = []

    if "random" in wanted:
        # random row split: neighbouring seconds of the same stream land on both sides (leakage)
        rng = np.random.default_rng(cfg.seed)
        u = rng.random(n)
        f = cfg.generalization.random_test_fraction
        te = u < f
        va = (u >= f) & (u < f + (1 - f) * cfg.model.val_fraction)
        tr = u >= f + (1 - f) * cfg.model.val_fraction
        rows.append({"protocol": "random rows", **_fit_eval(df, cfg, tr, va, te)})

    if "temporal" in wanted:
        # temporal split on the same cells: future vs past, but the same places
        ft = cfg.generalization.temporal_test_fraction
        t_test = T * (1 - ft)
        t_val = t_test - T * (1 - ft) * cfg.model.val_fraction
        te = t >= t_test
        va = (t >= t_val) & (t < t_test - cfg.model.purge_s)
        tr = t < t_val - cfg.model.purge_s
        rows.append({"protocol": "temporal (same cells)", **_fit_eval(df, cfg, tr, va, te)})

    if "unseen" in wanted or "unseen_future" in wanted:
        # leave-one-group-out, reusing the out-of-group predictions (no refit)
        strategy = cfg.model.default_strategy
        test = oof.filter((pl.col("strategy") == strategy) & (pl.col("split") == "test")).sort(["fold", "row"])
        rid = test["row"].to_numpy()
        thr = np.array([meta[f"{fo}|{strategy}|isotonic|thr_f1"] for fo in test["fold"].to_list()])
        y, p = test["y"].to_numpy(), test["p_isotonic"].to_numpy()
        groups = groups_for_bootstrap(df, cfg)[rid]
        if "unseen" in wanted:
            rows.append({"protocol": f"unseen {label}", **metrics_with_ci(y, p, thr, groups, cfg)})
        if "unseen_future" in wanted:
            late = t[rid] >= T * (1.0 - cfg.model.val_fraction)
            rows.append({"protocol": f"unseen {label} + future",
                         **metrics_with_ci(y[late], p[late], thr[late], groups[late], cfg)})
    return rows
