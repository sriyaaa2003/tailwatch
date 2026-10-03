"""Probabilistic regression: forecast the peak utilisation over the next horizon as an interval.

Quantile gradient boosting gives a raw interval; split conformalised quantile regression (CQR, Romano et al. 2019)
adds a finite-sample coverage guarantee *under exchangeability*. The held-out-district test deliberately breaks
exchangeability, and the report states how much coverage is lost.
"""
from __future__ import annotations

import numpy as np
import polars as pl
from sklearn.ensemble import HistGradientBoostingRegressor

from .config import Config
from .core import block_bootstrap, feature_cols, groups_for_bootstrap, lodo_folds


def conformal_margin(lo: np.ndarray, hi: np.ndarray, y: np.ndarray, alpha: float) -> float:
    """CQR score quantile: E_i = max(lo_i - y_i, y_i - hi_i); margin = ceil((n+1)(1-alpha))/n quantile."""
    s = np.maximum(lo - y, y - hi)
    n = len(s)
    k = min(n, int(np.ceil((n + 1) * (1 - alpha))))
    return float(np.sort(s)[k - 1])


def _qreg(cfg: Config, q: float) -> HistGradientBoostingRegressor:
    m = cfg.model
    return HistGradientBoostingRegressor(loss="quantile", quantile=q, max_iter=cfg.probabilistic.quantile_max_iter,
                                         learning_rate=m.learning_rate, max_leaf_nodes=m.max_leaf_nodes,
                                         min_samples_leaf=m.min_samples_leaf, early_stopping=False,
                                         random_state=cfg.seed)


def run(df: pl.DataFrame, cfg: Config) -> dict:
    a = cfg.probabilistic.alpha
    cols = feature_cols(df)
    thr = cfg.storm.utilisation_threshold
    folds = lodo_folds(df, cfg)
    groups = groups_for_bootstrap(df, cfg)
    pooled = {k: [] for k in ("y", "lo_raw", "hi_raw", "lo_c", "hi_c", "med", "rows")}
    per_fold = []
    for fold in folds:
        def get(mask):
            sub = df.filter(pl.Series(mask))
            return sub.select(cols).to_numpy().astype(np.float32), sub["y_fut_max"].to_numpy().astype(np.float64)
        Xtr, ytr = get(fold.train)
        Xva, yva = get(fold.val)
        Xte, yte = get(fold.test)
        lo_m, hi_m, med_m = (_qreg(cfg, q).fit(Xtr, ytr) for q in (a / 2, 1 - a / 2, 0.5))
        qhat = conformal_margin(lo_m.predict(Xva), hi_m.predict(Xva), yva, a)
        lo, hi, med = lo_m.predict(Xte), hi_m.predict(Xte), med_m.predict(Xte)
        lo = np.minimum(lo, hi)
        lo_c, hi_c = lo - qhat, hi + qhat
        # in-distribution check: coverage on a validation-like exchangeable sample would be ~1-alpha by construction
        per_fold.append({"fold": fold.name, "qhat": qhat,
                         "coverage_raw": float(((yte >= lo) & (yte <= hi)).mean()),
                         "coverage_cqr": float(((yte >= lo_c) & (yte <= hi_c)).mean()),
                         "width_raw": float((hi - lo).mean()), "width_cqr": float((hi_c - lo_c).mean())})
        for k, v in (("y", yte), ("lo_raw", lo), ("hi_raw", hi), ("lo_c", lo_c), ("hi_c", hi_c), ("med", med),
                     ("rows", np.flatnonzero(fold.test))):
            pooled[k].append(v)
    P = {k: np.concatenate(v) for k, v in pooled.items()}
    g = groups[P["rows"]]
    out = {"alpha": a, "nominal_coverage": 1 - a, "per_fold": per_fold, "pooled": {}}
    for tag, lo_k, hi_k in (("raw quantile interval", "lo_raw", "hi_raw"), ("conformalised (CQR)", "lo_c", "hi_c")):
        y, lo, hi = P["y"], P[lo_k], P[hi_k]
        cover = lambda i: float(((y[i] >= lo[i]) & (y[i] <= hi[i])).mean())
        storm = y >= thr
        c_lo, c_hi = block_bootstrap(g, cover, cfg)
        out["pooled"][tag] = {
            "coverage": cover(np.arange(len(y))), "coverage_lo": c_lo, "coverage_hi": c_hi,
            "coverage_when_storm": float(((y >= lo) & (y <= hi))[storm].mean()) if storm.any() else None,
            "mean_width": float((hi - lo).mean()),
            "interval_score": float(np.mean((hi - lo) + (2 / a) * (lo - y) * (y < lo) + (2 / a) * (y - hi) * (y > hi))),
        }
    out["median_mae"] = float(np.abs(P["med"] - P["y"]).mean())
    out["median_mae_persistence_baseline"] = float(np.abs(
        df["util_now"].to_numpy()[P["rows"]] - P["y"]).mean())
    out["plot"] = {"y": P["y"][::40].tolist(), "lo": P["lo_c"][::40].tolist(), "hi": P["hi_c"][::40].tolist(),
                   "med": P["med"][::40].tolist()}
    return out
