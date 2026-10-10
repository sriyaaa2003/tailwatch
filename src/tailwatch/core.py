"""Shared modelling core: splits, resampling strategies, calibrators, metrics, block bootstrap."""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import polars as pl
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, f1_score, log_loss

from .config import Config

EPS = 1e-6
VARIANTS = ("raw", "sigmoid", "isotonic")


# --------------------------------------------------------------------------- splits
@dataclass
class Fold:
    name: str
    train: np.ndarray   # boolean masks over the feature frame
    val: np.ndarray
    test: np.ndarray


META_COLS = {"cell_id", "t", "group", "row", "col", "top_label", "y_fut_max", "y", "level_now"}


def feature_cols(df: pl.DataFrame) -> list[str]:
    """Model inputs = every column that is not identity/label metadata, in frame order."""
    return [c for c in df.columns if c not in META_COLS]


def lodo_folds(df: pl.DataFrame, cfg: Config) -> list[Fold]:
    """Leave-one-group-out (a group is a mobility pattern or an app/mobility combination).

    Inside the training groups the validation set (threshold + calibration) is either the last `val_fraction` of time,
    separated from training by a purge gap so window overlap cannot leak (val_mode "time"), or every k-th stream,
    k = 1/val_fraction (val_mode "stream"; streams are separate recordings, so no window crosses the boundary)."""
    t = df["t"].to_numpy()
    grp = df["group"].to_numpy()
    folds = []
    if cfg.model.val_mode == "time":
        t_split = cfg.duration_s * (1.0 - cfg.model.val_fraction)
    else:
        k = max(2, round(1.0 / cfg.model.val_fraction))
        in_val = df["cell_id"].to_numpy() % k == 0
    for name in sorted(set(grp.tolist())):
        other = grp != name
        if cfg.model.val_mode == "time":
            folds.append(Fold(name, other & (t < t_split - cfg.model.purge_s), other & (t >= t_split), grp == name))
        else:
            folds.append(Fold(name, other & ~in_val, other & in_val, grp == name))
    return folds


def groups_for_bootstrap(df: pl.DataFrame, cfg: Config) -> np.ndarray:
    """Block ids = (cell, time block). Resampling these blocks respects within-block autocorrelation."""
    return (df["cell_id"].to_numpy().astype(np.int64) * 10_000 + df["t"].to_numpy() // cfg.eval.block_s)


# --------------------------------------------------------------------------- model
def make_hgb(cfg: Config, class_weight=None) -> HistGradientBoostingClassifier:
    m = cfg.model
    return HistGradientBoostingClassifier(
        max_iter=m.max_iter, learning_rate=m.learning_rate, max_leaf_nodes=m.max_leaf_nodes,
        min_samples_leaf=m.min_samples_leaf, class_weight=class_weight, early_stopping=False,
        random_state=cfg.seed)


def fit_strategy(strategy: str, X: np.ndarray, y: np.ndarray, cfg: Config) -> HistGradientBoostingClassifier:
    if strategy == "none":
        return make_hgb(cfg).fit(X, y)
    if strategy == "class_weight":
        return make_hgb(cfg, "balanced").fit(X, y)
    if strategy == "smote":
        Xr, yr = SMOTE(sampling_strategy=cfg.model.smote_ratio, random_state=cfg.seed).fit_resample(X, y)
        return make_hgb(cfg).fit(Xr, yr)
    if strategy == "undersample":
        Xr, yr = RandomUnderSampler(sampling_strategy=cfg.model.undersample_ratio,
                                    random_state=cfg.seed).fit_resample(X, y)
        return make_hgb(cfg).fit(Xr, yr)
    raise ValueError(strategy)


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


class Calibrator:
    """Fit on validation scores; maps raw model scores to probabilities. `raw` is the identity."""

    def __init__(self, variant: str):
        if variant not in VARIANTS:
            raise ValueError(variant)
        self.variant = variant
        self._m = None

    def fit(self, p: np.ndarray, y: np.ndarray) -> "Calibrator":
        if self.variant == "sigmoid":
            self._m = LogisticRegression(C=1e6, max_iter=1000).fit(_logit(p).reshape(-1, 1), y)
        elif self.variant == "isotonic":
            self._m = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(p, y)
        return self

    def predict(self, p: np.ndarray) -> np.ndarray:
        if self.variant == "raw":
            return p
        if self.variant == "sigmoid":
            return self._m.predict_proba(_logit(p).reshape(-1, 1))[:, 1]
        return self._m.predict(p)


# --------------------------------------------------------------------------- metrics
def best_f1_threshold(p: np.ndarray, y: np.ndarray) -> float:
    """Threshold maximising positive-class F1 on (validation) data."""
    order = np.argsort(-p)
    ps, ys = p[order], y[order]
    tp = np.cumsum(ys)
    fp = np.cumsum(1 - ys)
    fn = ys.sum() - tp
    f1 = 2 * tp / np.maximum(2 * tp + fp + fn, 1)
    # only cut between distinct score values
    distinct = np.r_[ps[1:] != ps[:-1], True]
    f1 = np.where(distinct, f1, -1)
    return float(ps[int(np.argmax(f1))])


def expected_calibration_error(p: np.ndarray, y: np.ndarray, bins: int) -> float:
    """Equal-mass ECE."""
    order = np.argsort(p)
    chunks = np.array_split(order, bins)
    n = len(p)
    return float(sum(len(c) / n * abs(p[c].mean() - y[c].mean()) for c in chunks if len(c)))


def reliability_curve(p: np.ndarray, y: np.ndarray, bins: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    order = np.argsort(p)
    chunks = [c for c in np.array_split(order, bins) if len(c)]
    return (np.array([p[c].mean() for c in chunks]), np.array([y[c].mean() for c in chunks]),
            np.array([len(c) for c in chunks]))


def _th(thr, idx):
    return thr if np.isscalar(thr) else thr[idx]


def macro_f1(y: np.ndarray, p: np.ndarray, thr) -> float:
    """`thr` may be a scalar or a per-row array (each fold applies its own validation-chosen threshold)."""
    return float(f1_score(y, (p >= thr).astype(int), average="macro", zero_division=0))


def point_metrics(y: np.ndarray, p: np.ndarray, thr, bins: int) -> dict[str, float]:
    return {
        "pr_auc": float(average_precision_score(y, p)) if y.sum() else float("nan"),
        "macro_f1": macro_f1(y, p, thr),
        "brier": float(brier_score_loss(y, np.clip(p, 0, 1))),
        "log_loss": float(log_loss(y, np.clip(p, EPS, 1 - EPS), labels=[0, 1])),
        "ece": expected_calibration_error(p, y, bins),
        "mean_p": float(p.mean()), "prevalence": float(y.mean()),
    }


def block_bootstrap(groups: np.ndarray, stat, cfg: Config, seed_offset: int = 0) -> tuple[float, float]:
    """Percentile CI of stat(index_array) under resampling of blocks."""
    rng = np.random.default_rng(cfg.seed + seed_offset)
    uniq, inv = np.unique(groups, return_inverse=True)
    members = [np.flatnonzero(inv == i) for i in range(len(uniq))]
    vals = []
    for _ in range(cfg.eval.n_boot):
        pick = rng.integers(0, len(members), len(members))
        idx = np.concatenate([members[i] for i in pick])
        v = stat(idx)
        if np.isfinite(v):
            vals.append(v)
    lo, hi = np.quantile(vals, [(1 - cfg.eval.ci_level) / 2, 1 - (1 - cfg.eval.ci_level) / 2])
    return float(lo), float(hi)


def metrics_with_ci(y: np.ndarray, p: np.ndarray, thr, groups: np.ndarray, cfg: Config) -> dict[str, float]:
    bins = cfg.eval.reliability_bins
    out = point_metrics(y, p, thr, bins)
    for key in ("pr_auc", "macro_f1", "brier", "ece"):
        fn = {
            "pr_auc": lambda i: average_precision_score(y[i], p[i]) if y[i].sum() else np.nan,
            "macro_f1": lambda i: macro_f1(y[i], p[i], _th(thr, i)),
            "brier": lambda i: brier_score_loss(y[i], np.clip(p[i], 0, 1)),
            "ece": lambda i: expected_calibration_error(p[i], y[i], bins),
        }[key]
        lo, hi = block_bootstrap(groups, fn, cfg)
        out[f"{key}_lo"], out[f"{key}_hi"] = lo, hi
    return out


def xy(df: pl.DataFrame, cfg: Config, cols: list[str] | None = None) -> tuple[np.ndarray, np.ndarray]:
    cols = cols or feature_cols(df)
    return df.select(cols).to_numpy().astype(np.float32), df["y"].to_numpy().astype(np.int8)


class Timer:
    def __enter__(self):
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *a):
        self.s = time.perf_counter() - self.t0
