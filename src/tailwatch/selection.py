"""Feature selection (mutual information, L1, tree importance) versus reduction (PCA): what each costs and gains."""
from __future__ import annotations

import warnings

import numpy as np
import polars as pl
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import StandardScaler

from .config import Config
from .core import Timer, block_bootstrap, feature_cols, fit_strategy, groups_for_bootstrap, lodo_folds, xy

METHODS = ("mutual_info", "l1", "tree", "pca")


def _rank(method: str, X: np.ndarray, y: np.ndarray, cfg: Config) -> np.ndarray:
    """Feature indices, best first. Fitted on a training subsample only."""
    rng = np.random.default_rng(cfg.seed)
    sub = rng.choice(len(y), size=min(cfg.selection.mi_subsample, len(y)), replace=False)
    Xs, ys = X[sub], y[sub]
    if method == "mutual_info":
        score = mutual_info_classif(Xs, ys, random_state=cfg.seed)
    elif method == "l1":
        Z = StandardScaler().fit_transform(Xs)
        with warnings.catch_warnings():  # sklearn 1.8 deprecates `penalty=`; kept for compatibility with older versions
            warnings.simplefilter("ignore")
            lr = LogisticRegression(penalty="l1", solver="liblinear", C=cfg.selection.l1_c, class_weight="balanced",
                                    max_iter=500).fit(Z, ys)
        score = np.abs(lr.coef_[0])
    elif method == "tree":
        rf = RandomForestClassifier(n_estimators=80, max_depth=8, min_samples_leaf=20, class_weight="balanced_subsample",
                                    n_jobs=-1, random_state=cfg.seed).fit(Xs, ys)
        score = rf.feature_importances_
    else:
        raise ValueError(method)
    return np.argsort(-score, kind="stable")


def run(df: pl.DataFrame, cfg: Config) -> dict:
    names = feature_cols(df)
    folds = lodo_folds(df, cfg)
    groups = groups_for_bootstrap(df, cfg)
    configs = [("all", len(names))] + [(m, k) for m in METHODS for k in cfg.selection.ks if k < len(names)]
    acc = {c: {"y": [], "p": [], "rows": [], "select_s": [], "fit_s": [], "pred_ms_per_1k": [], "chosen": []} for c in configs}
    for fold in folds:
        Xtr, ytr = xy(df.filter(pl.Series(fold.train)), cfg)
        Xte, yte = xy(df.filter(pl.Series(fold.test)), cfg)
        te_rows = np.flatnonzero(fold.test)
        rankings, rank_s = {}, {}
        for m in ("mutual_info", "l1", "tree"):
            with Timer() as tm:
                rankings[m] = _rank(m, Xtr, ytr, cfg)
            rank_s[m] = tm.s
        for method, k in configs:
            a = acc[(method, k)]
            if method == "all":
                cols = np.arange(len(names))
                s_time = 0.0
                with Timer() as tm:
                    model = fit_strategy(cfg.model.default_strategy, Xtr, ytr, cfg)
                with Timer() as tp:
                    p = model.predict_proba(Xte)[:, 1]
                a["chosen"].append(list(cols))
            elif method == "pca":
                with Timer() as ts:
                    scaler = StandardScaler().fit(Xtr)
                    pca = PCA(n_components=k, random_state=cfg.seed).fit(scaler.transform(Xtr))
                s_time = ts.s
                with Timer() as tm:
                    model = fit_strategy(cfg.model.default_strategy, pca.transform(scaler.transform(Xtr)), ytr, cfg)
                with Timer() as tp:
                    p = model.predict_proba(pca.transform(scaler.transform(Xte)))[:, 1]
                a["chosen"].append([])
            else:
                cols = rankings[method][:k]
                s_time = rank_s[method]
                with Timer() as tm:
                    model = fit_strategy(cfg.model.default_strategy, Xtr[:, cols], ytr, cfg)
                with Timer() as tp:
                    p = model.predict_proba(Xte[:, cols])[:, 1]
                a["chosen"].append(sorted(cols.tolist()))
            a["y"].append(yte); a["p"].append(p); a["rows"].append(te_rows)
            a["select_s"].append(s_time); a["fit_s"].append(tm.s)
            a["pred_ms_per_1k"].append(tp.s / len(yte) * 1000 * 1000)
    out = []
    for (method, k), a in acc.items():
        y, p, rows = np.concatenate(a["y"]), np.concatenate(a["p"]), np.concatenate(a["rows"])
        g = groups[rows]
        ap = float(average_precision_score(y, p))
        lo, hi = block_bootstrap(g, lambda i: average_precision_score(y[i], p[i]) if y[i].sum() else np.nan, cfg)
        chosen = [set(c) for c in a["chosen"]]
        stab = None
        if method not in ("all", "pca") and len(chosen) > 1:
            pairs = [(i, j) for i in range(len(chosen)) for j in range(i + 1, len(chosen))]
            stab = float(np.mean([len(chosen[i] & chosen[j]) / len(chosen[i] | chosen[j]) for i, j in pairs]))
        out.append({"method": method, "k": k, "pr_auc": ap, "pr_auc_lo": lo, "pr_auc_hi": hi,
                    "select_seconds": float(np.mean(a["select_s"])), "fit_seconds": float(np.mean(a["fit_s"])),
                    "predict_ms_per_1k_rows": float(np.mean(a["pred_ms_per_1k"])), "fold_jaccard_stability": stab,
                    "features": [names[i] for i in sorted(set.intersection(*chosen))] if method not in ("all", "pca") else None})
    return {"rows": out, "n_features_total": len(names)}
