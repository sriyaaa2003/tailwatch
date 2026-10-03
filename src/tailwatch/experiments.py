"""Out-of-district predictions and the analyses built on them: imbalance, calibration, alert policy."""
from __future__ import annotations

import numpy as np
import polars as pl

from .config import Config
from .core import (VARIANTS, Calibrator, Timer, best_f1_threshold, fit_strategy, groups_for_bootstrap,
                   lodo_folds, metrics_with_ci, reliability_curve, xy)
from .store import results_dir


def compute_oof(df: pl.DataFrame, cfg: Config) -> tuple[pl.DataFrame, dict]:
    """For each held-out district and strategy: fit on the other districts' early time, calibrate on their later
    time, predict the held-out district. Every test row is therefore predicted by a model that never saw its district."""
    folds = lodo_folds(df, cfg)
    rows, meta = [], {}
    for fold in folds:
        Xtr, ytr = xy(df.filter(pl.Series(fold.train)), cfg)
        Xva, yva = xy(df.filter(pl.Series(fold.val)), cfg)
        Xte, yte = xy(df.filter(pl.Series(fold.test)), cfg)
        te_idx, va_idx = np.flatnonzero(fold.test), np.flatnonzero(fold.val)
        for strategy in cfg.model.strategies:
            with Timer() as tm:
                model = fit_strategy(strategy, Xtr, ytr, cfg)
            pva, pte = model.predict_proba(Xva)[:, 1], model.predict_proba(Xte)[:, 1]
            meta[f"{fold.name}|{strategy}|fit_s"] = tm.s
            meta[f"{fold.name}|{strategy}|n_train"] = int(len(ytr))
            meta[f"{fold.name}|{strategy}|train_prev"] = float(ytr.mean())
            cal = {v: Calibrator(v).fit(pva, yva) for v in VARIANTS}
            for split, idx, y, p in (("val", va_idx, yva, pva), ("test", te_idx, yte, pte)):
                cols = {v: cal[v].predict(p) for v in VARIANTS}
                rows.append(pl.DataFrame({
                    "fold": fold.name, "strategy": strategy, "split": split, "row": idx, "y": y.astype(np.int8),
                    "p_raw": cols["raw"], "p_sigmoid": cols["sigmoid"], "p_isotonic": cols["isotonic"],
                }))
            for v in VARIANTS:
                pv = cal[v].predict(pva)
                meta[f"{fold.name}|{strategy}|{v}|thr_f1"] = best_f1_threshold(pv, yva)
    oof = pl.concat(rows)
    oof.write_parquet(results_dir(cfg) / "oof.parquet", compression="zstd")
    return oof, meta


def _test(oof: pl.DataFrame, strategy: str) -> pl.DataFrame:
    return oof.filter((pl.col("strategy") == strategy) & (pl.col("split") == "test")).sort(["fold", "row"])


def _thr_array(test: pl.DataFrame, meta: dict, strategy: str, variant: str) -> np.ndarray:
    return np.array([meta[f"{f}|{strategy}|{variant}|thr_f1"] for f in test["fold"].to_list()])


def analyse_imbalance(df: pl.DataFrame, oof: pl.DataFrame, meta: dict, cfg: Config) -> list[dict]:
    """Strategies compared on ranking (PR-AUC) and thresholded decision quality (macro-F1). Raw scores, no recalibration."""
    out = []
    groups_all = groups_for_bootstrap(df, cfg)
    for strategy in cfg.model.strategies:
        t = _test(oof, strategy)
        y, p = t["y"].to_numpy(), t["p_raw"].to_numpy()
        thr = _thr_array(t, meta, strategy, "raw")
        m = metrics_with_ci(y, p, thr, groups_all[t["row"].to_numpy()], cfg)
        per_fold = {}
        for f in sorted(set(t["fold"])):
            tf = t.filter(pl.col("fold") == f)
            from sklearn.metrics import average_precision_score
            per_fold[f] = float(average_precision_score(tf["y"].to_numpy(), tf["p_raw"].to_numpy()))
        m.update({"strategy": strategy, "per_fold_pr_auc": per_fold,
                  "fit_seconds_mean": float(np.mean([meta[f"{f}|{strategy}|fit_s"] for f in per_fold])),
                  "train_prevalence_mean": float(np.mean([meta[f"{f}|{strategy}|train_prev"] for f in per_fold]))})
        out.append(m)
    return out


def analyse_calibration(df: pl.DataFrame, oof: pl.DataFrame, meta: dict, cfg: Config) -> dict:
    groups_all = groups_for_bootstrap(df, cfg)
    table, curves = [], {}
    for strategy in cfg.model.strategies:
        t = _test(oof, strategy)
        y = t["y"].to_numpy()
        for v in VARIANTS:
            p = t[f"p_{v}"].to_numpy()
            thr = _thr_array(t, meta, strategy, v)
            m = metrics_with_ci(y, p, thr, groups_all[t["row"].to_numpy()], cfg)
            m.update({"strategy": strategy, "variant": v})
            table.append(m)
            mp, fr, cnt = reliability_curve(p, y, cfg.eval.reliability_bins)
            curves[f"{strategy}|{v}"] = {"mean_pred": mp.tolist(), "frac_pos": fr.tolist(), "count": cnt.tolist()}
    return {"table": table, "curves": curves}


# --------------------------------------------------------------------------- alert policy
def decision_cost(y: np.ndarray, p: np.ndarray, thr, c_miss: float, c_fa: float) -> float:
    """Expected cost per 1000 decisions."""
    alert = p >= thr
    fn = (y == 1) & ~alert
    fp = (y == 0) & alert
    return float((c_miss * fn + c_fa * fp).mean() * 1000.0)


def analyse_policy(df: pl.DataFrame, oof: pl.DataFrame, cfg: Config) -> dict:
    """Does calibration pay? Compare the textbook Bayes threshold c_fa/(c_fa+c_miss) (needs calibrated probabilities)
    against a threshold tuned on validation data, for every strategy x calibration variant."""
    from .core import block_bootstrap
    c_miss, c_fa = cfg.policy.cost_miss, cfg.policy.cost_false_alarm
    bayes = c_fa / (c_fa + c_miss)
    grid = np.linspace(0.0, 1.0, cfg.policy.threshold_grid)
    groups_all = groups_for_bootstrap(df, cfg)
    rows, curves = [], {}
    base_t = _test(oof, cfg.model.strategies[0])
    y0 = base_t["y"].to_numpy()
    never = decision_cost(y0, np.zeros_like(y0, dtype=float), 0.5, c_miss, c_fa)
    always = decision_cost(y0, np.ones_like(y0, dtype=float), 0.5, c_miss, c_fa)
    for strategy in cfg.model.strategies:
        t = _test(oof, strategy)
        v_all = oof.filter((pl.col("strategy") == strategy) & (pl.col("split") == "val"))
        y = t["y"].to_numpy()
        for variant in VARIANTS:
            p = t[f"p_{variant}"].to_numpy()
            tuned = np.empty(len(p))
            for f in sorted(set(t["fold"])):
                vf = v_all.filter(pl.col("fold") == f)
                yv, pv = vf["y"].to_numpy(), vf[f"p_{variant}"].to_numpy()
                costs = [decision_cost(yv, pv, g, c_miss, c_fa) for g in grid]
                tuned[(t["fold"] == f).to_numpy()] = grid[int(np.argmin(costs))]
            g = groups_all[t["row"].to_numpy()]
            row = {"strategy": strategy, "variant": variant}
            for name, thr in (("bayes", bayes), ("tuned", tuned)):
                row[f"cost_{name}"] = decision_cost(y, p, thr, c_miss, c_fa)
                lo, hi = block_bootstrap(g, lambda i, thr=thr: decision_cost(
                    y[i], p[i], thr if np.isscalar(thr) else thr[i], c_miss, c_fa), cfg)
                row[f"cost_{name}_lo"], row[f"cost_{name}_hi"] = lo, hi
            rows.append(row)
            curves[f"{strategy}|{variant}"] = [decision_cost(y, p, gv, c_miss, c_fa) for gv in grid]
    return {"c_miss": c_miss, "c_fa": c_fa, "bayes_threshold": bayes, "never_alert_cost": never,
            "always_alert_cost": always, "grid": grid.tolist(), "rows": rows, "curves": curves}


def per_fold_pr_auc(oof: pl.DataFrame, cfg: Config) -> dict:
    """PR-AUC inside each held-out district, per calibration variant. Within a district a monotone calibrator cannot
    change ranking; pooled PR-AUC mixes districts and therefore does depend on the per-district calibration maps."""
    from sklearn.metrics import average_precision_score
    out = {}
    for strategy in cfg.model.strategies:
        t = oof.filter((pl.col("strategy") == strategy) & (pl.col("split") == "test"))
        for f in sorted(set(t["fold"])):
            tf = t.filter(pl.col("fold") == f)
            y = tf["y"].to_numpy()
            out[f"{strategy}|{f}"] = {"prevalence": float(y.mean()),
                                      **{v: float(average_precision_score(y, tf[f"p_{v}"].to_numpy())) for v in VARIANTS}}
    return out
