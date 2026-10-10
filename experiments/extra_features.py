"""tailwatch v2: can better features, other learners and ensembles rank rare congestion bursts better on unseen groups?
Protocol identical to the repo: leave-one-group-out, validation = every k-th stream of the training groups, PR-AUC pooled over folds."""
import sys, time, json
import numpy as np, polars as pl
sys.path.insert(0, "src")
from sklearn.metrics import average_precision_score
from sklearn.ensemble import HistGradientBoostingClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from catboost import CatBoostClassifier
from tailwatch.config import load_config
from tailwatch import real
from tailwatch.core import lodo_folds, feature_cols, xy, make_hgb
from tailwatch.store import read_features

cfg = load_config("configs/default.yaml")
dcfg = real.derive_config(cfg)
feats = read_features(dcfg)
dcfg = real.derive_config(cfg, feats)
s = pl.read_parquet("data/msdata/streams.parquet").sort(["cell_id", "t"])
STORM = cfg.real.storm_mbps

def extended(s: pl.DataFrame) -> pl.DataFrame:
    u = (pl.col("load") / STORM)
    o = "cell_id"
    ex = [u.alias("_u")]
    d = s.with_columns(ex)
    u = pl.col("_u")
    cols = []
    for w in (120, 300):
        cols += [u.rolling_mean(w).over(o).alias(f"x_mean_{w}"), u.rolling_std(w).over(o).alias(f"x_std_{w}"), u.rolling_max(w).over(o).alias(f"x_max_{w}")]
    for hl in (3, 10, 30):
        cols.append(u.ewm_mean(half_life=hl, ignore_nulls=True).over(o).alias(f"x_ewm_{hl}"))
    for lv in (0.3, 0.6):
        above = (u >= lv).cast(pl.Float64)
        for w in (30, 60, 300):
            cols.append(above.rolling_sum(w).over(o).alias(f"x_cnt{int(lv*10)}_{w}"))
    cols += [u.rolling_quantile(0.9, window_size=60).over(o).alias("x_q90_60"), u.rolling_quantile(0.9, window_size=300).over(o).alias("x_q90_300"),
             (u - u.shift(1)).over(o).alias("x_d1"), (u - u.shift(3)).over(o).alias("x_d3"),
             pl.col("users").rolling_max(30).over(o).alias("x_users_max_30"), pl.col("buf").log1p().rolling_max(30).over(o).alias("x_buf_max_30"),
             pl.col("cqi").rolling_min(30).over(o).alias("x_cqi_min_30"), pl.col("mcs").rolling_mean(30).over(o).alias("x_mcs_mean_30"),
             pl.col("sinr").rolling_std(30).over(o).alias("x_sinr_std_30")]
    d = d.with_columns(cols)
    # time since the load last exceeded a fraction of the burst level
    for lv in (0.3, 0.6):
        last = pl.when(u >= lv).then(pl.col("t")).otherwise(None).forward_fill().over(o)
        d = d.with_columns(pl.min_horizontal((pl.col("t") - last).fill_null(600), pl.lit(600)).alias(f"x_since{int(lv*10)}"))
    keep = ["cell_id", "t"] + [c for c in d.columns if c.startswith("x_")]
    return d.select(keep)

t0 = time.time()
E = extended(s)
F = feats.join(E, on=["cell_id", "t"], how="left")
base_cols = feature_cols(feats)
ext_cols = [c for c in F.columns if c.startswith("x_")]
F = F.with_columns(pl.col(ext_cols).fill_nan(None).fill_null(0.0).cast(pl.Float32))
print("features", len(base_cols), "+", len(ext_cols), round(time.time() - t0), flush=True)
folds = lodo_folds(F, dcfg)

def models():
    return {
      "hgb_cw": lambda: HistGradientBoostingClassifier(max_iter=150, learning_rate=0.08, max_leaf_nodes=31, min_samples_leaf=40, class_weight="balanced", early_stopping=False, random_state=7),
      "hgb_plain": lambda: HistGradientBoostingClassifier(max_iter=150, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=100, l2_regularization=1.0, early_stopping=False, random_state=7),
      "cat": lambda: CatBoostClassifier(iterations=500, depth=6, learning_rate=0.05, l2_leaf_reg=5, verbose=0, random_seed=7, thread_count=8),
      "et": lambda: ExtraTreesClassifier(n_estimators=300, min_samples_leaf=20, max_features=0.5, n_jobs=8, random_state=7, class_weight="balanced_subsample"),
      "lr": lambda: make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=500, class_weight="balanced")),
    }

variants = [("base", base_cols), ("ext", base_cols + ext_cols)]
res, preds = {}, {}
for vname, cols in variants:
    for mname, mk in models().items():
        key = f"{mname}|{vname}"
        ys, ps = [], []
        per = {}
        for fold in folds:
            tr, te = F.filter(pl.Series(fold.train)), F.filter(pl.Series(fold.test))
            X, y = tr.select(cols).to_numpy().astype(np.float32), tr["y"].to_numpy()
            Xt, yt = te.select(cols).to_numpy().astype(np.float32), te["y"].to_numpy()
            m = mk().fit(X, y); p = m.predict_proba(Xt)[:, 1]
            ys.append(yt); ps.append(p); per[fold.name] = float(average_precision_score(yt, p))
        preds[key] = (np.concatenate(ys), np.concatenate(ps), ps)
        y_all = preds[key][0]
        res[key] = {"pooled": float(average_precision_score(y_all, preds[key][1])), "per_fold_mean": float(np.mean(list(per.values()))), "per_fold": per}
        print(key, round(res[key]["pooled"], 4), round(res[key]["per_fold_mean"], 4), round(time.time() - t0), flush=True)

def ens(keys):
    # average of within-fold rank-normalised scores (calibration-free, ranking only)
    from scipy.stats import rankdata
    ps_f = []
    for i in range(len(folds)):
        ps_f.append(np.mean([rankdata(preds[k][2][i]) / len(preds[k][2][i]) for k in keys], axis=0))
    p = np.concatenate(ps_f); y = preds[keys[0]][0]
    per = {folds[i].name: float(average_precision_score(preds[keys[0]][0][sum(len(x) for x in preds[keys[0]][2][:i]):sum(len(x) for x in preds[keys[0]][2][:i+1])], ps_f[i])) for i in range(len(folds))}
    return float(average_precision_score(y, p)), float(np.mean(list(per.values()))), per
for name, keys in {"ens_ext(hgb_cw,cat,et)": ["hgb_cw|ext", "cat|ext", "et|ext"], "ens_ext(cat,et)": ["cat|ext", "et|ext"], "ens_ext_all": [k for k in preds if k.endswith("|ext") and not k.startswith("lr")]}.items():
    a, b, per = ens(keys); res[name] = {"pooled": a, "per_fold_mean": b, "per_fold": per}; print(name, round(a, 4), round(b, 4), flush=True)
json.dump(res, open("results/v2/improve_msdata.json", "w"), indent=1)
