import sys, json
import numpy as np, polars as pl
sys.path.insert(0, "src")
from sklearn.metrics import average_precision_score
from sklearn.ensemble import ExtraTreesClassifier
from tailwatch.config import load_config
from tailwatch import real
from tailwatch.core import lodo_folds, feature_cols
from tailwatch.store import read_features
cfg = load_config("configs/default.yaml"); F = read_features(real.derive_config(cfg)); dcfg = real.derive_config(cfg, F)
cols = feature_cols(F); X = F.select(cols).to_numpy().astype(np.float32); y = F["y"].to_numpy(); folds = lodo_folds(F, dcfg)
out = {}
for leaf in (20, 60, 120, 250, 500):
    ys, ps = [], []
    for fo in folds:
        m = ExtraTreesClassifier(n_estimators=300, min_samples_leaf=leaf, max_features=0.3, n_jobs=8, random_state=7, class_weight="balanced_subsample").fit(X[fo.train], y[fo.train])
        ys.append(y[fo.test]); ps.append(m.predict_proba(X[fo.test])[:, 1])
    out[leaf] = float(average_precision_score(np.concatenate(ys), np.concatenate(ps))); print("leaf", leaf, round(out[leaf], 4), flush=True)
json.dump(out, open("results/v2/sens_leaf_msdata.json", "w"))
