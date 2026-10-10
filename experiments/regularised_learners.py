"""tailwatch v2, step 2: randomised, strongly regularised learners on the original features, on both data sets.
Hyper-parameters of ExtraTrees are chosen by a nested leave-one-group-out inside the training groups only."""
import sys, time, json, itertools
import numpy as np, polars as pl
sys.path.insert(0, "src")
from sklearn.metrics import average_precision_score
from sklearn.ensemble import HistGradientBoostingClassifier, ExtraTreesClassifier, RandomForestClassifier
from scipy.stats import rankdata
from tailwatch.config import load_config
from tailwatch import real, fiveg
from tailwatch.core import lodo_folds, feature_cols, groups_for_bootstrap, block_bootstrap
from tailwatch.store import read_features

cfg = load_config("configs/default.yaml")
DS = sys.argv[1] if len(sys.argv) > 1 else "msdata"
dcfg0 = real.derive_config(cfg) if DS == "msdata" else fiveg.derive_config(cfg)
F = read_features(dcfg0)
dcfg = (real if DS == "msdata" else fiveg).derive_config(cfg, F)
cols = feature_cols(F)
X = F.select(cols).to_numpy().astype(np.float32); y = F["y"].to_numpy(); grp = F["group"].to_numpy()
folds = lodo_folds(F, dcfg)
t0 = time.time()

def et(msl, mf, n=300, cw="balanced_subsample"):
    return lambda: ExtraTreesClassifier(n_estimators=n, min_samples_leaf=msl, max_features=mf, n_jobs=8, random_state=7, class_weight=cw)
GRID = {f"et_leaf{l}_mf{m}": et(l, m) for l, m in itertools.product((5, 20, 60), (0.3, 0.6, 1.0))}
BASE = {
  "hgb_cw(repo)": lambda: HistGradientBoostingClassifier(max_iter=150, learning_rate=0.08, max_leaf_nodes=31, min_samples_leaf=40, class_weight="balanced", early_stopping=False, random_state=7),
  "hgb_plain": lambda: HistGradientBoostingClassifier(max_iter=150, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=100, l2_regularization=1.0, early_stopping=False, random_state=7),
  "rf": lambda: RandomForestClassifier(n_estimators=300, min_samples_leaf=20, max_features=0.5, n_jobs=8, random_state=7, class_weight="balanced_subsample"),
}
def fit_pred(mk, tr_mask, te_mask):
    m = mk().fit(X[tr_mask], y[tr_mask]); return m.predict_proba(X[te_mask])[:, 1]

# nested selection of the ExtraTrees config: for every outer fold, inner leave-one-group-out over the remaining groups
import os
CH = f"results/v2/chosen_{DS}.json"
chosen = json.load(open(CH)) if os.path.exists(CH) else {}
for fo in (folds if not chosen else []):
    inner_groups = sorted(set(grp[fo.train | fo.val]))
    best, best_s = None, -1
    for name, mk in GRID.items():
        ys, ps = [], []
        for g in inner_groups:
            te_m = (grp == g) & (fo.train | fo.val); tr_m = (fo.train | fo.val) & (grp != g)
            if y[te_m].sum() == 0: continue
            ys.append(y[te_m]); ps.append(fit_pred(mk, tr_m, te_m))
        s = float(np.mean([average_precision_score(a, b) for a, b in zip(ys, ps)]))
        if s > best_s: best, best_s = name, s
    chosen[fo.name] = best
    print("outer", fo.name, "->", best, round(best_s, 3), round(time.time() - t0), flush=True)
    json.dump(chosen, open(CH, "w"))

preds, rows = {}, {}
def run(name, getmk):
    ys, ps, per = [], [], {}
    for fo in folds:
        p = fit_pred(getmk(fo), fo.train, fo.test); ys.append(y[fo.test]); ps.append(p); per[fo.name] = float(average_precision_score(y[fo.test], p))
    preds[name] = (ys, ps)
    yy, pp = np.concatenate(ys), np.concatenate(ps)
    rows[name] = {"pooled": float(average_precision_score(yy, pp)), "per_fold_mean": float(np.mean(list(per.values()))), "per_fold": per}
    print(name, round(rows[name]["pooled"], 4), round(rows[name]["per_fold_mean"], 4), round(time.time() - t0), flush=True)
for n, mk in BASE.items(): run(n, lambda fo, mk=mk: mk)
run("et_default(leaf20,mf0.5)", lambda fo: et(20, 0.5))
run("et_nested_selected", lambda fo: GRID[chosen[fo.name]])
def ens(names):
    ps = [np.mean([rankdata(preds[n][1][i]) / len(preds[n][1][i]) for n in names], axis=0) for i in range(len(folds))]
    pp = np.concatenate(ps); yy = np.concatenate(preds[names[0]][0])
    per = {folds[i].name: float(average_precision_score(preds[names[0]][0][i], ps[i])) for i in range(len(folds))}
    return {"pooled": float(average_precision_score(yy, pp)), "per_fold_mean": float(np.mean(list(per.values()))), "per_fold": per}, pp
for nm, names in {"ens(et_nested,hgb_plain)": ["et_nested_selected", "hgb_plain"], "ens(et_nested,hgb_plain,rf)": ["et_nested_selected", "hgb_plain", "rf"]}.items():
    r, pp = ens(names); rows[nm] = r; preds[nm] = (preds[names[0]][0], None); print(nm, round(r["pooled"], 4), round(r["per_fold_mean"], 4), flush=True)
# block-bootstrap CI of the difference in pooled PR-AUC: et_nested vs repo model
rid = np.concatenate([np.flatnonzero(fo.test) for fo in folds]); G = groups_for_bootstrap(F, dcfg)[rid]
yy = np.concatenate(preds["hgb_cw(repo)"][0]); a = np.concatenate(preds["et_nested_selected"][1]); b = np.concatenate(preds["hgb_cw(repo)"][1])
d = lambda i: average_precision_score(yy[i], a[i]) - average_precision_score(yy[i], b[i]) if yy[i].sum() else np.nan
lo, hi = block_bootstrap(G, d, dcfg)
rows["diff_et_nested_minus_repo"] = {"point": float(average_precision_score(yy, a) - average_precision_score(yy, b)), "ci95": [lo, hi]}
print("diff", rows["diff_et_nested_minus_repo"], flush=True)
rows["chosen"] = chosen; rows["prevalence"] = float(y.mean())
json.dump(rows, open(f"results/v2/improve2_{DS}.json", "w"), indent=1)
