"""results/comparison.md: do the twin's findings hold on the real trace? Every number is read from the two result sets."""
from __future__ import annotations

from .config import Config
from .real import derive_config
from .store import load_json, results_dir


def _ci(r: dict, key: str, nd: int = 3) -> str:
    return f"{r[key]:.{nd}f} [{r[key + '_lo']:.{nd}f}, {r[key + '_hi']:.{nd}f}]"


def _load(cfg: Config) -> dict:
    return {
        "data": load_json(cfg, "data_summary.json"), "imb": load_json(cfg, "imbalance.json"),
        "cal": load_json(cfg, "calibration.json")["table"], "pol": load_json(cfg, "policy.json"),
        "gen": load_json(cfg, "generalization.json"), "prob": load_json(cfg, "probabilistic.json"),
        "sel": load_json(cfg, "selection.json"),
    }


def write(cfg: Config) -> str:
    rcfg = derive_config(cfg)
    sets = [("digital twin", cfg, _load(cfg)), ("real trace (msData)", rcfg, _load(rcfg))]
    strat = cfg.model.default_strategy
    L: list[str] = []
    a = L.append
    a("# Twin vs real trace\n")
    a("Same code, same estimators; each column is its own data set and its own held-out groups (districts for the twin, "
      "mobility patterns for the real trace). Intervals are 95% block-bootstrap CIs. The two columns are different systems, "
      "so compare *patterns*, not absolute levels.\n")
    a("| | " + " | ".join(n for n, _, _ in sets) + " |")
    a("|---|" + "---|" * len(sets))

    def row(label: str, fn) -> None:
        a(f"| {label} | " + " | ".join(fn(c, d) for _, c, d in sets) + " |")

    row("rare-event rate", lambda c, d: f"{100 * d['data']['storm_rate']:.1f}% ({d['data']['storm_count']:,} events)")
    for s in cfg.model.strategies:
        row(f"PR-AUC, {s}", lambda c, d, s=s: _ci(next(r for r in d["imb"] if r["strategy"] == s), "pr_auc"))
    row(f"mean predicted probability, {strat} (true rate in row 1)",
        lambda c, d: f"{next(r for r in d['imb'] if r['strategy'] == strat)['mean_p']:.3f}")

    def cal(c, d, variant, key, nd):
        r = next(r for r in d["cal"] if r["strategy"] == strat and r["variant"] == variant)
        return f"{r[key]:.{nd}f}"
    row(f"ECE, {strat}, raw -> isotonic", lambda c, d: f"{cal(c, d, 'raw', 'ece', 4)} -> {cal(c, d, 'isotonic', 'ece', 4)}")
    row(f"Brier, {strat}, raw -> isotonic", lambda c, d: f"{cal(c, d, 'raw', 'brier', 4)} -> {cal(c, d, 'isotonic', 'brier', 4)}")

    def pol(c, d, variant, key):
        return next(r for r in d["pol"]["rows"] if r["strategy"] == strat and r["variant"] == variant)[key]
    row(f"alert cost per 1000, {strat}, Bayes threshold, raw -> isotonic",
        lambda c, d: f"{pol(c, d, 'raw', 'cost_bayes'):.0f} -> {pol(c, d, 'isotonic', 'cost_bayes'):.0f} "
                     f"(tuned threshold: {pol(c, d, 'raw', 'cost_tuned'):.0f})")

    def gen(c, d, name):
        r = next(r for r in d["gen"] if r["protocol"].startswith(name))
        return r
    row("PR-AUC, random row split", lambda c, d: _ci(gen(c, d, "random"), "pr_auc"))
    row("PR-AUC, unseen group", lambda c, d: _ci(gen(c, d, "unseen"), "pr_auc"))
    row("random split inflates PR-AUC by", lambda c, d: f"{100 * (gen(c, d, 'random')['pr_auc'] / gen(c, d, 'unseen')['pr_auc'] - 1):.0f}%")

    cq = "conformalised (CQR)"
    row("80% interval coverage (conformalised), overall", lambda c, d: _ci(d["prob"]["pooled"][cq], "coverage"))
    row("... on the rows where the rare event occurs",
        lambda c, d: f"{d['prob']['pooled'][cq]['coverage_when_storm']:.3f}")

    def sel(c, d, method, k):
        r = next((r for r in d["sel"]["rows"] if r["method"] == method and r["k"] == k), None)
        return _ci(r, "pr_auc") if r else "n/a"
    row("PR-AUC, all features", lambda c, d: f"{_ci(next(r for r in d['sel']['rows'] if r['method'] == 'all'), 'pr_auc')} "
                                              f"({d['sel']['n_features_total']} features)")
    for m in ("mutual_info", "l1", "tree", "pca"):
        row(f"PR-AUC, {m} with 6 features", lambda c, d, m=m: sel(c, d, m, 6))
    path = results_dir(cfg) / "comparison.md"
    path.write_text("\n".join(L) + "\n", encoding="utf-8")
    return str(path)
