"""Figures. Every plot is drawn from the JSON written by the experiment steps; nothing is computed here."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .config import Config  # noqa: E402
from .store import load_json, results_dir  # noqa: E402

COLORS = {"raw": "#c0392b", "sigmoid": "#e69f00", "isotonic": "#1f77b4"}


def _save(fig, cfg: Config, name: str) -> None:
    fig.tight_layout()
    fig.savefig(results_dir(cfg) / f"{name}.png", dpi=130)
    plt.close(fig)


def reliability(cfg: Config) -> None:
    cal = load_json(cfg, "calibration.json")["curves"]
    fig, axes = plt.subplots(1, len(cfg.model.strategies), figsize=(4 * len(cfg.model.strategies), 4), sharex=True, sharey=True)
    for ax, s in zip(np.atleast_1d(axes), cfg.model.strategies):
        ax.plot([0, 0.6], [0, 0.6], "k--", lw=1)
        for v in ("raw", "isotonic"):
            c = cal[f"{s}|{v}"]
            ax.plot(c["mean_pred"], c["frac_pos"], "o-", color=COLORS[v], label=v, ms=4)
        ax.set_title(s)
        ax.set_xlabel("predicted probability")
    np.atleast_1d(axes)[0].set_ylabel(f"observed {cfg.report.event_label} frequency")
    np.atleast_1d(axes)[0].legend()
    ax.set_xlim(0, 0.6); ax.set_ylim(0, 0.6)
    _save(fig, cfg, "reliability")


def cost_curves(cfg: Config) -> None:
    pol = load_json(cfg, "policy.json")
    fig, ax = plt.subplots(figsize=(6.5, 4))
    grid = pol["grid"]
    for s, ls in zip(cfg.model.strategies, ("-", "--", ":", "-.")):
        for v in ("raw", "isotonic"):
            ax.plot(grid, pol["curves"][f"{s}|{v}"], ls, color=COLORS[v], lw=1.4, label=f"{s}/{v}" if s == cfg.model.default_strategy else None)
    ax.axvline(pol["bayes_threshold"], color="k", lw=1)
    ax.text(pol["bayes_threshold"] + 0.01, ax.get_ylim()[1] * 0.9, "Bayes threshold\nc_fa/(c_fa+c_miss)", fontsize=8)
    ax.set_xlabel("alert threshold"); ax.set_ylabel(f"cost per 1000 decisions (held-out {cfg.report.group_label}s)")
    ax.set_title("Cost vs alert threshold (red = raw scores, blue = isotonic)")
    ax.set_ylim(0, min(ax.get_ylim()[1], pol["never_alert_cost"]))
    _save(fig, cfg, "cost_curve")


def generalization(cfg: Config) -> None:
    rows = load_json(cfg, "generalization.json")
    fig, ax = plt.subplots(figsize=(7, 3.8))
    y = np.arange(len(rows))
    v = np.array([r["pr_auc"] for r in rows]); lo = np.array([r["pr_auc_lo"] for r in rows]); hi = np.array([r["pr_auc_hi"] for r in rows])
    ax.barh(y, v, xerr=[v - lo, hi - v], color="#1f77b4", capsize=3)
    for i, r in enumerate(rows):
        ax.plot(r["prevalence"], i, "k|", ms=18)
    ax.set_yticks(y, [r["protocol"] for r in rows]); ax.invert_yaxis()
    ax.set_xlabel(f"PR-AUC (95% CI); black tick = no-skill level ({cfg.report.event_label} rate)", fontsize=9)
    _save(fig, cfg, "generalization")


def selection(cfg: Config) -> None:
    rows = load_json(cfg, "selection.json")["rows"]
    fig, ax = plt.subplots(figsize=(6.5, 4))
    base = next(r for r in rows if r["method"] == "all")
    ax.axhline(base["pr_auc"], color="k", lw=1, ls="--", label=f"all {load_json(cfg, 'selection.json')['n_features_total']} features")
    for m, mk in (("mutual_info", "o"), ("l1", "s"), ("tree", "^"), ("pca", "D")):
        rs = sorted((r for r in rows if r["method"] == m), key=lambda r: r["k"])
        ax.errorbar([r["k"] for r in rs], [r["pr_auc"] for r in rs],
                    yerr=[[r["pr_auc"] - r["pr_auc_lo"] for r in rs], [r["pr_auc_hi"] - r["pr_auc"] for r in rs]],
                    marker=mk, capsize=2, label=m, alpha=0.85)
    ax.set_xlabel("features kept (k)"); ax.set_ylabel(f"PR-AUC, unseen {cfg.report.group_label}s"); ax.legend(fontsize=8)
    _save(fig, cfg, "selection")


def intervals(cfg: Config) -> None:
    p = load_json(cfg, "probabilistic.json")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    y, lo, hi = np.array(p["plot"]["y"]), np.array(p["plot"]["lo"]), np.array(p["plot"]["hi"])
    order = np.argsort(np.array(p["plot"]["med"]))[::6][:250]
    axes[0].fill_between(range(len(order)), lo[order], hi[order], color="#1f77b4", alpha=0.3, label="80% CQR interval")
    axes[0].plot(y[order], "k.", ms=3, label="realised peak utilisation")
    axes[0].axhline(cfg.storm.utilisation_threshold, color="#c0392b", lw=1)
    axes[0].set_xlabel("held-out rows sorted by predicted median"); axes[0].legend(fontsize=8)
    names = [f["fold"] for f in p["per_fold"]]
    x = np.arange(len(names))
    axes[1].bar(x - 0.2, [f["coverage_raw"] for f in p["per_fold"]], 0.4, label="raw quantiles", color="#e69f00")
    axes[1].bar(x + 0.2, [f["coverage_cqr"] for f in p["per_fold"]], 0.4, label="conformalised", color="#1f77b4")
    axes[1].axhline(p["nominal_coverage"], color="k", ls="--", lw=1)
    axes[1].set_xticks(x, names); axes[1].set_ylabel(f"coverage on unseen {cfg.report.group_label}"); axes[1].legend(fontsize=8, loc="lower right")
    axes[1].set_ylim(0.5, 1.0)
    _save(fig, cfg, "intervals")


def imbalance(cfg: Config) -> None:
    rows = load_json(cfg, "imbalance.json")
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    for ax, key, lab in zip(axes, ("pr_auc", "macro_f1"), ("PR-AUC", "macro-F1")):
        v = np.array([r[key] for r in rows]); lo = np.array([r[f"{key}_lo"] for r in rows]); hi = np.array([r[f"{key}_hi"] for r in rows])
        ax.bar([r["strategy"] for r in rows], v, yerr=[v - lo, hi - v], capsize=3, color="#1f77b4")
        ax.set_ylabel(lab + " (95% CI)")
        ax.set_ylim(max(0, lo.min() - 0.08), hi.max() + 0.04)
    _save(fig, cfg, "imbalance")


def make_all(cfg: Config) -> None:
    for fn in (reliability, cost_curves, generalization, selection, intervals, imbalance):
        fn(cfg)
