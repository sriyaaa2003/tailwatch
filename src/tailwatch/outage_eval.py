"""Run the outage study and score every estimator against the simulator's ground truth.

Worlds before `outage.dev_worlds` are for choosing the method; every table below is computed on the remaining
worlds only. Confidence intervals are percentile bootstraps over scenarios."""
from __future__ import annotations

import numpy as np

from .config import Config
from .outage_est import blend, estimate, gls_total, loss_fraction, moving_average, prior_shares
from .outage_sim import build_world, handover_prior, run_scenario, sample_scenarios
from .store import results_dir, save_json

METHODS = ("level", "control")
TRUTH_KEYS = ("displaced_rate_mbps", "reconnected_rate_mbps", "loss_fraction", "degraded_excess_rate",
              "lost_user_share", "affected_users", "reconnected_users", "service_loss_user_seconds")


def sigma_key(s: float) -> str:
    return f"{s:g}"


def evaluate_one(world, out, handovers: dict[float, list[np.ndarray]], cfg: Config) -> dict:
    o = cfg.outage
    sc, tr, N = out.scenario, out.truth, out.neighbours
    rec = {"world": sc.world, "index": sc.index, "kind": sc.kind, "confounded": tr["confounded"],
           "failed": sc.failed, "neighbours": N, "window_s": sc.t1 - sc.t0,
           "truth": {k: tr[k] for k in TRUTH_KEYS}, "true_shares": {str(j): v for j, v in tr["shares"].items()},
           "absorbers": tr["absorbers"]}
    for m in METHODS:
        e = estimate(out.load, out.users, world.capacity, world.layout, sc.failed, sc.a0, sc.a1, o, m)
        D = e["displaced_rate_mbps"]
        r = {"D": D, "R_all": e["reconnected_rate_all_mbps"], "R_det": e["reconnected_rate_detected_mbps"],
             "detected": e["detected"], "loo_fpr": e["loo_false_positive_rate"], "deg_rate": e["degraded_excess_rate"],
             "data_share": {str(j): v for j, v in e["data_share"].items()}, "R_gls": {}, "prior": {}}
        for s, H in handovers.items():
            pi = prior_shares(world.layout, sc.failed, N, e["cf_failed_rate"], H)
            r["prior"][sigma_key(s)] = {str(j): v for j, v in pi.items()}
            r["R_gls"][sigma_key(s)] = gls_total(e["ebar"], e["se"], pi, D)
        # upper bound: the same estimator handed the true shares (not available in practice)
        true = tr["shares"] if tr["reconnected_mbit"] > 0 else {j: 1.0 / len(N) for j in N}
        r["R_gls_oracle"] = gls_total(e["ebar"], e["se"], {j: true.get(j, 0.0) for j in N}, D)
        rec[m] = r
    return rec


def example_series(world, out, cfg: Config) -> dict:
    """Curves for the example figure: the failed cell and its biggest true absorbers, observed vs counterfactual."""
    o = cfg.outage
    sc, tr = out.scenario, out.truth
    e = estimate(out.load, out.users, world.capacity, world.layout, sc.failed, sc.a0, sc.a1, o, "control", with_series=True)
    s = e["series"]
    lo, hi = max(sc.a0 - 1500, s["t0"]), min(world.load.shape[1], sc.a1 + 400)
    smooth = lambda x: moving_average(x[None, :], 31)[0]

    def cf_curve(c: int) -> list[float | None]:
        """Counterfactual exists from the pre-period to the end of the alarm window; later points are left empty."""
        out = np.full(hi - lo, np.nan)
        cf = smooth(s["cf"][c])
        n = min(len(cf) - (lo - s["t0"]), hi - lo)
        out[:n] = cf[lo - s["t0"]: lo - s["t0"] + n]
        return [None if np.isnan(v) else float(v) for v in out]

    nb_true = sorted(tr["rerouted_mbit"].items(), key=lambda kv: -kv[1])[:3]
    cells = list(sc.failed) + [int(j) for j, _ in nb_true]
    return {
        "t": list(range(lo, hi)), "alarm": [sc.a0, sc.a1], "true_window": [sc.t0, sc.t1], "failed": sc.failed,
        "cells": [{"cell": int(c), "role": "failed" if c in sc.failed else "absorber",
                   "observed": smooth(out.load[c])[lo:hi].tolist(), "baseline_truth": smooth(world.load[c])[lo:hi].tolist(),
                   "counterfactual": cf_curve(c) if c in s["cf"] else None,
                   "true_rate": tr["rerouted_mbit"].get(int(c), 0.0) / (sc.t1 - sc.t0),
                   "est_excess": e["ebar"].get(int(c)), "detected": int(c) in e["detected"]} for c in cells],
        "kind": sc.kind,
    }


def run_all(cfg: Config) -> dict:
    o = cfg.outage
    sigmas = sorted(set(o.noise_sweep) | {o.handover_noise_sigma})
    records, example = [], None
    for w in range(o.n_worlds):
        world = build_world(cfg, w)
        handovers = {s: handover_prior(world, s) for s in sigmas}
        for sc in sample_scenarios(cfg, world):
            out = run_scenario(world, sc)
            rec = evaluate_one(world, out, handovers, cfg)
            rec["dev"] = w < o.dev_worlds
            records.append(rec)
            if w == o.dev_worlds and sc.index == o.example_seed_index:
                example = example_series(world, out, cfg)
    return {"records": records, "example": example}


# ----------------------------------------------------------------------------------------------- aggregation
def bootstrap(records: list[dict], fn, cfg: Config, seed: int = 0) -> tuple[float, float, float]:
    rng = np.random.default_rng(cfg.seed + seed)
    n = len(records)
    vals = []
    for _ in range(cfg.outage.n_boot):
        idx = rng.integers(0, n, n)
        v = fn([records[i] for i in idx])
        if np.isfinite(v):
            vals.append(v)
    lo, hi = np.quantile(vals, [0.025, 0.975])
    return float(fn(records)), float(lo), float(hi)


def _pred_set(rec: dict, how: str, sigma: str, min_share: float) -> set[int]:
    if how == "uniform":
        return set(rec["neighbours"])
    if how == "prior":
        return {int(j) for j, v in rec["control"]["prior"][sigma].items() if v >= min_share}
    return set(rec[how]["detected"])


def f1_scores(records: list[dict], how: str, sigma: str, min_share: float) -> dict[str, float]:
    tp = fp = fn = 0
    for r in records:
        p, t = _pred_set(r, how, sigma, min_share), set(r["absorbers"])
        tp += len(p & t)
        fp += len(p - t)
        fn += len(t - p)
    prec = tp / (tp + fp) if tp + fp else float("nan")
    rec = tp / (tp + fn) if tp + fn else float("nan")
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return {"precision": prec, "recall": rec, "f1": f1}


def loss_estimators(sigma: str, prior_weights: list[float]) -> dict[str, tuple]:
    """name -> (method for D, function giving the reconnected rate R in Mbit/s)."""
    est = {
        "assume everything is lost": ("control", lambda r: 0.0),
        "assume half reconnects": ("control", lambda r: 0.5 * r["control"]["D"]),
        "level counterfactual, sum over neighbours": ("level", lambda r: r["level"]["R_all"]),
        "control counterfactual, sum over neighbours": ("control", lambda r: r["control"]["R_all"]),
        "control counterfactual, detected neighbours only": ("control", lambda r: r["control"]["R_det"]),
    }
    return est


def summarise(result: dict, cfg: Config) -> dict:
    o = cfg.outage
    recs = result["records"]
    test = [r for r in recs if not r["dev"]]
    dev = [r for r in recs if r["dev"]]
    sd = sigma_key(o.handover_noise_sigma)
    sweep = [sigma_key(s) for s in o.noise_sweep]
    out: dict = {"n_test": len(test), "n_dev": len(dev), "worlds": o.n_worlds, "dev_worlds": o.dev_worlds,
                 "default_sigma": sd, "kinds": {k: sum(r["kind"] == k for r in test) for k in ("single", "pair", "no coverage")},
                 "confounded": sum(r["confounded"] for r in test)}

    # ---- A. which neighbours absorbed traffic?
    det = []
    for name, how in (("flag every neighbour", "uniform"), (f"handover statistics only (sigma {sd})", "prior"),
                      ("data, level counterfactual", "level"), ("data, control counterfactual", "control")):
        row = {"method": name, **f1_scores(test, how, sd, o.absorber_min_share)}
        for k in ("precision", "recall", "f1"):
            _, lo, hi = bootstrap(test, lambda rs, k=k, how=how: f1_scores(rs, how, sd, o.absorber_min_share)[k], cfg, 1)
            row[k + "_lo"], row[k + "_hi"] = lo, hi
        nocov = [r for r in test if r["kind"] == "no coverage"]
        row["false_alarm_when_nobody_reconnects"] = float(np.mean([bool(_pred_set(r, how, sd, o.absorber_min_share)) for r in nocov])) if nocov else float("nan")
        det.append(row)
    out["absorbers"] = det

    # ---- B. how much was lost, how much reconnected?
    est = loss_estimators(sd, list(o.prior_weights))
    for s in sweep:
        est[f"control counterfactual + handover prior (sigma {s})"] = ("control", lambda r, s=s: r["control"]["R_gls"][s])
    est["control + true shares as prior (upper bound)"] = ("control", lambda r: r["control"]["R_gls_oracle"])
    rows = []

    def errs(rs, fn, m):
        d_loss = [abs(loss_fraction(r[m]["D"], max(fn(r), 0.0)) - r["truth"]["loss_fraction"]) for r in rs]
        d_rate = [abs(fn(r) - r["truth"]["reconnected_rate_mbps"]) for r in rs]
        bias = [fn(r) - r["truth"]["reconnected_rate_mbps"] for r in rs]
        return d_loss, d_rate, bias

    for name, (m, fn) in est.items():
        row = {"estimator": name}
        row["loss_mae_pp"], row["loss_mae_pp_lo"], row["loss_mae_pp_hi"] = (
            100 * v for v in bootstrap(test, lambda rs, fn=fn, m=m: float(np.mean(errs(rs, fn, m)[0])), cfg, 2))
        row["rate_mae_mbps"], row["rate_mae_mbps_lo"], row["rate_mae_mbps_hi"] = bootstrap(
            test, lambda rs, fn=fn, m=m: float(np.mean(errs(rs, fn, m)[1])), cfg, 3)
        row["rate_bias_mbps"] = float(np.mean(errs(test, fn, m)[2]))
        row["dev_loss_mae_pp"] = 100 * float(np.mean(errs(dev, fn, m)[0])) if dev else float("nan")
        rows.append(row)
    out["loss"] = rows
    out["displaced_rel_error_median"] = {m: float(np.median([abs(r[m]["D"] - r["truth"]["displaced_rate_mbps"]) / r["truth"]["displaced_rate_mbps"] for r in test])) for m in METHODS}
    out["truth_summary"] = {"mean_loss_fraction": float(np.mean([r["truth"]["loss_fraction"] for r in test])),
                            "mean_displaced_rate_mbps": float(np.mean([r["truth"]["displaced_rate_mbps"] for r in test])),
                            "mean_reconnected_rate_mbps": float(np.mean([r["truth"]["reconnected_rate_mbps"] for r in test]))}

    # ---- C. who took how much? (total variation distance between true and estimated shares; 0 is perfect)
    withshare = [r for r in test if r["truth"]["reconnected_rate_mbps"] > 0]

    def tvd(rs, share_fn):
        v = []
        for r in rs:
            t = r["true_shares"]
            e = share_fn(r)
            v.append(0.5 * sum(abs(t.get(j, 0.0) - e.get(j, 0.0)) for j in set(t) | set(e)))
        return float(np.mean(v))

    uniform = lambda r: {str(j): 1.0 / len(r["neighbours"]) for j in r["neighbours"]}
    shares = [{"method": "uniform over neighbours", **dict(zip(("tvd", "tvd_lo", "tvd_hi"), bootstrap(withshare, lambda rs: tvd(rs, uniform), cfg, 4)))},
              {"method": "data only (control counterfactual)", **dict(zip(("tvd", "tvd_lo", "tvd_hi"), bootstrap(withshare, lambda rs: tvd(rs, lambda r: r["control"]["data_share"]), cfg, 5)))}]
    grid = {}
    for s in sweep:
        grid[s] = {}
        for lam in o.prior_weights:
            f = lambda r, s=s, lam=lam: blend({int(j): v for j, v in r["control"]["data_share"].items()},
                                              {int(j): v for j, v in r["control"]["prior"][s].items()}, lam)
            g = lambda rs, f=f: tvd(rs, lambda r: {str(j): v for j, v in f(r).items()})
            grid[s][sigma_key(lam)] = float(g(withshare))
    out["shares"] = shares
    out["shares_grid"] = grid

    # ---- D. degradation in neighbouring cells (extra degraded user-seconds per second)
    deg = {}
    for m in METHODS:
        x = np.array([r[m]["deg_rate"] for r in test])
        y = np.array([r["truth"]["degraded_excess_rate"] for r in test])
        corr = lambda rs, m=m: float(np.corrcoef([r[m]["deg_rate"] for r in rs], [r["truth"]["degraded_excess_rate"] for r in rs])[0, 1])
        c, lo, hi = bootstrap(test, corr, cfg, 6)
        deg[m] = {"pearson_r": c, "pearson_r_lo": lo, "pearson_r_hi": hi, "mae": float(np.mean(np.abs(x - y))),
                  "mae_if_predicting_zero": float(np.mean(np.abs(y))), "true_mean": float(y.mean())}
    out["degradation"] = deg

    # ---- E. calibration of the placebo test (leave-one-out false-positive rate on control cells)
    out["placebo_fpr"] = {m: dict(zip(("fpr", "lo", "hi"), bootstrap(test, lambda rs, m=m: float(np.mean([r[m]["loo_fpr"] for r in rs])), cfg, 7))) for m in METHODS}
    out["alpha"] = o.alpha

    # ---- F. where does it break? headline metrics by subset
    def subset(rs):
        if len(rs) < 3:
            return None
        e_all = np.mean([abs(loss_fraction(r["control"]["D"], max(r["control"]["R_all"], 0)) - r["truth"]["loss_fraction"]) for r in rs])
        e_pri = np.mean([abs(loss_fraction(r["control"]["D"], max(r["control"]["R_gls"][sd], 0)) - r["truth"]["loss_fraction"]) for r in rs])
        return {"n": len(rs), "loss_mae_data_pp": 100 * float(e_all), "loss_mae_prior_pp": 100 * float(e_pri),
                "absorber_f1": f1_scores(rs, "control", sd, o.absorber_min_share)["f1"]}
    disp = np.array([r["truth"]["displaced_rate_mbps"] for r in test])
    q1, q2 = np.quantile(disp, [1 / 3, 2 / 3])
    dur = np.array([r["window_s"] for r in test])
    d1, d2 = np.quantile(dur, [1 / 3, 2 / 3])
    out["subsets"] = {name: subset(rs) for name, rs in (
        ("single-cell outage", [r for r in test if r["kind"] == "single"]),
        ("two-cell (site) outage", [r for r in test if r["kind"] == "pair"]),
        ("nobody can reconnect", [r for r in test if r["kind"] == "no coverage"]),
        ("outage overlaps a surge", [r for r in test if r["confounded"]]),
        ("no surge overlap", [r for r in test if not r["confounded"]]),
        (f"small outage (displaced < {q1:.0f} Mbit/s)", [r for r, x in zip(test, disp) if x < q1]),
        (f"medium outage ({q1:.0f}-{q2:.0f} Mbit/s)", [r for r, x in zip(test, disp) if q1 <= x < q2]),
        (f"large outage (> {q2:.0f} Mbit/s)", [r for r, x in zip(test, disp) if x >= q2]),
        (f"short outage (< {d1:.0f} s)", [r for r, x in zip(test, dur) if x < d1]),
        (f"medium-length outage ({d1:.0f}-{d2:.0f} s)", [r for r, x in zip(test, dur) if d1 <= x < d2]),
        (f"long outage (> {d2:.0f} s)", [r for r, x in zip(test, dur) if x >= d2]))}

    # ---- G. how much traffic must a neighbour absorb before the data finds it? (control counterfactual)
    edges = [0.0, 1.0, 3.0, 6.0, 1e9]
    power = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        tot = hit = 0
        for r in test:
            rate = r["truth"]["reconnected_rate_mbps"]
            flagged = set(r["control"]["detected"])
            for j, share in r["true_shares"].items():
                if lo <= share * rate < hi:
                    tot += 1
                    hit += int(int(j) in flagged)
        power.append({"absorbed_mbps_from": lo, "absorbed_mbps_to": None if hi > 1e8 else hi, "neighbours": tot,
                      "found": hit / tot if tot else float("nan")})
    out["detection_power"] = power
    return out


# ----------------------------------------------------------------------------------------------- figures + report
def make_figures(cfg: Config, summary: dict, example: dict | None) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    d = results_dir(cfg) / "outage"
    d.mkdir(parents=True, exist_ok=True)

    if example:
        t = np.array(example["t"])
        cells = example["cells"]
        fig, axes = plt.subplots(len(cells), 1, figsize=(9, 2.2 * len(cells)), sharex=True)
        for ax, c in zip(np.atleast_1d(axes), cells):
            ax.axvspan(*example["true_window"], color="#f4b6b6", alpha=0.35, lw=0)
            ax.axvline(example["alarm"][0], color="k", ls=":", lw=1)
            ax.axvline(example["alarm"][1], color="k", ls=":", lw=1)
            ax.plot(t, c["observed"], color="#1f77b4", lw=1.3, label="observed")
            ax.plot(t, c["baseline_truth"], color="#2ca02c", lw=1, ls="--", label="true counterfactual")
            if c["counterfactual"]:
                ax.plot(t, np.array([np.nan if v is None else v for v in c["counterfactual"]]), color="#7f7f7f", lw=1.2,
                        label="estimated counterfactual")
            tag = ("FAILED cell" if c["role"] == "failed" else
                   f"neighbour: true absorbed {c['true_rate']:.1f} Mbit/s, estimated excess {c['est_excess']:.1f}, "
                   + ("flagged" if c["detected"] else "not flagged"))
            ax.set_title(f"cell {c['cell']}: {tag}", fontsize=9, loc="left")
            ax.set_ylabel("Mbit/s")
        np.atleast_1d(axes)[0].legend(fontsize=7, ncol=3, loc="upper right")
        np.atleast_1d(axes)[-1].set_xlabel("time (s); pink = true outage, dotted = alarm window")
        fig.tight_layout()
        fig.savefig(d / "example.png", dpi=130)
        plt.close(fig)

    rows = summary["loss"]
    fig, ax = plt.subplots(figsize=(8.5, 4.4))
    y = np.arange(len(rows))
    v = np.array([r["loss_mae_pp"] for r in rows]); lo = np.array([r["loss_mae_pp_lo"] for r in rows]); hi = np.array([r["loss_mae_pp_hi"] for r in rows])
    colors = ["#bbbbbb" if r["estimator"].startswith("assume") else ("#9ecae1" if "handover" not in r["estimator"] and "true shares" not in r["estimator"] else "#1f77b4") for r in rows]
    ax.barh(y, v, xerr=[v - lo, hi - v], color=colors, capsize=2)
    ax.set_yticks(y, [r["estimator"] for r in rows], fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("loss-fraction error, percentage points (95% CI, lower is better)", fontsize=9)
    fig.tight_layout()
    fig.savefig(d / "loss_error.png", dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.2, 4))
    sweep = [r for r in rows if "handover prior" in r["estimator"]]
    sig = [float(r["estimator"].split("sigma ")[1].rstrip(")")) for r in sweep]
    ax.errorbar(sig, [r["loss_mae_pp"] for r in sweep], yerr=[[r["loss_mae_pp"] - r["loss_mae_pp_lo"] for r in sweep], [r["loss_mae_pp_hi"] - r["loss_mae_pp"] for r in sweep]],
                marker="o", capsize=2, color="#1f77b4", label="with handover prior")
    base = next(r for r in rows if r["estimator"].startswith("control counterfactual, sum"))
    ax.axhline(base["loss_mae_pp"], color="#d62728", ls="--", label="data only")
    half = next(r for r in rows if r["estimator"] == "assume half reconnects")
    ax.axhline(half["loss_mae_pp"], color="#7f7f7f", ls=":", label="always guess 50%")
    ax.set_xscale("log")
    ax.set_xlabel("noise in the handover statistics (log-normal sigma)")
    ax.set_ylabel("loss-fraction error (pp)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(d / "prior_noise.png", dpi=130)
    plt.close(fig)


def write_summary(cfg: Config, s: dict) -> str:
    o = cfg.outage
    L = []
    a = L.append
    pct = lambda x: f"{100 * x:.0f}%"
    a("# Cell-outage impact: estimates scored against ground truth\n")
    a(f"{o.n_worlds} simulated cities (seeds differ), {o.scenarios_per_world} outage scenarios each. World 0 was used to choose the "
      f"method; **every number below comes from the other {o.n_worlds - o.dev_worlds} worlds ({s['n_test']} scenarios)**: "
      f"{s['kinds']['single']} single-cell, {s['kinds']['pair']} two-cell, {s['kinds']['no coverage']} where nobody can reconnect; "
      f"{s['confounded']} overlap a scripted demand surge. Truth on average: {pct(s['truth_summary']['mean_loss_fraction'])} of displaced "
      f"traffic lost, {s['truth_summary']['mean_displaced_rate_mbps']:.1f} Mbit/s displaced per outage. Intervals are 95% bootstraps over scenarios.\n")
    a("## 1. Which neighbours absorbed traffic? (material absorber = at least "
      f"{pct(o.absorber_min_share)} of the rerouted traffic)\n")
    a("| method | precision | recall | F1 | false alarm when nobody reconnects |\n|---|---|---|---|---|")
    for r in s["absorbers"]:
        f = lambda k: f"{r[k]:.2f} [{r[k + '_lo']:.2f}, {r[k + '_hi']:.2f}]"
        a(f"| {r['method']} | {f('precision')} | {f('recall')} | {f('f1')} | {pct(r['false_alarm_when_nobody_reconnects'])} |")
    a("\n## 2. How much was lost and how much reconnected?\n")
    a("Error of the estimated **loss fraction** (share of displaced traffic that found no cell) and of the estimated reconnected rate. "
      f"Displaced traffic itself is estimated to within a median {pct(s['displaced_rel_error_median']['control'])} (control counterfactual).\n")
    a("| estimator | loss-fraction error, pp [95% CI] | reconnected-rate error, Mbit/s [95% CI] | bias, Mbit/s | dev-world error, pp |\n|---|---|---|---|---|")
    for r in s["loss"]:
        a(f"| {r['estimator']} | {r['loss_mae_pp']:.1f} [{r['loss_mae_pp_lo']:.1f}, {r['loss_mae_pp_hi']:.1f}] | "
          f"{r['rate_mae_mbps']:.1f} [{r['rate_mae_mbps_lo']:.1f}, {r['rate_mae_mbps_hi']:.1f}] | {r['rate_bias_mbps']:+.1f} | {r['dev_loss_mae_pp']:.1f} |")
    a("\n## 3. Who took how much? (total variation distance to the true shares; 0 is perfect, 1 is worst)\n")
    a("| method | distance [95% CI] |\n|---|---|")
    for r in s["shares"]:
        a(f"| {r['method']} | {r['tvd']:.2f} [{r['tvd_lo']:.2f}, {r['tvd_hi']:.2f}] |")
    lams = [sigma_key(x) for x in o.prior_weights]
    a("\nBlend of data and handover statistics, by noise in the statistics (rows) and weight on them (columns, 0 = data only, 1 = statistics only):\n")
    a("| handover noise sigma | " + " | ".join(f"weight {l}" for l in lams) + " |\n|---|" + "---|" * len(lams))
    for sg, row in s["shares_grid"].items():
        a(f"| {sg} | " + " | ".join(f"{row[l]:.2f}" for l in lams) + " |")
    a("\n## 4. Degradation in the neighbouring cells\n")
    a("Extra user-seconds per second below the served-fraction threshold, estimated from observed load and users against the cell's own pre-outage rate.\n")
    a("| counterfactual | correlation with truth [95% CI] | mean abs. error | error if predicting zero | true mean |\n|---|---|---|---|---|")
    for m, r in s["degradation"].items():
        a(f"| {m} | {r['pearson_r']:.2f} [{r['pearson_r_lo']:.2f}, {r['pearson_r_hi']:.2f}] | {r['mae']:.2f} | {r['mae_if_predicting_zero']:.2f} | {r['true_mean']:.2f} |")
    a(f"\n## 5. Is the absorber test honest?\n")
    a(f"False-positive rate of the placebo-in-space test on control cells (leave-one-out), nominal alpha = {o.alpha}:\n")
    a("| counterfactual | observed rate [95% CI] |\n|---|---|")
    for m, r in s["placebo_fpr"].items():
        a(f"| {m} | {r['fpr']:.3f} [{r['lo']:.3f}, {r['hi']:.3f}] |")
    a("\n## 6. Where does it break?\n")
    a("| subset | scenarios | loss error, data only (pp) | loss error, with handover prior (pp) | absorber F1 |\n|---|---|---|---|---|")
    for name, r in s["subsets"].items():
        if r:
            a(f"| {name} | {r['n']} | {r['loss_mae_data_pp']:.1f} | {r['loss_mae_prior_pp']:.1f} | {r['absorber_f1']:.2f} |")
    a("\n## 7. How much traffic must a neighbour absorb before the data finds it?\n")
    a("Share of neighbours flagged as absorbers (control counterfactual), by the traffic they truly took over:\n")
    a("| absorbed traffic, Mbit/s | neighbours | found |\n|---|---|---|")
    for r in s["detection_power"]:
        rng = f"{r['absorbed_mbps_from']:g} to {r['absorbed_mbps_to']:g}" if r["absorbed_mbps_to"] else f"{r['absorbed_mbps_from']:g} or more"
        a(f"| {rng} | {r['neighbours']} | {pct(r['found']) if r['neighbours'] else 'n/a'} |")
    path = results_dir(cfg) / "outage" / "summary.md"
    path.write_text("\n".join(L) + "\n", encoding="utf-8")
    return str(path)


def run_and_report(cfg: Config) -> str:
    result = run_all(cfg)
    summary = summarise(result, cfg)
    save_json(cfg, "outage/outage.json", summary)
    save_json(cfg, "outage/records.json", result["records"])
    save_json(cfg, "outage/example.json", result["example"])
    make_figures(cfg, summary, result["example"])
    return write_summary(cfg, summary)
