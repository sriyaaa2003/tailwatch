"""Render results/summary.md from the JSON artefacts. Every number in it is read from a file, never typed in."""
from __future__ import annotations

from .config import Config
from .store import load_json, results_dir


def ci(r: dict, key: str, nd: int = 3) -> str:
    return f"{r[key]:.{nd}f} [{r[key + '_lo']:.{nd}f}, {r[key + '_hi']:.{nd}f}]"


def _data_section(cfg: Config, ds: dict, a) -> None:
    if cfg.report.kind == "fiveg":
        _data_fiveg(cfg, ds, a)
    else:
        _data_msdata(cfg, ds, a)


def _data_msdata(cfg: Config, ds: dict, a) -> None:
    r = cfg.real
    cad = ds["record_cadence"]
    a("## Data (msData, Open RAN 5G testbed)\n")
    a(f"{ds['raw_rows']:,} per-UE records (median spacing {cad['median_ms']:.0f} ms, 5th-95th percentile {cad['p05_ms']:.0f}-{cad['p95_ms']:.0f} ms; "
      f"not 1 ms samples) -> cell-level load in {r.bin_s:g} s bins (sum of all UEs' downlink rate), {ds['streams']} contiguous streams "
      f"(silences over {r.max_gap_s:g} s split a stream, streams under {r.min_stream_s} s dropped), {ds['stream_seconds'] / 3600:.1f} h of cell time, "
      f"{ds['feature_rows']:,} decision points. **High-load burst** = cell load reaches {r.storm_mbps:g} Mbit/s within {r.horizon_s} s "
      f"(a configured level, about the 99.3rd percentile of 1 s load; the cell's capacity is not observed, so this is not verified congestion). "
      f"**Burst rate {100 * ds['storm_rate']:.1f}%** ({ds['storm_count']:,} events).\n")
    a("| mobility pattern | streams | decision points | burst rate |\n|---|---|---|---|")
    for d in ds["by_group"]:
        a(f"| {d['group']} | {d['streams']} | {d['rows']:,} | {100 * d['storm_rate']:.1f}% |")
    a("\nDescriptive only (the traffic label is never a model input): burst rate by the traffic class of the heaviest UE in the bin.\n")
    a("| heaviest UE's traffic | decision points | burst rate |\n|---|---|---|")
    for d in ds["by_top_label"]:
        a(f"| {d['top_label']} | {d['rows']:,} | {100 * d['storm_rate']:.1f}% |")
    try:
        b = load_json(cfg, "burstiness.json")
    except FileNotFoundError:
        return
    a("\n### Burstiness of the cell load\n")
    a(f"Streams of at least {b['min_stream_s']} s. Median [IQR]; iid-noise Hurst reference {b['iid_reference_H']:.2f}.\n")
    a("| streams | n | Hurst H | CV (std/mean) | autocorr lag 1 s | autocorr lag 10 s |\n|---|---|---|---|---|---|")
    rows = [("all", b["all"])] + list(b["by_group"].items())
    for name, v in rows:
        f = lambda k: f"{v[k]['median']:.2f} [{v[k]['q25']:.2f}, {v[k]['q75']:.2f}]" if v[k] else "n/a"
        a(f"| {name} | {v['n']} | {f('H')} | {f('cv')} | {f('acf1')} | {f('acf10')} |")


def _data_fiveg(cfg: Config, ds: dict, a) -> None:
    f = cfg.fiveg
    a("## Data (client-side 5G production traces, Irish operator)\n")
    a(f"{ds['streams']} streams, {ds['stream_seconds']:,} s ({ds['stream_seconds'] / 3600:.1f} h) of 1 Hz phone KPIs, {100 * ds['share_5g']:.0f}% of seconds on 5G, "
      f"{ds['feature_rows']:,} decision points (stride {f.stride_s} s, channel currently good: CQI >= {ds['cqi_now_min']}). "
      f"**Channel collapse** = the worst CQI in the next {ds['horizon_s']} s is <= {ds['cqi_event_max']}. "
      f"**Collapse rate {100 * ds['storm_rate']:.1f}%** ({ds['storm_count']:,} events).\n")
    a("| mobility / app | streams | seconds | decision points | collapse rate |\n|---|---|---|---|---|")
    for d in ds["by_group"]:
        a(f"| {d['group']} | {d['streams']} | {d['seconds']:,} | {d['rows']:,} | {100 * d['storm_rate']:.1f}% |")


def write(cfg: Config) -> str:
    ds = load_json(cfg, "data_summary.json")
    u, ev = cfg.report.group_label, cfg.report.event_label
    imb, cal = load_json(cfg, "imbalance.json"), load_json(cfg, "calibration.json")["table"]
    pol, gen = load_json(cfg, "policy.json"), load_json(cfg, "generalization.json")
    prob, sel = load_json(cfg, "probabilistic.json"), load_json(cfg, "selection.json")
    L: list[str] = []
    a = L.append

    _data_section(cfg, ds, a)

    a(f"\n## 1. Imbalance strategies (unseen {u}s, raw scores)\n")
    a("| strategy | PR-AUC [95% CI] | macro-F1 [95% CI] | Brier | mean predicted p (true rate "
      f"{imb[0]['prevalence']:.3f}) | train fit s |\n|---|---|---|---|---|---|")
    for r in imb:
        a(f"| {r['strategy']} | {ci(r, 'pr_auc')} | {ci(r, 'macro_f1')} | {r['brier']:.4f} | {r['mean_p']:.3f} | {r['fit_seconds_mean']:.1f} |")

    a(f"\n## 2. Calibration (unseen {u}s)\n")
    a("| strategy | calibration | Brier [95% CI] | ECE [95% CI] | PR-AUC |\n|---|---|---|---|---|")
    for r in cal:
        a(f"| {r['strategy']} | {r['variant']} | {ci(r, 'brier', 4)} | {ci(r, 'ece', 4)} | {r['pr_auc']:.3f} |")

    pf = load_json(cfg, "per_fold.json")
    a(f"\nPR-AUC inside each held-out {u} ({cfg.model.default_strategy}). A monotone calibrator (sigmoid) cannot change it; "
      f"isotonic can, through tied scores. Pooled PR-AUC above mixes {u}s, so it also depends on the per-{u} calibration maps.\n")
    a(f"| held-out {u} | {ev} rate | raw | sigmoid | isotonic |\n|---|---|---|---|---|")
    for k, v in pf.items():
        strat, district = k.split("|")
        if strat == cfg.model.default_strategy:
            a(f"| {district} | {100 * v['prevalence']:.1f}% | {v['raw']:.3f} | {v['sigmoid']:.3f} | {v['isotonic']:.3f} |")

    a("\n## 3. Alert policy: what does calibration buy?\n")
    a(f"Costs: missed {ev} = {pol['c_miss']:g}, false alarm = {pol['c_fa']:g}, so the Bayes threshold for a calibrated "
      f"probability is {pol['bayes_threshold']:.3f}. Cost per 1000 decisions (lower is better); never alert = "
      f"{pol['never_alert_cost']:.0f}, always alert = {pol['always_alert_cost']:.0f}.\n")
    a("| strategy | calibration | cost at Bayes threshold [95% CI] | cost at validation-tuned threshold |\n|---|---|---|---|")
    for r in pol["rows"]:
        a(f"| {r['strategy']} | {r['variant']} | {r['cost_bayes']:.0f} [{r['cost_bayes_lo']:.0f}, {r['cost_bayes_hi']:.0f}] | {r['cost_tuned']:.0f} |")

    a("\n## 4. Does the evaluation protocol flatter the model?\n")
    a(f"Same model ({cfg.model.default_strategy}, isotonic-calibrated), {len(gen)} ways of splitting the data. PR-AUC is only "
      f"comparable across rows with similar {ev} prevalence, so prevalence is shown.\n")
    a(f"| protocol | PR-AUC [95% CI] | macro-F1 | Brier | ECE | {ev} prevalence in test |\n|---|---|---|---|---|---|")
    for r in gen:
        a(f"| {r['protocol']} | {ci(r, 'pr_auc')} | {r['macro_f1']:.3f} | {r['brier']:.4f} | {r['ece']:.4f} | {r['prevalence']:.3f} |")

    a(f"\n## 5. Probabilistic regression: {cfg.report.peak_label} in the next "
      f"{cfg.storm.horizon_s} s ({int(100 * prob['nominal_coverage'])}% intervals)\n")
    a(f"| interval | coverage on unseen {u}s [95% CI] | coverage when a {ev} occurs | mean width | interval score |\n|---|---|---|---|---|")
    for k, v in prob["pooled"].items():
        a(f"| {k} | {ci(v, 'coverage')} | {v['coverage_when_storm']:.3f} | {v['mean_width']:.3f} | {v['interval_score']:.3f} |")
    a(f"\nPer held-out {u} coverage (raw / conformalised): " + ", ".join(
        f"{f['fold']} {f['coverage_raw']:.2f}/{f['coverage_cqr']:.2f}" for f in prob["per_fold"]) + ".")
    a(f"\nMedian forecast MAE {prob['median_mae']:.3f} vs {prob['median_mae_persistence_baseline']:.3f} for "
      "'peak = current level'.")

    a(f"\n## 6. Feature selection vs reduction (unseen {u}s)\n")
    a("| method | k | PR-AUC [95% CI] | selection s | fit s | predict ms / 1k rows | fold stability (Jaccard) |\n|---|---|---|---|---|---|---|")
    for r in sel["rows"]:
        st = "" if r["fold_jaccard_stability"] is None else f"{r['fold_jaccard_stability']:.2f}"
        a(f"| {r['method']} | {r['k']} | {ci(r, 'pr_auc')} | {r['select_seconds']:.2f} | {r['fit_seconds']:.2f} | {r['predict_ms_per_1k_rows']:.2f} | {st} |")
    top = next((r for r in sel["rows"] if r["method"] == "tree" and r["features"]), None)
    if top:
        a(f"\nFeatures chosen by tree importance in every fold at k={top['k']}: {', '.join(top['features'])}.")

    path = results_dir(cfg) / "summary.md"
    path.write_text("\n".join(L) + "\n", encoding="utf-8")
    return str(path)
