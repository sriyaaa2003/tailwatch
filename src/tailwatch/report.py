"""Render results/summary.md from the JSON artefacts. Every number in it is read from a file, never typed in."""
from __future__ import annotations

from .config import Config
from .store import load_json, results_dir


def ci(r: dict, key: str, nd: int = 3) -> str:
    return f"{r[key]:.{nd}f} [{r[key + '_lo']:.{nd}f}, {r[key + '_hi']:.{nd}f}]"


def write(cfg: Config) -> str:
    ds, hu = load_json(cfg, "data_summary.json"), load_json(cfg, "hurst.json")
    imb, cal = load_json(cfg, "imbalance.json"), load_json(cfg, "calibration.json")["table"]
    pol, gen = load_json(cfg, "policy.json"), load_json(cfg, "generalization.json")
    prob, sel = load_json(cfg, "probabilistic.json"), load_json(cfg, "selection.json")
    L: list[str] = []
    a = L.append

    a("## Data (synthetic digital twin)\n")
    a(f"{ds['raw_rows']:,} one-second cell observations -> {ds['feature_rows']:,} decision points (stride {cfg.features.stride_s} s, "
      f"already-congested rows removed). Storm = congestion (utilisation >= {cfg.storm.utilisation_threshold}) begins within "
      f"{cfg.storm.horizon_s} s. **Storm rate {100 * ds['storm_rate']:.1f}%** ({ds['storm_count']:,} events).\n")
    a("| district | decision points | storm rate |\n|---|---|---|")
    for d in ds["by_district"]:
        a(f"| {d['district']} | {d['rows']:,} | {100 * d['storm_rate']:.1f}% |")
    a("\nGenerator check (aggregated-variance Hurst exponent on load with surges and diurnal cycle switched off; "
      f"iid-noise reference H = {hu['H_iid_reference']:.2f}):\n")
    a("| cell | district | H measured | H theory (3-alpha_min)/2 |\n|---|---|---|---|")
    for r in hu["cells"]:
        a(f"| {r['cell_id']} | {r['district']} | {r['H_measured']:.2f} | {r['H_theory']:.2f} |")

    a("\n## 1. Imbalance strategies (unseen districts, raw scores)\n")
    a("| strategy | PR-AUC [95% CI] | macro-F1 [95% CI] | Brier | mean predicted p (true rate "
      f"{imb[0]['prevalence']:.3f}) | train fit s |\n|---|---|---|---|---|---|")
    for r in imb:
        a(f"| {r['strategy']} | {ci(r, 'pr_auc')} | {ci(r, 'macro_f1')} | {r['brier']:.4f} | {r['mean_p']:.3f} | {r['fit_seconds_mean']:.1f} |")

    a("\n## 2. Calibration (unseen districts)\n")
    a("| strategy | calibration | Brier [95% CI] | ECE [95% CI] | PR-AUC |\n|---|---|---|---|---|")
    for r in cal:
        a(f"| {r['strategy']} | {r['variant']} | {ci(r, 'brier', 4)} | {ci(r, 'ece', 4)} | {r['pr_auc']:.3f} |")

    pf = load_json(cfg, "per_fold.json")
    a(f"\nPR-AUC inside each held-out district ({cfg.model.default_strategy}). A monotone calibrator (sigmoid) cannot change it; "
      "isotonic can, through tied scores. Pooled PR-AUC above mixes districts, so it also depends on the per-district calibration maps.\n")
    a("| held-out district | storm rate | raw | sigmoid | isotonic |\n|---|---|---|---|---|")
    for k, v in pf.items():
        strat, district = k.split("|")
        if strat == cfg.model.default_strategy:
            a(f"| {district} | {100 * v['prevalence']:.1f}% | {v['raw']:.3f} | {v['sigmoid']:.3f} | {v['isotonic']:.3f} |")

    a("\n## 3. Alert policy: what does calibration buy?\n")
    a(f"Costs: missed storm = {pol['c_miss']:g}, false alarm = {pol['c_fa']:g}, so the Bayes threshold for a calibrated "
      f"probability is {pol['bayes_threshold']:.3f}. Cost per 1000 decisions (lower is better); never alert = "
      f"{pol['never_alert_cost']:.0f}, always alert = {pol['always_alert_cost']:.0f}.\n")
    a("| strategy | calibration | cost at Bayes threshold [95% CI] | cost at validation-tuned threshold |\n|---|---|---|---|")
    for r in pol["rows"]:
        a(f"| {r['strategy']} | {r['variant']} | {r['cost_bayes']:.0f} [{r['cost_bayes_lo']:.0f}, {r['cost_bayes_hi']:.0f}] | {r['cost_tuned']:.0f} |")

    a("\n## 4. Does the evaluation protocol flatter the model?\n")
    a(f"Same model ({cfg.model.default_strategy}, isotonic-calibrated), four splits. PR-AUC is only comparable across rows "
      "with similar storm prevalence, so prevalence is shown.\n")
    a("| protocol | PR-AUC [95% CI] | macro-F1 | Brier | ECE | storm prevalence in test |\n|---|---|---|---|---|---|")
    for r in gen:
        a(f"| {r['protocol']} | {ci(r, 'pr_auc')} | {r['macro_f1']:.3f} | {r['brier']:.4f} | {r['ece']:.4f} | {r['prevalence']:.3f} |")

    a("\n## 5. Probabilistic regression: peak utilisation in the next "
      f"{cfg.storm.horizon_s} s ({int(100 * prob['nominal_coverage'])}% intervals)\n")
    a("| interval | coverage on unseen districts [95% CI] | coverage when a storm occurs | mean width | interval score |\n|---|---|---|---|---|")
    for k, v in prob["pooled"].items():
        a(f"| {k} | {ci(v, 'coverage')} | {v['coverage_when_storm']:.3f} | {v['mean_width']:.3f} | {v['interval_score']:.3f} |")
    a("\nPer held-out district coverage (raw / conformalised): " + ", ".join(
        f"{f['fold']} {f['coverage_raw']:.2f}/{f['coverage_cqr']:.2f}" for f in prob["per_fold"]) + ".")
    a(f"\nMedian forecast MAE {prob['median_mae']:.3f} vs {prob['median_mae_persistence_baseline']:.3f} for "
      "'peak = current utilisation'.")

    a("\n## 6. Feature selection vs reduction (unseen districts)\n")
    a("| method | k | PR-AUC [95% CI] | selection s | fit s | predict ms / 1k rows | fold stability (Jaccard) |\n|---|---|---|---|---|---|---|")
    for r in sel["rows"]:
        st = "" if r["fold_jaccard_stability"] is None else f"{r['fold_jaccard_stability']:.2f}"
        a(f"| {r['method']} | {r['k']} | {ci(r, 'pr_auc')} | {r['select_seconds']:.2f} | {r['fit_seconds']:.2f} | {r['predict_ms_per_1k_rows']:.2f} | {st} |")
    top = next((r for r in sel["rows"] if r["method"] == "tree" and r["features"]), None)
    if top:
        a(f"\nFeatures chosen by tree importance in every fold at k={top['k']}: {', '.join(top['features'])}.")

    try:
        b = load_json(cfg, "bench.json")
    except FileNotFoundError:
        b = None
    if b:
        a("\n## 7. Scale (Parquet, rolling-window feature job)\n")
        a("| engine | scale | rows | status | wall s | peak MB |\n|---|---|---|---|---|---|")
        for r in b["runs"]:
            ok = r["status"] == "ok"
            a(f"| {r['engine']} | {r['multiplier']}x | {r['rows']:,} | {r['status']} | "
              f"{r['seconds']:.2f} ({r['seconds_min']:.2f}-{r['seconds_max']:.2f}) | {r['peak_mb']:.0f} |" if ok else f"| {r['engine']} | {r['multiplier']}x | {r['rows']:,} | {r['status']} | | |")
        a("\n| scale | rows | Parquet (zstd) MB | bytes/row |\n|---|---|---|---|")
        for m, d in b["datasets"].items():
            a(f"| {m}x | {d['rows']:,} | {d['parquet_bytes'] / 2**20:.1f} | {d['parquet_bytes'] / d['rows']:.1f} |")
        d1 = b["datasets"]["1"]
        a(f"\n1x as CSV: {d1['csv_bytes'] / 2**20:.1f} MB; Parquet: {d1['parquet_bytes'] / 2**20:.1f} MB "
          f"({d1['csv_bytes'] / d1['parquet_bytes']:.1f}x smaller); in-memory Arrow: {d1['in_memory_bytes'] / 2**20:.1f} MB.")
        if b.get("agreement"):
            a("\nCross-engine agreement on identical job output: " + ", ".join(f"{k}x {'yes' if v else 'NO'}" for k, v in b["agreement"].items()) + ".")

    path = results_dir(cfg) / "summary.md"
    path.write_text("\n".join(L) + "\n", encoding="utf-8")
    return str(path)
