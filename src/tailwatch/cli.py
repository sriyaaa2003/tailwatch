"""Command line: `tailwatch <command>`. Each step reads its inputs from disk, so steps can be re-run independently."""
from __future__ import annotations

import argparse
import logging
import sys

import polars as pl

from .config import load_config
from .store import data_dir, load_json, read_features, results_dir, save_json, write_parquet

log = logging.getLogger("tailwatch")


def cmd_simulate(cfg) -> None:
    from .features import build_features
    from .sim import simulate
    raw = simulate(cfg)
    raw_size = write_parquet(raw, data_dir(cfg) / "raw_1s.parquet")
    feats = build_features(raw, cfg)
    write_parquet(feats, data_dir(cfg) / "features.parquet")
    by_d = feats.group_by("group").agg(pl.len().alias("rows"), pl.col("y").mean().alias("storm_rate")).sort("group")
    save_json(cfg, "data_summary.json", {
        "raw_rows": raw.height, "raw_parquet_bytes": raw_size, "feature_rows": feats.height,
        "storm_rate": float(feats["y"].mean()), "storm_count": int(feats["y"].sum()),
        "by_group": by_d.to_dicts()})
    log.info("raw %d rows, %d feature rows, storm rate %.2f%%", raw.height, feats.height, 100 * feats["y"].mean())


def cmd_hurst(cfg) -> None:
    from .hurst import validate_generator
    res = validate_generator(cfg)
    save_json(cfg, "hurst.json", res)
    for r in res["cells"]:
        log.info("cell %d (%s): H=%.2f theory=%.2f", r["cell_id"], r["district"], r["H_measured"], r["H_theory"])


def cmd_experiments(cfg) -> None:
    from . import experiments as ex, generalization, probabilistic, selection
    df = read_features(cfg)
    log.info("fitting out-of-district models")
    oof, meta = ex.compute_oof(df, cfg)
    save_json(cfg, "oof_meta.json", meta)
    save_json(cfg, "imbalance.json", ex.analyse_imbalance(df, oof, meta, cfg))
    save_json(cfg, "calibration.json", ex.analyse_calibration(df, oof, meta, cfg))
    save_json(cfg, "policy.json", ex.analyse_policy(df, oof, cfg))
    log.info("generalisation protocols")
    save_json(cfg, "generalization.json", generalization.run(df, oof, meta, cfg))
    log.info("probabilistic regression")
    save_json(cfg, "probabilistic.json", probabilistic.run(df, cfg))
    log.info("feature selection")
    save_json(cfg, "selection.json", selection.run(df, cfg))


def cmd_bench(cfg) -> None:
    from . import scale
    res = scale.run(cfg)
    save_json(cfg, "bench.json", res)
    for r in res["runs"]:
        log.info("%s x%d: %s %s", r["engine"], r["multiplier"], r["status"],
                 f"{r['seconds']:.1f}s {r['peak_mb']:.0f}MB" if r["status"] == "ok" else "")


def cmd_radar(cfg) -> None:
    import polars as pl
    from . import radar
    oof = pl.read_parquet(results_dir(cfg) / "oof.parquet")
    log.info("wrote %s", radar.build(read_features(cfg), oof, cfg))


def _real_cfg(cfg, with_data: bool = True):
    from . import real
    df = read_features(real.derive_config(cfg)) if with_data else None
    return real.derive_config(cfg, df)


def cmd_real_download(cfg) -> None:
    from . import real
    log.info("dataset at %s", real.download(cfg))


def cmd_real_prepare(cfg) -> None:
    from . import real
    dcfg = real.derive_config(cfg)
    raw = real.load_raw(cfg)
    s = real.streams(real.cell_bins(raw, cfg), cfg)
    feats = real.build_real_features(s, dcfg)
    write_parquet(s, data_dir(dcfg) / "streams.parquet")
    write_parquet(feats, data_dir(dcfg) / "features.parquet")
    save_json(dcfg, "data_summary.json", real.summarise(raw, s, feats, dcfg))
    log.info("%d streams, %d decision points, high-load rate %.2f%%", s["cell_id"].n_unique(), feats.height,
             100 * feats["y"].mean())


def cmd_real_validate(cfg) -> None:
    from . import real
    dcfg = real.derive_config(cfg)
    s = pl.read_parquet(data_dir(dcfg) / "streams.parquet")
    res = real.compare_burstiness(s, cfg)
    save_json(dcfg, "burstiness.json", res)
    log.info("Hurst real median %.2f vs twin %.2f (iid %.2f)", res["real"]["H"]["median"], res["twin"]["H"]["median"],
             res["iid_reference_H"])


def cmd_real_experiments(cfg) -> None:
    cmd_experiments(_real_cfg(cfg))


def cmd_real_report(cfg) -> None:
    cmd_report(_real_cfg(cfg))


def cmd_compare(cfg) -> None:
    from . import compare
    log.info("wrote %s", compare.write(cfg))


def cmd_report(cfg) -> None:
    import polars as pl
    from . import experiments as ex, plots, report
    save_json(cfg, "per_fold.json", ex.per_fold_pr_auc(pl.read_parquet(results_dir(cfg) / "oof.parquet"), cfg))
    plots.make_all(cfg)
    log.info("wrote %s", report.write(cfg))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="tailwatch")
    ap.add_argument("--config", default=None, help="YAML config (default: $TAILWATCH_CONFIG or configs/default.yaml)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("simulate", "hurst", "experiments", "bench", "radar", "report", "all", "real-download", "real-prepare",
                 "real-validate", "real-experiments", "real-report", "real", "compare"):
        sub.add_parser(name)
    w = sub.add_parser("_bench-worker")
    w.add_argument("--engine"); w.add_argument("--path"); w.add_argument("--window", type=int)
    w.add_argument("--thr", type=float); w.add_argument("--stride", type=int); w.add_argument("--repeats", type=int)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", stream=sys.stderr)
    if args.cmd == "_bench-worker":
        from . import scale
        scale.worker(args.engine, args.path, args.window, args.thr, args.stride, args.repeats)
        return 0
    cfg = load_config(args.config)
    steps = {"simulate": cmd_simulate, "hurst": cmd_hurst, "experiments": cmd_experiments, "bench": cmd_bench,
             "radar": cmd_radar, "report": cmd_report, "real-download": cmd_real_download,
             "real-prepare": cmd_real_prepare, "real-validate": cmd_real_validate,
             "real-experiments": cmd_real_experiments, "real-report": cmd_real_report, "compare": cmd_compare}
    plan = {"all": ["simulate", "hurst", "experiments", "bench", "radar", "report"],
            "real": ["real-download", "real-prepare", "real-validate", "real-experiments", "real-report", "compare"]}
    for name in plan.get(args.cmd, [args.cmd]):
        log.info("== %s", name)
        steps[name](cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
