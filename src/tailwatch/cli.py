"""Command line: `tailwatch <command>`. Each step reads its inputs from disk, so steps can be re-run independently."""
from __future__ import annotations

import argparse
import logging
import sys

import polars as pl

from .config import load_config
from .store import data_dir, load_json, read_features, results_dir, save_json, write_parquet

log = logging.getLogger("tailwatch")


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


def _msdata_cfg(cfg, with_data: bool = True):
    from . import real
    df = read_features(real.derive_config(cfg)) if with_data else None
    return real.derive_config(cfg, df)


def _fiveg_cfg(cfg, with_data: bool = True):
    from . import fiveg
    df = read_features(fiveg.derive_config(cfg)) if with_data else None
    return fiveg.derive_config(cfg, df)


def cmd_msdata_download(cfg) -> None:
    from . import real
    log.info("dataset at %s", real.download(cfg))


def cmd_msdata_prepare(cfg) -> None:
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


def cmd_msdata_burstiness(cfg) -> None:
    from . import real
    dcfg = real.derive_config(cfg)
    s = pl.read_parquet(data_dir(dcfg) / "streams.parquet")
    res = real.burstiness(s, cfg)
    save_json(dcfg, "burstiness.json", res)
    log.info("Hurst median %.2f (iid reference %.2f)", res["all"]["H"]["median"], res["iid_reference_H"])


def cmd_msdata_experiments(cfg) -> None:
    cmd_experiments(_msdata_cfg(cfg))


def cmd_msdata_report(cfg) -> None:
    cmd_report(_msdata_cfg(cfg))


def cmd_fiveg_download(cfg) -> None:
    from . import fiveg
    log.info("dataset at %s", fiveg.download(cfg))


def cmd_fiveg_prepare(cfg) -> None:
    from . import fiveg
    dcfg = fiveg.derive_config(cfg)
    s = fiveg.load_streams(cfg)
    feats = fiveg.build_features(s, cfg)
    write_parquet(s, data_dir(dcfg) / "streams.parquet")
    write_parquet(feats, data_dir(dcfg) / "features.parquet")
    save_json(dcfg, "data_summary.json", fiveg.summarise(s, feats, cfg))
    log.info("%d streams, %d decision points, channel-collapse rate %.2f%%", s["cell_id"].n_unique(), feats.height,
             100 * feats["y"].mean())


def cmd_fiveg_experiments(cfg) -> None:
    cmd_experiments(_fiveg_cfg(cfg))


def cmd_fiveg_report(cfg) -> None:
    cmd_report(_fiveg_cfg(cfg))


def cmd_compare(cfg) -> None:
    from . import compare
    log.info("wrote %s", compare.write(cfg))


def cmd_report(cfg) -> None:
    import polars as pl
    from . import experiments as ex, plots, report
    save_json(cfg, "per_fold.json", ex.per_fold_pr_auc(pl.read_parquet(results_dir(cfg) / "oof.parquet"), cfg))
    plots.make_all(cfg)
    log.info("wrote %s", report.write(cfg))


PLANS = {
    "msdata": ["msdata-download", "msdata-prepare", "msdata-burstiness", "msdata-experiments", "msdata-report"],
    "fiveg": ["fiveg-download", "fiveg-prepare", "fiveg-experiments", "fiveg-report"],
}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="tailwatch")
    ap.add_argument("--config", default=None, help="YAML config (default: $TAILWATCH_CONFIG or configs/default.yaml)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="run every step for one data set")
    r.add_argument("dataset", choices=sorted(PLANS))
    for name in [s for plan in PLANS.values() for s in plan] + ["compare"]:
        sub.add_parser(name)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", stream=sys.stderr)
    cfg = load_config(args.config)
    table = {
        "msdata-download": cmd_msdata_download, "msdata-prepare": cmd_msdata_prepare,
        "msdata-burstiness": cmd_msdata_burstiness, "msdata-experiments": cmd_msdata_experiments,
        "msdata-report": cmd_msdata_report, "fiveg-download": cmd_fiveg_download, "fiveg-prepare": cmd_fiveg_prepare,
        "fiveg-experiments": cmd_fiveg_experiments, "fiveg-report": cmd_fiveg_report, "compare": cmd_compare,
    }
    plan = PLANS[args.dataset] if args.cmd == "run" else [args.cmd]
    for name in plan:
        log.info("== %s", name)
        table[name](cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
