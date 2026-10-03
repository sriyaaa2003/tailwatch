"""Real-trace study on msData (Open RAN 5G testbed, OpenIreland; 3.2 M records, one cell, 4 UEs).

What the raw data is: per-UE records of MAC/PHY metrics, one record per ~90 TTIs (about 155 ms of wall clock apart),
not 1 ms samples, plus a mobility pattern and a traffic label per record. There is no per-cell load column and the
cell's capacity is not observed. This module therefore builds a **cell-level load** by summing the downlink bit rate of
all UEs seen in the same time bin, cuts it into contiguous streams, and asks the same question as the twin: will
the load reach a configured high-load level within the next few seconds? The configured level is a labelling
convention (cfg.real.storm_mbps), not a measured congestion point.

The mobility pattern (car / bus / train / static / pedestrian) plays the part of the district: it is held out whole.
The traffic label is never a model input (a deployed forecaster would not have it); it is used only in descriptive
tables.
"""
from __future__ import annotations

import dataclasses
import logging
import urllib.request
from pathlib import Path

import numpy as np
import polars as pl

from .config import Config, Features, Paths, Report, Storm
from .hurst import hurst_aggvar

log = logging.getLogger("tailwatch.real")

RAW_COLS = ["timestamp", "mob_pattern", "label", "id_ue", "mac_dl_brate", "mac_dl_cqi", "mac_dl_mcs",
            "mac_dl_buffer", "phy_ul_pusch_sinr"]
RAW_SCHEMA = {"timestamp": pl.Float64, "mob_pattern": pl.String, "label": pl.String, "id_ue": pl.Int64,
              "mac_dl_brate": pl.Float64, "mac_dl_cqi": pl.Float64, "mac_dl_mcs": pl.Float64,
              "mac_dl_buffer": pl.Float64, "phy_ul_pusch_sinr": pl.Float64}


# ----------------------------------------------------------------------------------------------------- configuration
def derive_config(cfg: Config, df: pl.DataFrame | None = None) -> Config:
    """The shared pipeline reads everything from a Config; the real study is the same Config with the real-data
    settings swapped in (paths, storm level, horizon, windows, stream-based validation, group label)."""
    r = cfg.real
    t_end = int(df["t"].max()) + 1 if df is not None else cfg.sim.duration_s
    return dataclasses.replace(
        cfg,
        paths=Paths(data_dir=r.data_dir, results_dir=r.results_dir),
        sim=dataclasses.replace(cfg.sim, duration_s=t_end),
        storm=Storm(utilisation_threshold=1.0, horizon_s=r.horizon_s),   # util is load / storm_mbps
        features=Features(windows_s=r.windows_s, stride_s=r.stride_s),
        model=dataclasses.replace(cfg.model, val_mode="stream", purge_s=0),
        eval=dataclasses.replace(cfg.eval, n_boot=r.n_boot, block_s=r.block_s),
        generalization=dataclasses.replace(cfg.generalization, protocols=["random", "unseen"]),
        report=Report(group_label="mobility pattern", kind="real", event_label="burst",
                      peak_label="peak load relative to the configured high-load level"),
    )


def _dir(path: str) -> Path:
    from .store import ROOT
    p = Path(path)
    p = p if p.is_absolute() else ROOT / p
    p.mkdir(parents=True, exist_ok=True)
    return p


def raw_path(cfg: Config) -> Path:
    return _dir(cfg.real.data_dir) / cfg.real.filename


# ----------------------------------------------------------------------------------------------------- download
def download(cfg: Config) -> Path:
    """Fetch the CSV once. Verifies the byte count so a truncated transfer is never mistaken for the dataset."""
    r = cfg.real
    dest = raw_path(cfg)
    if dest.exists() and dest.stat().st_size == r.expected_bytes:
        log.info("dataset already present (%d bytes)", dest.stat().st_size)
        return dest
    part = dest.with_suffix(".part")
    log.info("downloading %s (%.0f MB)", r.url, r.expected_bytes / 2**20)
    with urllib.request.urlopen(r.url, timeout=60) as resp, open(part, "wb") as fh:
        while chunk := resp.read(1 << 20):
            fh.write(chunk)
    size = part.stat().st_size
    if size != r.expected_bytes:
        part.unlink()
        raise OSError(f"download is {size} bytes, expected {r.expected_bytes}: refusing to use it")
    part.replace(dest)
    return dest


# ----------------------------------------------------------------------------------------------------- aggregation
def load_raw(cfg: Config) -> pl.DataFrame:
    path = raw_path(cfg)
    if not path.exists():
        raise FileNotFoundError(f"{path} missing: run `tailwatch real-download`")
    return pl.read_csv(path, separator=";", columns=RAW_COLS, schema_overrides=RAW_SCHEMA)


def cell_bins(raw: pl.DataFrame, cfg: Config) -> pl.DataFrame:
    """Per (group, time bin): total downlink load over all UEs seen in that bin and the mean channel state.

    A UE without a record in a bin contributes nothing: the cell only reports a UE while it is scheduled, so an
    absent record is read as no traffic. This is an assumption about the testbed, stated in the README."""
    r = cfg.real
    g = r.group_col
    d = raw.with_columns((pl.col("timestamp") / r.timestamp_ticks_per_s / r.bin_s).floor().cast(pl.Int64).alias("bin"))
    ue = d.group_by([g, "bin", "id_ue"]).agg(
        (pl.col("mac_dl_brate").mean() / r.rate_unit_bps).alias("rate"),
        pl.col("mac_dl_cqi").mean().alias("cqi"), pl.col("mac_dl_mcs").mean().alias("mcs"),
        pl.col("mac_dl_buffer").mean().alias("buf"), pl.col("phy_ul_pusch_sinr").mean().alias("sinr"),
        pl.col("label").first().alias("label"))
    return (ue.group_by([g, "bin"]).agg(
        pl.col("rate").sum().alias("load"), pl.len().alias("users"), pl.col("cqi").mean(), pl.col("mcs").mean(),
        pl.col("buf").sum().alias("buf"), pl.col("sinr").mean(),
        pl.col("label").sort_by("rate").last().alias("top_label"))
        .rename({g: "group"}).sort(["group", "bin"]))


def streams(bins: pl.DataFrame, cfg: Config) -> pl.DataFrame:
    """Split each group's bin timeline at silences longer than max_gap_s, keep streams of at least min_stream_s,
    and fill the short silences inside a stream (load 0, channel state carried forward)."""
    r = cfg.real
    max_gap = max(1, int(round(r.max_gap_s / r.bin_s)))
    min_len = int(round(r.min_stream_s / r.bin_s))
    seg = bins.with_columns(
        (pl.col("bin").diff().over("group").fill_null(max_gap + 1) > max_gap).cum_sum().over("group").alias("seg"))
    spans = (seg.group_by(["group", "seg"]).agg(pl.col("bin").min().alias("b0"), pl.col("bin").max().alias("b1"))
             .filter(pl.col("b1") - pl.col("b0") + 1 >= min_len).sort(["group", "b0"])
             .with_columns(pl.int_range(pl.len()).cast(pl.Int32).alias("cell_id")))
    grid = (spans.with_columns(pl.int_ranges(pl.col("b0"), pl.col("b1") + 1).alias("bin"))
            .explode("bin", empty_as_null=False).select("group", "seg", "cell_id", "bin"))
    out = (grid.join(seg.drop("seg"), on=["group", "bin"], how="left")
           .sort(["cell_id", "bin"])
           .with_columns(pl.col("load", "users", "buf").fill_null(0.0),
                         pl.col("cqi", "mcs", "sinr").forward_fill().over("cell_id"),
                         pl.col("top_label").fill_null("none"))
           .drop_nulls(["cqi", "mcs", "sinr"]))
    t0 = out["bin"].min()
    return out.with_columns((pl.col("bin") - t0).cast(pl.Int32).alias("t")).drop("bin", "seg")


# ----------------------------------------------------------------------------------------------------- features
def real_feature_names(cfg: Config) -> list[str]:
    n = ["util_now", "users_now", "cqi_now", "mcs_now", "buffer_now", "sinr_now"]
    for w in cfg.features.windows_s:
        n += [f"util_mean_{w}", f"util_std_{w}", f"util_max_{w}", f"util_min_{w}", f"util_cv_{w}", f"util_delta_{w}",
              f"users_delta_{w}", f"cqi_mean_{w}", f"buffer_mean_{w}", f"sinr_mean_{w}"]
    return n


def build_real_features(s: pl.DataFrame, cfg: Config) -> pl.DataFrame:
    """Same construction as the twin: trailing windows for features, strictly forward window for the label.
    Windows and horizon are in seconds in the config and converted to bins here."""
    r = cfg.real
    to_bins = lambda sec: max(1, int(round(sec / r.bin_s)))
    H, stride = to_bins(r.horizon_s), to_bins(r.stride_s)
    df = s.with_columns((pl.col("load") / r.storm_mbps).alias("util"), pl.col("buf").log1p().alias("buf_l")) \
        .sort(["cell_id", "t"])
    u = pl.col("util")
    exprs: list[pl.Expr] = [u.alias("util_now"), pl.col("users").alias("users_now"), pl.col("cqi").alias("cqi_now"),
                            pl.col("mcs").alias("mcs_now"), pl.col("buf_l").alias("buffer_now"),
                            pl.col("sinr").alias("sinr_now")]
    for sec in cfg.features.windows_s:
        w = to_bins(sec)
        exprs += [
            u.rolling_mean(w).over("cell_id").alias(f"util_mean_{sec}"),
            u.rolling_std(w).over("cell_id").alias(f"util_std_{sec}"),
            u.rolling_max(w).over("cell_id").alias(f"util_max_{sec}"),
            u.rolling_min(w).over("cell_id").alias(f"util_min_{sec}"),
            (u - u.shift(w)).over("cell_id").alias(f"util_delta_{sec}"),
            (pl.col("users") - pl.col("users").shift(w)).over("cell_id").alias(f"users_delta_{sec}"),
            pl.col("cqi").rolling_mean(w).over("cell_id").alias(f"cqi_mean_{sec}"),
            pl.col("buf_l").rolling_mean(w).over("cell_id").alias(f"buffer_mean_{sec}"),
            pl.col("sinr").rolling_mean(w).over("cell_id").alias(f"sinr_mean_{sec}"),
        ]
    fut = u.reverse().rolling_max(H).reverse().shift(-1).over("cell_id").alias("y_fut_max")
    out = df.select("cell_id", "t", "group", "top_label", *exprs, fut)
    for sec in cfg.features.windows_s:
        out = out.with_columns((pl.col(f"util_std_{sec}") / (pl.col(f"util_mean_{sec}").abs() + 1e-6)).alias(f"util_cv_{sec}"))
    thr = cfg.storm.utilisation_threshold
    out = out.with_columns((pl.col("y_fut_max") >= thr).cast(pl.Int8).alias("y"))
    cols = real_feature_names(cfg)
    out = (out.filter((pl.col("t") % stride == 0) & (pl.col("util_now") < thr))
           .drop_nulls(cols + ["y_fut_max"])
           .with_columns(pl.col(cols).cast(pl.Float32)))
    return out.select(["cell_id", "t", "group", "top_label", *cols, "y_fut_max", "y"])


# ----------------------------------------------------------------------------------------------------- summaries
def record_cadence_ms(raw: pl.DataFrame, cfg: Config) -> dict:
    """Median wall-clock spacing between consecutive records of the same UE run (gaps under 1 s only)."""
    r = cfg.real
    d = (raw.select("mob_pattern", "label", "id_ue", (pl.col("timestamp") / r.timestamp_ticks_per_s).alias("s"))
         .sort(["mob_pattern", "label", "id_ue", "s"])
         .with_columns(pl.col("s").diff().over(["mob_pattern", "label", "id_ue"]).alias("dt"))
         .filter((pl.col("dt") > 0) & (pl.col("dt") < 1.0)))
    q = np.quantile(d["dt"].to_numpy() * 1000.0, [0.05, 0.5, 0.95])
    return {"p05_ms": float(q[0]), "median_ms": float(q[1]), "p95_ms": float(q[2])}


def summarise(raw: pl.DataFrame, s: pl.DataFrame, feats: pl.DataFrame, cfg: Config) -> dict:
    by_group = (feats.group_by("group").agg(pl.len().alias("rows"), pl.col("y").mean().alias("storm_rate"),
                                            pl.col("cell_id").n_unique().alias("streams")).sort("group"))
    by_label = (feats.group_by("top_label").agg(pl.len().alias("rows"), pl.col("y").mean().alias("storm_rate"))
                .sort("rows", descending=True))
    load = s["load"].to_numpy()
    return {
        "raw_rows": raw.height, "cell_bins": s.height, "streams": int(s["cell_id"].n_unique()),
        "stream_seconds": float(s.height * cfg.real.bin_s), "feature_rows": feats.height,
        "storm_rate": float(feats["y"].mean()), "storm_count": int(feats["y"].sum()),
        "storm_mbps": cfg.real.storm_mbps, "horizon_s": cfg.real.horizon_s,
        "load_quantiles_mbps": {str(q): float(np.quantile(load, q)) for q in (0.5, 0.9, 0.95, 0.99)},
        "frac_load_ge_storm": float((load >= cfg.real.storm_mbps).mean()),
        "record_cadence": record_cadence_ms(raw, cfg),
        "by_group": by_group.to_dicts(), "by_top_label": by_label.to_dicts(),
    }


# ----------------------------------------------------------------------------------------------------- twin vs real
def _acf(x: np.ndarray, lag: int) -> float:
    x = x - x.mean()
    den = float((x * x).sum())
    return float((x[:-lag] * x[lag:]).sum() / den) if den > 0 and len(x) > lag else float("nan")


def burst_stats(x: np.ndarray, cfg: Config) -> dict | None:
    if len(x) < 2 or x.mean() <= 0 or x.std() == 0:
        return None
    return {"H": hurst_aggvar(x, cfg.real.hurst_min_scale_s, cfg.real.hurst_max_scale_div),
            "cv": float(x.std(ddof=1) / x.mean()), "acf1": _acf(x, 1), "acf10": _acf(x, 10)}


def _summ(rows: list[dict]) -> dict:
    out = {"n": len(rows)}
    for k in ("H", "cv", "acf1", "acf10"):
        v = np.array([r[k] for r in rows if np.isfinite(r[k])])
        out[k] = ({"median": float(np.median(v)), "q25": float(np.quantile(v, 0.25)), "q75": float(np.quantile(v, 0.75))}
                  if len(v) else None)
    return out


def compare_burstiness(s: pl.DataFrame, cfg: Config) -> dict:
    """Does the twin's traffic look like the real trace? Same statistics, same estimator, same series length.

    Real: every stream of at least hurst_min_stream_s seconds. Twin: the plain generator (surges and diurnal cycle off)
    cut into non-overlapping chunks as long as the median real stream, so the Hurst estimator's finite-length bias
    is matched. Differences in level are expected (different system); the question is whether the *shape* agrees."""
    from .sim import simulate
    min_len = int(round(cfg.real.hurst_min_stream_s / cfg.real.bin_s))
    real_rows, lens = [], []
    for cid, g in s.group_by("cell_id", maintain_order=True):
        if g.height < min_len:
            continue
        st = burst_stats(g["load"].to_numpy().astype(float), cfg)
        if st:
            st.update({"group": g["group"][0], "len": g.height})
            real_rows.append(st)
            lens.append(g.height)
    L = int(np.median(lens)) if lens else min_len
    plain = dataclasses.replace(cfg.sim, events=[], diurnal_amplitude=0.0)

    def twin_chunks(c: Config) -> list[dict]:
        rows = []
        for _, g in simulate(c, sim=plain).group_by("cell_id", maintain_order=True):
            x = g["load_mbps"].to_numpy().astype(float)
            for i in range(len(x) // L):
                st = burst_stats(x[i * L:(i + 1) * L], cfg)
                if st:
                    rows.append(st)
        return rows

    twin_rows = twin_chunks(cfg)
    few = dataclasses.replace(cfg, city=dataclasses.replace(cfg.city, districts=[
        dataclasses.replace(d, n_sources=cfg.real.twin_few_sources) for d in cfg.city.districts]))
    twin_few_rows = twin_chunks(few)
    by_group = {}
    for grp in sorted({r["group"] for r in real_rows}):
        by_group[grp] = _summ([r for r in real_rows if r["group"] == grp])
    return {"chunk_len_s": L * cfg.real.bin_s, "min_stream_s": cfg.real.hurst_min_stream_s,
            "real": _summ(real_rows), "real_by_group": by_group, "twin": _summ(twin_rows),
            "twin_few_sources": {"sources_per_cell": cfg.real.twin_few_sources, **_summ(twin_few_rows)},
            "iid_reference_H": float(hurst_aggvar(np.random.default_rng(cfg.seed).normal(size=L * 8) + 5.0,
                                                  cfg.real.hurst_min_scale_s, cfg.real.hurst_max_scale_div))}
