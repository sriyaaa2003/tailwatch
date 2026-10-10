"""Client-side 5G production traces (Raca, Leahy, Sreenan, Quinlan: "Beyond Throughput, the Next Generation", MMSys 2020).

What the raw data is: 83 sessions (188,711 s in all) logged once per second by a phone on a major Irish operator's
network, in a car or on a desk, while it streamed Netflix or Amazon Prime video or downloaded a file. Each row has
radio KPIs (RSRP, RSRQ, SNR, CQI), the serving cell, the network mode (4G or 5G), speed, and the downlink rate.

The question mirrors the msData study: given the last seconds of KPIs, will the *channel collapse* in the next
`horizon_s` seconds? A collapse is the worst CQI over that horizon being at most `cqi_event_max` (CQI runs 0 to 15;
the modem reports it, and the scheduler's rate choice follows it). Only moments where the channel is currently good
(CQI >= `cqi_now_min`) are scored, so the task is forecasting a drop, not noticing one. The group that is held out
whole is the (mobility, app) pair. The app is never a model input.
"""
from __future__ import annotations

import dataclasses
import io
import logging
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import polars as pl

from .config import Config, Features, Paths, Report, Storm

log = logging.getLogger("tailwatch.fiveg")

COLS = ["Timestamp", "Speed", "CellID", "NetworkMode", "RSRP", "RSRQ", "SNR", "CQI", "DL_bitrate", "State"]
APPS = {"Amazon_Prime": "prime", "Netflix": "netflix", "Download": "download"}


def derive_config(cfg: Config, df: pl.DataFrame | None = None) -> Config:
    f = cfg.fiveg
    t_end = int(df["t"].max()) + 1 if df is not None else cfg.duration_s
    return dataclasses.replace(
        cfg,
        paths=Paths(data_dir=f.data_dir, results_dir=f.results_dir),
        duration_s=t_end,
        storm=Storm(utilisation_threshold=1.0, horizon_s=f.horizon_s),   # peak is (15 - worst CQI) / (15 - cqi_event_max)
        features=Features(windows_s=f.windows_s, stride_s=f.stride_s),
        model=dataclasses.replace(cfg.model, val_mode="stream", purge_s=0),
        eval=dataclasses.replace(cfg.eval, n_boot=f.n_boot, block_s=f.block_s),
        generalization=dataclasses.replace(cfg.generalization, protocols=["random", "unseen"]),
        report=Report(group_label="mobility / app", kind="fiveg", event_label="channel collapse",
                      peak_label="worst channel-quality deficit relative to the event level"),
    )


def _dir(path: str) -> Path:
    from .store import ROOT
    p = Path(path)
    p = p if p.is_absolute() else ROOT / p
    p.mkdir(parents=True, exist_ok=True)
    return p


def raw_path(cfg: Config) -> Path:
    return _dir(cfg.fiveg.data_dir) / cfg.fiveg.filename


def download(cfg: Config) -> Path:
    f = cfg.fiveg
    dest = raw_path(cfg)
    if dest.exists() and dest.stat().st_size == f.expected_bytes:
        return dest
    part = dest.with_suffix(".part")
    log.info("downloading %s", f.url)
    with urllib.request.urlopen(f.url, timeout=60) as resp, open(part, "wb") as fh:
        while chunk := resp.read(1 << 20):
            fh.write(chunk)
    if part.stat().st_size != f.expected_bytes:
        size = part.stat().st_size
        part.unlink()
        raise OSError(f"download is {size} bytes, expected {f.expected_bytes}: refusing to use it")
    part.replace(dest)
    return dest


def parse_session(csv_bytes: bytes) -> pl.DataFrame:
    """One session on a regular 1 s grid. Rows sharing a second are averaged (the logger sometimes writes twice)."""
    d = pl.read_csv(io.BytesIO(csv_bytes), null_values=["-"], columns=COLS, infer_schema_length=2000,
                    schema_overrides={"CQI": pl.Float64, "RSRP": pl.Float64, "RSRQ": pl.Float64, "SNR": pl.Float64,
                                      "DL_bitrate": pl.Float64, "Speed": pl.Float64, "CellID": pl.Int64})
    d = d.with_columns(pl.col("Timestamp").str.strptime(pl.Datetime, "%Y.%m.%d_%H.%M.%S").dt.epoch("s").alias("ts"))
    d = (d.group_by("ts", maintain_order=True)
         .agg(pl.col("CQI").mean(), pl.col("RSRP").mean(), pl.col("RSRQ").mean(), pl.col("SNR").mean(),
              pl.col("DL_bitrate").mean(), pl.col("Speed").mean(), pl.col("CellID").last(),
              (pl.col("NetworkMode") == "5G").mean().alias("nr"), (pl.col("State") == "D").mean().alias("active")))
    return d.sort("ts")


def load_streams(cfg: Config) -> pl.DataFrame:
    """Every session split at timestamp jumps longer than max_gap_s, streams shorter than min_stream_s dropped,
    short holes forward-filled. Columns: group, app, cell_id (stream id), t (seconds), and the KPIs."""
    f = cfg.fiveg
    path = raw_path(cfg)
    if not path.exists():
        raise FileNotFoundError(f"{path} missing: run `tailwatch fiveg-download`")
    out, cid, t_off = [], 0, 0
    with zipfile.ZipFile(path) as z:
        for name in sorted(n for n in z.namelist() if n.endswith(".csv") and not n.startswith("__MACOSX")):
            parts = Path(name).parts
            app = APPS[parts[1]]
            mob = parts[2].lower()
            s = parse_session(z.read(name))
            ts = s["ts"].to_numpy()
            if len(ts) < 2:
                continue
            seg = np.concatenate([[0], np.cumsum(np.diff(ts) > f.max_gap_s)])
            for k in np.unique(seg):
                part = s.filter(pl.Series(seg == k))
                if part.height < f.min_stream_s:
                    continue
                grid = pl.DataFrame({"ts": np.arange(part["ts"].min(), part["ts"].max() + 1)})
                part = (grid.join(part, on="ts", how="left").sort("ts")
                        .with_columns(pl.exclude("ts").forward_fill())
                        .drop_nulls(["CQI", "RSRP", "RSRQ", "SNR", "DL_bitrate", "Speed", "CellID"]))
                part = part.with_columns(
                    pl.lit(f"{mob} / {app}").alias("group"), pl.lit(app).alias("app"), pl.lit(cid).cast(pl.Int32).alias("cell_id"),
                    (pl.col("ts") - pl.col("ts").min()).cast(pl.Int32).alias("t"),
                    (pl.col("DL_bitrate") / f.dl_kbps_per_mbps).alias("dl"))
                out.append(part.drop("ts", "DL_bitrate"))
                cid += 1
    return pl.concat(out).sort(["cell_id", "t"])


def feature_names(cfg: Config) -> list[str]:
    n = ["cqi_now", "rsrp_now", "rsrq_now", "snr_now", "dl_now", "speed_now", "nr_now"]
    for w in cfg.fiveg.windows_s:
        n += [f"cqi_mean_{w}", f"cqi_std_{w}", f"cqi_min_{w}", f"cqi_delta_{w}", f"rsrp_mean_{w}", f"rsrp_delta_{w}",
              f"snr_mean_{w}", f"dl_mean_{w}", f"speed_mean_{w}", f"handovers_{w}", f"nr_share_{w}", f"active_share_{w}"]
    return n


def build_features(s: pl.DataFrame, cfg: Config) -> pl.DataFrame:
    """Features look strictly backward (trailing windows), the label strictly forward (the next horizon_s seconds)."""
    f = cfg.fiveg
    H, stride = f.horizon_s, f.stride_s
    df = s.sort(["cell_id", "t"]).with_columns(
        pl.col("dl").log1p().alias("dl_l"),
        (pl.col("CellID") != pl.col("CellID").shift(1).over("cell_id")).fill_null(False).cast(pl.Float64).alias("ho"))
    cqi = pl.col("CQI")
    exprs: list[pl.Expr] = [cqi.alias("cqi_now"), pl.col("RSRP").alias("rsrp_now"), pl.col("RSRQ").alias("rsrq_now"),
                            pl.col("SNR").alias("snr_now"), pl.col("dl_l").alias("dl_now"),
                            pl.col("Speed").alias("speed_now"), pl.col("nr").alias("nr_now")]
    for w in cfg.fiveg.windows_s:
        exprs += [
            cqi.rolling_mean(w).over("cell_id").alias(f"cqi_mean_{w}"),
            cqi.rolling_std(w).over("cell_id").alias(f"cqi_std_{w}"),
            cqi.rolling_min(w).over("cell_id").alias(f"cqi_min_{w}"),
            (cqi - cqi.shift(w)).over("cell_id").alias(f"cqi_delta_{w}"),
            pl.col("RSRP").rolling_mean(w).over("cell_id").alias(f"rsrp_mean_{w}"),
            (pl.col("RSRP") - pl.col("RSRP").shift(w)).over("cell_id").alias(f"rsrp_delta_{w}"),
            pl.col("SNR").rolling_mean(w).over("cell_id").alias(f"snr_mean_{w}"),
            pl.col("dl_l").rolling_mean(w).over("cell_id").alias(f"dl_mean_{w}"),
            pl.col("Speed").rolling_mean(w).over("cell_id").alias(f"speed_mean_{w}"),
            pl.col("ho").rolling_sum(w).over("cell_id").alias(f"handovers_{w}"),
            pl.col("nr").rolling_mean(w).over("cell_id").alias(f"nr_share_{w}"),
            pl.col("active").rolling_mean(w).over("cell_id").alias(f"active_share_{w}"),
        ]
    worst = cqi.reverse().rolling_min(H).reverse().shift(-1).over("cell_id")
    peak = ((15.0 - worst) / (15.0 - f.cqi_event_max)).alias("y_fut_max")
    out = df.select("cell_id", "t", "group", pl.col("app").alias("top_label"), cqi.alias("_cqi"),
                    ((15.0 - cqi) / (15.0 - f.cqi_event_max)).alias("level_now"), *exprs, peak)
    out = out.with_columns((pl.col("y_fut_max") >= 1.0).cast(pl.Int8).alias("y"))
    cols = feature_names(cfg)
    out = (out.filter((pl.col("t") % stride == 0) & (pl.col("_cqi") >= f.cqi_now_min))
           .drop_nulls(cols + ["y_fut_max"])
           .with_columns(pl.col(cols).cast(pl.Float32)))
    return out.select(["cell_id", "t", "group", "top_label", "level_now", *cols, "y_fut_max", "y"])


def summarise(s: pl.DataFrame, feats: pl.DataFrame, cfg: Config) -> dict:
    by_group = (feats.group_by("group").agg(pl.len().alias("rows"), pl.col("y").mean().alias("storm_rate"),
                                            pl.col("cell_id").n_unique().alias("streams")).sort("group"))
    sec = s.group_by("group").agg(pl.len().alias("seconds"))
    by_group = by_group.join(sec, on="group")
    f = cfg.fiveg
    return {
        "streams": int(s["cell_id"].n_unique()), "stream_seconds": int(s.height), "feature_rows": feats.height,
        "storm_rate": float(feats["y"].mean()), "storm_count": int(feats["y"].sum()),
        "cqi_now_min": f.cqi_now_min, "cqi_event_max": f.cqi_event_max, "horizon_s": f.horizon_s,
        "share_cqi_le_event": float((s["CQI"] <= f.cqi_event_max).mean()),
        "share_5g": float(s["nr"].mean()),
        "by_group": by_group.to_dicts(),
    }
