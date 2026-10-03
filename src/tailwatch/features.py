"""Feature + label engineering in Polars (lazy where possible). Labels look strictly forward, features strictly backward."""
from __future__ import annotations

import math

import polars as pl

from .config import Config
from .sim import build_layout


def feature_names(cfg: Config) -> list[str]:
    names = ["util_now", "users_now", "rsrp_now", "tod_sin", "tod_cos", "nbr_util_mean", "nbr_util_max"]
    for w in cfg.features.windows_s:
        names += [f"util_mean_{w}", f"util_std_{w}", f"util_max_{w}", f"util_min_{w}", f"util_cv_{w}",
                  f"util_delta_{w}", f"users_delta_{w}", f"rsrp_mean_{w}"]
    return names


def neighbor_edges(cfg: Config) -> pl.DataFrame:
    layout = build_layout(cfg)
    src, dst = [], []
    for cid, nbrs in enumerate(layout.neighbors):
        for n in nbrs:
            src.append(cid)
            dst.append(n)
    return pl.DataFrame({"cell_id": src, "nbr": dst}, schema={"cell_id": pl.Int32, "nbr": pl.Int32})


def build_features(raw: pl.DataFrame, cfg: Config) -> pl.DataFrame:
    """raw: 1 s long frame from sim.simulate. Returns one row per (cell, decision time) with features and labels.

    Labels: y = 1 if utilisation first reaches the threshold within (t, t+H], among rows not already congested.
    y_fut_max = max utilisation over (t, t+H], the regression target.
    """
    H = cfg.storm.horizon_s
    thr = cfg.storm.utilisation_threshold
    df = raw.sort(["cell_id", "t"])

    # spatial context: mean/max utilisation of the hex neighbours at the same instant
    nbr = (df.select(pl.col("cell_id").alias("nbr"), "t", pl.col("util").alias("nbr_util"))
           .join(neighbor_edges(cfg), on="nbr")
           .group_by(["cell_id", "t"])
           .agg(pl.col("nbr_util").mean().alias("nbr_util_mean"), pl.col("nbr_util").max().alias("nbr_util_max")))
    df = df.join(nbr, on=["cell_id", "t"], how="left").sort(["cell_id", "t"])

    exprs: list[pl.Expr] = [
        pl.col("util").alias("util_now"),
        pl.col("users").alias("users_now"),
        pl.col("rsrp_dbm").alias("rsrp_now"),
        (2 * math.pi * pl.col("t") / cfg.sim.diurnal_period_s).sin().alias("tod_sin"),
        (2 * math.pi * pl.col("t") / cfg.sim.diurnal_period_s).cos().alias("tod_cos"),
    ]
    for w in cfg.features.windows_s:
        u = pl.col("util")
        exprs += [
            u.rolling_mean(w).over("cell_id").alias(f"util_mean_{w}"),
            u.rolling_std(w).over("cell_id").alias(f"util_std_{w}"),
            u.rolling_max(w).over("cell_id").alias(f"util_max_{w}"),
            u.rolling_min(w).over("cell_id").alias(f"util_min_{w}"),
            (u - u.shift(w)).over("cell_id").alias(f"util_delta_{w}"),
            (pl.col("users") - pl.col("users").shift(w)).over("cell_id").alias(f"users_delta_{w}"),
            pl.col("rsrp_dbm").rolling_mean(w).over("cell_id").alias(f"rsrp_mean_{w}"),
        ]
    # forward window (t, t+H]: reverse -> rolling_max -> reverse gives max over [t, t+H-1]; shift(-1) moves it to (t, t+H]
    fut = pl.col("util").reverse().rolling_max(H).reverse().shift(-1).over("cell_id").alias("y_fut_max")
    out = df.select("cell_id", "t", "group", "row", "col", *exprs, "nbr_util_mean", "nbr_util_max", fut)
    for w in cfg.features.windows_s:
        out = out.with_columns((pl.col(f"util_std_{w}") / (pl.col(f"util_mean_{w}").abs() + 1e-6)).alias(f"util_cv_{w}"))
    out = out.with_columns(((pl.col("y_fut_max") >= thr).cast(pl.Int8)).alias("y"))

    cols = feature_names(cfg)
    out = (out.filter((pl.col("t") % cfg.features.stride_s == 0) & (pl.col("util_now") < thr))
           .drop_nulls(cols + ["y_fut_max"])
           .with_columns(pl.col(cols).cast(pl.Float32)))
    return out.select(["cell_id", "t", "group", "row", "col", *cols, "y_fut_max", "y"])
