"""Scale benchmark: the same rolling-window feature job on Parquet at 1x/10x/100x across engines.

Each (engine, scale) runs in its own subprocess so wall time and peak memory are isolated and a crash or timeout is
recorded as a result rather than killing the benchmark. Scaling replicates the base city's cells (new cell ids, same
time axis); it measures systems throughput, not modelling behaviour.
"""
from __future__ import annotations

import dataclasses
import json
import subprocess
import sys
import time

import polars as pl

from .config import Config
from .sim import simulate
from .store import data_dir, write_parquet


def _base(cfg: Config) -> pl.DataFrame:
    s = cfg.scale
    sim = dataclasses.replace(cfg.sim, duration_s=s.base_duration_s, events=[], diurnal_period_s=s.base_duration_s)
    return simulate(cfg, sim=sim, cells=list(range(s.base_cells))).select(
        "cell_id", "t", "users", "throughput_mbps", "rsrp_dbm", "util")


def prepare(cfg: Config) -> dict[int, dict]:
    base = _base(cfg)
    info = {}
    for m in cfg.scale.multipliers:
        path = data_dir(cfg) / f"scale_{m}x.parquet"
        big = pl.concat([base.with_columns(pl.col("cell_id") + i * cfg.scale.base_cells) for i in range(m)])
        size = write_parquet(big, path)
        info[m] = {"path": str(path), "rows": big.height, "parquet_bytes": size}
        del big
    csv_path = data_dir(cfg) / "scale_1x.csv"
    base.write_csv(csv_path)
    info[1]["csv_bytes"] = csv_path.stat().st_size
    csv_path.unlink()
    info[1]["in_memory_bytes"] = int(base.estimated_size())
    return info


# ------------------------------------------------------------------ the workload, one implementation per engine
def _job_polars(path: str, w: int, thr: float, stride: int) -> dict:
    lf = (pl.scan_parquet(path).sort(["cell_id", "t"])
          .with_columns(pl.col("util").rolling_mean(w).over("cell_id").alias("m"),
                        pl.col("util").rolling_std(w).over("cell_id").alias("s"),
                        pl.col("util").rolling_max(w).over("cell_id").alias("x"),
                        pl.col("throughput_mbps").rolling_mean(w).over("cell_id").alias("th"))
          .filter(pl.col("m").is_not_null() & (pl.col("t") % stride == 0))
          .select(pl.len().alias("n"), pl.col("s").mean().alias("mean_std"), pl.col("x").max().alias("max_x"),
                  (pl.col("x") >= thr).mean().alias("frac_hi")))
    return lf.collect().row(0, named=True)


def _job_pandas(path: str, w: int, thr: float, stride: int) -> dict:
    import pandas as pd
    df = pd.read_parquet(path).sort_values(["cell_id", "t"])
    g = df.groupby("cell_id")
    roll = lambda col, fn: getattr(g[col].rolling(w), fn)().reset_index(level=0, drop=True)
    df["m"], df["s"], df["x"], df["th"] = roll("util", "mean"), roll("util", "std"), roll("util", "max"), roll("throughput_mbps", "mean")
    df = df[df["m"].notna() & (df["t"] % stride == 0)]
    return {"n": len(df), "mean_std": df["s"].mean(), "max_x": df["x"].max(), "frac_hi": (df["x"] >= thr).mean()}


def _job_duckdb(path: str, w: int, thr: float, stride: int) -> dict:
    import duckdb
    q = f"""
    WITH f AS (
      SELECT t, avg(util) OVER win AS m, stddev_samp(util) OVER win AS s, max(util) OVER win AS x,
             avg(throughput_mbps) OVER win AS th, count(*) OVER win AS c
      FROM read_parquet('{path}')
      WINDOW win AS (PARTITION BY cell_id ORDER BY t ROWS BETWEEN {w - 1} PRECEDING AND CURRENT ROW))
    SELECT count(*) AS n, avg(s) AS mean_std, max(x) AS max_x, avg((x >= {thr})::DOUBLE) AS frac_hi
    FROM f WHERE c = {w} AND t % {stride} = 0"""
    n, mean_std, max_x, frac_hi = duckdb.connect().execute(q).fetchone()
    return {"n": n, "mean_std": mean_std, "max_x": max_x, "frac_hi": frac_hi}


JOBS = {"polars": _job_polars, "pandas": _job_pandas, "duckdb": _job_duckdb}


def _peak_mb() -> float:
    import psutil
    mi = psutil.Process().memory_info()
    return getattr(mi, "peak_wset", mi.rss) / 2**20


def worker(engine: str, path: str, w: int, thr: float, stride: int, repeats: int) -> None:
    """Import the engine first (cold imports and DLL loading are not the workload), then time `repeats` runs."""
    if engine == "pandas":
        import pandas  # noqa: F401
    elif engine == "duckdb":
        import duckdb  # noqa: F401
    secs, res = [], {}
    for _ in range(repeats):
        t0 = time.perf_counter()
        res = JOBS[engine](path, w, thr, stride)
        secs.append(time.perf_counter() - t0)
    secs.sort()
    print(json.dumps({"seconds": secs[len(secs) // 2], "seconds_min": secs[0], "seconds_max": secs[-1],
                      "peak_mb": _peak_mb(), "result": {k: float(v) for k, v in res.items()}}))


def run(cfg: Config) -> dict:
    info = prepare(cfg)
    out = {"datasets": info, "runs": []}
    for engine in cfg.scale.engines:
        for m in cfg.scale.multipliers:
            cmd = [sys.executable, "-m", "tailwatch.cli", "_bench-worker", "--engine", engine, "--path", info[m]["path"],
                   "--window", str(cfg.scale.window_s), "--thr", str(cfg.storm.utilisation_threshold),
                   "--stride", str(cfg.features.stride_s),
                   "--repeats", str(cfg.scale.repeats)]
            rec = {"engine": engine, "multiplier": m, "rows": info[m]["rows"]}
            try:
                p = subprocess.run(cmd, capture_output=True, text=True, timeout=cfg.scale.timeout_s * cfg.scale.repeats)
                if p.returncode != 0:
                    tail = (p.stderr.strip().splitlines() or ["unknown error"])[-1]
                    rec["status"] = f"failed: {tail[:200]}"
                else:
                    rec.update(json.loads(p.stdout.strip().splitlines()[-1]))
                    rec["status"] = "ok"
            except subprocess.TimeoutExpired:
                rec["status"] = f"timeout > {cfg.scale.timeout_s}s"
            out["runs"].append(rec)
            if rec["status"] != "ok":
                break  # a larger scale will not do better on this engine
    # cross-engine agreement on the same job (correctness check for the benchmark itself)
    for m in cfg.scale.multipliers:
        rs = [r for r in out["runs"] if r["multiplier"] == m and r["status"] == "ok"]
        if len(rs) > 1:
            ref = rs[0]["result"]
            out.setdefault("agreement", {})[str(m)] = all(
                abs(r["result"]["mean_std"] - ref["mean_std"]) < 1e-3 and r["result"]["n"] == ref["n"] for r in rs)
    return out
