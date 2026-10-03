"""Parquet persistence. All artefacts live under cfg.paths.* (relative paths resolve against the repo root)."""
from __future__ import annotations

import json
from pathlib import Path

import polars as pl

from .config import Config

ROOT = Path(__file__).resolve().parents[2]


def _dir(p: str) -> Path:
    path = Path(p)
    path = path if path.is_absolute() else ROOT / path
    path.mkdir(parents=True, exist_ok=True)
    return path


def data_dir(cfg: Config) -> Path:
    return _dir(cfg.paths.data_dir)


def results_dir(cfg: Config) -> Path:
    return _dir(cfg.paths.results_dir)


def write_parquet(df: pl.DataFrame, path: Path) -> int:
    df.write_parquet(path, compression="zstd")
    return path.stat().st_size


def read_features(cfg: Config) -> pl.DataFrame:
    path = data_dir(cfg) / "features.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing: run `tailwatch simulate` first")
    return pl.read_parquet(path)


def save_json(cfg: Config, name: str, obj) -> Path:
    path = results_dir(cfg) / name
    path.write_text(json.dumps(obj, indent=2, default=float), encoding="utf-8")
    return path


def load_json(cfg: Config, name: str):
    path = results_dir(cfg) / name
    if not path.exists():
        raise FileNotFoundError(f"{path} missing: run the preceding pipeline step")
    return json.loads(path.read_text(encoding="utf-8"))
