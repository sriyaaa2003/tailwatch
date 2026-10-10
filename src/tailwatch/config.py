"""Typed, validated configuration. YAML is the single source of truth; there are no code-side defaults."""
from __future__ import annotations

import dataclasses
import os
import typing
from dataclasses import dataclass
from pathlib import Path

import yaml


class ConfigError(ValueError):
    pass


@dataclass(frozen=True)
class Paths:
    data_dir: str
    results_dir: str


@dataclass(frozen=True)
class Storm:
    utilisation_threshold: float
    horizon_s: int


@dataclass(frozen=True)
class Features:
    windows_s: list[int]
    stride_s: int


@dataclass(frozen=True)
class Model:
    max_iter: int
    learning_rate: float
    max_leaf_nodes: int
    min_samples_leaf: int
    smote_ratio: float
    undersample_ratio: float
    val_fraction: float
    purge_s: int
    strategies: list[str]
    default_strategy: str
    val_mode: str                 # "time": last val_fraction of time; "stream": every k-th stream (k = 1/val_fraction)


@dataclass(frozen=True)
class Eval:
    n_boot: int
    block_s: int
    ci_level: float
    reliability_bins: int


@dataclass(frozen=True)
class Probabilistic:
    alpha: float
    quantile_max_iter: int


@dataclass(frozen=True)
class Selection:
    ks: list[int]
    l1_c: float
    mi_subsample: int


@dataclass(frozen=True)
class Policy:
    cost_miss: float
    cost_false_alarm: float
    threshold_grid: int


@dataclass(frozen=True)
class Generalization:
    random_test_fraction: float
    temporal_test_fraction: float
    protocols: list[str]          # subset of: random, temporal, unseen, unseen_future


@dataclass(frozen=True)
class Report:
    group_label: str              # what a held-out group is called in tables and plots
    kind: str                     # "msdata" | "fiveg": selects the data section of the report
    event_label: str              # what the rare event is called
    peak_label: str               # what the regression target is called in tables


@dataclass(frozen=True)
class Real:
    """msData (Open RAN 5G testbed). Nothing here applies to the client-side 5G study."""
    url: str
    expected_bytes: int
    filename: str
    results_dir: str
    data_dir: str
    group_col: str                # column that is held out as a whole group
    timestamp_ticks_per_s: float  # timestamp unit; inferred from epoch magnitude, see README
    bin_s: float
    max_gap_s: float              # a longer silence in the cell's record stream starts a new stream
    min_stream_s: int
    rate_unit_bps: float          # mac_dl_brate unit (bit/s) -> load is reported in Mbit/s
    storm_mbps: float             # configured "high load" level, NOT a measured capacity
    horizon_s: int
    windows_s: list[int]
    stride_s: int
    n_boot: int
    block_s: int
    hurst_min_stream_s: int       # only streams at least this long enter the Hurst estimate
    hurst_min_scale_s: int
    hurst_max_scale_div: int
    storm_tail_labels: list[str]  # traffic labels listed in the descriptive "who is present at a burst" table


@dataclass(frozen=True)
class FiveG:
    """Client-side 5G production traces (Raca et al., MMSys 2020): 1 Hz KPIs logged on a phone in a car or on a desk."""
    url: str
    expected_bytes: int
    filename: str
    data_dir: str
    results_dir: str
    max_gap_s: int                # a longer jump in the timestamps splits a session into two streams
    min_stream_s: int
    cqi_now_min: int              # decision points are rows whose channel is currently at least this good
    cqi_event_max: int            # event: the worst CQI in the next horizon_s seconds is at most this
    horizon_s: int
    windows_s: list[int]
    stride_s: int
    n_boot: int
    block_s: int
    dl_kbps_per_mbps: float       # DL_bitrate is logged in kbit/s


@dataclass(frozen=True)
class Config:
    seed: int
    duration_s: int               # length of the time axis of the data set in use; set by the data set's derive step
    paths: Paths
    storm: Storm
    features: Features
    model: Model
    eval: Eval
    probabilistic: Probabilistic
    selection: Selection
    policy: Policy
    generalization: Generalization
    report: Report
    real: Real
    fiveg: FiveG


def _build(tp: typing.Any, data: typing.Any, path: str) -> typing.Any:
    origin = typing.get_origin(tp)
    if dataclasses.is_dataclass(tp):
        if not isinstance(data, dict):
            raise ConfigError(f"{path}: expected a mapping")
        hints = typing.get_type_hints(tp)
        unknown = set(data) - set(hints)
        if unknown:
            raise ConfigError(f"{path}: unknown keys {sorted(unknown)}")
        kwargs = {}
        for f in dataclasses.fields(tp):
            if f.name not in data:
                raise ConfigError(f"{path}.{f.name}: missing")
            kwargs[f.name] = _build(hints[f.name], data[f.name], f"{path}.{f.name}")
        return tp(**kwargs)
    if origin is list:
        (arg,) = typing.get_args(tp)
        if not isinstance(data, list):
            raise ConfigError(f"{path}: expected a list")
        return [_build(arg, x, f"{path}[{i}]") for i, x in enumerate(data)]
    if tp is float:
        if isinstance(data, bool) or not isinstance(data, (int, float)):
            raise ConfigError(f"{path}: expected a number")
        return float(data)
    if tp is int:
        if isinstance(data, bool) or not isinstance(data, int):
            raise ConfigError(f"{path}: expected an integer")
        return data
    if not isinstance(data, tp):
        raise ConfigError(f"{path}: expected {tp.__name__}")
    return data


def validate(cfg: Config) -> None:
    if not 0 < cfg.storm.utilisation_threshold <= 1.5:
        raise ConfigError("storm.utilisation_threshold out of range")
    if cfg.model.default_strategy not in cfg.model.strategies:
        raise ConfigError("model.default_strategy must be listed in model.strategies")
    unknown = set(cfg.model.strategies) - {"none", "class_weight", "smote", "undersample"}
    if unknown:
        raise ConfigError(f"unknown strategies {sorted(unknown)}")
    if not 0 < cfg.probabilistic.alpha < 1:
        raise ConfigError("probabilistic.alpha must be in (0,1)")
    if not 0 < cfg.model.val_fraction < 0.5:
        raise ConfigError("model.val_fraction must be in (0,0.5)")
    if cfg.policy.cost_miss <= 0 or cfg.policy.cost_false_alarm <= 0:
        raise ConfigError("policy costs must be positive")
    if cfg.model.val_mode not in {"time", "stream"}:
        raise ConfigError("model.val_mode must be time|stream")
    bad = set(cfg.generalization.protocols) - {"random", "temporal", "unseen", "unseen_future"}
    if bad:
        raise ConfigError(f"unknown generalization protocols {sorted(bad)}")
    if cfg.report.kind not in {"msdata", "fiveg"}:
        raise ConfigError("report.kind must be msdata|fiveg")
    r = cfg.real
    if r.bin_s <= 0 or r.max_gap_s < r.bin_s or r.min_stream_s < 2 * r.horizon_s or r.storm_mbps <= 0:
        raise ConfigError("real: inconsistent bin/gap/stream/horizon/storm settings")
    f = cfg.fiveg
    if f.min_stream_s < 2 * f.horizon_s or not 0 <= f.cqi_event_max < f.cqi_now_min <= 15:
        raise ConfigError("fiveg: need min_stream_s >= 2 x horizon_s and 0 <= cqi_event_max < cqi_now_min <= 15")


def load_config(path: str | os.PathLike[str] | None = None) -> Config:
    """Load from `path`, else $TAILWATCH_CONFIG, else configs/default.yaml next to the repo root."""
    chosen = path or os.environ.get("TAILWATCH_CONFIG")
    if chosen is None:
        chosen = Path(__file__).resolve().parents[2] / "configs" / "default.yaml"
    with open(chosen, encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    cfg = _build(Config, raw, "config")
    validate(cfg)
    return cfg
