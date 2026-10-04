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
class District:
    name: str
    col_start: int
    col_end: int
    n_sources: int
    rate_mean_mbps: float
    rate_sigma: float
    on_alpha: float
    off_alpha: float
    target_mean_util: float


@dataclass(frozen=True)
class City:
    grid_rows: int
    grid_cols: int
    districts: list[District]


@dataclass(frozen=True)
class Event:
    district: str
    start_s: int
    duration_s: int
    ramp_s: int
    extra_source_factor: float


@dataclass(frozen=True)
class Sim:
    duration_s: int
    on_min_s: float
    off_min_s: float
    capacity_jitter_sigma: float
    diurnal_amplitude: float
    diurnal_period_s: int
    rsrp_base_dbm: float
    rsrp_util_slope_db: float
    rsrp_noise_db: float
    events: list[Event]


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
class Hurst:
    validate_cells: int
    min_scale_s: int
    max_scale_div: int


@dataclass(frozen=True)
class Scale:
    multipliers: list[int]
    base_cells: int
    base_duration_s: int
    window_s: int
    engines: list[str]
    timeout_s: int
    repeats: int


@dataclass(frozen=True)
class Radar:
    strategy: str
    variant: str
    lead_s: int
    window_s: int
    event_index: int


@dataclass(frozen=True)
class Report:
    group_label: str              # what a held-out group is called in tables and plots
    kind: str                     # "twin" | "real": selects the data section of the report
    event_label: str              # what the rare event is called ("storm" on the twin, "burst" on the real trace)
    peak_label: str               # what the regression target is called in tables


@dataclass(frozen=True)
class Real:
    """Real-trace study (msData, Open RAN 5G testbed). Nothing here applies to the digital twin."""
    url: str
    expected_bytes: int
    filename: str
    results_dir: str
    data_dir: str
    group_col: str                # column that plays the role of "district" (held out as a whole)
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
    hurst_min_stream_s: int       # only streams at least this long enter the Hurst comparison
    hurst_min_scale_s: int
    hurst_max_scale_div: int
    twin_few_sources: int         # sources per cell for the "testbed-sized" twin variant (the testbed has <= 4 UEs)
    storm_tail_labels: list[str]  # traffic labels listed in the descriptive "who is present at a burst" table


@dataclass(frozen=True)
class Outage:
    """Cell-outage impact study on the digital twin (known ground truth). Nothing here applies to the real trace."""
    n_worlds: int                 # independent simulated cities (different seeds)
    dev_worlds: int               # the first dev_worlds worlds are for choosing the method; results are reported on the rest
    world_seed_stride: int
    scenarios_per_world: int
    use_surge_events: bool        # scripted surges stay in the world: a realistic confounder for the estimator
    pair_fraction: float          # share of scenarios that take down two adjacent cells (site-level fault)
    no_coverage_fraction: float   # share of scenarios where nobody can reconnect (no neighbouring coverage)
    reconnect_prob_min: float     # per-scenario share of affected users that find a neighbouring cell
    reconnect_prob_max: float
    start_min_s: int              # earliest outage start (the estimator needs a pre-period)
    end_margin_s: int             # an outage ends at least this long before the series does
    duration_min_s: int
    duration_max_s: int
    reconnect_delay_min_s: int    # users are without service this long before they reselect a neighbour
    reconnect_delay_max_s: int
    return_delay_max_s: int       # users re-attach to the repaired cell up to this late
    routing_concentration: float  # Dirichlet concentration of the true neighbour-reselection weights
    handover_noise_sigma: float   # log-normal noise between true reselection weights and the handover statistics
    alarm_start_delay_max_s: int  # monitoring raises the alarm up to this late
    alarm_clear_delay_max_s: int
    degraded_served_fraction: float  # a second is degraded when served fraction (capacity / load) is below this
    absorber_min_share: float     # a neighbour is a "material absorber" when it takes at least this share of rerouted traffic
    pre_window_s: int
    guard_s: int                  # seconds before the alarm left out of the pre-period (alarm delay contamination)
    control_min_hops: int         # control cells are at least this many hops from every failed cell
    control_smooth_s: int
    level_block_s: int            # pre-period level = median of block means of this length (robust to a surge in the pre-period)
    alpha: float                  # significance level of the placebo-in-space absorber test
    prior_weights: list[float]    # weight on the handover prior when allocating traffic (0 = data only, 1 = prior only)
    noise_sweep: list[float]      # handover_noise_sigma values for the sensitivity table
    n_boot: int
    example_seed_index: int       # which scenario is drawn in the example figure


@dataclass(frozen=True)
class Config:
    seed: int
    paths: Paths
    city: City
    sim: Sim
    storm: Storm
    features: Features
    model: Model
    eval: Eval
    probabilistic: Probabilistic
    selection: Selection
    policy: Policy
    generalization: Generalization
    hurst: Hurst
    scale: Scale
    radar: Radar
    report: Report
    real: Real
    outage: Outage


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
    names = [d.name for d in cfg.city.districts]
    if len(set(names)) != len(names):
        raise ConfigError("city.districts: duplicate names")
    covered = set()
    for d in cfg.city.districts:
        if not (0 <= d.col_start < d.col_end <= cfg.city.grid_cols):
            raise ConfigError(f"district {d.name}: bad column range")
        cols = set(range(d.col_start, d.col_end))
        if covered & cols:
            raise ConfigError(f"district {d.name}: overlaps another district")
        covered |= cols
        if d.on_alpha <= 1 or d.off_alpha <= 1:
            raise ConfigError(f"district {d.name}: Pareto shapes must be > 1 (finite mean)")
        if not 0 < d.target_mean_util < 1:
            raise ConfigError(f"district {d.name}: target_mean_util must be in (0,1)")
    if covered != set(range(cfg.city.grid_cols)):
        raise ConfigError("city.districts must cover every grid column")
    for e in cfg.sim.events:
        if e.district not in names:
            raise ConfigError(f"event references unknown district {e.district!r}")
        if e.start_s + e.duration_s > cfg.sim.duration_s:
            raise ConfigError("event extends beyond sim.duration_s")
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
    if cfg.report.kind not in {"twin", "real"}:
        raise ConfigError("report.kind must be twin|real")
    r = cfg.real
    if r.bin_s <= 0 or r.max_gap_s < r.bin_s or r.min_stream_s < 2 * r.horizon_s or r.storm_mbps <= 0:
        raise ConfigError("real: inconsistent bin/gap/stream/horizon/storm settings")
    o = cfg.outage
    if o.n_worlds <= o.dev_worlds or o.dev_worlds < 0 or o.scenarios_per_world < 1:
        raise ConfigError("outage: need n_worlds > dev_worlds >= 0 and at least one scenario per world")
    if not (0 <= o.reconnect_prob_min <= o.reconnect_prob_max <= 1):
        raise ConfigError("outage: reconnect probabilities must satisfy 0 <= min <= max <= 1")
    if o.start_min_s < o.pre_window_s + o.guard_s + o.alarm_start_delay_max_s:
        raise ConfigError("outage.start_min_s must leave room for guard + pre_window")
    if o.start_min_s + o.duration_max_s + o.end_margin_s > cfg.sim.duration_s:
        raise ConfigError("outage: start_min_s + duration_max_s + end_margin_s exceeds sim.duration_s")
    if not (0 < o.alpha < 0.5) or not all(0 <= w <= 1 for w in o.prior_weights):
        raise ConfigError("outage: alpha must be in (0,0.5) and prior_weights in [0,1]")
    if cfg.radar.variant not in {"raw", "sigmoid", "isotonic"}:
        raise ConfigError("radar.variant must be raw|sigmoid|isotonic")


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
