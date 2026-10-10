import numpy as np
import polars as pl
import pytest

from tailwatch.config import load_config
from tailwatch.core import feature_cols, lodo_folds
from tailwatch.real import RAW_COLS, build_real_features, cell_bins, derive_config, real_feature_names, streams

TICKS = 100_000.0     # timestamp ticks per second (cfg.real.timestamp_ticks_per_s)


def records(group, ue, seconds, brate, label="youtube", per_s=10, t0=1.6e9):
    """per_s records per second for each listed second; brate may be a scalar or a dict {second: bit/s}."""
    rows = []
    for s in seconds:
        b = brate[s] if isinstance(brate, dict) else brate
        for k in range(per_s):
            rows.append({"timestamp": (t0 + s + k / per_s) * TICKS, "mob_pattern": group, "label": label, "id_ue": ue,
                         "mac_dl_brate": float(b), "mac_dl_cqi": 9.0, "mac_dl_mcs": 15.0, "mac_dl_buffer": 100.0,
                         "phy_ul_pusch_sinr": 25.0})
    return pl.DataFrame(rows)


@pytest.fixture(scope="module")
def cfg():
    return load_config()


@pytest.fixture(scope="module")
def raw():
    hole = set(range(60, 62))                                        # 2 s silence: shorter than max_gap_s, must be filled
    ue1 = records("car", 1, [s for s in range(0, 300) if s not in hole], 2e6)
    ue2 = records("car", 2, range(100, 200), 3e6, label="iot")       # a second UE for part of the stream
    burst = records("car", 3, range(150, 156), 9e6, label="dos-hulk-C")   # pushes load to 14 Mbit/s for 6 s
    second = records("car", 1, range(310, 460), 1e6)                 # after a 10 s silence: a new stream
    short = records("car", 1, range(500, 550), 1e6)                  # 50 s: below min_stream_s, dropped
    other = records("bus", 1, range(0, 200), 4e6)
    return pl.concat([ue1, ue2, burst, second, short, other]).select(RAW_COLS)


def test_cell_load_is_sum_over_ues_and_units_are_mbit_per_s(cfg, raw):
    b = cell_bins(raw, cfg).filter(pl.col("group") == "car")
    load = dict(zip(b["bin"].to_list(), b["load"].to_list()))
    base = min(load)                                                  # bins are absolute; offsets are relative to it
    assert load[base + 50] == pytest.approx(2.0)                      # UE 1 only
    assert load[base + 120] == pytest.approx(5.0)                     # UE 1 + UE 2
    assert load[base + 152] == pytest.approx(14.0)                    # + burst UE
    assert (base + 60) not in load                                    # no record, no bin (filled later)


def test_streams_split_on_gaps_drop_short_ones_and_fill_holes(cfg, raw):
    s = streams(cell_bins(raw, cfg), cfg)
    assert s.filter(pl.col("group") == "car")["cell_id"].n_unique() == 2      # 0..299 and 310..459; 500..549 dropped
    assert s.filter(pl.col("group") == "bus")["cell_id"].n_unique() == 1
    car = s.filter(pl.col("group") == "car")
    first = car.filter(pl.col("cell_id") == car["cell_id"].min()).sort("t")
    assert first.height == 300                                                # the 2 s hole is filled, not split
    assert first["t"].diff().drop_nulls().unique().to_list() == [1]           # regular 1-bin grid
    t0 = first["t"][0]
    hole_rows = first.filter(pl.col("t").is_in([t0 + 60, t0 + 61]))
    assert hole_rows["load"].to_list() == [0.0, 0.0] and hole_rows["users"].to_list() == [0.0, 0.0]
    assert hole_rows["cqi"].null_count() == 0                                 # channel state carried forward


def test_real_labels_match_a_naive_forward_window_and_features_look_back_only(cfg, raw):
    dcfg = derive_config(cfg)
    s = streams(cell_bins(raw, cfg), cfg)
    feats = build_real_features(s, dcfg)
    H, thr, mbps = dcfg.storm.horizon_s, dcfg.storm.utilisation_threshold, cfg.real.storm_mbps
    car1 = s.filter(pl.col("cell_id") == s.filter(pl.col("group") == "car")["cell_id"].min()).sort("t")
    util = (car1["load"].to_numpy() / mbps)
    t_of = car1["t"].to_numpy()
    sub = feats.filter(pl.col("cell_id") == car1["cell_id"][0])
    assert sub.height > 0 and sub["y"].sum() > 0                              # the burst is detected somewhere
    idx = {t: i for i, t in enumerate(t_of)}
    for t, y in zip(sub["t"].to_list(), sub["y"].to_list()):
        i = idx[t]
        assert util[i] < thr                                                  # already-high rows are excluded
        assert y == int(util[i + 1: i + H + 1].max() >= thr)
    w = dcfg.features.windows_s[0]
    t_probe = sub["t"].to_list()[20]
    i = idx[t_probe]
    row = sub.filter(pl.col("t") == t_probe)
    assert row[f"util_mean_{w}"][0] == pytest.approx(util[i - w + 1: i + 1].mean(), rel=1e-4)


def test_real_feature_frame_has_no_metadata_in_model_inputs(cfg, raw):
    dcfg = derive_config(cfg)
    feats = build_real_features(streams(cell_bins(raw, cfg), cfg), dcfg)
    assert feature_cols(feats) == real_feature_names(dcfg)                    # label text and ids are never inputs
    assert feats.select(feature_cols(feats)).null_count().sum_horizontal()[0] == 0


def test_stream_validation_folds_hold_out_whole_groups_and_never_mix_streams(cfg, raw):
    dcfg = derive_config(cfg)
    feats = build_real_features(streams(cell_bins(raw, cfg), cfg), dcfg)
    folds = lodo_folds(feats, dcfg)
    assert {f.name for f in folds} == {"car", "bus"}
    cid = feats["cell_id"].to_numpy()
    grp = feats["group"].to_numpy()
    for f in folds:
        assert not np.any(grp[f.train] == f.name) and not np.any(grp[f.val] == f.name)
        assert not (set(cid[f.train]) & set(cid[f.val]))                      # a stream is entirely train or entirely val
        assert not np.any(f.train & f.val)


def test_derive_config_swaps_in_msdata_settings_without_touching_the_base_config(cfg):
    d = derive_config(cfg)
    assert d.paths.results_dir == cfg.real.results_dir and cfg.paths.results_dir != d.paths.results_dir
    assert d.model.val_mode == "stream" and d.model.purge_s == 0
    assert d.storm.horizon_s == cfg.real.horizon_s and cfg.storm.horizon_s != cfg.real.horizon_s
    assert d.report.kind == "msdata" and cfg.report.kind == "msdata"
    assert d.generalization.protocols == ["random", "unseen"]
