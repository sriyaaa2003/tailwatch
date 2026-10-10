import numpy as np
import polars as pl
import pytest

from tailwatch.config import load_config
from tailwatch.core import feature_cols, lodo_folds
from tailwatch.fiveg import build_features, derive_config, feature_names, parse_session

HEADER = "Timestamp,Longitude,Latitude,Speed,Operatorname,CellID,NetworkMode,RSRP,RSRQ,SNR,CQI,RSSI,DL_bitrate,UL_bitrate,State,PINGAVG,PINGMIN,PINGMAX,PINGSTDEV,PINGLOSS,CELLHEX,NODEHEX,LACHEX,RAWCELLID,NRxRSRP,NRxRSRQ"


def session_csv(cqi, cell=lambda i: 12, duplicate_every=0):
    """Fixture rows laid out exactly like the logger's CSV (a unit-test input, not study data)."""
    lines = [HEADER]
    for i, q in enumerate(cqi):
        sec = f"2019.11.28_07.{i // 60:02d}.{i % 60:02d}"
        row = f"{sec},-8.38,51.93,0,B,{cell(i)},5G,-100,-10,8.0,{q},-,5000,0,D,-,-,-,-,-,C,A,B,1,-100,-1"
        lines.append(row)
        if duplicate_every and i % duplicate_every == 0:
            lines.append(row)
    return "\n".join(lines).encode()


@pytest.fixture(scope="module")
def cfg():
    return load_config()


def streams_from(cqi_by_stream, cfg):
    frames = []
    for cid, (group, cqi) in enumerate(cqi_by_stream):
        d = parse_session(session_csv(cqi, duplicate_every=7))
        d = d.with_columns(pl.lit(group).alias("group"), pl.lit("netflix").alias("app"), pl.lit(cid).cast(pl.Int32).alias("cell_id"),
                           (pl.col("ts") - pl.col("ts").min()).cast(pl.Int32).alias("t"), (pl.col("DL_bitrate") / 1000.0).alias("dl"))
        frames.append(d.drop("ts", "DL_bitrate"))
    return pl.concat(frames)


def test_parse_session_merges_rows_written_in_the_same_second():
    d = parse_session(session_csv([10] * 30, duplicate_every=7))
    assert d.height == 30 and d["ts"].diff().drop_nulls().unique().to_list() == [1]
    assert d["CQI"].to_list() == [10.0] * 30 and d["nr"].to_list() == [1.0] * 30


def test_labels_match_a_naive_forward_window_and_features_look_back_only(cfg):
    rng = np.random.default_rng(0)
    cqi = np.clip(np.round(11 + np.cumsum(rng.normal(0, 1.2, 300)) * 0.4), 0, 15)
    s = streams_from([("driving / netflix", cqi)], cfg)
    dcfg = derive_config(cfg)
    feats = build_features(s, cfg)
    f, H = cfg.fiveg, cfg.fiveg.horizon_s
    sub = feats.sort("t")
    assert sub.height > 0
    for t, y, now in zip(sub["t"].to_list(), sub["y"].to_list(), sub["cqi_now"].to_list()):
        assert cqi[t] >= f.cqi_now_min and now == cqi[t]                       # only currently-good rows are scored
        assert y == int(cqi[t + 1: t + H + 1].min() <= f.cqi_event_max)       # strictly forward label
    w = dcfg.features.windows_s[0]
    t_probe = sub["t"].to_list()[10]
    assert sub.filter(pl.col("t") == t_probe)[f"cqi_mean_{w}"][0] == pytest.approx(cqi[t_probe - w + 1: t_probe + 1].mean(), rel=1e-4)


def test_feature_frame_has_no_metadata_in_model_inputs(cfg):
    rng = np.random.default_rng(1)
    cqi = np.clip(np.round(11 + rng.normal(0, 2.0, 200)), 0, 15)
    feats = build_features(streams_from([("static / prime", cqi)], cfg), cfg)
    assert feature_cols(feats) == feature_names(derive_config(cfg))
    assert feats.select(feature_cols(feats)).null_count().sum_horizontal()[0] == 0


def test_handovers_count_serving_cell_changes(cfg):
    cqi = [12] * 120
    d = parse_session(session_csv(cqi, cell=lambda i: 1 if i < 60 else 2))
    d = d.with_columns(pl.lit("g").alias("group"), pl.lit("netflix").alias("app"), pl.lit(0).cast(pl.Int32).alias("cell_id"),
                       (pl.col("ts") - pl.col("ts").min()).cast(pl.Int32).alias("t"), pl.col("DL_bitrate").alias("dl")).drop("ts", "DL_bitrate")
    feats = build_features(d, cfg)
    w = cfg.fiveg.windows_s[0]
    row = feats.filter(pl.col("t") == 62)
    assert row[f"handovers_{w}"][0] == 1.0 and feats.filter(pl.col("t") == 40)[f"handovers_{w}"][0] == 0.0


def test_stream_folds_hold_out_whole_groups(cfg):
    rng = np.random.default_rng(2)
    spec = [(g, np.clip(np.round(11 + rng.normal(0, 2.2, 160)), 0, 15)) for g in ["driving / netflix"] * 6 + ["static / prime"] * 6]
    feats = build_features(streams_from(spec, cfg), cfg)
    folds = lodo_folds(feats, derive_config(cfg))
    cid, grp = feats["cell_id"].to_numpy(), feats["group"].to_numpy()
    assert {f.name for f in folds} == {"driving / netflix", "static / prime"}
    for f in folds:
        assert not np.any(grp[f.train] == f.name) and not (set(cid[f.train]) & set(cid[f.val]))
