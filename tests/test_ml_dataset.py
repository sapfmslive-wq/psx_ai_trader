"""Tests for src.models.dataset (Phase 15)."""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
import pytest

from src.models.dataset import build_dataset, split_chronological
from tests._ml_helpers import frame, make_config, random_walk

H = 5
N = 400


def _build(scale=1.0, with_indices=True, seed=1, **frame_kwargs):
    cfg = make_config()
    stock = frame(random_walk(N, seed), scale=scale, **frame_kwargs)
    indices = (
        {"KSE100": frame(random_walk(N, seed + 1, 45000.0, 30.0), scale=scale, volume_seed=3),
         "KMI30": frame(random_walk(N, seed + 2, 70000.0, 40.0), scale=scale, volume_seed=4)}
        if with_indices else {}
    )
    return build_dataset(stock, indices, cfg, H), stock, indices, cfg


def test_no_raw_price_level_features_are_used():
    ds, *_ = _build()
    forbidden = {"vwap", "obv", "macd_line", "macd_signal", "macd_histogram", "momentum_10"}
    for name in ds.feature_names:
        assert name not in forbidden, name
        assert not re.fullmatch(r"(sma|ema)_\d+", name), name
        assert not name.startswith(("bb_upper", "bb_lower", "bb_middle")), name


def test_features_are_scale_free_multiplying_every_price_by_ten_changes_nothing():
    base, *_ = _build(scale=1.0)
    scaled, *_ = _build(scale=10.0)
    assert base.feature_names == scaled.feature_names
    assert base.X.index.equals(scaled.X.index)
    np.testing.assert_allclose(base.X.to_numpy(), scaled.X.to_numpy(), rtol=1e-6, atol=1e-6)


def test_label_is_the_future_direction_and_never_a_feature():
    ds, stock, *_ = _build()
    close = stock["close"].to_numpy()
    for row in (0, 10, len(ds.y) // 2, len(ds.y) - 1):
        pos = int(ds.positions[row])
        expected = 1.0 if close[pos + H] > close[pos] else 0.0
        assert ds.y.iloc[row] == expected
    assert not any("label" in c or "future" in c for c in ds.feature_names)


def test_feature_matrix_has_no_lookahead():
    cfg = make_config()
    cutoff = 250
    stock = frame(random_walk(N, 1))
    idx_a = frame(random_walk(N, 2, 45000.0, 30.0), volume_seed=3)
    idx_b = frame(random_walk(N, 3, 70000.0, 40.0), volume_seed=4)
    before = build_dataset(stock, {"KSE100": idx_a, "KMI30": idx_b}, cfg, H)

    def mutate(df):
        out = df.copy()
        out.iloc[cutoff + 1 :, out.columns.get_indexer(["open", "high", "low", "close"])] *= 3.0
        return out

    after = build_dataset(mutate(stock), {"KSE100": mutate(idx_a), "KMI30": mutate(idx_b)}, cfg, H)

    cutoff_ts = stock.index[cutoff]
    shared = before.X.index.intersection(after.X.index)
    shared = shared[shared <= cutoff_ts]
    assert len(shared) > 100
    pd.testing.assert_frame_equal(before.X.loc[shared], after.X.loc[shared])


def test_all_nan_columns_are_dropped_not_fatal():
    ds, *_ = _build(with_volume=False)  # volume is NaN everywhere
    assert "relative_volume" in ds.dropped_all_nan_columns
    assert "relative_volume" not in ds.feature_names
    assert len(ds.X) > 100


def test_index_regime_features_appear_only_when_index_candles_exist():
    with_idx, *_ = _build(with_indices=True)
    without, *_ = _build(with_indices=False)
    assert any(c.startswith("KSE100_") for c in with_idx.feature_names)
    assert any(c.startswith("KMI30_") for c in with_idx.feature_names)
    assert any(c.startswith("relative_return_") for c in with_idx.feature_names)
    assert not any(c.startswith(("KSE100_", "KMI30_", "relative_return_")) for c in without.feature_names)


def test_warmup_and_unlabeled_tail_rows_are_dropped_and_counted():
    ds, *_ = _build()
    assert ds.n_dropped_feature_nan_rows > 0  # indicator warm-up
    assert ds.n_dropped_unlabeled_rows >= H  # last H bars have no future
    assert ds.X.notna().all().all() and ds.y.notna().all()
    assert len(ds.X) <= ds.n_input_bars - H


def test_split_is_chronological_non_overlapping_and_purged():
    positions = np.arange(500)
    s = split_chronological(positions, 0.6, 0.2, H)
    assert s.train.max() < s.validation.min() < s.validation.max() < s.test.min()
    assert positions[s.train].max() + H < positions[s.validation].min()
    assert positions[s.validation].max() + H < positions[s.test].min()
    assert s.purged_from_train == H and s.purged_from_validation == H
    assert len(s.test) == 500 - int(500 * 0.8)  # test is never trimmed
    assert set(s.train) & set(s.validation) == set() and set(s.validation) & set(s.test) == set()


def test_purge_uses_original_bar_positions_not_row_counts():
    # rows 40..59 were dropped upstream (NaN), so row i no longer equals bar i
    positions = np.array([p for p in range(120) if not 40 <= p < 60])
    s = split_chronological(positions, 0.5, 0.25, 5)
    assert positions[s.train].max() + 5 < positions[s.validation].min()
    assert positions[s.validation].max() + 5 < positions[s.test].min()


def test_non_positive_prices_are_refused_loudly_instead_of_inverting_labels():
    cfg = make_config()
    bad = frame(random_walk(N, 1))
    bad.iloc[100:120, bad.columns.get_indexer(["open", "high", "low", "close"])] = -5.0
    with pytest.raises(ValueError):
        build_dataset(bad, {}, cfg, H)
    good_stock = frame(random_walk(N, 1))
    with pytest.raises(ValueError):  # a bad INDEX frame is refused too
        build_dataset(good_stock, {"KSE100": bad}, cfg, H)


def test_split_refuses_tiny_or_degenerate_inputs():
    with pytest.raises(ValueError):
        split_chronological(np.arange(3), 0.6, 0.2, 5)
