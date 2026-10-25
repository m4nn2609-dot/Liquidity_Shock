import numpy as np
import pandas as pd
import pytest

from feature_engineering.features import (
    FEATURE_COLUMNS, MARKET_COLUMNS, MIN_SPREAD, add_market_features,
    average_true_range, build_features, build_index_features, clean_prices,
    corwin_schultz_spread, latest_snapshot, rolling_zscore,
)


def make_prices(n=60, ticker="AAA.NS", seed=0, volume=100.0):
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    return pd.DataFrame({
        "date": pd.bdate_range("2024-01-01", periods=n),
        "ticker": ticker,
        "open": close,
        "high": close * 1.01,
        "low": close * 0.99,
        "close": close,
        "adj_close": close,
        "volume": volume,
    })


def test_columns_and_no_infinities():
    f = build_features(make_prices())
    assert list(f.columns) == FEATURE_COLUMNS
    assert not np.isinf(f.select_dtypes("number")).any().any()


def test_adv_and_volume_ratio_known_values():
    f = build_features(make_prices(volume=100.0))
    assert f["adv20"].iloc[:19].isna().all()
    assert f["adv20"].iloc[19] == pytest.approx(100.0)
    assert f["volume_ratio"].iloc[:20].isna().all()
    assert f["volume_ratio"].iloc[20] == pytest.approx(1.0)


def test_volume_ratio_uses_previous_days_only():
    p = make_prices(volume=100.0)
    p.loc[30, "volume"] = 500.0                      # one spike on row 30
    f = build_features(p)
    assert f["volume_ratio"].iloc[30] == pytest.approx(5.0)


def test_no_lookahead():
    p = make_prices(n=400)
    p["volume"] = np.random.default_rng(5).uniform(50, 150, len(p))   # varying volume
    base = build_features(p)
    p2 = p.copy()
    p2.loc[350:, ["high", "low", "close", "adj_close", "open"]] *= 3.0   # change the future
    p2.loc[350:, "volume"] *= 7.0
    changed = build_features(p2)
    cols = [c for c in FEATURE_COLUMNS if c not in ("date", "ticker")]
    assert base[cols].iloc[300].notna().all()        # z-scores really are filled by then
    pd.testing.assert_frame_equal(base.loc[:349, cols], changed.loc[:349, cols])


def test_constant_price_gives_zero_range_and_zero_vol():
    p = make_prices()
    for c in ("open", "high", "low", "close", "adj_close"):
        p[c] = 100.0
    f = build_features(p)
    assert (f["hl_range"] == 0).all()
    assert f["vol20"].dropna().eq(0).all()
    assert f["spread20"].dropna().eq(MIN_SPREAD).all()


def test_clean_prices_removes_bad_rows():
    p = make_prices(n=40)
    p.loc[3, "volume"] = 0                           # holiday
    p.loc[5, "high"] = p.loc[5, "low"] * 0.9         # High < Low
    p.loc[7, "close"] = np.nan                       # missing
    p.loc[35, "volume"] = 0.5                        # glitch: far below 1% of median
    c = clean_prices(p)
    assert len(c) == 40 - 4
    assert c["date"].is_monotonic_increasing


def test_each_ticker_is_independent():
    a = make_prices(ticker="AAA.NS", seed=1)
    b = make_prices(ticker="BBB.NS", seed=2, volume=500.0)
    f = build_features(pd.concat([a, b], ignore_index=True))
    assert f.groupby("ticker")["adv20"].last().to_dict() == pytest.approx(
        {"AAA.NS": 100.0, "BBB.NS": 500.0})
    only_a = build_features(a)
    pd.testing.assert_frame_equal(
        f[f["ticker"] == "AAA.NS"].reset_index(drop=True), only_a)


def test_latest_snapshot_one_row_per_ticker_and_as_of():
    f = build_features(pd.concat([make_prices(ticker="AAA.NS"),
                                  make_prices(ticker="BBB.NS", seed=3)], ignore_index=True))
    snap = latest_snapshot(f)
    assert sorted(snap["ticker"]) == ["AAA.NS", "BBB.NS"]
    assert snap["date"].eq(f["date"].max()).all()
    early = latest_snapshot(f, as_of=f["date"].iloc[30])
    assert (early["date"] <= f["date"].iloc[30]).all()


def test_corwin_schultz_recovers_known_spread():
    """Simulate trades bouncing between bid and ask; the estimate should be close."""
    rng = np.random.default_rng(0)
    days, steps, true_spread, daily_vol = 3000, 200, 0.005, 0.01
    eff = np.cumsum(rng.normal(0, daily_vol / np.sqrt(steps), days * steps))
    side = rng.choice([-1, 1], days * steps)
    trades = (np.exp(eff) * (1 + side * true_spread / 2)).reshape(days, steps)
    high, low, close = (pd.Series(trades.max(1)), pd.Series(trades.min(1)),
                        pd.Series(trades[:, -1]))
    est = corwin_schultz_spread(high, low, close).mean()
    assert est == pytest.approx(true_spread, rel=0.15)


def test_corwin_schultz_first_day_is_nan():
    p = make_prices()
    s = corwin_schultz_spread(p["high"], p["low"], p["close"])
    assert np.isnan(s.iloc[0]) and s.iloc[1:].notna().all()


def test_atr_matches_hand_calculation():
    high = pd.Series([10.0, 12.0, 11.0, 13.0])
    low = pd.Series([8.0, 9.0, 9.0, 10.0])
    close = pd.Series([9.0, 11.0, 10.0, 12.0])
    atr = average_true_range(high, low, close, window=3)
    # true ranges: 2, 3, 2, 3  ->  atr[2] = 7/3, atr[3] = (7/3 * 2 + 3) / 3
    assert atr.iloc[:2].isna().all()
    assert atr.iloc[2] == pytest.approx(7 / 3)
    assert atr.iloc[3] == pytest.approx((7 / 3 * 2 + 3) / 3)


def test_zscore_known_value_and_constant_series():
    s = pd.Series(np.arange(1.0, 301.0))
    z = rolling_zscore(s, window=252, min_periods=126)
    last = s.iloc[-252:]
    assert z.iloc[-1] == pytest.approx((s.iloc[-1] - last.mean()) / last.std())
    assert z.iloc[:125].isna().all()
    assert rolling_zscore(pd.Series([5.0] * 300)).isna().all()      # std 0 -> NaN, not inf


def test_flat_price_gives_zero_trend_return_drawdown_atr():
    p = make_prices(n=300)
    for c in ("open", "high", "low", "close", "adj_close"):
        p[c] = 100.0
    f = build_features(p).iloc[-1]
    assert f["trend_50"] == 0 and f["ret_20d"] == 0
    assert f["drawdown252"] == 0 and f["atr_pct"] == 0


def test_rising_price_has_no_drawdown_and_positive_trend():
    p = make_prices(n=300)
    path = np.linspace(100, 200, 300)
    for c in ("open", "high", "low", "close", "adj_close"):
        p[c] = path
    f = build_features(p).iloc[-1]
    assert f["drawdown252"] == pytest.approx(0.0)
    assert f["trend_50"] > 0 and f["ret_20d"] > 0


def make_index(n=80, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "date": pd.bdate_range("2024-01-01", periods=n),
        "nifty": 10000 * np.exp(np.cumsum(rng.normal(0, 0.01, n))),
        "india_vix": 15.0,
    })


def test_index_features_basic_and_no_lookahead():
    idx = make_index()
    m = build_index_features(idx)
    assert list(m.columns) == ["date"] + MARKET_COLUMNS
    assert np.isnan(m["idx_return"].iloc[0])
    assert (m["idx_drawdown"] <= 0).all()
    idx2 = idx.copy()
    idx2.loc[60:, "nifty"] *= 0.5                       # crash in the future
    m2 = build_index_features(idx2)
    pd.testing.assert_frame_equal(m.loc[:59], m2.loc[:59])


def test_index_features_skip_missing_nifty_rows():
    idx = make_index()
    idx.loc[10, "nifty"] = np.nan
    m = build_index_features(idx)
    assert len(m) == len(idx) - 1 and m["idx_return"].iloc[1:].notna().all()


def test_add_market_features_keeps_every_stock_row():
    p = pd.concat([make_prices(ticker="AAA.NS"), make_prices(ticker="BBB.NS", seed=3)],
                  ignore_index=True)
    f = build_features(p)
    out = add_market_features(f, make_index(n=60))
    assert len(out) == len(f)
    assert set(MARKET_COLUMNS).issubset(out.columns)