import numpy as np
import pandas as pd
import pytest

from execution.liquidation import (
    BUCKET_LABELS, build_positions, days_to_liquidate, impact_cost_for_slices,
    liquidated_fraction_by_day, liquidation_report, load_portfolio,
    max_participation_slices, portfolio_summary, spread_cost,
)


def make_snapshot(tickers=("AAA.NS", "BBB.NS", "CCC.NS")):
    n = len(tickers)
    return pd.DataFrame({
        "ticker": list(tickers),
        "date": pd.Timestamp("2026-10-23"),
        "close": [100.0, 50.0, 200.0][:n],
        "adv20": [1_000_000.0, 200_000.0, 5_000_000.0][:n],
        "vol20": [0.02, 0.03, 0.01][:n],
        "spread20": [0.001, 0.004, 0.0005][:n],
        "amihud20": [0.0001, 0.001, 0.00002][:n],
    })


def make_portfolio(tickers=("AAA.NS", "BBB.NS", "CCC.NS")):
    n = len(tickers)
    return pd.DataFrame({"ticker": list(tickers), "shares": [500_000, 400_000, 100_000][:n],
                         "sector": ["X", "Y", "Z"][:n]})


def test_days_to_liquidate_known_value_and_series():
    assert days_to_liquidate(1000, 100, 0.10) == pytest.approx(100.0)
    s = days_to_liquidate(pd.Series([1000, 50]), pd.Series([100, 100]), 0.5)
    assert s.tolist() == pytest.approx([20.0, 1.0])


def test_halving_volume_doubles_days():
    assert days_to_liquidate(1000, 50, 0.1) == pytest.approx(2 * days_to_liquidate(1000, 100, 0.1))


def test_participation_must_be_between_zero_and_one():
    for bad in (0, -0.1, 1.5):
        with pytest.raises(ValueError):
            days_to_liquidate(1000, 100, bad)


def test_slices_add_up_and_respect_cap():
    s = max_participation_slices(1050, 100, 0.10)       # cap 10 per day
    assert s.sum() == pytest.approx(1050)
    assert len(s) == 105 and s.max() <= 10 + 1e-9
    s2 = max_participation_slices(25, 100, 0.10)        # last day is the remainder
    assert s2.tolist() == pytest.approx([10, 10, 5])


def test_slices_exact_multiple_has_no_tiny_extra_day():
    s = max_participation_slices(100, 100, 0.10)         # cap is 10 a day -> exactly 10 days
    assert len(s) == 10 and s.tolist() == pytest.approx([10.0] * 10)
    # float rounding (0.1 * 3 = 0.30000000000000004) must not create an extra day
    s2 = max_participation_slices(0.9, 3, 0.10)
    assert len(s2) == 3 and s2.sum() == pytest.approx(0.9)


def test_small_position_sells_in_one_day():
    s = max_participation_slices(5, 100, 0.10)
    assert s.tolist() == pytest.approx([5.0])


def test_slices_reject_bad_input():
    with pytest.raises(ValueError):
        max_participation_slices(0, 100)
    with pytest.raises(ValueError):
        max_participation_slices(100, float("nan"))


def test_spread_cost_known_value():
    assert spread_cost(1_000_000, 0.001) == pytest.approx(500.0)


def test_impact_known_value():
    # y * vol * sqrt(q/adv) * q * price = 1 * 0.02 * sqrt(0.1) * 10 * 50
    assert impact_cost_for_slices([10], adv=100, vol=0.02, price=50) == pytest.approx(
        0.02 * np.sqrt(0.1) * 10 * 50)


def test_impact_scales_with_y_and_volatility():
    base = impact_cost_for_slices([10], 100, 0.02, 50, y=1.0)
    assert impact_cost_for_slices([10], 100, 0.02, 50, y=2.0) == pytest.approx(2 * base)
    assert impact_cost_for_slices([10], 100, 0.04, 50, y=1.0) == pytest.approx(2 * base)


def test_selling_all_at_once_costs_more_than_spreading_it_out():
    all_at_once = impact_cost_for_slices([100], 100, 0.02, 50)
    spread_out = impact_cost_for_slices([10] * 10, 100, 0.02, 50)
    assert all_at_once > spread_out


def test_build_positions_and_value():
    pos = build_positions(make_portfolio(), make_snapshot())
    assert len(pos) == 3
    assert pos.loc[0, "position_value"] == pytest.approx(500_000 * 100.0)


def test_build_positions_raises_when_market_data_missing():
    with pytest.raises(ValueError, match="no market data"):
        build_positions(make_portfolio(("AAA.NS", "BBB.NS", "CCC.NS")), make_snapshot(("AAA.NS", "BBB.NS")))
    snap = make_snapshot()
    snap.loc[1, "adv20"] = np.nan
    with pytest.raises(ValueError, match="incomplete"):
        build_positions(make_portfolio(), snap)


def test_report_values_and_consistency():
    pos = build_positions(make_portfolio(), make_snapshot())
    rep = liquidation_report(pos, participation=0.10).set_index("ticker")
    assert rep.loc["AAA.NS", "days_to_liquidate"] == pytest.approx(5.0)     # 500k / (0.1 * 1M)
    assert rep.loc["BBB.NS", "days_to_liquidate"] == pytest.approx(20.0)    # 400k / (0.1 * 200k)
    assert rep.loc["AAA.NS", "trading_days"] == 5
    assert rep["total_cost"].tolist() == pytest.approx((rep["spread_cost"] + rep["impact_cost"]).tolist())
    expected_bps = 1e4 * rep["total_cost"] / rep["position_value"]
    assert rep["total_cost_bps"].tolist() == pytest.approx(expected_bps.tolist())
    assert rep.loc["AAA.NS", "spread_cost"] == pytest.approx(0.5 * 0.001 * 500_000 * 100.0)


def test_ranking_puts_hardest_to_sell_first():
    rep = liquidation_report(build_positions(make_portfolio(), make_snapshot()))
    assert rep["ticker"].iloc[0] == "BBB.NS"            # most days and highest Amihud
    assert rep["risk_rank"].tolist() == [1, 2, 3]
    assert rep["liquidity_risk_score"].is_monotonic_decreasing
    assert set(rep["bucket"].astype(str)) <= set(BUCKET_LABELS)


def test_stress_by_scaling_inputs_raises_days_and_cost():
    pos = build_positions(make_portfolio(), make_snapshot())
    normal = liquidation_report(pos)
    stressed_pos = pos.assign(adv=pos["adv"] * 0.5, spread=pos["spread"] * 3, vol=pos["vol"] * 1.5)
    stressed = liquidation_report(stressed_pos)
    assert (stressed.set_index("ticker")["days_to_liquidate"]
            > normal.set_index("ticker")["days_to_liquidate"]).all()
    assert (stressed.set_index("ticker")["total_cost"]
            > normal.set_index("ticker")["total_cost"]).all()


def test_more_selling_time_means_lower_impact():
    pos = build_positions(make_portfolio(), make_snapshot())
    fast = liquidation_report(pos, participation=0.50).set_index("ticker")["impact_cost"]
    slow = liquidation_report(pos, participation=0.05).set_index("ticker")["impact_cost"]
    assert (slow <= fast + 1e-9).all()
    # CCC is so small it sells in one day either way, so its cost does not change;
    # AAA and BBB need several days at the slow rate and get cheaper
    assert slow["AAA.NS"] < fast["AAA.NS"] and slow["BBB.NS"] < fast["BBB.NS"]
    assert slow["CCC.NS"] == pytest.approx(fast["CCC.NS"])


def test_portfolio_summary():
    rep = liquidation_report(build_positions(make_portfolio(), make_snapshot()))
    s = portfolio_summary(rep)
    assert s["longest_ticker"] == "BBB.NS" and s["longest_days"] == pytest.approx(20.0)
    assert s["total_cost_cr"] == pytest.approx(s["spread_cost_cr"] + s["impact_cost_cr"])
    assert s["total_value_cr"] == pytest.approx(rep["position_value"].sum() / 1e7)
    assert 0 < s["pct_value_over_10_days"] < 100


def test_liquidated_fraction_by_day():
    pos = build_positions(make_portfolio(), make_snapshot())
    prof = liquidated_fraction_by_day(pos, participation=0.10)
    assert prof["pct_value_liquidated"].is_monotonic_increasing
    assert prof["pct_value_liquidated"].iloc[-1] == pytest.approx(100.0)
    assert prof["day"].iloc[-1] == 20                   # BBB needs 20 days
    single = build_positions(make_portfolio(("AAA.NS",)), make_snapshot(("AAA.NS",)))
    p1 = liquidated_fraction_by_day(single, participation=0.10)   # 5 days
    assert p1.set_index("day").loc[2, "pct_value_liquidated"] == pytest.approx(40.0)


def test_load_portfolio_validates(tmp_path):
    good = tmp_path / "p.csv"
    good.write_text("ticker,shares,sector\nA.NS,100,X\nB.NS,200,Y\n")
    assert len(load_portfolio(good)) == 2
    dup = tmp_path / "d.csv"
    dup.write_text("ticker,shares,sector\nA.NS,100,X\nA.NS,200,Y\n")
    with pytest.raises(ValueError, match="duplicate"):
        load_portfolio(dup)
    neg = tmp_path / "n.csv"
    neg.write_text("ticker,shares,sector\nA.NS,-5,X\n")
    with pytest.raises(ValueError, match="shares"):
        load_portfolio(neg)