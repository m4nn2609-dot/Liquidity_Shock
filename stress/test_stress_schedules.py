"""Cross-checks: stress/scenarios.py and execution/schedules.py must agree with liquidation.py."""
import numpy as np
import pandas as pd
import pytest

from execution import schedules as sc
from execution.liquidation import (build_positions, impact_cost_for_slices, liquidation_report,
                                   max_participation_slices)
from stress.scenarios import PRESETS, apply_stress, book_cost, compare, monte_carlo


def make_book():
    snap = pd.DataFrame({"ticker": ["A", "B", "C"], "date": pd.Timestamp("2026-10-23"),
                         "close": [100.0, 50.0, 200.0], "adv20": [1e6, 2e5, 5e6],
                         "vol20": [0.02, 0.03, 0.01], "spread20": [0.001, 0.004, 0.0005],
                         "amihud20": [1e-4, 1e-3, 2e-5]})
    port = pd.DataFrame({"ticker": ["A", "B", "C"], "shares": [5e5, 4e5, 1e5], "sector": list("XYZ")})
    pos = build_positions(port, snap)
    book = pos.rename(columns={"adv": "adv20", "vol": "vol20", "spread": "spread20"})[
        ["ticker", "shares", "price", "adv20", "vol20", "spread20"]]
    return pos, book


# ---------------- scenarios.py ----------------
def test_book_cost_matches_liquidation_report():
    pos, book = make_book()
    rep = liquidation_report(pos).set_index("ticker")
    bc = book_cost(book).set_index("ticker").reindex(rep.index)
    assert bc["total_cost"].to_numpy() == pytest.approx(rep["total_cost"].to_numpy())
    assert bc["days"].to_numpy() == pytest.approx(rep["trading_days"].to_numpy())


def test_stressed_book_cost_matches_liquidation_report():
    pos, book = make_book()
    st = apply_stress(book, 0.5, 3.0, 1.5)
    stressed_pos = pos.assign(adv=pos["adv"] * 0.5, spread=pos["spread"] * 3, vol=pos["vol"] * 1.5)
    rep = liquidation_report(stressed_pos).set_index("ticker")
    bc = book_cost(st).set_index("ticker").reindex(rep.index)
    assert bc["total_cost"].to_numpy() == pytest.approx(rep["total_cost"].to_numpy())


def test_normal_preset_changes_nothing_and_stress_costs_more():
    _, book = make_book()
    table, s = compare(book, **PRESETS["Normal"])
    assert s["cost_stress"] == pytest.approx(s["cost_normal"])
    _, s2 = compare(book, **PRESETS["Severe (PS preset)"])
    assert s2["cost_stress"] > s2["cost_normal"] and s2["longest_exit_stress"] >= s2["longest_exit_normal"]


def test_monte_carlo_is_reproducible_and_ordered():
    _, book = make_book()
    t1, m1 = monte_carlo(book, runs=500, seed=1)
    t2, m2 = monte_carlo(book, runs=500, seed=1)
    assert np.array_equal(t1, t2) and len(t1) == 500
    assert m1["p99"] >= m1["p95"] >= m1["median"] > m1["base_cost"] * 0.5


# ---------------- schedules.py ----------------
S, ADV, SIG, PRICE, SPR = 4.0e5, 2.0e5, 0.03, 50.0, 0.004


def test_schedule_impact_equals_liquidation_impact():
    q = max_participation_slices(S, ADV, 0.10)
    assert sc.impact_cost(q, ADV, SIG, PRICE) == pytest.approx(impact_cost_for_slices(q, ADV, SIG, PRICE))
    cost = sc.evaluate(sc.at_cap(S, ADV, 0.10), ADV, SIG, PRICE, SPR)["total_cost"]
    pos, _ = make_book()
    assert cost == pytest.approx(liquidation_report(pos).set_index("ticker").loc["B", "total_cost"])


def test_every_plan_sells_exactly_the_position_and_respects_cap():
    tbl, plans = sc.compare_schedules(S, ADV, SIG, PRICE, SPR, p=0.10)
    for name, q in plans.items():
        assert q.sum() == pytest.approx(S, rel=1e-6), name
        assert (q >= -1e-9).all(), name
    for name in ("At cap (fastest legal)", "TWAP", "Front-loaded", "Optimized"):
        assert plans[name].max() <= 0.10 * ADV * 1.001, name
    assert bool(tbl.loc["Immediate", "within_cap"]) is False      # selling it all at once breaks the cap


def test_immediate_costs_more_impact_than_at_cap():
    imm = sc.impact_cost(sc.immediate(S), ADV, SIG, PRICE)
    cap = sc.impact_cost(sc.at_cap(S, ADV, 0.10), ADV, SIG, PRICE)
    assert imm > cap


def test_optimizer_is_at_least_as_good_as_twap_and_front_loaded():
    tbl, _ = sc.compare_schedules(S, ADV, SIG, PRICE, SPR, p=0.10, lam=0.1)
    obj = tbl["objective_bps"].astype(float)
    assert obj["Optimized"] <= obj["TWAP"] + 1e-6 and obj["Optimized"] <= obj["Front-loaded"] + 1e-6


def test_optimizer_rejects_impossible_deadline():
    with pytest.raises(ValueError, match="Infeasible"):
        sc.optimize_schedule(S, ADV, SIG, PRICE, p=0.10, deadline=sc.min_days(S, ADV, 0.10) - 1)


def test_waiting_risk_known_value():
    # sell 50 then 50: held after day 1 = 50, after day 2 = 0 -> risk = sigma * price * 50
    assert sc.waiting_risk([50, 50], 0.02, 10.0) == pytest.approx(0.02 * 10.0 * 50)


def test_efficient_frontier_trades_cost_for_risk():
    fr = sc.efficient_frontier(S, ADV, SIG, PRICE, SPR, p=0.10)
    assert fr["risk_bps"].is_monotonic_decreasing or (fr["risk_bps"].diff().dropna() <= 1e-6).all()
    assert (fr["cost_bps"].diff().dropna() >= -1e-6).all()


def test_vwap_uses_the_volume_profile():
    q = sc.vwap(100, [1, 1, 2])
    assert q.tolist() == pytest.approx([25, 25, 50])
    tbl, plans = sc.compare_schedules(S, ADV, SIG, PRICE, SPR, p=0.10, volumes=[1, 2, 3, 4])
    assert not np.allclose(plans["VWAP"], plans["TWAP"])