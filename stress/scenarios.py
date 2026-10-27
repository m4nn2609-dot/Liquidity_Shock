"""stress/scenarios.py
Normal vs stressed liquidation cost: presets, slider scenario, Monte Carlo.

`book` is a DataFrame with ONE ROW PER HOLDING and these columns
(rename them in COLS if your names differ):
    ticker, shares, price, adv20 (shares/day), vol20 (DAILY sigma), spread20 (fraction)

Costs use the "sell p * ADV every day" schedule, same formulas as liquidation.py.
To use YOUR liquidation_report instead, pass cost_fn=your_function to compare().
cost_fn(book, p, Y) must return a DataFrame with columns: ticker, days, total_cost, cost_bps.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

COLS = dict(ticker="ticker", shares="shares", price="price",
            adv="adv20", vol="vol20", spread="spread20")

PRESETS = {
    "Normal":              dict(adv_mult=1.00, spread_mult=1.0, vol_mult=1.00),
    "Moderate":            dict(adv_mult=0.70, spread_mult=2.0, vol_mult=1.25),
    "Severe (PS preset)":  dict(adv_mult=0.50, spread_mult=3.0, vol_mult=1.50),
    "Extreme":             dict(adv_mult=0.25, spread_mult=5.0, vol_mult=2.00),
}


def apply_stress(book, adv_mult=0.5, spread_mult=3.0, vol_mult=1.5):
    s = book.copy()
    s[COLS["adv"]] = s[COLS["adv"]] * adv_mult
    s[COLS["spread"]] = s[COLS["spread"]] * spread_mult
    s[COLS["vol"]] = s[COLS["vol"]] * vol_mult
    return s


def _cost_arrays(S, P, adv, sig, spr, p, Y):
    """Closed form for 'sell the cap every day': n full days of cap + one remainder day."""
    cap = p * adv
    n = np.floor(S / cap)
    r = S - n * cap
    impact = Y * sig * P * (n * cap ** 1.5 + r ** 1.5) / np.sqrt(adv)
    spread = 0.5 * spr * S * P
    days = np.ceil(S / cap)
    return days, spread, impact


def book_cost(book, p=0.10, Y=1.0):
    S = book[COLS["shares"]].to_numpy(float)
    P = book[COLS["price"]].to_numpy(float)
    days, spr, imp = _cost_arrays(S, P, book[COLS["adv"]].to_numpy(float),
                                  book[COLS["vol"]].to_numpy(float),
                                  book[COLS["spread"]].to_numpy(float), p, Y)
    out = pd.DataFrame({"ticker": book[COLS["ticker"]].values, "days": days,
                        "spread_cost": spr, "impact_cost": imp})
    out["total_cost"] = out["spread_cost"] + out["impact_cost"]
    out["cost_bps"] = 1e4 * out["total_cost"] / (S * P)
    return out


def compare(book, adv_mult=0.5, spread_mult=3.0, vol_mult=1.5, p=0.10, Y=1.0, cost_fn=book_cost):
    """Per-stock normal vs stressed table + portfolio totals (for the dashboard cards)."""
    base = cost_fn(book, p, Y)
    st = cost_fn(apply_stress(book, adv_mult, spread_mult, vol_mult), p, Y)
    cols = ["ticker", "days", "total_cost", "cost_bps"]
    table = base[cols].merge(st[cols], on="ticker", suffixes=("_normal", "_stress"))
    table["cost_increase_pct"] = 100 * (table["total_cost_stress"] / table["total_cost_normal"] - 1)
    value = (book[COLS["shares"]] * book[COLS["price"]]).sum()
    summary = {
        "cost_normal": float(base["total_cost"].sum()),
        "cost_stress": float(st["total_cost"].sum()),
        "bps_normal": float(1e4 * base["total_cost"].sum() / value),
        "bps_stress": float(1e4 * st["total_cost"].sum() / value),
        "longest_exit_normal": int(base["days"].max()),
        "longest_exit_stress": int(st["days"].max()),
    }
    return table, summary


def monte_carlo(book, runs=2000, p=0.10, Y=1.0, rho=0.7, seed=42,
                adv_mu=np.log(0.7), adv_sd=0.30,
                spr_mu=np.log(2.0), spr_sd=0.40,
                vol_mu=np.log(1.3), vol_sd=0.20):
    """All holdings are stressed TOGETHER: each run draws one market-wide shock (weight rho)
    plus a small shock per stock. Volume falls while spread and volatility rise in the same run.
    Returns (array of portfolio costs, summary dict)."""
    rng = np.random.default_rng(seed)
    n = len(book)
    S = book[COLS["shares"]].to_numpy(float)
    P = book[COLS["price"]].to_numpy(float)

    zc = rng.standard_normal((runs, 1))  # shared market shock

    def shock(mu, sd, sign):
        zi = rng.standard_normal((runs, n))
        return np.exp(mu + sign * sd * (np.sqrt(rho) * zc + np.sqrt(1 - rho) * zi))

    adv = book[COLS["adv"]].to_numpy(float) * shock(adv_mu, adv_sd, -1)
    spr = book[COLS["spread"]].to_numpy(float) * shock(spr_mu, spr_sd, +1)
    vol = book[COLS["vol"]].to_numpy(float) * shock(vol_mu, vol_sd, +1)

    _, spread_c, impact_c = _cost_arrays(S, P, adv, vol, spr, p, Y)
    total = (spread_c + impact_c).sum(axis=1)
    value = (S * P).sum()
    base = book_cost(book, p, Y)["total_cost"].sum()
    summary = {
        "base_cost": float(base),
        "median": float(np.median(total)),
        "p95": float(np.percentile(total, 95)),
        "p99": float(np.percentile(total, 99)),
        "median_bps": float(1e4 * np.median(total) / value),
        "p95_bps": float(1e4 * np.percentile(total, 95) / value),
    }
    return total, summary