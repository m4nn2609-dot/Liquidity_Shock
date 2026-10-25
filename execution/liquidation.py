"""
liquidation.py  -  how long, and how costly, is it to sell the portfolio?

Pure formulas, no ML. Takes the portfolio (ticker, shares, sector) and the latest
feature snapshot from feature_engineering.features.latest_snapshot().

Main steps
    positions = build_positions(portfolio, snapshot)
    report    = liquidation_report(positions, participation=0.10)
    summary   = portfolio_summary(report)
    profile   = liquidated_fraction_by_day(positions, participation=0.10)

Formulas (all prices in rupees, spread as a fraction, e.g. 0.001 = 10 bps)
    cap per day      = participation * ADV                  (max shares sold a day)
    days to liquidate= shares / (participation * ADV)
    spread cost      = 0.5 * spread * position value        (cross half the spread)
    impact cost/day  = Y * vol * sqrt(q / ADV) * q * price  (square-root impact,
                       q = shares sold that day, vol = daily volatility)
    total cost       = spread cost + impact cost

Selling plan used here: sell the maximum allowed every day (participation * ADV)
until the position is gone. Other plans (TWAP, VWAP, all at once) can reuse
`impact_cost_for_slices` with their own daily slices.

To simulate a crisis, scale the `adv`, `spread` and `vol` columns of `positions`
before calling liquidation_report (this is what the stress module does).

Limits to state in the report: spread comes from a High/Low estimate with a 1 bp
floor, impact uses a simple square-root model with a fixed Y, ADV is the last
20-day average, and no intraday data or circuit limits are modelled.

Run directly for a quick look (from the project root):
    python execution/liquidation.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_PARTICIPATION = 0.10   # sell at most 10% of daily volume
DEFAULT_IMPACT_Y = 1.0         # square-root impact coefficient (typical range 0.5 - 1.5)
CRORE = 1e7

BUCKET_EDGES = [0, 3, 12, 40, np.inf]
BUCKET_LABELS = ["Easy (<3d)", "Medium (3-12d)", "Hard (12-40d)", "Very hard (>40d)"]

POSITION_COLUMNS = ["ticker", "sector", "as_of", "shares", "price", "position_value",
                    "adv", "vol", "spread", "amihud"]


# --------------------------------------------------------------------------
# 1. Loading and building positions
# --------------------------------------------------------------------------
def load_portfolio(path="data/portfolio.csv") -> pd.DataFrame:
    """Read portfolio.csv (ticker, shares, sector) and validate it."""
    df = pd.read_csv(path)
    missing = [c for c in ("ticker", "shares", "sector") if c not in df.columns]
    if missing:
        raise ValueError(f"portfolio is missing columns: {missing}")
    df["ticker"] = df["ticker"].astype(str).str.strip()
    if df["ticker"].duplicated().any():
        raise ValueError("portfolio has duplicate tickers")
    if not (df["shares"] > 0).all():
        raise ValueError("every position must have shares > 0")
    return df[["ticker", "shares", "sector"]].reset_index(drop=True)


def build_positions(portfolio: pd.DataFrame, snapshot: pd.DataFrame) -> pd.DataFrame:
    """Join holdings with the latest market numbers (one row per holding).

    snapshot: output of latest_snapshot(), needs ticker, date, close, adv20,
    vol20, spread20 and amihud20. Raises an error rather than silently dropping
    a holding that has no market data.
    """
    need = ["ticker", "date", "close", "adv20", "vol20", "spread20", "amihud20"]
    absent = [c for c in need if c not in snapshot.columns]
    if absent:
        raise ValueError(f"snapshot is missing columns: {absent}")

    no_data = sorted(set(portfolio["ticker"]) - set(snapshot["ticker"]))
    if no_data:
        raise ValueError(f"no market data for: {no_data}")

    snap = snapshot[need].rename(columns={
        "date": "as_of", "close": "price", "adv20": "adv",
        "vol20": "vol", "spread20": "spread", "amihud20": "amihud"})
    pos = portfolio.merge(snap, on="ticker", how="left", validate="one_to_one")

    bad = pos[["price", "adv", "vol", "spread", "amihud"]].isna().any(axis=1)
    if bad.any():
        raise ValueError(f"incomplete market data for: {pos.loc[bad, 'ticker'].tolist()}")

    pos["position_value"] = pos["shares"] * pos["price"]
    return pos[POSITION_COLUMNS].reset_index(drop=True)


# --------------------------------------------------------------------------
# 2. Formulas
# --------------------------------------------------------------------------
def _check_participation(participation: float) -> None:
    if not (0 < participation <= 1):
        raise ValueError("participation must be greater than 0 and at most 1")


def days_to_liquidate(shares, adv, participation=DEFAULT_PARTICIPATION):
    """shares / (participation * ADV). Works on numbers or pandas Series."""
    _check_participation(participation)
    return shares / (participation * adv)


def spread_cost(position_value, spread):
    """Cost of crossing half the bid-ask spread on the whole position (rupees)."""
    return 0.5 * spread * position_value


def max_participation_slices(shares, adv, participation=DEFAULT_PARTICIPATION) -> np.ndarray:
    """Shares sold each day when selling the maximum allowed until done.

    Every day sells participation * ADV, except the last day, which sells
    what is left. The slices always add up to `shares`.
    """
    _check_participation(participation)
    if not (np.isfinite(shares) and np.isfinite(adv) and shares > 0 and adv > 0):
        raise ValueError("shares and adv must be positive numbers")
    cap = participation * adv
    n_days = max(int(np.ceil(shares / cap - 1e-9)), 1)
    slices = np.full(n_days, cap, dtype=float)
    slices[-1] = shares - cap * (n_days - 1)
    return slices


def impact_cost_for_slices(slices, adv, vol, price, y=DEFAULT_IMPACT_Y) -> float:
    """Square-root market-impact cost (rupees) of selling `slices` shares per day.

    On each day: price concession = y * vol * sqrt(q / adv), paid on the q shares
    sold that day. Concentrating the selling in fewer days costs more, because
    the concession grows faster than the amount sold.
    """
    q = np.asarray(slices, dtype=float)
    if (q < 0).any():
        raise ValueError("slices cannot be negative")
    return float(np.sum(y * vol * np.sqrt(q / adv) * q * price))


# --------------------------------------------------------------------------
# 3. Report, ranking and summary
# --------------------------------------------------------------------------
def liquidation_report(positions: pd.DataFrame,
                       participation: float = DEFAULT_PARTICIPATION,
                       impact_y: float = DEFAULT_IMPACT_Y) -> pd.DataFrame:
    """One row per holding with days, costs and a liquidity-risk ranking.

    liquidity_risk_score (0-100) is the average of the percentile rank of
    days_to_liquidate and of Amihud illiquidity. risk_rank 1 = hardest to sell.
    """
    _check_participation(participation)
    rows = []
    for r in positions.to_dict("records"):
        slices = max_participation_slices(r["shares"], r["adv"], participation)
        sp_cost = float(spread_cost(r["position_value"], r["spread"]))
        im_cost = impact_cost_for_slices(slices, r["adv"], r["vol"], r["price"], impact_y)
        total = sp_cost + im_cost
        rows.append({
            "ticker": r["ticker"], "sector": r["sector"], "as_of": r["as_of"],
            "shares": r["shares"], "price": r["price"],
            "position_value": r["position_value"],
            "position_cr": r["position_value"] / CRORE,
            "adv": r["adv"],
            "position_pct_adv": 100.0 * r["shares"] / r["adv"],
            "days_to_liquidate": float(days_to_liquidate(r["shares"], r["adv"], participation)),
            "trading_days": len(slices),
            "spread": r["spread"], "spread_cost": sp_cost, "impact_cost": im_cost,
            "total_cost": total,
            "total_cost_bps": 1e4 * total / r["position_value"],
            "vol": r["vol"], "amihud": r["amihud"],
        })
    rep = pd.DataFrame(rows)

    rep["liquidity_risk_score"] = 100.0 * (
        rep["days_to_liquidate"].rank(pct=True) + rep["amihud"].rank(pct=True)) / 2.0
    rep["risk_rank"] = rep["liquidity_risk_score"].rank(ascending=False, method="first").astype(int)
    rep["bucket"] = pd.cut(rep["days_to_liquidate"], bins=BUCKET_EDGES,
                           labels=BUCKET_LABELS, right=False)
    return rep.sort_values("risk_rank").reset_index(drop=True)


def portfolio_summary(report: pd.DataFrame) -> dict:
    """Headline numbers for the whole portfolio."""
    value = report["position_value"].sum()
    longest = report.loc[report["days_to_liquidate"].idxmax()]
    over_10 = report.loc[report["days_to_liquidate"] > 10, "position_value"].sum()
    return {
        "total_value_cr": value / CRORE,
        "spread_cost_cr": report["spread_cost"].sum() / CRORE,
        "impact_cost_cr": report["impact_cost"].sum() / CRORE,
        "total_cost_cr": report["total_cost"].sum() / CRORE,
        "total_cost_bps": 1e4 * report["total_cost"].sum() / value,
        "longest_days": float(longest["days_to_liquidate"]),
        "longest_ticker": longest["ticker"],
        "value_weighted_days": float((report["days_to_liquidate"] * report["position_value"]).sum() / value),
        "pct_value_over_10_days": 100.0 * over_10 / value,
    }


def liquidated_fraction_by_day(positions: pd.DataFrame,
                               participation: float = DEFAULT_PARTICIPATION,
                               max_days=None) -> pd.DataFrame:
    """% of the portfolio's VALUE that can be sold by the end of each day.

    Each position sells participation * ADV per day until gone. Gives the
    "liquidity profile" curve: e.g. 60% of value is sellable within 5 days.
    """
    _check_participation(participation)
    days_needed = days_to_liquidate(positions["shares"], positions["adv"], participation)
    if max_days is None:
        max_days = max(int(np.ceil(days_needed.max() - 1e-9)), 1)
    value = positions["position_value"].to_numpy()
    days = np.arange(1, max_days + 1)
    frac = np.minimum(1.0, days[:, None] / days_needed.to_numpy()[None, :])
    return pd.DataFrame({"day": days, "pct_value_liquidated": 100.0 * (frac * value).sum(axis=1) / value.sum()})


# --------------------------------------------------------------------------
# 4. Run as a script
# --------------------------------------------------------------------------
if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    from feature_engineering.features import build_features, latest_snapshot, load_prices

    snapshot = latest_snapshot(build_features(load_prices(root / "data" / "prices.csv")))
    positions = build_positions(load_portfolio(root / "data" / "portfolio.csv"), snapshot)
    report = liquidation_report(positions)

    show = report[["risk_rank", "ticker", "position_cr", "days_to_liquidate",
                   "total_cost_bps", "bucket"]].copy()
    print(show.round(1).to_string(index=False))
    print("\nPortfolio summary:")
    for k, v in portfolio_summary(report).items():
        print(f"  {k}: {v:,.2f}" if isinstance(v, float) else f"  {k}: {v}")