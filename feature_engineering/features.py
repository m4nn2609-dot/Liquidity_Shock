"""
features.py  -  ALL features for LiquidityShock, in one place (formulas only, no ML).

Combines the liquidity features (used by execution and stress) with the extra
clues the ML classifier needs (momentum, ATR, z-scores, drawdown, index features).

Input : prices table in long format with columns
        date, ticker, open, high, low, close, adj_close, volume
Output: one row per (ticker, date) with the columns in FEATURE_COLUMNS.

Column guide
  Liquidity (use these for execution / stress)
    adv20         average daily volume, last 20 days (shares)
    adv_value20   average daily traded value, last 20 days (rupees)
    spread20      estimated relative spread (0.001 = 10 bps), floored at 1 bp
    vol20         daily volatility of returns, last 20 days
    vol20_ann     the same, annualised
    amihud20      price move per rupee-crore traded (higher = less liquid)
  Clues for ML (do NOT use for cost calculations)
    volume_ratio  today's volume / average of the previous 20 days
    hl_range      today's (high - low) / close
    atr_pct       14-day average true range / close
    cs_clipped20  spread estimate averaged AFTER clipping at 0: inflated, but
                  it ranks stocks and days sensibly, so it is a useful clue
    ret_20d       20-day return
    trend_50      price / 50-day average - 1
    drawdown252   price / highest price of the last 252 days - 1
    *_z           z-scores of volume_ratio, hl_range, vol20 vs the last 252 days
  Market (optional, added with add_market_features)
    idx_return, idx_vol20, idx_drawdown   from the Nifty index

Known limitation: there is no real bid-ask data, so the spread is estimated
from daily High/Low (Corwin-Schultz). On liquid stocks the true spread (a few
bps) is far smaller than daily price noise, so `spread20` is often just the
1 bp floor there. Treat it as a rough guide and say so in the report.

No look-ahead: every rolling statistic on day t uses data up to and including
day t only. The one exception to "includes day t" is `volume_ratio`, which
compares day t's volume with the average of the PREVIOUS 20 days.

Usage from other files (run from the project root):
    from feature_engineering.features import (
        load_prices, load_index, build_features, add_market_features, latest_snapshot)
    prices   = load_prices("data/prices.csv")
    features = build_features(prices)
    today    = latest_snapshot(features)

Run directly to build and save data/features.csv:
    python feature_engineering/features.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

WINDOW = 20           # trading days used for rolling averages
ATR_WINDOW = 14
TREND_WINDOW = 50
RETURN_WINDOW = 20    # for ret_20d
Z_WINDOW = 252        # one trading year, for z-scores and drawdown
Z_MIN_PERIODS = 126   # z-scores start after half a year of history
TRADING_DAYS = 252    # for annualising volatility
CRORE = 1e7           # 1 crore = 10 million rupees
MIN_SPREAD = 0.00002   # 0.2 bp floor (reduced from 1bp to better reflect ultra-liquid NSE stocks)
MIN_VOLUME_FRACTION = 0.01   # drop days trading < 1% of the trailing 60-day median

REQUIRED_COLUMNS = ["date", "ticker", "high", "low", "close", "adj_close", "volume"]

FEATURE_COLUMNS = [
    "date", "ticker", "close", "volume",
    # liquidity
    "ret", "hl_range", "traded_value", "adv20", "adv_value20", "volume_ratio",
    "vol20", "vol20_ann", "amihud20", "spread_raw", "spread20",
    # extra clues for ML
    "atr_pct", "cs_clipped20", "ret_20d", "trend_50", "drawdown252",
    "volume_ratio_z", "hl_range_z", "vol20_z",
]

MARKET_COLUMNS = ["idx_return", "idx_vol20", "idx_drawdown"]


# --------------------------------------------------------------------------
# 1. Loading and cleaning
# --------------------------------------------------------------------------
def clean_prices(prices: pd.DataFrame) -> pd.DataFrame:
    """Drop unusable rows and sort by ticker and date.

    Removed: missing values, non-positive prices, zero-volume days
    (market holidays), rows where High < Low, duplicate (ticker, date), and
    obvious volume glitches (a day trading under 1% of that stock's trailing
    60-day median volume, e.g. a few thousand shares for a stock that normally
    trades hundreds of millions). The glitch test looks backwards only.
    """
    missing = [c for c in REQUIRED_COLUMNS if c not in prices.columns]
    if missing:
        raise ValueError(f"prices is missing columns: {missing}")

    df = prices.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.dropna(subset=["close", "adj_close", "high", "low", "volume"])
    df = df[(df["close"] > 0) & (df["adj_close"] > 0) & (df["low"] > 0)]
    df = df[df["volume"] > 0]
    df = df[df["high"] >= df["low"]]
    df = df.drop_duplicates(subset=["ticker", "date"], keep="last")
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)

    trailing_median = df.groupby("ticker")["volume"].transform(
        lambda v: v.rolling(60, min_periods=20).median())
    glitch = df["volume"] < MIN_VOLUME_FRACTION * trailing_median   # NaN -> False
    return df[~glitch].reset_index(drop=True)


def load_prices(path="data/prices.csv") -> pd.DataFrame:
    """Read prices.csv and return the cleaned table."""
    return clean_prices(pd.read_csv(path, parse_dates=["date"]))


def load_index(path="data/market_index.csv") -> pd.DataFrame:
    """Read market_index.csv (columns: date, nifty, india_vix)."""
    return pd.read_csv(path, parse_dates=["date"])


# --------------------------------------------------------------------------
# 2. Individual formulas
# --------------------------------------------------------------------------
def corwin_schultz_spread(high: pd.Series, low: pd.Series,
                          close: pd.Series = None) -> pd.Series:
    """Daily Corwin-Schultz (2012) bid-ask spread estimate from High and Low.

    Returns the RELATIVE spread as a fraction of price (0.001 = 0.1% = 10 bps).
    Uses two consecutive days, so the first day of a series is NaN.

    If `close` is given, the overnight-gap adjustment from the paper is applied:
    a gap between yesterday's close and today's range is removed so that it is
    not mistaken for spread.

    IMPORTANT: the daily estimate is NOISY and is often negative (about 45% of
    days on this data). It is returned UNCLIPPED on purpose: clipping each day
    at zero before averaging inflates the spread by several times. Average first
    (see `spread20` in build_features), then floor the average.
    Pass one ticker's data at a time, sorted by date.
    """
    k = 3.0 - 2.0 * np.sqrt(2.0)
    h, l = high.copy(), low.copy()

    if close is not None:
        prev_close = close.shift(1)
        up = (l - prev_close).clip(lower=0.0).fillna(0.0)    # range gapped up
        down = (prev_close - h).clip(lower=0.0).fillna(0.0)  # range gapped down
        h = h - up + down
        l = l - up + down

    hl = np.log(h / l) ** 2                            # (ln H/L)^2 for each day
    beta = hl + hl.shift(1)                            # today + yesterday
    two_day_high = pd.concat([h, h.shift(1)], axis=1).max(axis=1)
    two_day_low = pd.concat([l, l.shift(1)], axis=1).min(axis=1)
    gamma = np.log(two_day_high / two_day_low) ** 2

    alpha = (np.sqrt(2.0 * beta) - np.sqrt(beta)) / k - np.sqrt(gamma / k)
    spread = 2.0 * (np.exp(alpha) - 1.0) / (1.0 + np.exp(alpha))
    return spread.where(beta.notna())                  # first day stays NaN


def average_true_range(high: pd.Series, low: pd.Series, close: pd.Series,
                       window: int = ATR_WINDOW) -> pd.Series:
    """Wilder's Average True Range (same values as the `ta` library).

    True range = largest of (high-low, |high-prev close|, |low-prev close|).
    ATR starts as the mean of the first `window` true ranges, then updates as
    atr = (previous atr * (window-1) + true range) / window.
    The first window-1 values are NaN.
    """
    prev_close = close.shift(1)
    tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()],
                   axis=1).max(axis=1)                 # first row = high - low
    values = tr.to_numpy(dtype=float)
    atr = np.full(len(values), np.nan)
    if len(values) >= window:
        atr[window - 1] = values[:window].mean()
        for i in range(window, len(values)):
            atr[i] = (atr[i - 1] * (window - 1) + values[i]) / window
    return pd.Series(atr, index=high.index)


def rolling_zscore(s: pd.Series, window: int = Z_WINDOW,
                   min_periods: int = Z_MIN_PERIODS) -> pd.Series:
    """(value - rolling mean) / rolling std over the past `window` days."""
    mean = s.rolling(window, min_periods=min_periods).mean()
    std = s.rolling(window, min_periods=min_periods).std().replace(0.0, np.nan)
    return (s - mean) / std


def _features_one_ticker(g: pd.DataFrame, window: int) -> pd.DataFrame:
    """All features for ONE ticker. `g` must be sorted by date."""
    out = pd.DataFrame(index=g.index)
    out["date"] = g["date"]
    out["ticker"] = g["ticker"]
    out["close"] = g["close"]
    out["volume"] = g["volume"]

    # Daily log return (dividend/split adjusted)
    out["ret"] = np.log(g["adj_close"]).diff()

    # Daily high-low range as a fraction of close (a volatility clue, NOT a spread)
    out["hl_range"] = (g["high"] - g["low"]) / g["close"]

    # Traded value in rupees, and rolling averages of volume and value
    out["traded_value"] = g["close"] * g["volume"]
    out["adv20"] = g["volume"].rolling(window, min_periods=window).mean()
    out["adv_value20"] = out["traded_value"].rolling(window, min_periods=window).mean()

    # Volume ratio: today vs the average of the PREVIOUS `window` days
    prev_adv = g["volume"].shift(1).rolling(window, min_periods=window).mean()
    out["volume_ratio"] = g["volume"] / prev_adv

    # Volatility of daily returns (daily and annualised)
    out["vol20"] = out["ret"].rolling(window, min_periods=window).std()
    out["vol20_ann"] = out["vol20"] * np.sqrt(TRADING_DAYS)

    # Amihud illiquidity: average of |return| per crore of traded value.
    # We use a small epsilon to prevent division by zero for ghost stocks.
    traded_crore = out["traded_value"] / CRORE
    amihud_daily = out["ret"].abs() / traded_crore.replace(0.0, np.nan)
    # Fill NaNs (where volume was 0) with a high penalty value (e.g., 10x the median)
    # to reflect extreme illiquidity without breaking the rolling mean with 'inf'.
    amihud_daily = amihud_daily.fillna(amihud_daily.median() * 10)
    out["amihud20"] = amihud_daily.rolling(window, min_periods=window).mean()

    # Spread: noisy daily Corwin-Schultz value (unclipped), then the rolling
    # average, floored at MIN_SPREAD. The floor is an assumption: real quoted
    # spreads on liquid NSE stocks are only a few bps, and a smoothed estimate
    # that is zero or negative just means "too small to measure".
    out["spread_raw"] = corwin_schultz_spread(g["high"], g["low"], g["close"])
    out["spread20"] = (out["spread_raw"].rolling(window, min_periods=window).mean()
                       .clip(lower=MIN_SPREAD))

    # ---- extra clues for the ML model -------------------------------------
    out["atr_pct"] = average_true_range(g["high"], g["low"], g["close"]) / g["close"]

    # Spread averaged after clipping each day at 0 (inflated, ML clue only)
    out["cs_clipped20"] = (out["spread_raw"].clip(lower=0.0)
                           .rolling(window, min_periods=window).mean())

    adj = g["adj_close"]
    out["ret_20d"] = adj.pct_change(RETURN_WINDOW)
    out["trend_50"] = adj / adj.rolling(TREND_WINDOW, min_periods=TREND_WINDOW).mean() - 1.0
    out["drawdown252"] = adj / adj.rolling(Z_WINDOW, min_periods=60).max() - 1.0

    out["volume_ratio_z"] = rolling_zscore(out["volume_ratio"])
    out["hl_range_z"] = rolling_zscore(out["hl_range"])
    out["vol20_z"] = rolling_zscore(out["vol20"])

    return out


# --------------------------------------------------------------------------
# 3. Main entry points
# --------------------------------------------------------------------------
def build_features(prices: pd.DataFrame, window: int = WINDOW) -> pd.DataFrame:
    """Compute all stock-level features for every ticker and date.

    `prices` can be raw or already cleaned; it is cleaned here either way.
    Early rows of each ticker have NaN where a rolling window is not full yet.
    """
    df = clean_prices(prices)
    parts = [_features_one_ticker(g, window) for _, g in df.groupby("ticker", sort=True)]
    if not parts:
        return pd.DataFrame(columns=FEATURE_COLUMNS)
    feats = pd.concat(parts, ignore_index=True)
    return feats[FEATURE_COLUMNS].sort_values(["ticker", "date"]).reset_index(drop=True)


def build_index_features(index: pd.DataFrame, window: int = WINDOW) -> pd.DataFrame:
    """Market-level features from the Nifty series (one row per date).

    idx_return    daily log return of Nifty
    idx_vol20     20-day volatility of that return
    idx_drawdown  Nifty / its highest level so far - 1
    India VIX is deliberately NOT a feature: it is the source of the stress
    label, and using it as an input too would make the classifier look perfect
    without learning anything.
    """
    df = index.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.dropna(subset=["nifty"])
    df = df[df["nifty"] > 0].drop_duplicates("date", keep="last").sort_values("date")
    nifty = df["nifty"].reset_index(drop=True)

    out = pd.DataFrame({"date": df["date"].reset_index(drop=True)})
    out["idx_return"] = np.log(nifty).diff()
    out["idx_vol20"] = out["idx_return"].rolling(window, min_periods=window).std()
    out["idx_drawdown"] = nifty / nifty.cummax() - 1.0
    return out


def add_market_features(features: pd.DataFrame, index: pd.DataFrame) -> pd.DataFrame:
    """Attach idx_return, idx_vol20 and idx_drawdown to the stock features by date."""
    market = build_index_features(index)
    return features.merge(market, on="date", how="left")


def latest_snapshot(features: pd.DataFrame, as_of=None) -> pd.DataFrame:
    """Most recent complete feature row for each ticker (one row per ticker).

    as_of: optional date. Only rows on or before it are used, e.g.
           latest_snapshot(features, "2020-03-24") gives the picture at the
           Covid crash, using no later data.
    Rows where adv20, vol20 or spread20 are still NaN are skipped.
    """
    df = features
    if as_of is not None:
        df = df[df["date"] <= pd.Timestamp(as_of)]
    df = df.dropna(subset=["adv20", "vol20", "spread20"])
    return (df.sort_values(["ticker", "date"])
              .groupby("ticker", as_index=False)
              .tail(1)
              .reset_index(drop=True))


# --------------------------------------------------------------------------
# 4. Run as a script
# --------------------------------------------------------------------------
if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    prices = load_prices(root / "data" / "prices.csv")
    feats = build_features(prices)

    out_path = root / "data" / "features.csv"
    feats.to_csv(out_path, index=False)
    print(f"Saved {out_path}  ({len(feats)} rows, {feats['ticker'].nunique()} tickers, "
          f"{len(FEATURE_COLUMNS)} columns)")

    snap = latest_snapshot(feats)
    show = snap[["ticker", "date", "adv20", "vol20", "amihud20", "atr_pct", "trend_50"]].copy()
    show["date"] = show["date"].dt.date
    print("\nLatest snapshot:")
    print(show.round(5).to_string(index=False))