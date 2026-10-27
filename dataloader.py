"""
Download historical data once and save to CSV.

Usage:  python download_data.py
Output: data/prices.csv         (long format: one row per date and ticker)
        data/market_index.csv   (Nifty 50 and India VIX closes)
"""
import time
from datetime import date
from pathlib import Path

import pandas as pd
import yfinance as yf

START = "2018-01-01"
END = date.today().isoformat()

# Mix of large caps and small/mid caps so liquidity differs across stocks.
# Edit freely. Tickers that fail are skipped and reported.
TICKERS = [
    # Large caps
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS",
    "ICICIBANK.NS", "SBIN.NS", "ITC.NS", "LT.NS",
    # Mid caps
    "PERSISTENT.NS", "VOLTAS.NS", "NHPC.NS", "IRFC.NS", "PNB.NS",
    # Small caps / less liquid
    "IDEA.NS", "SUZLON.NS", "YESBANK.NS", "RPOWER.NS",
    "HFCL.NS", "JPPOWER.NS", "TRIDENT.NS",
]

INDEX_TICKERS = {"^NSEI": "nifty", "^INDIAVIX": "india_vix"}

DATA_DIR = Path("data")


def fetch_one(ticker, retries=3):
    """Download one ticker with simple retry. Returns DataFrame or None."""
    for attempt in range(1, retries + 1):
        try:
            df = yf.download(ticker, start=START, end=END,
                             auto_adjust=False, progress=False)
            if df is not None and not df.empty:
                # Newer yfinance returns MultiIndex columns; flatten them.
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                return df
        except Exception as e:
            print(f"  {ticker} attempt {attempt} failed: {e}")
        time.sleep(2)
    return None


def download_prices():
    frames, failed = [], []
    for t in TICKERS:
        print(f"Downloading {t} ...")
        df = fetch_one(t)
        if df is None:
            failed.append(t)
            continue
        df = df.reset_index()
        df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]
        df["ticker"] = t
        keep = ["date", "ticker", "open", "high", "low",
                "close", "adj_close", "volume"]
        frames.append(df[keep])

    if not frames:
        raise SystemExit("No data downloaded. Check internet or use a Kaggle CSV.")

    prices = pd.concat(frames, ignore_index=True)
    prices = prices.sort_values(["ticker", "date"]).reset_index(drop=True)
    prices.to_csv(DATA_DIR / "prices.csv", index=False)
    print(f"\nSaved data/prices.csv  ({len(prices)} rows, "
          f"{prices['ticker'].nunique()} tickers)")
    if failed:
        print("FAILED tickers (replace or remove them):", failed)


def download_index():
    series = {}
    for sym, name in INDEX_TICKERS.items():
        print(f"Downloading {sym} ...")
        df = fetch_one(sym)
        if df is None:
            print(f"  Could not get {sym}, skipping.")
            continue
        series[name] = df["Close"]
    if not series:
        print("No index data saved.")
        return
    idx = pd.DataFrame(series)
    idx.index.name = "date"
    idx.reset_index().to_csv(DATA_DIR / "market_index.csv", index=False)
    print(f"Saved data/market_index.csv ({len(idx)} rows)")


if __name__ == "__main__":
    DATA_DIR.mkdir(exist_ok=True)
    download_prices()
    download_index()
