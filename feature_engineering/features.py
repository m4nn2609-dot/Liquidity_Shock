import numpy as np
import pandas as pd
from ta.trend import SMAIndicator
from ta.volatility import AverageTrueRange


def log_return(df):
    # kept in numpy: ta's version is in percent (x100)
    return np.log(df['Close'] / df['Close'].shift(1)).rename('log_return')


def adv(df, window=20):
    return df['Volume'].rolling(window).mean().rename(f'adv_{window}')


def dollar_adv(df, window=20):
    return (df['Close'] * df['Volume']).rolling(window).mean().rename(f'dollar_adv_{window}')


def volume_ratio(df, window=20):
    a = adv(df, window).replace(0, np.nan)
    return (df['Volume'] / a).rename('volume_ratio')


def amihud(df, window=20):
    dollar_vol = (df['Close'] * df['Volume']).replace(0, np.nan)
    daily = log_return(df).abs() / dollar_vol
    return (daily.rolling(window, min_periods=window // 2).mean() * 1e6).rename('amihud')


def realized_vol(df, window=20):
    return log_return(df).rolling(window).std().rename('realized_vol')


def atr_pct(df, window=14):
    atr = AverageTrueRange(df['High'], df['Low'], df['Close'], window=window).average_true_range()
    atr.iloc[:window - 1] = np.nan
    return (atr / df['Close']).rename('atr_pct')


def spread_proxy(df):
    return ((df['High'] - df['Low']) / df['Close']).rename('spread_proxy')


def corwin_schultz(df, window=20):
    h, l = df['High'], df['Low']
    beta = np.log(h / l) ** 2 + np.log(h.shift(1) / l.shift(1)) ** 2
    h2 = np.maximum(h, h.shift(1))
    l2 = np.minimum(l, l.shift(1))
    gamma = np.log(h2 / l2) ** 2
    k = 3 - 2 * np.sqrt(2)
    alpha = (np.sqrt(2 * beta) - np.sqrt(beta)) / k - np.sqrt(gamma / k)
    s = 2 * (np.exp(alpha) - 1) / (1 + np.exp(alpha))
    return s.clip(lower=0).rolling(window).mean().rename('cs_spread')


def momentum(df):
    close = df['Close']
    sma50 = SMAIndicator(close, window=50).sma_indicator()
    return pd.concat([
        close.pct_change(20).rename('ret_20d'),
        (close / sma50 - 1).rename('trend_50'),
    ], axis=1)

def zscore(series, window=252):
    mean = series.rolling(window).mean()
    std = series.rolling(window).std()
    return ((series - mean) / std).rename(f'{series.name}_z')


def drawdown(df):
    return (df['Close'] / df['Close'].cummax() - 1).rename('drawdown')


def index_features(df):
    return pd.concat([
        log_return(df).rename('idx_return'),
        realized_vol(df).rename('idx_vol'),
        drawdown(df).rename('idx_drawdown'),
    ], axis=1)


def build_features(df):
    vr = volume_ratio(df)
    sp = spread_proxy(df)
    rv = realized_vol(df)
    return pd.concat([
        log_return(df),
        adv(df),
        dollar_adv(df),
        vr,
        rv,
        atr_pct(df),
        amihud(df),
        sp,
        corwin_schultz(df),
        momentum(df),
        zscore(vr),
        zscore(sp),
        zscore(rv),
    ], axis=1)