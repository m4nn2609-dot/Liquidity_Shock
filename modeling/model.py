"""modeling/model.py
Normal vs stressed classifier.

Label : y = 1 if india_vix > 80th percentile of VIX in the TRAINING years, else 0.
        india_vix is the answer key and is NEVER an input.
Split : train on earlier years, test on later years (no random split).
Models: logistic regression (scaled) + random forest, both class-weighted.

Run from the project root:   python -m modeling.model
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

FEATURES = ["volume_ratio", "volume_ratio_z", "log_amihud20", "spread20", "cs_clipped20",
            "ret", "hl_range", "hl_range_z", "vol20", "vol20_z", "atr_pct",
            "ret_20d", "trend_50", "drawdown252"]
MARKET = ["idx_return", "idx_vol20", "idx_drawdown"]


def _day(series):
    """'2020-03-02 00:00:00+05:30' or '2020-03-02' -> plain date. Cuts the text, so no timezone shift."""
    return pd.to_datetime(series.astype(str).str[:10])


def load(path="../data/features.csv", market_path="../data/market_index.csv", vix_col="india_vix"):
    """Reads features.csv, then adds what features.csv does not contain, from market_index.csv:
       - india_vix (the answer key, found by name if it is called something else)
       - idx_return, idx_vol20, idx_drawdown (built by feature_engineering.features.build_index_features)"""
    df = pd.read_csv(path)
    if "date" not in df.columns:
        df = df.rename(columns={df.columns[0]: "date"})
    df["date"] = _day(df["date"])

    need_vix = "india_vix" not in df.columns
    need_idx = [c for c in MARKET if c not in df.columns]
    if need_vix or need_idx:
        mk = pd.read_csv(market_path)
        if "date" not in mk.columns:
            mk = mk.rename(columns={mk.columns[0]: "date"})
        mk["date"] = _day(mk["date"])
        mk = mk.drop_duplicates("date", keep="last")
        if need_vix:
            vc = vix_col if vix_col in mk.columns else next((c for c in mk.columns if "vix" in str(c).lower()), None)
            if vc is None:
                raise KeyError(f"no VIX column in {market_path}; columns are {mk.columns.tolist()}")
            df = df.merge(mk[["date", vc]].rename(columns={vc: "india_vix"}), on="date", how="left")
        if need_idx:
            try:
                from feature_engineering.features import build_index_features
                df = df.merge(build_index_features(mk)[["date"] + need_idx], on="date", how="left")
            except Exception as e:  # noqa: BLE001
                print(f"  [warn] could not build index features ({type(e).__name__}: {e})")

    df = df.dropna(subset=["india_vix"])  # days without a VIX value cannot be labelled
    if "amihud20" in df.columns:
        df["log_amihud20"] = np.log1p(df["amihud20"])
    return df.sort_values(["date", "ticker"]).reset_index(drop=True)


def make_label(df, train_end="2023-12-31", q=0.80):
    train = df["date"] <= pd.Timestamp(train_end)
    thr = df.loc[train].drop_duplicates("date")["india_vix"].quantile(q)  # TRAINING years only, one value per date
    out = df.copy()
    out["y"] = (out["india_vix"] > thr).astype(int)
    out["is_train"] = train
    return out, thr


def get_xy(df, use_market):
    cols = [c for c in FEATURES + (MARKET if use_market else []) if c in df.columns]
    missing = [c for c in FEATURES + (MARKET if use_market else []) if c not in df.columns]
    if missing:
        print("  [warn] columns not found, skipped:", missing)
    X = df[cols].replace([np.inf, -np.inf], np.nan)
    keep = X.notna().all(axis=1)
    return X[keep], df.loc[keep], cols


def models():
    return {
        "Logistic": make_pipeline(StandardScaler(),
                                  LogisticRegression(class_weight="balanced", max_iter=2000)),
        "RandomForest": RandomForestClassifier(n_estimators=300, max_depth=8, min_samples_leaf=50,
                                               class_weight="balanced_subsample",
                                               n_jobs=-1, random_state=42),
    }


def score(y_true, proba, thr=0.5):
    pred = (proba >= thr).astype(int)
    return {
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "f1": f1_score(y_true, pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, proba),
        "pr_auc": average_precision_score(y_true, proba),
    }


def run(path="data/features.csv", train_end="2023-12-31", use_market=True):
    df, thr = make_label(load(path), train_end)
    X, d, cols = get_xy(df, use_market)
    tr, te = d["is_train"].to_numpy(), ~d["is_train"].to_numpy()
    ytr, yte = d.loc[tr, "y"], d.loc[te, "y"]
    print(f"VIX threshold (train 80th pct): {thr:.2f}")
    print(f"stress share  train: {ytr.mean():.1%}   test: {yte.mean():.1%}   features: {len(cols)}")

    rows, fitted = {}, {}
    for name, m in models().items():
        m.fit(X[tr], ytr)
        proba = m.predict_proba(X[te])[:, 1]
        r = score(yte, proba)
        # every stock shares the same label on a date, so also score one number per DATE
        by_date = pd.DataFrame({"date": d.loc[te, "date"].values, "p": proba, "y": yte.values}) \
                    .groupby("date").agg(p=("p", "mean"), y=("y", "first"))
        r["roc_auc_by_date"] = roc_auc_score(by_date["y"], by_date["p"]) if by_date["y"].nunique() > 1 else np.nan
        rows[name], fitted[name] = r, m
    results = pd.DataFrame(rows).T.round(3)
    results.attrs["baseline_pr_auc"] = round(float(yte.mean()), 3)  # PR-AUC of random guessing
    return results, fitted, cols


def feature_importance(fitted, cols):
    rf = fitted["RandomForest"]
    return pd.Series(rf.feature_importances_, index=cols).sort_values(ascending=False)


if __name__ == "__main__":
    for use_market in (False, True):
        print(f"\n=== with market features: {use_market} ===")
        res, fit, cols = run(use_market=use_market)
        print(res)
        print("PR-AUC of random guessing:", res.attrs["baseline_pr_auc"])
    print("\nTop features (RF):")
    print(feature_importance(fit, cols).head(8))