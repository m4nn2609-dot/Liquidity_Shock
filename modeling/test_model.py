import numpy as np
import pandas as pd
import pytest

from modeling import model as m


def make_features(n_days=1500, tickers=("AAA", "BBB"), seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2018-01-01", periods=n_days)
    stress = (np.sin(np.arange(n_days) / 40) > 0.6).astype(float)         # pretend stress episodes
    rows = []
    for t in tickers:
        d = pd.DataFrame({"date": dates, "ticker": t})
        for c in m.FEATURES:
            if c != "log_amihud20":
                d[c] = rng.normal(size=n_days) + 2.0 * stress
        d["amihud20"] = np.exp(rng.normal(-9, 1, n_days) + stress)
        rows.append(d)
    f = pd.concat(rows, ignore_index=True)
    nifty = 10000 * np.exp(np.cumsum(rng.normal(0, 0.01, n_days)))
    mk = pd.DataFrame({"date": dates, "nifty": nifty, "india_vix": 14 + 20 * stress + rng.normal(0, 1, n_days)})
    return f, mk


def test_vix_is_never_an_input():
    assert "india_vix" not in m.FEATURES + m.MARKET


def test_threshold_uses_training_years_only(tmp_path):
    f, mk = make_features()
    mk_path = tmp_path / "mk.csv"
    mk.to_csv(mk_path, index=False)
    df1, thr1 = m.make_label(m.load(features=f, market_path=mk_path), "2021-12-31")
    mk2 = mk.copy()
    mk2.loc[mk2["date"] > "2021-12-31", "india_vix"] += 100          # change the TEST years only
    mk2.to_csv(mk_path, index=False)
    df2, thr2 = m.make_label(m.load(features=f, market_path=mk_path), "2021-12-31")
    assert thr1 == pytest.approx(thr2)
    train = df1["is_train"]
    assert (df1.loc[train, "y"].to_numpy() == df2.loc[train, "y"].to_numpy()).all()


def test_split_is_by_time(tmp_path):
    f, mk = make_features()
    p = tmp_path / "mk.csv"
    mk.to_csv(p, index=False)
    df, _ = m.make_label(m.load(features=f, market_path=p), "2021-12-31")
    assert df.loc[df["is_train"], "date"].max() < df.loc[~df["is_train"], "date"].min()


def test_log_amihud_is_a_real_log(tmp_path):
    f, mk = make_features()
    mk.to_csv(tmp_path / "mk.csv", index=False)
    df = m.load(features=f, market_path=tmp_path / "mk.csv")
    assert df["log_amihud20"].min() < -5                              # log1p would stay near 0


def test_run_end_to_end_on_synthetic_data(tmp_path):
    f, mk = make_features()
    mk.to_csv(tmp_path / "mk.csv", index=False)
    df = m.load(features=f, market_path=tmp_path / "mk.csv")
    df, thr = m.make_label(df, "2021-12-31")
    X, d, cols = m.get_xy(df, use_market=True)
    tr = d["is_train"].to_numpy()
    model = m.models()["RandomForest"].fit(X[tr], d.loc[tr, "y"])
    proba = model.predict_proba(X[~tr])[:, 1]
    r = m.score(d.loc[~tr, "y"], proba)
    assert set(r) == {"precision", "recall", "f1", "roc_auc", "pr_auc"}
    assert r["roc_auc"] > 0.8                                         # the pretend signal is strong