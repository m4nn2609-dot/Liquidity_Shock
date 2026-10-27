"""audit.py  -  checks the whole LiquidityShock pipeline, layer by layer.

Put this file in the PROJECT ROOT (next to dataloader.py) and run:
    python audit.py
Paste the whole output back. Every check prints PASS / WARN / FAIL / ERROR / SKIP.
Nothing here changes your files.
"""
from __future__ import annotations

import os
import subprocess
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, os.getcwd())

RESULTS = []
STATE = {}
GUIDE_COST_CRORE = 50.6   # from your process guide, 10% participation
GUIDE_BPS = 74


class Warn(Exception):
    pass


class Skip(Exception):
    pass


def check(name):
    def deco(fn):
        try:
            msg = fn()
            RESULTS.append(("PASS", name, msg or ""))
        except Warn as e:
            RESULTS.append(("WARN", name, str(e)))
        except Skip as e:
            RESULTS.append(("SKIP", name, str(e)))
        except AssertionError as e:
            RESULTS.append(("FAIL", name, str(e)))
        except Exception as e:  # noqa: BLE001
            RESULTS.append(("ERROR", name, f"{type(e).__name__}: {e}"))
        print(f"[{RESULTS[-1][0]:5}] {name}  {RESULTS[-1][2]}")
        return fn
    return deco


def section(title):
    print(f"\n=== {title} ===")


def find(df, *names):
    """Case-insensitive column lookup. Returns the real column name or None."""
    low = {c.lower().replace(" ", "_"): c for c in df.columns}
    for n in names:
        if n.lower() in low:
            return low[n.lower()]
    return None


def day(series):
    """Date text -> plain date by cutting to YYYY-MM-DD (no timezone conversion, so no one-day shifts)."""
    return pd.to_datetime(series.astype(str).str[:10])


def need(key):
    if key not in STATE:
        raise Skip(f"needs '{key}' from an earlier check that failed")
    return STATE[key]


# ------------------------------------------------------------------ 1. files
section("1. FILES")
FILES = {k: f"data/{k}.csv" for k in ("prices", "market_index", "portfolio", "features")}


@check("data files exist")
def _():
    missing = [f for f in FILES.values() if not os.path.exists(f)]
    assert not missing, f"missing: {missing}  (run from the project root; features.csv comes from features_check.ipynb)"
    return "all 4 found"


for key in ("prices", "market_index", "portfolio", "features"):
    if os.path.exists(FILES[key]):
        STATE[key] = pd.read_csv(FILES[key])

# ------------------------------------------------------------------ 2. prices
section("2. RAW PRICES (data/prices.csv)")


@check("prices: columns and tickers")
def _():
    df = need("prices").copy()
    cols = {k: find(df, k) for k in ("date", "ticker", "open", "high", "low", "close", "volume")}
    missing = [k for k, v in cols.items() if v is None]
    assert not missing, f"missing columns {missing}; found {df.columns.tolist()}"
    df = df.rename(columns={v: k for k, v in cols.items()})
    df["date"] = day(df["date"])
    STATE["px"] = df
    n = df["ticker"].nunique()
    if n != 20:
        raise Warn(f"{n} tickers (guide says 20)")
    return f"{n} tickers, {df['date'].min().date()} to {df['date'].max().date()}"


@check("prices: no High < Low, no bad close")
def _():
    df = need("px")
    bad_hl = int((df["high"] < df["low"]).sum())
    bad_c = int((df["close"] <= 0).sum() + df["close"].isna().sum())
    assert bad_hl == 0 and bad_c == 0, f"High<Low rows: {bad_hl}, bad close rows: {bad_c}"


@check("prices: no duplicate (date, ticker)")
def _():
    df = need("px")
    d = int(df.duplicated(["date", "ticker"]).sum())
    assert d == 0, f"{d} duplicate rows"


@check("prices: zero-volume days per ticker")
def _():
    df = need("px")
    z = (df["volume"] <= 0).groupby(df["ticker"]).mean()
    bad = z[z > 0.02]
    if len(bad):
        raise Warn("more than 2% zero-volume days: " + ", ".join(f"{t} {v:.1%}" for t, v in bad.items()))
    return f"worst ticker {z.max():.2%}"


@check("prices: data is recent")
def _():
    last = need("px")["date"].max()
    age = (pd.Timestamp.today().normalize() - last.normalize()).days
    if age > 10:
        raise Warn(f"latest date {last.date()} is {age} days old")
    return f"latest {last.date()}"


# ------------------------------------------------------------------ 3. market
section("3. MARKET INDEX (data/market_index.csv)")


@check("market: Nifty and VIX columns, VIX range")
def _():
    mk = need("market_index").copy()
    dc = find(mk, "date") or mk.columns[0]
    mk = mk.rename(columns={dc: "date"})
    mk["date"] = day(mk["date"])
    vix = [c for c in mk.columns if "vix" in c.lower()]
    assert vix, f"no VIX-like column in {mk.columns.tolist()}"
    v = mk[vix[0]].dropna()
    STATE["mk"], STATE["vix_col"] = mk, vix[0]
    assert 5 < v.min() and v.max() < 120, f"VIX values look wrong: min {v.min():.1f}, max {v.max():.1f}"
    return f"column '{vix[0]}', range {v.min():.1f} to {v.max():.1f}, p80 {v.quantile(.8):.1f}"


@check("market: dates overlap prices")
def _():
    a, b = set(need("px")["date"]), set(need("mk")["date"])
    share = len(a & b) / len(a)
    assert share > 0.9, f"only {share:.0%} of price dates have a market row (timezone or format mismatch?)"
    return f"{share:.0%} overlap"


# ------------------------------------------------------------------ 4. portfolio
section("4. PORTFOLIO (data/portfolio.csv)")


@check("portfolio: shares positive, tickers known")
def _():
    p = need("portfolio")
    assert "ticker" in p.columns and "shares" in p.columns, f"columns: {p.columns.tolist()}"
    assert (p["shares"] > 0).all(), "non-positive shares"
    unknown = set(p["ticker"]) - set(need("px")["ticker"])
    assert not unknown, f"tickers not in prices.csv: {sorted(unknown)}"
    return f"{len(p)} holdings"


# ------------------------------------------------------------------ 5. features
section("5. FEATURES (data/features.csv)")
NEEDED = ["ticker", "date", "close", "adv20", "vol20", "spread20"]


@check("features: required columns")
def _():
    f = need("features").copy()
    dc = find(f, "date") or f.columns[0]
    f = f.rename(columns={dc: "date"})
    f["date"] = day(f["date"])
    miss = [c for c in NEEDED if c not in f.columns]
    assert not miss, f"missing {miss}; found {f.columns.tolist()}"
    STATE["feat"] = f
    return f"{f.shape[1]} columns, {len(f):,} rows"


@check("features: no inf, no duplicate (date, ticker)")
def _():
    f = need("feat")
    num = f.select_dtypes("number")
    inf = int(np.isinf(num.to_numpy()).sum())
    dup = int(f.duplicated(["date", "ticker"]).sum())
    assert inf == 0 and dup == 0, f"inf values: {inf}, duplicate rows: {dup}"


@check("features: latest row per ticker has no NaN in book columns")
def _():
    f = need("feat")
    last = f.sort_values("date").groupby("ticker").tail(1)
    bad = last[last[["adv20", "vol20", "spread20", "close"]].isna().any(axis=1)]["ticker"].tolist()
    assert not bad, f"NaN in latest row for {bad}"
    STATE["latest"] = last


@check("features: UNITS (daily vol, spread as a fraction, positive ADV)")
def _():
    last = need("latest")
    v, s, a = last["vol20"], last["spread20"], last["adv20"]
    assert (a > 0).all(), "adv20 has zeros or negatives"
    assert v.median() < 0.10, f"median vol20 = {v.median():.3f}; looks ANNUALISED, the code expects daily sigma"
    assert v.median() > 0.002, f"median vol20 = {v.median():.4f}; suspiciously small"
    assert s.max() < 0.10, f"spread20 max {s.max():.3f}; looks like percent, code expects a fraction (0.001 = 10 bps)"
    floor = float((s <= 0.000101).mean())
    msg = f"vol20 median {v.median():.3f}, spread20 median {s.median()*1e4:.1f} bps, {floor:.0%} of stocks on the 1 bp floor"
    return msg


# ------------------------------------------------------------------ 6. book + liquidation
section("6. BOOK AND LIQUIDATION")


@check("book builds from portfolio + latest features")
def _():
    p, last = need("portfolio"), need("latest")
    book = (p.merge(last[["ticker", "close", "adv20", "vol20", "spread20"]], on="ticker")
              .rename(columns={"close": "price"}))
    assert len(book) == len(p), f"merge dropped {len(p) - len(book)} holdings"
    assert book[["shares", "price", "adv20", "vol20", "spread20"]].notna().all().all(), "NaN in book"
    STATE["book"] = book
    days = np.ceil(book["shares"] / (0.10 * book["adv20"]))
    worst = book.assign(days=days).sort_values("days").iloc[-1]
    return f"{len(book)} holdings, slowest {worst['ticker']} {int(worst['days'])} days"


@check("book cost vs your guide (50.6 crore, 74 bps)")
def _():
    from stress.scenarios import book_cost
    book = need("book")
    c = book_cost(book)["total_cost"].sum()
    value = (book["shares"] * book["price"]).sum()
    cr, bps = c / 1e7, 1e4 * c / value
    msg = f"{cr:.1f} crore, {bps:.0f} bps (guide: {GUIDE_COST_CRORE} crore, {GUIDE_BPS} bps)"
    if abs(cr / GUIDE_COST_CRORE - 1) > 0.10:
        raise Warn(msg + "  -> differs by >10%; fine if data or portfolio changed since the guide")
    return msg


@check("liquidation.py imports and exposes its functions")
def _():
    try:
        import execution.liquidation as lq
    except ModuleNotFoundError as e:
        raise AssertionError(f"cannot import execution/liquidation.py ({e}); run audit.py from the project root")
    names = [n for n in dir(lq) if not n.startswith("_") and callable(getattr(lq, n))]
    want = {"days_to_liquidate", "spread_cost", "liquidation_report", "portfolio_summary"}
    miss = want - set(names)
    if miss:
        raise Warn(f"functions not found by these names: {sorted(miss)}  (found: {names[:12]})")
    return "core functions present"


# ------------------------------------------------------------------ 7. stress
section("7. STRESS SCENARIOS")


@check("stress: more stress -> higher cost, none -> unchanged")
def _():
    from stress.scenarios import compare
    book = need("book")
    _, base = compare(book, 1, 1, 1)
    _, a = compare(book, 0.5, 1, 1)
    _, s = compare(book, 1, 3, 1)
    _, v = compare(book, 1, 1, 1.5)
    _, all_ = compare(book, 0.5, 3, 1.5)
    assert abs(base["cost_stress"] - base["cost_normal"]) < 1e-6, "multipliers of 1 changed the cost"
    assert a["cost_stress"] > base["cost_normal"], "halving volume did not raise cost"
    assert s["cost_stress"] > base["cost_normal"], "wider spread did not raise cost"
    assert v["cost_stress"] > base["cost_normal"], "higher volatility did not raise cost"
    assert all_["cost_stress"] > max(a["cost_stress"], s["cost_stress"], v["cost_stress"]), "combined stress not the worst"
    assert all_["longest_exit_stress"] == 2 * base["longest_exit_normal"] or \
        all_["longest_exit_stress"] >= 2 * base["longest_exit_normal"] - 1, "halving ADV should double the longest exit"
    return f"{all_['bps_normal']:.0f} -> {all_['bps_stress']:.0f} bps, longest exit {all_['longest_exit_normal']} -> {all_['longest_exit_stress']} days"


@check("monte carlo: ordering and reproducibility")
def _():
    from stress.scenarios import monte_carlo
    book = need("book")
    t1, m1 = monte_carlo(book, runs=1000, seed=1)
    t2, m2 = monte_carlo(book, runs=1000, seed=1)
    assert np.allclose(t1, t2), "same seed gave different results"
    assert m1["median"] <= m1["p95"] <= m1["p99"], "percentiles out of order"
    assert m1["median"] > m1["base_cost"], "median stressed cost should exceed normal cost"
    return f"normal {m1['base_cost']/1e7:.1f} < median {m1['median']/1e7:.1f} < p95 {m1['p95']/1e7:.1f} crore"


# ------------------------------------------------------------------ 8. schedules
section("8. SCHEDULES AND OPTIMIZER (slowest stock)")


@check("schedules: sums, cap, ordering")
def _():
    from execution.schedules import compare_schedules
    book = need("book")
    row = book.assign(days=np.ceil(book["shares"] / (0.10 * book["adv20"]))).sort_values("days").iloc[-1]
    tbl, plans = compare_schedules(row["shares"], row["adv20"], row["vol20"], row["price"], row["spread20"], p=0.10, lam=0.1)
    for k, q in plans.items():
        assert abs(q.sum() - row["shares"]) < 1, f"{k}: slices sum to {q.sum():.0f}, not {row['shares']:.0f}"
        assert (q >= -1e-9).all(), f"{k}: negative slice"
    legal = tbl[tbl["within_cap"].astype(bool)]
    assert not tbl.loc["Immediate", "within_cap"], "'Immediate' should break the participation cap"
    assert tbl.loc["Optimized", "objective_bps"] <= legal["objective_bps"].min() + 1e-6, "Optimized is not the best legal schedule"
    assert tbl.loc["Immediate", "total_cost"] > tbl.loc["At cap (fastest legal)", "total_cost"], "dumping at once should cost more"
    STATE["row"] = row
    return f"{row['ticker']}: immediate {tbl.loc['Immediate','cost_bps']:.0f} bps vs at-cap {tbl.loc['At cap (fastest legal)','cost_bps']:.0f} bps"


@check("optimizer: cost up, risk down as lambda grows")
def _():
    from execution.schedules import efficient_frontier
    r = need("row")
    fr = efficient_frontier(r["shares"], r["adv20"], r["vol20"], r["price"], r["spread20"], p=0.10,
                            lams=(0.0, 0.1, 0.5, 2.0))
    assert fr["cost_bps"].is_monotonic_increasing, f"cost not rising with lambda: {fr['cost_bps'].round(1).tolist()}"
    assert fr["risk_bps"].is_monotonic_decreasing, f"risk not falling with lambda: {fr['risk_bps'].round(1).tolist()}"


@check("optimizer: infeasible deadline raises a clear error")
def _():
    from execution.schedules import optimize_schedule
    r = need("row")
    try:
        optimize_schedule(r["shares"], r["adv20"], r["vol20"], r["price"], 0.10, deadline=2)
    except ValueError:
        return
    raise AssertionError("no error for a 2-day deadline on a long exit")


# ------------------------------------------------------------------ 9. ML
section("9. ML CLASSIFIER")


@check("ML: VIX label and NO leakage of VIX into inputs")
def _():
    from modeling import model as M
    banned = {"india_vix", "date", "ticker", "y", "is_train"}
    leaked = banned & set(M.FEATURES + M.MARKET)
    assert not leaked, f"answer-key or ID columns used as inputs: {leaked}"
    df = M.load()
    d, thr = M.make_label(df, "2023-12-31")
    tr = d[d["is_train"]]
    te = d[~d["is_train"]]
    msg = f"VIX threshold {thr:.1f} (guide ~21.1), stress share train {tr['y'].mean():.0%}, test {te['y'].mean():.0%}"
    if te["y"].sum() == 0 or tr["y"].sum() == 0:
        raise AssertionError("one split has no stressed days. " + msg)
    return msg


@check("ML: models run on a time split, scores sane")
def _():
    from modeling import model as M
    out = []
    for use_market in (False, True):
        res, _, cols = M.run(use_market=use_market)
        top = res["roc_auc"].max()
        out.append(f"market={use_market}: best ROC-AUC {top:.2f}, PR-AUC {res['pr_auc'].max():.2f} "
                   f"(random {res.attrs['baseline_pr_auc']}), by-date AUC {res['roc_auc_by_date'].max():.2f}")
        assert top > 0.5, f"model no better than chance ({out[-1]})"
        if top > 0.98:
            raise Warn("ROC-AUC above 0.98 is suspicious, check for leakage. " + " | ".join(out))
    return " | ".join(out)


# ------------------------------------------------------------------ 10. tests and repo
section("10. TESTS AND REPO HYGIENE")


@check("pytest")
def _():
    if not os.path.isdir("tests") and not any(f.startswith("test_") for f in os.listdir(".")):
        raise Skip("no tests folder")
    r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-x", "--disable-warnings"],
                       capture_output=True, text=True, timeout=600)
    text = (r.stdout.strip() or r.stderr.strip())
    last = (text.splitlines() or ["(no output)"])[-1]
    if "No module named pytest" in text:
        raise Skip("pytest is not installed in this environment (pip install pytest)")
    if r.returncode == 5:
        raise Skip("pytest found no tests")
    assert r.returncode == 0, last + "   (run `python -m pytest -x` to see which test failed)"
    return last


@check("git: data/features.csv is not committed (remove before the freeze)")
def _():
    r = subprocess.run(["git", "ls-files", "data/features.csv"], capture_output=True, text=True)
    if r.returncode != 0:
        raise Skip("not a git repo here")
    if r.stdout.strip():
        raise Warn("features.csv is tracked by git; your rules say to remove it before the demo")


@check(".gitignore covers .venv and notebook checkpoints")
def _():
    g = open(".gitignore").read() if os.path.exists(".gitignore") else ""
    miss = [x for x in (".venv", ".ipynb_checkpoints", "__pycache__") if x not in g]
    if miss:
        raise Warn(f"not in .gitignore: {miss}")


# ------------------------------------------------------------------ summary
print("\n" + "=" * 60)
counts = {k: sum(1 for r in RESULTS if r[0] == k) for k in ("PASS", "WARN", "FAIL", "ERROR", "SKIP")}
print("SUMMARY:", counts)
bad = [r for r in RESULTS if r[0] in ("FAIL", "ERROR")]
if bad:
    print("\nFIX THESE FIRST (in order):")
    for s, n, m in bad:
        print(f" - [{s}] {n}: {m}")
sys.exit(1 if bad else 0)
