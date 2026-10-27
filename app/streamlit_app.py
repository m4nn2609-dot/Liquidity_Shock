"""
LiquidityShock dashboard.   Run from the project root:   streamlit run app/streamlit_app.py

Everything shown here is computed by the project modules (no numbers are typed in):
  feature_engineering/features.py  market numbers      execution/liquidation.py  days and costs
  stress/scenarios.py              crisis what-ifs     execution/schedules.py    selling plans, optimizer
  modeling/model.py                normal vs stressed detector
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from execution import schedules
from execution.liquidation import (BUCKET_LABELS, build_positions, liquidated_fraction_by_day,
                                   liquidation_report, load_portfolio, portfolio_summary)
from feature_engineering.features import build_features, latest_snapshot, load_prices
from modeling import model
from stress.scenarios import PRESETS, compare, monte_carlo

CRORE = 1e7
st.set_page_config(page_title="LiquidityShock", page_icon="📉", layout="wide")


# ------------------------------------------------------------------ cached work
@st.cache_data(show_spinner="Building features from prices.csv ...")
def get_features():
    return build_features(load_prices(ROOT / "data" / "prices.csv"))


@st.cache_data(show_spinner=False)
def run_monte_carlo(book, runs, p, Y):
    return monte_carlo(book, runs=runs, p=p, Y=Y)


@st.cache_data(show_spinner="Optimising selling schedules ...")
def run_schedules(S, adv, sigma, price, spread, p, Y, lam, D, vols):
    return schedules.compare_schedules(S, adv, sigma, price, spread, p=p, Y=Y, lam=lam, deadline=D,
                                       volumes=np.array(vols) if vols else None)


@st.cache_data(show_spinner="Tracing the efficient frontier (several optimisations) ...")
def run_frontier(S, adv, sigma, price, spread, p, Y, D):
    return schedules.efficient_frontier(S, adv, sigma, price, spread, p=p, Y=Y, deadline=D,
                                        lams=(0.0, 0.05, 0.2, 0.5, 2.0))


@st.cache_resource(show_spinner="Training the stress detector (a few seconds) ...")
def train_ml(_features, train_end="2023-12-31"):
    out, fits = {}, {}
    for use_market in (False, True):
        res, fit, cols = model.run(features=_features, train_end=train_end, use_market=use_market)
        out[use_market] = dict(res=res, importance=model.feature_importance(fit, cols),
                               baseline=res.attrs["baseline_pr_auc"])
        fits[use_market] = fit
    df, thr = model.make_label(model.load(features=_features), train_end)
    X, d, _ = model.get_xy(df, use_market=False)
    proba = fits[False]["RandomForest"].predict_proba(X)[:, 1]
    by_date = (pd.DataFrame({"date": d["date"].values, "probability": proba,
                             "vix": d["india_vix"].values, "stressed": d["y"].values})
               .groupby("date").agg(probability=("probability", "mean"), vix=("vix", "first"),
                                    stressed=("stressed", "first")).reset_index())
    te = by_date[by_date["date"] > pd.Timestamp(train_end)]
    stats = dict(false_alarm=float((te.loc[te["stressed"] == 0, "probability"] >= 0.5).mean()),
                 caught=float((te.loc[te["stressed"] == 1, "probability"] >= 0.5).mean()),
                 n_stressed=int(te["stressed"].sum()), n_days=len(te))
    return (out, by_date, thr, float(d.loc[d["is_train"], "y"].mean()),
            float(d.loc[~d["is_train"], "y"].mean()), stats)


# ------------------------------------------------------------------ sidebar
st.sidebar.title("Settings")
p = st.sidebar.slider("Participation rate (max % of daily volume)", 5, 30, 10, format="%d%%") / 100
Y = st.sidebar.slider("Market-impact coefficient Y", 0.5, 1.5, 1.0, 0.1)
preset = st.sidebar.selectbox("Stress scenario", list(PRESETS) + ["Custom"], index=2)
if preset == "Custom":
    params = dict(adv_mult=st.sidebar.slider("Volume multiplier", 0.1, 1.5, 0.5, 0.05),
                  spread_mult=st.sidebar.slider("Spread multiplier", 1.0, 6.0, 3.0, 0.5),
                  vol_mult=st.sidebar.slider("Volatility multiplier", 1.0, 3.0, 1.5, 0.25))
else:
    params = PRESETS[preset]
    st.sidebar.caption(f"Volume x{params['adv_mult']}, spread x{params['spread_mult']}, "
                       f"volatility x{params['vol_mult']}")
uploaded = st.sidebar.file_uploader("Portfolio CSV (ticker, shares, sector)", type="csv")

# ------------------------------------------------------------------ data
features = get_features()
snapshot = latest_snapshot(features)
try:
    portfolio = load_portfolio(uploaded if uploaded is not None else ROOT / "data" / "portfolio.csv")
    positions = build_positions(portfolio, snapshot)
except (ValueError, KeyError) as e:
    st.error(f"Could not use that portfolio: {e}")
    st.stop()

report = liquidation_report(positions, participation=p, impact_y=Y)
summary = portfolio_summary(report)
book = positions.rename(columns={"adv": "adv20", "vol": "vol20", "spread": "spread20"})[
    ["ticker", "shares", "price", "adv20", "vol20", "spread20"]]

st.title("📉 LiquidityShock")
st.caption("Market liquidity risk and execution simulator. "
           f"Market data up to {positions['as_of'].max().date()}; selling at most {p:.0%} of daily volume.")

tab_over, tab_stress, tab_sched, tab_ml, tab_method = st.tabs(
    ["Overview", "Stress test", "Selling schedules", "Stress detector (ML)", "Method and limits"])

# ------------------------------------------------------------------ overview
with tab_over:
    c = st.columns(5)
    c[0].metric("Portfolio value", f"₹{summary['total_value_cr']:,.0f} cr")
    c[1].metric("Cost to liquidate", f"₹{summary['total_cost_cr']:,.1f} cr",
                f"{summary['total_cost_bps']:.0f} bps", delta_color="off", delta_arrow="off")
    c[2].metric("Longest exit", f"{summary['longest_days']:.0f} days",
                summary["longest_ticker"], delta_color="off", delta_arrow="off")
    c[3].metric("Value-weighted days", f"{summary['value_weighted_days']:.1f}")
    c[4].metric("Value needing > 10 days", f"{summary['pct_value_over_10_days']:.0f}%")
    impact_share = 100 * summary["impact_cost_cr"] / summary["total_cost_cr"]
    st.caption(f"Market impact is {impact_share:.0f}% of the cost; the spread is the rest.")

    st.subheader("Liquidity-risk ranking (hardest to sell first)")
    show = report[["risk_rank", "ticker", "sector", "position_cr", "days_to_liquidate",
                   "total_cost_bps", "bucket", "liquidity_risk_score"]].copy()
    show["bucket"] = show["bucket"].astype(str)
    show.columns = ["Rank", "Ticker", "Sector", "Position (₹ cr)", "Days to sell",
                    "Cost (bps)", "Bucket", "Risk score"]
    st.dataframe(show.round({"Position (₹ cr)": 1, "Days to sell": 1, "Cost (bps)": 1, "Risk score": 0}),
                 hide_index=True)
    st.download_button("Download ranking (CSV)", report.to_csv(index=False),
                       "liquidation_report.csv", "text/csv")

    left, right = st.columns(2)
    with left:
        st.subheader("How fast can we turn it into cash?")
        prof = liquidated_fraction_by_day(positions, p)
        fig = px.line(prof, x="day", y="pct_value_liquidated",
                      labels={"day": "Trading day", "pct_value_liquidated": "% of portfolio value sold"})
        fig.update_yaxes(range=[0, 100])
        st.plotly_chart(fig)

    with right:
        st.subheader("Days to liquidate, by stock")
        bar = report.sort_values("days_to_liquidate")
        fig = px.bar(bar, x="days_to_liquidate", y="ticker", color=bar["bucket"].astype(str), orientation="h",
                     category_orders={"color": BUCKET_LABELS}, height=520,
                     labels={"days_to_liquidate": "Days to liquidate", "ticker": "", "color": "Bucket"})
        st.plotly_chart(fig)

# ------------------------------------------------------------------ stress
with tab_stress:
    table, s = compare(book, p=p, Y=Y, **params)
    st.subheader(f"Normal vs stressed: {preset}")
    c = st.columns(4)
    c[0].metric("Cost, normal", f"₹{s['cost_normal'] / CRORE:,.1f} cr", f"{s['bps_normal']:.0f} bps", delta_color="off", delta_arrow="off")
    c[1].metric("Cost, stressed", f"₹{s['cost_stress'] / CRORE:,.1f} cr", f"{s['bps_stress']:.0f} bps", delta_color="off", delta_arrow="off")
    c[2].metric("Extra cost", f"+{100 * (s['cost_stress'] / s['cost_normal'] - 1):.0f}%")
    c[3].metric("Longest exit", f"{s['longest_exit_stress']} days", f"was {s['longest_exit_normal']}", delta_color="off", delta_arrow="off")

    t = table.copy()
    t["days_normal"], t["days_stress"] = t["days_normal"].astype(int), t["days_stress"].astype(int)
    t = t[["ticker", "days_normal", "days_stress", "cost_bps_normal", "cost_bps_stress", "cost_increase_pct"]]
    t.columns = ["Ticker", "Days (normal)", "Days (stress)", "Cost bps (normal)", "Cost bps (stress)", "Extra cost %"]
    a, b = st.columns([3, 2])
    a.dataframe(t.round(1), hide_index=True)
    long = table.melt(id_vars="ticker", value_vars=["days_normal", "days_stress"], var_name="case", value_name="days")
    b.plotly_chart(px.bar(long, x="days", y="ticker", color="case", barmode="group", orientation="h", height=520,
                          labels={"days": "Days to liquidate", "ticker": ""}))

    st.subheader("Everything stressed together (Monte Carlo)")
    runs = st.slider("Number of simulated scenarios", 500, 5000, 2000, 500)
    total, mc = run_monte_carlo(book, runs, p, Y)
    c = st.columns(4)
    c[0].metric("Normal", f"₹{mc['base_cost'] / CRORE:,.1f} cr")
    c[1].metric("Median stress", f"₹{mc['median'] / CRORE:,.1f} cr", f"{mc['median_bps']:.0f} bps", delta_color="off", delta_arrow="off")
    c[2].metric("95th percentile", f"₹{mc['p95'] / CRORE:,.1f} cr", f"{mc['p95_bps']:.0f} bps", delta_color="off", delta_arrow="off")
    c[3].metric("99th percentile", f"₹{mc['p99'] / CRORE:,.1f} cr")
    fig = go.Figure(go.Histogram(x=total / CRORE, nbinsx=50, marker_color="lightgray"))
    for label, val, color in [("normal", mc["base_cost"], "green"), ("median", mc["median"], "orange"),
                              ("95th pct", mc["p95"], "red")]:
        fig.add_vline(x=val / CRORE, line_color=color, annotation_text=label)
    fig.update_layout(xaxis_title="Total liquidation cost (₹ crore)", yaxis_title="Scenarios", showlegend=False)
    st.plotly_chart(fig)
    st.caption("Each scenario draws one market-wide shock (volume down, spreads and volatility up) plus a small "
               "shock per stock. In the real Covid crash volume actually rose while volatility exploded, "
               "so 'volume falls' is a what-if, not history.")

# ------------------------------------------------------------------ schedules
with tab_sched:
    st.subheader("Sell it all now, or spread it out?")
    tick = st.selectbox("Stock", list(report["ticker"]), index=0)
    r = positions.set_index("ticker").loc[tick]
    S, adv, sig, price, spr = (float(r["shares"]), float(r["adv"]), float(r["vol"]),
                               float(r["price"]), float(r["spread"]))
    d_min = schedules.min_days(S, adv, p)
    c1, c2 = st.columns(2)
    mult = c1.slider("Deadline (x the minimum days)", 1.0, 3.0, 2.0, 0.25)
    lam = c2.slider("Risk aversion λ (higher = sell faster)", 0.0, 2.0, 0.1, 0.05)
    D = max(int(round(mult * d_min)), d_min)
    vols = tuple(features.loc[features["ticker"] == tick, "volume"].tail(D).to_numpy())
    tbl, plans = run_schedules(S, adv, sig, price, spr, p, Y, lam, D, vols)

    t = tbl.copy()
    t["within_cap"] = t["within_cap"].astype(bool)
    for col in t.columns.drop("within_cap"):
        t[col] = pd.to_numeric(t[col])
    best = t["objective_bps"].idxmin()
    st.caption(f"{tick}: {S:,.0f} shares, at least {d_min} days at {p:.0%} participation, deadline {D} days. "
               f"Lowest cost + λ·risk: **{best}**.")
    out = pd.DataFrame({"Days": t["days"].astype(int), "Expected cost (bps)": t["cost_bps"],
                        "Waiting risk (bps)": t["risk_bps"], "Cost + λ·risk (bps)": t["objective_bps"],
                        "Peak % of daily volume": 100 * t["peak_participation"], "Within cap": t["within_cap"]})
    st.dataframe(out.round(1))
    st.caption("Waiting risk = one standard deviation of the value change while shares are still unsold. "
               "VWAP uses the stock's last D daily volumes as its expected volume profile.")

    a, b = st.columns(2)
    fig = go.Figure()
    for name, q in plans.items():
        x = np.arange(1, len(q) + 1)
        fig.add_trace(go.Scatter(x=x, y=np.cumsum(q) / S * 100, name=name,
                                 mode="lines+markers" if len(q) <= 3 else "lines"))
    fig.update_layout(xaxis_title="Trading day", yaxis_title="% of position sold")
    a.plotly_chart(fig)
    if b.checkbox("Show the efficient frontier (extra optimisations; slow for very long plans)"):
        frontier = run_frontier(S, adv, sig, price, spr, p, Y, D)
        fig = px.line(frontier, x="cost_bps", y="risk_bps", text="lam", markers=True,
                      labels={"cost_bps": "Expected cost (bps)", "risk_bps": "Waiting risk (bps)"},
                      title="Efficient frontier (labels are λ)")
        fig.update_traces(textposition="top right")
        b.plotly_chart(fig)

# ------------------------------------------------------------------ ML
with tab_ml:
    st.subheader("Is today a normal or a stressed market?")
    st.write("A model learns from stock-level clues (volume ratio, volatility, ATR, Amihud, drawdown ...) "
             "to label each day normal or stressed. The answer key is India VIX above its 80th percentile "
             "in the training years (2018-2023); VIX itself is never an input. Test years: 2024 onwards.")
    if st.checkbox("Train and show the detector"):
        ml, by_date, thr, share_tr, share_te, stats = train_ml(features)
        c = st.columns(4)
        c[0].metric("VIX stress threshold", f"{thr:.1f}")
        c[1].metric("Stressed days, train", f"{share_tr:.0%}")
        c[2].metric("Stressed days, test", f"{share_te:.0%}")
        last = by_date.iloc[-1]
        c[3].metric(f"Model stress score, {last['date'].date()}", f"{last['probability']:.0%}",
                    f"India VIX {last['vix']:.1f} (stress line {thr:.1f})", delta_color="off", delta_arrow="off")
        st.caption(f"Read the score with care: on the test years a score of 50% or more appeared on "
                   f"{stats['false_alarm']:.0%} of normal days (false alarms) and on {stats['caught']:.0%} of the "
                   f"{stats['n_stressed']} stressed days (out of {stats['n_days']}). It is an early-warning score, "
                   f"not a verdict. On the latest date the real India VIX is "
                   f"{'below' if last['vix'] < thr else 'above'} the stress line.")

        for use_market, title in [(False, "Stock-level clues only"), (True, "Plus Nifty index clues")]:
            st.markdown(f"**{title}**")
            a, b = st.columns([3, 2])
            a.dataframe(ml[use_market]["res"].round(3))
            imp = ml[use_market]["importance"].head(8).iloc[::-1]
            b.plotly_chart(px.bar(imp, orientation="h", labels={"value": "Importance", "index": ""}, height=300))
        st.caption(f"A random guess scores PR-AUC {ml[False]['baseline']:.3f}. Stressed days are rare in the test "
                   "years, so precision is low and scores are noisy. With Nifty clues the model mostly reads "
                   "idx_vol20 and idx_drawdown, which closely track VIX, so treat that version with care.")

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=by_date["date"], y=by_date["probability"], name="Stress probability (avg of stocks)"))
        fig.add_trace(go.Scatter(x=by_date["date"], y=by_date["vix"], name="India VIX", yaxis="y2",
                                 line=dict(color="gray", width=1)))
        fig.add_shape(type="line", x0="2024-01-01", x1="2024-01-01", yref="paper", y0=0, y1=1,
                      line=dict(dash="dash"))
        fig.update_layout(yaxis=dict(title="Probability", range=[0, 1]),
                          yaxis2=dict(title="India VIX", overlaying="y", side="right"),
                          legend=dict(orientation="h"))
        st.plotly_chart(fig)
        st.caption("Dashed line = start of the test years. The model detects stress on the day; it does not forecast it.")

# ------------------------------------------------------------------ method
with tab_method:
    st.subheader("How the numbers are made")
    st.markdown("""
- **Days to liquidate** = shares / (participation x ADV), ADV = 20-day average volume.
- **Spread cost** = 0.5 x spread x position value. The spread is *estimated* from daily High/Low (Corwin-Schultz), floored at 1 bp.
- **Market impact** per day = Y x daily volatility x sqrt(q / ADV) x q x price (q = shares sold that day).
- **Waiting risk** = volatility x price x sqrt(sum of squared unsold shares): the price can move while we wait.
- **Optimizer** minimises impact + λ x waiting risk, with every day's sale capped at participation x ADV.
- **Stress** scales volume, spread and volatility; **Monte Carlo** stresses all stocks together.
""")
    st.subheader("Limits to keep in mind")
    st.markdown("""
- No historical bid-ask data exists for free, so spreads are estimates and often sit on the 1 bp floor for liquid stocks.
- Impact uses a simple square-root model with a fixed coefficient Y (try 0.5 to 1.5).
- ADV is a 20-day average; no intraday data, circuit limits or settlement effects.
- Prices are Yahoo Finance data; a few glitch days are removed. IRFC listed in 2021, so it has no 2020 history.
- The ML label is a market-fear proxy (India VIX). Stress is a small sample, so scores are noisy.
""")