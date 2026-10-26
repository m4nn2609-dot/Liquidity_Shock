"""execution/schedules.py
Immediate vs TWAP vs VWAP vs front-loaded schedules, plus an optimizer.

Formulas (from the process guide):
  impact cost  = sum_d  Y * sigma * sqrt(q_d / ADV) * q_d * price
  spread cost  = 0.5 * spread * shares * price
  waiting risk = sigma * price * sqrt(sum_d x_d^2),  x_d = shares still held after day d
  optimizer    : minimise impact + lam * risk,  sum q_d = S,  0 <= q_d <= p * ADV,  D <= deadline

Inputs for ONE stock (take them from features.py, do not recompute):
  shares S, adv (shares/day), sigma (DAILY volatility, vol20), price, spread (fraction, spread20)

If your liquidation.py already has impact_cost_for_slices with the same formula,
import it and delete impact_cost() below, so the numbers match everywhere.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize


# ---------- cost pieces ----------
def spread_cost(shares, spread, price):
    return 0.5 * spread * shares * price


def impact_cost(slices, adv, sigma, price, Y=1.0):
    q = np.clip(np.asarray(slices, dtype=float), 0, None)
    return float(np.sum(Y * sigma * np.sqrt(q / adv) * q * price))


def waiting_risk(slices, sigma, price):
    q = np.clip(np.asarray(slices, dtype=float), 0, None)
    held = q.sum() - np.cumsum(q)  # shares still held after each day
    return float(sigma * price * np.sqrt(np.sum(held ** 2)))


# ---------- schedules (all return shares sold per day) ----------
def min_days(S, adv, p):
    return int(np.ceil(S / (p * adv)))


def immediate(S):
    return np.array([float(S)])


def twap(S, D):
    return np.full(D, S / D)


def vwap(S, volumes):
    """volumes = expected volume per day (e.g. the last D daily volumes). Flat -> same as TWAP."""
    v = np.asarray(volumes, dtype=float)
    return S * v / v.sum()


def front_loaded(S, D, cap=None):
    """Sell more early, less later (linearly falling weights).
    If cap (= p * ADV) is given, no day exceeds it; the excess moves to later days."""
    w = np.arange(D, 0, -1, dtype=float)
    q = S * w / w.sum()
    if cap is None:
        return q
    if cap * D < S - 1e-6:
        raise ValueError("cap * D < S: cannot fit the position inside the deadline")
    for _ in range(100):  # clip, then re-spread the excess over days that still have room
        over = np.clip(q - cap, 0, None)
        if over.sum() < 1e-6:
            break
        q = np.minimum(q, cap)
        room = cap - q
        q = q + over.sum() * room / room.sum()
    return q


def at_cap(S, adv, p):
    """Fastest legal schedule: sell p * ADV every day until done."""
    cap = p * adv
    n = int(np.ceil(S / cap))
    q = np.full(n, cap)
    q[-1] = S - cap * (n - 1)
    return q


# ---------- optimizer ----------
def optimize_schedule(S, adv, sigma, price, p=0.10, deadline=None, lam=0.1, Y=1.0):
    """Best split of S shares over <= deadline days. Convex problem, so SLSQP finds the global optimum."""
    cap = p * adv
    d_min = min_days(S, adv, p)
    D = int(deadline) if deadline else 2 * d_min
    if D < d_min:
        raise ValueError(f"Infeasible: need at least {d_min} days at {p:.0%} participation, deadline is {D}.")

    value = S * price

    def objective(w):  # w = fraction of S sold each day; result in bps of position value
        q = np.clip(w, 0, None) * S
        return 1e4 * (impact_cost(q, adv, sigma, price, Y) + lam * waiting_risk(q, sigma, price)) / value

    x0 = np.full(D, 1.0 / D)
    res = minimize(
        objective, x0, method="SLSQP",
        bounds=[(0.0, min(1.0, cap / S))] * D,
        constraints=[{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}],
        options={"maxiter": 400, "ftol": 1e-10},
    )
    w = np.clip(res.x, 0, None)
    q = S * w / w.sum()
    return q


# ---------- compare everything ----------
def evaluate(slices, adv, sigma, price, spread, p=0.10, Y=1.0, lam=0.1):
    slices = np.asarray(slices, dtype=float)
    S = slices.sum()
    value = S * price
    imp = impact_cost(slices, adv, sigma, price, Y)
    spr = spread_cost(S, spread, price)
    risk = waiting_risk(slices, sigma, price)
    return {
        "days": int((slices > 1e-6 * S).sum()),
        "spread_cost": spr,
        "impact_cost": imp,
        "total_cost": spr + imp,
        "cost_bps": 1e4 * (spr + imp) / value,
        "waiting_risk": risk,
        "risk_bps": 1e4 * risk / value,
        "objective_bps": 1e4 * (imp + lam * risk) / value,
        "peak_participation": slices.max() / adv,
        "within_cap": bool(slices.max() <= p * adv * 1.0001),
    }


def compare_schedules(S, adv, sigma, price, spread, p=0.10, Y=1.0, lam=0.1,
                      deadline=None, volumes=None):
    """One row per strategy. Feed this DataFrame straight into the dashboard."""
    d_min = min_days(S, adv, p)
    D = int(deadline) if deadline else 2 * d_min
    D = max(D, d_min)
    vols = volumes if volumes is not None else np.ones(D)
    vols = np.resize(np.asarray(vols, dtype=float), D)

    plans = {
        "Immediate": immediate(S),
        "At cap (fastest legal)": at_cap(S, adv, p),
        "TWAP": twap(S, D),
        "VWAP": vwap(S, vols),
        "Front-loaded": front_loaded(S, D, cap=p * adv),
        "Optimized": optimize_schedule(S, adv, sigma, price, p, D, lam, Y),
    }
    rows = {name: evaluate(q, adv, sigma, price, spread, p, Y, lam) for name, q in plans.items()}
    return pd.DataFrame(rows).T, plans


def efficient_frontier(S, adv, sigma, price, spread, p=0.10, Y=1.0, deadline=None,
                       lams=(0.0, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0)):
    """Impact cost vs waiting risk as lam changes. Plot cost_bps (x) against risk_bps (y)."""
    out = []
    for lam in lams:
        q = optimize_schedule(S, adv, sigma, price, p, deadline, lam, Y)
        r = evaluate(q, adv, sigma, price, spread, p, Y, lam)
        out.append({"lam": lam, "days": r["days"], "cost_bps": r["cost_bps"], "risk_bps": r["risk_bps"]})
    return pd.DataFrame(out)