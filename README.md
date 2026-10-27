# LiquidityShock

**Market Liquidity Risk & Execution Simulator** (QuantActive Code Conflux, PS05)

> **Status:** work in progress. The structure, features and roadmap below are *proposed* and will change as we build.

A portfolio can look liquid in normal markets and become very expensive to exit in a crisis: volumes fall, spreads widen, and large orders move prices against you. LiquidityShock estimates what it would cost, and how long it would take, to liquidate a portfolio under normal and stressed conditions.

## What it does

- Loads historical prices and volumes for NSE stocks (via yfinance)
- Builds liquidity, volatility, spread and trend features
- Estimates days to liquidate under a participation limit
- Estimates transaction cost from spreads and market impact
- Compares immediate vs staged liquidation schedules
- Stresses the portfolio (lower volume, wider spreads, correlated selling)
- Ranks holdings by liquidity risk
- Classifies normal vs stressed liquidity conditions with ML

## Project structure

```
Liquidity_Shock/
├── data/                              CSV data only
│   ├── prices.csv                     market data
│   ├── market_index.csv               Nifty and VIX
│   ├── portfolio.csv                  your holdings
│   ├── features.csv                   (generated, to be removed later)
│   └── data_checks.ipynb              
│
├── dataloader.py                      downloads the CSVs (run once)
│
├── feature_engineering/               STEP 1: numbers from prices
│   ├── __init__.py
│   ├── features.py                    ADV, volatility, spread, Amihud, ML clues
│   ├── features_check.ipynb
|   └── test_features.py      
│
├── execution/                         STEP 2: days and costs
│   ├── __init__.py
│   ├── liquidation.py                 the file we just built
│   └── schedules.py                   TO DO: sell-today vs staged, optimizer
│
├── stress/                            STEP 3: crisis versions
│   └── scenarios.py                   TO DO
│
├── modeling/                          STEP 4: ML (in parallel)
│   └── model.py                       TO DO: normal vs stressed classifier
│
├── app/                               STEP 5: what judges see
│   └── streamlit_app.py               TO DO
│
├── tests/
│   ├── test_features.py               17 tests
│   └── test_liquidation.py            20 tests
│
├── conftest.py                        empty, lets tests find the folders
├── requirements.txt  .gitignore  README.md
└── docs/                              report, slides, demo video
```

_This layout is a proposal. Folders and files may be renamed, split or merged as the project evolves._

## Features (per stock)

| Group | Features |
|---|---|
| Liquidity | `adv_20`, `dollar_adv_20`, `volume_ratio`, `amihud` |
| Spread | `spread_proxy`, `cs_spread` (Corwin-Schultz estimator) |
| Volatility | `realized_vol`, `atr_pct` |
| Trend | `ret_20d`, `trend_50` |
| Z-scores | `volume_ratio_z`, `spread_proxy_z`, `realized_vol_z` |
| Market-wide (from `^NSEI`) | `idx_return`, `idx_vol`, `idx_drawdown` |

All features use only past data, with no look-ahead.

**Note on spreads:** free sources do not provide historical bid-ask data, so spreads are estimated from daily High/Low prices. These are proxies, not quoted spreads.

## Getting started

```bash
git clone <repo-url>
cd Liquidity_Shock

python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux

pip install -r requirements.txt
```

Run the API from the project root:

```bash
uvicorn backend.app:app --reload
```

Then open http://127.0.0.1:8000/docs to try the endpoints.

## API

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Server check |
| GET | `/ticker` | List of supported tickers |
| GET | `/features/{ticker}` | Feature table (planned) |
| POST | `/liquidate` | Days to liquidate and cost for a portfolio (planned) |
| POST | `/stress` | Normal vs stressed comparison (planned) |

## Roadmap

- [x] Project structure
- [x] Feature functions
- [x] API skeleton and schemas
- [ ] Data cleaning (drop zero-volume rows, strip timezones)
- [ ] Portfolio config and participation limit
- [ ] Days to liquidate, spread cost, liquidity ranking
- [ ] Market impact model
- [ ] Immediate vs staged schedules and optimizer
- [ ] Stress scenarios
- [ ] Normal vs stressed classifier (`modeling/`)
- [ ] Dashboard and demo

## Team

- Mann ([m4nn2609](https://github.com/m4nn2609))
- Sagnik([plasmy-05](https://github.com/plasmy-05))
