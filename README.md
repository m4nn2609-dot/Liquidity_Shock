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
├── data/                  # data loading
│   ├── dataloader.py      # downloads OHLCV, returns DataFrames
│   └── data_test.ipynb    # testing the loader
├── eda/
│   └── eda.ipynb          # exploratory analysis
├── feature_engineering/
│   ├── features.py        # all feature functions
│   └── fe_test.ipynb      # testing and plotting features
├── modeling/              # main ML work
│   ├── model.py           # normal vs stressed classifier, training and evaluation
│   └── modeling.ipynb     # experiments, validation, plots
├── backend/               # FastAPI service
│   ├── app.py             # app and endpoints
│   └── schemas.py         # Pydantic request/response models
├── main.py                # entry point
├── requirements.txt
└── README.md
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
- _add teammates here_
