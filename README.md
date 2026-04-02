# Trading Assistant — Setup & Run Guide

## Requirements
Python 3.10+

## Install dependencies
```bash
pip install -r requirements.txt
```

## File structure
Place all files in the same directory:
```
trading_assistant/
├── config.py
├── database.py
├── universe.py
├── data_fetch.py
├── scoring.py
├── risk.py
├── learning.py
├── backtest.py
├── app.py
├── requirements.txt
└── data/        ← auto-created on first run
└── logs/        ← auto-created on first run
```

## First run (important order)
1. Start the app:
   ```bash
   streamlit run app.py
   ```
2. In the sidebar → click **Refresh Universe** (~2-5 min)
   - Fetches S&P 500 from Wikipedia
   - Applies Filter A / Filter B criteria
   - Caches sector data

3. Navigate to **Daily Trades** → click **Generate Today's Trades**

## Pages
| Page | Purpose |
|------|---------|
| 📈 Daily Trades | Generate up to 3 signals with full explanation |
| 👁 Watchlist | Score any ticker, track custom list |
| 📚 Trade History | View/update trades, export CSV |
| 📊 Performance | Live stats, equity curve, backtest |
| 🧠 Strategy Insights | Weights, learning curves, reset |

## Configuration
All tunables are in `config.py`:
- `CAPITAL` — your portfolio size (default $100,000)
- `RISK_PER_TRADE` — fraction risked per trade (default 1%)
- `ATR_STOP_MULT` / `ATR_TARGET_MULT` — stop/target multipliers (default 1×/2×)

## Disclaimer
Educational use only. Not financial advice.
