# Institutional Binance & Zuperior Quantitative Signal Engine

A production-grade, modular quantitative trading system and algorithmic signal engine written in Python 3.11+. The system is architected for high expectancy and strict risk preservation across **Binance (USDⓈ-M Futures & Spot)** and **Zuperior MetaTrader 5 (MT5)**.

It pairs non-repainting execution logic with an institutional 3-stage regime filter, real-time dynamic precision quantization (`tickSize`, `stepSize`, `minNotional`), intraday High-Water Mark Drawdown circuit breakers, and an event-driven backtesting validator modeling taker fees, slippage, and funding debits.

---

## 1. System Architecture

```
forex/
├── config/                     # Configuration & Environment
│   ├── settings.py             # AppSettings, TradingMode, ExecutionEnvironment, Risk parameters
│   └── symbols.py              # Symbol specifications (BTCUSDT, ETHUSDT, SOLUSDT, PAXGUSDT)
├── core/                       # Quantitative Core Pipeline
│   ├── exchange_metadata.py    # Binance exchangeInfo dynamic cache & precision rounding
│   ├── market_stream.py        # Async WebSocket stream (kline, bookTicker, markPrice)
│   ├── regime_filter.py        # 4-State market classifier (Trending, Mean-Reverting, Chop)
│   ├── signal_engine.py        # Zero-repaint closed-bar quantitative strategies
│   └── risk_manager.py         # ATR position sizing, 1:2.5 min R:R, 3% Daily DD Kill-Switch
├── execution/                  # Institutional Order Execution & Telemetry
│   └── binance_router.py       # HMAC-SHA256 authenticated router, drift sync, SQLite/CSV audit
├── backtest/                   # Quantitative Validation
│   └── validator.py            # Event-driven backtester (maker/taker fees, funding, bracket simulation)
├── strategies/                 # MT5 & Multi-Asset Strategies
│   ├── base_strategy.py        # Abstract strategy base & TradeSignal model
│   ├── gold_strategy.py        # London/NY liquidity sweeps
│   ├── btc_strategy.py         # Volatility squeeze breakouts
│   └── nasdaq_strategy.py      # US Cash Open ORB + VWAP
├── tests/                      # Automated Verification
│   └── test_binance_engine.py  # 13-test Pytest suite (Precision, Regimes, Risk, Router)
├── data/                       # Telemetry & Trade Database
│   ├── trades.db               # SQLite trade ledger with full audit trails
│   └── trades.csv              # CSV audit log
├── main.py                     # Unified CLI Interface
├── .env.example                # Sample environment template
├── requirements.txt            # Python dependencies
└── README.md                   # System documentation
```

---

## 2. Core Quantitative Pillars

### 2.1 Dynamic Precision & Notional Quantization
Binance enforces strict order filters that reject orders with precision mismatches.
- **Prices:** Quantized strictly to `tickSize` multiples using exact `Decimal` arithmetic.
- **Quantities:** Quantized strictly downward (`ROUND_DOWN`) to `stepSize` increments to ensure the maximum calculated dollar risk is never breached.
- **Notional Guard:** Orders below `minNotional` or outside `minQty`/`maxQty` bounds are caught and validated prior to dispatch.

### 2.2 Async WebSocket Multiplexing & Zero Repainting
- Replaces heavy REST polling with Binance WebSocket streams:
  - `<symbol>@kline_5m`: Consumes live bars, but triggers strategies strictly when `kline.x == True` (candle close).
  - `<symbol>@bookTicker`: Real-time bid/ask spread tracking in basis points (bps).
  - `<symbol>@markPrice`: Mark price and 8-hour funding rates for liquidation-proof stop execution.
- Exponential backoff and auto-reconnect handles transient network drops.

### 2.3 4-State Regime Filter & Funding Guard
Signals must pass a multi-metric regime gatekeeper before order generation:
- **`TRENDING_BULL` / `TRENDING_BEAR`:** ADX(14) > 25.0, ATR percentile > 40th, Fast EMA > Slow EMA. Activates Squeeze Momentum breakouts.
- **`MEAN_REVERTING`:** ADX(14) < 20.0, compressed ATR volatility. Activates Liquidity Sweep reversals.
- **`CHOP_NO_TRADE`:** Indeterminate volatility or mixed momentum. System halts entries.
- **Spread Guard:** Rejects entries if live spread exceeds symbol cap (e.g. > 3.0 bps for BTC, > 6.0 bps for SOL).
- **Funding Rate Guard:** Blocks Longs if 8h funding rate > +0.05% (crowded long premium); blocks Shorts if funding < -0.05% (crowded short discount).

### 2.4 Risk Preservation & Drawdown Kill-Switch
- **Fixed Fractional Risk:** Dollar risk per trade is hard-capped at 1.0% of portfolio equity:
  $$\text{Quantity} = \frac{\text{Equity} \times \text{Risk\%}}{|\text{Entry} - \text{StopLoss}|}$$
- **Minimum 1:2.5 R:R Ratio:** Trades with TP1 below 2.5 times the stop distance are discarded.
- **High-Water Mark Drawdown Kill-Switch:** Tracks intraday equity against the day's peak. If daily drawdown reaches or exceeds **3.0%**, the engine triggers an emergency halt, blocking new entries until 00:00 UTC.

### 2.5 Execution Safety & Telemetry
- **Clock Drift Sync:** Automatically calculates server timestamp drift ($\Delta t = \text{server\_time} - \text{local\_time}$) to prevent Binance `-1021 INVALID_TIMESTAMP` rejections.
- **Bracket Exits:** Stop Loss and Take Profit orders are dispatched with `reduceOnly=True` to eliminate accidental position flips.
- **Dual Telemetry:** Every execution event and filled bracket is logged simultaneously to `data/trades.db` (SQLite) and `data/trades.csv`.

---

## 3. Installation & Setup

### Requirements
- Python 3.11+
- Linux (Ubuntu/Pop!_OS/Debian), macOS, or Windows

```bash
# Clone the repository
git clone <repository_url>
cd forex

# Install dependencies
pip install -r requirements.txt
```

### Environment Configuration
Copy the template `.env.example` to `.env`:
```bash
cp .env.example .env
```
Configure your credentials in `.env`:
```ini
# Binance Configuration
BINANCE_API_KEY="your_api_key_here"
BINANCE_API_SECRET="your_api_secret_here"
TRADING_MODE="USD_M_FUTURES"         # USD_M_FUTURES or SPOT
EXECUTION_ENV="PAPER"                # PAPER or LIVE
RISK_PER_TRADE_PERCENT="1.0"
DAILY_MAX_DRAWDOWN_PERCENT="3.0"

# Telegram Bot (Optional)
TELEGRAM_BOT_TOKEN="your_bot_token"
TELEGRAM_CHAT_ID="your_chat_id"
```

---

## 4. Usage & Execution Modes

### 1. Verify Exchange Filters & Precision
Validate live connection to Binance endpoints and verify `tickSize`, `stepSize`, `minNotional`, and filter rules:
```bash
python3 main.py --verify-metadata
```

### 2. Single Market Scan (`--once`)
Run a single evaluation pass across all registered symbols (`BTCUSDT`, `ETHUSDT`, `SOLUSDT`, `PAXGUSDT`), displaying closed-bar prices, spread in bps, and market regime states:
```bash
python3 main.py --once
```

### 3. Continuous Paper Trading (`--paper`)
Run live WebSocket market feeds and paper execute trades with zero financial risk:
```bash
python3 main.py --paper
```

### 4. Production Live Execution (`--live`)
Deploy live execution with real order routing, `reduceOnly=True` exit brackets, and SQLite telemetry:
```bash
python3 main.py --live
```

### 5. Event-Driven Backtesting (`--backtest`)
Run an event-driven backtest on real historical Binance klines, simulating spread crossing slippage, Binance taker fees (0.05%), maker fees (0.02%), and 8-hour funding debits/credits:
```bash
# Backtest BTCUSDT over the last 14 days
python3 main.py --backtest BTCUSDT --days 14

# Backtest ETHUSDT over the last 30 days
python3 main.py --backtest ETHUSDT --days 30
```

---

## 5. Automated Verification Suite

Run the full pytest suite to verify precision quantizers, regime filters, risk circuit breakers, and clock-drift synchronization:
```bash
pytest -v tests/test_binance_engine.py
```

### Verified Test Cases:
1. `test_tick_size_rounding`: Precision quantizer rounds prices to exact tick size.
2. `test_step_size_rounding_down`: Quantity rounds down to step size (protecting dollar risk limit).
3. `test_min_notional_validation`: Orders below Binance minimum notional threshold are rejected.
4. `test_wide_spread_rejection`: Gatekeeper blocks execution when spread exceeds bps cap.
5. `test_extreme_positive_funding_blocks_longs`: Blocks Longs during crowded funding premium (> +0.05%).
6. `test_extreme_negative_funding_blocks_shorts`: Blocks Shorts during crowded funding discount (< -0.05%).
7. `test_chop_no_trade_regime`: Blocks trades when ADX and ATR detect low-volatility chop.
8. `test_fixed_fractional_sizing_math`: Verifies base asset position size formula.
9. `test_daily_drawdown_kill_switch`: Verifies portfolio shutdown upon 3.0% daily drawdown breach.
10. `test_minimum_rr_ratio_enforcement`: Rejects signals offering less than 1:2.5 risk-to-reward.
11. `test_clock_drift_sync`: Adjusts client request timestamps according to exchange drift.
12. `test_http_429_retry_after_handling`: Backs off and honors HTTP 429 rate limit cooldowns.
13. `test_deterministic_client_order_id`: Generates unique, reproducible `newClientOrderId` strings.

---

## 6. License & Disclaimer

This software is for research and educational purposes. Trading cryptocurrency futures and foreign exchange carries significant risk of capital loss. Always test strategies thoroughly in paper-trading environments before committing live capital.
