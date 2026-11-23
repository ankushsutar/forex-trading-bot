# Zuperior MetaTrader 5 (MT5) Algorithmic Signal Engine

A modular, production-grade algorithmic trading signal engine written in Python 3.11+ tailored specifically for **Zuperior MetaTrader 5 (MT5)** accounts. It generates institutional-grade trade alerts with exact Entry, Stop Loss, Take Profit 1 (1:2.5 R:R), Take Profit 2 (1:3.5+ R:R), and exact MT5 Lot Sizes normalized to broker volume steps for **Gold (XAUUSD)**, **Bitcoin (BTCUSD)**, and **Nasdaq (USTEC/NAS100)**.

Dispatches alerts across terminal console, **Telegram Bot (HTML/Markdown)**, and **Webhooks (Discord/Slack/n8n/Zapier)** with built-in deduplication.

---

## 1. Project Architecture

```
forex/
├── config.py                 # Central config, Zuperior credentials, dynamic symbol mapping, session windows
├── indicators.py             # Pure NumPy/pandas vector indicators (EMA, SMA, ATR, ADX, Bollinger, Keltner, VWAP)
├── mt5_connector.py          # MT5 API wrapper, auto symbol resolver, contract specs, live ticks & OHLCV feeds
├── regime_filter.py          # 3-Stage gatekeeper pipeline (Live spread vs 1.8x median, session windows, MTF regime)
├── risk_engine.py            # Precision lot sizing, dynamic ATR stops, 1:2.5 min R:R, confluence gatekeeper
├── notifier.py               # Telegram, Webhook (Discord/Slack), and Console notifications with signal deduplication
├── backtester.py             # Historical backtest engine using Zuperior's actual historical spread column
├── main.py                   # Master engine CLI with --verify-symbols, --backtest, and live scanning loop
├── strategies/
│   ├── __init__.py
│   ├── base_strategy.py      # Abstract strategy interface & TradeSignal dataclass
│   ├── gold_strategy.py      # London/NY session liquidity sweep + pullback reversal
│   ├── btc_strategy.py       # Volatility squeeze breakout (BB inside KC) + volume spike confirmation
│   └── nasdaq_strategy.py    # US Cash Open (14:30 UTC) Opening Range Breakout (ORB) + VWAP alignment
├── .env.example              # Environment variables template
├── requirements.txt          # Production dependencies
└── README.md                 # Complete documentation
```

---

## 2. Strict 3-Stage Zuperior Gatekeeper Pipeline

Before any signal is emitted to Telegram, Webhooks, or Console, it must pass through 3 strict validation gatekeepers:

```
┌────────────────────────────────────────────────────────────────────────┐
│                   STAGE 1: SPREAD & SESSION FILTER                     │
│  • Rejects if Live Spread > 1.8 * Rolling Median Spread                │
│  • Rejects if Spread > Absolute Cap (Gold $0.60, BTC $50, NAS 3.5 pts) │
│  • Enforces Session Windows:                                           │
│    - Gold: 12:00 - 16:30 UTC (London / NY Morning Liquidity Overlap)   │
│    - Nasdaq: 14:30 - 17:00 UTC (US Cash Open & Power Hours)            │
│    - Bitcoin: 24/7 (Excluding low-volatility weekend chop)             │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │ Passed
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                   STAGE 2: MULTI-TIMEFRAME REGIME                      │
│  • Macro Bias: H1 50 EMA vs 200 EMA alignment                          │
│  • Micro Regime: M5/M15 ADX(14) & ATR(14):                             │
│    - Breakout (Nasdaq/BTC): ADX > 25.0 & Expanding ATR                 │
│    - Liquidity Sweep (Gold): ADX < 22.0 (Range fade) or exhaustion     │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │ Passed
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│              STAGE 3: CONFLUENCE & EXACT MT5 LOT SIZING                │
│  • Minimum 3+ Aligned Technical Confluences                            │
│  • Minimum 1:2.5 Risk-to-Reward Ratio Enforced                         │
│  • Exact Lot Sizing using Zuperior MT5 Equity, Tick Value & Step:      │
│      Risk_USD = Live_Equity * 1.0%                                     │
│      Risk_Per_Lot = (Price_Distance / trade_tick_size) * tick_value    │
│      Lot_Size = floor(Risk_USD / Risk_Per_Lot, volume_step)            │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │ Emitted
                                    ▼
                             TRADE ALERT DISPATCH
```

---

## 3. Strategies Overview

### 1. Gold Strategy (`strategies/gold_strategy.py`)
- **Asset:** `XAUUSD` / `GOLD` (Auto-resolved to `.pro`, `.raw`, `.ecn`, etc.)
- **Liquidity Window:** 12:00 – 16:30 UTC (London / New York overlap).
- **Core Mechanics:**
  - Establishes Asian Session (00:00 - 08:00 UTC) and Early London (08:00 - 12:00 UTC) Highs & Lows.
  - Detects liquidity sweeps: Price wicks beyond session extremes to hunt stops, then closes back inside the range with a rejection shadow (wick >= 40% of candle).
  - Enters on Market Structure Shift (MSS) / EMA pullback on M5.
  - Invalidation Stop Loss: Extreme sweep wick $\pm\ 0.5 \times \text{ATR}$.
  - Minimum TP1: 1:2.5 R:R; TP2: Opposite session extreme (1:3.5+ R:R).

### 2. Bitcoin Strategy (`strategies/btc_strategy.py`)
- **Asset:** `BTCUSD` / `BTCUSDT`
- **Liquidity Window:** 24/7 with weekend low-volatility chop filter.
- **Core Mechanics:**
  - Identifies volatility compression: Bollinger Bands (20, 2.0 std dev) squeeze inside Keltner Channels (20, 1.5 ATR mult).
  - Fires upon squeeze release: Bands expand outside Keltner Channel with positive/negative Squeeze Momentum.
  - Volume Confirmation: M15 tick volume $\ge 1.8\times$ 20-period volume SMA.
  - Macro Alignment: H1 50 EMA > 200 EMA for Buys, EMA50 < EMA200 for Sells.
  - Minimum TP1: 1:2.5 R:R; TP2: 1:4.0 R:R runner.

### 3. Nasdaq Strategy (`strategies/nasdaq_strategy.py`)
- **Asset:** `USTEC` / `NAS100` / `US100`
- **Liquidity Window:** 14:30 – 17:00 UTC (US Cash Open).
- **Core Mechanics:**
  - Computes the 15-minute Opening Range (14:30 – 14:45 UTC, 3 M5 candles) establishing `ORB_High` and `ORB_Low`.
  - Anchors intraday Volume-Weighted Average Price (VWAP) to 14:30 UTC Cash Open.
  - Bullish: M5 candle closes above `ORB_High`, Price > VWAP, M5 ADX > 25, H1 EMA50 > EMA200.
  - Bearish: M5 candle closes below `ORB_Low`, Price < VWAP, M5 ADX > 25, H1 EMA50 < EMA200.
  - Stop Loss: Opening range midpoint or `ORB_Level` $\pm\ 1.5 \times \text{ATR}$.
  - Minimum TP1: 1:2.5 R:R; TP2: 1:3.5+ R:R.

---

## 4. Setup & Installation

### Prerequisites
- Python 3.11+
- Zuperior Account (or use built-in Mock / Simulation mode if you haven't opened one yet)

### Understanding Zuperior vs. MT5 Accounts
> [!NOTE]
> **You do NOT need to register a separate account with MetaQuotes/MetaTrader.**
> Zuperior is the broker that provides your MT5 trading account directly:
> 1. Log in to your **Zuperior Client Dashboard** (web portal).
> 2. Navigate to **"Trading Accounts"** or **"Accounts"** → Click **"Open Account"** (Demo or Live).
> 3. Select **"MetaTrader 5 (MT5)"** as the platform, choose your leverage and account currency (USD).
> 4. Zuperior instantly generates your:
>    - **MT5 Login Number** (e.g. `1089542`)
>    - **MT5 Trader Password**
>    - **Server Name** (e.g. `Zuperior-Demo` or `Zuperior-Live`)
> 5. If you do not have an MT5 account yet, keep `USE_MOCK_MT5=true` in `.env` to run paper trading and strategy testing immediately with Zuperior specifications!

### 1. Clone & Install Dependencies
```bash
git clone https://github.com/your-username/zuperior-signal-engine.git
cd forex

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install requirements
pip install -r requirements.txt
```

### 2. Environment Configuration (.env)
A ready-to-run [`.env`](file:///home/cwd/ankush/agent/forex/.env) file is pre-packaged in the project. You can edit it or copy from [`.env.example`](file:///home/cwd/ankush/agent/forex/.env.example):
```bash
# Edit .env with your credentials or toggle USE_MOCK_MT5=true
nano .env
```
Key configuration settings in `.env`:
```env
# If you don't have an active MT5 terminal running, keep USE_MOCK_MT5=true
USE_MOCK_MT5=true

# When ready to connect live to your Zuperior MT5 terminal:
MT5_LOGIN=1089542
MT5_PASSWORD=YourZuperiorTradingPassword
MT5_SERVER=Zuperior-Demo
USE_MOCK_MT5=false

RISK_PERCENT=1.0
MIN_RR_RATIO=2.5

ENABLE_CONSOLE=true
ENABLE_TELEGRAM=false
ENABLE_WEBHOOK=false
```

---

## 5. MetaTrader 5 Terminal Configuration

To allow Python to communicate with your Zuperior MT5 terminal:

1. **Enable Algo Trading:** Click the **Algo Trading** button in the main MT5 top toolbar (it should display a green play icon).
2. **Configure EA & WebRequest Permissions:**
   - In MT5, navigate to: `Tools -> Options -> Expert Advisors`.
   - Check **"Allow algorithmic trading"**.
   - Check **"Allow DLL imports"**.
   - Check **"Allow WebRequest for listed URL"** and add:
     - `https://api.telegram.org`
     - Your Webhook URL (e.g. `https://discord.com`, `https://hooks.slack.com`)
3. **Symbol Suffixes on Zuperior:**
   - Zuperior accounts frequently append suffixes to tickers depending on account type:
     - **Pro Account:** `XAUUSD.pro`, `BTCUSD.pro`, `USTEC.pro`
     - **Raw Account:** `XAUUSD.raw`, `BTCUSD.raw`, `USTEC.raw`
     - **ECN Account:** `XAUUSD.ecn`, `BTCUSD.ecn`, `USTEC.ecn`
     - **Standard Account:** `XAUUSD`, `BTCUSD`, `USTEC` / `NAS100` / `US100`
   - The engine **auto-detects** these suffixes from MT5 Market Watch dynamically.

---

## 6. Linux & Wine Execution Notes

The official `MetaTrader5` Python package provides native C-bindings designed for Windows. If running on Linux:

### Option A: Running under Wine (Recommended for Linux MT5)
1. Install Wine and install the Windows 64-bit MT5 terminal:
   ```bash
   wine /path/to/zuperior5setup.exe
   ```
2. Install Windows Python 3.11 inside the Wine prefix:
   ```bash
   wine python-3.11.9-amd64.exe
   ```
3. Run the engine using Wine's Python:
   ```bash
   wine python main.py --verify-symbols
   wine python main.py
   ```

### Option B: Automatic Simulation Bridge for Linux Development
When running natively in a standard Linux environment where `MetaTrader5` is not installed, `mt5_connector.py` automatically initializes the **Zuperior Mock Bridge**:
- Accurately simulates Zuperior account profiles, contract specifications, tick values, spreads, and OHLCV bars.
- Allows running `--verify-symbols`, backtests, and unit tests seamlessly without crashing.

---

## 7. CLI Usage

### 1. Verify Symbols & Data Flow (`--verify-symbols`)
Connects to Zuperior MT5, confirms account equity, tests dynamic symbol detection for Gold, Bitcoin, and Nasdaq, validates contract sizes, tick values, volume steps, and confirms live data flow:
```bash
python main.py --verify-symbols
```
*Sample Output:*
```
╔══════════════════════════════════════════════════════════════════════════════╗
║           ZUPERIOR METATRADER 5: SYMBOL & DATA FLOW VERIFICATION             ║
╚══════════════════════════════════════════════════════════════════════════════╝

Zuperior Account Profile:
  • Login:      1089542
  • Server:     Zuperior-Demo (Zuperior Financial Services)
  • Currency:   USD | Leverage: 1:500
  • Balance:    $10,000.00
  • Live Equity:$10,000.00

Asset      Broker Symbol  Bid/Ask              Spread (Pts/$)     Contract   Tick Val   Min Step   Data Flow 
─────────────────────────────────────────────────────────────────────────────────────────────────────────
GOLD       XAUUSD.pro     2650.00 / 2650.16    16 pts ($0.16)     100        $1.00      0.01 (min 0.01) OK (100b)
BITCOIN    BTCUSD.pro     96500.00 / 96512.00  1200 pts ($12.00)  1          $0.01      0.01 (min 0.01) OK (100b)
NASDAQ     USTEC.pro      20850.00 / 20850.90  90 pts ($0.90)     20         $0.20      0.01 (min 0.01) OK (100b)
─────────────────────────────────────────────────────────────────────────────────────────────────────────

✔ ALL ASSET SYMBOLS, CONTRACT SPECIFICATIONS & DATA FLOW CONFIRMED.
```

### 2. Single Market Scan (`--once`)
Runs a single scan iteration across all 3 assets and outputs any passing signals:
```bash
python main.py --once
```

### 3. Historical Backtesting (`--backtest`)
Runs multi-timeframe backtesting on Zuperior's actual historical rates and spread data:
```bash
# Backtest Gold for 30 days
python backtester.py --asset GOLD --days 30

# Backtest Nasdaq for 14 days
python backtester.py --asset NASDAQ --days 14

# Backtest all assets
python main.py --backtest ALL --days 30
```
*Sample Institutional Performance Report:*
```
╔══════════════════════════════════════════════════════════════════════╗
║           ZUPERIOR MT5 STRATEGY BACKTEST RESULTS (30 DAYS)           ║
╚══════════════════════════════════════════════════════════════════════╝
Asset: NASDAQ | Symbol: USTEC.pro
──────────────────────────────────────────────────────────────────────
Total Trades:           24  (Wins: 17 | Losses: 7)
Win Rate:               70.83%
Profit Factor:          3.45
Net Profit ($):         $2,480.12
Ending Equity ($):      $12,480.12
Max Drawdown:           2.15% ($268.30)
Expectancy:             $103.34 / trade (1.28 R)
Sharpe Ratio:           2.84
──────────────────────────────────────────────────────────────────────
```

### 4. Continuous Live Scanning
Runs the live scanning loop at the configured interval (default: 30 seconds):
```bash
python main.py --interval 30
```

---

## 8. Alert Format Preview

### Telegram Alert (HTML)
```
🚨 ZUPERIOR MT5 SIGNAL ENGINE 🚨
━━━━━━━━━━━━━━━━━━━━━━━━
🎯 Asset: GOLD (XAUUSD.pro)
⚡ Action: 🟢 BUY (M5)
📊 Strategy: Gold London/NY Liquidity Sweep

💰 ENTRY: 2650.50
🛑 STOP LOSS: 2642.00 (Dist: 8.50)
🎯 TAKE PROFIT 1: 2671.75 (R:R 1:2.50)
🚀 TAKE PROFIT 2: 2680.25

⚖️ Zuperior Sizing & Risk:
• Recommended Lot Size: 0.11 Lots
• Risk: $100.00 (1.0% of $10,000.00)
• Live Spread: $0.16

🔍 Confluence Checklist (3+ Aligned):
• Session liquidity low swept (2643.20)
• M5 rejection wick / close back inside range (Wick: 52.4%)
• Pullback confirmation above EMA9 (2650.50 >= 2648.80)
• Peak London/NY liquidity window active (13:15 UTC)
• Zuperior live spread verified (0.16 <= 0.60)

🕒 Bar Timestamp: 2026-11-23 13:15:00 UTC
```

---

## 9. License & Disclaimer
This software is provided for educational and algorithmic research purposes only. Trading Forex, Cryptocurrencies, and CFDs carries a high level of risk to your capital and you should only trade with money you can afford to lose.
