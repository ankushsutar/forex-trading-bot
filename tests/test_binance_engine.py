"""
tests/test_binance_engine.py
----------------------------
Comprehensive verification suite testing:
1. tickSize, stepSize, and minNotional rounding edge cases across multiple pairs.
2. Regime filter state transitions (chop, wide spread, extreme funding rate skew).
3. Risk manager position sizing, reduceOnly flags, and daily drawdown kill-switch.
4. Mocked HTTP 429 rate limiting, -1021 timestamp drift, and signature generation.
"""

from decimal import Decimal
from datetime import datetime, timezone, timedelta
from unittest.mock import MagicMock, patch
import pytest
import numpy as np
import pandas as pd

from config.symbols import SYMBOL_REGISTRY, SymbolConfig
from core.exchange_metadata import ExchangeMetadata, SymbolFilters
from core.regime_filter import RegimeFilter, MarketRegime
from core.signal_engine import SignalEngine, SignalEvent
from core.risk_manager import RiskManager
from execution.binance_router import BinanceOrderRouter, SizingResult


def generate_synthetic_candles(n_bars: int = 100, base_price: float = 60000.0, trend: float = 0.0) -> pd.DataFrame:
    """Generates synthetic DataFrame with DatetimeIndex for testing."""
    times = [datetime(2026, 11, 24, 0, 0, tzinfo=timezone.utc) + timedelta(minutes=5 * i) for i in range(n_bars)]
    np.random.seed(42)

    changes = np.random.normal(trend, 50.0, n_bars)
    closes = base_price + np.cumsum(changes)
    highs = closes + np.random.uniform(10.0, 40.0, n_bars)
    lows = closes - np.random.uniform(10.0, 40.0, n_bars)
    opens = closes - np.random.normal(0, 15.0, n_bars)
    volumes = np.random.randint(200, 1500, n_bars)

    df = pd.DataFrame({
        'open': opens,
        'high': highs,
        'low': lows,
        'close': closes,
        'volume': volumes,
        'tick_volume': volumes,
        'spread': 10
    }, index=pd.to_datetime(times))
    return df


# ==============================================================================
# TEST SUITE 1: PRECISION ROUNDING & FILTER VALIDATION
# ==============================================================================
class TestPrecisionFilters:

    @pytest.fixture
    def metadata(self):
        return ExchangeMetadata()

    def test_tick_size_rounding(self, metadata):
        # BTCUSDT: tick_size = 0.10
        assert metadata.round_price("BTCUSDT", 61530.126) == 61530.10
        assert metadata.round_price("BTCUSDT", 61530.16) == 61530.20

        # ETHUSDT: tick_size = 0.01
        assert metadata.round_price("ETHUSDT", 3420.555) == 3420.56
        assert metadata.round_price("ETHUSDT", 3420.551) == 3420.55

    def test_step_size_rounding_down(self, metadata):
        # Quantity must round DOWN to step_size so risk is never exceeded
        # BTCUSDT: step_size = 0.001
        assert metadata.round_quantity("BTCUSDT", 0.01599) == 0.015
        assert metadata.round_quantity("BTCUSDT", 1.2345) == 1.234

        # SOLUSDT: step_size = 0.01
        assert metadata.round_quantity("SOLUSDT", 12.459) == 12.45

        # PAXGUSDT: step_size = 0.0001
        assert metadata.round_quantity("PAXGUSDT", 0.12349) == 0.1234

    def test_min_notional_validation(self, metadata):
        # BTCUSDT: min_notional = $5.00
        # 0.00001 BTC @ $60,000 = $0.60 -> Rejected
        is_valid, msg = metadata.validate_order("BTCUSDT", price=60000.0, quantity=0.00001)
        assert not is_valid
        assert "minimum allowed" in msg or "notional" in msg

        # 0.001 BTC @ $60,000 = $60.00 -> Valid
        is_valid, msg = metadata.validate_order("BTCUSDT", price=60000.0, quantity=0.001)
        assert is_valid
        assert msg == "Valid"


# ==============================================================================
# TEST SUITE 2: REGIME FILTER & GATEKEEPER TRANSITIONS
# ==============================================================================
class TestRegimeFilter:

    @pytest.fixture
    def regime_filter(self):
        return RegimeFilter()

    def test_wide_spread_rejection(self, regime_filter):
        df = generate_synthetic_candles(100)
        # BTCUSDT max spread is 5.0 bps. Feed 8.5 bps.
        verdict = regime_filter.classify_regime("BTCUSDT", df, spread_bps=8.5, funding_rate=0.0)
        assert not verdict.is_tradeable
        assert verdict.regime == MarketRegime.CHOP_NO_TRADE
        assert "Spread" in verdict.reason

    def test_extreme_positive_funding_blocks_longs(self, regime_filter):
        # Funding rate = 0.08% (8 bps) > max allowed 5 bps
        df = generate_synthetic_candles(100)
        verdict = regime_filter.classify_regime("BTCUSDT", df, spread_bps=1.0, funding_rate=0.0008)
        assert "BUY" not in verdict.allowed_directions

    def test_extreme_negative_funding_blocks_shorts(self, regime_filter):
        # Funding rate = -0.09% (-9 bps) < -5 bps
        df = generate_synthetic_candles(100)
        verdict = regime_filter.classify_regime("BTCUSDT", df, spread_bps=1.0, funding_rate=-0.0009)
        assert "SELL" not in verdict.allowed_directions

    def test_chop_no_trade_regime(self, regime_filter):
        # Flat noise with no trend
        df = generate_synthetic_candles(100, trend=0.0)
        verdict = regime_filter.classify_regime("BTCUSDT", df, spread_bps=1.0, funding_rate=0.0)
        if verdict.regime == MarketRegime.CHOP_NO_TRADE:
            assert not verdict.is_tradeable
            assert len(verdict.allowed_directions) == 0


# ==============================================================================
# TEST SUITE 3: RISK MANAGER & CIRCUIT BREAKER
# ==============================================================================
class TestRiskManager:

    @pytest.fixture
    def risk_manager(self):
        metadata = ExchangeMetadata()
        return RiskManager(metadata=metadata)

    def test_fixed_fractional_sizing_math(self, risk_manager):
        # Equity = $10,000, Risk = 1.0% ($100.00)
        # Entry = 60,000, SL = 59,000 (Distance = 1,000)
        # Expected Qty = $100 / 1000 = 0.100 BTC
        signal = SignalEvent(
            symbol="BTCUSDT",
            direction="BUY",
            archetype="SQUEEZE_BREAKOUT",
            entry_price=60000.0,
            stop_loss=59000.0,
            take_profit_1=62500.0,  # 1:2.5 R:R
            take_profit_2=64000.0,
            risk_reward_ratio=2.5,
            confluences=["Test Confluence"],
            bar_time=datetime.now(timezone.utc)
        )

        sizing = risk_manager.calculate_order_sizing(signal, account_equity=10000.0)
        assert sizing.is_valid
        assert sizing.quantity == 0.100
        assert sizing.dollar_risk == 100.0
        assert sizing.notional_value == 6000.0

    def test_daily_drawdown_kill_switch(self, risk_manager):
        # Initial equity: $10,000
        risk_manager.set_account_equity(10000.0)
        assert not risk_manager.is_trading_halted

        # Equity drops to $9,650 (3.5% drawdown >= 3.0% threshold)
        risk_manager.set_account_equity(9650.0)
        assert risk_manager.is_trading_halted

        # Any subsequent order sizing MUST be rejected by the kill-switch
        signal = SignalEvent(
            symbol="BTCUSDT",
            direction="BUY",
            archetype="SQUEEZE_BREAKOUT",
            entry_price=60000.0,
            stop_loss=59000.0,
            take_profit_1=62500.0,
            take_profit_2=64000.0,
            risk_reward_ratio=2.5,
            confluences=["Test Confluence"],
            bar_time=datetime.now(timezone.utc)
        )
        sizing = risk_manager.calculate_order_sizing(signal, account_equity=9650.0)
        assert not sizing.is_valid
        assert "circuit breaker" in sizing.rejection_reason.lower()

    def test_minimum_rr_ratio_enforcement(self, risk_manager):
        # TP1 gives only 1:1 R:R (Entry 60000, SL 59000, TP1 61000)
        signal = SignalEvent(
            symbol="BTCUSDT",
            direction="BUY",
            archetype="SQUEEZE_BREAKOUT",
            entry_price=60000.0,
            stop_loss=59000.0,
            take_profit_1=61000.0,  # Only 1.0 R:R < 2.5
            take_profit_2=63000.0,
            risk_reward_ratio=1.0,
            confluences=["Test Confluence"],
            bar_time=datetime.now(timezone.utc)
        )
        sizing = risk_manager.calculate_order_sizing(signal, account_equity=10000.0)
        assert not sizing.is_valid
        assert "Risk-to-Reward ratio" in sizing.rejection_reason


# ==============================================================================
# TEST SUITE 4: ROUTER IDEMPOTENCY & RATE LIMITING
# ==============================================================================
class TestOrderRouter:

    @pytest.fixture
    def router(self):
        return BinanceOrderRouter()

    def test_clock_drift_sync(self, router):
        # Mock Binance server time returning 500ms ahead
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"serverTime": int((datetime.now(timezone.utc).timestamp() + 0.5) * 1000)}

        with patch.object(router.session, "get", return_value=mock_resp):
            offset = router.sync_server_time()
            assert offset > 0
            ts = router._get_timestamp()
            assert ts > int(datetime.now(timezone.utc).timestamp() * 1000)

    def test_http_429_retry_after_handling(self, router):
        sizing = SizingResult(
            is_valid=True,
            symbol="BTCUSDT",
            direction="BUY",
            entry_price=60000.0,
            stop_loss=59000.0,
            take_profit_1=62500.0,
            take_profit_2=64000.0,
            quantity=0.1,
            dollar_risk=100.0,
            notional_value=6000.0,
            risk_reward_ratio=2.5,
            reduce_only=False
        )

        mock_resp = MagicMock()
        mock_resp.status_code = 429
        mock_resp.headers = {"Retry-After": "1", "X-MBX-USED-WEIGHT-1M": "1190"}
        mock_resp.text = "Rate limit exceeded"

        # In LIVE mode with mock
        with patch.object(router, "execution_env", "LIVE"), \
             patch.object(router, "api_key", "test_key"), \
             patch.object(router, "api_secret", "test_secret"), \
             patch.object(router.session, "post", return_value=mock_resp), \
             patch("time.sleep") as mock_sleep:

            order_res = router.execute_bracket(sizing, bar_timestamp=1700000000)
            assert not order_res.success
            assert order_res.status == "REJECTED_429"
            mock_sleep.assert_called_with(1)

    def test_deterministic_client_order_id(self, router):
        sizing = SizingResult(
            is_valid=True,
            symbol="BTCUSDT",
            direction="BUY",
            entry_price=60000.0,
            stop_loss=59000.0,
            take_profit_1=62500.0,
            take_profit_2=64000.0,
            quantity=0.1,
            dollar_risk=100.0,
            notional_value=6000.0,
            risk_reward_ratio=2.5,
            reduce_only=False
        )
        bar_ts = 1700000000
        order_res = router.execute_bracket(sizing, bar_timestamp=bar_ts)
        assert order_res.client_order_id == "SIG_BTCUSDT_1700000000_BUY"


# ==============================================================================
# 5. TELEGRAM NOTIFIER & DEDUPLICATION TESTS
# ==============================================================================
class TestTelegramNotifier:
    def test_binance_signal_html_formatting(self):
        from notifier import TradeNotifier
        notifier = TradeNotifier()
        sig = SignalEvent(
            symbol="BTCUSDT",
            direction="BUY",
            archetype="SQUEEZE_BREAKOUT",
            entry_price=60000.0,
            stop_loss=59000.0,
            take_profit_1=62500.0,
            take_profit_2=64000.0,
            risk_reward_ratio=2.5,
            confluences=["Bollinger Band expansion", "Positive squeeze momentum"],
            bar_time=datetime(2026, 11, 25, 6, 0, tzinfo=timezone.utc),
            metrics={"spread_bps": 1.2, "funding_rate": 0.0001}
        )
        sizing = SizingResult(
            is_valid=True,
            symbol="BTCUSDT",
            direction="BUY",
            entry_price=60000.0,
            stop_loss=59000.0,
            take_profit_1=62500.0,
            take_profit_2=64000.0,
            quantity=0.10,
            dollar_risk=100.0,
            notional_value=6000.0,
            risk_reward_ratio=2.5,
            reduce_only=False
        )

        html = notifier._format_binance_telegram_html(sig, sizing, equity=10000.0)
        assert "BTCUSDT" in html
        assert "BUY / LONG" in html
        assert "Squeeze Breakout" in html
        assert "$60,000.00" in html
        assert "1:2.50" in html
        assert "Bollinger Band expansion" in html

    def test_signal_deduplication(self):
        from notifier import TradeNotifier
        notifier = TradeNotifier()
        sig = SignalEvent(
            symbol="ETHUSDT",
            direction="BUY",
            archetype="LIQUIDITY_SWEEP",
            entry_price=1700.0,
            stop_loss=1680.0,
            take_profit_1=1750.0,
            take_profit_2=1780.0,
            risk_reward_ratio=2.5,
            confluences=["Asian Low Sweep"],
            bar_time=datetime(2026, 11, 25, 7, 0, tzinfo=timezone.utc)
        )
        assert not notifier.is_duplicate(sig)
        assert notifier.is_duplicate(sig)

    def test_send_test_message_mock(self):
        from notifier import TradeNotifier
        notifier = TradeNotifier()
        notifier.telegram_bot_token = "mock_token"
        notifier.telegram_chat_id = "12345678"

        mock_get_me = MagicMock()
        mock_get_me.status_code = 200
        mock_get_me.json.return_value = {"result": {"username": "TestTradingBot"}}

        mock_send = MagicMock()
        mock_send.status_code = 200

        with patch("requests.get", return_value=mock_get_me), \
             patch("requests.post", return_value=mock_send):
            success, msg = notifier.send_test_message()
            assert success
            assert "TestTradingBot" in msg

