"""
indicators.py
-------------
High-performance, standalone technical indicators implemented in pure NumPy and pandas.
Provides EMA, SMA, ATR (Wilder's), ADX (+DI, -DI, DX), Bollinger Bands, Keltner Channels,
Squeeze Momentum, Session-Anchored VWAP, and Rolling Spread statistics.
"""

from typing import Tuple, Optional
import numpy as np
import pandas as pd


def calculate_sma(series: pd.Series, period: int) -> pd.Series:
    """Calculate Simple Moving Average."""
    return series.rolling(window=period, min_periods=period).mean()


def calculate_ema(series: pd.Series, period: int) -> pd.Series:
    """Calculate Exponential Moving Average with exponential weighting."""
    return series.ewm(span=period, adjust=False).mean()


def _wilder_rma(series: pd.Series, period: int) -> pd.Series:
    """Wilder's Running Moving Average (RMA), equivalent to ewm(alpha=1/period)."""
    return series.ewm(alpha=1.0 / period, adjust=False).mean()


def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Calculate Average True Range (ATR) using Wilder's smoothing method.
    Requires 'high', 'low', 'close' columns.
    """
    high = df['high']
    low = df['low']
    prev_close = df['close'].shift(1)

    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()

    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = _wilder_rma(tr, period)
    return atr


def calculate_adx(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """
    Calculate Average Directional Index (ADX) along with +DI and -DI.
    Uses Wilder's smoothing technique.
    Returns DataFrame with columns: ['adx', 'plus_di', 'minus_di'].
    """
    high = df['high']
    low = df['low']
    prev_high = high.shift(1)
    prev_low = low.shift(1)
    prev_close = df['close'].shift(1)

    # True Range
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    # Directional Movement
    up_move = high - prev_high
    down_move = prev_low - low

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    plus_dm_series = pd.Series(plus_dm, index=df.index)
    minus_dm_series = pd.Series(minus_dm, index=df.index)

    # Smoothed TR and DM using Wilder's RMA
    smoothed_tr = _wilder_rma(tr, period)
    smoothed_plus_dm = _wilder_rma(plus_dm_series, period)
    smoothed_minus_dm = _wilder_rma(minus_dm_series, period)

    # Prevent division by zero
    safe_tr = smoothed_tr.replace(0, np.nan)

    plus_di = 100.0 * (smoothed_plus_dm / safe_tr)
    minus_di = 100.0 * (smoothed_minus_dm / safe_tr)

    di_sum = (plus_di + minus_di).replace(0, np.nan)
    dx = 100.0 * ((plus_di - minus_di).abs() / di_sum)
    adx = _wilder_rma(dx, period)

    return pd.DataFrame({
        'adx': adx.fillna(0.0),
        'plus_di': plus_di.fillna(0.0),
        'minus_di': minus_di.fillna(0.0)
    }, index=df.index)


def calculate_bollinger_bands(
    series: pd.Series,
    period: int = 20,
    std_dev: float = 2.0
) -> pd.DataFrame:
    """
    Calculate Bollinger Bands: Mid (SMA), Upper, Lower, and Bandwidth.
    """
    mid = series.rolling(window=period, min_periods=period).mean()
    std = series.rolling(window=period, min_periods=period).std(ddof=0)

    upper = mid + (std_dev * std)
    lower = mid - (std_dev * std)
    bandwidth = (upper - lower) / mid.replace(0, np.nan)

    return pd.DataFrame({
        'bb_mid': mid,
        'bb_upper': upper,
        'bb_lower': lower,
        'bb_width': bandwidth
    }, index=series.index)


def calculate_keltner_channels(
    df: pd.DataFrame,
    period: int = 20,
    atr_multiplier: float = 1.5,
    atr_period: int = 14
) -> pd.DataFrame:
    """
    Calculate Keltner Channels: Mid (EMA), Upper, and Lower based on ATR.
    """
    mid = calculate_ema(df['close'], period)
    atr = calculate_atr(df, period=atr_period)

    upper = mid + (atr_multiplier * atr)
    lower = mid - (atr_multiplier * atr)

    return pd.DataFrame({
        'kc_mid': mid,
        'kc_upper': upper,
        'kc_lower': lower
    }, index=df.index)


def detect_squeeze(
    df: pd.DataFrame,
    bb_period: int = 20,
    bb_std: float = 2.0,
    kc_period: int = 20,
    kc_mult: float = 1.5
) -> pd.Series:
    """
    Detect John Carter's Volatility Squeeze (Bollinger Bands inside Keltner Channel).
    Returns boolean Series where True indicates market is in volatility compression.
    """
    bb = calculate_bollinger_bands(df['close'], period=bb_period, std_dev=bb_std)
    kc = calculate_keltner_channels(df, period=kc_period, atr_multiplier=kc_mult)

    is_squeeze = (bb['bb_upper'] < kc['kc_upper']) & (bb['bb_lower'] > kc['kc_lower'])
    return is_squeeze


def calculate_squeeze_momentum(
    df: pd.DataFrame,
    bb_period: int = 20,
    kc_period: int = 20
) -> pd.Series:
    """
    Calculate LazyBear Squeeze Momentum oscillator.
    Computes linear regression slope of (Close - Midline).
    Midline = (Donchian Midpoint + 20-SMA) / 2.
    """
    close = df['close']
    highest_high = df['high'].rolling(window=kc_period).max()
    lowest_low = df['low'].rolling(window=kc_period).min()
    donchian_mid = (highest_high + lowest_low) / 2.0
    sma_mid = close.rolling(window=kc_period).mean()

    delta = close - ((donchian_mid + sma_mid) / 2.0)

    # Linear regression slope over kc_period
    def _linreg_slope(y: np.ndarray) -> float:
        if len(y) < kc_period or np.isnan(y).any():
            return 0.0
        x = np.arange(len(y))
        x_mean = x.mean()
        y_mean = y.mean()
        denominator = np.sum((x - x_mean) ** 2)
        if denominator == 0:
            return 0.0
        slope = np.sum((x - x_mean) * (y - y_mean)) / denominator
        # Value at end of line:
        intercept = y_mean - slope * x_mean
        return float(intercept + slope * (len(y) - 1))

    momentum = delta.rolling(window=kc_period).apply(_linreg_slope, raw=True)
    return momentum.fillna(0.0)


def calculate_vwap(
    df: pd.DataFrame,
    anchor_hour: int = 14,
    anchor_minute: int = 30
) -> pd.Series:
    """
    Calculate intraday Volume-Weighted Average Price (VWAP) anchored to a specific session open.
    Default anchor: 14:30 UTC (US Cash Open).
    If no anchor timestamp matches within a day, falls back to daily session open (00:00 UTC).
    """
    volume = df['tick_volume'] if 'tick_volume' in df.columns else df['volume']
    typical_price = (df['high'] + df['low'] + df['close']) / 3.0
    pv = typical_price * volume

    # Generate session groups based on anchor
    # Identifies each new session start
    times = df.index
    is_anchor = (times.hour == anchor_hour) & (times.minute == anchor_minute)

    # Also reset at start of new day if anchor hasn't occurred yet
    is_new_day = (times.day != pd.Series(times.day, index=times).shift(1))
    reset_trigger = is_anchor | (is_new_day & (times.hour < anchor_hour))

    # Group ID increments whenever reset_trigger is True
    group_id = reset_trigger.cumsum()

    # Calculate cumulative sums per group
    cum_pv = pv.groupby(group_id).cumsum()
    cum_vol = volume.groupby(group_id).cumsum()

    safe_vol = cum_vol.replace(0, np.nan)
    vwap = cum_pv / safe_vol
    return vwap.fillna(typical_price)


def calculate_rolling_spread_median(df: pd.DataFrame, window: int = 50) -> pd.Series:
    """
    Calculate rolling median spread from MT5 historical spread column.
    """
    if 'spread' not in df.columns:
        return pd.Series(0.0, index=df.index)
    return df['spread'].rolling(window=window, min_periods=5).median().bfill()


def find_swing_levels(df: pd.DataFrame, window: int = 5) -> Tuple[pd.Series, pd.Series]:
    """
    Identify fractal swing highs and swing lows over a rolling window.
    A swing high is highest among window bars before and after.
    A swing low is lowest among window bars before and after.
    """
    high = df['high']
    low = df['low']

    is_swing_high = pd.Series(False, index=df.index)
    is_swing_low = pd.Series(False, index=df.index)

    for i in range(window, len(df) - window):
        center_high = high.iloc[i]
        center_low = low.iloc[i]

        left_highs = high.iloc[i - window:i]
        right_highs = high.iloc[i + 1:i + window + 1]

        left_lows = low.iloc[i - window:i]
        right_lows = low.iloc[i + 1:i + window + 1]

        if (center_high > left_highs.max()) and (center_high >= right_highs.max()):
            is_swing_high.iloc[i] = True

        if (center_low < left_lows.min()) and (center_low <= right_lows.min()):
            is_swing_low.iloc[i] = True

    return is_swing_high, is_swing_low


def get_session_high_low(
    df: pd.DataFrame,
    start_hour: int,
    start_minute: int,
    end_hour: int,
    end_minute: int
) -> Tuple[Optional[float], Optional[float]]:
    """
    Retrieve high and low prices for a specific intraday time window.
    Useful for extracting Asian session (00:00 - 08:00 UTC) or London morning levels.
    """
    times = df.index
    # Filter bars within the time window
    time_of_day = times.hour * 60 + times.minute
    start_val = start_hour * 60 + start_minute
    end_val = end_hour * 60 + end_minute

    if start_val < end_val:
        mask = (time_of_day >= start_val) & (time_of_day < end_val)
    else:  # Overnight session wrapping midnight
        mask = (time_of_day >= start_val) | (time_of_day < end_val)

    # Restrict to the most recent completed day session
    session_bars = df[mask]
    if session_bars.empty:
        return None, None

    # Get bars from the most recent day in the filtered set
    latest_date = session_bars.index[-1].date()
    current_day_session = session_bars[session_bars.index.date == latest_date]

    if current_day_session.empty:
        return None, None

    return float(current_day_session['high'].max()), float(current_day_session['low'].min())
