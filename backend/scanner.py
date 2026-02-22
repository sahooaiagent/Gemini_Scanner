"""
Gemini Scanner — AMA PRO TEMA + Qwen Logic
============================================
Faithful Python port of the AMA_PRO_TEMA and MA Qwen Pine Script indicators.
Uses Triple Exponential Moving Averages (TEMA) with adaptive parameters
based on market regime detection.

Signal: Check if longValid or shortValid occurred on the PREVIOUS closed candle.
"""

import yfinance as yf
import pandas as pd
import numpy as np
import time
import datetime
import logging
from concurrent.futures import ThreadPoolExecutor

# Shared thread pool for CPU intensive calculations
executor = ThreadPoolExecutor(max_workers=4)

# Map UI index names to yfinance tickers
TICKER_MAP = {
    "NIFTY": "^NSEI",
    "NIFTY 50": "^NSEI",
    "BANKNIFTY": "^NSEBANK",
    "BANK NIFTY": "^NSEBANK",
    "DOW JONES": "^DJI",
    "NASDAQ": "^IXIC"
}

# =============================================================================
# TECHNICAL INDICATOR FUNCTIONS
# =============================================================================

def calculate_ema(series, length):
    """Standard EMA"""
    return series.ewm(span=length, adjust=False).mean()

def calculate_tema(series, length):
    """
    Triple Exponential Moving Average (TEMA)
    TEMA = 3*EMA1 - 3*EMA2 + EMA3
    Reduces lag compared to standard EMA.
    """
    ema1 = series.ewm(span=length, adjust=False).mean()
    ema2 = ema1.ewm(span=length, adjust=False).mean()
    ema3 = ema2.ewm(span=length, adjust=False).mean()
    return 3 * ema1 - 3 * ema2 + ema3

def calculate_atr(high, low, close, length):
    """Average True Range using RMA (Wilder's smoothing)"""
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.ewm(alpha=1/length, adjust=False).mean()

def calculate_adx(high, low, close, length):
    """ADX matching Pine Script calcADX"""
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/length, adjust=False).mean()

    up = high.diff()
    down = -low.diff()

    plus_dm = np.where((up > down) & (up > 0), up, 0.0)
    minus_dm = np.where((down > up) & (down > 0), down, 0.0)

    plus_dm_s = pd.Series(plus_dm, index=high.index).ewm(alpha=1/length, adjust=False).mean()
    minus_dm_s = pd.Series(minus_dm, index=high.index).ewm(alpha=1/length, adjust=False).mean()

    plus_di = 100 * (plus_dm_s / atr.replace(0, np.nan)).fillna(0)
    minus_di = 100 * (minus_dm_s / atr.replace(0, np.nan)).fillna(0)

    di_sum = plus_di + minus_di
    di_sum = di_sum.replace(0, 1e-10)
    dx = 100 * abs(plus_di - minus_di) / di_sum
    return dx.ewm(alpha=1/length, adjust=False).mean()

def calculate_rsi(series, length=14):
    """RSI matching Pine Script ta.rsi"""
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1/length, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/length, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return (100 - (100 / (1 + rs))).fillna(50)

def calculate_vwap(df):
    """
    Session VWAP matching TradingView ta.vwap.
    Resets at UTC midnight (00:00) each day.
    """
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    dates = df.index.date
    vwap = pd.Series(np.nan, index=df.index)

    cum_tp_vol = 0.0
    cum_vol = 0.0
    prev_date = None

    for i in range(len(df)):
        current_date = dates[i]
        if current_date != prev_date:
            cum_tp_vol = 0.0
            cum_vol = 0.0
            prev_date = current_date

        cum_tp_vol += typical_price.iloc[i] * df['volume'].iloc[i]
        cum_vol += df['volume'].iloc[i]

        if cum_vol > 0:
            vwap.iloc[i] = cum_tp_vol / cum_vol

    return vwap

# =============================================================================
# DATA FETCHING (with retry logic for JSONDecodeError)
# =============================================================================

def fetch_yfinance_data(ticker, tf_input, retries=3):
    """
    Fetches data using yfinance with retry logic.
    Maps timeframes to yfinance intervals, resamples when needed.
    """
    # Clean up input string
    tf_clean = tf_input.lower().strip()

    # Explicit mapping for reliability
    mapping = {
        '5min': ('5m', '60d', None),
        '10min': ('5m', '60d', '10min'),
        '15min': ('15m', '60d', None),
        '30min': ('30m', '60d', None),
        '45min': ('15m', '60d', '45min'),
        '1hr': ('1h', '730d', None),
        '2hr': ('1h', '730d', '2h'),
        '4hr': ('1h', '730d', '4h'),
        '8hr': ('1h', '730d', '8h'),
        '12hr': ('1h', '730d', '12h'),
        '1 day': ('1d', '5y', None),
        '2 day': ('1d', '5y', '2d'),
        '1 week': ('1wk', 'max', None)
    }

    yf_interval, period, resample_freq = mapping.get(tf_clean, ('15m', '60d', None))

    for attempt in range(1, retries + 1):
        try:
            logging.info(f"Fetching data for {ticker} at {yf_interval} interval (attempt {attempt})...")
            data = yf.download(ticker, period=period, interval=yf_interval, progress=False)

            if data is None or data.empty:
                logging.warning(f"No data returned for {ticker} at {tf_input}.")
                return None

            logging.info(f"Raw data head: {data.shape} | Columns: {list(data.columns)}")

            # Clean up yfinance multi-index columns
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = [col[0].lower() for col in data.columns]
            else:
                data.columns = [c.lower() for c in data.columns]

            logging.info(f"Cleaned columns: {list(data.columns)}")

            if resample_freq:
                logging.info(f"Resampling to {resample_freq} for {ticker}...")
                data = data.resample(resample_freq).agg({
                    'open': 'first', 'high': 'max',
                    'low': 'min', 'close': 'last',
                    'volume': 'sum'
                }).dropna()

            final_df = data.tail(500)
            logging.info(f"Final DF length for {ticker} {tf_input}: {len(final_df)}")
            return final_df

        except Exception as e:
            logging.error(f"Attempt {attempt} failed for {ticker}: {str(e)}")
            if attempt < retries:
                wait_time = attempt * 2
                logging.info(f"Retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                logging.error(f"All {retries} attempts failed for {ticker} at {tf_input}.")
                return None

# =============================================================================
# AMA PRO TEMA LOGIC — Faithful Port of Pine Script
# =============================================================================

def apply_ama_pro_tema(df, tf_input="1 day", **kwargs):
    """
    Applies the full AMA PRO TEMA logic:
    1. Market regime detection (ADX, volatility ratio, EMA alignment)
    2. Adaptive TEMA period calculation (with tfMultiplier)
    3. TEMA crossover signals
    4. Signal filtering: longValid / shortValid (min bars between + regime conflict)
    5. Check the PREVIOUS closed candle (index -2) for a valid signal

    Returns: (signal, crossover_angle, tema_gap_pct) or (None, None, None)
    """
    if df is None or len(df) < 200:
        return None, None, None

    # Drop the last row (current forming/incomplete candle) so all logic
    # runs exclusively on closed candles. After this, df.iloc[-1] is the
    # latest CLOSED candle.
    df = df.iloc[:-1].copy()

    if len(df) < 200:
        return None, None, None

    try:
        # === PINE SCRIPT PARAMETERS ===
        i_emaFastMin, i_emaFastMax = 8, 21
        i_emaSlowMin, i_emaSlowMax = 21, 55
        i_adxLength = 14
        i_adxThreshold = 25
        i_volLookback = 50

        # User defined parameters
        i_minBarsBetween = kwargs.get('min_bars_between', 3)
        adaptation_speed = kwargs.get('adaptation_speed', 'Medium')

        # Sensitivity multiplier: High=1.5, Medium=1.0, Low=0.5 (as per Pine Script)
        sensitivity_mult = 1.5 if adaptation_speed == 'High' else 0.5 if adaptation_speed == 'Low' else 1.0

        # =================================================================
        # SECTION 3: MARKET REGIME DETECTION
        # =================================================================
        df['ADX'] = calculate_adx(df['high'], df['low'], df['close'], i_adxLength)

        # Volatility
        df['ATR'] = calculate_atr(df['high'], df['low'], df['close'], 14)
        df['returns'] = np.log(df['close'] / df['close'].shift(1))
        df['volatility'] = df['returns'].rolling(window=i_volLookback).std(ddof=0) * np.sqrt(252) * 100
        df['hist_vol'] = df['volatility'].rolling(window=i_volLookback).mean()
        df['vol_ratio'] = (df['volatility'] / df['hist_vol'].replace(0, np.nan)).fillna(1.0)

        # Trend alignment
        df['ema20'] = calculate_ema(df['close'], 20)
        df['ema50'] = calculate_ema(df['close'], 50)
        df['ema200'] = calculate_ema(df['close'], 200)

        # Rate of change for momentum
        df['roc10'] = df['close'].pct_change(10) * 100
        df['roc20'] = df['close'].pct_change(20) * 100
        df['momentum'] = (df['roc10'] + df['roc20']) / 2

        # Regime classification
        df['volRegime'] = np.select(
            [df['vol_ratio'] > 1.3, df['vol_ratio'] < 0.7],
            ['High', 'Low'], default='Normal'
        )
        df['trendRegime'] = np.where(df['ADX'] > i_adxThreshold, 'Trending', 'Ranging')

        trend_up = (df['close'] > df['ema20']) & (df['ema20'] > df['ema50']) & (df['ema50'] > df['ema200'])
        trend_down = (df['close'] < df['ema20']) & (df['ema20'] < df['ema50']) & (df['ema50'] < df['ema200'])
        df['trendAlignment'] = np.select([trend_up, trend_down], [1, -1], default=0)
        df['directionRegime'] = np.select([trend_up, trend_down], ['Bullish', 'Bearish'], default='Neutral')

        # Stable regime with confirmation counter (matching Pine Script exactly)
        # The regime only switches after i_regimeStability (3) consecutive bars
        # of a new regime, preventing noisy flip-flopping.
        i_regimeStability = 3
        stable_regime = "Neutral-Normal-Ranging"
        regime_counter = 0

        n_rows = len(df)
        stable_bullish = np.zeros(n_rows, dtype=bool)
        stable_bearish = np.zeros(n_rows, dtype=bool)
        stable_high_vol = np.zeros(n_rows, dtype=bool)
        stable_low_vol = np.zeros(n_rows, dtype=bool)
        stable_trending = np.zeros(n_rows, dtype=bool)
        stable_ranging = np.zeros(n_rows, dtype=bool)

        dir_vals = df['directionRegime'].values
        vol_vals = df['volRegime'].values
        trend_vals = df['trendRegime'].values

        for i in range(n_rows):
            current_regime = f"{dir_vals[i]}-{vol_vals[i]}-{trend_vals[i]}"
            if current_regime != stable_regime:
                regime_counter += 1
                if regime_counter >= i_regimeStability:
                    stable_regime = current_regime
                    regime_counter = 0
            else:
                regime_counter = 0

            stable_bullish[i] = "Bullish" in stable_regime
            stable_bearish[i] = "Bearish" in stable_regime
            stable_high_vol[i] = "High" in stable_regime
            stable_low_vol[i] = "Low" in stable_regime
            stable_trending[i] = "Trending" in stable_regime
            stable_ranging[i] = "Ranging" in stable_regime

        df['regimeIsBullish'] = stable_bullish
        df['regimeIsBearish'] = stable_bearish
        df['regimeIsHighVol'] = stable_high_vol
        df['regimeIsLowVol'] = stable_low_vol
        df['regimeIsTrending'] = stable_trending
        df['regimeIsRanging'] = stable_ranging

        # =================================================================
        # SECTION 4: ADAPTIVE PARAMETERS — TEMA VERSION
        # =================================================================
        vol_adjust = np.select(
            [df['regimeIsHighVol'], df['regimeIsLowVol']],
            [0.7, 1.3], default=1.0
        )
        trend_adjust = np.where(df['regimeIsTrending'], 0.8, 1.2)
        # tfMultiplier matching Pine Script:
        # timeframe.isintraday: multiplier<=5 -> 0.8, <=60 -> 1.0, >60 -> 1.2
        # timeframe.isdaily: 1.3, weekly+: 1.5
        tf_clean = tf_input.lower().strip()
        if 'min' in tf_clean:
            try:
                m = int(tf_clean.replace('min', ''))
                tf_multiplier = 0.8 if m <= 5 else 1.0 if m <= 60 else 1.2
            except: tf_multiplier = 1.0
        elif 'hr' in tf_clean:
            try:
                h = int(tf_clean.replace('hr', ''))
                minutes = h * 60
                tf_multiplier = 1.0 if minutes <= 60 else 1.2
            except: tf_multiplier = 1.0
        elif 'day' in tf_clean:
            tf_multiplier = 1.3
        elif 'week' in tf_clean:
            tf_multiplier = 1.5
        else:
            tf_multiplier = 1.0

        combined_adjust = vol_adjust * trend_adjust * tf_multiplier * sensitivity_mult
        adjust_factor = np.clip(1.0 / combined_adjust, 0.5, 1.5)

        fast_range = i_emaFastMax - i_emaFastMin
        slow_range = i_emaSlowMax - i_emaSlowMin

        adaptive_fast = i_emaFastMin + fast_range * (1 - adjust_factor)
        adaptive_slow = i_emaSlowMin + slow_range * (1 - adjust_factor)

        # Ensure minimum separation of 6 (TEMA needs slightly larger)
        adaptive_slow = np.maximum(adaptive_slow, adaptive_fast + 6)

        # =================================================================
        # PRE-CALCULATE TEMA VALUES (matching Pine Script periods)
        # =================================================================
        temas = {p: calculate_tema(df['close'], p) for p in [8, 10, 12, 14, 16, 18, 21, 26, 30, 34, 38, 42, 47, 55]}

        df['temaFast'] = np.select(
            [adaptive_fast <= 9, adaptive_fast <= 11, adaptive_fast <= 13,
             adaptive_fast <= 15, adaptive_fast <= 17, adaptive_fast <= 19],
            [temas[8], temas[10], temas[12], temas[14], temas[16], temas[18]],
            default=temas[21]
        )

        df['temaSlow'] = np.select(
            [adaptive_slow <= 28, adaptive_slow <= 32, adaptive_slow <= 36,
             adaptive_slow <= 40, adaptive_slow <= 44, adaptive_slow <= 51],
            [temas[26], temas[30], temas[34], temas[38], temas[42], temas[47]],
            default=temas[55]
        )

        # =================================================================
        # SECTION 5: STRATEGY LOGIC — TEMA crossovers
        # =================================================================
        df['longCondition'] = (df['temaFast'] > df['temaSlow']) & (df['temaFast'].shift(1) <= df['temaSlow'].shift(1))
        df['shortCondition'] = (df['temaFast'] < df['temaSlow']) & (df['temaFast'].shift(1) >= df['temaSlow'].shift(1))

        # =================================================================
        # SECTION 6: SIGNAL FILTERING — longValid / shortValid
        # =================================================================
        # Track bars since last long/short condition (iterative, matching Pine Script)
        n = len(df)
        bars_since_long = np.full(n, 999, dtype=int)
        bars_since_short = np.full(n, 999, dtype=int)
        long_valid = np.zeros(n, dtype=bool)
        short_valid = np.zeros(n, dtype=bool)

        long_cond = df['longCondition'].values
        short_cond = df['shortCondition'].values
        is_bullish = df['regimeIsBullish'].values
        is_bearish = df['regimeIsBearish'].values
        momentum_vals = df['momentum'].values

        for i in range(1, n):
            if long_cond[i]:
                bars_since_long[i] = 0
            else:
                bars_since_long[i] = bars_since_long[i-1] + 1

            if short_cond[i]:
                bars_since_short[i] = 0
            else:
                bars_since_short[i] = bars_since_short[i-1] + 1

            # longValid = longCondition AND barsSinceLastLong >= minBarsBetween
            lv = long_cond[i] and (bars_since_long[i-1] >= i_minBarsBetween if i > 0 else True)
            sv = short_cond[i] and (bars_since_short[i-1] >= i_minBarsBetween if i > 0 else True)

            # Resolve conflicts: if both long and short are valid on same bar
            if lv and sv:
                if is_bullish[i]:
                    sv = False
                elif is_bearish[i]:
                    lv = False
                else:
                    if momentum_vals[i] > 0:
                        sv = False
                    else:
                        lv = False

            long_valid[i] = lv
            short_valid[i] = sv

        df['longValid'] = long_valid
        df['shortValid'] = short_valid

        # =================================================================
        # DEBUG: Log the state of last 10 candles
        # =================================================================
        logging.info("--- Signal check on last 10 candles ---")
        for k in range(10, 0, -1):
            idx_k = -k
            if abs(idx_k) < len(df):
                row = df.iloc[idx_k]
                ts = df.index[idx_k] if hasattr(df.index[idx_k], 'strftime') else str(df.index[idx_k])
                logging.info(
                    f"  Candle[{idx_k}] {ts} | "
                    f"Close={row['close']:.2f} | "
                    f"TEMA_F={row['temaFast']:.2f} TEMA_S={row['temaSlow']:.2f} | "
                    f"longCond={row['longCondition']} shortCond={row['shortCondition']} | "
                    f"longValid={row['longValid']} shortValid={row['shortValid']}"
                )

        # =================================================================
        # SIGNAL CHECK — LATEST CLOSED CANDLE ONLY
        # =================================================================
        # The forming candle was already dropped before processing, so
        # df.iloc[-1] is the latest CLOSED candle.  Only report a signal
        # if longValid or shortValid fired on this candle — stale signals
        # from older candles are not actionable.

        signal = None
        crossover_angle = None
        tema_gap_pct = None

        last_row = df.iloc[-1]
        last_ts = df.index[-1]

        if last_row['longValid']:
            signal = "LONG"
        elif last_row['shortValid']:
            signal = "SHORT"

        if signal:
            logging.info(f"  >>> {signal} signal on latest closed candle {last_ts}")

            # Calculate TEMA gap percentage
            fast_val = last_row['temaFast']
            slow_val = last_row['temaSlow']
            if slow_val != 0:
                tema_gap_pct = round((fast_val - slow_val) / slow_val * 100, 3)

            # Calculate crossover angle
            try:
                angle_lookback = 3
                if len(df) > angle_lookback + 1:
                    fast_now = df['temaFast'].iloc[-1]
                    fast_prev = df['temaFast'].iloc[-1 - angle_lookback]
                    slow_now = df['temaSlow'].iloc[-1]
                    slow_prev = df['temaSlow'].iloc[-1 - angle_lookback]
                    price = df['close'].iloc[-1]

                    fast_slope = (fast_now - fast_prev) / angle_lookback
                    slow_slope = (slow_now - slow_prev) / angle_lookback
                    slope_diff = (fast_slope - slow_slope) / price
                    crossover_angle = round(np.degrees(np.arctan(slope_diff * 100)), 2)
            except Exception:
                crossover_angle = 0.0
        else:
            logging.info(f"  No signal on latest closed candle {last_ts}.")

        return signal, crossover_angle, tema_gap_pct

    except Exception as e:
        logging.error(f"Error in AMA PRO TEMA calculation: {str(e)}")
        import traceback
        logging.error(traceback.format_exc())
        return None, None, None

# =============================================================================
# MA QWEN LOGIC — Faithful Port of Pine Script
# =============================================================================

def apply_qwen_scanner(df, tf_input="1 day", **kwargs):
    """
    Applies the MA Qwen indicator logic (matching QwenPre alert):
    1. Volatility & Regime detection (lowVol, highVolMomentum, panicSelling)
    2. Indicators: EMA(12), EMA(26), RSI(14), Volume Spike, Bollinger Bands
    3. Strategy modes: mean_reversion, trend, neutral
    4. longCondition / shortCondition computed for all bars
    5. buyToPlot / sellToPlot with 5-bar dedup (no repeated QL/QS within 5 bars)
    6. Signal = buyToPlot or sellToPlot on the latest CLOSED candle (= QwenPre)

    Returns: (signal, None, None) — angle/tema_gap not applicable for Qwen
    """
    if df is None or len(df) < 100:
        return None, None, None

    # Drop the last row (current forming/incomplete candle)
    df = df.iloc[:-1].copy()

    if len(df) < 100:
        return None, None, None

    try:
        # === INPUTS ===
        i_volLookbackHours = 24

        # === Timeframe Handling ===
        tf_clean = tf_input.lower().strip()
        if 'min' in tf_clean:
            try:
                timeframe_in_minutes = int(tf_clean.replace('min', ''))
            except:
                timeframe_in_minutes = 15
        elif 'hr' in tf_clean:
            try:
                timeframe_in_minutes = int(tf_clean.replace('hr', '')) * 60
            except:
                timeframe_in_minutes = 60
        elif 'day' in tf_clean:
            timeframe_in_minutes = 1440
        elif 'week' in tf_clean:
            timeframe_in_minutes = 10080
        else:
            timeframe_in_minutes = 15

        bars_per_hour = 60 / max(1, timeframe_in_minutes)
        vol_lookback_bars = max(20, round(i_volLookbackHours * bars_per_hour))

        # === Volatility & Regime ===
        df['pctReturn'] = (df['close'] - df['close'].shift(1)) / df['close'].shift(1)
        df['volatility_q'] = df['pctReturn'].rolling(window=vol_lookback_bars, min_periods=1).std()

        # priceChange24h = close / close[vol_lookback_bars] - 1
        shift_bars = min(vol_lookback_bars, len(df) - 1)
        df['priceChange24h'] = df['close'] / df['close'].shift(shift_bars) - 1
        df['priceChange24h'] = df['priceChange24h'].fillna(0)

        df['lowVol'] = (df['volatility_q'] < 0.012) & (df['priceChange24h'].abs() < 0.008)
        df['highVolMomentum'] = (df['volatility_q'] > 0.035) & (df['priceChange24h'] > 0.025)
        df['panicSelling'] = df['priceChange24h'] < -0.03

        # === Indicators ===
        df['emaFast12'] = calculate_ema(df['close'], 12)
        df['emaSlow26'] = calculate_ema(df['close'], 26)
        df['rsi'] = calculate_rsi(df['close'], 14)
        df['volumeSMA30'] = df['volume'].rolling(window=30, min_periods=1).mean()
        df['volumeSpike'] = df['volume'] > df['volumeSMA30'] * 1.3

        # Bollinger Bands (adaptive)
        n = len(df)
        bb_upper = pd.Series(np.nan, index=df.index)
        bb_lower = pd.Series(np.nan, index=df.index)

        for i in range(max(24, 0), n):
            vol_val = df['volatility_q'].iloc[i] if not pd.isna(df['volatility_q'].iloc[i]) else 0
            bb_length = 14 if vol_val > 0.03 else 24
            start = max(0, i - bb_length + 1)
            window = df['close'].iloc[start:i + 1]
            basis = window.mean()
            dev = window.std()
            if not pd.isna(dev):
                bb_upper.iloc[i] = basis + 2.1 * dev
                bb_lower.iloc[i] = basis - 2.1 * dev

        df['bbUpper'] = bb_upper
        df['bbLower'] = bb_lower

        # VWAP (only for neutral mode)
        df['vwap'] = calculate_vwap(df)

        # === Strategy Logic ===
        # Determine mode for each bar
        modes = []
        for i in range(n):
            if df['lowVol'].iloc[i]:
                modes.append('mean_reversion')
            elif df['highVolMomentum'].iloc[i] or df['panicSelling'].iloc[i]:
                modes.append('trend')
            else:
                modes.append('neutral')
        df['mode'] = modes

        # =================================================================
        # COMPUTE longCondition / shortCondition FOR ALL BARS
        # =================================================================
        long_cond = np.zeros(n, dtype=bool)
        short_cond = np.zeros(n, dtype=bool)

        close_vals = df['close'].values
        ema_fast_vals = df['emaFast12'].values
        ema_slow_vals = df['emaSlow26'].values
        rsi_vals = df['rsi'].values
        bb_up_vals = df['bbUpper'].values
        bb_lo_vals = df['bbLower'].values
        vwap_vals = df['vwap'].values
        high_vol_mom_vals = df['highVolMomentum'].values
        panic_vals = df['panicSelling'].values
        vol_spike_vals = df['volumeSpike'].values

        for i in range(1, n):
            m = modes[i]
            if m == 'mean_reversion':
                if not np.isnan(bb_lo_vals[i]) and not np.isnan(rsi_vals[i]):
                    long_cond[i] = (close_vals[i] <= bb_lo_vals[i]) and (rsi_vals[i] < 28) and (close_vals[i] < ema_slow_vals[i])
                    short_cond[i] = (close_vals[i] >= bb_up_vals[i]) and (rsi_vals[i] > 72) and (close_vals[i] > ema_slow_vals[i])
            elif m == 'trend':
                if high_vol_mom_vals[i] or (panic_vals[i] and rsi_vals[i] < 25 and vol_spike_vals[i]):
                    long_cond[i] = (close_vals[i] > ema_fast_vals[i]) and (ema_fast_vals[i] > ema_slow_vals[i])
                short_cond[i] = False  # Conservative: no shorts in panic
            else:  # neutral
                if not np.isnan(vwap_vals[i]):
                    long_cond[i] = (close_vals[i] > ema_fast_vals[i]) and (ema_fast_vals[i] > ema_slow_vals[i]) and (close_vals[i] > vwap_vals[i])
                    short_cond[i] = (close_vals[i] < ema_fast_vals[i]) and (ema_fast_vals[i] < ema_slow_vals[i]) and (close_vals[i] < vwap_vals[i])

        # =================================================================
        # 5-BAR DEDUPLICATION — matches Pine Script buyToPlot / sellToPlot
        # =================================================================
        lookback_bars = 5
        last_idx = n - 1

        # Check buyToPlot on the last closed candle
        buy_to_plot = False
        if long_cond[last_idx]:
            had_recent_buy = False
            for j in range(1, min(lookback_bars + 1, last_idx + 1)):
                if long_cond[last_idx - j]:
                    had_recent_buy = True
                    break
            buy_to_plot = not had_recent_buy

        # Check sellToPlot on the last closed candle
        sell_to_plot = False
        if short_cond[last_idx]:
            had_recent_sell = False
            for j in range(1, min(lookback_bars + 1, last_idx + 1)):
                if short_cond[last_idx - j]:
                    had_recent_sell = True
                    break
            sell_to_plot = not had_recent_sell

        # =================================================================
        # SIGNAL — matches QwenPre alert (buyToPlot[1] / sellToPlot[1])
        # =================================================================
        signal = None
        if buy_to_plot:
            signal = "LONG"
        elif sell_to_plot:
            signal = "SHORT"

        if signal:
            last_ts = df.index[-1]
            logging.info(f"  >>> Qwen {signal} signal on latest closed candle {last_ts}")

        # Debug logging for the last candle
        logging.info(
            f"  Qwen debug | mode={modes[last_idx]} | "
            f"close={close_vals[last_idx]:.4f} | "
            f"emaFast={ema_fast_vals[last_idx]:.4f} emaSlow={ema_slow_vals[last_idx]:.4f} | "
            f"rsi={rsi_vals[last_idx]:.2f} | "
            f"longCond={long_cond[last_idx]} shortCond={short_cond[last_idx]} | "
            f"buyToPlot={buy_to_plot} sellToPlot={sell_to_plot}"
        )

        return signal, None, None

    except Exception as e:
        logging.error(f"Error in Qwen calculation: {str(e)}")
        import traceback
        logging.error(traceback.format_exc())
        return None, None, None

# =============================================================================
# MAIN SCAN ENTRY POINT
# =============================================================================

def run_scan(indices, timeframes, log_file, **kwargs):
    """
    Main entrypoint called by the API.
    Scans selected indices across selected timeframes.
    Supports scanner_type: 'ama_pro', 'qwen', or 'both'.
    """
    results = []

    # Map friendly names to yfinance tickers
    index_map = {
        'NIFTY': '^NSEI',
        'BANKNIFTY': '^NSEBANK',
        'DOW JONES': '^DJI',
        'NASDAQ': '^IXIC'
    }

    # Parameters to pass down
    adaptation_speed = kwargs.get('adaptation_speed', 'Medium')
    min_bars_between = kwargs.get('min_bars_between', 3)
    scanner_type = kwargs.get('scanner_type', 'ama_pro')

    run_ama = scanner_type in ('ama_pro', 'both')
    run_qwen = scanner_type in ('qwen', 'both')

    for index_name in indices:
        ticker = index_map.get(index_name)
        if not ticker:
            logging.warning(f"Unknown index '{index_name}', skipping.")
            continue

        logging.info(f"==> Starting scan for {index_name} ({ticker}) <==")

        # Fetch daily change once per index
        daily_change_pct = "N/A"
        try:
            d1_data = fetch_yfinance_data(ticker, "1 day")
            if d1_data is not None and len(d1_data) >= 2:
                prev_close = float(d1_data['close'].iloc[-2])
                curr_price = float(d1_data['close'].iloc[-1])
                daily_change_pct = f"{((curr_price - prev_close) / prev_close) * 100:+.2f}%"
        except Exception as e:
            logging.error(f"Failed to fetch daily change for {index_name}: {e}")

        for tf in timeframes:
            try:
                logging.info(f"Analyzing {index_name} on {tf} timeframe...")
                df = fetch_yfinance_data(ticker, tf)

                if df is None:
                    logging.warning(f"No data for {index_name} on {tf}.")
                    continue

                ama_signal = None
                qwen_signal = None

                # Run AMA Pro scanner
                if run_ama and len(df) >= 200:
                    signal, angle, tema_gap = apply_ama_pro_tema(
                        df.copy(),
                        tf_input=tf,
                        adaptation_speed=adaptation_speed,
                        min_bars_between=min_bars_between
                    )
                    if signal:
                        ama_signal = (signal, angle, tema_gap)

                # Run Qwen scanner
                if run_qwen and len(df) >= 100:
                    signal_q, _, _ = apply_qwen_scanner(
                        df.copy(),
                        tf_input=tf
                    )
                    if signal_q:
                        qwen_signal = signal_q

                # Build results based on scanner type
                if scanner_type == 'both':
                    if ama_signal and qwen_signal:
                        if ama_signal[0] == qwen_signal:
                            # Same signal from both — mark as "Both"
                            results.append({
                                'Crypto Name': index_name,
                                'Timeperiod': tf,
                                'Signal': ama_signal[0],
                                'Angle': f"{ama_signal[1]:.2f}°" if ama_signal[1] is not None else "N/A",
                                'TEMA Gap': f"{ama_signal[2]:+.3f}%" if ama_signal[2] is not None else "N/A",
                                'Daily Change': daily_change_pct,
                                'Scanner': 'Both',
                                'Timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            })
                        else:
                            # Different signals — add separate rows
                            results.append({
                                'Crypto Name': index_name,
                                'Timeperiod': tf,
                                'Signal': ama_signal[0],
                                'Angle': f"{ama_signal[1]:.2f}°" if ama_signal[1] is not None else "N/A",
                                'TEMA Gap': f"{ama_signal[2]:+.3f}%" if ama_signal[2] is not None else "N/A",
                                'Daily Change': daily_change_pct,
                                'Scanner': 'AMA Pro',
                                'Timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            })
                            results.append({
                                'Crypto Name': index_name,
                                'Timeperiod': tf,
                                'Signal': qwen_signal,
                                'Angle': 'N/A',
                                'TEMA Gap': 'N/A',
                                'Daily Change': daily_change_pct,
                                'Scanner': 'Qwen',
                                'Timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            })
                    elif ama_signal:
                        results.append({
                            'Crypto Name': index_name,
                            'Timeperiod': tf,
                            'Signal': ama_signal[0],
                            'Angle': f"{ama_signal[1]:.2f}°" if ama_signal[1] is not None else "N/A",
                            'TEMA Gap': f"{ama_signal[2]:+.3f}%" if ama_signal[2] is not None else "N/A",
                            'Daily Change': daily_change_pct,
                            'Scanner': 'AMA Pro',
                            'Timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        })
                    elif qwen_signal:
                        results.append({
                            'Crypto Name': index_name,
                            'Timeperiod': tf,
                            'Signal': qwen_signal,
                            'Angle': 'N/A',
                            'TEMA Gap': 'N/A',
                            'Daily Change': daily_change_pct,
                            'Scanner': 'Qwen',
                            'Timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        })
                elif scanner_type == 'ama_pro' and ama_signal:
                    results.append({
                        'Crypto Name': index_name,
                        'Timeperiod': tf,
                        'Signal': ama_signal[0],
                        'Angle': f"{ama_signal[1]:.2f}°" if ama_signal[1] is not None else "N/A",
                        'TEMA Gap': f"{ama_signal[2]:+.3f}%" if ama_signal[2] is not None else "N/A",
                        'Daily Change': daily_change_pct,
                        'Timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    })
                elif scanner_type == 'qwen' and qwen_signal:
                    results.append({
                        'Crypto Name': index_name,
                        'Timeperiod': tf,
                        'Signal': qwen_signal,
                        'Angle': 'N/A',
                        'TEMA Gap': 'N/A',
                        'Daily Change': daily_change_pct,
                        'Timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    })
                else:
                    logging.info(f"No signal for {index_name} on {tf}.")

            except Exception as e:
                logging.error(f"Error scanning {index_name} on {tf}: {str(e)}")

            time.sleep(1.5)  # Rate limiting delay

    logging.info(f"Scan complete. Total signals found: {len(results)}")
    return results
