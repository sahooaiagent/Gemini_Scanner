"""
Gemini Scanner — AMA PRO TEMA Logic
====================================
Faithful Python port of the AMA_PRO_TEMA Pine Script indicator.
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

# =============================================================================
# DATA FETCHING (with retry logic for JSONDecodeError)
# =============================================================================

def fetch_yfinance_data(ticker, tf_input, retries=3):
    """
    Fetches data using yfinance with retry logic.
    Maps timeframes to yfinance intervals, resamples when needed.
    """
    tf_str = tf_input.replace('min', 'm').replace('hr', 'h').replace(' day', 'd').replace(' week', 'wk').replace(' ', '')
    
    yf_interval = '15m'
    period = '60d'
    resample_freq = None
    
    if tf_str in ['15m', '30m']:
        yf_interval = tf_str
        period = '60d'
    elif tf_str == '45m':
        yf_interval = '15m'
        period = '60d'
        resample_freq = '45min'
    elif tf_str == '1h':
        yf_interval = '1h'
        period = '730d'
    elif tf_str in ['2h', '4h']:
        yf_interval = '1h'
        period = '730d'
        resample_freq = tf_str
    elif tf_str == '1d':
        yf_interval = '1d'
        period = '5y'
    elif tf_str == '1wk':
        yf_interval = '1wk'
        period = 'max'

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

def apply_ama_pro_tema(df, **kwargs):
    """
    Applies the full AMA PRO TEMA logic:
    1. Market regime detection (ADX, volatility ratio, EMA alignment)
    2. Adaptive TEMA period calculation
    3. TEMA crossover signals
    4. Signal filtering: longValid / shortValid (min bars between + regime conflict)
    5. Check the PREVIOUS closed candle for a valid signal
    
    Returns: (signal, crossover_angle) or (None, None)
    """
    if df is None or len(df) < 200:
        return None, None
    
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
        df['volatility'] = df['returns'].rolling(window=i_volLookback).std() * np.sqrt(252) * 100
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
        
        # Regime flags (using stable regime with confirmation counter)
        # For vectorized operation, use rolling mode or direct regime
        df['regimeIsBullish'] = df['directionRegime'] == 'Bullish'
        df['regimeIsBearish'] = df['directionRegime'] == 'Bearish'
        df['regimeIsHighVol'] = df['volRegime'] == 'High'
        df['regimeIsLowVol'] = df['volRegime'] == 'Low'
        df['regimeIsTrending'] = df['trendRegime'] == 'Trending'
        df['regimeIsRanging'] = df['trendRegime'] == 'Ranging'
        
        # =================================================================
        # SECTION 4: ADAPTIVE PARAMETERS — TEMA VERSION
        # =================================================================
        vol_adjust = np.select(
            [df['regimeIsHighVol'], df['regimeIsLowVol']],
            [0.7, 1.3], default=1.0
        )
        trend_adjust = np.where(df['regimeIsTrending'], 0.8, 1.2)
        tf_multiplier = 1.0  # Will be set per-timeframe later if needed
        
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
        # CHECK THE PREVIOUS CANDLE for longValid / shortValid
        # =================================================================
        # Per user requirement: ONLY check the previous completed candle.
        # Index -1 = Current/Forming candle (Live)
        # Index -2 = Previous completed candle
        
        signal = None
        crossover_angle = None
        signal_idx = -2  # ALWAYS check the previous closed candle
        
        if abs(signal_idx) < len(df):
            if df['longValid'].iloc[signal_idx]:
                signal = "LONG"
                logging.info(f"  >>> LONG VALID found at previous candle[{signal_idx}]")
            elif df['shortValid'].iloc[signal_idx]:
                signal = "SHORT"
                logging.info(f"  >>> SHORT VALID found at previous candle[{signal_idx}]")
        
        if signal is None:
            logging.info(f"  No valid signal on previous candle (index {signal_idx}).")
            return None, None
        
        # Calculate crossover angle
        try:
            angle_lookback = 3
            if abs(signal_idx) + angle_lookback < len(df):
                fast_now = df['temaFast'].iloc[signal_idx]
                fast_prev = df['temaFast'].iloc[signal_idx - angle_lookback]
                slow_now = df['temaSlow'].iloc[signal_idx]
                slow_prev = df['temaSlow'].iloc[signal_idx - angle_lookback]
                price = df['close'].iloc[signal_idx]
                
                fast_slope = (fast_now - fast_prev) / angle_lookback
                slow_slope = (slow_now - slow_prev) / angle_lookback
                slope_diff = (fast_slope - slow_slope) / price
                crossover_angle = round(np.degrees(np.arctan(slope_diff * 100)), 2)
        except Exception:
            crossover_angle = None
        
        return signal, crossover_angle
        
    except Exception as e:
        logging.error(f"Error in AMA PRO TEMA calculation: {str(e)}")
        import traceback
        logging.error(traceback.format_exc())
        return None, None

# =============================================================================
# MAIN SCAN ENTRY POINT
# =============================================================================

def run_scan(indices, timeframes, log_file, **kwargs):
    """
    Main entrypoint called by the API.
    Scans selected indices across selected timeframes using AMA PRO TEMA logic.
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
                
                if df is not None and len(df) >= 200:
                    signal, angle = apply_ama_pro_tema(df)
                    if signal:
                        logging.info(f"*** SIGNAL FOUND *** {index_name} | {tf} | {signal}")
                        results.append({
                            'Crypto Name': index_name,
                            'Timeperiod': tf,
                            'Signal': signal,
                            'Angle': f"{angle:.2f}°" if angle is not None else "N/A",
                            'Daily Change': daily_change_pct,
                            'Timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        })
                    else:
                        logging.info(f"No signal for {index_name} on {tf}.")
                else:
                    logging.warning(f"Insufficient data for {index_name} on {tf}.")
                    
            except Exception as e:
                logging.error(f"Error scanning {index_name} on {tf}: {str(e)}")
            
            time.sleep(1.5)  # Rate limiting delay
    
    logging.info(f"Scan complete. Total signals found: {len(results)}")
    return results
