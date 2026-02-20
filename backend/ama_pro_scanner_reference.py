import ccxt
import pandas as pd
import numpy as np
import time
import datetime
import warnings
import sys
import glob as file_glob
from concurrent.futures import ThreadPoolExecutor, as_completed
import smtplib
import ssl
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import os

# Load .env file if it exists
if os.path.exists('.env'):
    with open('.env') as f:
        for line in f:
            if '=' in line:
                key, value = line.strip().split('=', 1)
                os.environ[key] = value

# Suppress warnings
warnings.filterwarnings("ignore")

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
EXCHANGE_ID = 'binance'  # Default, will be overridden by command line
# SYMBOL_LIMIT will be set dynamically in main()
MAX_WORKERS = 2     # Very safe parallelism to avoid IP bans
TIMEFRAMES = ['15m', '30m', '1h', '2h', '4h', '1d']
# Valid intervals in CCXT/Binance/MEXC: 1m, 3m, 5m, 15m, 30m, 1h, 2h, 4h, 6h, 8h, 12h, 1d, 3d, 1w, 1M

# Gmail Configuration
GMAIL_USER = 'sahooaiagent@gmail.com'
GMAIL_PASSWORD = os.environ.get('GMAIL_APP_PASSWORD', '') # Set this in your environment or replace with App Password
RECEIVER_EMAIL = 'sahooaiagent@gmail.com'

CCXT_TIMEFRAMES = {
    '15m': '15m',
    '30m': '30m',
    '45m': '15m', # Resample from 15m
    '1h': '1h',
    '2h': '1h',   # Resample from 1h
    '3h': '1h',   # Resample from 1h
    '4h': '4h',
    '5h': '1h',   # Resample from 1h
    '12h': '4h',  # Resample from 4h
    '1d': '1d',
    '2d': '1d',   # Resample from 1d
    '1w': '1w',
    '1M': '1M'
}

# -----------------------------------------------------------------------------
# Technical Indicator Functions (Manual Implementation)
# -----------------------------------------------------------------------------

def calculate_ema(series, length):
    return series.ewm(span=length, adjust=False).mean()

def calculate_sma(series, length):
    return series.rolling(window=length).mean()

def calculate_rsi(series, length):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).fillna(0)
    loss = (-delta.where(delta < 0, 0)).fillna(0)
    
    avg_gain = gain.ewm(alpha=1/length, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/length, adjust=False).mean()
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_atr(high, low, close, length):
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/length, adjust=False).mean() # RMA is roughly EMA
    return atr

def calculate_adx(high, low, close, length):
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/length, adjust=False).mean() # Pine RMA
    
    up = high - high.shift()
    down = low.shift() - low
    
    plus_dm = np.where((up > down) & (up > 0), up, 0)
    minus_dm = np.where((down > up) & (down > 0), down, 0)
    
    plus_dm_s = pd.Series(plus_dm, index=high.index).ewm(alpha=1/length, adjust=False).mean()
    minus_dm_s = pd.Series(minus_dm, index=high.index).ewm(alpha=1/length, adjust=False).mean()
    
    plus_di = 100 * (plus_dm_s / atr)
    minus_di = 100 * (minus_dm_s / atr)
    
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = dx.ewm(alpha=1/length, adjust=False).mean()
    return adx

def calculate_stdev(series, length):
    return series.rolling(window=length).std()

def calculate_bb(series, length, mult):
    basis = calculate_sma(series, length)
    dev = calculate_stdev(series, length)
    upper = basis + mult * dev
    lower = basis - mult * dev
    return upper, lower

# -----------------------------------------------------------------------------
# Exchange Logic
# -----------------------------------------------------------------------------

def get_top_symbols(exchange, limit=50):
    print(f"Fetching top {limit} symbols from {EXCHANGE_ID.upper()} Futures...")
    try:
        tickers = exchange.fetch_tickers()
        
        # Filter for USDT pairs (Binance Futures format: SYMBOL/USDT:USDT)
        usdt_pairs = {k: v for k, v in tickers.items() if '/USDT' in k or ':USDT' in k}
        
        # Sort by 24h quote volume
        sorted_pairs = sorted(usdt_pairs.items(), key=lambda x: x[1]['quoteVolume'], reverse=True)
        
        top_symbols = [pair[0] for pair in sorted_pairs[:limit]]
        return top_symbols
    except Exception as e:
        if "418" in str(e) or "429" in str(e):
            print("Initial symbol fetch rate limited. Retrying in 30s...")
            time.sleep(30)
            return get_top_symbols(exchange, limit)
        print(f"Error fetching symbols: {e}")
        return []

def fetch_ohlcv(exchange, symbol, timeframe, limit=300):
    max_retries = 3
    for attempt in range(max_retries):
        try:
            # Handle resampling for non-standard timeframes
            fetch_tf = CCXT_TIMEFRAMES.get(timeframe, timeframe)
            
            # If we need to resample (e.g., 5h), we need more data
            fetch_limit = limit * 5 if timeframe in ['3h', '5h'] else limit
            
            ohlcv = exchange.fetch_ohlcv(symbol, fetch_tf, limit=fetch_limit)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            if timeframe == '3h':
                df = df.resample('3h').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'})
            elif timeframe == '5h':
                df = df.resample('5h').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'})
            elif timeframe == '45m':
                df = df.resample('45min').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'})
            elif timeframe == '2h':
                df = df.resample('2h').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'})
            elif timeframe == '12h':
                df = df.resample('12h').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'})
            elif timeframe == '2d':
                df = df.resample('2d').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'})
                
            return df.dropna()
        except Exception as e:
            if "418" in str(e) or "429" in str(e):
                wait_time = (attempt + 1) * 30 # Wait 30, 60, 90s
                print(f"Rate limited on {symbol}. Waiting {wait_time}s...")
                time.sleep(wait_time)
            else:
                if attempt == max_retries - 1:
                    print(f"Error fetching data for {symbol} {timeframe}: {e}")
                time.sleep(1)
    return None

# -----------------------------------------------------------------------------
# AMA Pro Logic
# -----------------------------------------------------------------------------

def apply_ama_pro_logic(df):
    if df is None or len(df) < 100:
        return None
    
    # --- Inputs (Default) ---
    i_emaFastMin = 8
    i_emaFastMax = 21
    i_emaSlowMin = 21
    i_emaSlowMax = 55
    i_rsiMin = 10
    i_rsiMax = 21
    i_bbLengthMin = 14
    i_bbLengthMax = 34
    
    i_adxLength = 14
    i_adxThreshold = 25
    i_volLookback = 50
    i_regimeStability = 3
    i_minBarsBetween = 3
    i_useVolumeFilter = False
    
    assetVolFactor = 1.5 # Crypto
    assetTrendFactor = 1.3 # Crypto
    
    # --- Regime Detection ---
    df['ADX'] = calculate_adx(df['high'], df['low'], df['close'], i_adxLength)
    
    # Volatility
    df['ATR'] = calculate_atr(df['high'], df['low'], df['close'], 14)
    df['returns'] = np.log(df['close'] / df['close'].shift(1))
    df['volatility'] = df['returns'].rolling(window=i_volLookback).std() * np.sqrt(252) * 100
    df['hist_vol'] = df['volatility'].rolling(window=i_volLookback).mean()
    df['vol_ratio'] = df['volatility'] / df['hist_vol']
    
    # Trend
    df['ema20'] = calculate_ema(df['close'], 20)
    df['ema50'] = calculate_ema(df['close'], 50)
    df['ema200'] = calculate_ema(df['close'], 200)
    
    # Regime Logic (Vectorized)
    conditions = [
        (df['vol_ratio'] > 1.3),
        (df['vol_ratio'] < 0.7)
    ]
    choices = ['High', 'Low']
    df['volRegime'] = np.select(conditions, choices, default='Normal')
    
    df['trendRegime'] = np.where(df['ADX'] > i_adxThreshold, 'Trending', 'Ranging')
    
    trend_up = (df['close'] > df['ema20']) & (df['ema20'] > df['ema50']) & (df['ema50'] > df['ema200'])
    trend_down = (df['close'] < df['ema20']) & (df['ema20'] < df['ema50']) & (df['ema50'] < df['ema200'])
    
    df['directionRegime'] = np.select([trend_up, trend_down], ['Bullish', 'Bearish'], default='Neutral')
    
    # --- Adaptive Calculation (Simplified for Python) ---
    # In Pine, this runs bar by bar. In Python, we can approximate by calculating the 'adjustFactor' series
    
    # 1. Calculate Adjust Factor Series
    # Factors based on regime
    vol_adjust = np.select(
        [df['volRegime'] == 'High', df['volRegime'] == 'Low'],
        [0.7, 1.3], default=1.0
    )
    trend_adjust = np.where(df['trendRegime'] == 'Trending', 0.8, 1.2)
    
    # Timeframe multiplier (Assuming typical timeframe > 1h for this scan)
    # 1h=60, 4h=240, 1d=1440. 
    # Logic: isintraday ? (<=5? 0.8 : <=60? 1.0 : 1.2) : (isdaily? 1.3 : 1.5)
    # We will assume multiplier 1.2 for >1h intraday, 1.3 for daily
    # For simplicity, let's use 1.2 as a baseline for H-timeframes
    tf_multiplier = 1.2 
    
    sensitivity_mult = 1.0 # Medium
    
    combined_adjust = vol_adjust * trend_adjust * tf_multiplier * sensitivity_mult
    adjust_factor = np.clip(1.0 / combined_adjust, 0.5, 1.5)
    
    
    # 2. Adaptive Periods
    fast_range = i_emaFastMax - i_emaFastMin
    slow_range = i_emaSlowMax - i_emaSlowMin
    
    # Calculate Adaptive Lengths (Series)
    adaptive_fast_len = i_emaFastMin + fast_range * (1 - adjust_factor)
    adaptive_slow_len = i_emaSlowMin + slow_range * (1 - adjust_factor)
    
    # Ensure separation
    adaptive_slow_len = np.maximum(adaptive_slow_len, adaptive_fast_len + 5)
    
    # Important: Calculating Dynamic EMA in Python Vectorized is hard. 
    # We will use the *current* bar's adaptive length to calculate the EMA for the *current* decision.
    # However, standard library functions like ewm or rolling require a fixed window.
    # APPROACH: We will pre-calculate a set of EMAs (e.g. 8, 10, 12... 55) and select the valid one based on the adaptive length for that row.
    
    ema_periods = [8, 10, 12, 14, 16, 18, 21, 26, 30, 34, 38, 42, 47, 55]
    emas = {}
    for p in ema_periods:
        emas[p] = calculate_ema(df['close'], p)
        
    # Select EMA Fast
    # Logic from Pine: 
    # <=9->8, <=11->10, <=13->12, <=15->14, <=17->16, <=19->18, else 21
    df['adaptiveEmaFast'] = np.select(
        [
            adaptive_fast_len <= 9,
            adaptive_fast_len <= 11,
            adaptive_fast_len <= 13,
            adaptive_fast_len <= 15,
            adaptive_fast_len <= 17,
            adaptive_fast_len <= 19
        ],
        [emas[8], emas[10], emas[12], emas[14], emas[16], emas[18]],
        default=emas[21]
    )
    
    # Select EMA Slow
    # Logic: <=28->26, <=32->30, <=36->34, <=40->38, <=44->42, <=51->47, else 55
    df['adaptiveEmaSlow'] = np.select(
        [
            adaptive_slow_len <= 28,
            adaptive_slow_len <= 32,
            adaptive_slow_len <= 36,
            adaptive_slow_len <= 40,
            adaptive_slow_len <= 44,
            adaptive_slow_len <= 51
        ],
        [emas[26], emas[30], emas[34], emas[38], emas[42], emas[47]],
        default=emas[55]
    )
    
    # --- Strategy Logic ---
    # Strategy Mode
    df['strategyMode'] = np.select(
        [
            (df['trendRegime'] == 'Ranging') & (df['volRegime'] == 'Low'),
            (df['trendRegime'] == 'Trending') & (df['volRegime'] == 'High')
        ],
        ['mean_reversion', 'trend_following'],
        default='balanced'
    )
    
    # Conditions
    # All modes use crossovers of adaptive EMAs as base signal in the simplified code
    # Long: Crossover, Short: Crossunder
    
    # Create shifted columns for crossover check
    df['prev_fast'] = df['adaptiveEmaFast'].shift(1)
    df['prev_slow'] = df['adaptiveEmaSlow'].shift(1)
    
    # Crossover: Fast > Slow AND PrevFast <= PrevSlow
    df['longCondition'] = (df['adaptiveEmaFast'] > df['adaptiveEmaSlow']) & (df['prev_fast'] <= df['prev_slow'])
    # Crossunder: Fast < Slow AND PrevFast >= PrevSlow
    df['shortCondition'] = (df['adaptiveEmaFast'] < df['adaptiveEmaSlow']) & (df['prev_fast'] >= df['prev_slow'])
    
    # --- Additional Filters to Reduce False Signals ---
    
    # 1. EMA Separation Filter: Require minimum separation between EMAs
    min_ema_separation_pct = 0.15  # 0.15% minimum separation
    ema_separation_pct = abs(df['adaptiveEmaFast'] - df['adaptiveEmaSlow']) / df['close'] * 100
    df['ema_separation_valid'] = ema_separation_pct >= min_ema_separation_pct
    
    # 2. Price Confirmation Filter: Price should align with signal direction
    # For LONG: price should be at or above fast EMA (within 0.2% tolerance)
    # For SHORT: price should be at or below fast EMA (within 0.2% tolerance)
    df['price_confirms_long'] = df['close'] >= df['adaptiveEmaFast'] * 0.998
    df['price_confirms_short'] = df['close'] <= df['adaptiveEmaFast'] * 1.002
    
    # --- Filtering (Last 5 Candles) ---
    # Check if BUY/SELL signals appeared in the last 5 candles
    # and validate that price hasn't invalidated the signal

    signal = None
    crossover_angle = None

    # Look back at the last 5 closed candles (excluding current incomplete candle)
    lookback_candles = min(5, len(df) - 1)

    for i in range(2, lookback_candles + 2):  # Check candles -2 to -6
        candle_idx = -i
        candle = df.iloc[candle_idx]

        # Check for LONG signal
        if candle['longCondition']:
            # Enhanced Regime Filter check for Long
            is_choppy = (candle['trendRegime'] == 'Ranging') and (candle['directionRegime'] == 'Neutral')

            if (candle['directionRegime'] == 'Bearish' or
                not candle['ema_separation_valid'] or
                not candle['price_confirms_long'] or
                is_choppy):
                continue  # Skip this candle

            # Price Validation: Check if ANY candle after the signal has crossed above the buy candle's high
            # Get all candles from the signal candle to the most recent candle
            buy_candle_high = df['high'].iloc[candle_idx]
            candles_after_signal = df.iloc[candle_idx:]

            # Check if the highest high after (and including) the signal candle has exceeded the signal candle's high
            max_high_after = candles_after_signal['high'].iloc[1:].max() if len(candles_after_signal) > 1 else 0

            if max_high_after > buy_candle_high:  # Price has crossed above, signal invalidated
                continue

            # Valid LONG signal found
            signal = "LONG"

            # Calculate crossover angle for this candle
            angle_lookback = min(3, abs(candle_idx) - 1)
            if angle_lookback > 0:
                fast_slope = (df['adaptiveEmaFast'].iloc[candle_idx] - df['adaptiveEmaFast'].iloc[candle_idx - angle_lookback]) / angle_lookback
                slow_slope = (df['adaptiveEmaSlow'].iloc[candle_idx] - df['adaptiveEmaSlow'].iloc[candle_idx - angle_lookback]) / angle_lookback
                slope_diff = (fast_slope - slow_slope) / candle['close']
                crossover_angle = np.degrees(np.arctan(slope_diff * 100))
            break  # Found a valid signal, stop searching

        # Check for SHORT signal
        if candle['shortCondition']:
            # Enhanced Regime Filter check for Short
            is_choppy = (candle['trendRegime'] == 'Ranging') and (candle['directionRegime'] == 'Neutral')

            if (candle['directionRegime'] == 'Bullish' or
                not candle['ema_separation_valid'] or
                not candle['price_confirms_short'] or
                is_choppy):
                continue  # Skip this candle

            # Price Validation: Check if ANY candle after the signal has crossed below the sell candle's low
            # Get all candles from the signal candle to the most recent candle
            sell_candle_low = df['low'].iloc[candle_idx]
            candles_after_signal = df.iloc[candle_idx:]

            # Check if the lowest low after (and including) the signal candle has gone below the signal candle's low
            min_low_after = candles_after_signal['low'].iloc[1:].min() if len(candles_after_signal) > 1 else float('inf')

            if min_low_after < sell_candle_low:  # Price has crossed below, signal invalidated
                continue

            # Valid SHORT signal found
            signal = "SHORT"

            # Calculate crossover angle for this candle
            angle_lookback = min(3, abs(candle_idx) - 1)
            if angle_lookback > 0:
                fast_slope = (df['adaptiveEmaFast'].iloc[candle_idx] - df['adaptiveEmaFast'].iloc[candle_idx - angle_lookback]) / angle_lookback
                slow_slope = (df['adaptiveEmaSlow'].iloc[candle_idx] - df['adaptiveEmaSlow'].iloc[candle_idx - angle_lookback]) / angle_lookback
                slope_diff = (fast_slope - slow_slope) / candle['close']
                crossover_angle = np.degrees(np.arctan(slope_diff * 100))
            break  # Found a valid signal, stop searching

    return signal, crossover_angle

# -----------------------------------------------------------------------------
# Notification Logic
# -----------------------------------------------------------------------------

def send_gmail_notification(signals):
    """Sends found signals or status report via Gmail."""
    if not GMAIL_PASSWORD:
        print("\n[WARNING] Gmail App Password not set. Skipping email notification.")
        print("To enable email, set GMAIL_APP_PASSWORD environment variable or update the script.")
        return

    print(f"Sending email notification to {RECEIVER_EMAIL}...")
    
    # Create the email content
    if signals:
        subject = f"AMA Pro Scanner Alert: {len(signals)} Signals Found"
        body_text = f"AMA Pro Scanner detected the following {len(signals)} signals:\n\n"
        body_text += f"{'Crypto Name':<15} | {'Timeperiod':<10} | {'Signal':<10} | {'Angle':<10} | {'Timestamp'}\n"
        body_text += "-" * 85 + "\n"
        for s in signals:
            body_text += f"{s['Crypto Name']:<15} | {s['Timeperiod']:<10} | {s['Signal']:<10} | {s['Angle']:<10} | {s['Timestamp']}\n"
    else:
        subject = "AMA Pro Scanner: Daily Status (No Signals)"
        body_text = "AMA Pro Scanner completed its daily run. No signals were detected matching your criteria.\n"

    body_text += "\n\nHappy Trading!"

    # Create the MIME message
    message = MIMEMultipart()
    message["From"] = GMAIL_USER
    message["To"] = RECEIVER_EMAIL
    message["Subject"] = subject
    message.attach(MIMEText(body_text, "plain"))

    # Send the email
    context = ssl.create_default_context()
    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as server:
            server.login(GMAIL_USER, GMAIL_PASSWORD)
            server.sendmail(GMAIL_USER, RECEIVER_EMAIL, message.as_string())
        print("Email notification sent successfully.")
    except Exception as e:
        print(f"Failed to send email: {e}")

# -----------------------------------------------------------------------------
# Cleanup Functions
# -----------------------------------------------------------------------------

def cleanup_old_results(keep_last=10):
    """Remove old CSV result files, keeping only the most recent ones."""
    try:
        csv_files = file_glob.glob("ama_pro_scan_results_*.csv")
        if len(csv_files) > keep_last:
            # Sort by modification time
            csv_files.sort(key=os.path.getmtime, reverse=True)
            # Remove older files
            for old_file in csv_files[keep_last:]:
                os.remove(old_file)
                print(f"🗑️  Cleaned up old result file: {old_file}")
    except Exception as e:
        print(f"Warning: Could not cleanup old files: {e}")

# -----------------------------------------------------------------------------
# Worker Function
# -----------------------------------------------------------------------------

def scan_symbol(exchange, symbol):
    """Worker function to scan a single symbol across all timeframes."""
    found_signals = []

    # Fetch sources to resample if needed
    h1_df_source = None
    h4_df_source = None
    d1_df_source = None
    m15_df_source = None

    try:
        # Optimization: Fetch base timeframes for resampling (pre-check which ones we need)
        needs_h1 = any(tf in ['1h', '2h', '3h', '5h'] for tf in TIMEFRAMES)
        needs_h4 = any(tf in ['12h'] for tf in TIMEFRAMES)
        needs_d1 = any(tf in ['1d', '2d'] for tf in TIMEFRAMES)
        needs_m15 = any(tf in ['45m'] for tf in TIMEFRAMES)

        if needs_h1:
            h1_df_source = fetch_ohlcv(exchange, symbol, '1h', limit=1000)
        if needs_h4:
            h4_df_source = fetch_ohlcv(exchange, symbol, '4h', limit=1000)
        if needs_d1:
            d1_df_source = fetch_ohlcv(exchange, symbol, '1d', limit=500)
        if needs_m15:
            m15_df_source = fetch_ohlcv(exchange, symbol, '15m', limit=1000)
    except Exception:
        pass

    # Calculate Daily Change (reuse d1_df_source if available to avoid redundant fetch)
    daily_change_pct = "N/A"
    try:
        # Reuse d1_df_source if we already fetched it
        daily_df = d1_df_source if d1_df_source is not None else fetch_ohlcv(exchange, symbol, '1d', limit=2)
        if daily_df is not None and len(daily_df) >= 2:
            prev_close = daily_df['close'].iloc[-2]
            curr_price = daily_df['close'].iloc[-1]
            daily_change_pct = f"{((curr_price - prev_close) / prev_close) * 100:.2f}%"
    except Exception:
        pass

    for tf in TIMEFRAMES:
        try:
            df = None
            # Resample timeframes from sources
            if tf in ['1h', '2h', '3h', '5h'] and h1_df_source is not None:
                if tf == '1h':
                    df = h1_df_source.tail(300)
                else:
                    df = h1_df_source.resample(tf).agg({
                        'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'
                    }).dropna().tail(300)
            elif tf == '12h' and h4_df_source is not None:
                df = h4_df_source.resample('12h').agg({
                    'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'
                }).dropna().tail(300)
            elif tf == '2d' and d1_df_source is not None:
                df = d1_df_source.resample('2d').agg({
                    'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'
                }).dropna().tail(300)
            elif tf == '45m' and m15_df_source is not None:
                df = m15_df_source.resample('45min').agg({
                    'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'
                }).dropna().tail(300)
            else:
                # Fetch directly for 15m, 30m, 4h, 1d, 1w, 1M
                df = fetch_ohlcv(exchange, symbol, tf, limit=300)
            
            if df is not None and not df.empty:
                result = apply_ama_pro_logic(df)
                if result:
                    signal, angle = result
                    if signal:
                        found_signals.append({
                            'Crypto Name': symbol,
                            'Timeperiod': tf,
                            'Signal': signal,
                            'Angle': f"{angle:.2f}°" if angle is not None else "N/A",
                            'Daily Change': daily_change_pct,
                            'Timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        })
        except Exception:
            pass
            
    # Tiny pause between symbols to spread requests
    time.sleep(0.5)
    return found_signals

# -----------------------------------------------------------------------------
# Main Execution
# -----------------------------------------------------------------------------

def main():
    global TIMEFRAMES, EXCHANGE_ID
    start_time = time.time()

    # Handle dynamic parameters
    import argparse
    parser = argparse.ArgumentParser(add_help=False) # Don't conflict with positional args
    parser.add_argument('--exchange', type=str, default='binance', help='Exchange to use (binance or mexc)')
    parser.add_argument('--timeframes', type=str, help='Comma-separated timeframes')
    args, unknown = parser.parse_known_args()

    # Set exchange
    EXCHANGE_ID = args.exchange.lower()
    exchange_name = "Binance" if EXCHANGE_ID == "binance" else "MEXC"
    print(f"Starting AMA Pro Logic Scanner ({exchange_name} Edition)...")

    # User Configuration: Number of symbols
    # Check if a limit was passed as a command line argument
    if len(sys.argv) > 1:
        try:
            current_symbol_limit = int(sys.argv[1])
            print(f"Using symbol limit from argument: {current_symbol_limit}")
        except ValueError:
            print(f"Invalid argument '{sys.argv[1]}'. Using default: 100")
            current_symbol_limit = 100
    else:
        # If output is redirected (not a TTY), skip interactive input
        if not sys.stdout.isatty():
            print("Non-interactive mode (redirection) detected. Using default symbol limit: 100")
            current_symbol_limit = 100
        else:
            try:
                limit_input = input("Enter the number of crypto to be searched (100, 200, 300, 400, 500) [Default 100]: ").strip()
                current_symbol_limit = int(limit_input) if limit_input else 100
            except (EOFError, ValueError):
                print("Invalid input or EOF. Using default: 100")
                current_symbol_limit = 100

    if args.timeframes:
        TIMEFRAMES = [tf.strip() for tf in args.timeframes.split(',')]
        print(f"Overriding timeframes from CLI: {TIMEFRAMES}")

    print(f"Exchange: {exchange_name}")
    print(f"Timeframes: {TIMEFRAMES}")
    print(f"Max Workers: {MAX_WORKERS}")

    # Initialize Exchange
    if EXCHANGE_ID == 'mexc':
        exchange = ccxt.mexc({
            'options': {'defaultType': 'swap'},  # MEXC uses 'swap' for futures
            'enableRateLimit': True
        })
    else:  # binance
        exchange = ccxt.binance({
            'options': {'defaultType': 'future'},
            'enableRateLimit': True
        })
    
    # Get Top Symbols
    symbols = get_top_symbols(exchange, limit=current_symbol_limit)
    num_symbols = len(symbols)
    print(f"Found {num_symbols} symbols. Starting parallel scan...")
    
    all_results = []
    
    # Use ThreadPoolExecutor for parallel processing
    print(f"\n🔍 Scanning {num_symbols} symbols across {len(TIMEFRAMES)} timeframes...")
    print("=" * 60)

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Create a dictionary to keep track of futures
        future_to_symbol = {executor.submit(scan_symbol, exchange, symbol): symbol for symbol in symbols}

        count = 0
        signals_found = 0
        for future in as_completed(future_to_symbol):
            symbol = future_to_symbol[future]
            try:
                results = future.result()
                if results:
                    all_results.extend(results)
                    signals_found += len(results)
                    for res in results:
                        print(f"✅ SIGNAL FOUND: {res['Crypto Name']} [{res['Timeperiod']}] -> {res['Signal']} (Angle: {res['Angle']})")
            except Exception as e:
                print(f"⚠️  Error scanning {symbol}: {e}")

            count += 1
            if count % 10 == 0 or count == num_symbols:
                progress_pct = (count / num_symbols) * 100
                print(f"📊 Progress: {count}/{num_symbols} ({progress_pct:.1f}%) | Signals Found: {signals_found}")
        
    # Process and Output Results
    print("\n" + "=" * 60)
    results_df = pd.DataFrame(all_results)

    # Always save a CSV (even if empty)
    timestamp_str = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"ama_pro_scan_results_{timestamp_str}.csv"
    results_df.to_csv(filename, index=False)
    print(f"💾 Results saved to {filename}")

    if not all_results:
        print("\n❌ No signals found matching criteria.")
    else:
        print(f"\n✅ {len(all_results)} Total Signals Found!")
        print("\n" + results_df.to_string(index=False))

    # Always send Gmail Notification (status report)
    print("\n📧 Sending email notification...")
    send_gmail_notification(all_results)

    duration = time.time() - start_time
    print("\n" + "=" * 60)
    print(f"✅ SCAN COMPLETED in {duration/60:.2f} minutes ({duration:.1f} seconds)")
    print(f"📊 Summary: {len(all_results)} signals from {num_symbols} symbols")
    print("=" * 60)

    # Cleanup old result files
    cleanup_old_results(keep_last=10)

if __name__ == "__main__":
    main()
