import requests
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta
import time

def fetch_candles_from_binance(symbol="BTCUSDT", interval="5m", limit=1000):
    url = "https://api.binance.com/api/v3/klines"
    
    interval_minutes = {
        '1m': 1, '3m': 3, '5m': 5, '15m': 15, '30m': 30,
        '1h': 60, '2h': 120, '4h': 240, '6h': 360, '8h': 480, '12h': 720,
        '1d': 1440, '3d': 4320, '1w': 10080, '1M': 43200
    }
    
    minutes_back = limit * interval_minutes.get(interval, 5)
    start_time = int((datetime.now() - timedelta(minutes=minutes_back)).timestamp() * 1000)
    
    all_data = []
    while limit > 0:
        request_limit = min(limit, 1000)
        params = {
            "symbol": symbol,
            "interval": interval,
            "limit": request_limit,
            "startTime": start_time
        }
        
        try:
            r = requests.get(url, params=params)
            r.raise_for_status()
            data = r.json()
            
            if not data:
                break
                
            all_data.extend(data)
            start_time = data[-1][0] + 1
            limit -= request_limit
            time.sleep(0.1)
            
        except Exception as e:
            print(f"Error fetching data for {symbol}: {e}")
            return None

    rows = []
    for d in all_data:
        rows.append({
            "opentime": pd.to_datetime(d[0], unit='ms', utc=True),
            "closetime": pd.to_datetime(d[6], unit='ms', utc=True),
            "open": float(d[1]),
            "high": float(d[2]),
            "low": float(d[3]),
            "close": float(d[4]),
            "volume": float(d[5]),
            "number_of_trades": d[8],
            "quote_asset_volume": float(d[7]),
            "taker_buy_base_volume": float(d[9]),
            "taker_buy_quote_volume": float(d[10])
        })

    df = pd.DataFrame(rows).sort_values("closetime").reset_index(drop=True)
    now_utc = datetime.now(timezone.utc)
    df = df[df["closetime"] <= now_utc].reset_index(drop=True)
    return df

def fetch_current_price(symbol="BTCUSDT"):
    url = "https://api.binance.com/api/v3/ticker/price"
    params = {"symbol": symbol}
    try:
        r = requests.get(url, params=params)
        r.raise_for_status()
        data = r.json()
        return float(data['price'])
    except Exception as e:
        print(f"Error fetching current price for {symbol}: {e}")
        return None

def detect_hh_hl_lh_ll_columns(df, interval):
    """Add HH, HL, LH, LL boolean columns to dataframe based on swing detection"""
    # Initialize columns
    df['hh'] = False
    df['hl'] = False
    df['lh'] = False
    df['ll'] = False
    
    # Swing window based on interval
    swing_window_dict = {
        '1m': 65, '3m': 50, '5m': 50, '15m': 70, '30m': 85,
        '1h': 100, '2h': 100, '4h': 90, '6h': 90, '8h': 90, '12h': 90,
        '1d': 60, '3d': 60, '1w': 60, '1M': 60
    }
    swing_window = swing_window_dict.get(interval, 50)
    
    # Minimum price change threshold (based on ATR)
    min_price_change = df['atr'] * 0.5 if 'atr' in df.columns else df['close'] * 0.01

    # Detect HH/HL/LH/LL with local pivot check
    for i in range(swing_window, len(df)):
        # Local pivot check: Is this a local max/min within a small window (e.g., 5 candles)?
        local_window = 5  # Check 5 candles before and after
        is_local_high = df['high'].iloc[i] == df['high'].iloc[max(0, i-local_window):i+local_window+1].max()
        is_local_low = df['low'].iloc[i] == df['low'].iloc[max(0, i-local_window):i+local_window+1].min()
        
        # HH: Higher high (significant high above recent max)
        if is_local_high and df['high'].iloc[i] > df['high'].iloc[i-swing_window:i].max():
            price_diff = df['high'].iloc[i] - df['high'].iloc[i-swing_window:i].max()
            if price_diff >= min_price_change.iloc[i]:
                df.loc[df.index[i], 'hh'] = True
        
        # HL: Higher low (significant low above recent min)
        if is_local_low and df['low'].iloc[i] > df['low'].iloc[i-swing_window:i].min():
            price_diff = df['low'].iloc[i] - df['low'].iloc[i-swing_window:i].min()
            if price_diff >= min_price_change.iloc[i]:
                df.loc[df.index[i], 'hl'] = True
        
        # LH: Lower high (significant high below recent max)
        if is_local_high and df['high'].iloc[i] < df['high'].iloc[i-swing_window:i].max():
            price_diff = df['high'].iloc[i-swing_window:i].max() - df['high'].iloc[i]
            if price_diff >= min_price_change.iloc[i]:
                df.loc[df.index[i], 'lh'] = True
        
        # LL: Lower low (significant low below recent min)
        if is_local_low and df['low'].iloc[i] < df['low'].iloc[i-swing_window:i].min():
            price_diff = df['low'].iloc[i-swing_window:i].min() - df['low'].iloc[i]
            if price_diff >= min_price_change.iloc[i]:
                df.loc[df.index[i], 'll'] = True
    
    return df



def detect_pivots(df, window=10):
    """
    Detect pivot highs and lows in the dataframe
    """
    df = df.copy()
    
    # Initialize pivot columns
    df['pivot_high'] = False
    df['pivot_low'] = False
    
    # Detect pivot highs
    for i in range(window, len(df) - window):
        if all(df['high'].iloc[i] >= df['high'].iloc[i-window:i+window+1]):
            df.loc[df.index[i], 'pivot_high'] = True
    
    # Detect pivot lows
    for i in range(window, len(df) - window):
        if all(df['low'].iloc[i] <= df['low'].iloc[i-window:i+window+1]):
            df.loc[df.index[i], 'pivot_low'] = True
    
    return df



def calculate_indicators(df, rsi_period=14, ema_period=20, atr_period=14, bb_period=20,
                        macd_fast=12, macd_slow=26, macd_signal=9,
                        ichimoku_tenkan=9, ichimoku_kijun=26, ichimoku_senkou_b=52,
                        wyckoff_ema_short=20, wyckoff_ema_long=50):
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=rsi_period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_period).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    df['ema'] = df['close'].ewm(span=ema_period, adjust=False).mean()
    df['ema50'] = df['close'].ewm(span=50, adjust=False).mean()
    
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    df['atr'] = true_range.rolling(window=atr_period).mean()
    df['inverseATR'] = 1 / df['atr']
    
    df['bb_sma'] = df['close'].rolling(window=bb_period).mean()
    bb_std = df['close'].rolling(window=bb_period).std()
    df['bb_upper'] = df['bb_sma'] + (bb_std * 2)
    df['bb_lower'] = df['bb_sma'] - (bb_std * 2)
    df['bb_percentB'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
    
    # MACD calculation
    ema_fast = df['close'].ewm(span=macd_fast, adjust=False).mean()
    ema_slow = df['close'].ewm(span=macd_slow, adjust=False).mean()
    df['macd'] = ema_fast - ema_slow
    df['macd_signal'] = df['macd'].ewm(span=macd_signal, adjust=False).mean()
    df['macd_histogram'] = df['macd'] - df['macd_signal']
    
    # Ichimoku Components
    high_tenkan = df['high'].rolling(window=ichimoku_tenkan).max()
    low_tenkan = df['low'].rolling(window=ichimoku_tenkan).min()
    df['Tenkan_sen'] = (high_tenkan + low_tenkan) / 2
    
    high_kijun = df['high'].rolling(window=ichimoku_kijun).max()
    low_kijun = df['low'].rolling(window=ichimoku_kijun).min()
    df['Kijun_sen'] = (high_kijun + low_kijun) / 2
    
    df['Senkou_Span_A'] = ((df['Tenkan_sen'] + df['Kijun_sen']) / 2).shift(ichimoku_kijun)
    df['Senkou_Span_A_unshifted'] = (df['Tenkan_sen'] + df['Kijun_sen']) / 2
    
    high_senkou_b = df['high'].rolling(window=ichimoku_senkou_b).max()
    low_senkou_b = df['low'].rolling(window=ichimoku_senkou_b).min()
    df['Senkou_Span_B'] = ((high_senkou_b + low_senkou_b) / 2).shift(ichimoku_kijun)
    df['Senkou_Span_B_unshifted'] = (high_senkou_b + low_senkou_b) / 2
    
    # Wyckoff-specific indicators
    df['wyckoff_ema_short'] = df['close'].ewm(span=wyckoff_ema_short, adjust=False).mean()
    df['wyckoff_ema_long'] = df['close'].ewm(span=wyckoff_ema_long, adjust=False).mean()
    
    # Support/Resistance for Wyckoff
    lookback = 20
    df['support'] = df['low'].rolling(window=lookback).min()
    df['resistance'] = df['high'].rolling(window=lookback).max()
    
    # Average Volume for Wyckoff
    df['avg_volume'] = df['volume'].rolling(window=20).mean()
    df['volume_spike'] = df['volume'] > df['avg_volume'] * 1.5
    df['volume_spike_ema'] = df['volume_spike'].ewm(span=20, adjust=False).mean()
    
    # On-Balance Volume (OBV)
    df['obv'] = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()

    df = detect_pivots(df, window=10)
    
    return df