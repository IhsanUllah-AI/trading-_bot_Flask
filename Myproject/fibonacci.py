import pandas as pd
import numpy as np
from utils import fetch_candles_from_binance, fetch_current_price, calculate_indicators
from plotly.subplots import make_subplots
import plotly.graph_objects as go

def compute_fib_levels(df, window=20, fib_ratios=[0.236, 0.382, 0.5, 0.618, 0.786, 1.0]):
    if len(df) < window:
        window = len(df)
    
    latest = df.iloc[-1]
    tail = df.tail(window)

    high = tail["high"].max()
    low = tail["low"].min()
    close = latest["close"]

    move = high - low
    if move == 0:
        return "sideways", {}, high, low, close, latest
    
    pos = (close - low) / move

    if pos > 0.6:
        trend = "uptrend"
        levels = {f"support_{int(r*100)}": high - move * r for r in fib_ratios}
    elif pos < 0.4:
        trend = "downtrend"
        levels = {f"resistance_{int(r*100)}": low + move * r for r in fib_ratios}
    else:
        trend = "sideways"
        levels = {}

    return trend, levels, high, low, close, latest

def generate_signals(df, trend, levels, latest, high, low, symbol, fib_threshold=0.01):
    signals = []
    confidence = 0
    buy_signals = []
    sell_signals = []
    buy_descriptions = []
    sell_descriptions = []

    entry_price = None
    stop_loss = None
    take_profit = None
    trade_action = None
    fib_signal_triggered = False

    current_price = fetch_current_price(symbol)
    if current_price is None:
        current_price = latest['close']

    # RSI Signals (weight 0.25)
    rsi = latest['rsi']
    if rsi < 30:
        buy_signals.append(("RSI Oversold", "BUY", "green", 0.25))
        buy_descriptions.append("RSI < 30: Potential oversold condition, suggesting a buy.")
    elif rsi > 70:
        sell_signals.append(("RSI Overbought", "SELL", "red", 0.25))
        sell_descriptions.append("RSI > 70: Potential overbought condition, suggesting a sell.")

    # Bollinger Bands Signals (weight 0.20)
    bb_percent = latest['bb_percentB']
    if bb_percent < 0:
        buy_signals.append(("Price Below Lower BB", "BUY", "green", 0.20))
        buy_descriptions.append("Price below lower Bollinger Band: Suggests oversold, potential buy.")
    elif bb_percent > 1:
        sell_signals.append(("Price Above Upper BB", "SELL", "red", 0.20))
        sell_descriptions.append("Price above upper Bollinger Band: Suggests overbought, potential sell.")

    # EMA Signal (weight 0.15)
    if current_price > latest['ema']:
        buy_signals.append(("Price Above EMA", "BUY", "blue", 0.15))
        buy_descriptions.append("Price above 20-period EMA: Bullish momentum, favoring buys.")
    else:
        sell_signals.append(("Price Below EMA", "SELL", "orange", 0.15))
        sell_descriptions.append("Price below 20-period EMA: Bearish momentum, favoring sells.")

    # Fibonacci Signals with Indicator Confirmation (weight 0.40)
    if trend != "sideways" and levels:
        fib_levels = list(levels.values())
        fib_keys = list(levels.keys())

        closest_idx = np.argmin(np.abs(np.array(fib_levels) - current_price))
        closest_level = fib_levels[closest_idx]
        level_name = fib_keys[closest_idx]
        distance_percent = abs(current_price - closest_level) / closest_level * 100

        has_buy_indicator = rsi < 30 or bb_percent < 0 or current_price > latest['ema']
        has_sell_indicator = rsi > 70 or bb_percent > 1 or current_price < latest['ema']

        if abs(current_price - closest_level) / closest_level <= fib_threshold:
            if trend == "uptrend" and has_buy_indicator:
                buy_signals.append((f"Near {level_name} Support", "BUY", "green", 0.40))
                buy_descriptions.append(f"Price near {level_name} support with indicator confirmation: Strong buy signal.")
                entry_price = current_price
                stop_loss = entry_price * 0.99  # 1% risk
                take_profit = entry_price * 1.02  # 2% reward, ensuring 1:2 ratio
                fib_signal_triggered = True
                trade_action = "BUY"
            elif trend == "downtrend" and has_sell_indicator:
                sell_signals.append((f"Near {level_name} Resistance", "SELL", "red", 0.40))
                sell_descriptions.append(f"Price near {level_name} resistance with indicator confirmation: Strong sell signal.")
                entry_price = current_price
                stop_loss = entry_price * 1.01  # 1% risk
                take_profit = entry_price * 0.98  # 2% reward, ensuring 1:2 ratio
                fib_signal_triggered = True
                trade_action = "SELL"

    # Fallback to ATR for 1:2 ratio
    atr = latest['atr']
    if fib_signal_triggered and entry_price is not None:
        if trade_action == "BUY":
            stop_loss = entry_price - atr  # risk = atr
            take_profit = entry_price + (2 * atr)  # reward = 2 * atr, 1:2
        elif trade_action == "SELL":
            stop_loss = entry_price + atr  # risk = atr
            take_profit = entry_price - (2 * atr)  # reward = 2 * atr, 1:2

    signals = buy_signals + sell_signals
    signal_descriptions = buy_descriptions + sell_descriptions

    buy_confidence = sum(weight for _, _, _, weight in buy_signals)
    sell_confidence = sum(weight for _, _, _, weight in sell_signals)

    if buy_confidence > 0 and sell_confidence > 0:
        conflict_penalty = min(buy_confidence, sell_confidence) * 0.5
        confidence = max(buy_confidence, sell_confidence) - conflict_penalty
        trade_action = "BUY" if buy_confidence > sell_confidence and fib_signal_triggered and trade_action == "BUY" else "SELL" if sell_confidence > buy_confidence and fib_signal_triggered and trade_action == "SELL" else None
    else:
        confidence = buy_confidence + sell_confidence
        trade_action = "BUY" if buy_confidence > 0 and fib_signal_triggered else "SELL" if sell_confidence > 0 and fib_signal_triggered else None

    confidence = min(confidence, 1.0)

    # Apply 38% confidence threshold
    if confidence < 0.38:
        trade_action = None
        entry_price = None
        stop_loss = None
        take_profit = None

    return signals, confidence, entry_price, stop_loss, take_profit, signal_descriptions, trade_action, fib_signal_triggered

def create_fibonacci_chart(df, trend, levels, window, entry_price=None, stop_loss=None, take_profit=None):
    fig = make_subplots(
        rows=3, cols=1, 
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=(f'Price with Fibonacci Levels (Based on {window} candles)', 'Volume', 'RSI'),
        row_width=[0.2, 0.2, 0.6]
    )
    
    fig.add_trace(
        go.Candlestick(
            x=df['closetime'],
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name='Price'
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=df['closetime'],
            y=df['ema'],
            name='EMA (20)',
            line=dict(color='orange', width=1.5)
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=df['closetime'],
            y=df['bb_upper'],
            name='BB Upper',
            line=dict(color='rgba(200, 200, 200, 0.5)', width=1),
            showlegend=False
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=df['closetime'],
            y=df['bb_lower'],
            name='BB Lower',
            line=dict(color='rgba(200, 200, 200, 0.5)', width=1),
            fill='tonexty',
            fillcolor='rgba(200, 200, 200, 0.1)',
            showlegend=False
        ),
        row=1, col=1
    )
    
    colors = ['#FF6B6B', '#FF9E6B', '#FFD166', '#06D6A0', '#118AB2', '#073B4C']
    fib_ratios = [0.236, 0.382, 0.5, 0.618, 0.786, 1.0]
    
    for i, (ratio, color) in enumerate(zip(fib_ratios, colors)):
        if i < len(levels):
            level_value = list(levels.values())[i]
            if not np.isnan(level_value):
                level_name = list(levels.keys())[i]
                fig.add_hline(
                    y=level_value, 
                    line_dash="dash", 
                    line_color=color,
                    annotation_text=f"{level_name}: {level_value:.2f}",
                    annotation_position="top right",
                    row=1, col=1
                )
    
    if entry_price is not None:
        fig.add_hline(
            y=entry_price, 
            line_dash="solid", 
            line_color="blue",
            annotation_text=f"Entry (Current Price): {entry_price:.2f}",
            row=1, col=1
        )
    
    if stop_loss is not None:
        fig.add_hline(
            y=stop_loss, 
            line_dash="solid", 
            line_color="red",
            annotation_text=f"Stop Loss: {stop_loss:.2f}",
            row=1, col=1
        )
    
    if take_profit is not None:
        fig.add_hline(
            y=take_profit, 
            line_dash="solid", 
            line_color="green",
            annotation_text=f"Take Profit: {take_profit:.2f}",
            row=1, col=1
        )
    
    colors_volume = ['red' if row['open'] > row['close'] else 'green' for _, row in df.iterrows()]
    fig.add_trace(
        go.Bar(
            x=df['closetime'],
            y=df['volume'],
            name='Volume',
            marker_color=colors_volume
        ),
        row=2, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=df['closetime'],
            y=df['rsi'],
            name='RSI',
            line=dict(color='purple', width=1.5)
        ),
        row=3, col=1
    )
    
    fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1, annotation_text="Overbought (70)")
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1, annotation_text="Oversold (30)")
    fig.add_hline(y=50, line_dash="dot", line_color="gray", row=3, col=1)
    
    fig.update_layout(
        height=800,
        title=f"Fibonacci Analysis - {trend.capitalize()}",
        yaxis_title='Price',
        xaxis_rangeslider_visible=False,
        template="plotly_dark",
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    fig.update_yaxes(title_text="RSI", row=3, col=1)
    fig.update_xaxes(title_text="Date", row=3, col=1)
    
    return fig

def run_fibonacci_analysis(symbol, interval, candle_limit, window, fib_threshold):
    df_live = fetch_candles_from_binance(symbol, interval, candle_limit)
    if df_live is None:
        return None
        
    df_live = calculate_indicators(df_live)
    trend, levels, high, low, close, latest = compute_fib_levels(df_live, window=window)
    signals, confidence, entry_price, stop_loss, take_profit, signal_descriptions, trade_action, fib_signal_triggered = generate_signals(
        df_live, trend, levels, latest, high, low, symbol, fib_threshold
    )
    
    current_price = fetch_current_price(symbol) or close
    chart_html = create_fibonacci_chart(df_live, trend, levels, window, entry_price, stop_loss, take_profit).to_html(full_html=False)
    
    # Generate Fibonacci levels HTML
    fib_html = None
    closest_info = ""
    fib_explanation = ""
    
    if levels:
        is_support = any('support' in key for key in levels.keys())
        level_type = "Support" if is_support else "Resistance"
        fib_df = pd.DataFrame.from_dict(levels, orient='index', columns=['Price Level'])
        fib_df.index.name = 'Fibonacci Level'
        fib_df['Type'] = fib_df.index.map(lambda x: 'Support' if 'support' in x else 'Resistance')
        fib_df['Distance %'] = ((current_price - fib_df['Price Level']) / fib_df['Price Level']) * 100
        fib_df['Strength'] = fib_df.index.map(lambda x: 'Strong' if any(r in x for r in ['0618', '0500', '0786']) else 'Moderate' if any(r in x for r in ['0382', '0236']) else 'Weak')
        fib_html = fib_df.to_html(formatters={'Price Level': '{:.8f}'.format, 'Distance %': '{:.2f}%'.format}, escape=False)
        closest_level = fib_df.iloc[np.argmin(np.abs(fib_df['Distance %']))]
        closest_info = f"Closest {level_type} level: {closest_level.name} at {closest_level['Price Level']:.4f} ({closest_level['Distance %']:.2f}% from current price)"
        
        if is_support:
            fib_explanation = f'''
            <div class="fib-explanation">
            <h4>📈 Uptrend Fibonacci Support Levels Analysis</h4>
            <p>In an <strong>uptrend</strong>, support levels are where price might bounce up.</p>
            <p><strong>Current Price: {current_price:.4f}</strong></p>
            <ul>
            <li><strong>23.6% Support ({levels.get('support_23', 0):.4f})</strong> - Shallow pullback. Buy with confirmation.</li>
            <li><strong>38.2% Support ({levels.get('support_38', 0):.4f})</strong> - Common pullback. Good buy spot.</li>
            <li><strong>50.0% Support ({levels.get('support_50', 0):.4f})</strong> - Psychological level. Buy if holds.</li>
            <li><strong>61.8% Support ({levels.get('support_61', 0):.4f})</strong> - Golden ratio. Strong buy signal.</li>
            <li><strong>78.6% Support ({levels.get('support_78', 0):.4f})</strong> - Deep pullback. Risky buy.</li>
            <li><strong>100% Support ({levels.get('support_100', 0):.4f})</strong> - Trend reversal if hit.</li>
            </ul>
            </div>
            '''
        else:
            fib_explanation = f'''
            <div class="fib-explanation">
            <h4>📉 Downtrend Fibonacci Resistance Levels Analysis</h4>
            <p>In a <strong>downtrend</strong>, resistance levels are where price might fall back.</p>
            <p><strong>Current Price: {current_price:.4f}</strong></p>
            <ul>
            <li><strong>23.6% Resistance ({levels.get('resistance_23', 0):.4f})</strong> - Shallow bounce. Sell with confirmation.</li>
            <li><strong>38.2% Resistance ({levels.get('resistance_38', 0):.4f})</strong> - Common bounce. Good sell spot.</li>
            <li><strong>50.0% Resistance ({levels.get('resistance_50', 0):.4f})</strong> - Psychological level. Sell if rejects.</li>
            <li><strong>61.8% Resistance ({levels.get('resistance_61', 0):.4f})</strong> - Golden ratio. Strong sell signal.</li>
            <li><strong>78.6% Resistance ({levels.get('resistance_78', 0):.4f})</strong> - Deep bounce. Risky sell.</li>
            <li><strong>100% Resistance ({levels.get('resistance_100', 0):.4f})</strong> - Trend reversal if hit.</li>
            </ul>
            </div>
            '''
    else:
        closest_info = "Market is in sideways movement - No strong Fibonacci levels identified"
    
    # Calculate risk/reward ratio
    risk_reward_ratio = None
    if entry_price and stop_loss and take_profit and trade_action:
        if trade_action == "BUY":
            risk_reward_ratio = abs((take_profit - entry_price) / (entry_price - stop_loss))
        else:
            risk_reward_ratio = abs((entry_price - take_profit) / (stop_loss - entry_price))
    
    # Technical indicators
    rsi_value = latest['rsi']
    ema_value = latest['ema']
    ema_rel = "Above" if current_price > ema_value else "Below"
    atr_value = latest['atr']
    bb_value = latest['bb_percentB'] * 100
    bb_status = "Overbought" if bb_value > 100 else "Oversold" if bb_value < 0 else "Neutral"
    price_change = ((latest['close'] - latest['open']) / latest['open']) * 100
    
    confidence_class = "confidence-high" if confidence > 0.6 else "confidence-medium" if confidence >= 0.38 else "confidence-low"
    
    return {
        'symbol': symbol,
        'interval': interval,
        'current_price': current_price,
        'swing_high': high,
        'swing_low': low,
        'trend': trend,
        'trend_class': f"trend-{trend}",
        'signals': signals,
        'confidence': confidence,
        'confidence_class': confidence_class,
        'entry_price': entry_price,
        'stop_loss': stop_loss,
        'take_profit': take_profit,
        'trade_action': trade_action,
        'signal_descriptions': signal_descriptions,
        'risk_reward_ratio': risk_reward_ratio,
        'chart_html': chart_html,
        'fib_html': fib_html,
        'closest_info': closest_info,
        'fib_explanation': fib_explanation,
        'rsi_value': rsi_value,
        'ema_value': ema_value,
        'ema_rel': ema_rel,
        'atr_value': atr_value,
        'bb_value': bb_value,
        'bb_status': bb_status,
        'price_stats': {
            'open': latest['open'],
            'high': latest['high'],
            'low': latest['low'],
            'close': latest['close'],
            'change': price_change
        },
        'volume_stats': {
            'volume': latest['volume'],
            'quote_volume': latest['quote_asset_volume'],
            'trades': latest['number_of_trades']
        },
        'last_candle': latest['closetime'],
        'data_from': df_live['closetime'].min(),
        'data_to': df_live['closetime'].max()
    }