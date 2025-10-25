import pandas as pd
import numpy as np
from utils import fetch_candles_from_binance, fetch_current_price, calculate_indicators
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from datetime import timedelta

def detect_ichimoku_signals(df, current_price, interval):
    if len(df) < 52:
        return {'trend': 'none', 'confidence': 'low', 'signals': [], 'cloud_bullish': False, 'reasons_not_met': [], 'ichimoku_values': {}}

    last = df.iloc[-1]
    prev = df.iloc[-2] if len(df) > 1 else last

    # Trend determination using cloud at current candle (26 candles back)
    if len(df) >= 26:
        cloud_idx = -26
        cloud_span_a = df['Senkou_Span_A_unshifted'].iloc[cloud_idx] if pd.notnull(df['Senkou_Span_A_unshifted'].iloc[cloud_idx]) else last['close']
        cloud_span_b = df['Senkou_Span_B_unshifted'].iloc[cloud_idx] if pd.notnull(df['Senkou_Span_B_unshifted'].iloc[cloud_idx]) else last['close']
        cloud_top = max(cloud_span_a, cloud_span_b)
        cloud_bottom = min(cloud_span_a, cloud_span_b)
        cloud_bullish = cloud_span_a > cloud_span_b
    else:
        cloud_top = last['close']
        cloud_bottom = last['close']
        cloud_bullish = False

    if last['close'] > cloud_top:
        trend = 'uptrend'
    elif last['close'] < cloud_bottom:
        trend = 'downtrend'
    else:
        trend = 'sideways'

    # Future cloud color (26 candles ahead)
    future_cloud_bullish = False
    if pd.notnull(last['Senkou_Span_A']) and pd.notnull(last['Senkou_Span_B']):
        future_cloud_bullish = last['Senkou_Span_A'] > last['Senkou_Span_B']

    # TK Configuration (current position)
    bull_config = last['Tenkan_sen'] > last['Kijun_sen']
    bear_config = last['Tenkan_sen'] < last['Kijun_sen']

    # TK distance check
    tk_diff = abs(last['Tenkan_sen'] - last['Kijun_sen']) / last['atr'] if pd.notnull(last['atr']) and last['atr'] > 0 else 0
    tk_not_close = tk_diff >= 0.5

    # Chikou confirmation
    if len(df) > 26:
        close_26_ago = df['close'].iloc[-27]
        chikou_bull = current_price > close_26_ago
        chikou_bear = current_price < close_26_ago
    else:
        chikou_bull = False
        chikou_bear = False

    # Volume trend
    volume_rising = False
    volume_sell_increasing = False
    if len(df) > 5:
        recent_volumes = df['volume'].tail(5)
        volume_rising = recent_volumes.is_monotonic_increasing
        sell_volumes = df['volume'].tail(5)[df['close'].tail(5) < df['open'].tail(5)]
        if len(sell_volumes) >= 2:
            volume_sell_increasing = sell_volumes.is_monotonic_increasing

    # Momentum filters
    rsi_bull = last['rsi'] > 50 if pd.notnull(last['rsi']) else False
    rsi_bear = last['rsi'] < 50 if pd.notnull(last['rsi']) else False
    macd_bull = last['macd'] > 0 if pd.notnull(last['macd']) else False
    macd_bear = last['macd'] < 0 if pd.notnull(last['macd']) else False

    # Three Ichimoku confirmations
    bull_three_met = bull_config and tk_not_close and chikou_bull and trend == 'uptrend' and cloud_bullish
    bear_three_met = bear_config and tk_not_close and chikou_bear and trend == 'downtrend' and not cloud_bullish

    # Confidence score
    score = 0
    if bull_three_met or bear_three_met:
        score = 3
        if future_cloud_bullish if bull_three_met else not future_cloud_bullish:
            score += 1
        if volume_rising if bull_three_met else volume_sell_increasing:
            score += 1
        if (rsi_bull or macd_bull) if bull_three_met else (rsi_bear or macd_bear):
            score += 1

    confidence = 'high' if score >= 5 else 'medium' if score >= 4 else 'low'

    # Signals
    signals = []
    reasons_not_met = []
    if not (bull_config or bear_config):
        reasons_not_met.append("Current Tenkan-sen and Kijun-sen configuration not met.")
    if not tk_not_close:
        reasons_not_met.append("Tenkan-sen and Kijun-sen are too close.")
    if not (chikou_bull or chikou_bear):
        reasons_not_met.append("Chikou Span condition not met.")
    if not ((trend == 'uptrend' and cloud_bullish) or (trend == 'downtrend' and not cloud_bullish)):
        reasons_not_met.append("Price vs Cloud condition not met.")
    if len(reasons_not_met) >= 2:
        reasons_not_met.append("Multiple Ichimoku conditions not met.")

    atr = last['atr'] if pd.notnull(last['atr']) else 0
    reason_parts = []
    if bull_three_met and confidence in ['high', 'medium']:
        reason_parts.append("Tenkan above Kijun with distance, Price above Green Cloud, Chikou confirmation")
        if future_cloud_bullish:
            reason_parts.append("Future Green Cloud")
        if volume_rising:
            reason_parts.append("Rising Volume")
        if rsi_bull or macd_bull:
            reason_parts.append("Momentum (RSI or MACD) confirmed")
        reason = "Bullish Ichimoku: " + ", ".join(reason_parts)
        sl = current_price - 1.5 * atr
        tp = current_price + 3 * atr
        signals.append({
            'type': 'BUY',
            'entry_price': current_price,
            'sl': sl,
            'tp': tp,
            'reason': reason
        })
    elif bear_three_met and confidence in ['high', 'medium']:
        reason_parts.append("Tenkan below Kijun with distance, Price below Red Cloud, Chikou confirmation")
        if not future_cloud_bullish:
            reason_parts.append("Future Red Cloud")
        if volume_sell_increasing:
            reason_parts.append("Sell Volume Increasing")
        if rsi_bear or macd_bear:
            reason_parts.append("Momentum (RSI or MACD) confirmed")
        reason = "Bearish Ichimoku: " + ", ".join(reason_parts)
        sl = current_price + 1.5 * atr
        tp = current_price - 3 * atr
        signals.append({
            'type': 'SELL',
            'entry_price': current_price,
            'sl': sl,
            'tp': tp,
            'reason': reason
        })

    ichimoku_values = {
        'Tenkan_sen': last['Tenkan_sen'],
        'Kijun_sen': last['Kijun_sen'],
        'Senkou_Span_A': last['Senkou_Span_A'] if pd.notnull(last['Senkou_Span_A']) else None,
        'Senkou_Span_B': last['Senkou_Span_B'] if pd.notnull(last['Senkou_Span_B']) else None,
        'close': last['close']
    }

    return {
        'trend': trend,
        'confidence': confidence,
        'signals': signals,
        'cloud_bullish': cloud_bullish,
        'reasons_not_met': reasons_not_met,
        'ichimoku_values': ichimoku_values
    }

def create_ichimoku_chart(raw_df, df_live, trend, indicators_to_show, show_ema, show_bb, interval):
    num_subplots = len(indicators_to_show) + 1
    row_heights = [0.6] + [0.4 / len(indicators_to_show)] * len(indicators_to_show) if indicators_to_show else [1.0]
    
    subplot_titles = [f'Ichimoku Cloud Analysis - {trend.capitalize()}']
    subplot_titles.extend(indicators_to_show)
    
    fig = make_subplots(
        rows=num_subplots, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=subplot_titles,
        row_heights=row_heights
    )
    
    fig.add_trace(
        go.Candlestick(
            x=raw_df['closetime'],
            open=raw_df['open'],
            high=raw_df['high'],
            low=raw_df['low'],
            close=raw_df['close'],
            name='Price'
        ),
        row=1, col=1
    )
    
    if show_ema:
        fig.add_trace(
            go.Scatter(
                x=df_live['closetime'],
                y=df_live['ema'],
                name='EMA (20)',
                line=dict(color='orange', width=1.5)
            ),
            row=1, col=1
        )
    
    if show_bb:
        fig.add_trace(
            go.Scatter(
                x=df_live['closetime'],
                y=df_live['bb_upper'],
                name='BB Upper',
                line=dict(color='rgba(200, 200, 200, 0.5)', width=1),
                showlegend=False
            ),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=df_live['closetime'],
                y=df_live['bb_lower'],
                name='BB Lower',
                line=dict(color='rgba(200, 200, 200, 0.5)', width=1),
                fill='tonexty',
                fillcolor='rgba(200, 200, 200, 0.1)',
                showlegend=False
            ),
            row=1, col=1
        )
    
    # Ichimoku Lines
    fig.add_trace(
        go.Scatter(
            x=df_live['closetime'],
            y=df_live['Tenkan_sen'],
            name='Tenkan-sen',
            line=dict(color='#0000FF', width=1.5)
        ),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=df_live['closetime'],
            y=df_live['Kijun_sen'],
            name='Kijun-sen',
            line=dict(color='#FF0000', width=1.5)
        ),
        row=1, col=1
    )
    
    # Chikou Span
    ichimoku_kijun = 26
    if len(df_live) > ichimoku_kijun:
        chikou_x = df_live['closetime'][:-ichimoku_kijun]
        chikou_y = df_live['close'][ichimoku_kijun:]
        fig.add_trace(
            go.Scatter(
                x=chikou_x,
                y=chikou_y,
                name='Chikou Span',
                line=dict(color='#800080', width=1.5)
            ),
            row=1, col=1
        )
    
    # Calculate time shift for cloud
    interval_minutes = {
        '1m': 1, '3m': 3, '5m': 5, '15m': 15, '30m': 30,
        '1h': 60, '2h': 120, '4h': 240, '6h': 360, '8h': 480, '12h': 720,
        '1d': 1440, '3d': 4320, '1w': 10080, '1M': 43200
    }
    period_minutes = interval_minutes.get(interval, 5)
    shift_timedelta = timedelta(minutes=period_minutes * ichimoku_kijun)
    
    # Shift cloud x-axis
    cloud_closetime = df_live['closetime'] + shift_timedelta
    
    # Plot Senkou Span lines
    fig.add_trace(
        go.Scatter(
            x=cloud_closetime,
            y=df_live['Senkou_Span_A'],
            name='Senkou Span A',
            line=dict(color='#00FF00', width=1, dash='dot')
        ),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=cloud_closetime,
            y=df_live['Senkou_Span_B'],
            name='Senkou Span B',
            line=dict(color='#FF0000', width=1, dash='dot')
        ),
        row=1, col=1
    )
    
    # Single cloud fill with dynamic color
    df_cloud = df_live[['closetime', 'Senkou_Span_A', 'Senkou_Span_B']].copy()
    df_cloud['closetime'] = cloud_closetime
    df_cloud['upper'] = df_cloud[['Senkou_Span_A', 'Senkou_Span_B']].max(axis=1)
    df_cloud['lower'] = df_cloud[['Senkou_Span_A', 'Senkou_Span_B']].min(axis=1)
    df_cloud['color'] = np.where(df_cloud['Senkou_Span_A'] >= df_cloud['Senkou_Span_B'], 'rgba(144, 238, 144, 0.15)', 'rgba(255, 182, 193, 0.15)')
    
    # Split cloud into segments
    segments = []
    current_segment = {'closetime': [], 'upper': [], 'lower': [], 'color': df_cloud['color'].iloc[0]}
    for i in range(len(df_cloud)):
        if i > 0 and (df_cloud['color'].iloc[i] != current_segment['color'] or pd.isna(df_cloud['upper'].iloc[i]) or pd.isna(df_cloud['lower'].iloc[i])):
            if current_segment['closetime']:
                segments.append(current_segment)
            current_segment = {'closetime': [], 'upper': [], 'lower': [], 'color': df_cloud['color'].iloc[i] if not pd.isna(df_cloud['upper'].iloc[i]) else None}
        if not pd.isna(df_cloud['upper'].iloc[i]) and not pd.isna(df_cloud['lower'].iloc[i]):
            current_segment['closetime'].append(df_cloud['closetime'].iloc[i])
            current_segment['upper'].append(df_cloud['upper'].iloc[i])
            current_segment['lower'].append(df_cloud['lower'].iloc[i])
    if current_segment['closetime']:
        segments.append(current_segment)
    
    # Add cloud segments
    cloud_legend_added = False
    for segment in segments:
        if segment['color']:
            fig.add_trace(
                go.Scatter(
                    x=segment['closetime'],
                    y=segment['upper'],
                    mode='lines',
                    line=dict(color='rgba(0,0,0,0)'),
                    showlegend=False
                ),
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(
                    x=segment['closetime'],
                    y=segment['lower'],
                    mode='lines',
                    fill='tonexty',
                    fillcolor=segment['color'],
                    line=dict(color='rgba(0,0,0,0)'),
                    name='Cloud',
                    showlegend=not cloud_legend_added
                ),
                row=1, col=1
            )
            cloud_legend_added = True
    
    current_row = 2
    if "Volume" in indicators_to_show:
        colors_volume = ['red' if row['open'] > row['close'] else 'green' for _, row in raw_df.iterrows()]
        fig.add_trace(
            go.Bar(
                x=raw_df['closetime'],
                y=raw_df['volume'],
                name='Volume',
                marker_color=colors_volume
            ),
            row=current_row, col=1
        )
        fig.update_yaxes(title_text="Volume", row=current_row, col=1)
        current_row += 1
    
    if "RSI" in indicators_to_show:
        fig.add_trace(
            go.Scatter(
                x=df_live['closetime'],
                y=df_live['rsi'],
                name='RSI',
                line=dict(color='purple', width=1.5)
            ),
            row=current_row, col=1
        )
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=current_row, col=1, annotation_text="Overbought (70)")
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=current_row, col=1, annotation_text="Oversold (30)")
        fig.add_hline(y=50, line_dash="dot", line_color="gray", row=current_row, col=1)
        fig.update_yaxes(title_text="RSI", row=current_row, col=1)
        current_row += 1
    
    if "MACD" in indicators_to_show:
        fig.add_trace(
            go.Scatter(
                x=df_live['closetime'],
                y=df_live['macd'],
                name='MACD Line',
                line=dict(color='blue', width=1.5)
            ),
            row=current_row, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=df_live['closetime'],
                y=df_live['macd_signal'],
                name='Signal Line',
                line=dict(color='orange', width=1.5)
            ),
            row=current_row, col=1
        )
        histogram_colors = ['green' if val >= 0 else 'red' for val in df_live['macd_histogram']]
        fig.add_trace(
            go.Bar(
                x=df_live['closetime'],
                y=df_live['macd_histogram'],
                name='MACD Histogram',
                marker_color=histogram_colors,
                opacity=0.5
            ),
            row=current_row, col=1
        )
        fig.add_hline(y=0, line_dash="dot", line_color="gray", row=current_row, col=1)
        fig.update_yaxes(title_text="MACD", row=current_row, col=1)
    
    fig.update_layout(
        height=900,
        title=f"Ichimoku Cloud Analysis - {trend.capitalize()}",
        yaxis_title='Price',
        template="plotly_dark",
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        dragmode='zoom',
        hovermode='x unified'
    )
    fig.update_xaxes(
        title_text="Date",
        row=num_subplots, col=1,
        rangeselector=dict(
            buttons=list([
                dict(count=1, label="1m", step="month", stepmode="backward"),
                dict(count=6, label="6m", step="month", stepmode="backward"),
                dict(count=1, label="YTD", step="year", stepmode="todate"),
                dict(count=1, label="1y", step="year", stepmode="backward"),
                dict(step="all")
            ])
        )
    )
    
    if len(raw_df) > 0:
        initial_start_idx = max(0, len(raw_df) - 200)
        initial_start = raw_df['closetime'].iloc[initial_start_idx]
        initial_end = raw_df['closetime'].iloc[-1] + shift_timedelta
        for r in range(1, num_subplots + 1):
            fig.update_xaxes(range=[initial_start, initial_end], row=r, col=1, matches='x')
    
    return fig

def run_ichimoku_analysis(symbol, interval, candle_limit, show_ema=True, show_bb=True, show_volume=True, show_rsi=True, show_macd=True):
    print(f"🔍 Starting Ichimoku analysis for {symbol} {interval}")
    
    try:
        df_live = fetch_candles_from_binance(symbol, interval, candle_limit)
        if df_live is None or df_live.empty:
            print(f"❌ No data fetched for {symbol}")
            return None
            
        print(f"✅ Data fetched: {len(df_live)} candles")
        
        raw_df = df_live.copy()
        df_live = calculate_indicators(df_live)
        
        print(f"✅ Indicators calculated")
        
        current_price = fetch_current_price(symbol) or df_live['close'].iloc[-1]
        
        ichimoku_data = detect_ichimoku_signals(df_live, current_price, interval)
        
        # Create chart
        indicators_to_show = []
        if show_volume:
            indicators_to_show.append("Volume")
        if show_rsi:
            indicators_to_show.append("RSI")
        if show_macd:
            indicators_to_show.append("MACD")
        
        chart = create_ichimoku_chart(raw_df, df_live, ichimoku_data['trend'], indicators_to_show, show_ema, show_bb, interval)
        
        # Get technical indicators
        latest = df_live.iloc[-1]
        tech_indicators = {
            'rsi': latest['rsi'],
            'ema': latest['ema'],
            'atr': latest['atr'],
            'bb_percent': latest['bb_percentB'] * 100,
            'tenkan_sen': latest['Tenkan_sen'],
            'kijun_sen': latest['Kijun_sen'],
            'senkou_span_a': latest['Senkou_Span_A'],
            'senkou_span_b': latest['Senkou_Span_B']
        }
        
        result = {
            'symbol': symbol,
            'interval': interval,
            'current_price': current_price,
            'trend': ichimoku_data['trend'],
            'confidence': ichimoku_data['confidence'],
            'signals': ichimoku_data['signals'],
            'cloud_bullish': ichimoku_data['cloud_bullish'],
            'reasons_not_met': ichimoku_data['reasons_not_met'],
            'ichimoku_values': ichimoku_data['ichimoku_values'],
            'chart_html': chart.to_html(full_html=False),
            'technical_indicators': tech_indicators
        }
        
        print(f"✅ Ichimoku analysis completed for {symbol}")
        return result
        
    except Exception as e:
        print(f"❌ Error in Ichimoku analysis for {symbol}: {e}")
        import traceback
        traceback.print_exc()
        return None