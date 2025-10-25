import pandas as pd
import numpy as np
from utils import fetch_candles_from_binance, fetch_current_price, calculate_indicators
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import math

def calculate_gann_fan_angles(df, pivot_price, pivot_index, trend_direction='bullish', price_scale=None):
    angles = {}
    gann_ratios = {
        '1x1': (1, 1),    # 45°
        '1x2': (1, 2),    # 63.75°
        '2x1': (2, 1),    # 26.25°
        '1x4': (1, 4),    # 75°
        '4x1': (4, 1),    # 15°
        '1x8': (1, 8),    # 82.5°
        '8x1': (8, 1),    # 7.5°
        '3x1': (3, 1),    # 18.75°
        '1x3': (1, 3)     # 71.25°
    }
    
    if price_scale is None:
        if len(df) > 0 and 'atr' in df and pd.notnull(df['atr'].iloc[-1]):
            price_scale = df['atr'].iloc[-1] * 0.1
        else:
            price_scale = df['close'].iloc[-1] * 0.001 if len(df) > 0 else 0.01

    for angle_name, (price_ratio, time_ratio) in gann_ratios.items():
        slope = (price_scale * price_ratio) / time_ratio
        if trend_direction == 'bullish':
            angle_values = []
            for i in range(len(df)):
                if i >= pivot_index:
                    time_diff = i - pivot_index
                    angle_value = pivot_price + (time_diff * slope)
                    angle_values.append(angle_value)
                else:
                    angle_values.append(np.nan)
        else:
            angle_values = []
            for i in range(len(df)):
                if i >= pivot_index:
                    time_diff = i - pivot_index
                    angle_value = pivot_price - (time_diff * slope)
                    angle_values.append(angle_value)
                else:
                    angle_values.append(np.nan)
        angles[angle_name] = angle_values
    
    return angles, price_scale

def gann_square_of_9(price, levels=8):
    if price <= 0:
        return []
    sqrt_price = math.sqrt(price)
    base_levels = []
    for i in range(-levels, levels + 1):
        level = (sqrt_price + i * 0.125) ** 2
        if level > 0:
            base_levels.append(level)
    unique_levels = sorted(set(round(level, 4) for level in base_levels))
    price_range = price * 0.5, price * 1.5
    filtered_levels = [level for level in unique_levels if price_range[0] <= level <= price_range[1]]
    return filtered_levels[:16]

def gann_box(df, interval, lookback=50):
    if len(df) == 0:
        return [], []
    
    effective_lookback = min(lookback, len(df))
    price_min = df['low'].tail(effective_lookback).min()
    price_max = df['high'].tail(effective_lookback).max()
    price_range = price_max - price_min
    ratios = [0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875]
    price_levels = [price_min + r * price_range for r in ratios]
    
    # Get the current time and ensure it's a pandas Timestamp
    time_current = pd.to_datetime(df['closetime'].iloc[-1])
    
    # Calculate time levels using proper pandas frequency arithmetic
    time_levels = []
    
    # Determine appropriate time delta based on interval
    if 'm' in interval:
        # For minute intervals, use minutes
        minutes_per_ratio = 24 * 60  # 24 hours in minutes
        for r in ratios:
            minutes_to_add = r * minutes_per_ratio
            time_point = time_current + pd.Timedelta(minutes=minutes_to_add)
            time_levels.append(time_point)
    elif 'h' in interval:
        # For hour intervals, use hours
        hours_per_ratio = 24  # 24 hours
        for r in ratios:
            hours_to_add = r * hours_per_ratio
            time_point = time_current + pd.Timedelta(hours=hours_to_add)
            time_levels.append(time_point)
    elif 'd' in interval:
        # For day intervals, use days
        days_per_ratio = 30  # 30 days as base
        for r in ratios:
            days_to_add = r * days_per_ratio
            time_point = time_current + pd.Timedelta(days=days_to_add)
            time_levels.append(time_point)
    else:
        # Default fallback - use hours
        for r in ratios:
            hours_to_add = r * 24
            time_point = time_current + pd.Timedelta(hours=hours_to_add)
            time_levels.append(time_point)
    
    return price_levels, time_levels

def gann_square_fixed(high, low, current_price):
    if high <= low:
        return {}
    price_range = high - low
    ratios = [0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0]
    levels = {}
    for ratio in ratios:
        level = low + ratio * price_range
        levels[f'{int(ratio*100)}%'] = level
    extension_ratios = [1.125, 1.25, 1.375, 1.5, 1.625, 1.75, 1.875, 2.0]
    for ratio in extension_ratios:
        level = low + ratio * price_range
        levels[f'{int(ratio*100)}%'] = level
    return levels

def generate_gann_analysis_details(df, symbol, interval, current_price, gann_subtools, pivot_choice):
    """Generate detailed analysis results for each Gann tool"""
    details = {}
    
    if len(df) == 0:
        return details
    
    # Determine trend and pivot
    use_pivot_low = False
    trend_direction = 'bullish'
    if pivot_choice == 'Auto (based on trend)':
        if 'ema' in df and 'ema50' in df and len(df) > 0:
            trend_direction = 'bullish' if df['ema'].iloc[-1] > df['ema50'].iloc[-1] else 'bearish'
            use_pivot_low = (trend_direction == 'bullish')
    elif pivot_choice == 'Latest Pivot Low':
        use_pivot_low = True
        trend_direction = 'bullish'
    elif pivot_choice == 'Latest Pivot High':
        use_pivot_low = False
        trend_direction = 'bearish'
    
    # Calculate significant levels
    lookback = min(50, len(df))
    significant_low = df['low'].tail(lookback).min()
    significant_high = df['high'].tail(lookback).max()
    
    last = df.iloc[-1]
    
    # Gann Fan details
    if 'Gann Fan' in gann_subtools:
        pivot_price = 0
        pivot_type = 'N/A'
        key_levels = []
        if 'pivot_low_price' in last and pd.notnull(last['pivot_low_price']):
            pivot_price = last['pivot_low_price']
            pivot_type = 'Pivot Low'
        elif 'pivot_high_price' in last and pd.notnull(last['pivot_high_price']):
            pivot_price = last['pivot_high_price']
            pivot_type = 'Pivot High'
        
        for angle_name in ['1x1', '2x1', '1x2']:
            col = f'gann_{angle_name}'
            if col in last and pd.notnull(last[col]):
                key_levels.append(f"{angle_name}: {last[col]:.4f}")
        
        signal = 'Neutral'
        strength = 'Low'
        if key_levels:
            if current_price > last.get('gann_1x1', 0):
                signal = 'Bullish'
            elif current_price < last.get('gann_1x1', 0):
                signal = 'Bearish'
            strength = 'Medium'
        
        details['Gann Fan'] = {
            'description': 'Gann Fan lines show support and resistance at various angles from a pivot point.',
            'pivot_price': pivot_price,
            'current_angle': '45°',
            'key_levels': ', '.join(key_levels) if key_levels else 'No levels detected',
            'signal': signal,
            'strength': strength,
            'pivot_type': pivot_type,
            'trend_direction': trend_direction
        }
    
    # Gann Square details
    if 'Gann Square' in gann_subtools:
        square_levels = last.get('gann_square_levels_current', [])
        nearest_level = min(square_levels, key=lambda x: abs(x - current_price)) if square_levels else 0
        
        details['Gann Square'] = {
            'description': 'Gann Square identifies price levels based on squared price and time relationships.',
            'square_root': math.sqrt(current_price) if current_price > 0 else 0,
            'price_square': current_price ** 2,
            'key_levels': f"Nearest: {nearest_level:.4f}, Levels: {len(square_levels)}",
            'signal': 'Consolidation' if len(square_levels) > 5 else 'Trending',
            'strength': 'Strong' if abs(current_price - nearest_level) < current_price * 0.02 else 'Medium',
            'levels_count': len(square_levels),
            'nearest_level': nearest_level
        }
    
    # Gann Box details
    if 'Gann Box' in gann_subtools:
        price_levels = last.get('gann_box_price_levels', [])
        if price_levels:
            box_top = max(price_levels)
            box_bottom = min(price_levels)
            box_center = (box_top + box_bottom) / 2
            signal = 'Inside Box' if box_bottom <= current_price <= box_top else 'Outside Box'
            strength = 'Strong'
        else:
            box_top = significant_high
            box_bottom = significant_low
            box_center = (box_top + box_bottom) / 2
            signal = 'No Box Calculated'
            strength = 'Low'
        
        details['Gann Box'] = {
            'description': 'Gann Box creates a box around price action to identify breakout points.',
            'box_top': box_top,
            'box_bottom': box_bottom,
            'box_center': box_center,
            'signal': signal,
            'strength': strength,
            'price_levels_count': len(price_levels),
            'time_levels_count': len(last.get('gann_box_time_levels', []))
        }
    
    # Gann Square Fixed details
    if 'Gann Square Fixed' in gann_subtools:
        fixed_levels = last.get('gann_square_fixed_levels', {})
        support_count = len([l for l in fixed_levels.values() if l < current_price])
        resistance_count = len([l for l in fixed_levels.values() if l > current_price])
        
        details['Gann Square Fixed'] = {
            'description': 'Fixed Gann Square uses fixed intervals to project support and resistance.',
            'fixed_interval': (significant_high - significant_low) * 0.125,
            'levels_count': len(fixed_levels),
            'key_levels': f"{support_count} support, {resistance_count} resistance",
            'signal': 'Multiple Levels Active',
            'strength': 'Medium',
            'support_levels': support_count,
            'resistance_levels': resistance_count
        }
    
    return details

def create_gann_chart_compatible(df, symbol, interval, gann_tool, analysis_details, show_ema, show_bb, show_volume, show_rsi, show_macd):
    if len(df) == 0:
        return "<p>No data available for chart</p>"
    
    current_price = df['close'].iloc[-1]
    trend = analysis_details.get(gann_tool, {}).get('signal', 'Neutral')  # Use signal as trend for chart title
    
    indicators_to_show = []
    if show_volume:
        indicators_to_show.append("Volume")
    if show_rsi:
        indicators_to_show.append("RSI")
    if show_macd:
        indicators_to_show.append("MACD")
    
    num_subplots = len(indicators_to_show) + 1
    row_heights = [0.6] + [0.4 / len(indicators_to_show)] * len(indicators_to_show) if indicators_to_show else [1.0]
    subplot_titles = [f'{gann_tool} Analysis - {trend.capitalize()} Trend'] + indicators_to_show
    
    fig = make_subplots(
        rows=num_subplots, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=subplot_titles,
        row_heights=row_heights
    )
    
    fig.add_trace(
        go.Candlestick(
            x=df['closetime'],
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name='Price',
            increasing_line_color='#26A69A',
            decreasing_line_color='#EF5350',
            increasing_fillcolor='#26A69A',
            decreasing_fillcolor='#EF5350',
            line=dict(width=0.5)
        ),
        row=1, col=1
    )
    
    if show_ema and 'ema' in df and 'ema50' in df:
        fig.add_trace(
            go.Scatter(
                x=df['closetime'],
                y=df['ema'],
                name='EMA (20)',
                line=dict(color='#FFA500', width=1.5)
            ),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=df['closetime'],
                y=df['ema50'],
                name='EMA (50)',
                line=dict(color='#800080', width=1.5)
            ),
            row=1, col=1
        )
    
    if show_bb and 'bb_upper' in df and 'bb_lower' in df:
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
    
    # Add Gann tool specific elements
    if gann_tool == 'Gann Fan':
        angle_styles = [
            ('gann_1x1', '#0000FF', 'dash', '1x1 (45°)'),
            ('gann_2x1', '#00FF00', 'dot', '2x1 (26.25°)'),
            ('gann_1x2', '#FF0000', 'dot', '1x2 (63.75°)'),
            ('gann_4x1', '#00FFFF', 'dashdot', '4x1 (15°)'),
            ('gann_1x4', '#FF00FF', 'dashdot', '1x4 (75°)'),
            ('gann_8x1', '#FFFF00', 'solid', '8x1 (7.5°)'),
            ('gann_3x1', '#FF4500', 'dash', '3x1 (18.75°)'),
            ('gann_1x3', '#4B0082', 'dot', '1x3 (71.25°)'),
            ('gann_1x8', '#808000', 'dashdot', '1x8 (82.5°)')
        ]
        for angle, color, dash, label in angle_styles:
            if angle in df and not df[angle].isna().all():
                valid_data = df[df[angle].notna()]
                if not valid_data.empty:
                    fig.add_trace(
                        go.Scatter(
                            x=valid_data['closetime'],
                            y=valid_data[angle],
                            name=label,
                            line=dict(color=color, width=2 if angle == 'gann_1x1' else 1.5, dash=dash),
                            opacity=0.8
                        ),
                        row=1, col=1
                    )
        
        if 'pivot_low' in df:
            pivot_lows = df[df['pivot_low'] == True]
            if not pivot_lows.empty:
                fig.add_trace(
                    go.Scatter(
                        x=pivot_lows['closetime'],
                        y=pivot_lows['low'],
                        mode='markers+text',
                        name='Pivot Low',
                        marker=dict(symbol='triangle-up', color='#00FF00', size=12),
                        text=['PL' for _ in range(len(pivot_lows))],
                        textposition='bottom center',
                        textfont=dict(color='#00FF00', size=10)
                    ),
                    row=1, col=1
                )
        if 'pivot_high' in df:
            pivot_highs = df[df['pivot_high'] == True]
            if not pivot_highs.empty:
                fig.add_trace(
                    go.Scatter(
                        x=pivot_highs['closetime'],
                        y=pivot_highs['high'],
                        mode='markers+text',
                        name='Pivot High',
                        marker=dict(symbol='triangle-down', color='#FF0000', size=12),
                        text=['PH' for _ in range(len(pivot_highs))],
                        textposition='top center',
                        textfont=dict(color='#FF0000', size=10)
                    ),
                    row=1, col=1
                )
    
    if gann_tool == 'Gann Square':
        levels_low = df['gann_square_levels_low'].iloc[-1][:8] if 'gann_square_levels_low' in df else []
        levels_high = df['gann_square_levels_high'].iloc[-1][:8] if 'gann_square_levels_high' in df else []
        levels_current = df['gann_square_levels_current'].iloc[-1][:8] if 'gann_square_levels_current' in df else []
        for i, level in enumerate(levels_low):
            fig.add_hline(
                y=level, 
                line_dash="dot", 
                line_color="#00FFFF", 
                row=1, col=1, 
                annotation_text=f"SqL{i+1}",
                annotation_position="right",
                opacity=0.7
            )
        for i, level in enumerate(levels_high):
            fig.add_hline(
                y=level, 
                line_dash="dot", 
                line_color="#FF00FF", 
                row=1, col=1, 
                annotation_text=f"SqH{i+1}",
                annotation_position="right",
                opacity=0.7
            )
        for i, level in enumerate(levels_current[:4]):
            fig.add_hline(
                y=level, 
                line_dash="solid", 
                line_color="#FFFFFF", 
                line_width=2,
                row=1, col=1, 
                annotation_text=f"Curr{i+1}",
                annotation_position="left",
                opacity=0.9
            )
    
    if gann_tool == 'Gann Square Fixed':
        levels_dict = df['gann_square_fixed_levels'].iloc[-1] if 'gann_square_fixed_levels' in df else {}
        if levels_dict:
            for ratio, level in list(levels_dict.items())[:10]:
                fig.add_hline(
                    y=level, 
                    line_dash="solid" if '100' in ratio else "dot", 
                    line_color="#008000", 
                    row=1, col=1, 
                    annotation_text=f"{ratio}",
                    annotation_position="right",
                    opacity=0.8
                )
    
    if gann_tool == 'Gann Box':
        price_levels = df['gann_box_price_levels'].iloc[-1] if 'gann_box_price_levels' in df else []
        time_levels = df['gann_box_time_levels'].iloc[-1] if 'gann_box_time_levels' in df else []
        
        for i, price_level in enumerate(price_levels):
            fig.add_hline(
                y=price_level,
                line_dash="dash",
                line_color="#FFA500",
                row=1, col=1,
                annotation_text=f"BoxP{i+1}",
                annotation_position="left",
                opacity=0.7
            )
        
        # Fixed: Ensure time_levels are properly converted to datetime
        for i, time_level in enumerate(time_levels):
            try:
                # Convert to datetime if not already
                if not isinstance(time_level, pd.Timestamp):
                    time_level = pd.to_datetime(time_level)
                
                fig.add_vline(
                    x=time_level,
                    line_dash="dash",
                    line_color="#800080",
                    row=1, col=1,
                    annotation_text=f"BoxT{i+1}",
                    annotation_position="top",
                    opacity=0.7
                )
            except Exception as e:
                print(f"Warning: Could not add time level {i}: {e}")
                continue
    
    current_row = 2
    if show_volume and 'volume' in df:
        colors_volume = ['#EF5350' if o > c else '#26A69A' for o, c in zip(df['open'], df['close'])]
        fig.add_trace(
            go.Bar(
                x=df['closetime'],
                y=df['volume'],
                name='Volume',
                marker_color=colors_volume,
                opacity=0.8
            ),
            row=current_row, col=1
        )
        current_row += 1
    
    if show_rsi and 'rsi' in df:
        fig.add_trace(
            go.Scatter(
                x=df['closetime'],
                y=df['rsi'],
                name='RSI',
                line=dict(color='#800080', width=2)
            ),
            row=current_row, col=1
        )
        fig.add_hline(y=70, line_dash="dash", line_color="#EF5350", row=current_row, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="#26A69A", row=current_row, col=1)
        fig.add_hline(y=50, line_dash="dot", line_color="#808080", row=current_row, col=1)
        current_row += 1
    
    if show_macd and 'macd' in df:
        fig.add_trace(
            go.Scatter(
                x=df['closetime'],
                y=df['macd'],
                name='MACD Line',
                line=dict(color='#0000FF', width=2)
            ),
            row=current_row, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=df['closetime'],
                y=df['macd_signal'],
                name='Signal Line',
                line=dict(color='#FFA500', width=2)
            ),
            row=current_row, col=1
        )
        if 'macd_histogram' in df:
            histogram_colors = ['#26A69A' if val >= 0 else '#EF5350' for val in df['macd_histogram']]
            fig.add_trace(
                go.Bar(
                    x=df['closetime'],
                    y=df['macd_histogram'],
                    name='MACD Histogram',
                    marker_color=histogram_colors,
                    opacity=0.8
                ),
                row=current_row, col=1
            )
        fig.add_hline(y=0, line_dash="dot", line_color="#808080", row=current_row, col=1)
    
    fig.update_layout(
        height=800,
        title=f"{gann_tool} Analysis - {symbol} ({interval}) | Price: {current_price:.4f}",
        yaxis_title='Price (USDT)',
        template="plotly_dark",
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode='x unified',
        plot_bgcolor='#141414',
        paper_bgcolor='#141414',
        font=dict(color='#FFFFFF', family="Arial", size=12),
        margin=dict(l=60, r=60, t=100, b=60)
    )
    
    fig.update_xaxes(
        rangeselector=dict(
            buttons=list([
                dict(count=1, label="1m", step="month", stepmode="backward"),
                dict(count=6, label="6m", step="month", stepmode="backward"),
                dict(count=1, label="YTD", step="year", stepmode="todate"),
                dict(count=1, label="1y", step="year", stepmode="backward"),
                dict(step="all")
            ]),
            bgcolor='#2A2A2A',
            font=dict(color='#FFFFFF')
        ),
        showgrid=True,
        gridcolor='rgba(255,255,255,0.05)',
        zeroline=False,
        tickformat='%Y-%m-%d %H:%M'
    )
    
    fig.update_yaxes(
        showgrid=True,
        gridcolor='rgba(255,255,255,0.05)',
        zeroline=False,
        tickformat='.4f'
    )
    
    # Initial zoom to last 200 candles
    if len(df) > 0:
        initial_start_idx = max(0, len(df) - 200)
        initial_start = df['closetime'].iloc[initial_start_idx]
        initial_end = df['closetime'].iloc[-1]
        fig.update_xaxes(range=[initial_start, initial_end], row=1, col=1)
    
    return fig.to_html(full_html=False, include_plotlyjs=False)

def run_gann_analysis(symbol, interval, candle_limit, gann_subtools, pivot_choice, 
                     show_ema=True, show_bb=True, show_volume=True, show_rsi=True, show_macd=True):
    """Main Gann analysis function compatible with Flask app"""
    print(f"🔍 Starting Gann analysis for {symbol} {interval}")
    
    try:
        # Fetch and prepare data
        df = fetch_candles_from_binance(symbol, interval, candle_limit)
        if df is None or df.empty:
            print(f"❌ No data fetched for {symbol}")
            return None
            
        print(f"✅ Data fetched: {len(df)} candles")
        
        # Calculate indicators
        df = calculate_indicators(df)
        current_price = fetch_current_price(symbol) or df['close'].iloc[-1]
        
        print(f"✅ Indicators calculated, current price: {current_price}")
        
        # Calculate pivot highs and lows
        pivot_window = 5  # Lookback periods on each side
        df['pivot_low'] = df['low'] == df['low'].rolling(window=2 * pivot_window + 1, center=True).min()
        df['pivot_high'] = df['high'] == df['high'].rolling(window=2 * pivot_window + 1, center=True).max()
        df['pivot_low'] = df['pivot_low'].fillna(False)
        df['pivot_high'] = df['pivot_high'].fillna(False)
        
        print(f"✅ Pivots calculated: {df['pivot_low'].sum()} lows, {df['pivot_high'].sum()} highs")
        
        # Determine trend and pivot
        use_pivot_low = False
        trend_direction = 'bullish'
        if pivot_choice == 'Auto (based on trend)':
            if 'ema' in df and 'ema50' in df and len(df) > 0:
                trend_direction = 'bullish' if df['ema'].iloc[-1] > df['ema50'].iloc[-1] else 'bearish'
                use_pivot_low = (trend_direction == 'bullish')
        elif pivot_choice == 'Latest Pivot Low':
            use_pivot_low = True
            trend_direction = 'bullish'
        elif pivot_choice == 'Latest Pivot High':
            use_pivot_low = False
            trend_direction = 'bearish'
        
        # Calculate significant levels
        lookback = min(50, len(df))
        significant_low = df['low'].tail(lookback).min()
        significant_high = df['high'].tail(lookback).max()
        
        # Add Gann calculations to df based on selected tools
        if 'Gann Fan' in gann_subtools:
            pivot_idx = None
            pivot_price = None
            if use_pivot_low:
                pivot_lows = df[df['pivot_low'] == True]
                if not pivot_lows.empty:
                    pivot_idx = pivot_lows.index[-1]
                    pivot_price = df['low'].iloc[pivot_idx]
                    df['pivot_low_price'] = np.nan
                    df.loc[pivot_idx:, 'pivot_low_price'] = pivot_price
            else:
                pivot_highs = df[df['pivot_high'] == True]
                if not pivot_highs.empty:
                    pivot_idx = pivot_highs.index[-1]
                    pivot_price = df['high'].iloc[pivot_idx]
                    df['pivot_high_price'] = np.nan
                    df.loc[pivot_idx:, 'pivot_high_price'] = pivot_price
            
            if pivot_idx is not None:
                recent_atr = df['atr'].iloc[pivot_idx:].mean() if 'atr' in df else 0
                price_scale = recent_atr * 0.1 if recent_atr > 0 else current_price * 0.001
                angles, calculated_scale = calculate_gann_fan_angles(df, pivot_price, pivot_idx, trend_direction, price_scale)
                for angle_name, values in angles.items():
                    df[f'gann_{angle_name}'] = values
                df['gann_price_scale'] = calculated_scale
        
        if 'Gann Square' in gann_subtools:
            df['gann_square_levels_low'] = [gann_square_of_9(significant_low)] * len(df)
            df['gann_square_levels_high'] = [gann_square_of_9(significant_high)] * len(df)
            df['gann_square_levels_current'] = [gann_square_of_9(current_price)] * len(df)
        
        if 'Gann Box' in gann_subtools:
            price_levels, time_levels = gann_box(df, interval, 50)
            df['gann_box_price_levels'] = [price_levels] * len(df)
            df['gann_box_time_levels'] = [time_levels] * len(df)
        
        if 'Gann Square Fixed' in gann_subtools:
            fixed_levels = gann_square_fixed(significant_high, significant_low, current_price)
            df['gann_square_fixed_levels'] = [fixed_levels] * len(df)
        
        # Generate analysis details
        analysis_details = generate_gann_analysis_details(
            df, symbol, interval, current_price, gann_subtools, pivot_choice
        )
        
        # Create charts for each selected tool
        charts = {}
        for tool in gann_subtools:
            if tool in analysis_details:
                try:
                    chart_html = create_gann_chart_compatible(
                        df, symbol, interval, tool, analysis_details, 
                        show_ema, show_bb, show_volume, show_rsi, show_macd
                    )
                    charts[tool] = chart_html
                    print(f"✅ Chart created for {tool}")
                except Exception as e:
                    print(f"❌ Error creating chart for {tool}: {e}")
                    charts[tool] = f"<p>Error creating {tool} chart</p>"
            else:
                charts[tool] = f"<p>No analysis available for {tool}</p>"
        
        # Determine overall trend
        trend = 'uptrend' if trend_direction == 'bullish' else 'downtrend' if trend_direction == 'bearish' else 'none'
        confidence = 'medium'
        
        # Calculate nearest support and resistance from Gann levels
        nearest_support = current_price * 0.98  # Default fallback
        nearest_resistance = current_price * 1.02  # Default fallback
        
        support_levels = []
        resistance_levels = []
        
        for tool_details in analysis_details.values():
            if 'box_bottom' in tool_details and tool_details['box_bottom'] < current_price:
                support_levels.append(tool_details['box_bottom'])
            if 'box_top' in tool_details and tool_details['box_top'] > current_price:
                resistance_levels.append(tool_details['box_top'])
            if 'pivot_price' in tool_details and tool_details['pivot_price'] < current_price:
                support_levels.append(tool_details['pivot_price'])
        
        if support_levels:
            nearest_support = max(support_levels)
        if resistance_levels:
            nearest_resistance = min(resistance_levels)
        
        # Determine price position
        if current_price <= nearest_support * 1.01:
            price_position = 'Near Support'
        elif current_price >= nearest_resistance * 0.99:
            price_position = 'Near Resistance'
        else:
            price_position = 'Between Levels'
        
        # Build the final result compatible with Flask app expectations
        result = {
            'symbol': symbol,
            'interval': interval,
            'current_price': current_price,
            'trend': trend,
            'confidence': confidence,
            'pivot_type': pivot_choice,
            'price_position': price_position,
            'nearest_support': nearest_support,
            'nearest_resistance': nearest_resistance,
            'subtools': gann_subtools,
            'charts': charts,
            'gann_analysis_details': analysis_details,
            'technical_indicators': {
                'rsi': df['rsi'].iloc[-1] if 'rsi' in df.columns and len(df) > 0 else 50,
                'ema': df['ema'].iloc[-1] if 'ema' in df.columns and len(df) > 0 else current_price,
                'atr': df['atr'].iloc[-1] if 'atr' in df.columns and len(df) > 0 else 0,
                'bb_percent': df['bb_percentB'].iloc[-1] * 100 if 'bb_percentB' in df.columns and len(df) > 0 else 50
            }
        }
        
        print(f"✅ Gann analysis completed for {symbol} with {len(gann_subtools)} tools")
        return result
        
    except Exception as e:
        print(f"❌ Error in Gann analysis for {symbol}: {e}")
        import traceback
        traceback.print_exc()
        return None