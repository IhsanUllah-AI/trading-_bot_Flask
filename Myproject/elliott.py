import pandas as pd
import numpy as np
from utils import fetch_candles_from_binance, fetch_current_price, calculate_indicators
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from datetime import datetime, timezone

def find_pivots(df, delta):
    pivots = []
    if len(df) == 0:
        return pivots

    last_pivot_idx = 0
    last_pivot_price = df['close'].iloc[0]
    last_pivot_type = None
    direction = 0

    for i in range(1, len(df)):
        high = df['high'].iloc[i]
        low = df['low'].iloc[i]

        if high > last_pivot_price:
            direction = 1
            last_pivot_price = df['low'].iloc[0]
            last_pivot_type = 'L'
            pivots.append((0, last_pivot_price, 'L'))
            last_pivot_idx = 0
            break
        elif low < last_pivot_price:
            direction = -1
            last_pivot_price = df['high'].iloc[0]
            last_pivot_type = 'H'
            pivots.append((0, last_pivot_price, 'H'))
            last_pivot_idx = 0
            break

    if direction == 0:
        return pivots

    for i in range(last_pivot_idx + 1, len(df)):
        high = df['high'].iloc[i]
        low = df['low'].iloc[i]
        reversal_amount = last_pivot_price * delta

        if direction == 1:
            if high > last_pivot_price:
                last_pivot_price = high
                last_pivot_idx = i
            if low < last_pivot_price - reversal_amount:
                pivots.append((last_pivot_idx, df['high'].iloc[last_pivot_idx], 'H'))
                direction = -1
                last_pivot_price = low
                last_pivot_idx = i
                last_pivot_type = 'H'
        elif direction == -1:
            if low < last_pivot_price:
                last_pivot_price = low
                last_pivot_idx = i
            if high > last_pivot_price + reversal_amount:
                pivots.append((last_pivot_idx, df['low'].iloc[last_pivot_idx], 'L'))
                direction = 1
                last_pivot_price = high
                last_pivot_idx = i
                last_pivot_type = 'L'

    if direction == 1:
        pivots.append((last_pivot_idx, df['high'].iloc[last_pivot_idx], 'H?'))
    elif direction == -1:
        pivots.append((last_pivot_idx, df['low'].iloc[last_pivot_idx], 'L?'))

    return pivots

def detect_elliott_waves_from_pivots(pivots, df):
    if len(pivots) < 3:
        return {'historical_patterns': [], 'partial_waves': {}, 'confidence': 'low', 'direction': 'none', 'fib_historical': {}, 'fib_partial': {}, 'signals': []}

    historical_patterns = []
    partial_waves = {}
    confidence = 'low'
    direction = 'none'
    fib_historical = {}
    fib_partial = {}
    signals = []

    historical_completes = find_all_complete_impulses(pivots)
    for historical_complete in historical_completes:
        if historical_complete['waves']:
            pattern = {'impulse': historical_complete['waves'], 'fib_impulse': historical_complete['fib']}
            if 'confidence' in historical_complete and historical_complete['confidence'] > confidence:
                confidence = historical_complete['confidence']
            if 'direction' in historical_complete:
                direction = historical_complete['direction']

            # FIXED: Proper ABC detection after complete impulse
            start_idx = historical_complete['start_idx'] + 6
            fib_abc = {}
            abc = {}
            if len(pivots) > start_idx and len(pivots) - start_idx >= 3:
                abc_data = detect_abc_correction(pivots[start_idx:start_idx+3], historical_complete['direction'], historical_complete['waves']['5']['price'])
                if abc_data['abc']:
                    abc = abc_data['abc']
                    fib_abc = abc_data['fib_abc']
                    confidence = 'high' if confidence == 'medium' else confidence
            pattern['abc'] = abc
            pattern['fib_abc'] = fib_abc
            historical_patterns.append(pattern)

    if historical_patterns:
        last_historical = historical_completes[-1]
        confidence = last_historical['confidence']
        direction = last_historical['direction']
        last_pattern = historical_patterns[-1]
        fib_historical = {**last_pattern['fib_impulse'], **last_pattern['fib_abc']}

    # FIXED: Proper partial wave detection
    if len(pivots) >= 5:
        partial_data = detect_partial_up_to_wave4(pivots[-5:], df)
        if partial_data['partial_waves']:
            partial_waves = partial_data['partial_waves']
            confidence = max(confidence, partial_data['confidence']) if isinstance(confidence, str) else partial_data['confidence']
            direction = partial_data['direction']
            fib_partial.update(partial_data['fib'])
            signals = partial_data['signals']

    if not partial_waves and len(pivots) >= 3:
        partial_data = detect_partial_up_to_wave2(pivots[-3:], df)
        if partial_data['partial_waves']:
            partial_waves = partial_data['partial_waves']
            confidence = max(confidence, partial_data['confidence']) if isinstance(confidence, str) else partial_data['confidence']
            direction = partial_data['direction']
            fib_partial.update(partial_data['fib'])
            signals = partial_data['signals']

    return {'historical_patterns': historical_patterns, 'partial_waves': partial_waves, 'confidence': confidence, 'direction': direction, 'fib_historical': fib_historical, 'fib_partial': fib_partial, 'signals': signals}

def find_all_complete_impulses(pivots):
    completes = []
    i = 0
    while i <= len(pivots) - 6:
        candidate_pivots = pivots[i:i+6]
        complete_data = detect_complete_impulse(candidate_pivots)
        if complete_data['waves']:
            complete_data['start_idx'] = i
            completes.append(complete_data)
            i += 6
        else:
            i += 1
    return completes

def detect_complete_impulse(last_pivots):
    types = [p[2].replace('?', '') for p in last_pivots]
    if types != ['L', 'H', 'L', 'H', 'L', 'H'] and types != ['H', 'L', 'H', 'L', 'H', 'L']:
        return {'waves': {}, 'confidence': 'low', 'direction': 'none', 'fib': {}}

    is_bullish = types[0] == 'L' and last_pivots[-1][1] > last_pivots[0][1]
    is_bearish = types[0] == 'H' and last_pivots[-1][1] < last_pivots[0][1]

    if not (is_bullish or is_bearish):
        return {'waves': {}, 'confidence': 'low', 'direction': 'none', 'fib': {}}

    direction = 'bullish' if is_bullish else 'bearish'
    p0, p1, p2, p3, p4, p5 = [p[1] for p in last_pivots]

    w1 = abs(p1 - p0)
    w2 = abs(p2 - p1)
    w3 = abs(p3 - p2)
    w4 = abs(p4 - p3)
    w5 = abs(p5 - p4)

    if is_bullish:
        if p2 <= p0 or p4 <= p1:
            return {'waves': {}, 'confidence': 'low', 'direction': 'none', 'fib': {}}
    else:
        if p2 >= p0 or p4 >= p1:
            return {'waves': {}, 'confidence': 'low', 'direction': 'none', 'fib': {}}

    if w3 <= min(w1, w5):
        return {'waves': {}, 'confidence': 'low', 'direction': 'none', 'fib': {}}

    if is_bullish:
        w2_retr = (p1 - p2) / (p1 - p0)
        w3_ext = w3 / w1
        w4_retr = (p3 - p4) / (p3 - p2)
        w5_ext = w5 / w1
    else:
        w2_retr = (p2 - p1) / (p0 - p1)
        w3_ext = w3 / w1
        w4_retr = (p4 - p3) / (p2 - p3)
        w5_ext = w5 / w1

    fib = {
        'Wave 2 Retrace': w2_retr,
        'Wave 3 Extension': w3_ext,
        'Wave 4 Retrace': w4_retr,
        'Wave 5 Extension': w5_ext
    }

    score = 0
    if 0.382 <= w2_retr <= 0.618:
        score += 2
    elif 0.236 <= w2_retr <= 0.786:
        score += 1

    if 1.0 <= w3_ext <= 2.618:
        score += 2 if abs(w3_ext - 1.618) < 0.3 else 1

    if 0.236 <= w4_retr <= 0.382:
        score += 1

    if abs(w5_ext - 1.0) < 0.2 or abs(w5_ext - 0.618) < 0.2 or abs(w5_ext - 1.618) < 0.2:
        score += 1

    confidence = 'high' if score >= 4 else 'medium' if score >= 3 else 'low'

    waves = {}
    labels = ['0', '1', '2', '3', '4', '5']
    for j, label in enumerate(labels):
        idx, price, ptype = last_pivots[j]
        waves[label] = {'index': idx, 'price': price, 'type': ptype.replace('?', '')}

    return {'waves': waves, 'confidence': confidence, 'direction': direction, 'fib': fib}

def detect_partial_up_to_wave2(last_pivots, df):
    types = [p[2].replace('?', '') for p in last_pivots]
    if types != ['L', 'H', 'L'] and types != ['H', 'L', 'H']:
        return {'partial_waves': {}, 'signals': [], 'confidence': 'low', 'direction': 'none', 'fib': {}}

    is_bullish = types[0] == 'L' and last_pivots[1][1] > last_pivots[0][1]
    is_bearish = types[0] == 'H' and last_pivots[1][1] < last_pivots[0][1]

    if not (is_bullish or is_bearish):
        return {'partial_waves': {}, 'signals': [], 'confidence': 'low', 'direction': 'none', 'fib': {}}

    direction = 'bullish' if is_bullish else 'bearish'
    p0, p1, p2 = [p[1] for p in last_pivots]

    w1 = abs(p1 - p0)
    w2 = abs(p2 - p1)

    if is_bullish:
        if p2 <= p0:
            return {'partial_waves': {}, 'signals': [], 'confidence': 'low', 'direction': 'none', 'fib': {}}
    else:
        if p2 >= p0:
            return {'partial_waves': {}, 'signals': [], 'confidence': 'low', 'direction': 'none', 'fib': {}}

    if is_bullish:
        w2_retr = (p1 - p2) / (p1 - p0)
    else:
        w2_retr = (p2 - p1) / (p0 - p1)

    fib = {'Wave 2 Retrace': w2_retr}

    score = 0
    if 0.382 <= w2_retr <= 0.618:
        score += 2
    elif 0.236 <= w2_retr <= 0.786:
        score += 1

    last_idx = last_pivots[-1][0]
    rsi_at_w2 = df['rsi'].iloc[last_idx]
    if is_bullish and rsi_at_w2 > 30:
        score += 1
    elif is_bearish and rsi_at_w2 < 70:
        score += 1

    confidence = 'high' if score >= 4 else 'medium' if score >= 3 else 'low'

    partial_waves = {}
    labels = ['0?', '1?', '2?']
    for j, label in enumerate(labels):
        idx, price, ptype = last_pivots[j]
        partial_waves[label] = {'index': idx, 'price': price, 'type': ptype.replace('?', '')}

    signals = []
    if confidence != 'low':
        atr = df['atr'].iloc[-1]
        current_price = df['close'].iloc[-1]
        if is_bullish:
            sl = p2 - atr
            tp = p2 + 1.0 * w1
            # FIXED: Use 'type' instead of 'action' to match your template
            signals.append({'type': 'BUY', 'entry_price': current_price, 'sl': sl, 'tp': tp, 'reason': 'After Wave 2?, expect Wave 3 up'})
        else:
            sl = p2 + atr
            tp = p2 - 1.0 * w1
            signals.append({'type': 'SELL', 'entry_price': current_price, 'sl': sl, 'tp': tp, 'reason': 'After Wave 2?, expect Wave 3 down'})

    return {'partial_waves': partial_waves, 'signals': signals, 'confidence': confidence, 'direction': direction, 'fib': fib}

def detect_partial_up_to_wave4(last_pivots, df):
    types = [p[2].replace('?', '') for p in last_pivots]
    if types != ['L', 'H', 'L', 'H', 'L'] and types != ['H', 'L', 'H', 'L', 'H']:
        return {'partial_waves': {}, 'signals': [], 'confidence': 'low', 'direction': 'none', 'fib': {}}

    is_bullish = types[0] == 'L' and last_pivots[3][1] > last_pivots[1][1]
    is_bearish = types[0] == 'H' and last_pivots[3][1] < last_pivots[1][1]

    if not (is_bullish or is_bearish):
        return {'partial_waves': {}, 'signals': [], 'confidence': 'low', 'direction': 'none', 'fib': {}}

    direction = 'bullish' if is_bullish else 'bearish'
    p0, p1, p2, p3, p4 = [p[1] for p in last_pivots]

    w1 = abs(p1 - p0)
    w2 = abs(p2 - p1)
    w3 = abs(p3 - p2)
    w4 = abs(p4 - p3)

    if is_bullish:
        if p2 <= p0 or p4 <= p1:
            return {'partial_waves': {}, 'signals': [], 'confidence': 'low', 'direction': 'none', 'fib': {}}
    else:
        if p2 >= p0 or p4 >= p1:
            return {'partial_waves': {}, 'signals': [], 'confidence': 'low', 'direction': 'none', 'fib': {}}

    if w3 <= w1:
        return {'partial_waves': {}, 'signals': [], 'confidence': 'low', 'direction': 'none', 'fib': {}}

    if is_bullish:
        w2_retr = (p1 - p2) / (p1 - p0)
        w3_ext = w3 / w1
        w4_retr = (p3 - p4) / (p3 - p2)
    else:
        w2_retr = (p2 - p1) / (p0 - p1)
        w3_ext = w3 / w1
        w4_retr = (p4 - p3) / (p2 - p3)

    fib = {
        'Wave 2 Retrace': w2_retr,
        'Wave 3 Extension': w3_ext,
        'Wave 4 Retrace': w4_retr
    }

    score = 0
    if 0.382 <= w2_retr <= 0.618:
        score += 2
    elif 0.236 <= w2_retr <= 0.786:
        score += 1

    if 1.0 <= w3_ext <= 2.618:
        score += 2 if abs(w3_ext - 1.618) < 0.3 else 1

    if 0.236 <= w4_retr <= 0.382:
        score += 1

    last_idx = last_pivots[-1][0]
    rsi_at_w4 = df['rsi'].iloc[last_idx]
    if is_bullish and rsi_at_w4 > 30:
        score += 1
    elif is_bearish and rsi_at_w4 < 70:
        score += 1

    confidence = 'high' if score >= 4 else 'medium' if score >= 3 else 'low'

    partial_waves = {}
    labels = ['0?', '1?', '2?', '3?', '4?']
    for j, label in enumerate(labels):
        idx, price, ptype = last_pivots[j]
        partial_waves[label] = {'index': idx, 'price': price, 'type': ptype.replace('?', '')}

    signals = []
    if confidence != 'low':
        atr = df['atr'].iloc[-1]
        current_price = df['close'].iloc[-1]
        if is_bullish:
            sl = p4 - atr
            tp = p4 + 0.38 * w1
            signals.append({'type': 'BUY', 'entry_price': current_price, 'sl': sl, 'tp': tp, 'reason': 'After Wave 4?, expect Wave 5 up'})
        else:
            sl = p4 + atr
            tp = p4 - 0.38 * w1
            signals.append({'type': 'SELL', 'entry_price': current_price, 'sl': sl, 'tp': tp, 'reason': 'After Wave 4?, expect Wave 5 down'})

    return {'partial_waves': partial_waves, 'signals': signals, 'confidence': confidence, 'direction': direction, 'fib': fib}

def detect_abc_correction(last_pivots, direction, wave5_price):
    types = [p[2].replace('?', '') for p in last_pivots]
    
    if len(last_pivots) < 3:
        return {'abc': {}, 'fib_abc': {}}
        
    pa, pb, pc = [p[1] for p in last_pivots]

    add_abc = False
    fib_abc = {}
    if direction == 'bullish' and types == ['L', 'H', 'L']:
        add_abc = True
        a_len = wave5_price - pa
        b_retr = (pb - pa) / a_len if a_len != 0 else 0
        c_len = pb - pc
        c_ext = c_len / a_len if a_len != 0 else 0
    elif direction == 'bearish' and types == ['H', 'L', 'H']:
        add_abc = True
        a_len = pa - wave5_price
        b_retr = (pa - pb) / a_len if a_len != 0 else 0
        c_len = pc - pb
        c_ext = c_len / a_len if a_len != 0 else 0

    if add_abc:
        score = 0
        if 0.382 < b_retr < 0.786:
            score += 1
        if abs(c_ext - 1.0) < 0.2 or abs(c_ext - 1.618) < 0.2:
            score += 1

        if score > 0:
            fib_abc = {'B Retrace': b_retr, 'C Extension': c_ext}
            abc = {}
            labels = ['A', 'B', 'C']
            for j, label in enumerate(labels):
                idx, price, ptype = last_pivots[j]
                abc[label] = {'index': idx, 'price': price, 'type': ptype.replace('?', '')}
            return {'abc': abc, 'fib_abc': fib_abc}

    return {'abc': {}, 'fib_abc': {}}

def create_elliott_chart(raw_df, df_live, historical_patterns, partial_waves, trend, indicators_to_show, show_ema, show_bb):
    num_subplots = len(indicators_to_show) + 1
    row_heights = [0.6] + [0.4 / len(indicators_to_show)] * len(indicators_to_show) if indicators_to_show else [1.0]
    
    subplot_titles = [f'Elliott Wave Analysis - {trend.capitalize()}']
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
                name=f'EMA (20)',
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
    
    price_range = raw_df['high'].max() - raw_df['low'].min()
    offset = price_range * 0.05
    
    colors_wave = {'0': 'gray', '1': 'blue', '2': 'red', '3': 'blue', '4': 'red', '5': 'blue', 
                   'A': 'green', 'B': 'red', 'C': 'green', '0?': 'gray', '1?': 'blue', '2?': 'red', '3?': 'blue', '4?': 'red'}
    
    # Plot historical patterns with ABC corrections
    for hist_idx, pattern in enumerate(historical_patterns):
        prefix = f"H{hist_idx+1}-"
        wave_points = []
        for label in ['0', '1', '2', '3', '4', '5']:
            if label in pattern['impulse']:
                idx = pattern['impulse'][label]['index']
                price = pattern['impulse'][label]['price']
                ptype = pattern['impulse'][label]['type']
                offset_price = price + offset if ptype == 'H' else price - offset
                wave_points.append((df_live['closetime'].iloc[idx], offset_price))
                fig.add_annotation(
                    x=df_live['closetime'].iloc[idx],
                    y=offset_price,
                    text=f"{prefix}{label}",
                    showarrow=True,
                    arrowhead=2,
                    arrowsize=1,
                    arrowwidth=2,
                    arrowcolor=colors_wave.get(label, 'gray'),
                    opacity=0.6,
                    ax=0,
                    ay=-30 if ptype == 'H' else 30,
                    row=1, col=1
                )
        
        if wave_points:
            x_vals, y_vals = zip(*wave_points)
            fig.add_trace(
                go.Scatter(
                    x=x_vals,
                    y=y_vals,
                    mode='lines+markers',
                    name=f'{prefix}Impulse',
                    line=dict(color='blue', width=2, dash='solid'),
                    opacity=0.5,
                    showlegend=(hist_idx == 0)
                ),
                row=1, col=1
            )

        # Plot ABC corrections
        if pattern['abc']:
            abc_prefix = f"{prefix}ABC-"
            abc_points = []
            for label in ['A', 'B', 'C']:
                if label in pattern['abc']:
                    idx = pattern['abc'][label]['index']
                    price = pattern['abc'][label]['price']
                    ptype = pattern['abc'][label]['type']
                    offset_price = price + offset if ptype == 'H' else price - offset
                    abc_points.append((df_live['closetime'].iloc[idx], offset_price))
                    fig.add_annotation(
                        x=df_live['closetime'].iloc[idx],
                        y=offset_price,
                        text=f"{abc_prefix}{label}",
                        showarrow=True,
                        arrowhead=2,
                        arrowsize=1,
                        arrowwidth=2,
                        arrowcolor=colors_wave.get(label, 'green'),
                        opacity=0.6,
                        ax=0,
                        ay=-30 if ptype == 'H' else 30,
                        row=1, col=1
                    )
            
            if abc_points:
                x_vals, y_vals = zip(*abc_points)
                fig.add_trace(
                    go.Scatter(
                        x=x_vals,
                        y=y_vals,
                        mode='lines+markers',
                        name=f'{abc_prefix}',
                        line=dict(color='green', width=2, dash='solid'),
                        opacity=0.5,
                        showlegend=(hist_idx == 0)
                    ),
                    row=1, col=1
                )
    
    # Plot partial waves
    partial_points = []
    for label in ['0?', '1?', '2?', '3?', '4?']:
        if label in partial_waves:
            idx = partial_waves[label]['index']
            price = partial_waves[label]['price']
            ptype = partial_waves[label]['type']
            offset_price = price + offset if ptype == 'H' else price - offset
            partial_points.append((df_live['closetime'].iloc[idx], offset_price))
            fig.add_annotation(
                x=df_live['closetime'].iloc[idx],
                y=offset_price,
                text=label,
                showarrow=True,
                arrowhead=2,
                arrowsize=1,
                arrowwidth=2,
                arrowcolor=colors_wave.get(label, 'gray'),
                ax=0,
                ay=-30 if ptype == 'H' else 30,
                row=1, col=1
            )
    
    if partial_points:
        x_vals, y_vals = zip(*partial_points)
        fig.add_trace(
            go.Scatter(
                x=x_vals,
                y=y_vals,
                mode='lines+markers',
                name='Current Partial',
                line=dict(color='yellow', width=2, dash='dash'),
                showlegend=True
            ),
            row=1, col=1
        )
    
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
        title=f"Elliott Wave Analysis - {trend.capitalize()}",
        yaxis_title='Price',
        xaxis_rangeslider_visible=True,
        template="plotly_dark",
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        dragmode='zoom',
        hovermode='x unified'
    )
    fig.update_xaxes(
        title_text="Date", 
        row=num_subplots, col=1,
        rangeslider_visible=True,
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
        initial_end = raw_df['closetime'].iloc[-1]
        fig.update_xaxes(range=[initial_start, initial_end], row=1, col=1)
    
    return fig

def run_elliott_analysis(symbol, interval, candle_limit, thresholds, selected_degrees, use_smoothing=False, smooth_period=3, 
                        show_ema=True, show_bb=True, show_volume=True, show_rsi=True, show_macd=True):
    print(f"🔍 Starting Elliott Wave analysis for {symbol} {interval}")
    
    try:
        df_live = fetch_candles_from_binance(symbol, interval, candle_limit)
        if df_live is None or df_live.empty:
            print(f"❌ No data fetched for {symbol}")
            return None
            
        print(f"✅ Data fetched: {len(df_live)} candles")
        
        raw_df = df_live.copy()
        df_live = calculate_indicators(df_live)
        
        print(f"✅ Indicators calculated")
        
        # Add MACD calculation for Elliott Wave
        macd_fast, macd_slow, macd_signal = 12, 26, 9
        ema_fast = df_live['close'].ewm(span=macd_fast, adjust=False).mean()
        ema_slow = df_live['close'].ewm(span=macd_slow, adjust=False).mean()
        df_live['macd'] = ema_fast - ema_slow
        df_live['macd_signal'] = df_live['macd'].ewm(span=macd_signal, adjust=False).mean()
        df_live['macd_histogram'] = df_live['macd'] - df_live['macd_signal']
        
        if use_smoothing:
            df_live['high'] = df_live['high'].ewm(span=smooth_period, adjust=False).mean()
            df_live['low'] = df_live['low'].ewm(span=smooth_period, adjust=False).mean()
            df_live['close'] = df_live['close'].ewm(span=smooth_period, adjust=False).mean()
        
        current_price = fetch_current_price(symbol) or df_live['close'].iloc[-1]
        
        wave_data_by_degree = {}
        for degree in selected_degrees:
            delta = thresholds[degree]
            pivots = find_pivots(df_live, delta)
            wave_data = detect_elliott_waves_from_pivots(pivots, df_live)
            wave_data_by_degree[degree] = wave_data
        
        # Create charts for each degree
        charts = {}
        indicators_to_show = []
        if show_volume:
            indicators_to_show.append("Volume")
        if show_rsi:
            indicators_to_show.append("RSI")
        if show_macd:
            indicators_to_show.append("MACD")
        
        for degree in selected_degrees:
            wave_data = wave_data_by_degree[degree]
            chart = create_elliott_chart(raw_df, df_live, wave_data['historical_patterns'], 
                                       wave_data['partial_waves'], wave_data['direction'], 
                                       indicators_to_show, show_ema, show_bb)
            charts[degree] = chart.to_html(full_html=False)
        
        result = {
            'symbol': symbol,
            'interval': interval,
            'current_price': current_price,
            'wave_data_by_degree': wave_data_by_degree,
            'charts': charts,
            'technical_indicators': {
                'rsi': df_live['rsi'].iloc[-1] if 'rsi' in df_live.columns and len(df_live) > 0 else 0,
                'ema': df_live['ema'].iloc[-1] if 'ema' in df_live.columns and len(df_live) > 0 else 0,
                'atr': df_live['atr'].iloc[-1] if 'atr' in df_live.columns and len(df_live) > 0 else 0,
                'bb_percent': df_live['bb_percentB'].iloc[-1] * 100 if 'bb_percentB' in df_live.columns and len(df_live) > 0 else 0
            }
        }
        
        print(f"✅ Elliott Wave analysis completed for {symbol}")
        return result
        
    except Exception as e:
        print(f"❌ Error in Elliott Wave analysis for {symbol}: {e}")
        import traceback
        traceback.print_exc()
        return None