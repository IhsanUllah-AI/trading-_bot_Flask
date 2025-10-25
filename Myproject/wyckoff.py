import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timezone
from utils import fetch_candles_from_binance, fetch_current_price, calculate_indicators, detect_hh_hl_lh_ll_columns

def get_zigzag(df, interval):
    depth_dict = {
        '1m': 65, '3m': 50, '5m': 50, '15m': 70, '30m': 85,
        '1h': 100, '2h': 100, '4h': 90, '6h': 90, '8h': 90, '12h': 90,
        '1d': 60, '3d': 60, '1w': 60, '1M': 60
    }
    depth = depth_dict.get(interval, 50)

    pivots = []
    last_pivot_hl = 0
    last_pivot_index = 0
    last_pivot = None

    for i in range(depth, len(df)):
        is_high_pivot = df['high'].iloc[i] == df['high'].iloc[i - depth:i + depth + 1].max()
        is_low_pivot = df['low'].iloc[i] == df['low'].iloc[i - depth:i + depth + 1].min()

        cur_pivot_hl = 0
        cur_pivot_value = None
        if is_high_pivot:
            cur_pivot_hl = 1
            cur_pivot_value = df['high'].iloc[i]
        elif is_low_pivot:
            cur_pivot_hl = -1
            cur_pivot_value = df['low'].iloc[i]

        if cur_pivot_hl == 0:
            continue

        if last_pivot_hl == 0:
            last_pivot_hl = cur_pivot_hl
            last_pivot_index = i
            last_pivot = cur_pivot_value
            continue

        if cur_pivot_hl == last_pivot_hl:
            if cur_pivot_hl == 1 and cur_pivot_value > last_pivot:
                last_pivot = cur_pivot_value
                last_pivot_index = i
            elif cur_pivot_hl == -1 and cur_pivot_value < last_pivot:
                last_pivot = cur_pivot_value
                last_pivot_index = i
            continue

        pivots.append((df['closetime'].iloc[last_pivot_index], last_pivot, last_pivot_hl))

        last_pivot_hl = cur_pivot_hl
        last_pivot_index = i
        last_pivot = cur_pivot_value

    if last_pivot_index > 0:
        pivots.append((df['closetime'].iloc[last_pivot_index], last_pivot, last_pivot_hl))

    return pivots

def detect_wyckoff_signals(df, current_price, interval):
    if len(df) < 50:
        return {'phase': 'none', 'confidence': 'low', 'signals': [], 'reasons_not_met': [], 'sideways_count': 0, 'hh_in_range': False, 'll_in_range': False, 'hh_detected': False, 'hl_detected': False, 'lh_detected': False, 'll_detected': False}

    last = df.iloc[-1]
    
    # Initialize
    phase = 'none'
    confidence = 'low'
    signals = []
    reasons_not_met = []
    sideways_count = 0

    # Support and Resistance
    support = df['support'].iloc[-1]
    resistance = df['resistance'].iloc[-1]
    range_threshold_acc = last['atr'] * 2 if pd.notnull(last['atr']) else 0.01 * last['close']
    range_threshold_dist = last['atr'] if pd.notnull(last['atr']) else 0.005 * last['close']

    # Timeframe-based sideways thresholds
    sideways_thresholds = {
        '1m': 200,  # ~3 hours
        '15m': 80,  # ~1-2 days
        '1h': 40,   # ~2 days
        '4h': 30    # ~5 days
    }
    sideways_min = sideways_thresholds.get(interval, 15)

    # HH/LL count window from table
    hhll_count_window = {
        '1m': 65, '3m': 50, '5m': 50, '15m': 70, '30m': 85,
        '1h': 100, '2h': 100, '4h': 90, '6h': 90, '8h': 90, '12h': 90,
        '1d': 60, '3d': 60, '1w': 60, '1M': 60
    }
    count_window = hhll_count_window.get(interval, 30)

    # Volume Checks
    rolling_volume_avg = df['avg_volume'].iloc[-1]
    volume_high = last['volume'] > 1.5 * rolling_volume_avg if pd.notnull(rolling_volume_avg) else False
    volume_spike = last['volume'] > 1.5 * rolling_volume_avg if pd.notnull(rolling_volume_avg) else False
    sell_volume_high = False
    if len(df) > 5:
        sell_volumes = df['volume'].tail(5)[df['close'].tail(5) < df['open'].tail(5)]
        sell_volume_high = sell_volumes.iloc[-1] > 1.5 * rolling_volume_avg if len(sell_volumes) > 0 else False

    # ATR Compression
    atr_avg = df['atr'].tail(20).mean()
    atr_compression = last['atr'] < atr_avg if pd.notnull(atr_avg) and pd.notnull(last['atr']) else False

    # RSI Divergence (for reference, not mandatory for signals)
    rsi_div_bull = False
    rsi_div_bear = False
    pivots = get_zigzag(df, interval)
    if pivots and len(pivots) >= 2:
        high_pivots = [p for p in pivots if p[2] == 1]
        low_pivots = [p for p in pivots if p[2] == -1]
        if len(high_pivots) >= 2:
            latest_high_idx = df['high'].tail(50).idxmax()
            prev_high_idx = df['high'].tail(50).iloc[:-1].idxmax()
            latest_rsi = df['rsi'].iloc[latest_high_idx]
            prev_rsi = df['rsi'].iloc[prev_high_idx]
            if df['high'].iloc[latest_high_idx] >= df['high'].iloc[prev_high_idx] and latest_rsi < prev_rsi:
                rsi_div_bear = True
        if len(low_pivots) >= 2:
            latest_low_idx = df['low'].tail(50).idxmin()
            prev_low_idx = df['low'].tail(50).iloc[:-1].idxmin()
            latest_rsi = df['rsi'].iloc[latest_low_idx]
            prev_rsi = df['rsi'].iloc[prev_low_idx]
            if df['low'].iloc[latest_low_idx] <= df['low'].iloc[prev_low_idx] and latest_rsi > prev_rsi:
                rsi_div_bull = True

    # OBV Direction
    obv_rising = False
    obv_falling = False
    if len(df) > 20:
        obv_tail = df['obv'].tail(20)
        obv_rising = obv_tail.iloc[-1] > obv_tail.mean()
        obv_falling = obv_tail.iloc[-1] < obv_tail.mean()

    rsi_acc = last['rsi'] < 40  # For reference only
    rsi_dist = last['rsi'] > 60  # For reference only
    rsi_markup = last['rsi'] > 50  # For reference only
    rsi_markdown = last['rsi'] < 50  # For reference only

    # EMA and MACD
    ema_bull = last['ema'] > last['ema50']
    ema_bear = last['ema'] < last['ema50']
    macd_bull = last['macd'] > 0 and last['macd'] > last['macd_signal']
    macd_bear = last['macd'] < 0 and last['macd'] < last['macd_signal']

    # Sideways Range Check (for Accumulation/Distribution)
    sideways_lookback = max(sideways_min, 30)
    ll_in_range = False
    hh_in_range = False
    volume_spike_in_range = False
    low_vol = False
    if len(df) >= sideways_lookback:
        recent_df = df.tail(sideways_lookback)
        in_range = (recent_df['close'] >= support - range_threshold_acc) & (recent_df['close'] <= resistance + range_threshold_dist)
        sideways_count = in_range.sum()
        low_vol = recent_df['close'].pct_change().std() < 0.01
        ll_in_range = recent_df['ll'].tail(count_window).sum() >= 2
        hh_in_range = recent_df['hh'].tail(count_window).sum() >= 2
        volume_spike_in_range = (recent_df['volume'] > recent_df['avg_volume'] * 1.5).any()

    hh_detected = False
    hl_detected = False
    lh_detected = False
    ll_detected = False

    # Last high/low pivots for TP
    last_high_pivot = max([p[1] for p in pivots if p[2] == 1], default=resistance) if pivots else resistance
    last_low_pivot = min([p[1] for p in pivots if p[2] == -1], default=support) if pivots else support

    # Phase Detection
    # Accumulation: Sideways, LL, ATR compression
    if sideways_count >= sideways_min and low_vol and ll_in_range and atr_compression:
        phase = 'accumulation'
        ll_detected = ll_in_range
        score = 3  # Base: sideways range + low volatility + LL + ATR compression
        if volume_spike_in_range or volume_high:
            score += 1  # Bonus: Volume
        if ema_bull:
            score += 1  # Secondary: EMA bullish
        if macd_bull:
            score += 1  # Secondary: MACD bullish
        spring = last['low'] < support and last['close'] > support
        if spring:
            score += 1  # Booster: Spring
        if rsi_acc:
            score += 1  # Optional: RSI oversold
        if rsi_div_bull:
            score += 1  # RSI divergence
        if obv_rising:
            score += 1  # OBV direction
        confidence = 'high' if score >= 6 else 'medium' if score >= 5 else 'low'
        
        # Calculate SL/TP for Accumulation (Buy)
        atr = last['atr'] if pd.notnull(last['atr']) else 0.01 * last['close']
        sl_buy = min(df['low'].tail(20).min() - 0.5 * atr, support - 0.5 * atr) if spring else support - 0.5 * atr
        tp_buy = last_high_pivot + atr  # Target swing high + buffer for breakout
        risk = current_price - sl_buy
        reward = tp_buy - current_price
        rr_ratio = reward / risk if risk > 0 else 0
        
        if rr_ratio < 1.5:
            reasons_not_met.append(f"Accumulation signal skipped: RR ratio {rr_ratio:.2f} < 1.5")
        elif (confidence == 'medium' and rsi_div_bull) or (confidence == 'high' and obv_rising):
            reason = f"Accumulation: Sideways range ({sideways_count}/{sideways_min} candles), Low volatility, Multiple lower lows, ATR compression"
            if volume_spike_in_range or volume_high:
                reason += ", Volume spike/high (bonus)"
            if rsi_div_bull:
                reason += ", RSI bullish divergence"
            elif rsi_acc:
                reason += ", RSI oversold (<40)"
            if spring:
                reason += ", Spring pattern"
            if ema_bull:
                reason += ", EMA20 > EMA50"
            if macd_bull:
                reason += ", MACD bullish"
            if obv_rising:
                reason += ", OBV rising"
            reason += f", RR: {rr_ratio:.2f}"
            signals.append({
                'type': 'BUY',
                'entry_price': current_price,
                'sl': sl_buy,
                'tp': tp_buy,
                'reason': reason
            })
        else:
            if confidence == 'medium' and not rsi_div_bull:
                reasons_not_met.append("Accumulation conditions not met: Missing RSI bullish divergence for medium confidence.")
            if confidence == 'high' and not obv_rising:
                reasons_not_met.append("Accumulation conditions not met: Missing OBV rising for high confidence.")

    # Distribution: Sideways, HH, ATR compression
    elif sideways_count >= sideways_min and low_vol and hh_in_range and atr_compression:
        phase = 'distribution'
        hh_detected = hh_in_range
        score = 3  # Base: sideways range + low volatility + HH + ATR compression
        if volume_spike_in_range or not volume_high:
            score += 1  # Bonus: Volume
        if ema_bear:
            score += 1  # Secondary: EMA bearish
        if macd_bear:
            score += 1  # Secondary: MACD bearish
        utad = last['high'] > resistance and last['close'] < resistance
        if utad:
            score += 1  # Booster: UTAD
        if rsi_dist:
            score += 1  # Optional: RSI overbought
        if rsi_div_bear:
            score += 1  # RSI divergence
        if obv_falling:
            score += 1  # OBV direction
        confidence = 'high' if score >= 6 else 'medium' if score >= 5 else 'low'
        
        # Calculate SL/TP for Distribution (Sell)
        atr = last['atr'] if pd.notnull(last['atr']) else 0.01 * last['close']
        sl_sell = max(df['high'].tail(20).max() + 0.5 * atr, resistance + 0.5 * atr) if utad else resistance + 0.5 * atr
        tp_sell = last_low_pivot - atr  # Target swing low - buffer
        risk = sl_sell - current_price
        reward = current_price - tp_sell
        rr_ratio = reward / risk if risk > 0 else 0
        
        if rr_ratio < 1.5:
            reasons_not_met.append(f"Distribution signal skipped: RR ratio {rr_ratio:.2f} < 1.5")
        elif (confidence == 'medium' and rsi_div_bear) or (confidence == 'high' and obv_falling):
            reason = f"Distribution: Sideways range ({sideways_count}/{sideways_min} candles), Low volatility, Multiple higher highs, ATR compression"
            if volume_spike_in_range or not volume_high:
                reason += ", Volume spike/low (bonus)"
            if rsi_div_bear:
                reason += ", RSI bearish divergence"
            elif rsi_dist:
                reason += ", RSI overbought (>60)"
            if utad:
                reason += ", UTAD pattern"
            if ema_bear:
                reason += ", EMA20 < EMA50"
            if macd_bear:
                reason += ", MACD bearish"
            if obv_falling:
                reason += ", OBV falling"
            reason += f", RR: {rr_ratio:.2f}"
            signals.append({
                'type': 'SELL',
                'entry_price': current_price,
                'sl': sl_sell,
                'tp': tp_sell,
                'reason': reason
            })
        else:
            if confidence == 'medium' and not rsi_div_bear:
                reasons_not_met.append("Distribution conditions not met: Missing RSI bearish divergence for medium confidence.")
            if confidence == 'high' and not obv_falling:
                reasons_not_met.append("Distribution conditions not met: Missing OBV falling for high confidence.")

    # Markup: HH/HL, EMA bullish
    elif len(df) >= count_window:
        recent_df = df.tail(count_window)
        hh_count = recent_df['hh'].sum()
        hl_count = recent_df['hl'].sum()
        hh_detected = hh_count >= 2
        hl_detected = hl_count >= 2
        if (hh_detected or hl_detected) and ema_bull:
            phase = 'markup'
            score = 3  # Base: HH or HL + EMA bullish
            if volume_high:
                score += 1  # Bonus: Volume high
            if macd_bull:
                score += 1  # Secondary: MACD bullish
            pullback = abs(last['close'] - last['ema']) < last['atr']
            if pullback:
                score += 1  # Booster: Pullback
            if rsi_markup:
                score += 1  # Optional: RSI bullish
            if rsi_div_bull:
                score += 1  # RSI divergence
            if obv_rising:
                score += 1  # OBV direction
            confidence = 'high' if score >= 6 else 'medium' if score >= 5 else 'low'
            
            # Calculate SL/TP for Markup (Buy, trend)
            atr = last['atr'] if pd.notnull(last['atr']) else 0.01 * last['close']
            sl_buy = last['ema'] - atr  # Below EMA for trailing-like stop
            risk = current_price - sl_buy
            tp_buy = current_price + 2 * risk  # 1:2 RR fixed
            rr_ratio = 2.0  # Enforced
            
            if (confidence == 'medium' and rsi_div_bull) or (confidence == 'high' and obv_rising):
                reason = f"Markup: Multiple higher highs or lows, EMA20 > EMA50"
                if volume_high:
                    reason += ", High volume (bonus)"
                if rsi_div_bull:
                    reason += ", RSI bullish divergence"
                elif rsi_markup:
                    reason += ", RSI bullish (>50)"
                if pullback:
                    reason += ", Pullback to EMA20"
                if macd_bull:
                    reason += ", MACD bullish"
                if obv_rising:
                    reason += ", OBV rising"
                reason += f", RR: {rr_ratio:.2f}"
                signals.append({
                    'type': 'BUY',
                    'entry_price': current_price,
                    'sl': sl_buy,
                    'tp': tp_buy,
                    'reason': reason
                })
            else:
                if confidence == 'medium' and not rsi_div_bull:
                    reasons_not_met.append("Markup conditions not met: Missing RSI bullish divergence for medium confidence.")
                if confidence == 'high' and not obv_rising:
                    reasons_not_met.append("Markup conditions not met: Missing OBV rising for high confidence.")

    # Markdown: LH/LL, EMA bearish
    elif len(df) >= count_window:
        recent_df = df.tail(count_window)
        lh_count = recent_df['lh'].sum()
        ll_count = recent_df['ll'].sum()
        lh_detected = lh_count >= 2
        ll_detected = ll_count >= 2
        if (lh_detected or ll_detected) and ema_bear:
            phase = 'markdown'
            score = 3  # Base: LH or LL + EMA bearish
            if sell_volume_high:
                score += 1  # Bonus: Sell volume high
            if macd_bear:
                score += 1  # Secondary: MACD bearish
            pullback = abs(last['close'] - last['ema']) < last['atr']
            if pullback:
                score += 1  # Booster: Pullback
            if rsi_markdown:
                score += 1  # Optional: RSI bearish
            if rsi_div_bear:
                score += 1  # RSI divergence
            if obv_falling:
                score += 1  # OBV direction
            confidence = 'high' if score >= 6 else 'medium' if score >= 5 else 'low'
            
            # Calculate SL/TP for Markdown (Sell, trend)
            atr = last['atr'] if pd.notnull(last['atr']) else 0.01 * last['close']
            sl_sell = last['ema'] + atr  # Above EMA for trailing-like stop
            risk = sl_sell - current_price
            tp_sell = current_price - 2 * risk  # 1:2 RR fixed
            rr_ratio = 2.0  # Enforced
            
            if (confidence == 'medium' and rsi_div_bear) or (confidence == 'high' and obv_falling):
                reason = f"Markdown: Multiple lower highs or lows, EMA20 < EMA50"
                if sell_volume_high:
                    reason += ", High sell volume (bonus)"
                if rsi_div_bear:
                    reason += ", RSI bearish divergence"
                elif rsi_markdown:
                    reason += ", RSI bearish (<50)"
                if pullback:
                    reason += ", Pullback to EMA20"
                if macd_bear:
                    reason += ", MACD bearish"
                if obv_falling:
                    reason += ", OBV falling"
                reason += f", RR: {rr_ratio:.2f}"
                signals.append({
                    'type': 'SELL',
                    'entry_price': current_price,
                    'sl': sl_sell,
                    'tp': tp_sell,
                    'reason': reason
                })
            else:
                if confidence == 'medium' and not rsi_div_bear:
                    reasons_not_met.append("Markdown conditions not met: Missing RSI bearish divergence for medium confidence.")
                if confidence == 'high' and not obv_falling:
                    reasons_not_met.append("Markdown conditions not met: Missing OBV falling for high confidence.")

    else:
        reasons_not_met.append("No clear Wyckoff phase detected: Monitor price action, volume, and indicators.")

    return {'phase': phase, 'confidence': confidence, 'signals': signals, 'reasons_not_met': reasons_not_met, 'sideways_count': sideways_count, 'hh_in_range': hh_in_range, 'll_in_range': ll_in_range, 'hh_detected': hh_detected, 'hl_detected': hl_detected, 'lh_detected': lh_detected, 'll_detected': ll_detected}

def create_wyckoff_chart(df_live, phase, indicators_to_show, show_ema, show_bb, interval, hh_in_range, ll_in_range, hh_detected, hl_detected, lh_detected, ll_detected):
    num_subplots = len(indicators_to_show) + 1
    row_heights = [0.6] + [0.4 / len(indicators_to_show)] * len(indicators_to_show) if indicators_to_show else [1.0]
    
    subplot_titles = [f'Wyckoff Method Analysis - {phase.capitalize()}']
    subplot_titles.extend(indicators_to_show)
    
    fig = make_subplots(
        rows=num_subplots, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=subplot_titles,
        row_heights=row_heights
    )
    
    # Candlestick
    fig.add_trace(
        go.Candlestick(
            x=df_live['closetime'],
            open=df_live['open'],
            high=df_live['high'],
            low=df_live['low'],
            close=df_live['close'],
            name='Price'
        ),
        row=1, col=1
    )
    
    # EMAs
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
        fig.add_trace(
            go.Scatter(
                x=df_live['closetime'],
                y=df_live['ema50'],
                name='EMA (50)',
                line=dict(color='purple', width=1.5)
            ),
            row=1, col=1
        )
    
    # Bollinger Bands
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
    
    # Support and Resistance
    fig.add_hline(y=df_live['support'].iloc[-1], line_dash="dash", line_color="green", row=1, col=1, annotation_text="Support")
    fig.add_hline(y=df_live['resistance'].iloc[-1], line_dash="dash", line_color="red", row=1, col=1, annotation_text="Resistance")
    
    # Zigzag Line and Labels
    pivots = get_zigzag(df_live, interval)
    if pivots:
        x = [p[0] for p in pivots]
        y = [p[1] for p in pivots]
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode='lines+markers',
                name='ZigZag',
                line=dict(color='white', width=2),
                marker=dict(size=8, symbol='circle', color='white')
            ),
            row=1, col=1
        )

        high_pivots = [p for p in pivots if p[2] == 1]
        low_pivots = [p for p in pivots if p[2] == -1]

        for i in range(1, len(high_pivots)):
            curr_x, curr_y, _ = high_pivots[i]
            prev_x, prev_y, _ = high_pivots[i-1]
            label = 'HH' if curr_y > prev_y else 'LH'
            fig.add_annotation(
                x=curr_x,
                y=curr_y,
                text=f"{label}: {curr_y:.2f}",
                showarrow=True,
                arrowhead=1,
                arrowsize=1,
                arrowwidth=1,
                arrowcolor='white',
                ax=0,
                ay=40,
                font=dict(color='white', size=12),
                align='center'
            )

        for i in range(1, len(low_pivots)):
            curr_x, curr_y, _ = low_pivots[i]
            prev_x, prev_y, _ = low_pivots[i-1]
            label = 'HL' if curr_y > prev_y else 'LL'
            fig.add_annotation(
                x=curr_x,
                y=curr_y,
                text=f"{label}: {curr_y:.2f}",
                showarrow=True,
                arrowhead=1,
                arrowsize=1,
                arrowwidth=1,
                arrowcolor='white',
                ax=0,
                ay=-40,
                font=dict(color='white', size=12),
                align='center'
            )

    # Add markers for detected HH, HL, LH, LL from boolean columns (only significant swings)
    # HH: Green triangle-up at high
    hh_df = df_live[df_live['hh']]
    if not hh_df.empty:
        fig.add_trace(
            go.Scatter(
                x=hh_df['closetime'],
                y=hh_df['high'],
                mode='markers+text',
                name='HH',
                marker=dict(symbol='triangle-up', color='lime', size=12),
                text=['HH' for _ in range(len(hh_df))],
                textposition='top center',
                showlegend=True
            ),
            row=1, col=1
        )

    # HL: Green triangle-up at low (higher low is bullish)
    hl_df = df_live[df_live['hl']]
    if not hl_df.empty:
        fig.add_trace(
            go.Scatter(
                x=hl_df['closetime'],
                y=hl_df['low'],
                mode='markers+text',
                name='HL',
                marker=dict(symbol='triangle-up', color='lime', size=12),
                text=['HL' for _ in range(len(hl_df))],
                textposition='bottom center',
                showlegend=True
            ),
            row=1, col=1
        )

    # LH: Red triangle-down at high (lower high is bearish)
    lh_df = df_live[df_live['lh']]
    if not lh_df.empty:
        fig.add_trace(
            go.Scatter(
                x=lh_df['closetime'],
                y=lh_df['high'],
                mode='markers+text',
                name='LH',
                marker=dict(symbol='triangle-down', color='red', size=12),
                text=['LH' for _ in range(len(lh_df))],
                textposition='top center',
                showlegend=True
            ),
            row=1, col=1
        )

    # LL: Red triangle-down at low
    ll_df = df_live[df_live['ll']]
    if not ll_df.empty:
        fig.add_trace(
            go.Scatter(
                x=ll_df['closetime'],
                y=ll_df['low'],
                mode='markers+text',
                name='LL',
                marker=dict(symbol='triangle-down', color='red', size=12),
                text=['LL' for _ in range(len(ll_df))],
                textposition='bottom center',
                showlegend=True
            ),
            row=1, col=1
        )

    # Phase annotation
    if phase != 'none':
        fig.add_annotation(
            x=df_live['closetime'].iloc[-1],
            y=df_live['high'].max() * 1.05,
            text=f"Current Phase: {phase.capitalize()}",
            showarrow=False,
            font=dict(size=16, color="white"),
            bgcolor="rgba(0,0,0,0.5)",
            bordercolor="white",
            borderwidth=1,
            borderpad=4,
            align="center"
        )
        if phase in ['accumulation', 'distribution']:
            recent_start = df_live['closetime'].iloc[-20] if len(df_live) > 20 else df_live['closetime'].iloc[0]
            fig.add_shape(
                type="rect",
                x0=recent_start,
                y0=df_live['support'].iloc[-1],
                x1=df_live['closetime'].iloc[-1],
                y1=df_live['resistance'].iloc[-1],
                fillcolor="rgba(255, 255, 0, 0.1)" if phase == 'accumulation' else "rgba(255, 0, 0, 0.1)",
                line=dict(width=0),
                row=1, col=1
            )
            hh_detected_text = "Yes" if hh_in_range else "No"
            ll_detected_text = "Yes" if ll_in_range else "No"
            detection_text = f"Multiple HH Detected: {hh_detected_text}<br>Multiple LL Detected: {ll_detected_text}"
            fig.add_annotation(
                x=df_live['closetime'].iloc[-1],
                y=df_live['high'].max() * 0.95,
                text=detection_text,
                showarrow=False,
                font=dict(size=12, color="white"),
                bgcolor="rgba(139, 69, 19, 0.5)",  # Sepia-like color
                bordercolor="white",
                borderwidth=1,
                borderpad=4,
                align="center"
            )
        elif phase == 'markup':
            hh_detected_text = "Yes" if hh_detected else "No"
            hl_detected_text = "Yes" if hl_detected else "No"
            detection_text = f"Multiple HH Detected: {hh_detected_text}<br>Multiple HL Detected: {hl_detected_text}"
            fig.add_annotation(
                x=df_live['closetime'].iloc[-1],
                y=df_live['high'].max() * 0.95,
                text=detection_text,
                showarrow=False,
                font=dict(size=12, color="white"),
                bgcolor="rgba(0, 128, 0, 0.5)",  # Green for bullish
                bordercolor="white",
                borderwidth=1,
                borderpad=4,
                align="center"
            )
        elif phase == 'markdown':
            lh_detected_text = "Yes" if lh_detected else "No"
            ll_detected_text = "Yes" if ll_detected else "No"
            detection_text = f"Multiple LH Detected: {lh_detected_text}<br>Multiple LL Detected: {ll_detected_text}"
            fig.add_annotation(
                x=df_live['closetime'].iloc[-1],
                y=df_live['high'].max() * 0.95,
                text=detection_text,
                showarrow=False,
                font=dict(size=12, color="white"),
                bgcolor="rgba(128, 0, 0, 0.5)",  # Red for bearish
                bordercolor="white",
                borderwidth=1,
                borderpad=4,
                align="center"
            )

    # Volume
    current_row = 2
    if "Volume" in indicators_to_show:
        colors_volume = ['rgba(255,0,0,0.8)' if row['open'] > row['close'] else 'rgba(0,255,0,0.8)' for _, row in df_live.iterrows()]
        fig.add_trace(
            go.Bar(
                x=df_live['closetime'],
                y=df_live['volume'],
                name='Volume',
                marker_color=colors_volume,
                opacity=0.8
            ),
            row=current_row, col=1
        )
        # Add average volume line
        fig.add_trace(
            go.Scatter(
                x=df_live['closetime'],
                y=df_live['avg_volume'],
                name='Avg Volume (20)',
                line=dict(color='yellow', width=1.5, dash='dash')
            ),
            row=current_row, col=1
        )
        fig.update_yaxes(title_text="Volume", row=current_row, col=1, gridcolor='rgba(255,255,255,0.1)')
        current_row += 1
    
    # RSI
    if "RSI" in indicators_to_show:
        fig.add_trace(
            go.Scatter(
                x=df_live['closetime'],
                y=df_live['rsi'],
                name='RSI',
                line=dict(color='purple', width=2)
            ),
            row=current_row, col=1
        )
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=current_row, col=1, annotation_text="Overbought (70)")
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=current_row, col=1, annotation_text="Oversold (30)")
        fig.add_hline(y=50, line_dash="dot", line_color="gray", row=current_row, col=1)
        fig.update_yaxes(title_text="RSI", row=current_row, col=1, gridcolor='rgba(255,255,255,0.1)')
        current_row += 1
    
    # MACD
    if "MACD" in indicators_to_show:
        fig.add_trace(
            go.Scatter(
                x=df_live['closetime'],
                y=df_live['macd'],
                name='MACD Line',
                line=dict(color='blue', width=2)
            ),
            row=current_row, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=df_live['closetime'],
                y=df_live['macd_signal'],
                name='Signal Line',
                line=dict(color='orange', width=2)
            ),
            row=current_row, col=1
        )
        histogram_colors = ['rgba(0,255,0,0.8)' if val >= 0 else 'rgba(255,0,0,0.8)' for val in df_live['macd_histogram']]
        fig.add_trace(
            go.Bar(
                x=df_live['closetime'],
                y=df_live['macd_histogram'],
                name='MACD Histogram',
                marker_color=histogram_colors,
                opacity=0.8
            ),
            row=current_row, col=1
        )
        fig.add_hline(y=0, line_dash="dot", line_color="gray", row=current_row, col=1)
        fig.update_yaxes(title_text="MACD", row=current_row, col=1, gridcolor='rgba(255,255,255,0.1)')
    
    # Adjust zoom to show count_window candles
    count_window = {
        '1m': 65, '3m': 50, '5m': 50, '15m': 70, '30m': 85,
        '1h': 100, '2h': 100, '4h': 90, '6h': 90, '8h': 90, '12h': 90,
        '1d': 60, '3d': 60, '1w': 60, '1M': 60
    }.get(interval, 30)
    
    fig.update_layout(
        height=900,
        title=f"Wyckoff Method Analysis - {phase.capitalize()}",
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
    
    if len(df_live) > 0:
        initial_start_idx = max(0, len(df_live) - count_window)
        initial_start = df_live['closetime'].iloc[initial_start_idx]
        initial_end = df_live['closetime'].iloc[-1]
        for r in range(1, num_subplots + 1):
            fig.update_xaxes(range=[initial_start, initial_end], row=r, col=1, matches='x')
    
    return fig

def run_wyckoff_analysis(symbol, interval, candle_limit=1000, show_ema=True, show_bb=True, 
                        show_volume=True, show_rsi=True, show_macd=True):
    try:
        # Fetch data
        df = fetch_candles_from_binance(symbol, interval, candle_limit)
        if df is None or len(df) < 50:
            return None
        
        # Calculate indicators
        df = calculate_indicators(df)
        
        # Add HH/HL/LH/LL detection columns
        df = detect_hh_hl_lh_ll_columns(df, interval)
        
        # Get current price
        current_price = fetch_current_price(symbol) or df['close'].iloc[-1]
        
        # Detect Wyckoff signals
        analysis = detect_wyckoff_signals(df, current_price, interval)
        
        # Create chart
        chart_html = create_wyckoff_chart(
            df, analysis['phase'], [], show_ema, show_bb, 
            interval, analysis['hh_in_range'], analysis['ll_in_range'], 
            analysis['hh_detected'], analysis['hl_detected'], 
            analysis['lh_detected'], analysis['ll_detected']
        ).to_html(include_plotlyjs=False, div_id=f"wyckoff-chart-{symbol}")
        
        # Prepare result
        result = {
            'symbol': symbol,
            'interval': interval,
            'current_price': current_price,
            'phase': analysis['phase'],
            'confidence': analysis['confidence'],
            'signals': analysis['signals'],
            'reasons_not_met': analysis['reasons_not_met'],
            'technical_indicators': {
                'rsi': df['rsi'].iloc[-1],
                'ema_short': df['ema'].iloc[-1],
                'ema_long': df['ema50'].iloc[-1],
                'atr': df['atr'].iloc[-1],
                'bb_percent': df['bb_percentB'].iloc[-1] * 100,
                'support': df['support'].iloc[-1],
                'resistance': df['resistance'].iloc[-1],
                'volume': df['volume'].iloc[-1],
                'avg_volume': df['avg_volume'].iloc[-1],
                'macd': df['macd'].iloc[-1],
                'macd_signal': df['macd_signal'].iloc[-1]
            },
            'chart_html': chart_html,
            'sideways_count': analysis['sideways_count'],
            'hh_count': df['hh'].tail(30).sum(),
            'hl_count': df['hl'].tail(30).sum(),
            'lh_count': df['lh'].tail(30).sum(),
            'll_count': df['ll'].tail(30).sum()
        }
        
        return result
        
    except Exception as e:
        print(f"Error in Wyckoff analysis for {symbol}: {e}")
        return None