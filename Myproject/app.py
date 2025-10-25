import json
import os
from datetime import datetime, timezone, timedelta
from flask import Flask, make_response, render_template, request, session, jsonify
from flask_apscheduler import APScheduler
import threading
import time
import sqlite3

# Import database functions
from database import (
    init_db, load_budget, save_budget, load_active_trades, save_active_trades,
    load_trade_history, save_trade_history, load_analysis_config, save_analysis_config,
    load_combined_settings, save_combined_settings, load_analysis_state, save_analysis_state
)

# Initialize database
init_db()

# Analysis state tracking - loaded from database
analysis_state = load_analysis_state()
last_analysis_start_time = analysis_state.get('last_analysis_start_time')
last_analysis_end_time = analysis_state.get('last_analysis_end_time')
analysis_in_progress = analysis_state.get('analysis_in_progress', False)

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'

scheduler = APScheduler()
scheduler.init_app(app)


# Import analysis functions
from fibonacci import run_fibonacci_analysis
from elliott import run_elliott_analysis
from ichimoku import run_ichimoku_analysis
from wyckoff import run_wyckoff_analysis
from gann import run_gann_analysis

# Trade constants
FIXED_TRADE_AMOUNT = 250.0
BINANCE_FEE_RATE = 0.001

# Add these imports at the top
import threading
from contextlib import contextmanager

# Global locks for trade management
analysis_lock = threading.Lock()
trade_management_lock = threading.Lock()

@contextmanager
def safe_trade_closure():
    """Context manager to ensure only one process closes trades at a time"""
    acquired = trade_management_lock.acquire(blocking=True, timeout=5)
    if not acquired:
        raise Exception("Could not acquire trade management lock")
    try:
        yield
    finally:
        trade_management_lock.release()


def close_single_trade(tool, trade_key, trade, current_price, session_data, reason=""):
    """
    Centralized function to close a single trade safely
    Returns: (success: bool, closed_trade: dict or None)
    """
    try:
        action = trade.get('action')
        stop_loss = trade.get('stop_loss')
        take_profit = trade.get('take_profit')
        entry_price = trade.get('entry_price')
        net_investment = trade.get('net_investment')
        entry_fee = trade.get('entry_fee', 0.0)
        invested_amount = trade.get('invested_amount', FIXED_TRADE_AMOUNT)
        
        # Validation
        if action not in ['BUY', 'SELL']:
            print(f"⚠️ Invalid action for {tool}:{trade_key}: {action}")
            return False, None
            
        if stop_loss is None or take_profit is None:
            print(f"⚠️ Missing SL/TP for {tool}:{trade_key}")
            return False, None
            
        if entry_price is None or net_investment is None:
            print(f"⚠️ Missing entry data for {tool}:{trade_key}")
            return False, None
        
        # Calculate position and fees
        position_size = net_investment / entry_price
        position_value = position_size * current_price
        closing_fee, _ = calculate_trade_costs(position_value, is_opening=False)
        
        # Calculate profit/loss
        if action == "BUY":
            gross_profit_usd = (current_price - entry_price) * position_size
        else:  # SELL
            gross_profit_usd = (entry_price - current_price) * position_size
        
        # Net profit after ALL fees
        net_profit_usd = gross_profit_usd - entry_fee - closing_fee
        gross_profit_percent = (gross_profit_usd / invested_amount) * 100
        net_profit_percent = (net_profit_usd / invested_amount) * 100
        outcome = 'win' if net_profit_usd > 0 else 'loss'
        
        # Create closed trade record
        closed_trade = trade.copy()
        closed_trade.update({
            'outcome': outcome,
            'close_price': current_price,
            'profit_pct': gross_profit_percent,
            'net_profit_pct': net_profit_percent,
            'net_profit_usd': net_profit_usd,
            'gross_profit_usd': gross_profit_usd,
            'closing_fee': closing_fee,
            'total_fees': entry_fee + closing_fee,
            'close_time': datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S'),
            'interval': trade.get('interval', 'N/A'),
            'close_reason': reason  # NEW: Track why trade was closed
        })
        
        # Add to history
        session_data['trade_history'][tool].append(closed_trade)
        
        # Update budget
        update_budget(
            invested_amount,
            closing_fee,
            "return",
            net_profit_usd=net_profit_usd
        )
        
        # Remove from active trades
        del session_data['active_trades'][tool][trade_key]
        
        # Log closure
        print(f"✅ {tool.capitalize()} trade CLOSED for {trade_key}: {outcome.upper()} ({reason})")
        print(f"   Entry: ${entry_price:.4f}, Close: ${current_price:.4f}")
        print(f"   Net P/L: ${net_profit_usd:.2f} ({net_profit_percent:.2f}%)")
        print(f"   Fees: Open ${entry_fee:.2f} + Close ${closing_fee:.2f}")
        
        return True, closed_trade
        
    except Exception as e:
        print(f"❌ Error closing trade {tool}:{trade_key}: {e}")
        import traceback
        traceback.print_exc()
        return False, None

def update_analysis_state():
    """Update analysis state in database"""
    global last_analysis_start_time, last_analysis_end_time, analysis_in_progress
    state_data = {
        'last_analysis_start_time': last_analysis_start_time,
        'last_analysis_end_time': last_analysis_end_time,
        'analysis_in_progress': analysis_in_progress
    }
    save_analysis_state(state_data)

def calculate_trade_costs(investment_amount, is_opening=True, position_value=None):
    """Calculate fees according to Binance standard spot trading - FIXED"""
    fee_rate = BINANCE_FEE_RATE
    
    if is_opening:
        fee_amount = investment_amount * fee_rate
        net_amount = investment_amount - fee_amount
        return fee_amount, net_amount
    else:
        if position_value is None:
            position_value = investment_amount
        fee_amount = position_value * fee_rate
        return fee_amount, position_value - fee_amount
    
def calculate_total_invested():
    """Calculate total invested from active trades"""
    active_trades, _ = load_trade_data()
    total_invested = 0.0
    
    for tool in active_trades:
        for trade_key, trade in active_trades[tool].items():
            total_invested += trade.get('invested_amount', 0.0)
    
    return total_invested


def update_budget(investment_amount, fee_amount, action="use", net_profit_usd=0.0):
    """Update budget when trade is opened or closed - FIXED"""
    budget = load_budget()
    
    if action == "use":
        # When opening trade: reserve the full $250 investment amount
        budget['used_budget'] += investment_amount
        budget['total_fees'] += fee_amount
        budget['remaining_budget'] = budget['total_budget'] - budget['used_budget']
        
        print(f"Budget Update - OPEN: Used +${investment_amount:.2f}, Total Used=${budget['used_budget']:.2f}, Remaining=${budget['remaining_budget']:.2f}")
        
    elif action == "return":
        # When closing trade: free up the $250 investment amount
        budget['used_budget'] -= investment_amount
        
        # CRITICAL FIX: Prevent negative used_budget
        if budget['used_budget'] < -0.01:
            print(f"⚠️ ERROR: used_budget became negative: {budget['used_budget']:.2f}")
            budget['used_budget'] = calculate_total_invested()
        
        # Ensure used_budget never goes below 0
        budget['used_budget'] = max(0, budget['used_budget'])
        
        # Add closing fee to total fees
        budget['total_fees'] += fee_amount
        
        # Update total budget with net profit/loss
        budget['total_budget'] += net_profit_usd
        
        # Recalculate remaining budget
        budget['remaining_budget'] = budget['total_budget'] - budget['used_budget']
        
        print(f"Budget Update - CLOSE: Used -${investment_amount:.2f}, Total Used=${budget['used_budget']:.2f}")
        print(f"   Total=${budget['total_budget']:.2f}, Remaining=${budget['remaining_budget']:.2f}, P/L=${net_profit_usd:.2f}")
    
    # SYNC CHECK: Verify used_budget matches active trades
    actual_invested = calculate_total_invested()
    if abs(budget['used_budget'] - actual_invested) > 1.0:
        print(f"⚠️ SYNC ERROR: used_budget={budget['used_budget']:.2f} vs actual={actual_invested:.2f}")
        print(f"✅ Auto-correcting to {actual_invested:.2f}")
        budget['used_budget'] = actual_invested
        budget['remaining_budget'] = budget['total_budget'] - budget['used_budget']
    
    save_budget(budget)
    return True



def can_open_trade():
    """Check if there's enough budget to open a new $250 trade"""
    budget = load_budget()
    total_cost = FIXED_TRADE_AMOUNT
    return budget['remaining_budget'] >= total_cost, total_cost, FIXED_TRADE_AMOUNT

def load_trade_data():
    """Load trade data from database"""
    active_trades = load_active_trades()
    trade_history = load_trade_history()
    return active_trades, trade_history

def save_trade_data(active_trades, trade_history):
    """Save trade data to database"""
    save_active_trades(active_trades)
    save_trade_history(trade_history)
    return True

def load_complete_trade_history():
    """Load complete trade history from database"""
    return load_trade_history()

def normalize_analysis(analysis, tool):
    """Normalize analysis output to ensure required keys"""
    if not analysis:
        return None
    default_keys = {
        'fibonacci': {
            'symbol': 'N/A',
            'interval': 'N/A',
            'current_price': 0.0,
            'trend': 'none',
            'confidence': 'low',
            'signals': [],
            'signal_descriptions': [],
            'trade_action': None,
            'entry_price': None,
            'stop_loss': None,
            'take_profit': None,
            'risk_reward_ratio': None,
            'swing_high': 0.0,
            'swing_low': 0.0,
            'trend_class': 'trend-none',
            'confidence_class': 'confidence-low',
            'last_candle': 'N/A',
            'data_from': 'N/A',
            'data_to': 'N/A',
            'chart_html': '',
            'fib_html': '',
            'closest_info': 'No levels detected',
            'fib_explanation': '',
            'rsi_value': 0.0,
            'ema_value': 0.0,
            'ema_rel': 'N/A',
            'atr_value': 0.0,
            'bb_value': 0.0,
            'bb_status': 'N/A'
        },
        'elliott': {
            'symbol': 'N/A',
            'interval': 'N/A',
            'current_price': 0.0,
            'wave_data_by_degree': {},
            'charts': {},
            'technical_indicators': {
                'rsi': 0.0,
                'ema': 0.0,
                'bb_percent': 0.0
            }
        },
        'ichimoku': {
            'symbol': 'N/A',
            'interval': 'N/A',
            'current_price': 0.0,
            'trend': 'none',
            'confidence': 'low',
            'cloud_bullish': False,
            'signals': [],
            'reasons_not_met': [],
            'chart_html': '',
            'technical_indicators': {
                'rsi': 0.0,
                'ema': 0.0,
                'atr': 0.0,
                'bb_percent': 0.0,
                'tenkan_sen': 0.0,
                'kijun_sen': 0.0,
                'senkou_span_a': 0.0,
                'senkou_span_b': 0.0
            }
        },
        'wyckoff': {
            'symbol': 'N/A',
            'interval': 'N/A',
            'current_price': 0.0,
            'phase': 'none',
            'confidence': 'low',
            'sideways_count': 0,
            'signals': [],
            'reasons_not_met': [],
            'chart_html': '',
            'technical_indicators': {
                'rsi': 0.0,
                'ema_short': 0.0,
                'ema_long': 0.0,
                'atr': 0.0,
                'bb_percent': 0.0,
                'support': 0.0,
                'resistance': 0.0
            }
        },
        'gann': {
            'symbol': 'N/A',
            'interval': 'N/A',
            'current_price': 0.0,
            'trend': 'none',
            'confidence': 'low',
            'pivot_type': 'N/A',
            'price_position': 'N/A',
            'nearest_support': 0.0,
            'nearest_resistance': 0.0,
            'subtools': ['Gann Fan', 'Gann Square', 'Gann Box', 'Gann Square Fixed'],
            'charts': {},
            'technical_indicators': {
                'rsi': 0.0,
                'ema': 0.0,
                'atr': 0.0,
                'bb_percent': 0.0
            }
        }
    }
    normalized = default_keys[tool].copy()
    if analysis:
        normalized.update(analysis)
    return normalized

def run_analysis_for_tool(tool, symbols, interval, candle_limit, config):
    """Run analysis for a specific tool and return results. Uses config dict instead of request_form."""
    analyses = {}
    
    if tool == 'fibonacci':
        window = config.get('fib_window', 60)
        fib_threshold = config.get('fib_threshold', 0.003)
        
        for symbol in symbols:
            try:
                analysis = run_fibonacci_analysis(symbol, interval, candle_limit, window, fib_threshold)
                if analysis:
                    analysis = normalize_analysis(analysis, 'fibonacci')
                    analyses[symbol] = analysis
            except Exception as e:
                print(f"Error running Fibonacci analysis for {symbol}: {e}")
                
    elif tool == 'elliott':
        thresholds = config.get('elliott_thresholds', {'Minor': 0.005, 'Intermediate': 0.02, 'Major': 0.05})
        
        selected_degrees = config.get('elliott_degrees', ['Minor', 'Intermediate', 'Major'])
        use_smoothing = config.get('use_smoothing', False)
        smooth_period = config.get('smooth_period', 3)
        
        show_ema = config.get('show_ema', False)
        show_bb = config.get('show_bb', False)
        show_volume = config.get('show_volume', False)
        show_rsi = config.get('show_rsi', False)
        show_macd = config.get('show_macd', False)
        
        for symbol in symbols:
            try:
                analysis = run_elliott_analysis(
                    symbol, interval, candle_limit, thresholds, selected_degrees,
                    use_smoothing, smooth_period, show_ema, show_bb, show_volume, show_rsi, show_macd
                )
                if analysis:
                    analysis = normalize_analysis(analysis, 'elliott')
                    analyses[symbol] = analysis
            except Exception as e:
                print(f"Error running Elliott analysis for {symbol}: {e}")
                
    elif tool == 'ichimoku':
        show_ema_ichimoku = config.get('show_ema_ichimoku', False)
        show_bb_ichimoku = config.get('show_bb_ichimoku', False)
        show_volume_ichimoku = config.get('show_volume_ichimoku', False)
        show_rsi_ichimoku = config.get('show_rsi_ichimoku', False)
        show_macd_ichimoku = config.get('show_macd_ichimoku', False)
        
        for symbol in symbols:
            try:
                analysis = run_ichimoku_analysis(
                    symbol, interval, candle_limit,
                    show_ema_ichimoku, show_bb_ichimoku, show_volume_ichimoku, 
                    show_rsi_ichimoku, show_macd_ichimoku
                )
                if analysis:
                    analysis = normalize_analysis(analysis, 'ichimoku')
                    analyses[symbol] = analysis
            except Exception as e:
                print(f"Error running Ichimoku analysis for {symbol}: {e}")
                
    elif tool == 'wyckoff':
        show_ema_wyckoff = config.get('show_ema_wyckoff', False)
        show_bb_wyckoff = config.get('show_bb_wyckoff', False)
        show_volume_wyckoff = config.get('show_volume_wyckoff', False)
        show_rsi_wyckoff = config.get('show_rsi_wyckoff', False)
        show_macd_wyckoff = config.get('show_macd_wyckoff', False)
        
        for symbol in symbols:
            try:
                analysis = run_wyckoff_analysis(
                    symbol, interval, candle_limit,
                    show_ema_wyckoff, show_bb_wyckoff, show_volume_wyckoff,
                    show_rsi_wyckoff, show_macd_wyckoff
                )
                if analysis:
                    analysis = normalize_analysis(analysis, 'wyckoff')
                    analyses[symbol] = analysis
            except Exception as e:
                print(f"Error running Wyckoff analysis for {symbol}: {e}")
                
    elif tool == 'gann':
        gann_subtools = config.get('gann_subtools', ['Gann Fan', 'Gann Square', 'Gann Box', 'Gann Square Fixed'])
        
        pivot_choice = config.get('pivot_choice', 'Auto (based on trend)')
        show_ema_gann = config.get('show_ema_gann', False)
        show_bb_gann = config.get('show_bb_gann', False)
        show_volume_gann = config.get('show_volume_gann', False)
        show_rsi_gann = config.get('show_rsi_gann', False)
        show_macd_gann = config.get('show_macd_gann', False)
        
        for symbol in symbols:
            try:
                analysis = run_gann_analysis(
                    symbol, interval, candle_limit,
                    gann_subtools, pivot_choice,
                    show_ema_gann, show_bb_gann, show_volume_gann, show_rsi_gann, show_macd_gann
                )
                if analysis:
                    analysis = normalize_analysis(analysis, 'gann')
                    if not analysis.get('subtools'):
                        analysis['subtools'] = gann_subtools
                    analyses[symbol] = analysis
            except Exception as e:
                print(f"Error running Gann analysis for {symbol}: {e}")
    
    return analyses

def convert_confidence_to_numeric(confidence_str):
    """Convert confidence string to numeric value"""
    confidence_map = {
        'very high': 0.9,
        'high': 0.8,
        'medium': 0.6,
        'low': 0.4,
        'very low': 0.2
    }
    return confidence_map.get(confidence_str.lower(), 0.5)

def convert_numeric_to_confidence(confidence_numeric):
    """Convert numeric confidence to string"""
    if confidence_numeric >= 0.8:
        return 'High'
    elif confidence_numeric >= 0.6:
        return 'Medium'
    elif confidence_numeric >= 0.4:
        return 'Low'
    else:
        return 'Very Low'

def analyze_gann_information(gann_data, current_price):
    """Analyze Gann data to extract trading bias"""
    if not gann_data:
        return {'bias': 'neutral', 'confidence': 'low', 'reasons': ['No Gann data available']}
    
    reasons = []
    bullish_signals = 0
    bearish_signals = 0
    
    # Analyze Gann levels
    if gann_data.get('price_position') == 'Near Support':
        bullish_signals += 1
        reasons.append("Price near Gann support level")
    
    if gann_data.get('price_position') == 'Near Resistance':
        bearish_signals += 1
        reasons.append("Price near Gann resistance level")
    
    # Analyze Gann tools
    analysis_details = gann_data.get('gann_analysis_details', {})
    for tool_name, tool_data in analysis_details.items():
        signal = tool_data.get('signal', '').lower()
        if 'bullish' in signal:
            bullish_signals += 1
            reasons.append(f"{tool_name}: Bullish signal")
        elif 'bearish' in signal:
            bearish_signals += 1
            reasons.append(f"{tool_name}: Bearish signal")
    
    # Determine overall bias
    if bullish_signals > bearish_signals:
        bias = 'bullish'
        confidence = 'medium' if (bullish_signals - bearish_signals) >= 2 else 'low'
    elif bearish_signals > bullish_signals:
        bias = 'bearish'
        confidence = 'medium' if (bearish_signals - bullish_signals) >= 2 else 'low'
    else:
        bias = 'neutral'
        confidence = 'low'
        reasons.append("Mixed signals from Gann tools")
    
    return {
        'bias': bias,
        'confidence': confidence,
        'reasons': reasons,
        'detailed_analysis': analysis_details
    }

def calculate_tp_sl_rr(current_price, action, atr, rr_ratio):
    """Calculate take profit and stop loss with risk-reward ratio"""
    if action == "BUY":
        stop_loss = current_price - (atr * 1.5)
        take_profit = current_price + ((current_price - stop_loss) * rr_ratio)
    else:  # SELL
        stop_loss = current_price + (atr * 1.5)
        take_profit = current_price - ((stop_loss - current_price) * rr_ratio)
    
    return stop_loss, take_profit


def manage_trades(tool, analyses, session_data, interval, enable_buy=True, enable_sell=True):
    """
    FIXED: Opens trades and saves IMMEDIATELY to database
    No batch saving at end - each trade saved individually
    """
    active_trades = session_data['active_trades']
    budget = load_budget()
    
    for symbol, analysis in analyses.items():
        current_price = analysis['current_price']
        
        if tool == 'elliott':
            for degree in analysis.get('wave_data_by_degree', {}).keys():
                trade_key = f"{symbol}_{degree}"
                wave_data = analysis['wave_data_by_degree'].get(degree, {})
                
                # Check if new trade should be opened
                if trade_key not in active_trades[tool] and wave_data.get('signals'):
                    for signal in wave_data['signals']:
                        action_type = signal['type']
                        if (action_type == 'BUY' and not enable_buy) or (action_type == 'SELL' and not enable_sell):
                            continue
                            
                        # Use Elliott's own SL/TP values
                        entry_price = signal.get('entry_price', current_price)
                        stop_loss = signal.get('sl')
                        take_profit = signal.get('tp')
                        
                        # Validate Elliott's SL/TP
                        if stop_loss is None or take_profit is None:
                            print(f"⚠️ Elliott {degree} missing SL/TP for {symbol}, skipping trade")
                            continue
                            
                        can_trade, total_cost, investment_amount = can_open_trade()
                        
                        if can_trade:
                            entry_fee, net_investment = calculate_trade_costs(investment_amount, is_opening=True)
                            
                            active_trade = {
                                'symbol': symbol,
                                'degree': degree,
                                'action': action_type,
                                'entry_price': entry_price,
                                'stop_loss': stop_loss,
                                'take_profit': take_profit,
                                'entry_time': datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S'),
                                'reason': signal['reason'],
                                'interval': interval,
                                'position_size': 100 * (investment_amount / budget['total_budget']),
                                'invested_amount': investment_amount,
                                'entry_fee': entry_fee,
                                'net_investment': net_investment
                            }
                            active_trades[tool][trade_key] = active_trade
                            
                            # SAVE IMMEDIATELY to database
                            save_active_trades(active_trades)
                            
                            update_budget(investment_amount, entry_fee, "use")
                            print(f"🎯 {tool.capitalize()} trade OPENED for {symbol} ({degree}): {action_type}")
                            print(f"   Entry: ${entry_price:.4f}, SL: ${stop_loss:.4f}, TP: ${take_profit:.4f}")
                            print(f"   Investment: ${investment_amount:.2f}, Fee: ${entry_fee:.2f}")
                            print(f"   ✅ Saved to database immediately")
                            break
                        else:
                            print(f"❌ Insufficient budget for {tool} on {symbol} ({degree})")
                        break
        
        elif tool == 'ichimoku':
            trade_key = symbol
            
            # Check if new trade should be opened
            if trade_key not in active_trades[tool]:
                signals = analysis.get('signals', [])
                
                for signal in signals:
                    action_type = signal['type']
                    if (action_type == 'BUY' and not enable_buy) or (action_type == 'SELL' and not enable_sell):
                        continue
                    
                    # Use Ichimoku's own SL/TP values
                    entry_price = signal.get('entry_price', current_price)
                    stop_loss = signal.get('sl')
                    take_profit = signal.get('tp')
                    
                    # Validate Ichimoku's SL/TP
                    if stop_loss is None or take_profit is None:
                        print(f"⚠️ Ichimoku missing SL/TP for {symbol}, skipping trade")
                        continue
                        
                    can_trade, total_cost, investment_amount = can_open_trade()
                    
                    if can_trade:
                        entry_fee, net_investment = calculate_trade_costs(investment_amount, is_opening=True)
                        
                        active_trade = {
                            'symbol': symbol,
                            'action': action_type,
                            'entry_price': entry_price,
                            'stop_loss': stop_loss,
                            'take_profit': take_profit,
                            'entry_time': datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S'),
                            'signals': analysis.get('signals', []),
                            'confidence': analysis.get('confidence'),
                            'interval': interval,
                            'position_size': 100 * (investment_amount / budget['total_budget']),
                            'invested_amount': investment_amount,
                            'entry_fee': entry_fee,
                            'net_investment': net_investment,
                            'reason': signal['reason']
                        }
                        active_trades[tool][trade_key] = active_trade
                        
                        # SAVE IMMEDIATELY to database
                        save_active_trades(active_trades)
                        
                        update_budget(investment_amount, entry_fee, "use")
                        print(f"🎯 {tool.capitalize()} trade OPENED for {symbol}: {action_type}")
                        print(f"   Entry: ${entry_price:.4f}, SL: ${stop_loss:.4f}, TP: ${take_profit:.4f}")
                        print(f"   Investment: ${investment_amount:.2f}, Fee: ${entry_fee:.2f}")
                        print(f"   ✅ Saved to database immediately")
                        break
                    else:
                        print(f"❌ Insufficient budget for {tool} on {symbol}")
        
        elif tool == 'wyckoff':
            trade_key = symbol
            
            # Check if new trade should be opened
            if trade_key not in active_trades[tool]:
                signals = analysis.get('signals', [])
                
                for signal in signals:
                    action_type = signal['type']
                    if (action_type == 'BUY' and not enable_buy) or (action_type == 'SELL' and not enable_sell):
                        continue
                    
                    # Use Wyckoff's own SL/TP values
                    entry_price = signal.get('entry_price', current_price)
                    stop_loss = signal.get('sl')
                    take_profit = signal.get('tp')
                    
                    # Validate Wyckoff's SL/TP
                    if stop_loss is None or take_profit is None:
                        print(f"⚠️ Wyckoff missing SL/TP for {symbol}, skipping trade")
                        continue
                        
                    can_trade, total_cost, investment_amount = can_open_trade()
                    
                    if can_trade:
                        entry_fee, net_investment = calculate_trade_costs(investment_amount, is_opening=True)
                        
                        active_trade = {
                            'symbol': symbol,
                            'action': action_type,
                            'entry_price': entry_price,
                            'stop_loss': stop_loss,
                            'take_profit': take_profit,
                            'entry_time': datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S'),
                            'signals': analysis.get('signals', []),
                            'confidence': analysis.get('confidence'),
                            'interval': interval,
                            'position_size': 100 * (investment_amount / budget['total_budget']),
                            'invested_amount': investment_amount,
                            'entry_fee': entry_fee,
                            'net_investment': net_investment,
                            'reason': signal['reason']
                        }
                        active_trades[tool][trade_key] = active_trade
                        
                        # SAVE IMMEDIATELY to database
                        save_active_trades(active_trades)
                        
                        update_budget(investment_amount, entry_fee, "use")
                        print(f"🎯 {tool.capitalize()} trade OPENED for {symbol}: {action_type}")
                        print(f"   Entry: ${entry_price:.4f}, SL: ${stop_loss:.4f}, TP: ${take_profit:.4f}")
                        print(f"   Investment: ${investment_amount:.2f}, Fee: ${entry_fee:.2f}")
                        print(f"   ✅ Saved to database immediately")
                        break
                    else:
                        print(f"❌ Insufficient budget for {tool} on {symbol}")
        
        elif tool == 'fibonacci':
            trade_key = symbol
            
            # Check if new trade should be opened
            if trade_key not in active_trades[tool]:
                trade_action = analysis.get('trade_action')
                entry_price = analysis.get('entry_price', current_price)
                
                # Use Fibonacci's own SL/TP values
                stop_loss = analysis.get('stop_loss')
                take_profit = analysis.get('take_profit')
                
                if trade_action in ['BUY', 'SELL']:
                    if (trade_action == 'BUY' and not enable_buy) or (trade_action == 'SELL' and not enable_sell):
                        continue
                    
                    # Validate Fibonacci's SL/TP
                    if stop_loss is None or take_profit is None:
                        print(f"⚠️ Fibonacci missing SL/TP for {symbol}, skipping trade")
                        continue
                        
                    can_trade, total_cost, investment_amount = can_open_trade()
                    
                    if can_trade:
                        entry_fee, net_investment = calculate_trade_costs(investment_amount, is_opening=True)
                        
                        active_trade = {
                            'symbol': symbol,
                            'action': trade_action,
                            'entry_price': entry_price,
                            'stop_loss': stop_loss,
                            'take_profit': take_profit,
                            'entry_time': datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S'),
                            'signals': analysis.get('signals', []),
                            'signal_descriptions': analysis.get('signal_descriptions', []),
                            'confidence': analysis.get('confidence'),
                            'interval': interval,
                            'position_size': 100 * (investment_amount / budget['total_budget']),
                            'invested_amount': investment_amount,
                            'entry_fee': entry_fee,
                            'net_investment': net_investment
                        }
                        active_trades[tool][trade_key] = active_trade
                        
                        # SAVE IMMEDIATELY to database
                        save_active_trades(active_trades)
                        
                        update_budget(investment_amount, entry_fee, "use")
                        print(f"🎯 {tool.capitalize()} trade OPENED for {symbol}: {trade_action}")
                        print(f"   Entry: ${entry_price:.4f}, SL: ${stop_loss:.4f}, TP: ${take_profit:.4f}")
                        print(f"   Investment: ${investment_amount:.2f}, Fee: ${entry_fee:.2f}")
                        print(f"   ✅ Saved to database immediately")
                    else:
                        print(f"❌ Insufficient budget for {tool} on {symbol}")
        
        elif tool == 'combined':
            trade_key = symbol
            
            # Check if new trade should be opened
            if trade_key not in active_trades[tool]:
                trade_action = analysis.get('action')
                entry_price = analysis.get('entry_price', current_price)
                
                # Use Combined's own SL/TP values (consensus from tools)
                stop_loss = analysis.get('stop_loss')
                take_profit = analysis.get('take_profit')
                
                if trade_action in ['BUY', 'SELL']:
                    if (trade_action == 'BUY' and not enable_buy) or (trade_action == 'SELL' and not enable_sell):
                        continue
                    
                    # For combined, we can allow calculated SL/TP as fallback
                    if stop_loss is None or take_profit is None:
                        print(f"⚠️ Combined missing SL/TP for {symbol}, using 1:2 RR calculation")
                        atr = analysis.get('atr_value', current_price * 0.02)
                        if atr <= 0:
                            atr = current_price * 0.02
                        stop_loss, take_profit = calculate_tp_sl_rr(current_price, trade_action, atr, 2.0)
                        
                    can_trade, total_cost, investment_amount = can_open_trade()
                    
                    if can_trade:
                        entry_fee, net_investment = calculate_trade_costs(investment_amount, is_opening=True)
                        
                        active_trade = {
                            'symbol': symbol,
                            'action': trade_action,
                            'entry_price': entry_price,
                            'stop_loss': stop_loss,
                            'take_profit': take_profit,
                            'entry_time': datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S'),
                            'confidence': analysis.get('confidence'),
                            'interval': interval,
                            'position_size': analysis.get('position_size', 50),
                            'invested_amount': investment_amount,
                            'entry_fee': entry_fee,
                            'net_investment': net_investment,
                            'reasons': analysis.get('reasons', []),
                            'agreement_level': analysis.get('agreement_level', 'medium')
                        }
                        active_trades[tool][trade_key] = active_trade
                        
                        # SAVE IMMEDIATELY to database
                        save_active_trades(active_trades)
                        
                        update_budget(investment_amount, entry_fee, "use")
                        print(f"🎯 {tool.capitalize()} trade OPENED for {symbol}: {trade_action}")
                        print(f"   Entry: ${entry_price:.4f}, SL: ${stop_loss:.4f}, TP: ${take_profit:.4f}")
                        print(f"   Position Size: {analysis.get('position_size', 50)}%")
                        print(f"   Investment: ${investment_amount:.2f}, Fee: ${entry_fee:.2f}")
                        print(f"   ✅ Saved to database immediately")
                    else:
                        print(f"❌ Insufficient budget for {tool} on {symbol}")


def generate_combined_signal(tool_signals, current_price, symbol, interval, tool_weights, combined_settings):
    """Generate combined trading signal based on all tool signals with weighted scoring"""
    
    buy_signals = []
    sell_signals = []
    hold_signals = []
    
    confidence_threshold = combined_settings.get('confidence_threshold', 0.6)
    min_tool_agreement = combined_settings.get('min_tool_agreement', 2)
    rr_ratio_map = {"1:1": 1.0, "1:1.5": 1.5, "1:2": 2.0, "1:2.5": 2.5, "1:3": 3.0}
    selected_rr_ratio = rr_ratio_map.get(combined_settings.get('risk_reward_ratio', '1:2'), 2.0)
    
    # Collect signals from all tools
    for tool, signal_data in tool_signals.items():
        if tool == "gann":
            # Special handling for Gann's indirect information
            gann_analysis = analyze_gann_information(signal_data, current_price)
            action = 'BUY' if gann_analysis['bias'] == 'bullish' else 'SELL' if gann_analysis['bias'] == 'bearish' else 'HOLD'
            confidence_numeric = convert_confidence_to_numeric(gann_analysis['confidence'])
            reason = " | ".join(gann_analysis['reasons']) if gann_analysis['reasons'] else "No clear bias - Neutral position"
            signal_info = {
                'tool': tool,
                'action': action,
                'confidence': confidence_numeric,
                'reason': reason,
                'entry_price': current_price,
                'stop_loss': None,
                'take_profit': None,
                'weight': tool_weights.get(tool, 0.5),
                'is_direct': False,
                'confidence_original': gann_analysis['confidence'],  # This should be string
                'detailed_analysis': gann_analysis['detailed_analysis']
            }
            if action == 'BUY':
                buy_signals.append(signal_info)
            elif action == 'SELL':
                sell_signals.append(signal_info)
            else:
                hold_signals.append(signal_info)
            continue

        # Generalized handling for other tools (Fibonacci, Elliott, Ichimoku, Wyckoff)
        confidence_raw = signal_data.get('confidence', 'Low')
        if isinstance(confidence_raw, (int, float)):
            confidence_str = convert_numeric_to_confidence(confidence_raw)
        else:
            confidence_str = str(confidence_raw)
        confidence_numeric = convert_confidence_to_numeric(confidence_str)

        action = 'HOLD'
        reason = 'No clear signal from tool'
        entry_price = current_price
        stop_loss = None
        take_profit = None

        trade_action = signal_data.get('trade_action')
        signals = signal_data.get('signals', [])

        if tool == 'elliott':
            # Special aggregation for Elliott: collect all signals across degrees
            all_signals = []
            for degree_data in signal_data.get('wave_data_by_degree', {}).values():
                all_signals.extend(degree_data.get('signals', []))
            signals = all_signals  # Override with aggregated

        if trade_action:  # Fibonacci-style
            action = trade_action.upper()
            reason = ' | '.join(signal_data.get('signal_descriptions', ['No description provided']))
            entry_price = signal_data.get('entry_price', current_price)
            stop_loss = signal_data.get('stop_loss')
            take_profit = signal_data.get('take_profit')
        elif signals:  # Signal-based (Elliott, Ichimoku, Wyckoff)
            # Take the first signal (or aggregate if needed)
            signal = signals[0]
            if isinstance(signal, dict):
                # Dict format (assumed for Elliott-like)
                action_type = signal.get('type') or signal.get('action')
                if action_type:
                    action = action_type.upper()
                reason = signal.get('reason', 'No reason provided')
                entry_price = signal.get('entry_price', current_price)
                stop_loss = signal.get('sl') or signal.get('stop_loss')
                take_profit = signal.get('tp') or signal.get('take_profit')
            elif isinstance(signal, str):
                # String format (possible for Ichimoku/Wyckoff signals)
                reason = signal  # Use string as reason
                # Fallback to 'trend' or 'phase' for action
                trend = signal_data.get('trend', 'none').lower()
                phase = signal_data.get('phase', 'none').lower()
                if 'bullish' in trend or 'accumulation' in phase or 'markup' in phase:
                    action = 'BUY'
                elif 'bearish' in trend or 'distribution' in phase or 'markdown' in phase:
                    action = 'SELL'
        else:
            # UPDATED: No fallback to trend/phase for action - set to HOLD to require explicit signals
            action = 'HOLD'
            reason = 'No clear signal from tool'

        signal_info = {
            'tool': tool,
            'action': action,
            'confidence': confidence_numeric,
            'reason': reason,
            'entry_price': entry_price,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'weight': tool_weights.get(tool, 1.0),
            'is_direct': True,
            'confidence_original': confidence_str  # Always string for display
        }
        
        if action == "BUY":
            buy_signals.append(signal_info)
        elif action == "SELL":
            sell_signals.append(signal_info)
        else:
            hold_signals.append(signal_info)
    
    # Calculate weighted combined signal strength
    buy_strength = sum(float(sig['confidence']) * float(sig['weight']) for sig in buy_signals)
    sell_strength = sum(float(sig['confidence']) * float(sig['weight']) for sig in sell_signals)
    hold_strength = sum(float(sig['confidence']) * float(sig['weight']) for sig in hold_signals)
    
    # Only count direct tools for agreement (exclude Gann)
    direct_tools = [sig for sig in buy_signals + sell_signals + hold_signals if sig['is_direct']]
    total_direct_tools = len(set(sig['tool'] for sig in direct_tools))  # Number of tools that provided a signal
    buy_count = len([sig for sig in buy_signals if sig['is_direct']])
    sell_count = len([sig for sig in sell_signals if sig['is_direct']])
    hold_count = len([sig for sig in hold_signals if sig['is_direct']])
    
    # Calculate agreement percentage based on direct tools only
    total_signals = buy_count + sell_count + hold_count
    agreement_percentage = max(buy_count, sell_count, hold_count) / total_signals if total_signals > 0 else 0
    
    # Determine combined action with weighted scoring
    weighted_buy_score = buy_strength * agreement_percentage if buy_count > 0 else 0
    weighted_sell_score = sell_strength * agreement_percentage if sell_count > 0 else 0
    weighted_hold_score = hold_strength * agreement_percentage if hold_count > 0 else 0
    
    max_score = max(weighted_buy_score, weighted_sell_score, weighted_hold_score)
    
    reasons = []
    
    if max_score == weighted_buy_score and buy_count >= min_tool_agreement and weighted_buy_score >= confidence_threshold:
        combined_action = "BUY"
        combined_confidence_numeric = weighted_buy_score
        reasons = [f"{sig['tool']} ({sig['weight']}x): {sig['reason']}" for sig in buy_signals]
        agreement_level = 'high' if agreement_percentage >= 0.75 else 'medium' if agreement_percentage >= 0.5 else 'low'
        
    elif max_score == weighted_sell_score and sell_count >= min_tool_agreement and weighted_sell_score >= confidence_threshold:
        combined_action = "SELL"
        combined_confidence_numeric = weighted_sell_score
        reasons = [f"{sig['tool']} ({sig['weight']}x): {sig['reason']}" for sig in sell_signals]
        agreement_level = 'high' if agreement_percentage >= 0.75 else 'medium' if agreement_percentage >= 0.5 else 'low'
        
    else:
        # Default to HOLD if no clear buy/sell or if hold is strongest
        combined_action = "HOLD"
        combined_confidence_numeric = weighted_hold_score if weighted_hold_score > 0 else 0.5
        if total_signals == 0:
            reasons.append("No signals from any tools")
        if max(buy_count, sell_count) < min_tool_agreement:
            reasons.append(f"Insufficient tool agreement: {max(buy_count, sell_count)}/{min_tool_agreement} tools agree on BUY/SELL")
        if max(weighted_buy_score, weighted_sell_score) < confidence_threshold:
            reasons.append(f"Confidence score {max(weighted_buy_score, weighted_sell_score):.2f} below threshold {confidence_threshold}")
        if not reasons:
            reasons = ["Insufficient tool agreement or confidence"]
        reasons += [f"{sig['tool']} ({sig['weight']}x): {sig['reason']}" for sig in hold_signals]
        agreement_level = 'medium' if combined_confidence_numeric > 0.5 else 'low'
    
    # Convert numeric confidence back to string for display
    combined_confidence = convert_numeric_to_confidence(combined_confidence_numeric)
    
    # Calculate position sizing based on confidence and agreement
    if combined_action in ["BUY", "SELL"] and combined_confidence_numeric > 0:
        # Dynamic position sizing based on confidence and agreement
        base_size = min(combined_confidence_numeric * 100, 100)
        agreement_multiplier = 1.0 if agreement_level == 'high' else 0.7 if agreement_level == 'medium' else 0.5
        position_size = base_size * agreement_multiplier
        
        # Calculate stop loss and take profit with selected risk-reward ratio
        atr = tool_signals.get('fibonacci', {}).get('atr_value', current_price * 0.02)
        if atr <= 0:
            atr = current_price * 0.02
        stop_loss, take_profit = calculate_tp_sl_rr(current_price, combined_action, atr, selected_rr_ratio)
        
    else:
        position_size = 0
        stop_loss = None
        take_profit = None
    
    return {
        'action': combined_action,
        'confidence': combined_confidence,  # This is now a string
        'confidence_numeric': combined_confidence_numeric,  # This is the numeric value
        'position_size': position_size,
        'entry_price': current_price,
        'stop_loss': stop_loss,
        'take_profit': take_profit,
        'reasons': reasons,
        'agreement_level': agreement_level,
        'risk_reward_ratio': selected_rr_ratio,
        'tool_breakdown': {
            'buy_signals': buy_signals,
            'sell_signals': sell_signals,
            'hold_signals': hold_signals,
            'total_direct_tools': total_direct_tools,
            'buy_count': buy_count,
            'sell_count': sell_count,
            'hold_count': hold_count,
            'agreement_percentage': agreement_percentage
        },
        'current_price': current_price  # Add for manage_trades
    }


def run_scheduled_analysis():
    """Run scheduled analysis - trades saved immediately, no batch save at end"""
    global last_analysis_start_time, last_analysis_end_time, analysis_in_progress
    
    start_time = datetime.now(timezone.utc)
    print(f"\n{'*'*60}")
    print(f"🔄 ANALYSIS STARTED at {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'*'*60}")
    
    analysis_in_progress = True
    last_analysis_start_time = start_time
    update_analysis_state()
    print(f"✅ Analysis state marked as IN PROGRESS in database")
    
    try:
        config = load_analysis_config()
        combined_settings = load_combined_settings()
        symbols = config['symbols']
        interval = config['interval']
        candle_limit = config['candle_limit']
        selected_tools = config['selected_tools']
        enable_buy = config.get('enable_buy', True)
        enable_sell = config.get('enable_sell', True)

        print(f"📊 Config: {len(symbols)} symbols, {interval} interval, {candle_limit} candles")
        print(f"🛠️  Selected tools: {', '.join(selected_tools)}")
        print(f"{'*'*60}\n")

        # Load trade data from database
        active_trades, trade_history = load_trade_data()
        session_data = {'active_trades': active_trades, 'trade_history': trade_history}

        tool_results = {}
        
        # Run analyses for all selected tools (EXCLUDING COMBINED)
        analysis_tools = [tool for tool in selected_tools if tool != 'combined']
        
        for tool in analysis_tools:
            print(f"🔍 Running {tool.upper()} analysis...")
            tool_analyses = {}
            for symbol in symbols:
                try:
                    result = run_analysis_for_tool(tool, [symbol], interval, candle_limit, config)
                    if result:
                        tool_analyses.update(result)
                except Exception as e:
                    print(f"⚠️ Skipping {symbol} for {tool} - Error: {e}")
                    continue
            
            if tool_analyses:
                # Manage trades - each trade saved immediately in manage_trades()
                if tool != 'combined':
                    manage_trades(tool, tool_analyses, session_data, interval, enable_buy=enable_buy, enable_sell=enable_sell)
                    tool_results[tool] = tool_analyses
                    # NO SAVE HERE - trades already saved immediately in manage_trades()
                    print(f"✅ {tool.upper()} analysis completed for {len(tool_analyses)} symbols\n")
        
        # Handle combined analysis
        if 'combined' in selected_tools:
            print(f"{'='*60}")
            print("🔗 GENERATING COMBINED SIGNALS...")
            print(f"{'='*60}")
            combined_analyses = {}
            
            for symbol in symbols:
                try:
                    active_trade_count = 0
                    active_tool_names = []
                    
                    for tool in analysis_tools:
                        if tool == 'gann':
                            continue
                        
                        if tool == 'elliott':
                            for degree in ['Minor', 'Intermediate', 'Major']:
                                trade_key = f"{symbol}_{degree}"
                                if trade_key in session_data['active_trades'].get(tool, {}):
                                    active_trade_count += 1
                                    active_tool_names.append(f"{tool}({degree})")
                                    break
                        else:
                            trade_key = symbol
                            if trade_key in session_data['active_trades'].get(tool, {}):
                                active_trade_count += 1
                                active_tool_names.append(tool)
                    
                    min_tool_agreement = combined_settings.get('min_tool_agreement', 2)
                    
                    print(f"\n📊 {symbol}:")
                    print(f"   Active trades: {active_trade_count}/{len(analysis_tools)-1} tools")
                    print(f"   Required minimum: {min_tool_agreement} tools")
                    if active_tool_names:
                        print(f"   Tools with active trades: {', '.join(active_tool_names)}")
                    
                    if active_trade_count >= min_tool_agreement:
                        symbol_tool_signals = {tool: data[symbol] for tool, data in tool_results.items() if symbol in data}
                        
                        if symbol_tool_signals:
                            current_price = next(iter(symbol_tool_signals.values()))['current_price']
                            combined_signal = generate_combined_signal(
                                symbol_tool_signals, current_price, symbol, interval,
                                combined_settings['tool_weights'], combined_settings
                            )
                            
                            combined_signal['active_trade_count'] = active_trade_count
                            combined_signal['reasons'].insert(0, f"✅ {active_trade_count} tools have active trades for {symbol}")
                            
                            combined_analyses[symbol] = combined_signal
                            print(f"   ✅ Combined signal GENERATED: {combined_signal['action']}")
                    else:
                        print(f"   ❌ Combined signal SKIPPED: Only {active_trade_count}/{min_tool_agreement} tools have active trades")
                        
                except Exception as e:
                    print(f"⚠️ Skipping {symbol} for Combined - Error: {e}")
                    continue

            # Manage combined trades - saved immediately in manage_trades()
            if combined_analyses:
                print(f"\n{'='*60}")
                print(f"📊 Managing {len(combined_analyses)} combined trade(s)...")
                print(f"{'='*60}")
                manage_trades('combined', combined_analyses, session_data, interval, enable_buy=enable_buy, enable_sell=enable_sell)
                # NO SAVE HERE - trades already saved immediately
            else:
                print(f"\n⚠️ No combined trades generated - insufficient tool agreement")
            
            print(f"{'='*60}\n")
        
        # Update analysis completion state
        end_time = datetime.now(timezone.utc)
        duration = (end_time - start_time).total_seconds()
        
        last_analysis_end_time = end_time
        analysis_in_progress = False
        update_analysis_state()
        
        print(f"\n{'*'*60}")
        print(f"✅ ANALYSIS COMPLETED SUCCESSFULLY")
        print(f"{'*'*60}")
        print(f"⏱️  Duration: {duration:.1f} seconds")
        print(f"🕐 Completed at: {end_time.strftime('%H:%M:%S')}")
        print(f"✅ Analysis state marked as FREE in database")
        print(f"{'*'*60}\n")
        
    except Exception as e:
        end_time = datetime.now(timezone.utc)
        duration = (end_time - start_time).total_seconds()
        
        analysis_in_progress = False
        last_analysis_end_time = end_time
        update_analysis_state()
        
        print(f"\n{'*'*60}")
        print(f"❌ ANALYSIS FAILED")
        print(f"{'*'*60}")
        print(f"⏱️  Failed after: {duration:.1f} seconds")
        print(f"❌ Error: {e}")
        print(f"{'*'*60}\n")
        
        import traceback
        traceback.print_exc()
        raise
    finally:
        analysis_in_progress = False
        update_analysis_state()

@scheduler.task('interval', id='auto_analysis', minutes=5, max_instances=1)
def scheduled_task():
    """
    MODIFIED: Analysis task that ONLY opens trades, never closes them
    """
    global analysis_in_progress, last_analysis_start_time, last_analysis_end_time
    
    current_time = datetime.now(timezone.utc)
    
    # Reload state from database
    analysis_state = load_analysis_state()
    db_analysis_in_progress = analysis_state.get('analysis_in_progress', False)
    last_analysis_end_time_loaded = analysis_state.get('last_analysis_end_time')
    
    # Parse datetime if string
    if last_analysis_end_time_loaded and isinstance(last_analysis_end_time_loaded, str):
        try:
            last_analysis_end_time_loaded = datetime.strptime(
                last_analysis_end_time_loaded, '%Y-%m-%d %H:%M:%S.%f%z'
            )
        except ValueError:
            last_analysis_end_time_loaded = None
    
    analysis_in_progress = db_analysis_in_progress
    
    print(f"\n{'='*60}")
    print(f"📅 Scheduled Task at {current_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🔍 Status: {'IN PROGRESS' if db_analysis_in_progress else 'FREE'}")
    print(f"{'='*60}")
    
    # Skip if analysis already running
    if db_analysis_in_progress:
        print(f"⏭️ SKIPPING - previous analysis still running\n")
        return
    
    # Skip if no manual run yet
    if last_analysis_end_time_loaded is None:
        print(f"⏭️ SKIPPING - waiting for first manual run\n")
        return
    
    # Try to acquire analysis lock
    if not analysis_lock.acquire(blocking=False):
        print("⏭️ SKIPPING - lock already held\n")
        return
    
    try:
        print(f"🔓 Lock ACQUIRED\n")
        run_scheduled_analysis()
    except Exception as e:
        print(f"\n❌ CRITICAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        analysis_in_progress = False
        update_analysis_state()
    finally:
        analysis_lock.release()
        print(f"\n🔒 Lock RELEASED\n")

# Add global flag to prevent concurrent monitor runs
monitor_lock_flag = False
monitor_last_run = None

@scheduler.task('interval', id='monitor_active_trades', seconds=15, max_instances=1)  # Changed from 10 to 15 seconds
def monitor_active_trades():
    """
    FIXED: Monitors trades with strict locking to prevent duplicate closures
    - Increased interval to 15 seconds
    - Global lock flag prevents concurrent runs
    - Immediate database saves after each closure
    - Cooldown period after closing trades
    """
    global monitor_lock_flag, monitor_last_run
    
    # Check if another monitor is already running
    if monitor_lock_flag:
        print(f"⏭️ MONITOR: Already running, skipping this cycle")
        return
    
    # Check cooldown period (wait at least 12 seconds after last run)
    current_time = datetime.now(timezone.utc)
    if monitor_last_run:
        time_since_last = (current_time - monitor_last_run).total_seconds()
        if time_since_last < 12:
            print(f"⏭️ MONITOR: Cooldown period ({12 - time_since_last:.1f}s remaining)")
            return
    
    try:
        # Set lock flag
        monitor_lock_flag = True
        monitor_last_run = current_time
        
        # Use dedicated lock for trade management
        with safe_trade_closure():
            # Load fresh data from database
            active_trades = load_active_trades()
            trade_history = load_trade_history()
            
            if not active_trades or all(len(trades) == 0 for trades in active_trades.values()):
                return
            
            # Collect ALL unique symbols from active trades
            symbols = set()
            for tool in active_trades:
                for trade_key, trade in active_trades[tool].items():
                    symbols.add(trade['symbol'])
            
            if not symbols:
                return
            
            print(f"\n{'='*60}")
            print(f"🔍 MONITOR: Checking {len(symbols)} symbols with active trades")
            print(f"📊 Total active trades: {sum(len(trades) for trades in active_trades.values())}")
            print(f"{'='*60}")
            
            # Fetch current prices for ALL symbols with active trades
            from utils import fetch_current_price
            current_prices = {}
            for symbol in symbols:
                try:
                    current_prices[symbol] = fetch_current_price(symbol)
                    print(f"   {symbol}: ${current_prices[symbol]:.4f}")
                except Exception as e:
                    print(f"⚠️ Error fetching price for {symbol}: {e}")
                    continue
            
            trades_closed = 0
            trades_checked = 0
            
            # Create a snapshot of trades to check (prevents modification during iteration)
            trades_snapshot = []
            for tool in active_trades.keys():
                for trade_key, trade in active_trades[tool].items():
                    trades_snapshot.append({
                        'tool': tool,
                        'trade_key': trade_key,
                        'symbol': trade['symbol'],
                        'action': trade.get('action'),
                        'entry_price': trade.get('entry_price'),
                        'stop_loss': trade.get('stop_loss'),
                        'take_profit': trade.get('take_profit')
                    })
            
            print(f"📋 Processing {len(trades_snapshot)} trades...")
            
            # Process each trade with fresh data reloads
            for trade_snapshot in trades_snapshot:
                trades_checked += 1
                
                tool = trade_snapshot['tool']
                trade_key = trade_snapshot['trade_key']
                symbol = trade_snapshot['symbol']
                
                # CRITICAL: Reload fresh data before EACH trade
                active_trades = load_active_trades()
                trade_history = load_trade_history()
                
                # Verify trade still exists (not already closed)
                if tool not in active_trades or trade_key not in active_trades[tool]:
                    print(f"⏭️ [{trades_checked}/{len(trades_snapshot)}] {tool}:{trade_key} - Already closed, skipping")
                    continue
                
                # Get full trade data from fresh load
                trade = active_trades[tool][trade_key]
                
                if symbol not in current_prices:
                    print(f"⚠️ [{trades_checked}/{len(trades_snapshot)}] No price for {symbol}, skipping")
                    continue
                
                current_price = current_prices[symbol]
                action = trade.get('action')
                stop_loss = trade.get('stop_loss')
                take_profit = trade.get('take_profit')
                entry_price = trade.get('entry_price')
                
                # Validate trade data
                if None in [action, stop_loss, take_profit, entry_price]:
                    print(f"⚠️ [{trades_checked}/{len(trades_snapshot)}] Invalid data for {tool}:{trade_key}")
                    continue
                
                # Check if SL or TP hit
                hit_sl = False
                hit_tp = False
                
                try:
                    if action == "BUY":
                        hit_sl = current_price <= stop_loss
                        hit_tp = current_price >= take_profit
                    elif action == "SELL":
                        hit_sl = current_price >= stop_loss
                        hit_tp = current_price <= take_profit
                    else:
                        print(f"⚠️ [{trades_checked}/{len(trades_snapshot)}] Invalid action '{action}'")
                        continue
                    
                    # Log check status
                    if not (hit_sl or hit_tp):
                        status = f"Open (${current_price:.4f} between SL:${stop_loss:.4f} - TP:${take_profit:.4f})"
                        print(f"✅ [{trades_checked}/{len(trades_snapshot)}] {tool}:{trade_key} - {status}")
                        continue
                    
                    # SL or TP hit - proceed with closure
                    reason = "TP Hit" if hit_tp else "SL Hit"
                    print(f"\n{'*'*60}")
                    print(f"🎯 [{trades_checked}/{len(trades_snapshot)}] CLOSING TRADE: {tool}:{trade_key}")
                    print(f"   Symbol: {symbol}")
                    print(f"   Reason: {reason}")
                    print(f"   Entry: ${entry_price:.4f} → Current: ${current_price:.4f}")
                    print(f"{'*'*60}")
                    
                    # Create session_data with FRESH data
                    session_data = {
                        'active_trades': active_trades,
                        'trade_history': trade_history
                    }
                    
                    # Close the trade
                    success, closed_trade = close_single_trade(
                        tool, trade_key, trade, current_price, 
                        session_data, reason=reason
                    )
                    
                    if success:
                        # CRITICAL: Save to database IMMEDIATELY
                        try:
                            print(f"💾 Saving to database...")
                            save_active_trades(session_data['active_trades'])
                            save_trade_history(session_data['trade_history'])
                            
                            # Small delay to ensure database write completes
                            time.sleep(0.5)
                            
                            # Verify removal from database
                            verify_active = load_active_trades()
                            if tool in verify_active and trade_key in verify_active.get(tool, {}):
                                print(f"❌ ERROR: {tool}:{trade_key} STILL IN DATABASE!")
                                # Force remove it
                                del verify_active[tool][trade_key]
                                if not verify_active[tool]:
                                    del verify_active[tool]
                                save_active_trades(verify_active)
                                print(f"🔧 Force removed {tool}:{trade_key}")
                            else:
                                print(f"✅ Verified: {tool}:{trade_key} removed from database")
                            
                            trades_closed += 1
                            print(f"{'*'*60}\n")
                            
                            # Add cooldown after closing trade
                            time.sleep(1)
                            
                        except Exception as save_error:
                            print(f"❌ CRITICAL: Failed to save closure: {save_error}")
                            import traceback
                            traceback.print_exc()
                    else:
                        print(f"❌ Failed to close {tool}:{trade_key}")
                        
                except (TypeError, ValueError) as e:
                    print(f"⚠️ [{trades_checked}/{len(trades_snapshot)}] Price comparison error: {e}")
                    continue
            
            # Final summary
            if trades_closed > 0:
                # Get final count from database
                final_active_trades = load_active_trades()
                remaining_trades = sum(len(trades) for trades in final_active_trades.values())
                
                print(f"\n{'='*60}")
                print(f"✅ MONITOR COMPLETE")
                print(f"   Checked: {trades_checked} trades")
                print(f"   Closed: {trades_closed} trades")
                print(f"   Remaining: {remaining_trades} active trades")
                
                # Log remaining active trades
                if remaining_trades > 0:
                    print(f"\n📋 Active Trades Still Open:")
                    for tool, trades in final_active_trades.items():
                        for tk, t in trades.items():
                            print(f"   {tool}:{tk} - {t['symbol']} @ ${t['entry_price']:.4f}")
                
                print(f"{'='*60}\n")
            else:
                print(f"\n✅ MONITOR: Checked {trades_checked} trades - All within SL/TP bounds\n")
                
    except Exception as e:
        print(f"❌ Error in monitor_active_trades: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Always release lock flag
        monitor_lock_flag = False
        print(f"🔓 Monitor lock released\n")


# Initialize session data
def init_session():
    # Load trade data from database
    active_trades, trade_history = load_trade_data()
    
    # Initialize session with loaded data
    if 'active_trades' not in session:
        session['active_trades'] = active_trades
    if 'trade_history' not in session:
        session['trade_history'] = trade_history
    
    # Load combined settings
    combined_settings = load_combined_settings()
    if 'combined_settings' not in session:
        session['combined_settings'] = combined_settings
    
    # Load and initialize budget
    budget = load_budget()
    session['budget'] = budget

@app.route('/', methods=['GET', 'POST'])
def index():
    init_session()
    
    popular_symbols = ["BTCUSDT", "ZKCUSDT", "DEGOUSDT", "BELUSDT", "ETHUSDT", "BNBUSDT", "ADAUSDT", "XRPUSDT", "SOLUSDT", "DOTUSDT", "DOGEUSDT", 
                       "LTCUSDT", "LINKUSDT", "AVAXUSDT", "UNIUSDT", "ATOMUSDT"]
    intervals = ["1m", "3m", "5m", "15m", "1h", "2h", "3h", "4h", "1d"]
    
    fibonacci_analyses = []
    elliott_analyses = {}
    ichimoku_analyses = {}
    wyckoff_analyses = {}
    gann_analyses = {}
    combined_analyses = {}
    
    combined_settings = session.get('combined_settings', load_combined_settings())
    saved_config = load_analysis_config()
    config = saved_config.copy()
    budget = load_budget()
    
    if request.method == 'POST':
        global analysis_in_progress, last_analysis_end_time
        
        # Don't create backup if analysis already in progress
        if not analysis_in_progress:
            analysis_in_progress = True
            update_analysis_state()
        
        try:
            with analysis_lock:
                # Handle combined settings update
                if 'confidence_threshold' in request.form:
                    combined_settings['confidence_threshold'] = float(request.form.get('confidence_threshold', 0.6))
                    combined_settings['min_tool_agreement'] = int(request.form.get('min_tool_agreement', 2))
                    combined_settings['risk_reward_ratio'] = request.form.get('risk_reward_ratio', '1:2')
                    for tool in ['fibonacci', 'elliott', 'ichimoku', 'wyckoff', 'gann']:
                        weight_key = f"{tool}_weight"
                        if weight_key in request.form:
                            combined_settings['tool_weights'][tool] = float(request.form.get(weight_key, 1.0))
                    
                    session['combined_settings'] = combined_settings
                    save_combined_settings(combined_settings)
                
                # Get form inputs
                selected_tools = request.form.getlist('tools') or ['fibonacci', 'elliott', 'ichimoku', 'wyckoff', 'gann']
                symbols = request.form.getlist('symbols')
                custom_symbols_input = request.form.get('custom_symbols', '').upper()
                custom_symbols = [s.strip() for s in custom_symbols_input.split(',') if s.strip()]
                for custom_symbol in custom_symbols:
                    if custom_symbol and custom_symbol not in symbols:
                        symbols.append(custom_symbol)
                
                if not symbols:
                    symbols = ['BTCUSDT']
                
                interval = request.form.get('interval', '5m')
                candle_limit = int(request.form.get('candle_limit', 1000))

                # Update config
                config.update({
                    'selected_tools': selected_tools,
                    'symbols': symbols,
                    'custom_symbols': custom_symbols_input,
                    'interval': interval,
                    'candle_limit': candle_limit,
                    'fib_window': int(request.form.get('fib_window', 50)),
                    'fib_threshold': float(request.form.get('fib_threshold', 0.2)) / 100,
                    'elliott_thresholds': {
                        'Minor': float(request.form.get('minor_threshold', 0.5)) / 100,
                        'Intermediate': float(request.form.get('intermediate_threshold', 2.0)) / 100,
                        'Major': float(request.form.get('major_threshold', 5.0)) / 100
                    },
                    'elliott_degrees': request.form.getlist('elliott_degrees') or ['Minor', 'Intermediate', 'Major'],
                    'use_smoothing': 'use_smoothing' in request.form,
                    'smooth_period': int(request.form.get('smooth_period', 5)),
                    'show_ema': 'show_ema' in request.form,
                    'show_bb': 'show_bb' in request.form,
                    'show_volume': 'show_volume' in request.form,
                    'show_rsi': 'show_rsi' in request.form,
                    'show_macd': 'show_macd' in request.form,
                    'show_ema_ichimoku': 'show_ema_ichimoku' in request.form,
                    'show_bb_ichimoku': 'show_bb_ichimoku' in request.form,
                    'show_volume_ichimoku': 'show_volume_ichimoku' in request.form,
                    'show_rsi_ichimoku': 'show_rsi_ichimoku' in request.form,
                    'show_macd_ichimoku': 'show_macd_ichimoku' in request.form,
                    'show_ema_wyckoff': 'show_ema_wyckoff' in request.form,
                    'show_bb_wyckoff': 'show_bb_wyckoff' in request.form,
                    'show_volume_wyckoff': 'show_volume_wyckoff' in request.form,
                    'show_rsi_wyckoff': 'show_rsi_wyckoff' in request.form,
                    'show_macd_wyckoff': 'show_macd_wyckoff' in request.form,
                    'gann_subtools': request.form.getlist('gann_subtools') or ['Gann Fan', 'Gann Square', 'Gann Box', 'Gann Square Fixed'],
                    'pivot_choice': request.form.get('pivot_choice', 'Auto (based on trend)'),
                    'show_ema_gann': 'show_ema_gann' in request.form,
                    'show_bb_gann': 'show_bb_gann' in request.form,
                    'show_volume_gann': 'show_volume_gann' in request.form,
                    'show_rsi_gann': 'show_rsi_gann' in request.form,
                    'show_macd_gann': 'show_macd_gann' in request.form,
                    'enable_buy': 'enable_buy' in request.form,
                    'enable_sell': 'enable_sell' in request.form
                })
                save_analysis_config(config)
                
                # Run analyses for all selected tools
                tool_results = {}
                
                if 'fibonacci' in selected_tools:
                    print(f"\n{'='*60}")
                    print(f"🔍 Running FIBONACCI analysis...")
                    print(f"{'='*60}")
                    fibonacci_analyses_dict = run_analysis_for_tool('fibonacci', symbols, interval, candle_limit, config)
                    manage_trades('fibonacci', fibonacci_analyses_dict, session, interval, enable_buy=config['enable_buy'], enable_sell=config['enable_sell'])
                    tool_results['fibonacci'] = fibonacci_analyses_dict
                    # Trades saved immediately in manage_trades() - NO save_trade_data() here
                    print(f"✅ Fibonacci analysis completed\n")
                
                if 'elliott' in selected_tools:
                    print(f"\n{'='*60}")
                    print(f"🔍 Running ELLIOTT analysis...")
                    print(f"{'='*60}")
                    elliott_analyses = run_analysis_for_tool('elliott', symbols, interval, candle_limit, config)
                    manage_trades('elliott', elliott_analyses, session, interval, enable_buy=config['enable_buy'], enable_sell=config['enable_sell'])
                    tool_results['elliott'] = elliott_analyses
                    # Trades saved immediately in manage_trades() - NO save_trade_data() here
                    print(f"✅ Elliott analysis completed\n")
                
                if 'ichimoku' in selected_tools:
                    print(f"\n{'='*60}")
                    print(f"🔍 Running ICHIMOKU analysis...")
                    print(f"{'='*60}")
                    ichimoku_analyses = run_analysis_for_tool('ichimoku', symbols, interval, candle_limit, config)
                    manage_trades('ichimoku', ichimoku_analyses, session, interval, enable_buy=config['enable_buy'], enable_sell=config['enable_sell'])
                    tool_results['ichimoku'] = ichimoku_analyses
                    # Trades saved immediately in manage_trades() - NO save_trade_data() here
                    print(f"✅ Ichimoku analysis completed\n")
                
                if 'wyckoff' in selected_tools:
                    print(f"\n{'='*60}")
                    print(f"🔍 Running WYCKOFF analysis...")
                    print(f"{'='*60}")
                    wyckoff_analyses = run_analysis_for_tool('wyckoff', symbols, interval, candle_limit, config)
                    manage_trades('wyckoff', wyckoff_analyses, session, interval, enable_buy=config['enable_buy'], enable_sell=config['enable_sell'])
                    tool_results['wyckoff'] = wyckoff_analyses
                    # Trades saved immediately in manage_trades() - NO save_trade_data() here
                    print(f"✅ Wyckoff analysis completed\n")
                
                if 'gann' in selected_tools:
                    print(f"\n{'='*60}")
                    print(f"🔍 Running GANN analysis...")
                    print(f"{'='*60}")
                    gann_analyses = run_analysis_for_tool('gann', symbols, interval, candle_limit, config)
                    tool_results['gann'] = gann_analyses
                    print(f"✅ Gann analysis completed\n")
                
                # Generate combined analysis
                print(f"\n{'='*60}")
                print(f"🔗 GENERATING COMBINED SIGNALS...")
                print(f"{'='*60}")
                
                combined_analyses = {}
                for symbol in symbols:
                    symbol_tool_signals = {tool: data[symbol] for tool, data in tool_results.items() if symbol in data}
                    if symbol_tool_signals:
                        current_price = next(iter(symbol_tool_signals.values()))['current_price']
                        combined_signal = generate_combined_signal(
                            symbol_tool_signals, current_price, symbol, interval,
                            combined_settings['tool_weights'], combined_settings
                        )
                        combined_analyses[symbol] = combined_signal
                        print(f"✅ Combined signal generated for {symbol}: {combined_signal['action']}")
                
                # Manage combined trades
                if combined_analyses:
                    print(f"\n{'='*60}")
                    print(f"📊 Managing {len(combined_analyses)} combined trade(s)...")
                    print(f"{'='*60}")
                    manage_trades('combined', combined_analyses, session, interval, enable_buy=config['enable_buy'], enable_sell=config['enable_sell'])
                    # Trades saved immediately in manage_trades() - NO save_trade_data() here
                    print(f"✅ Combined trades managed\n")
                
                # Update analysis state
                analysis_in_progress = False
                last_analysis_end_time = datetime.now(timezone.utc)
                update_analysis_state()
                print(f"✅ Analysis state marked as FREE")
                
                # Reload budget after trades
                budget = load_budget()
                session['budget'] = budget
                session.modified = True
                
                # Set results for template
                fibonacci_analyses = list(tool_results.get('fibonacci', {}).values())
                elliott_analyses = tool_results.get('elliott', {})
                ichimoku_analyses = tool_results.get('ichimoku', {})
                wyckoff_analyses = tool_results.get('wyckoff', {})
                gann_analyses = tool_results.get('gann', {})
                combined_analyses = combined_analyses
                
                print(f"\n{'*'*60}")
                print(f"✅ MANUAL ANALYSIS COMPLETED SUCCESSFULLY")
                print(f"{'*'*60}\n")
                
        except Exception as e:
            analysis_in_progress = False
            update_analysis_state()
            print(f"❌ Manual analysis failed: {e}")
            import traceback
            traceback.print_exc()
        finally:
            analysis_in_progress = False
            update_analysis_state()
            print(f"✅ Finally: Manual analysis state ensured as FREE")
    
    # For GET requests, use saved config
    else:
        config = saved_config
        symbols = config.get('symbols', ['BTCUSDT'])
        selected_symbols = symbols
        selected_tools = config.get('selected_tools', ['fibonacci', 'elliott', 'ichimoku', 'wyckoff', 'gann'])
    
    # Load trade data from database
    active_trades, trade_history = load_trade_data()
    session['active_trades'] = active_trades
    session['trade_history'] = trade_history
    
    # Load complete trade history for statistics
    complete_trade_history = load_complete_trade_history()
    
    # Calculate wins/losses
    fibonacci_wins = sum(1 for t in complete_trade_history.get('fibonacci', []) if t.get('outcome') == 'win')
    fibonacci_losses = len(complete_trade_history.get('fibonacci', [])) - fibonacci_wins
    elliott_wins = sum(1 for t in complete_trade_history.get('elliott', []) if t.get('outcome') == 'win')
    elliott_losses = len(complete_trade_history.get('elliott', [])) - elliott_wins
    ichimoku_wins = sum(1 for t in complete_trade_history.get('ichimoku', []) if t.get('outcome') == 'win')
    ichimoku_losses = len(complete_trade_history.get('ichimoku', [])) - ichimoku_wins
    wyckoff_wins = sum(1 for t in complete_trade_history.get('wyckoff', []) if t.get('outcome') == 'win')
    wyckoff_losses = len(complete_trade_history.get('wyckoff', [])) - wyckoff_wins
    combined_wins = sum(1 for t in complete_trade_history.get('combined', []) if t.get('outcome') == 'win')
    combined_losses = len(complete_trade_history.get('combined', [])) - combined_wins
    
    # Update budget with total invested
    budget['total_invested'] = calculate_total_invested()
    save_budget(budget)
    
    # Get selected symbols for template
    selected_symbols = request.form.getlist('symbols') if request.method == 'POST' else config.get('symbols', [])
    
    return render_template('index.html', 
                         fibonacci_analyses=fibonacci_analyses,
                         elliott_analyses=elliott_analyses,
                         ichimoku_analyses=ichimoku_analyses,
                         wyckoff_analyses=wyckoff_analyses,
                         gann_analyses=gann_analyses,
                         combined_analyses=combined_analyses,
                         active_trades=active_trades,
                         trade_history=complete_trade_history,
                         fibonacci_wins=fibonacci_wins,
                         fibonacci_losses=fibonacci_losses,
                         elliott_wins=elliott_wins,
                         elliott_losses=elliott_losses,
                         ichimoku_wins=ichimoku_wins,
                         ichimoku_losses=ichimoku_losses,
                         wyckoff_wins=wyckoff_wins,
                         wyckoff_losses=wyckoff_losses,
                         combined_wins=combined_wins,
                         combined_losses=combined_losses,
                         popular_symbols=popular_symbols,
                         intervals=intervals,
                         combined_settings=combined_settings,
                         budget=budget,
                         config=config,
                         selected_symbols=selected_symbols,
                         zip=zip)

@app.route('/refresh_price/<symbol>')
def refresh_price(symbol):
    from utils import fetch_current_price
    try:
        price = fetch_current_price(symbol)
        return jsonify({'price': price})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/refresh_data', methods=['POST'])
def refresh_data():
    """Refresh ALL data from database without running analysis"""
    try:
        # Load all current data from database
        budget = load_budget()
        active_trades = load_active_trades()
        trade_history = load_complete_trade_history()
        
        # Update session with latest data
        session['budget'] = budget
        session['active_trades'] = active_trades
        session['trade_history'] = trade_history
        session.modified = True
        
        # Calculate statistics
        fibonacci_wins = sum(1 for t in trade_history.get('fibonacci', []) if t.get('outcome') == 'win')
        fibonacci_losses = len(trade_history.get('fibonacci', [])) - fibonacci_wins
        elliott_wins = sum(1 for t in trade_history.get('elliott', []) if t.get('outcome') == 'win')
        elliott_losses = len(trade_history.get('elliott', [])) - elliott_wins
        ichimoku_wins = sum(1 for t in trade_history.get('ichimoku', []) if t.get('outcome') == 'win')
        ichimoku_losses = len(trade_history.get('ichimoku', [])) - ichimoku_wins
        wyckoff_wins = sum(1 for t in trade_history.get('wyckoff', []) if t.get('outcome') == 'win')
        wyckoff_losses = len(trade_history.get('wyckoff', [])) - wyckoff_wins
        combined_wins = sum(1 for t in trade_history.get('combined', []) if t.get('outcome') == 'win')
        combined_losses = len(trade_history.get('combined', [])) - combined_wins
        
        # Calculate profit/loss summary
        total_profit_loss = 0.0
        total_gross_profit = 0.0
        total_fees = 0.0
        total_trades = 0
        
        for tool in trade_history:
            for trade in trade_history[tool]:
                total_profit_loss += trade.get('net_profit_usd', 0.0)
                total_gross_profit += trade.get('gross_profit_usd', 0.0)
                total_fees += trade.get('total_fees', 0.0)
                total_trades += 1
        
        win_rate = (combined_wins + fibonacci_wins + elliott_wins + ichimoku_wins + wyckoff_wins) / total_trades * 100 if total_trades > 0 else 0
        
        return jsonify({
            'status': 'success',
            'budget': budget,
            'active_trades': active_trades,
            'trade_history': trade_history,
            'statistics': {
                'fibonacci_wins': fibonacci_wins,
                'fibonacci_losses': fibonacci_losses,
                'elliott_wins': elliott_wins,
                'elliott_losses': elliott_losses,
                'ichimoku_wins': ichimoku_wins,
                'ichimoku_losses': ichimoku_losses,
                'wyckoff_wins': wyckoff_wins,
                'wyckoff_losses': wyckoff_losses,
                'combined_wins': combined_wins,
                'combined_losses': combined_losses,
                'total_profit_loss': total_profit_loss,
                'total_gross_profit': total_gross_profit,
                'total_fees': total_fees,
                'total_trades': total_trades,
                'win_rate': win_rate,
                'total_invested': calculate_total_invested()
            }
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500
    

def close_single_trade(tool, trade_key, trade, current_price, session_data, reason=""):
    """
    CORRECTED: Enhanced with better validation and error handling
    """
    try:
        # Validate input parameters
        if not trade or not isinstance(trade, dict):
            print(f"⚠️ Invalid trade data for {tool}:{trade_key}")
            return False, None
            
        action = trade.get('action')
        stop_loss = trade.get('stop_loss')
        take_profit = trade.get('take_profit')
        entry_price = trade.get('entry_price')
        net_investment = trade.get('net_investment')
        entry_fee = trade.get('entry_fee', 0.0)
        invested_amount = trade.get('invested_amount', FIXED_TRADE_AMOUNT)
        
        # Enhanced validation
        if action not in ['BUY', 'SELL']:
            print(f"⚠️ Invalid action for {tool}:{trade_key}: {action}")
            return False, None
            
        if None in [stop_loss, take_profit, entry_price, net_investment]:
            print(f"⚠️ Missing critical data for {tool}:{trade_key}")
            print(f"   SL: {stop_loss}, TP: {take_profit}, Entry: {entry_price}, Net: {net_investment}")
            return False, None
        
        # Verify trade still exists in session data
        if tool not in session_data['active_trades'] or trade_key not in session_data['active_trades'][tool]:
            print(f"⚠️ Trade {tool}:{trade_key} already closed or not found")
            return False, None
        
        # Calculate position and fees
        position_size = net_investment / entry_price
        position_value = position_size * current_price
        closing_fee, _ = calculate_trade_costs(position_value, is_opening=False)
        
        # Calculate profit/loss
        if action == "BUY":
            gross_profit_usd = (current_price - entry_price) * position_size
        else:  # SELL
            gross_profit_usd = (entry_price - current_price) * position_size
        
        # Net profit after ALL fees
        net_profit_usd = gross_profit_usd - entry_fee - closing_fee
        gross_profit_percent = (gross_profit_usd / invested_amount) * 100
        net_profit_percent = (net_profit_usd / invested_amount) * 100
        outcome = 'win' if net_profit_usd > 0 else 'loss'
        
        # Create closed trade record
        closed_trade = trade.copy()
        closed_trade.update({
            'outcome': outcome,
            'close_price': current_price,
            'profit_pct': gross_profit_percent,
            'net_profit_pct': net_profit_percent,
            'net_profit_usd': net_profit_usd,
            'gross_profit_usd': gross_profit_usd,
            'closing_fee': closing_fee,
            'total_fees': entry_fee + closing_fee,
            'close_time': datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S'),
            'interval': trade.get('interval', 'N/A'),
            'close_reason': reason
        })
        
        # Add to history
        if tool not in session_data['trade_history']:
            session_data['trade_history'][tool] = []
        session_data['trade_history'][tool].append(closed_trade)
        
        # Update budget
        try:
            update_budget(
                invested_amount,
                closing_fee,
                "return",
                net_profit_usd=net_profit_usd
            )
        except Exception as e:
            print(f"⚠️ Error updating budget for {tool}:{trade_key}: {e}")
            return False, None
        
        # Remove from active trades
        try:
            del session_data['active_trades'][tool][trade_key]
            # Also remove the tool if no more trades
            if not session_data['active_trades'][tool]:
                del session_data['active_trades'][tool]
        except KeyError:
            print(f"⚠️ Trade {tool}:{trade_key} already removed")
            return False, None
        
        # Log closure
        print(f"✅ {tool.capitalize()} trade CLOSED for {trade_key}: {outcome.upper()} ({reason})")
        print(f"   Entry: ${entry_price:.4f}, Close: ${current_price:.4f}")
        print(f"   Net P/L: ${net_profit_usd:.2f} ({net_profit_percent:.2f}%)")
        print(f"   Fees: Open ${entry_fee:.2f} + Close ${closing_fee:.2f}")
        
        return True, closed_trade
        
    except Exception as e:
        print(f"❌ Critical error closing trade {tool}:{trade_key}: {e}")
        import traceback
        traceback.print_exc()
        return False, None

@app.route('/update_budget', methods=['POST'])
def update_budget_route():
    try:
        data = request.get_json()
        new_budget = float(data['total_budget'])
        
        budget = load_budget()
        budget['total_budget'] = new_budget
        budget['remaining_budget'] = new_budget - budget['used_budget']
        budget['initial_budget'] = new_budget
        
        save_budget(budget)
        session['budget'] = budget
        
        return jsonify({'status': 'success', 'budget': budget})
    except Exception as e:
        print(f"Error updating budget: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/reset_budget', methods=['POST'])
def reset_budget():
    """Reset budget to initial state"""
    try:
        default_budget = {
            'total_budget': 5000.0,
            'used_budget': 0.0,
            'remaining_budget': 5000.0,
            'initial_budget': 5000.0,
            'total_fees': 0.0,
            'total_invested': 0.0
        }
        save_budget(default_budget)
        session['budget'] = default_budget
        return jsonify({'status': 'success', 'budget': default_budget})
    except Exception as e:
        print(f"Error resetting budget: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500
    

@app.route('/download_db', methods=['GET'])
def download_db():
    """Download all database data as JSON"""
    try:
        # Load all data from database
        budget = load_budget()
        active_trades = load_active_trades()
        trade_history = load_complete_trade_history()
        analysis_config = load_analysis_config()
        combined_settings = load_combined_settings()
        analysis_state_data = load_analysis_state()
        
        # Combine all data
        all_data = {
            'budget': budget,
            'active_trades': active_trades,
            'trade_history': trade_history,
            'analysis_config': analysis_config,
            'combined_settings': combined_settings,
            'analysis_state': analysis_state_data,
            'export_timestamp': datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # Create JSON response
        response = make_response(json.dumps(all_data, indent=2))
        response.headers['Content-Type'] = 'application/json'
        response.headers['Content-Disposition'] = f'attachment; filename=trading_data_{datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")}.json'
        
        return response
    except Exception as e:
        print(f"Error downloading database: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/delete_all_data', methods=['POST'])
def delete_all_data():
    """Delete all data from database tables"""
    try:
        conn = sqlite3.connect('trading_bot.db')
        cursor = conn.cursor()
        
        # Delete all records from each table
        cursor.execute("DELETE FROM budget")
        cursor.execute("DELETE FROM active_trades")
        cursor.execute("DELETE FROM trade_history")
        cursor.execute("DELETE FROM analysis_config")
        cursor.execute("DELETE FROM combined_settings")
        cursor.execute("DELETE FROM analysis_state")
        
        conn.commit()
        conn.close()
        
        # Reinitialize with default values
        init_db()
        
        # Clear session data
        session.clear()
        
        # Reload default data into session
        init_session()
        
        return jsonify({
            'status': 'success',
            'message': 'All data deleted successfully. Database reinitialized with defaults.'
        })
    except Exception as e:
        print(f"Error deleting data: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500


if __name__ == '__main__':
    import atexit
    # Only start scheduler if not in reloader process
    if not app.debug or os.environ.get("WERKZEUG_RUN_MAIN") == "true":
        scheduler.start()
        atexit.register(lambda: scheduler.shutdown())
    
    app.run(host='0.0.0.0', port=5000, debug=True)
