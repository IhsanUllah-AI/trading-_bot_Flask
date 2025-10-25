import sqlite3
import json
from datetime import datetime, timezone
import os

# Database file path
DB_PATH = 'trading_bot.db'

def get_db_connection():
    """Create database connection"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row  # This enables column access by name
    return conn

def init_db():
    # Check if database file exists
    if os.path.exists(DB_PATH):
        conn = get_db_connection()
        cur = conn.cursor()
        
        # Check if 'budget' table exists and has data
        cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='budget'")
        budget_table_exists = cur.fetchone()
        
        if budget_table_exists:
            # Check if budget table has any rows
            cur.execute("SELECT COUNT(*) FROM budget")
            budget_count = cur.fetchone()[0]
            if budget_count > 0:
                print("✅ Database already initialized with budget data! Skipping re-initialization.")
                cur.close()
                conn.close()
                return
        
        cur.close()
        conn.close()
    
    # Proceed with initialization if database doesn't exist or budget table is empty
    conn = get_db_connection()
    cur = conn.cursor()
    
    # Budget table
    cur.execute('''
        CREATE TABLE IF NOT EXISTS budget (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            total_budget REAL DEFAULT 5000.00,
            used_budget REAL DEFAULT 0.00,
            remaining_budget REAL DEFAULT 5000.00,
            initial_budget REAL DEFAULT 5000.00,
            total_fees REAL DEFAULT 0.00,
            total_invested REAL DEFAULT 0.00,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Active trades table
    cur.execute('''
        CREATE TABLE IF NOT EXISTS active_trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            tool TEXT NOT NULL,
            trade_key TEXT NOT NULL,
            symbol TEXT NOT NULL,
            degree TEXT,
            action TEXT NOT NULL,
            entry_price REAL NOT NULL,
            stop_loss REAL,
            take_profit REAL,
            entry_time TIMESTAMP NOT NULL,
            reason TEXT,
            interval TEXT,
            position_size REAL,
            invested_amount REAL NOT NULL,
            entry_fee REAL NOT NULL,
            net_investment REAL NOT NULL,
            signals TEXT,
            signal_descriptions TEXT,
            confidence TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(tool, trade_key)
        )
    ''')
    
    # Trade history table
    cur.execute('''
        CREATE TABLE IF NOT EXISTS trade_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            tool TEXT NOT NULL,
            trade_key TEXT NOT NULL,
            symbol TEXT NOT NULL,
            degree TEXT,
            action TEXT NOT NULL,
            entry_price REAL NOT NULL,
            stop_loss REAL,
            take_profit REAL,
            entry_time TIMESTAMP NOT NULL,
            close_price REAL NOT NULL,
            close_time TIMESTAMP NOT NULL,
            outcome TEXT NOT NULL,
            profit_pct REAL,
            net_profit_pct REAL,
            net_profit_usd REAL,
            gross_profit_usd REAL,
            closing_fee REAL,
            total_fees REAL,
            invested_amount REAL NOT NULL,
            entry_fee REAL NOT NULL,
            net_investment REAL NOT NULL,
            position_size REAL,
            interval TEXT,
            reason TEXT,
            signals TEXT,
            signal_descriptions TEXT,
            confidence TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Analysis config table
    cur.execute('''
        CREATE TABLE IF NOT EXISTS analysis_config (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            config_name TEXT DEFAULT 'default',
            config_data TEXT NOT NULL,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(config_name)
        )
    ''')
    
    # Combined settings table
    cur.execute('''
        CREATE TABLE IF NOT EXISTS combined_settings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            settings_name TEXT DEFAULT 'default',
            settings_data TEXT NOT NULL,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(settings_name)
        )
    ''')
    
    # Analysis state table
    cur.execute('''
        CREATE TABLE IF NOT EXISTS analysis_state (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            state_name TEXT DEFAULT 'current',
            last_analysis_start_time TIMESTAMP,
            last_analysis_end_time TIMESTAMP,
            analysis_in_progress BOOLEAN DEFAULT FALSE,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(state_name)
        )
    ''')
    
    # Insert default budget only if budget table is empty
    cur.execute("SELECT COUNT(*) FROM budget")
    if cur.fetchone()[0] == 0:
        cur.execute('''
            INSERT INTO budget (total_budget, used_budget, remaining_budget, initial_budget, total_fees, total_invested)
            VALUES (5000.00, 0.00, 5000.00, 5000.00, 0.00, 0.00)
        ''')
        print("✅ Inserted default budget values into empty budget table.")
    
    # Insert default analysis state
    cur.execute('''
        INSERT OR IGNORE INTO analysis_state (state_name, analysis_in_progress)
        VALUES ('current', FALSE)
    ''')
    
    conn.commit()
    cur.close()
    conn.close()
    print("✅ SQLite database initialized successfully!")
    
def load_budget():
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute('SELECT * FROM budget ORDER BY id DESC LIMIT 1')
    budget = cur.fetchone()
    cur.close()
    conn.close()
    
    if budget:
        loaded_budget = {
            'total_budget': budget['total_budget'],
            'used_budget': budget['used_budget'],
            'remaining_budget': budget['remaining_budget'],
            'initial_budget': budget['initial_budget'],
            'total_fees': budget['total_fees'],
            'total_invested': budget['total_invested']
        }
    else:
        # Return default budget if no data exists
        loaded_budget = {
            'total_budget': 5000.0,
            'used_budget': 0.0,
            'remaining_budget': 5000.0,
            'initial_budget': 5000.0,
            'total_fees': 0.0,
            'total_invested': 0.0
        }
        save_budget(loaded_budget)
        return loaded_budget
    
    # AUTO-FIX: Verify used_budget matches active trades
    from app import calculate_total_invested  # Import here to avoid circular import
    actual_invested = calculate_total_invested()
    
    if abs(loaded_budget['used_budget'] - actual_invested) > 1.0:
        print(f"🔧 Auto-fixing budget: stored={loaded_budget['used_budget']:.2f}, actual={actual_invested:.2f}")
        loaded_budget['used_budget'] = actual_invested
        loaded_budget['remaining_budget'] = loaded_budget['total_budget'] - loaded_budget['used_budget']
        save_budget(loaded_budget)
    
    return loaded_budget

def save_budget(budget_data):
    conn = get_db_connection()
    cur = conn.cursor()
    
    # Update existing budget or insert new
    cur.execute('''
        INSERT OR REPLACE INTO budget (
            id, total_budget, used_budget, remaining_budget,
            initial_budget, total_fees, total_invested, updated_at
        )
        SELECT id, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP
        FROM budget ORDER BY id DESC LIMIT 1
    ''', (
        budget_data['total_budget'], budget_data['used_budget'], budget_data['remaining_budget'],
        budget_data['initial_budget'], budget_data['total_fees'], budget_data['total_invested']
    ))
    
    # If no rows were affected (i.e., table was empty), insert new
    if cur.rowcount == 0:
        cur.execute('''
            INSERT INTO budget (
                total_budget, used_budget, remaining_budget,
                initial_budget, total_fees, total_invested, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
        ''', (
            budget_data['total_budget'], budget_data['used_budget'], budget_data['remaining_budget'],
            budget_data['initial_budget'], budget_data['total_fees'], budget_data['total_invested']
        ))
    
    conn.commit()
    cur.close()
    conn.close()

# Active trades functions
def load_active_trades():
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute('SELECT * FROM active_trades ORDER BY entry_time')
    trades = cur.fetchall()
    cur.close()
    conn.close()
    
    # Convert to the expected format
    active_trades = {'fibonacci': {}, 'elliott': {}, 'ichimoku': {}, 'wyckoff': {}, 'gann': {}, 'combined': {}}
    
    for trade in trades:
        tool = trade['tool']
        trade_key = trade['trade_key']
        
        # Convert to dictionary
        trade_dict = dict(trade)
        
        # Convert JSON strings back to lists
        if trade_dict.get('signals'):
            try:
                trade_dict['signals'] = json.loads(trade_dict['signals'])
            except:
                trade_dict['signals'] = []
        else:
            trade_dict['signals'] = []
            
        if trade_dict.get('signal_descriptions'):
            try:
                trade_dict['signal_descriptions'] = json.loads(trade_dict['signal_descriptions'])
            except:
                trade_dict['signal_descriptions'] = []
        else:
            trade_dict['signal_descriptions'] = []
        
        active_trades[tool][trade_key] = trade_dict
    
    return active_trades

def save_active_trades(active_trades):
    conn = get_db_connection()
    cur = conn.cursor()
    
    # Clear existing active trades
    cur.execute('DELETE FROM active_trades')
    
    # Insert new active trades
    for tool, trades in active_trades.items():
        for trade_key, trade in trades.items():
            cur.execute('''
                INSERT INTO active_trades 
                (tool, trade_key, symbol, degree, action, entry_price, stop_loss, take_profit,
                 entry_time, reason, interval, position_size, invested_amount, entry_fee,
                 net_investment, signals, signal_descriptions, confidence)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                tool, trade_key, trade['symbol'], trade.get('degree'),
                trade['action'], trade['entry_price'], trade.get('stop_loss'),
                trade.get('take_profit'), trade['entry_time'], trade.get('reason'),
                trade.get('interval'), trade.get('position_size'), trade['invested_amount'],
                trade['entry_fee'], trade['net_investment'],
                json.dumps(trade.get('signals', [])),
                json.dumps(trade.get('signal_descriptions', [])),
                trade.get('confidence')
            ))
    
    conn.commit()
    cur.close()
    conn.close()

# Trade history functions
def load_trade_history():
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute('SELECT * FROM trade_history ORDER BY close_time DESC')
    trades = cur.fetchall()
    cur.close()
    conn.close()
    
    # Convert to the expected format
    trade_history = {'fibonacci': [], 'elliott': [], 'ichimoku': [], 'wyckoff': [], 'gann': [], 'combined': []}
    
    for trade in trades:
        tool = trade['tool']
        
        # Convert to dictionary
        trade_dict = dict(trade)
        
        # Convert JSON strings back to lists
        if trade_dict.get('signals'):
            try:
                trade_dict['signals'] = json.loads(trade_dict['signals'])
            except:
                trade_dict['signals'] = []
        else:
            trade_dict['signals'] = []
            
        if trade_dict.get('signal_descriptions'):
            try:
                trade_dict['signal_descriptions'] = json.loads(trade_dict['signal_descriptions'])
            except:
                trade_dict['signal_descriptions'] = []
        else:
            trade_dict['signal_descriptions'] = []
        
        trade_history[tool].append(trade_dict)
    
    return trade_history

def save_trade_history(trade_history):
    conn = get_db_connection()
    cur = conn.cursor()
    
    # Note: We don't clear trade history as it's append-only
    for tool, trades in trade_history.items():
        for trade in trades:
            # Check if trade already exists (based on unique identifier)
            cur.execute('''
                SELECT id FROM trade_history 
                WHERE tool = ? AND trade_key = ? AND close_time = ?
            ''', (tool, trade['trade_key'], trade['close_time']))
            
            if not cur.fetchone():  # Only insert if it doesn't exist
                cur.execute('''
                    INSERT INTO trade_history 
                    (tool, trade_key, symbol, degree, action, entry_price, stop_loss, take_profit,
                     entry_time, close_price, close_time, outcome, profit_pct, net_profit_pct,
                     net_profit_usd, gross_profit_usd, closing_fee, total_fees, invested_amount,
                     entry_fee, net_investment, position_size, interval, reason, signals, signal_descriptions, confidence)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    tool, trade['trade_key'], trade['symbol'], trade.get('degree'),
                    trade['action'], trade['entry_price'], trade.get('stop_loss'),
                    trade.get('take_profit'), trade['entry_time'], trade['close_price'],
                    trade['close_time'], trade['outcome'], trade.get('profit_pct'),
                    trade.get('net_profit_pct'), trade.get('net_profit_usd'),
                    trade.get('gross_profit_usd'), trade.get('closing_fee'),
                    trade.get('total_fees'), trade['invested_amount'], trade['entry_fee'],
                    trade['net_investment'], trade.get('position_size'), trade.get('interval'),
                    trade.get('reason'), json.dumps(trade.get('signals', [])),
                    json.dumps(trade.get('signal_descriptions', [])), trade.get('confidence')
                ))
    
    conn.commit()
    cur.close()
    conn.close()

# Analysis config functions
def load_analysis_config():
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("SELECT config_data FROM analysis_config WHERE config_name = 'default'")
    result = cur.fetchone()
    cur.close()
    conn.close()
    
    if result:
        return json.loads(result['config_data'])
    else:
        # Return default config
        default_config = {
            'selected_tools': ['fibonacci', 'elliott', 'ichimoku', 'wyckoff', 'gann'],
            'symbols': ['BTCUSDT'],
            'interval': '5m',
            'candle_limit': 1000,
            'fib_window': 50,
            'fib_threshold': 0.002,
            'elliott_thresholds': {'Minor': 0.005, 'Intermediate': 0.020, 'Major': 0.05},
            'elliott_degrees': ['Minor', 'Intermediate', 'Major'],
            'use_smoothing': True,
            'smooth_period': 5,
            'show_ema': True,
            'show_bb': True,
            'show_volume': True,
            'show_rsi': True,
            'show_macd': True,
            'show_ema_ichimoku': True,
            'show_bb_ichimoku': True,
            'show_volume_ichimoku': True,
            'show_rsi_ichimoku': True,
            'show_macd_ichimoku': True,
            'show_ema_wyckoff': True,
            'show_bb_wyckoff': True,
            'show_volume_wyckoff': True,
            'show_rsi_wyckoff': True,
            'show_macd_wyckoff': True,
            'gann_subtools': ['Gann Fan', 'Gann Square', 'Gann Box', 'Gann Square Fixed'],
            'pivot_choice': 'Auto (based on trend)',
            'show_ema_gann': True,
            'show_bb_gann': True,
            'show_volume_gann': True,
            'show_rsi_gann': True,
            'show_macd_gann': True,
            'enable_buy': True,
            'enable_sell': False
        }
        save_analysis_config(default_config)
        return default_config

def save_analysis_config(config_data):
    conn = get_db_connection()
    cur = conn.cursor()
    
    cur.execute('''
        INSERT OR REPLACE INTO analysis_config (config_name, config_data, updated_at)
        VALUES ('default', ?, CURRENT_TIMESTAMP)
    ''', (json.dumps(config_data),))
    
    conn.commit()
    cur.close()
    conn.close()

# Combined settings functions
def load_combined_settings():
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("SELECT settings_data FROM combined_settings WHERE settings_name = 'default'")
    result = cur.fetchone()
    cur.close()
    conn.close()
    
    if result:
        return json.loads(result['settings_data'])
    else:
        default_settings = {
            'confidence_threshold': 0.6,
            'min_tool_agreement': 2,
            'risk_reward_ratio': '1:2',
            'tool_weights': {
                'fibonacci': 1.0,
                'elliott': 1.0,
                'ichimoku': 1.0,
                'wyckoff': 1.0,
                'gann': 0.5
            }
        }
        save_combined_settings(default_settings)
        return default_settings

def save_combined_settings(settings_data):
    conn = get_db_connection()
    cur = conn.cursor()
    
    cur.execute('''
        INSERT OR REPLACE INTO combined_settings (settings_name, settings_data, updated_at)
        VALUES ('default', ?, CURRENT_TIMESTAMP)
    ''', (json.dumps(settings_data),))
    
    conn.commit()
    cur.close()
    conn.close()

# Analysis state functions
def load_analysis_state():
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("SELECT * FROM analysis_state WHERE state_name = 'current'")
    result = cur.fetchone()
    cur.close()
    conn.close()
    
    if result:
        return {
            'last_analysis_start_time': result['last_analysis_start_time'],
            'last_analysis_end_time': result['last_analysis_end_time'],
            'analysis_in_progress': bool(result['analysis_in_progress'])
        }
    else:
        return {
            'last_analysis_start_time': None,
            'last_analysis_end_time': None,
            'analysis_in_progress': False
        }

def save_analysis_state(state_data):
    conn = get_db_connection()
    cur = conn.cursor()
    
    cur.execute('''
        INSERT OR REPLACE INTO analysis_state 
        (state_name, last_analysis_start_time, last_analysis_end_time, analysis_in_progress, updated_at)
        VALUES ('current', ?, ?, ?, CURRENT_TIMESTAMP)
    ''', (
        state_data.get('last_analysis_start_time'),
        state_data.get('last_analysis_end_time'),
        state_data.get('analysis_in_progress', False)
    ))
    
    conn.commit()
    cur.close()
    conn.close()