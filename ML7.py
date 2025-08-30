# ═══════════════════════════════════════════════════════════════════════════════
# 🔮 SPX PROPHET - PART 1: FOUNDATION & DATA HANDLING 📊
# ═══════════════════════════════════════════════════════════════════════════════

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import pytz
from datetime import datetime, date, time, timedelta
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# ═══════════════════════════════════════════════════════════════════════════════
# 🌎 CORE CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════
CT_TZ = pytz.timezone('America/Chicago')
UTC_TZ = pytz.UTC

# Trading parameters
RTH_START = "08:30"  # RTH start in CT
RTH_END = "14:30"    # RTH end in CT
SPX_ANCHOR_START = "17:00"  # SPX anchor window start CT
SPX_ANCHOR_END = "19:30"    # SPX anchor window end CT

# Default slopes per 30-min block
SPX_SLOPES = {
    'high': -0.2792,
    'close': -0.2792, 
    'low': -0.2792,
    'skyline': 0.268,
    'baseline': -0.235
}

STOCK_SLOPES = {
    'AAPL': 0.0155, 'MSFT': 0.0541, 'NVDA': 0.0086, 'AMZN': 0.0139,
    'GOOGL': 0.0122, 'TSLA': 0.0285, 'META': 0.0674, 'NFLX': 0.0230
}

# ═══════════════════════════════════════════════════════════════════════════════
# 🎨 STREAMLIT CONFIGURATION & STYLING
# ═══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="🔮 SPX Prophet Analytics",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced colorful styling
st.markdown("""
<style>
    .main { 
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        color: white;
    }
    .metric-container { 
        background: linear-gradient(135deg, rgba(255,255,255,0.1), rgba(255,255,255,0.05));
        padding: 1.2rem;
        border-radius: 15px;
        border: 2px solid rgba(255,255,255,0.2);
        backdrop-filter: blur(10px);
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
    }
    .metric-container:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 35px rgba(0,0,0,0.2);
    }
    .stTab { 
        background: linear-gradient(135deg, rgba(255,255,255,0.1), rgba(255,255,255,0.05));
        border-radius: 10px;
        padding: 10px;
        margin: 5px;
    }
    .success-box {
        background: linear-gradient(135deg, #00C851, #007E33);
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #00ff88;
    }
    .warning-box {
        background: linear-gradient(135deg, #ffbb33, #ff8800);
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #ffff00;
    }
    .info-box {
        background: linear-gradient(135deg, #33b5e5, #0099cc);
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #00ddff;
    }
    .stDataFrame {
        background: rgba(255,255,255,0.95);
        border-radius: 10px;
        overflow: hidden;
    }
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# 📊 DATA FETCHING FUNCTIONS (FIXED)
# ═══════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=60)
def fetch_live_data(symbol: str, start_date: date, end_date: date) -> pd.DataFrame:
    """Fetch live yfinance data with proper OHLC extraction"""
    try:
        # Add buffer days to ensure we get data
        buffer_start = start_date - timedelta(days=5)
        buffer_end = end_date + timedelta(days=2)
        
        ticker = yf.Ticker(symbol)
        
        # Fetch with prepost=True to get extended hours data
        df = ticker.history(
            start=buffer_start.strftime('%Y-%m-%d'), 
            end=buffer_end.strftime('%Y-%m-%d'),
            interval="30m", 
            prepost=True,
            auto_adjust=False,
            back_adjust=False
        )
        
        if df.empty:
            st.error(f"⚠️ No data returned for {symbol}")
            return pd.DataFrame()
        
        # Fix column names - yfinance sometimes returns MultiIndex
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0] if isinstance(col, tuple) else str(col) for col in df.columns]
        
        # Ensure we have the required columns
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            st.error(f"❌ Missing columns for {symbol}: {missing_cols}")
            return pd.DataFrame()
        
        # Convert timezone to CT
        if df.index.tz is None:
            df.index = df.index.tz_localize('US/Eastern')
        df.index = df.index.tz_convert(CT_TZ)
        
        # Filter to requested date range
        start_dt = CT_TZ.localize(datetime.combine(start_date, time(0, 0)))
        end_dt = CT_TZ.localize(datetime.combine(end_date, time(23, 59)))
        df = df.loc[start_dt:end_dt]
        
        # Data quality validation
        if not validate_ohlc_data(df):
            st.warning(f"⚠️ Data quality issues detected for {symbol}")
        
        return df
        
    except Exception as e:
        st.error(f"❌ Error fetching {symbol}: {str(e)}")
        return pd.DataFrame()

def validate_ohlc_data(df: pd.DataFrame) -> bool:
    """Validate OHLC data quality"""
    if df.empty:
        return False
    
    # Check for required columns
    if not all(col in df.columns for col in ['Open', 'High', 'Low', 'Close']):
        return False
    
    # Check for valid price relationships
    invalid_bars = (
        (df['High'] < df['Low']) | 
        (df['High'] < df['Open']) | 
        (df['High'] < df['Close']) |
        (df['Low'] > df['Open']) | 
        (df['Low'] > df['Close']) |
        (df['Close'] <= 0) |
        (df['High'] <= 0)
    )
    
    if invalid_bars.any():
        return False
    
    return True

@st.cache_data(ttl=300)
def fetch_historical_data(symbol: str, days_back: int = 30) -> pd.DataFrame:
    """Fetch historical data for analysis"""
    end_date = datetime.now(CT_TZ).date()
    start_date = end_date - timedelta(days=days_back)
    return fetch_live_data(symbol, start_date, end_date)

# ═══════════════════════════════════════════════════════════════════════════════
# ⏰ TIME HANDLING FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def rth_slots_ct(target_date: date) -> List[datetime]:
    """Generate RTH 30-min slots in CT timezone"""
    start_dt = datetime.combine(target_date, time(8, 30))
    start_ct = CT_TZ.localize(start_dt)
    
    slots = []
    current = start_ct
    end_ct = CT_TZ.localize(datetime.combine(target_date, time(14, 30)))
    
    while current <= end_ct:
        slots.append(current)
        current += timedelta(minutes=30)
        
    return slots

def format_ct_time(dt: datetime) -> str:
    """Format datetime to CT time string"""
    if dt.tzinfo is None:
        dt = CT_TZ.localize(dt)
    elif dt.tzinfo != CT_TZ:
        dt = dt.astimezone(CT_TZ)
    return dt.strftime("%H:%M")

def get_session_window(df: pd.DataFrame, start_time: str, end_time: str) -> pd.DataFrame:
    """Filter dataframe to specific session window in CT"""
    if df.empty:
        return df
    return df.between_time(start_time, end_time)

def get_daily_ohlc(df: pd.DataFrame, target_date: date) -> Dict:
    """Extract actual daily OHLC from 30-min data"""
    if df.empty:
        return {}
    
    # Filter to target date
    start_dt = CT_TZ.localize(datetime.combine(target_date, time(0, 0)))
    end_dt = CT_TZ.localize(datetime.combine(target_date, time(23, 59)))
    
    day_data = df.loc[start_dt:end_dt]
    
    if day_data.empty:
        return {}
    
    # Calculate actual OHLC for the day
    day_open = day_data.iloc[0]['Open']
    day_high = day_data['High'].max()
    day_low = day_data['Low'].min()
    day_close = day_data.iloc[-1]['Close']
    
    # Get timestamps for high and low
    high_time = day_data[day_data['High'] == day_high].index[0]
    low_time = day_data[day_data['Low'] == day_low].index[0]
    open_time = day_data.index[0]
    close_time = day_data.index[-1]
    
    return {
        'open': (day_open, open_time),
        'high': (day_high, high_time),
        'low': (day_low, low_time),
        'close': (day_close, close_time)
    }

# ═══════════════════════════════════════════════════════════════════════════════
# 📈 SWING DETECTION FUNCTIONS (FIXED)
# ═══════════════════════════════════════════════════════════════════════════════

def detect_swings_simple(df: pd.DataFrame) -> pd.DataFrame:
    """Find highest and lowest close prices in dataset"""
    if df.empty or len(df) < 2:
        return df.copy()
    
    df_swings = df.copy()
    df_swings['swing_high'] = False
    df_swings['swing_low'] = False
    
    # Find highest and lowest close prices
    if 'Close' in df_swings.columns:
        max_close_idx = df_swings['Close'].idxmax()
        min_close_idx = df_swings['Close'].idxmin()
        
        # Mark them as swings
        df_swings.loc[max_close_idx, 'swing_high'] = True
        df_swings.loc[min_close_idx, 'swing_low'] = True
    
    return df_swings

def get_anchor_points(df_swings: pd.DataFrame) -> Tuple[Optional[Tuple], Optional[Tuple]]:
    """Extract skyline (highest) and baseline (lowest) close prices"""
    skyline = None
    baseline = None
    
    if df_swings.empty or 'Close' not in df_swings.columns:
        return skyline, baseline
    
    swing_highs = df_swings[df_swings.get('swing_high', False) == True]
    swing_lows = df_swings[df_swings.get('swing_low', False) == True]
    
    if not swing_highs.empty:
        best_high = swing_highs.loc[swing_highs['Close'].idxmax()]
        skyline = (best_high['Close'], best_high.name)
    
    if not swing_lows.empty:
        best_low = swing_lows.loc[swing_lows['Close'].idxmin()]
        baseline = (best_low['Close'], best_low.name)
    
    return skyline, baseline

# ═══════════════════════════════════════════════════════════════════════════════
# 📊 PROJECTION FUNCTIONS (FIXED)
# ═══════════════════════════════════════════════════════════════════════════════

def project_anchor_line(anchor_price: float, anchor_time: datetime, 
                       slope: float, target_date: date) -> pd.DataFrame:
    """Project anchor line across RTH using slope per 30-min block"""
    rth_slots = rth_slots_ct(target_date)
    projections = []
    
    for slot_time in rth_slots:
        # Calculate 30-min blocks from anchor to slot
        time_diff = slot_time - anchor_time
        blocks = time_diff.total_seconds() / 1800  # 1800 seconds = 30 minutes
        
        projected_price = anchor_price + (slope * blocks)
        
        projections.append({
            'Time': format_ct_time(slot_time),
            'Price': round(projected_price, 2),
            'Blocks': round(blocks, 1),
            'Anchor_Price': round(anchor_price, 2),
            'Slope': slope
        })
    
    return pd.DataFrame(projections)

# ═══════════════════════════════════════════════════════════════════════════════
# 📊 INDICATORS & ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

def calculate_ema(series: pd.Series, span: int) -> pd.Series:
    """Calculate EMA for given span"""
    return series.ewm(span=span).mean()

def calculate_vwap(df: pd.DataFrame) -> pd.Series:
    """Calculate intraday VWAP"""
    if df.empty or 'Volume' not in df.columns:
        return pd.Series(index=df.index, dtype=float)
    
    typical_price = (df['High'] + df['Low'] + df['Close']) / 3
    cum_volume = df['Volume'].cumsum()
    cum_vol_price = (typical_price * df['Volume']).cumsum()
    
    # Avoid division by zero
    vwap = cum_vol_price / cum_volume
    vwap = vwap.fillna(method='ffill').fillna(typical_price)
    
    return vwap

def calculate_es_spx_offset(es_data: pd.DataFrame, spx_data: pd.DataFrame) -> float:
    """Calculate ES to SPX offset using most recent overlapping data"""
    try:
        if es_data.empty or spx_data.empty:
            return 0.0
        
        # Get the most recent common timeframe
        es_last = es_data.iloc[-1]['Close']
        spx_last = spx_data.iloc[-1]['Close']
        
        offset = spx_last - es_last
        return round(offset, 1)
        
    except Exception as e:
        st.warning(f"⚠️ Offset calculation error: {str(e)}")
        return 0.0

# ═══════════════════════════════════════════════════════════════════════════════
# 🔧 SESSION STATE INITIALIZATION
# ═══════════════════════════════════════════════════════════════════════════════

if 'theme' not in st.session_state:
    st.session_state.theme = 'Dark'

if 'spx_slopes' not in st.session_state:
    st.session_state.spx_slopes = SPX_SLOPES.copy()

if 'stock_slopes' not in st.session_state:
    st.session_state.stock_slopes = STOCK_SLOPES.copy()

if 'current_offset' not in st.session_state:
    st.session_state.current_offset = 0.0

# ═══════════════════════════════════════════════════════════════════════════════
# 🎛️ SIDEBAR CONTROLS
# ═══════════════════════════════════════════════════════════════════════════════

st.sidebar.title("🔮 SPX Prophet Analytics")
st.sidebar.markdown("---")

# Theme selector
theme = st.sidebar.radio("🎨 Theme", ["Dark", "Light"], key="ui_theme")
st.session_state.theme = theme

st.sidebar.markdown("---")

# SPX slope controls with colorful headers
st.sidebar.markdown("### 📈 SPX Slopes (per 30-min)")
st.sidebar.caption("🎯 Adjust projection slopes for each anchor type")

for slope_name, default_value in SPX_SLOPES.items():
    icon_map = {
        'high': '🔴', 'close': '🟡', 'low': '🟢', 
        'skyline': '🔥', 'baseline': '🏔️'
    }
    
    icon = icon_map.get(slope_name, '📊')
    display_name = slope_name.title()
    
    slope_value = st.sidebar.number_input(
        f"{icon} {display_name}",
        value=st.session_state.spx_slopes[slope_name],
        step=0.0001, format="%.4f",
        key=f"sb_spx_{slope_name}"
    )
    st.session_state.spx_slopes[slope_name] = slope_value

st.sidebar.markdown("---")

# Stock slope controls
st.sidebar.markdown("### 🏢 Stock Slopes (magnitude)")
st.sidebar.caption("📊 Individual stock projection parameters")

# Collapsible stock controls
with st.sidebar.expander("🔧 Stock Slope Settings", expanded=False):
    for ticker, default_slope in STOCK_SLOPES.items():
        current_slope = st.number_input(
            f"📈 {ticker}", 
            value=st.session_state.stock_slopes.get(ticker, default_slope),
            step=0.0001, format="%.4f",
            key=f"sb_stk_{ticker}"
        )
        st.session_state.stock_slopes[ticker] = current_slope

# ═══════════════════════════════════════════════════════════════════════════════
# 🏠 MAIN APP HEADER
# ═══════════════════════════════════════════════════════════════════════════════

# Hero section with colorful design
st.markdown("""
<div style="text-align: center; padding: 2rem; background: linear-gradient(135deg, rgba(255,255,255,0.1), rgba(255,255,255,0.05)); border-radius: 20px; margin: 1rem 0; border: 2px solid rgba(255,255,255,0.2);">
    <h1 style="font-size: 3rem; margin: 0; background: linear-gradient(135deg, #ff6b6b, #4ecdc4, #45b7d1, #f9ca24); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;">
        🔮 SPX Prophet Analytics 📊
    </h1>
    <p style="font-size: 1.3rem; margin: 1rem 0; opacity: 0.9;">
        🚀 Advanced Trading Analytics with Live Market Data Integration ⚡
    </p>
</div>
""", unsafe_allow_html=True)

# Live market status with colorful metrics
col1, col2, col3 = st.columns(3)

with col1:
    current_time_ct = datetime.now(CT_TZ)
    st.markdown(f"""
    <div class="metric-container">
        <h3>⏰ Current Time (CT)</h3>
        <h2>{current_time_ct.strftime("%H:%M:%S")}</h2>
        <p>{current_time_ct.strftime("%A, %B %d")}</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    market_open = current_time_ct.replace(hour=8, minute=30, second=0, microsecond=0)
    market_close = current_time_ct.replace(hour=14, minute=30, second=0, microsecond=0)
    is_rth = market_open <= current_time_ct <= market_close
    
    status_color = "#00ff88" if is_rth else "#ff6b6b"
    status_text = "🟢 MARKET OPEN" if is_rth else "🔴 MARKET CLOSED"
    
    st.markdown(f"""
    <div class="metric-container">
        <h3>📊 Market Status</h3>
        <h2 style="color: {status_color};">{status_text}</h2>
        <p>RTH: 08:30 - 14:30 CT</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="metric-container">
        <h3>🔄 ES→SPX Offset</h3>
        <h2>{st.session_state.current_offset:+.1f}</h2>
        <p>Live market differential</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# Data validation status
if st.button("🔍 Test Data Connection", key="test_connection"):
    with st.spinner("🔄 Testing market data connection..."):
        test_data = fetch_live_data("^GSPC", datetime.now().date() - timedelta(days=1), datetime.now().date())
        
        if not test_data.empty:
            st.success("✅ Market data connection successful!")
            st.info(f"📊 Retrieved {len(test_data)} data points for SPX")
        else:
            st.error("❌ Market data connection failed!")

st.markdown("---")

# ═══════════════════════════════════════════════════════════════════════════════
# ✅ PART 1 COMPLETE - FOUNDATION
# ═══════════════════════════════════════════════════════════════════════════════








# ═══════════════════════════════════════════════════════════════════════════════
# SPX PROPHET - PART 2: SPX ANCHORS TAB
# ═══════════════════════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════════════════════
# ENTRY/EXIT ANALYSIS FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def calculate_entry_exit_table(projection_df: pd.DataFrame, anchor_type: str) -> pd.DataFrame:
    """Calculate entry/exit analysis based on anchor bounce strategy"""
    if projection_df.empty:
        return pd.DataFrame()
    
    analysis_rows = []
    
    # Determine anchor characteristics
    is_skyline = anchor_type.upper() in ['SKYLINE', 'HIGH'] 
    is_baseline = anchor_type.upper() in ['BASELINE', 'LOW']
    
    for idx, row in projection_df.iterrows():
        time_slot = row['Time']
        anchor_price = row['Price']
        
        # Enhanced TP calculations based on market structure
        if is_skyline:
            # Skyline expects downward move after bounce
            volatility_adj = anchor_price * 0.015  # 1.5% typical move
            tp1_distance = volatility_adj * 0.6    # Conservative TP1
            tp2_distance = volatility_adj * 1.8    # Extended TP2
            
            entry_price = anchor_price
            tp1_price = anchor_price - tp1_distance
            tp2_price = anchor_price - tp2_distance
            direction = "BUY"  # Buy the bounce off skyline
            
            # Stop above skyline with retest buffer
            stop_price = anchor_price + (anchor_price * 0.008)  # 0.8% above
            
        elif is_baseline:
            # Baseline expects upward move after bounce  
            volatility_adj = anchor_price * 0.015
            tp1_distance = volatility_adj * 0.6
            tp2_distance = volatility_adj * 1.8
            
            entry_price = anchor_price
            tp1_price = anchor_price + tp1_distance
            tp2_price = anchor_price + tp2_distance
            direction = "BUY"  # Buy the bounce off baseline
            
            # Stop below baseline with retest buffer
            stop_price = max(0.01, anchor_price - (anchor_price * 0.008))  # 0.8% below
            
        else:
            # High/Close/Low anchors - more conservative
            volatility_adj = anchor_price * 0.012
            tp1_distance = volatility_adj * 0.5
            tp2_distance = volatility_adj * 1.4
            
            entry_price = anchor_price
            tp1_price = anchor_price + tp1_distance if anchor_type.upper() != 'HIGH' else anchor_price - tp1_distance
            tp2_price = anchor_price + tp2_distance if anchor_type.upper() != 'HIGH' else anchor_price - tp2_distance
            direction = "BUY" if anchor_type.upper() != 'HIGH' else "SELL"
            
            stop_distance = anchor_price * 0.006
            stop_price = entry_price - stop_distance if direction == "BUY" else entry_price + stop_distance
        
        risk_amount = abs(entry_price - stop_price)
        
        # Enhanced probability calculations
        entry_prob = calculate_anchor_entry_probability(anchor_type, time_slot)
        tp1_prob = calculate_anchor_target_probability(anchor_type, 1)
        tp2_prob = calculate_anchor_target_probability(anchor_type, 2)
        
        # Risk-reward ratios
        rr1 = abs(tp1_price - entry_price) / risk_amount if risk_amount > 0 else 0
        rr2 = abs(tp2_price - entry_price) / risk_amount if risk_amount > 0 else 0
        
        analysis_rows.append({
            'Time': time_slot,
            'Direction': direction,
            'Entry': round(entry_price, 2),
            'Stop': round(stop_price, 2),
            'TP1': round(tp1_price, 2),
            'TP2': round(tp2_price, 2),
            'Risk': round(risk_amount, 2),
            'RR1': f"{rr1:.1f}",
            'RR2': f"{rr2:.1f}",
            'Entry_Prob': f"{entry_prob:.0f}%",
            'TP1_Prob': f"{tp1_prob:.0f}%",
            'TP2_Prob': f"{tp2_prob:.0f}%",
            'Anchor_Type': anchor_type
        })
    
    return pd.DataFrame(analysis_rows)

def calculate_anchor_entry_probability(anchor_type: str, time_slot: str) -> float:
    """Calculate entry probability based on your anchor strategy"""
    base_probs = {
        'SKYLINE': 90.0,  # Your stated 90% for main anchors
        'BASELINE': 90.0,
        'HIGH': 75.0,
        'CLOSE': 80.0,
        'LOW': 75.0
    }
    
    base_prob = base_probs.get(anchor_type.upper(), 70.0)
    
    # Time-of-day adjustments
    hour = int(time_slot.split(':')[0])
    if hour in [8, 9]:  # Market open - higher volatility = higher probability
        time_adj = 8
    elif hour in [13, 14]:  # End of day momentum
        time_adj = 5
    else:
        time_adj = 0
    
    return min(95, base_prob + time_adj)

def calculate_anchor_target_probability(anchor_type: str, target_num: int) -> float:
    """Calculate target probability based on anchor strength"""
    if anchor_type.upper() in ['SKYLINE', 'BASELINE']:
        if target_num == 1:
            return 85.0  # High probability for TP1 on main anchors
        else:
            return 68.0  # TP2 still strong but lower
    else:
        if target_num == 1:
            return 75.0
        else:
            return 55.0

# Create main tabs
tab1, tab2, tab3, tab4 = st.tabs(["📈 SPX Anchors", "📚 Stock Anchors", "✅ Signals & EMA", "🧮 Contract Tool"])

with tab1:
    st.markdown("""
    <div style="background: linear-gradient(135deg, rgba(255,255,255,0.1), rgba(255,255,255,0.05)); padding: 1.5rem; border-radius: 15px; margin: 1rem 0; border: 2px solid rgba(255,255,255,0.2);">
        <h2>📈 SPX Anchor Analysis</h2>
        <p>Live ES futures data for anchor detection and SPX projections with your bounce strategy</p>
    </div>
    """, unsafe_allow_html=True)
    
    # ═══════════════════════════════════════════════════════════════════════════════
    # INPUT CONTROLS WITH ENHANCED UI
    # ═══════════════════════════════════════════════════════════════════════════════
    
    col1, col2 = st.columns(2)
    with col1:
        prev_day = st.date_input(
            "📅 Previous Trading Day", 
            value=datetime.now(CT_TZ).date() - timedelta(days=1),
            key="spx_prev_day",
            help="Date to analyze for anchor points"
        )
        
        # Show weekday for clarity
        weekday = prev_day.strftime("%A")
        st.caption(f"📆 Selected: {weekday}")
    
    with col2:
        proj_day = st.date_input(
            "🎯 Projection Day",
            value=prev_day + timedelta(days=1),
            key="spx_proj_day",
            help="RTH day to project entries for"
        )
        
        proj_weekday = proj_day.strftime("%A") 
        st.caption(f"📈 Projecting for: {proj_weekday}")
    
    # ES to SPX offset controls
    st.markdown("### 🔄 ES→SPX Offset Configuration")
    
    offset_col1, offset_col2 = st.columns(2)
    with offset_col1:
        manual_offset = st.number_input(
            "Manual Offset", 
            value=st.session_state.current_offset,
            step=0.1, format="%.1f", key="spx_offset_manual",
            help="Manual ES to SPX price differential"
        )
        st.session_state.current_offset = manual_offset
    
    with offset_col2:
        if st.button("🔄 Update from Live Data", key="spx_update_offset", type="secondary"):
            with st.spinner("Fetching live market data..."):
                # Get recent data for offset calculation
                es_data = fetch_live_data("ES=F", prev_day - timedelta(days=2), prev_day + timedelta(days=1))
                spx_data = fetch_live_data("^GSPC", prev_day - timedelta(days=2), prev_day + timedelta(days=1))
                
                if not es_data.empty and not spx_data.empty:
                    new_offset = calculate_es_spx_offset(es_data, spx_data)
                    st.session_state.current_offset = new_offset
                    st.success(f"✅ Offset updated: {new_offset:+.1f}")
                    st.rerun()
                else:
                    st.error("❌ Failed to fetch offset data")
    
    st.markdown("---")
    
    # ═══════════════════════════════════════════════════════════════════════════════
    # ENHANCED DATA ANALYSIS
    # ═══════════════════════════════════════════════════════════════════════════════
    
    if st.button("🚀 Generate SPX Anchors", key="spx_generate", type="primary"):
        with st.spinner("🔍 Analyzing market data..."):
            try:
                # Fetch ES futures data for the selected date
                es_data = fetch_live_data("ES=F", prev_day, prev_day)
                
                if es_data.empty:
                    st.error(f"❌ No ES futures data for {prev_day}")
                else:
                    st.success(f"✅ Retrieved {len(es_data)} ES data points")
                    
                    # Get anchor window data (17:00-19:30 CT)
                    anchor_window = get_session_window(es_data, SPX_ANCHOR_START, SPX_ANCHOR_END)
                    
                    if anchor_window.empty:
                        st.warning("⚠️ No data in ES anchor window (17:00-19:30 CT)")
                        # Use full day data as fallback
                        anchor_window = es_data
                    
                    # Store ES anchor data
                    st.session_state.es_anchor_data = anchor_window
                    
                    # Get SPX data for High/Close/Low anchors
                    spx_data = fetch_live_data("^GSPC", prev_day, prev_day)
                    
                    if not spx_data.empty:
                        # Extract actual daily OHLC
                        daily_ohlc = get_daily_ohlc(spx_data, prev_day)
                        
                        if daily_ohlc:
                            st.session_state.spx_manual_anchors = daily_ohlc
                            st.info("📊 SPX OHLC anchors extracted successfully")
                        else:
                            st.warning("⚠️ Could not extract SPX OHLC data")
                    else:
                        # Use ES data as fallback with offset
                        st.warning("⚠️ No SPX data, using ES with offset")
                        es_daily_ohlc = get_daily_ohlc(anchor_window, prev_day)
                        
                        if es_daily_ohlc:
                            # Apply offset to ES data
                            adjusted_ohlc = {}
                            for key, (price, timestamp) in es_daily_ohlc.items():
                                adjusted_ohlc[key] = (price + manual_offset, timestamp)
                            st.session_state.spx_manual_anchors = adjusted_ohlc
                    
                    st.session_state.spx_analysis_ready = True
                    st.success("🎯 SPX anchor analysis ready!")
                    
            except Exception as e:
                st.error(f"❌ Analysis error: {str(e)}")
    
    # ═══════════════════════════════════════════════════════════════════════════════
    # ENHANCED RESULTS DISPLAY
    # ═══════════════════════════════════════════════════════════════════════════════
    
    if st.session_state.get('spx_analysis_ready', False):
        st.markdown("## 📊 SPX Anchor Projections & Analysis")
        
        # Process swing detection for Skyline/Baseline
        es_data = st.session_state.get('es_anchor_data', pd.DataFrame())
        skyline_anchor = None
        baseline_anchor = None
        
        if not es_data.empty:
            es_swings = detect_swings_simple(es_data)
            skyline_anchor, baseline_anchor = get_anchor_points(es_swings)
        
        # Show anchor summary
        if st.session_state.get('spx_manual_anchors'):
            manual_anchors = st.session_state.spx_manual_anchors
            
            st.markdown("### 📋 Detected Anchors Summary")
            summary_cols = st.columns(5)
            
            anchor_names = ['high', 'close', 'low']
            anchor_icons = ['🔴', '🟡', '🟢']
            
            for i, (name, icon) in enumerate(zip(anchor_names, anchor_icons)):
                if i < len(summary_cols) and name in manual_anchors:
                    price, timestamp = manual_anchors[name]
                    with summary_cols[i]:
                        st.markdown(f"""
                        <div style="text-align: center; padding: 1rem; background: rgba(255,255,255,0.1); border-radius: 10px;">
                            <h4>{icon} {name.title()}</h4>
                            <h3>${price:.2f}</h3>
                            <p>{format_ct_time(timestamp)}</p>
                        </div>
                        """, unsafe_allow_html=True)
            
            # Swing anchors
            with summary_cols[3]:
                if skyline_anchor:
                    price, timestamp = skyline_anchor
                    st.markdown(f"""
                    <div style="text-align: center; padding: 1rem; background: rgba(255,100,100,0.2); border-radius: 10px;">
                        <h4>🔥 Skyline</h4>
                        <h3>${price:.2f}</h3>
                        <p>{format_ct_time(timestamp)}</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.warning("No Skyline")
            
            with summary_cols[4]:
                if baseline_anchor:
                    price, timestamp = baseline_anchor
                    st.markdown(f"""
                    <div style="text-align: center; padding: 1rem; background: rgba(100,100,255,0.2); border-radius: 10px;">
                        <h4>🏔️ Baseline</h4>
                        <h3>${price:.2f}</h3>
                        <p>{format_ct_time(timestamp)}</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.warning("No Baseline")
        
        # Enhanced projection tabs
        projection_tabs = st.tabs(["🔴 High", "🟡 Close", "🟢 Low", "🔥 Skyline", "🏔️ Baseline"])
        
        # Process each anchor type
        if st.session_state.get('spx_manual_anchors'):
            manual_anchors = st.session_state.spx_manual_anchors
            
            # High Anchor
            with projection_tabs[0]:
                if 'high' in manual_anchors:
                    price, timestamp = manual_anchors['high']
                    anchor_time_ct = timestamp.astimezone(CT_TZ)
                    
                    high_proj = project_anchor_line(
                        price, anchor_time_ct, 
                        st.session_state.spx_slopes['high'], proj_day
                    )
                    
                    if not high_proj.empty:
                        st.markdown("#### 🔴 High Anchor Price Projection")
                        st.dataframe(high_proj, use_container_width=True, hide_index=True)
                        
                        high_analysis = calculate_entry_exit_table(high_proj, "HIGH")
                        if not high_analysis.empty:
                            st.markdown("#### 🎯 Entry/Exit Strategy")
                            st.dataframe(high_analysis, use_container_width=True, hide_index=True)
                    else:
                        st.error("Failed to generate high projection")
                else:
                    st.warning("⚠️ No high anchor data available")
            
            # Close Anchor
            with projection_tabs[1]:
                if 'close' in manual_anchors:
                    price, timestamp = manual_anchors['close']
                    anchor_time_ct = timestamp.astimezone(CT_TZ)
                    
                    close_proj = project_anchor_line(
                        price, anchor_time_ct,
                        st.session_state.spx_slopes['close'], proj_day
                    )
                    
                    if not close_proj.empty:
                        st.markdown("#### 🟡 Close Anchor Price Projection")
                        st.dataframe(close_proj, use_container_width=True, hide_index=True)
                        
                        close_analysis = calculate_entry_exit_table(close_proj, "CLOSE")
                        if not close_analysis.empty:
                            st.markdown("#### 🎯 Entry/Exit Strategy")
                            st.dataframe(close_analysis, use_container_width=True, hide_index=True)
                else:
                    st.warning("⚠️ No close anchor data available")
            
            # Low Anchor
            with projection_tabs[2]:
                if 'low' in manual_anchors:
                    price, timestamp = manual_anchors['low']
                    anchor_time_ct = timestamp.astimezone(CT_TZ)
                    
                    low_proj = project_anchor_line(
                        price, anchor_time_ct,
                        st.session_state.spx_slopes['low'], proj_day
                    )
                    
                    if not low_proj.empty:
                        st.markdown("#### 🟢 Low Anchor Price Projection")
                        st.dataframe(low_proj, use_container_width=True, hide_index=True)
                        
                        low_analysis = calculate_entry_exit_table(low_proj, "LOW")
                        if not low_analysis.empty:
                            st.markdown("#### 🎯 Entry/Exit Strategy")
                            st.dataframe(low_analysis, use_container_width=True, hide_index=True)
                else:
                    st.warning("⚠️ No low anchor data available")
        
        # Swing-based projections (Skyline/Baseline)
        with projection_tabs[3]:  # Skyline
            if skyline_anchor:
                sky_price, sky_time = skyline_anchor
                spx_sky_price = sky_price + st.session_state.current_offset
                sky_time_ct = sky_time.astimezone(CT_TZ)
                
                skyline_proj = project_anchor_line(
                    spx_sky_price, sky_time_ct,
                    st.session_state.spx_slopes['skyline'], proj_day
                )
                
                if not skyline_proj.empty:
                    st.markdown("#### 🔥 Skyline Projection (90% Probability Zone)")
                    st.dataframe(skyline_proj, use_container_width=True, hide_index=True)
                    
                    sky_analysis = calculate_entry_exit_table(skyline_proj, "SKYLINE")
                    if not sky_analysis.empty:
                        st.markdown("#### 🎯 Skyline Bounce Strategy")
                        st.info("💡 Strategy: Bearish candle touches from above + closes above = BUY signal")
                        st.dataframe(sky_analysis, use_container_width=True, hide_index=True)
            else:
                st.warning("⚠️ No skyline anchor detected in ES data")
        
        with projection_tabs[4]:  # Baseline
            if baseline_anchor:
                base_price, base_time = baseline_anchor
                spx_base_price = base_price + st.session_state.current_offset
                base_time_ct = base_time.astimezone(CT_TZ)
                
                baseline_proj = project_anchor_line(
                    spx_base_price, base_time_ct,
                    st.session_state.spx_slopes['baseline'], proj_day
                )
                
                if not baseline_proj.empty:
                    st.markdown("#### 🏔️ Baseline Projection (90% Probability Zone)")
                    st.dataframe(baseline_proj, use_container_width=True, hide_index=True)
                    
                    base_analysis = calculate_entry_exit_table(baseline_proj, "BASELINE")
                    if not base_analysis.empty:
                        st.markdown("#### 🎯 Baseline Bounce Strategy")
                        st.info("💡 Strategy: Bearish candle touches from above + closes above = BUY signal")
                        st.dataframe(base_analysis, use_container_width=True, hide_index=True)
            else:
                st.warning("⚠️ No baseline anchor detected in ES data")
    
    else:
        st.info("👆 Configure your dates and click 'Generate SPX Anchors' to begin analysis")

# ═══════════════════════════════════════════════════════════════════════════════
# END OF SPX ANCHORS TAB  
# ═══════════════════════════════════════════════════════════════════════════════
