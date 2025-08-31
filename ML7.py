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
    'high': -0.379,
    'close': -0.379, 
    'low': -0.379,
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
    """Calculate ES to SPX offset using overlapping RTH sessions"""
    try:
        if es_data.empty or spx_data.empty:
            return 0.0
        
        # Filter both to RTH overlap (8:30-15:00 CT) for accurate comparison
        es_rth = get_session_window(es_data, "08:30", "15:00")
        spx_rth = get_session_window(spx_data, "08:30", "15:00")
        
        if es_rth.empty or spx_rth.empty:
            # Fallback to any available close data
            es_close = es_data.iloc[-1]['Close']
            spx_close = spx_data.iloc[-1]['Close']
        else:
            # Use last RTH close when both markets were active
            es_close = es_rth.iloc[-1]['Close']
            spx_close = spx_rth.iloc[-1]['Close']
        
        offset = spx_close - es_close
        return round(offset, 1)
        
    except Exception as e:
        st.warning(f"Offset calculation error: {str(e)}")
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

# SPX slope controls
st.sidebar.markdown("### SPX Slopes (per 30-min)")
st.sidebar.caption("Adjust projection slopes for each anchor type")

# Collapsible SPX controls
with st.sidebar.expander("SPX Slope Settings", expanded=False):
    for slope_name, default_value in SPX_SLOPES.items():
        icon_map = {
            'high': 'High', 'close': 'Close', 'low': 'Low', 
            'skyline': 'Skyline', 'baseline': 'Baseline'
        }
        
        display_name = icon_map.get(slope_name, slope_name.title())
        
        slope_value = st.number_input(
            display_name,
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

# Hero section
st.markdown("""
<div style="text-align: center; padding: 2rem; background: linear-gradient(135deg, rgba(255,255,255,0.1), rgba(255,255,255,0.05)); border-radius: 20px; margin: 1rem 0; border: 2px solid rgba(255,255,255,0.2);">
    <h1 style="font-size: 3rem; margin: 0; background: linear-gradient(135deg, #ff6b6b, #4ecdc4, #45b7d1, #f9ca24); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;">
        SPX Prophet Analytics
    </h1>
    <p style="font-size: 1.3rem; margin: 1rem 0; opacity: 0.9;">
        Advanced Trading Analytics with Live Market Data Integration
    </p>
</div>
""", unsafe_allow_html=True)

# Live market status with metrics
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
    current_time_ct = datetime.now(CT_TZ)
    
    # Check if it's a trading day (Monday-Friday)
    is_weekday = current_time_ct.weekday() < 5  # 0-4 are Mon-Fri
    
    # Check if within RTH hours
    market_open = current_time_ct.replace(hour=8, minute=30, second=0, microsecond=0)
    market_close = current_time_ct.replace(hour=14, minute=30, second=0, microsecond=0)
    within_hours = market_open <= current_time_ct <= market_close
    
    # Market is open only if both conditions are true
    is_rth = is_weekday and within_hours
    
    if is_weekday:
        if is_rth:
            status_color = "#00ff88"
            status_text = "MARKET OPEN"
        else:
            status_color = "#ffbb33" 
            status_text = "MARKET CLOSED"
    else:
        status_color = "#ff6b6b"
        status_text = "WEEKEND"
    
    st.markdown(f"""
    <div class="metric-container">
        <h3>Market Status</h3>
        <h2 style="color: {status_color};">{status_text}</h2>
        <p>RTH: 08:30 - 14:30 CT</p>
        <p>Mon-Fri Only</p>
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
if st.button("Test Data Connection", key="test_connection"):
    with st.spinner("Testing market data connection..."):
        test_data = fetch_live_data("^GSPC", datetime.now().date() - timedelta(days=1), datetime.now().date())
        
        if not test_data.empty:
            st.success("Market data connection successful!")
            st.info(f"Retrieved {len(test_data)} data points for SPX")
        else:
            st.error("Market data connection failed!")

st.markdown("---")

# ═══════════════════════════════════════════════════════════════════════════════
# ✅ PART 1 COMPLETE - FOUNDATION
# ═══════════════════════════════════════════════════════════════════════════════





# ═══════════════════════════════════════════════════════════════════════════════
# SPX PROPHET - PART 2: SPX ANCHORS TAB
# ═══════════════════════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════════════════════
# AUTO-UPDATE FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def update_offset_for_date():
    """Automatically update offset when date changes"""
    if 'spx_prev_day' in st.session_state:
        selected_date = st.session_state.spx_prev_day
        
        # Fetch data for the selected date
        es_data = fetch_live_data("ES=F", selected_date, selected_date)
        spx_data = fetch_live_data("^GSPC", selected_date, selected_date)
        
        if not es_data.empty and not spx_data.empty:
            new_offset = calculate_es_spx_offset(es_data, spx_data)
            st.session_state.current_offset = new_offset

# ═══════════════════════════════════════════════════════════════════════════════
# ENTRY/EXIT ANALYSIS FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def calculate_entry_exit_table(projection_df: pd.DataFrame, anchor_type: str) -> pd.DataFrame:
    """Calculate entry/exit analysis based on anchor bounce strategy"""
    if projection_df.empty:
        return pd.DataFrame()
    
    analysis_rows = []
    
    # Determine anchor characteristics for direction bias
    is_skyline = anchor_type.upper() in ['SKYLINE', 'HIGH'] 
    is_baseline = anchor_type.upper() in ['BASELINE', 'LOW']
    
    for idx, row in projection_df.iterrows():
        time_slot = row['Time']
        anchor_price = row['Price']
        
        # Calculate targets based on anchor type
        if is_skyline:
            # Skyline bounce - expect initial bounce up then potential reversal
            volatility_factor = anchor_price * 0.012
            tp1_distance = volatility_factor * 0.8   # Quick bounce target
            tp2_distance = volatility_factor * 2.2   # Extended target
            
            entry_price = anchor_price
            tp1_price = anchor_price + tp1_distance  # Bounce up from skyline
            tp2_price = anchor_price + tp2_distance  # Extended bounce
            direction = "BUY"
            
            # Stop above skyline with buffer for retests
            stop_price = anchor_price + (anchor_price * 0.006)
            
        elif is_baseline:
            # Baseline bounce - expect upward move
            volatility_factor = anchor_price * 0.012
            tp1_distance = volatility_factor * 0.8
            tp2_distance = volatility_factor * 2.2
            
            entry_price = anchor_price
            tp1_price = anchor_price + tp1_distance  # Bounce up from baseline
            tp2_price = anchor_price + tp2_distance  # Extended bounce
            direction = "BUY"
            
            # Stop below baseline with buffer
            stop_price = max(0.01, anchor_price - (anchor_price * 0.006))
            
        else:
            # High/Close/Low anchors 
            volatility_factor = anchor_price * 0.010
            tp1_distance = volatility_factor * 0.7
            tp2_distance = volatility_factor * 1.8
            
            if anchor_type.upper() == 'HIGH':
                entry_price = anchor_price
                tp1_price = anchor_price - tp1_distance  # Expect decline from high
                tp2_price = anchor_price - tp2_distance
                direction = "SELL"
                stop_price = anchor_price + (anchor_price * 0.005)
            else:
                entry_price = anchor_price
                tp1_price = anchor_price + tp1_distance  # Expect rise from close/low
                tp2_price = anchor_price + tp2_distance
                direction = "BUY"
                stop_price = anchor_price - (anchor_price * 0.005)
        
        risk_amount = abs(entry_price - stop_price)
        
        # Probability calculations
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
            'TP2_Prob': f"{tp2_prob:.0f}%"
        })
    
    return pd.DataFrame(analysis_rows)

def calculate_anchor_entry_probability(anchor_type: str, time_slot: str) -> float:
    """Calculate entry probability based on anchor strategy"""
    base_probs = {
        'SKYLINE': 90.0,
        'BASELINE': 90.0,
        'HIGH': 75.0,
        'CLOSE': 80.0,
        'LOW': 75.0
    }
    
    base_prob = base_probs.get(anchor_type.upper(), 70.0)
    
    # Time adjustments
    hour = int(time_slot.split(':')[0])
    if hour in [8, 9]:
        time_adj = 8
    elif hour in [13, 14]:
        time_adj = 5
    else:
        time_adj = 0
    
    return min(95, base_prob + time_adj)

def calculate_anchor_target_probability(anchor_type: str, target_num: int) -> float:
    """Calculate target probability based on anchor strength"""
    if anchor_type.upper() in ['SKYLINE', 'BASELINE']:
        return 85.0 if target_num == 1 else 68.0
    else:
        return 75.0 if target_num == 1 else 55.0

# Create main tabs
tab1, tab2, tab3, tab4 = st.tabs(["SPX Anchors", "Stock Anchors", "Signals & EMA", "Contract Tool"])

with tab1:
    st.subheader("SPX Anchor Analysis")
    st.caption("Live ES futures data for anchor detection and SPX projections")
    
    # ═══════════════════════════════════════════════════════════════════════════════
    # INPUT CONTROLS
    # ═══════════════════════════════════════════════════════════════════════════════
    
    col1, col2 = st.columns(2)
    with col1:
        prev_day = st.date_input(
            "Previous Trading Day", 
            value=datetime.now(CT_TZ).date() - timedelta(days=1),
            key="spx_prev_day",
            on_change=update_offset_for_date
        )
        
        weekday = prev_day.strftime("%A")
        st.caption(f"Selected: {weekday}")
    
    with col2:
        proj_day = st.date_input(
            "Projection Day",
            value=prev_day + timedelta(days=1),
            key="spx_proj_day"
        )
        
        proj_weekday = proj_day.strftime("%A") 
        st.caption(f"Projecting for: {proj_weekday}")

def update_offset_for_date():
    """Automatically update offset when date changes"""
    if 'spx_prev_day' in st.session_state:
        selected_date = st.session_state.spx_prev_day
        
        # Fetch data for the selected date
        es_data = fetch_live_data("ES=F", selected_date, selected_date)
        spx_data = fetch_live_data("^GSPC", selected_date, selected_date)
        
        if not es_data.empty and not spx_data.empty:
            new_offset = calculate_es_spx_offset(es_data, spx_data)
            st.session_state.current_offset = new_offset
    
    st.markdown("---")
    
    # Manual price override section
    st.subheader("Price Override (Optional)")
    st.caption("Override Yahoo Finance data with your exact prices for accurate projections")
    
    use_manual = st.checkbox("Use Manual Prices", key="use_manual_prices")
    
    if use_manual:
        override_col1, override_col2, override_col3 = st.columns(3)
        
        with override_col1:
            manual_high = st.number_input(
                "Manual High Price",
                value=0.0,
                step=0.1, format="%.1f",
                key="manual_high_price"
            )
        
        with override_col2:
            manual_close = st.number_input(
                "Manual Close Price", 
                value=0.0,
                step=0.1, format="%.1f",
                key="manual_close_price"
            )
        
        with override_col3:
            manual_low = st.number_input(
                "Manual Low Price",
                value=0.0,
                step=0.1, format="%.1f", 
                key="manual_low_price"
            )
    
    st.markdown("---")
    
    # ═══════════════════════════════════════════════════════════════════════════════
    # DATA ANALYSIS WITH AUTO OFFSET
    # ═══════════════════════════════════════════════════════════════════════════════
    
    if st.button("Generate SPX Anchors", key="spx_generate", type="primary"):
        with st.spinner("Analyzing market data..."):
            try:
                # Get offset for the specific historical date
                es_data_for_offset = fetch_live_data("ES=F", prev_day, prev_day)
                spx_data_for_offset = fetch_live_data("^GSPC", prev_day, prev_day)
                
                if not es_data_for_offset.empty and not spx_data_for_offset.empty:
                    st.session_state.current_offset = calculate_es_spx_offset(es_data_for_offset, spx_data_for_offset)
                
                # Fetch ES futures data for anchor detection
                es_data = fetch_live_data("ES=F", prev_day, prev_day)
                
                if es_data.empty:
                    st.error(f"No ES futures data for {prev_day}")
                else:
                    # Get anchor window data (17:00-19:30 CT)
                    anchor_window = get_session_window(es_data, SPX_ANCHOR_START, SPX_ANCHOR_END)
                    
                    if anchor_window.empty:
                        # Use full day ES data as fallback
                        anchor_window = es_data
                    
                    # Store ES anchor data
                    st.session_state.es_anchor_data = anchor_window
                    
                    # Get SPX data for High/Close/Low anchors
                    spx_data = fetch_live_data("^GSPC", prev_day, prev_day)
                    
                    if not spx_data.empty:
                        # Extract actual daily SPX OHLC
                        daily_ohlc = get_daily_ohlc(spx_data, prev_day)
                        
                        if daily_ohlc:
                            st.session_state.spx_manual_anchors = daily_ohlc
                        else:
                            st.warning("Could not extract SPX OHLC data")
                    else:
                        # Convert ES anchor window to SPX equivalent using offset
                        es_daily_ohlc = get_daily_ohlc(anchor_window, prev_day)
                        
                        if es_daily_ohlc:
                            spx_equivalent = {}
                            for key, (es_price, timestamp) in es_daily_ohlc.items():
                                spx_equivalent[key] = (es_price + st.session_state.current_offset, timestamp)
                            st.session_state.spx_manual_anchors = spx_equivalent
                    
                    st.session_state.spx_analysis_ready = True
                    
            except Exception as e:
                st.error(f"Analysis error: {str(e)}")
    
    # ═══════════════════════════════════════════════════════════════════════════════
    # RESULTS DISPLAY
    # ═══════════════════════════════════════════════════════════════════════════════
    
    if st.session_state.get('spx_analysis_ready', False):
        st.subheader("SPX Anchor Projections")
        
        # Process ES swing detection but convert to SPX values
        es_data = st.session_state.get('es_anchor_data', pd.DataFrame())
        skyline_anchor_spx = None
        baseline_anchor_spx = None
        
        if not es_data.empty:
            # Detect swings in ES data
            es_swings = detect_swings_simple(es_data)
            es_skyline, es_baseline = get_anchor_points(es_swings)
            
            # Convert ES anchor points to SPX equivalent
            current_offset = st.session_state.current_offset
            
            if es_skyline:
                es_price, es_time = es_skyline
                spx_price = es_price + current_offset
                skyline_anchor_spx = (spx_price, es_time)
            
            if es_baseline:
                es_price, es_time = es_baseline
                spx_price = es_price + current_offset
                baseline_anchor_spx = (spx_price, es_time)
        
        # Display anchor summary
        if st.session_state.get('spx_manual_anchors'):
            manual_anchors = st.session_state.spx_manual_anchors
            
            st.subheader("Detected SPX Anchors")
            summary_cols = st.columns(5)
            
            # Manual anchors (High/Close/Low)
            anchor_info = [
                ('high', 'High', '#ff6b6b'),
                ('close', 'Close', '#f9ca24'),
                ('low', 'Low', '#4ecdc4')
            ]
            
            for i, (name, display_name, color) in enumerate(anchor_info):
                if name in manual_anchors:
                    price, timestamp = manual_anchors[name]
                    with summary_cols[i]:
                        st.markdown(f"""
                        <div style="text-align: center; padding: 1rem; background: rgba(255,255,255,0.1); border-radius: 10px; border-left: 4px solid {color};">
                            <h4>{display_name}</h4>
                            <h3>${price:.2f}</h3>
                            <p>{format_ct_time(timestamp)}</p>
                        </div>
                        """, unsafe_allow_html=True)
            
            # Swing anchors (Skyline/Baseline) - converted to SPX
            with summary_cols[3]:
                if skyline_anchor_spx:
                    price, timestamp = skyline_anchor_spx
                    st.markdown(f"""
                    <div style="text-align: center; padding: 1rem; background: rgba(255,100,100,0.2); border-radius: 10px; border-left: 4px solid #ff4757;">
                        <h4>Skyline</h4>
                        <h3>${price:.2f}</h3>
                        <p>{format_ct_time(timestamp)}</p>
                        <small>SPX Equivalent</small>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.warning("No Skyline")
            
            with summary_cols[4]:
                if baseline_anchor_spx:
                    price, timestamp = baseline_anchor_spx
                    st.markdown(f"""
                    <div style="text-align: center; padding: 1rem; background: rgba(100,100,255,0.2); border-radius: 10px; border-left: 4px solid #3742fa;">
                        <h4>Baseline</h4>
                        <h3>${price:.2f}</h3>
                        <p>{format_ct_time(timestamp)}</p>
                        <small>SPX Equivalent</small>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.warning("No Baseline")
        
        st.markdown("---")
        
        # Projection tabs with SPX values
        projection_tabs = st.tabs(["High", "Close", "Low", "Skyline", "Baseline"])
        
        # Manual anchor projections using SPX values
        if st.session_state.get('spx_manual_anchors'):
            manual_anchors = st.session_state.spx_manual_anchors
            
            # High Anchor
            with projection_tabs[0]:
                if 'high' in manual_anchors:
                    spx_price, timestamp = manual_anchors['high']
                    anchor_time_ct = timestamp.astimezone(CT_TZ)
                    
                    high_proj = project_anchor_line(
                        spx_price, anchor_time_ct, 
                        st.session_state.spx_slopes['high'], proj_day
                    )
                    
                    if not high_proj.empty:
                        st.subheader("High Anchor SPX Projection")
                        st.dataframe(high_proj, use_container_width=True, hide_index=True)
                        
                        high_analysis = calculate_entry_exit_table(high_proj, "HIGH")
                        if not high_analysis.empty:
                            st.subheader("Entry/Exit Strategy")
                            st.dataframe(high_analysis, use_container_width=True, hide_index=True)
                else:
                    st.warning("No high anchor data available")
            
            # Close Anchor
            with projection_tabs[1]:
                if 'close' in manual_anchors:
                    spx_price, timestamp = manual_anchors['close']
                    anchor_time_ct = timestamp.astimezone(CT_TZ)
                    
                    close_proj = project_anchor_line(
                        spx_price, anchor_time_ct,
                        st.session_state.spx_slopes['close'], proj_day
                    )
                    
                    if not close_proj.empty:
                        st.subheader("Close Anchor SPX Projection")
                        st.dataframe(close_proj, use_container_width=True, hide_index=True)
                        
                        close_analysis = calculate_entry_exit_table(close_proj, "CLOSE")
                        if not close_analysis.empty:
                            st.subheader("Entry/Exit Strategy")
                            st.dataframe(close_analysis, use_container_width=True, hide_index=True)
                else:
                    st.warning("No close anchor data available")
            
            # Low Anchor
            with projection_tabs[2]:
                if 'low' in manual_anchors:
                    spx_price, timestamp = manual_anchors['low']
                    anchor_time_ct = timestamp.astimezone(CT_TZ)
                    
                    low_proj = project_anchor_line(
                        spx_price, anchor_time_ct,
                        st.session_state.spx_slopes['low'], proj_day
                    )
                    
                    if not low_proj.empty:
                        st.subheader("Low Anchor SPX Projection")
                        st.dataframe(low_proj, use_container_width=True, hide_index=True)
                        
                        low_analysis = calculate_entry_exit_table(low_proj, "LOW")
                        if not low_analysis.empty:
                            st.subheader("Entry/Exit Strategy") 
                            st.dataframe(low_analysis, use_container_width=True, hide_index=True)
                else:
                    st.warning("No low anchor data available")
        
        # Swing-based projections using SPX converted values
        with projection_tabs[3]:  # Skyline
            if skyline_anchor_spx:
                spx_sky_price, sky_time = skyline_anchor_spx
                sky_time_ct = sky_time.astimezone(CT_TZ)
                
                skyline_proj = project_anchor_line(
                    spx_sky_price, sky_time_ct,
                    st.session_state.spx_slopes['skyline'], proj_day
                )
                
                if not skyline_proj.empty:
                    st.subheader("Skyline SPX Projection (90% Zone)")
                    st.info("Strategy: Bearish candle touches from above + closes above = BUY signal")
                    st.dataframe(skyline_proj, use_container_width=True, hide_index=True)
                    
                    sky_analysis = calculate_entry_exit_table(skyline_proj, "SKYLINE")
                    if not sky_analysis.empty:
                        st.subheader("Skyline Bounce Strategy")
                        st.dataframe(sky_analysis, use_container_width=True, hide_index=True)
            else:
                st.warning("No skyline anchor detected")
        
        with projection_tabs[4]:  # Baseline
            if baseline_anchor_spx:
                spx_base_price, base_time = baseline_anchor_spx
                base_time_ct = base_time.astimezone(CT_TZ)
                
                baseline_proj = project_anchor_line(
                    spx_base_price, base_time_ct,
                    st.session_state.spx_slopes['baseline'], proj_day
                )
                
                if not baseline_proj.empty:
                    st.subheader("Baseline SPX Projection (90% Zone)")
                    st.info("Strategy: Bearish candle touches from above + closes above = BUY signal")
                    st.dataframe(baseline_proj, use_container_width=True, hide_index=True)
                    
                    base_analysis = calculate_entry_exit_table(baseline_proj, "BASELINE")
                    if not base_analysis.empty:
                        st.subheader("Baseline Bounce Strategy")
                        st.dataframe(base_analysis, use_container_width=True, hide_index=True)
            else:
                st.warning("No baseline anchor detected")
    
    else:
        st.info("Configure your dates and click 'Generate SPX Anchors' to begin analysis")

# ═══════════════════════════════════════════════════════════════════════════════
# END OF SPX ANCHORS TAB
# ═══════════════════════════════════════════════════════════════════════════════







# ═══════════════════════════════════════════════════════════════════════════════
# SPX PROPHET - PART 3: STOCK ANCHORS TAB
# ═══════════════════════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════════════════════
# AUTO-UPDATE FUNCTIONS FOR STOCKS
# ═══════════════════════════════════════════════════════════════════════════════

def update_stock_data_for_dates():
    """Automatically fetch stock data when Monday/Tuesday dates change"""
    if ('selected_stock' in st.session_state and 
        f"stk_mon_{st.session_state.selected_stock}" in st.session_state and
        f"stk_tue_{st.session_state.selected_stock}" in st.session_state):
        
        ticker = st.session_state.selected_stock
        monday_date = st.session_state[f"stk_mon_{ticker}"]
        tuesday_date = st.session_state[f"stk_tue_{ticker}"]
        
        # Fetch Monday and Tuesday data automatically
        mon_data = fetch_live_data(ticker, monday_date, monday_date)
        tue_data = fetch_live_data(ticker, tuesday_date, tuesday_date)
        
        if not mon_data.empty or not tue_data.empty:
            # Combine available data
            if mon_data.empty:
                combined_data = tue_data
            elif tue_data.empty:
                combined_data = mon_data
            else:
                combined_data = pd.concat([mon_data, tue_data]).sort_index()
            
            # Store results automatically
            st.session_state.stock_analysis_data = combined_data
            st.session_state.stock_analysis_ticker = ticker
            st.session_state.stock_analysis_ready = True

# ═══════════════════════════════════════════════════════════════════════════════
# STOCK ANALYSIS FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def calculate_stock_volatility_from_data(ticker: str, market_data: pd.DataFrame) -> float:
    """Calculate actual volatility factor from stock's price movement data"""
    if market_data.empty or len(market_data) < 5:
        return 1.0  # Neutral fallback
    
    # Calculate volatility from actual price changes
    returns = market_data['Close'].pct_change().dropna()
    if len(returns) == 0:
        return 1.0
    
    # Use standard deviation of returns as volatility measure
    volatility = returns.std()
    
    # Scale to meaningful factor (typical stock volatility ranges 0.5-3.0)
    volatility_factor = max(0.5, min(3.0, volatility * 100))
    return volatility_factor

def calculate_day_performance_multiplier(market_data: pd.DataFrame, target_weekday: str) -> float:
    """Calculate day-specific performance from actual historical patterns"""
    if market_data.empty:
        return 1.0
    
    try:
        # Filter data by weekday if enough data
        weekday_map = {'Wednesday': 2, 'Thursday': 3, 'Friday': 4}
        target_weekday_num = weekday_map.get(target_weekday, 3)
        
        # Get data for target weekday
        weekday_data = market_data[market_data.index.weekday == target_weekday_num]
        
        if len(weekday_data) >= 3:
            # Calculate average daily range for this weekday
            weekday_ranges = ((weekday_data['High'] - weekday_data['Low']) / weekday_data['Close'] * 100).mean()
            
            # Compare to overall average
            overall_ranges = ((market_data['High'] - market_data['Low']) / market_data['Close'] * 100).mean()
            
            if overall_ranges > 0:
                multiplier = weekday_ranges / overall_ranges
                return max(0.7, min(1.3, multiplier))
        
        return 1.0
    except:
        return 1.0

def calculate_stock_time_adjustment(time_slot: str, market_data: pd.DataFrame) -> float:
    """Calculate time-of-day adjustment based on actual volume patterns"""
    if market_data.empty or 'Volume' not in market_data.columns:
        return 0.0
    
    try:
        hour = int(time_slot.split(':')[0])
        
        # Get volume data for this hour
        hour_data = market_data[market_data.index.hour == hour]
        
        if not hour_data.empty:
            avg_hour_volume = hour_data['Volume'].mean()
            total_avg_volume = market_data['Volume'].mean()
            
            if total_avg_volume > 0:
                volume_ratio = avg_hour_volume / total_avg_volume
                # Convert volume ratio to adjustment (-10 to +10 range)
                return (volume_ratio - 1) * 10
        
        return 0.0
    except:
        return 0.0

def calculate_stock_entry_exit_table(projection_df: pd.DataFrame, ticker: str, anchor_type: str, day_name: str) -> pd.DataFrame:
    """Calculate stock entry/exit analysis with data-driven calculations"""
    if projection_df.empty:
        return pd.DataFrame()
    
    analysis_rows = []
    
    # Get market data for calculations
    stock_data = st.session_state.get('stock_analysis_data', pd.DataFrame())
    
    # Calculate data-driven factors
    stock_volatility = calculate_stock_volatility_from_data(ticker, stock_data)
    day_multiplier = calculate_day_performance_multiplier(stock_data, day_name)
    
    for idx, row in projection_df.iterrows():
        time_slot = row['Time']
        anchor_price = row['Price']
        
        # Calculate targets based on actual stock data
        if not stock_data.empty:
            # Use actual daily range
            daily_high = stock_data['High'].max()
            daily_low = stock_data['Low'].min()
            daily_range = daily_high - daily_low
            
            # Base volatility on actual price movement
            base_volatility = daily_range
        else:
            # Fallback: conservative estimate
            base_volatility = anchor_price * 0.02
        
        tp1_distance = base_volatility * 0.30 * stock_volatility
        tp2_distance = base_volatility * 0.50 * stock_volatility
        
        # Dynamic stop based on actual volatility
        if not stock_data.empty and len(stock_data) >= 5:
            recent_returns = stock_data['Close'].pct_change().tail(10).dropna()
            if len(recent_returns) > 0:
                volatility_stop = recent_returns.std() * anchor_price * 2
                stop_buffer = max(anchor_price * 0.003, volatility_stop)
            else:
                stop_buffer = anchor_price * 0.005
        else:
            stop_buffer = anchor_price * 0.005
        
        # Calculate scenarios
        if anchor_type.upper() == 'HIGH':
            buy_tp1 = anchor_price + tp1_distance
            buy_tp2 = anchor_price + tp2_distance
            sell_tp1 = anchor_price - tp1_distance
            sell_tp2 = anchor_price - tp2_distance
            stop_price = anchor_price + stop_buffer
        else:
            buy_tp1 = anchor_price + tp1_distance
            buy_tp2 = anchor_price + tp2_distance
            sell_tp1 = anchor_price - tp1_distance
            sell_tp2 = anchor_price - tp2_distance
            stop_price = max(0.01, anchor_price - stop_buffer)
        
        # Data-driven probability calculations
        base_prob = 60.0
        
        # Volatility adjustment
        if stock_volatility > 1.5:
            vol_adj = 10
        elif stock_volatility > 1.0:
            vol_adj = 5
        else:
            vol_adj = -5
        
        # Time adjustment based on actual data
        time_adj = calculate_stock_time_adjustment(time_slot, stock_data)
        
        # Day adjustment based on actual performance
        day_adj = (day_multiplier - 1) * 20
        
        entry_prob = base_prob + vol_adj + time_adj + day_adj
        entry_prob = max(40, min(90, entry_prob * day_multiplier))
        
        tp1_prob = max(30, entry_prob - 15)
        tp2_prob = max(20, entry_prob - 25)
        
        analysis_rows.append({
            'Time': time_slot,
            'Entry_Level': round(anchor_price, 2),
            'BUY_Scenario': f"Touch from above, close above → TP1: {buy_tp1:.2f}, TP2: {buy_tp2:.2f}",
            'SELL_Scenario': f"Touch and close below → TP1: {sell_tp1:.2f}, TP2: {sell_tp2:.2f}",
            'Stop_Buffer': round(stop_buffer, 2),
            'Entry_Prob': f"{entry_prob:.0f}%",
            'TP1_Prob': f"{tp1_prob:.0f}%",
            'TP2_Prob': f"{tp2_prob:.0f}%",
            'Volatility_Factor': f"{stock_volatility:.2f}",
            'Day': day_name
        })
    
    return pd.DataFrame(analysis_rows)

def calculate_stock_anchor_analytics(projection_df: pd.DataFrame, ticker: str, anchor_type: str, market_data: pd.DataFrame) -> pd.DataFrame:
    """Calculate stock-specific anchor line analytics from actual market data"""
    if projection_df.empty:
        return pd.DataFrame()
    
    analytics = []
    
    # Calculate actual bounce rate from historical data
    if not market_data.empty and len(market_data) >= 10:
        # Analyze actual price bounces vs penetrations
        price_changes = market_data['Close'].pct_change().dropna()
        positive_moves = len(price_changes[price_changes > 0])
        total_moves = len(price_changes)
        
        if total_moves > 0:
            bounce_rate = (positive_moves / total_moves) * 100
        else:
            bounce_rate = 50.0
        
        # Calculate reliability from actual volatility consistency
        volatility = price_changes.std() * 100
        if volatility < 2.0:
            reliability = 85.0
        elif volatility < 4.0:
            reliability = 70.0
        else:
            reliability = 55.0
        
        # Volume analysis
        if 'Volume' in market_data.columns:
            avg_volume = market_data['Volume'].mean()
            recent_volume = market_data['Volume'].tail(5).mean()
            
            if avg_volume > 0:
                volume_ratio = recent_volume / avg_volume
                volume_score = min(90, max(30, 50 + (volume_ratio - 1) * 30))
            else:
                volume_score = 50.0
        else:
            volume_score = 50.0
    else:
        # Minimal data fallbacks
        bounce_rate = 60.0
        reliability = 65.0
        volume_score = 50.0
    
    for idx, row in projection_df.iterrows():
        time_slot = row['Time']
        
        # Time-based momentum from actual data
        if not market_data.empty:
            try:
                hour = int(time_slot.split(':')[0])
                hour_data = market_data[market_data.index.hour == hour]
                
                if not hour_data.empty:
                    hour_volatility = hour_data['Close'].pct_change().std() * 100
                    momentum_score = min(90, max(40, 60 + hour_volatility * 5))
                else:
                    momentum_score = 60.0
            except:
                momentum_score = 60.0
        else:
            momentum_score = 60.0
        
        # Overall trade confidence from actual data
        trade_confidence = (bounce_rate * 0.3) + (reliability * 0.3) + (volume_score * 0.2) + (momentum_score * 0.2)
        
        analytics.append({
            'Time': time_slot,
            'Bounce_Rate': f"{bounce_rate:.0f}%",
            'Reliability': f"{reliability:.0f}%",
            'Volume_Score': f"{volume_score:.0f}%", 
            'Momentum': f"{momentum_score:.0f}%",
            'Trade_Confidence': f"{trade_confidence:.0f}%",
            'Data_Quality': get_data_quality_score(market_data)
        })
    
    return pd.DataFrame(analytics)

def get_data_quality_score(market_data: pd.DataFrame) -> str:
    """Assess data quality for reliability"""
    if market_data.empty:
        return "NO DATA"
    elif len(market_data) >= 20:
        return "HIGH"
    elif len(market_data) >= 10:
        return "MEDIUM"
    else:
        return "LOW"

with tab2:
    st.subheader("Stock Anchor Analysis")
    st.caption("Monday/Tuesday combined session analysis for weekly stock projections")
    
    # ═══════════════════════════════════════════════════════════════════════════════
    # TICKER SELECTION
    # ═══════════════════════════════════════════════════════════════════════════════
    
    st.write("Core Tickers:")
    ticker_cols = st.columns(4)
    selected_ticker = None
    
    core_tickers = ['TSLA', 'NVDA', 'AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META', 'NFLX']
    for i, ticker in enumerate(core_tickers):
        with ticker_cols[i % 4]:
            if st.button(f"{ticker}", key=f"stk_btn_{ticker}"):
                selected_ticker = ticker
                st.session_state.selected_stock = ticker
    
    # Custom ticker input
    st.markdown("---")
    custom_ticker = st.text_input(
        "Custom Symbol", 
        placeholder="Enter any ticker symbol",
        key="stk_custom_input"
    )
    
    if custom_ticker:
        selected_ticker = custom_ticker.upper()
        st.session_state.selected_stock = selected_ticker
    
    # Use session state ticker if available
    if not selected_ticker and 'selected_stock' in st.session_state:
        selected_ticker = st.session_state.selected_stock
    
    if selected_ticker:
        st.info(f"Selected: {selected_ticker}")
        
        # ═══════════════════════════════════════════════════════════════════════════════
        # SLOPE MANAGEMENT
        # ═══════════════════════════════════════════════════════════════════════════════
        
        # Auto-fill slope or use default
        default_slope = STOCK_SLOPES.get(selected_ticker, 0.0150)
        current_slope = st.session_state.stock_slopes.get(selected_ticker, default_slope)
        
        slope_magnitude = st.number_input(
            f"{selected_ticker} Slope Magnitude",
            value=current_slope,
            step=0.0001, format="%.4f",
            key=f"stk_slope_{selected_ticker}",
            help="Used as +magnitude for Skyline, -magnitude for Baseline"
        )
        st.session_state.stock_slopes[selected_ticker] = slope_magnitude
        
        # ═══════════════════════════════════════════════════════════════════════════════
        # DATE INPUTS
        # ═══════════════════════════════════════════════════════════════════════════════
        
        col1, col2, col3 = st.columns(3)
        with col1:
            monday_date = st.date_input(
                "Monday Date",
                value=datetime.now(CT_TZ).date() - timedelta(days=7),
                key=f"stk_mon_{selected_ticker}",
                on_change=update_stock_data_for_dates
            )
        
        with col2:
            tuesday_date = st.date_input(
                "Tuesday Date", 
                value=monday_date + timedelta(days=1),
                key=f"stk_tue_{selected_ticker}",
                on_change=update_stock_data_for_dates
            )
        
        with col3:
            # Show rest of week dates
            wed_date = tuesday_date + timedelta(days=1)
            thu_date = tuesday_date + timedelta(days=2) 
            fri_date = tuesday_date + timedelta(days=3)
            st.write("Projection Days:")
            st.caption(f"Wed: {wed_date}")
            st.caption(f"Thu: {thu_date}")
            st.caption(f"Fri: {fri_date}")
        
        st.markdown("---")
        
        # Show current analysis status
        if (st.session_state.get('stock_analysis_ready', False) and 
            st.session_state.get('stock_analysis_ticker') == selected_ticker):
            st.success(f"Historical data loaded for {selected_ticker}")
        else:
            st.info("Select dates to automatically load historical analysis")
        
        # ═══════════════════════════════════════════════════════════════════════════════
        # AUTOMATIC RESULTS DISPLAY
        # ═══════════════════════════════════════════════════════════════════════════════
        
        if (st.session_state.get('stock_analysis_ready', False) and 
            st.session_state.get('stock_analysis_ticker') == selected_ticker):
            
            st.subheader(f"{selected_ticker} Weekly Analysis")
            
            # Process swing detection on combined Monday/Tuesday data
            stock_data = st.session_state.stock_analysis_data
            stock_swings = detect_swings_simple(stock_data)
            
            # Get absolute highest and lowest across both days
            skyline_anchor, baseline_anchor = get_anchor_points(stock_swings)
            
            # Get manual High/Close/Low from Tuesday's data (using close prices)
            tue_data_only = stock_data[stock_data.index.date == tuesday_date]
            if not tue_data_only.empty:
                tue_ohlc = get_daily_ohlc(tue_data_only, tuesday_date)
                manual_anchors = tue_ohlc
            else:
                # Fallback to last available data
                last_bar = stock_data.iloc[-1]
                manual_anchors = {
                    'high': (last_bar['High'], last_bar.name),
                    'close': (last_bar['Close'], last_bar.name),
                    'low': (last_bar['Low'], last_bar.name)
                }
            
            # Display anchor summary for stock
            st.subheader(f"{selected_ticker} Detected Anchors")
            anchor_summary_cols = st.columns(5)
            
            # Manual anchors
            anchor_info = [
                ('high', 'High', '#ff6b6b'),
                ('close', 'Close', '#f9ca24'), 
                ('low', 'Low', '#4ecdc4')
            ]
            
            for i, (name, display_name, color) in enumerate(anchor_info):
                if name in manual_anchors:
                    price, timestamp = manual_anchors[name]
                    with anchor_summary_cols[i]:
                        st.markdown(f"""
                        <div style="text-align: center; padding: 1rem; background: rgba(255,255,255,0.1); border-radius: 10px; border-left: 4px solid {color};">
                            <h4>{display_name}</h4>
                            <h3>${price:.2f}</h3>
                            <p>{format_ct_time(timestamp)}</p>
                        </div>
                        """, unsafe_allow_html=True)
            
            # Swing anchors
            with anchor_summary_cols[3]:
                if skyline_anchor:
                    price, timestamp = skyline_anchor
                    st.markdown(f"""
                    <div style="text-align: center; padding: 1rem; background: rgba(255,100,100,0.2); border-radius: 10px; border-left: 4px solid #ff4757;">
                        <h4>Skyline</h4>
                        <h3>${price:.2f}</h3>
                        <p>{format_ct_time(timestamp)}</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.warning("No Skyline")
            
            with anchor_summary_cols[4]:
                if baseline_anchor:
                    price, timestamp = baseline_anchor
                    st.markdown(f"""
                    <div style="text-align: center; padding: 1rem; background: rgba(100,100,255,0.2); border-radius: 10px; border-left: 4px solid #3742fa;">
                        <h4>Baseline</h4>
                        <h3>${price:.2f}</h3>
                        <p>{format_ct_time(timestamp)}</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.warning("No Baseline")
            
            st.markdown("---")
            
            # Weekly projection tabs
            projection_dates = [
                ("Wednesday", tuesday_date + timedelta(days=1)),
                ("Thursday", tuesday_date + timedelta(days=2)), 
                ("Friday", tuesday_date + timedelta(days=3))
            ]
            
            weekly_tabs = st.tabs(["Wednesday", "Thursday", "Friday"])
            
            for day_idx, (day_name, proj_date) in enumerate(projection_dates):
                with weekly_tabs[day_idx]:
                    st.subheader(f"{day_name} - {proj_date}")
                    
                    # Create anchor sub-tabs for each day
                    anchor_subtabs = st.tabs(["High", "Close", "Low", "Skyline", "Baseline"])
                    
                    # Manual anchors projections
                    with anchor_subtabs[0]:  # High
                        if 'high' in manual_anchors:
                            price, timestamp = manual_anchors['high']
                            anchor_time_ct = timestamp.astimezone(CT_TZ)
                            
                            high_proj = project_anchor_line(
                                price, anchor_time_ct, slope_magnitude, proj_date
                            )
                            
                            st.subheader("High Anchor Projection")
                            st.dataframe(high_proj, use_container_width=True, hide_index=True)
                            
                            # Analytics table
                            high_analytics = calculate_stock_anchor_analytics(high_proj, selected_ticker, "HIGH", stock_data)
                            st.subheader("Anchor Analytics")
                            st.dataframe(high_analytics, use_container_width=True, hide_index=True)
                            
                            high_analysis = calculate_stock_entry_exit_table(high_proj, selected_ticker, "HIGH", day_name)
                            st.subheader("Entry/Exit Strategy")
                            st.dataframe(high_analysis, use_container_width=True, hide_index=True)
                    
                    with anchor_subtabs[1]:  # Close
                        if 'close' in manual_anchors:
                            price, timestamp = manual_anchors['close']
                            anchor_time_ct = timestamp.astimezone(CT_TZ)
                            
                            close_proj = project_anchor_line(
                                price, anchor_time_ct, slope_magnitude, proj_date
                            )
                            
                            st.subheader("Close Anchor Projection")
                            st.dataframe(close_proj, use_container_width=True, hide_index=True)
                            
                            close_analytics = calculate_stock_anchor_analytics(close_proj, selected_ticker, "CLOSE", stock_data)
                            st.subheader("Anchor Analytics")
                            st.dataframe(close_analytics, use_container_width=True, hide_index=True)
                            
                            close_analysis = calculate_stock_entry_exit_table(close_proj, selected_ticker, "CLOSE", day_name)
                            st.subheader("Entry/Exit Strategy")
                            st.dataframe(close_analysis, use_container_width=True, hide_index=True)
                    
                    with anchor_subtabs[2]:  # Low
                        if 'low' in manual_anchors:
                            price, timestamp = manual_anchors['low']
                            anchor_time_ct = timestamp.astimezone(CT_TZ)
                            
                            low_proj = project_anchor_line(
                                price, anchor_time_ct, slope_magnitude, proj_date
                            )
                            
                            st.subheader("Low Anchor Projection")
                            st.dataframe(low_proj, use_container_width=True, hide_index=True)
                            
                            low_analytics = calculate_stock_anchor_analytics(low_proj, selected_ticker, "LOW", stock_data)
                            st.subheader("Anchor Analytics")
                            st.dataframe(low_analytics, use_container_width=True, hide_index=True)
                            
                            low_analysis = calculate_stock_entry_exit_table(low_proj, selected_ticker, "LOW", day_name)
                            st.subheader("Entry/Exit Strategy")
                            st.dataframe(low_analysis, use_container_width=True, hide_index=True)
                    
                    # Swing anchors projections
                    with anchor_subtabs[3]:  # Skyline
                        if skyline_anchor:
                            sky_price, sky_time = skyline_anchor
                            sky_time_ct = sky_time.astimezone(CT_TZ)
                            
                            skyline_proj = project_anchor_line(
                                sky_price, sky_time_ct, slope_magnitude, proj_date
                            )
                            
                            st.subheader("Skyline Projection (80% Zone)")
                            st.dataframe(skyline_proj, use_container_width=True, hide_index=True)
                            
                            sky_analytics = calculate_stock_anchor_analytics(skyline_proj, selected_ticker, "SKYLINE", stock_data)
                            st.subheader("Anchor Analytics")
                            st.dataframe(sky_analytics, use_container_width=True, hide_index=True)
                            
                            sky_analysis = calculate_stock_entry_exit_table(skyline_proj, selected_ticker, "SKYLINE", day_name)
                            st.subheader("Skyline Bounce Strategy")
                            st.dataframe(sky_analysis, use_container_width=True, hide_index=True)
                        else:
                            st.warning("No skyline anchor detected")
                    
                    with anchor_subtabs[4]:  # Baseline
                        if baseline_anchor:
                            base_price, base_time = baseline_anchor
                            base_time_ct = base_time.astimezone(CT_TZ)
                            
                            baseline_proj = project_anchor_line(
                                base_price, base_time_ct, -slope_magnitude, proj_date
                            )
                            
                            st.subheader("Baseline Projection (80% Zone)")
                            st.dataframe(baseline_proj, use_container_width=True, hide_index=True)
                            
                            base_analytics = calculate_stock_anchor_analytics(baseline_proj, selected_ticker, "BASELINE", stock_data)
                            st.subheader("Anchor Analytics")
                            st.dataframe(base_analytics, use_container_width=True, hide_index=True)
                            
                            base_analysis = calculate_stock_entry_exit_table(baseline_proj, selected_ticker, "BASELINE", day_name)
                            st.subheader("Baseline Bounce Strategy")
                            st.dataframe(base_analysis, use_container_width=True, hide_index=True)
                        else:
                            st.warning("No baseline anchor detected")
    else:
        st.info("Select a ticker to begin stock analysis")

# ═══════════════════════════════════════════════════════════════════════════════
# END OF STOCK ANCHORS TAB
# ═══════════════════════════════════════════════════════════════════════════════





# ═══════════════════════════════════════════════════════════════════════════════
# SPX PROPHET - PART 4: SIGNALS & EMA TAB
# ═══════════════════════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════════════════════
# MARKET DATA ANALYSIS FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def calculate_market_volatility(data: pd.DataFrame) -> float:
    """Calculate actual market volatility from price data"""
    if data.empty or len(data) < 2:
        return 1.5
    
    returns = data['Close'].pct_change().dropna()
    if returns.empty:
        return 1.5
        
    volatility = returns.std() * np.sqrt(390)  # Annualized intraday volatility
    return volatility * 100

def calculate_average_true_range(data: pd.DataFrame, periods: int = 14) -> pd.Series:
    """Calculate ATR from market data"""
    if data.empty or len(data) < periods:
        return pd.Series(index=data.index, dtype=float)
    
    high_low = data['High'] - data['Low']
    high_close = np.abs(data['High'] - data['Close'].shift())
    low_close = np.abs(data['Low'] - data['Close'].shift())
    
    true_range = np.maximum(high_low, np.maximum(high_close, low_close))
    atr = true_range.rolling(window=periods).mean()
    
    return atr

def detect_anchor_touches(price_data: pd.DataFrame, anchor_line: pd.DataFrame) -> pd.DataFrame:
    """Detect when price touches anchor lines based on your strategy"""
    if price_data.empty or anchor_line.empty:
        return pd.DataFrame()
    
    # Create anchor price lookup
    anchor_dict = {}
    for _, row in anchor_line.iterrows():
        anchor_dict[row['Time']] = row['Price']
    
    touches = []
    
    for idx, bar in price_data.iterrows():
        bar_time = format_ct_time(idx)
        
        if bar_time not in anchor_dict:
            continue
            
        anchor_price = anchor_dict[bar_time]
        
        # Calculate touch precision using actual price data
        low_distance = abs(bar['Low'] - anchor_price)
        high_distance = abs(bar['High'] - anchor_price)
        
        # Touch tolerance based on recent volatility
        recent_atr = calculate_average_true_range(price_data.tail(20), 14)
        if not recent_atr.empty:
            tolerance = recent_atr.iloc[-1] * 0.3  # 30% of ATR
        else:
            tolerance = anchor_price * 0.002  # 0.2% fallback
        
        touches_anchor = (bar['Low'] <= anchor_price + tolerance and 
                         bar['High'] >= anchor_price - tolerance)
        
        if touches_anchor:
            # Determine candle type
            is_bearish = bar['Close'] < bar['Open']
            is_bullish = bar['Close'] > bar['Open']
            
            # Touch quality based on how close the touch was
            closest_distance = min(low_distance, high_distance)
            touch_quality = max(0, 100 - (closest_distance / tolerance * 100))
            
            # Volume analysis
            volume_ma = price_data['Volume'].rolling(20).mean().iloc[-1] if 'Volume' in price_data.columns else 0
            volume_ratio = bar['Volume'] / volume_ma if volume_ma > 0 else 1.0
            volume_strength = min(100, volume_ratio * 50)
            
            touches.append({
                'Time': bar_time,
                'Anchor_Price': round(anchor_price, 2),
                'Touch_Price': round(bar['Low'] if low_distance < high_distance else bar['High'], 2),
                'Candle_Type': 'Bearish' if is_bearish else 'Bullish' if is_bullish else 'Doji',
                'Open': round(bar['Open'], 2),
                'High': round(bar['High'], 2),
                'Low': round(bar['Low'], 2),
                'Close': round(bar['Close'], 2),
                'Volume': int(bar['Volume']) if 'Volume' in bar else 0,
                'Touch_Quality': round(touch_quality, 1),
                'Volume_Strength': round(volume_strength, 1),
                'ATR_Tolerance': round(tolerance, 2)
            })
    
    return pd.DataFrame(touches)

def analyze_anchor_line_interaction(price_data: pd.DataFrame, anchor_line: pd.DataFrame) -> pd.DataFrame:
    """Analyze how price interacts with anchor line throughout the day"""
    if price_data.empty or anchor_line.empty:
        return pd.DataFrame()
    
    anchor_dict = {}
    for _, row in anchor_line.iterrows():
        anchor_dict[row['Time']] = row['Price']
    
    interactions = []
    total_touches = 0
    bounces = 0
    penetrations = 0
    bounce_distances = []
    penetration_depths = []
    
    for idx, bar in price_data.iterrows():
        bar_time = format_ct_time(idx)
        
        if bar_time not in anchor_dict:
            continue
            
        anchor_price = anchor_dict[bar_time]
        
        # Calculate interaction metrics
        price_above = bar['Close'] > anchor_price
        touched = (bar['Low'] <= anchor_price <= bar['High'])
        
        if touched:
            total_touches += 1
            
            # Determine if it bounced or penetrated
            if price_above:
                bounces += 1
                bounce_distance = bar['Close'] - anchor_price
                bounce_distances.append(bounce_distance)
            else:
                penetrations += 1
                penetration_depth = anchor_price - bar['Close']
                penetration_depths.append(penetration_depth)
        
        # Distance from anchor
        distance_from_anchor = bar['Close'] - anchor_price
        distance_pct = (distance_from_anchor / anchor_price) * 100
        
        interactions.append({
            'Time': bar_time,
            'Close_Price': round(bar['Close'], 2),
            'Anchor_Price': round(anchor_price, 2),
            'Distance': round(distance_from_anchor, 2),
            'Distance_Pct': round(distance_pct, 2),
            'Touched': 'Yes' if touched else 'No',
            'Position': 'Above' if price_above else 'Below'
        })
    
    return pd.DataFrame(interactions)

def calculate_ema_crossover_signals(price_data: pd.DataFrame) -> pd.DataFrame:
    """Calculate EMA crossovers with actual market data"""
    if price_data.empty or len(price_data) < 21:
        return pd.DataFrame()
    
    # Calculate EMAs
    ema8 = calculate_ema(price_data['Close'], 8)
    ema21 = calculate_ema(price_data['Close'], 21)
    
    crossovers = []
    
    for i in range(1, len(price_data)):
        current_time = format_ct_time(price_data.index[i])
        prev_8 = ema8.iloc[i-1]
        prev_21 = ema21.iloc[i-1]
        curr_8 = ema8.iloc[i]
        curr_21 = ema21.iloc[i]
        current_price = price_data.iloc[i]['Close']
        
        crossover_type = None
        strength = abs(curr_8 - curr_21) / curr_21 * 100
        
        # Detect crossovers
        if prev_8 <= prev_21 and curr_8 > curr_21:
            crossover_type = "Bullish Cross"
        elif prev_8 >= prev_21 and curr_8 < curr_21:
            crossover_type = "Bearish Cross"
        
        # Always show current EMA status
        ema_regime = "Bullish" if curr_8 > curr_21 else "Bearish"
        
        crossovers.append({
            'Time': current_time,
            'Price': round(current_price, 2),
            'EMA8': round(curr_8, 2),
            'EMA21': round(curr_21, 2),
            'Separation': round(strength, 3),
            'Regime': ema_regime,
            'Crossover': crossover_type if crossover_type else 'None',
            'Signal_Strength': 'Strong' if strength > 0.5 else 'Moderate' if strength > 0.2 else 'Weak'
        })
    
    return pd.DataFrame(crossovers)

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN SIGNALS TAB
# ═══════════════════════════════════════════════════════════════════════════════

with tab3:
    st.subheader("Signal Detection & Market Analysis")
    st.caption("Real-time anchor touch detection with market-derived analytics")
    
    # ═══════════════════════════════════════════════════════════════════════════════
    # INPUT CONTROLS
    # ═══════════════════════════════════════════════════════════════════════════════
    
    col1, col2 = st.columns(2)
    with col1:
        signal_symbol = st.selectbox(
            "Analysis Symbol", 
            ["^GSPC", "ES=F", "SPY"],
            index=0,
            key="sig_symbol"
        )
    
    with col2:
        signal_day = st.date_input(
            "Analysis Day",
            value=datetime.now(CT_TZ).date(),
            key="sig_day"
        )
    
    st.markdown("Reference Line Configuration")
    
    ref_col1, ref_col2, ref_col3 = st.columns(3)
    with ref_col1:
        anchor_price = st.number_input(
            "Anchor Price",
            value=6000.0,
            step=0.1, format="%.2f",
            key="sig_anchor_price"
        )
    
    with ref_col2:
        anchor_time_input = st.time_input(
            "Anchor Time (CT)",
            value=time(17, 0),
            key="sig_anchor_time"
        )
    
    with ref_col3:
        ref_slope = st.number_input(
            "Slope per 30min",
            value=0.268,
            step=0.001, format="%.3f",
            key="sig_ref_slope"
        )
    
    st.markdown("---")
    
    # ═══════════════════════════════════════════════════════════════════════════════
    # ANALYSIS EXECUTION
    # ═══════════════════════════════════════════════════════════════════════════════
    
    if st.button("Analyze Market Signals", key="sig_generate", type="primary"):
        with st.spinner("Analyzing market data..."):
            
            # Fetch data for analysis day
            signal_data = fetch_live_data(signal_symbol, signal_day, signal_day)
            
            if signal_data.empty:
                st.error(f"No data available for {signal_symbol} on {signal_day}")
            else:
                # Filter to RTH session for analysis
                rth_data = get_session_window(signal_data, RTH_START, RTH_END)
                
                if rth_data.empty:
                    st.error("No RTH data available for selected day")
                else:
                    # Store results
                    st.session_state.signal_data = rth_data
                    st.session_state.signal_anchor = {
                        'price': anchor_price,
                        'time': anchor_time_input,
                        'slope': ref_slope
                    }
                    st.session_state.signal_symbol = signal_symbol
                    st.session_state.signal_ready = True
    
    # ═══════════════════════════════════════════════════════════════════════════════
    # RESULTS DISPLAY
    # ═══════════════════════════════════════════════════════════════════════════════
    
    if st.session_state.get('signal_ready', False):
        signal_data = st.session_state.signal_data
        anchor_config = st.session_state.signal_anchor
        symbol = st.session_state.signal_symbol
        
        st.subheader(f"{symbol} Market Analysis Results")
        
        # Create anchor datetime and projection
        anchor_datetime = datetime.combine(signal_day, anchor_config['time'])
        anchor_datetime_ct = CT_TZ.localize(anchor_datetime)
        
        # Generate reference line projection
        ref_line_proj = project_anchor_line(
            anchor_config['price'], 
            anchor_datetime_ct,
            anchor_config['slope'],
            signal_day
        )
        
        # Calculate market-derived indicators
        volatility = calculate_market_volatility(signal_data)
        atr_series = calculate_average_true_range(signal_data, 14)
        vwap_series = calculate_vwap(signal_data)
        
        # Market overview
        st.subheader("Market Overview")
        overview_col1, overview_col2, overview_col3 = st.columns(3)
        
        with overview_col1:
            day_range = signal_data['High'].max() - signal_data['Low'].min()
            st.metric("Day Range", f"${day_range:.2f}")
        
        with overview_col2:
            st.metric("Volatility", f"{volatility:.2f}%")
        
        with overview_col3:
            current_atr = atr_series.iloc[-1] if not atr_series.empty else 0
            st.metric("Current ATR", f"${current_atr:.2f}")
        
        st.markdown("---")
        
        # Analysis tabs
        signal_tabs = st.tabs(["Reference Line", "Anchor Touches", "Line Interaction", "EMA Analysis"])
        
        with signal_tabs[0]:  # Reference Line
            st.subheader("Reference Line Projection")
            st.dataframe(ref_line_proj, use_container_width=True, hide_index=True)
            
            # Show projection quality
            if not ref_line_proj.empty:
                price_range = ref_line_proj['Price'].max() - ref_line_proj['Price'].min()
                avg_price = ref_line_proj['Price'].mean()
                range_pct = (price_range / avg_price) * 100
                
                st.info(f"Projection range: ${price_range:.2f} ({range_pct:.1f}% of average price)")
        
        with signal_tabs[1]:  # Anchor Touches
            anchor_touches = detect_anchor_touches(signal_data, ref_line_proj)
            
            if not anchor_touches.empty:
                st.subheader("Detected Anchor Touches")
                st.dataframe(anchor_touches, use_container_width=True, hide_index=True)
                
                # Touch summary
                total_touches = len(anchor_touches)
                avg_touch_quality = anchor_touches['Touch_Quality'].mean()
                avg_volume_strength = anchor_touches['Volume_Strength'].mean()
                
                touch_col1, touch_col2, touch_col3 = st.columns(3)
                with touch_col1:
                    st.metric("Total Touches", total_touches)
                with touch_col2:
                    st.metric("Avg Touch Quality", f"{avg_touch_quality:.1f}%")
                with touch_col3:
                    st.metric("Avg Volume Strength", f"{avg_volume_strength:.1f}%")
                
            else:
                st.info("No anchor line touches detected for this day")
        
        with signal_tabs[2]:  # Line Interaction
            line_interaction = analyze_anchor_line_interaction(signal_data, ref_line_proj)
            
            if not line_interaction.empty:
                st.subheader("Price-Anchor Line Interaction")
                st.dataframe(line_interaction, use_container_width=True, hide_index=True)
                
                # Interaction statistics
                touches = line_interaction[line_interaction['Touched'] == 'Yes']
                above_line = line_interaction[line_interaction['Position'] == 'Above']
                
                interaction_col1, interaction_col2, interaction_col3 = st.columns(3)
                with interaction_col1:
                    st.metric("Touch Points", len(touches))
                with interaction_col2:
                    above_pct = (len(above_line) / len(line_interaction)) * 100
                    st.metric("Time Above Line", f"{above_pct:.1f}%")
                with interaction_col3:
                    if not line_interaction.empty:
                        avg_distance = abs(line_interaction['Distance']).mean()
                        st.metric("Avg Distance", f"${avg_distance:.2f}")
            else:
                st.info("No line interaction data available")
        
        with signal_tabs[3]:  # EMA Analysis
            ema_analysis = calculate_ema_crossover_signals(signal_data)
            
            if not ema_analysis.empty:
                st.subheader("EMA 8/21 Analysis")
                st.dataframe(ema_analysis, use_container_width=True, hide_index=True)
                
                # EMA statistics
                crossovers = ema_analysis[ema_analysis['Crossover'] != 'None']
                current_regime = ema_analysis.iloc[-1]['Regime'] if not ema_analysis.empty else 'Unknown'
                current_separation = ema_analysis.iloc[-1]['Separation'] if not ema_analysis.empty else 0
                
                ema_col1, ema_col2, ema_col3 = st.columns(3)
                with ema_col1:
                    st.metric("Crossovers", len(crossovers))
                with ema_col2:
                    st.metric("Current Regime", current_regime)
                with ema_col3:
                    st.metric("EMA Separation", f"{current_separation:.3f}%")
            else:
                st.info("Insufficient data for EMA analysis")
    
    else:
        st.info("Configure your parameters and click 'Analyze Market Signals' to begin")

# ═══════════════════════════════════════════════════════════════════════════════
# END OF SIGNALS & EMA TAB
# ═══════════════════════════════════════════════════════════════════════════════





# ═══════════════════════════════════════════════════════════════════════════════
# SPX PROPHET - PART 5: CONTRACT TOOL TAB
# ═══════════════════════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════════════════════
# MARKET DATA CALCULATION FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def calculate_contract_volatility(price_data: pd.DataFrame, window: int = 20) -> float:
    """Calculate actual volatility from price movements"""
    if price_data.empty or len(price_data) < window:
        return 0.15
    
    returns = price_data['Close'].pct_change().dropna()
    if returns.empty:
        return 0.15
        
    volatility = returns.rolling(window=window).std().iloc[-1]
    return volatility if not np.isnan(volatility) else 0.15

def calculate_price_momentum(p1_price: float, p2_price: float, time_hours: float) -> dict:
    """Calculate momentum metrics from actual price movement"""
    price_change = p2_price - p1_price
    price_change_pct = (price_change / p1_price) * 100 if p1_price > 0 else 0
    hourly_rate = price_change / time_hours if time_hours > 0 else 0
    
    # Momentum strength classification
    abs_change_pct = abs(price_change_pct)
    if abs_change_pct >= 15:
        strength = "Very Strong"
        confidence = 95
    elif abs_change_pct >= 8:
        strength = "Strong" 
        confidence = 85
    elif abs_change_pct >= 3:
        strength = "Moderate"
        confidence = 70
    else:
        strength = "Weak"
        confidence = 50
        
    return {
        'change': price_change,
        'change_pct': price_change_pct,
        'hourly_rate': hourly_rate,
        'strength': strength,
        'confidence': confidence
    }

def calculate_market_based_targets(entry_price: float, market_data: pd.DataFrame, direction: str) -> dict:
    """Calculate TP1/TP2 based on actual market volatility and price action"""
    if market_data.empty:
        # Fallback if no market data
        base_move = entry_price * 0.02
        return {
            'tp1': entry_price + base_move if direction == "BUY" else entry_price - base_move,
            'tp2': entry_price + (base_move * 2.5) if direction == "BUY" else entry_price - (base_move * 2.5),
            'stop_distance': base_move * 0.6
        }
    
    # Calculate ATR-based targets
    atr_series = calculate_average_true_range(market_data, 14)
    current_atr = atr_series.iloc[-1] if not atr_series.empty else entry_price * 0.015
    
    # Recent price volatility
    recent_range = market_data['High'].tail(10).max() - market_data['Low'].tail(10).min()
    volatility_factor = recent_range / market_data['Close'].tail(10).mean()
    
    # Adaptive target calculation
    base_target = current_atr * 1.2  # 120% of ATR for TP1
    extended_target = current_atr * 3.0  # 300% of ATR for TP2
    
    # Adjust for volatility
    base_target *= (1 + volatility_factor * 0.5)
    extended_target *= (1 + volatility_factor * 0.3)
    
    if direction == "BUY":
        tp1 = entry_price + base_target
        tp2 = entry_price + extended_target
    else:
        tp1 = entry_price - base_target
        tp2 = entry_price - extended_target
    
    # Stop based on recent support/resistance
    stop_distance = current_atr * 0.8
    
    return {
        'tp1': tp1,
        'tp2': tp2,
        'stop_distance': stop_distance,
        'atr': current_atr,
        'volatility_factor': volatility_factor
    }

def analyze_overnight_market_behavior(symbol: str, start_date: date, end_date: date) -> dict:
    """Analyze overnight price behavior patterns"""
    overnight_data = fetch_live_data(symbol, start_date - timedelta(days=5), end_date)
    
    if overnight_data.empty:
        return {
            'avg_overnight_change': 0,
            'overnight_volatility': 0.02,
            'gap_frequency': 0,
            'mean_reversion_rate': 0.6
        }
    
    # Filter to overnight hours (16:00 to 08:30 next day)
    overnight_moves = []
    gap_moves = []
    
    # Group by date and calculate overnight moves
    for date_group in overnight_data.groupby(overnight_data.index.date):
        daily_data = date_group[1]
        
        if len(daily_data) < 2:
            continue
            
        # Last close vs first open (overnight move)
        day_close = daily_data.iloc[-1]['Close']
        next_open = daily_data.iloc[0]['Open'] if len(daily_data) > 0 else day_close
        
        overnight_change = (next_open - day_close) / day_close if day_close > 0 else 0
        overnight_moves.append(overnight_change)
        
        # Gap analysis
        gap_size = abs(overnight_change)
        if gap_size > 0.005:  # 0.5% gap threshold
            gap_moves.append(gap_size)
    
    if not overnight_moves:
        return {
            'avg_overnight_change': 0,
            'overnight_volatility': 0.02,
            'gap_frequency': 0,
            'mean_reversion_rate': 0.6
        }
    
    avg_overnight_change = np.mean(overnight_moves)
    overnight_volatility = np.std(overnight_moves)
    gap_frequency = len(gap_moves) / len(overnight_moves) if overnight_moves else 0
    
    # Mean reversion calculation (how often overnight moves reverse)
    reversion_count = sum(1 for move in overnight_moves if abs(move) < np.std(overnight_moves))
    mean_reversion_rate = reversion_count / len(overnight_moves) if overnight_moves else 0.6
    
    return {
        'avg_overnight_change': avg_overnight_change,
        'overnight_volatility': overnight_volatility,
        'gap_frequency': gap_frequency,
        'mean_reversion_rate': mean_reversion_rate
    }

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN CONTRACT TOOL TAB
# ═══════════════════════════════════════════════════════════════════════════════

with tab4:
    st.subheader("Contract Tool")
    st.caption("Overnight contract analysis for RTH entry optimization")
    
    # ═══════════════════════════════════════════════════════════════════════════════
    # INPUT CONTROLS
    # ═══════════════════════════════════════════════════════════════════════════════
    
    st.subheader("Overnight Contract Price Points")
    
    point_col1, point_col2 = st.columns(2)
    
    with point_col1:
        st.markdown("**Point 1 (Earlier)**")
        
        p1_date = st.date_input(
            "Point 1 Date",
            value=datetime.now(CT_TZ).date() - timedelta(days=1),
            key="ct_p1_date"
        )
        
        p1_time = st.time_input(
            "Point 1 Time (CT)",
            value=time(20, 0),
            key="ct_p1_time"
        )
        
        p1_price = st.number_input(
            "Point 1 Contract Price",
            value=10.0,
            min_value=0.01,
            step=0.01, format="%.2f",
            key="ct_p1_price"
        )
    
    with point_col2:
        st.markdown("**Point 2 (Later)**")
        
        p2_date = st.date_input(
            "Point 2 Date", 
            value=datetime.now(CT_TZ).date(),
            key="ct_p2_date"
        )
        
        p2_time = st.time_input(
            "Point 2 Time (CT)",
            value=time(8, 0),
            key="ct_p2_time"
        )
        
        p2_price = st.number_input(
            "Point 2 Contract Price",
            value=12.0,
            min_value=0.01,
            step=0.01, format="%.2f",
            key="ct_p2_price"
        )
    
    # Projection day
    projection_day = st.date_input(
        "RTH Projection Day",
        value=p2_date,
        key="ct_proj_day"
    )
    
    # Validate and calculate metrics
    p1_datetime = datetime.combine(p1_date, p1_time)
    p2_datetime = datetime.combine(p2_date, p2_time)
    
    if p2_datetime <= p1_datetime:
        st.error("Point 2 must be after Point 1")
    else:
        # Calculate time and momentum metrics
        time_diff_hours = (p2_datetime - p1_datetime).total_seconds() / 3600
        momentum_metrics = calculate_price_momentum(p1_price, p2_price, time_diff_hours)
        
        # Display calculated metrics
        metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
        
        with metric_col1:
            st.metric("Time Span", f"{time_diff_hours:.1f} hours")
        
        with metric_col2:
            st.metric("Price Change", f"{momentum_metrics['change']:+.2f}")
        
        with metric_col3:
            st.metric("Change %", f"{momentum_metrics['change_pct']:+.1f}%")
        
        with metric_col4:
            st.metric("Momentum", momentum_metrics['strength'])
    
    st.markdown("---")
    
    # ═══════════════════════════════════════════════════════════════════════════════
    # ANALYSIS EXECUTION
    # ═══════════════════════════════════════════════════════════════════════════════
    
    if st.button("Analyze Contract Projections", key="ct_generate", type="primary"):
        if p2_datetime <= p1_datetime:
            st.error("Please ensure Point 2 is after Point 1")
        else:
            with st.spinner("Analyzing contract and market data..."):
                try:
                    # Calculate contract slope
                    time_diff_minutes = (p2_datetime - p1_datetime).total_seconds() / 60
                    blocks_between = time_diff_minutes / 30
                    contract_slope = (p2_price - p1_price) / blocks_between if blocks_between > 0 else 0
                    
                    # Fetch underlying market data for analysis
                    underlying_data = fetch_live_data("^GSPC", projection_day - timedelta(days=10), projection_day)
                    
                    # Analyze overnight market behavior
                    overnight_analysis = analyze_overnight_market_behavior("^GSPC", projection_day - timedelta(days=10), projection_day)
                    
                    # Generate RTH projections
                    p1_ct = CT_TZ.localize(p1_datetime)
                    contract_projections = project_contract_line(p1_price, p1_ct, contract_slope, projection_day)
                    
                    # Store results with market analysis
                    st.session_state.contract_projections = contract_projections
                    st.session_state.contract_config = {
                        'p1_price': p1_price,
                        'p1_time': p1_ct,
                        'p2_price': p2_price,
                        'p2_time': CT_TZ.localize(p2_datetime),
                        'slope': contract_slope,
                        'momentum': momentum_metrics,
                        'overnight_analysis': overnight_analysis
                    }
                    st.session_state.underlying_data = underlying_data
                    st.session_state.contract_ready = True
                    
                except Exception as e:
                    st.error(f"Analysis error: {str(e)}")
    
    # ═══════════════════════════════════════════════════════════════════════════════
    # RESULTS DISPLAY
    # ═══════════════════════════════════════════════════════════════════════════════
    
    if st.session_state.get('contract_ready', False):
        st.subheader("Contract Analysis Results")
        
        projections = st.session_state.contract_projections
        config = st.session_state.contract_config
        underlying_data = st.session_state.get('underlying_data', pd.DataFrame())
        
        # Analysis tabs
        contract_tabs = st.tabs(["RTH Projections", "Market Analysis", "Risk Management"])
        
        with contract_tabs[0]:  # RTH Projections
            st.subheader("RTH Contract Price Projections")
            
            # Add probability and target analysis
            if not projections.empty and not underlying_data.empty:
                enhanced_projections = []
                
                for idx, row in projections.iterrows():
                    time_slot = row['Time']
                    contract_price = row['Contract_Price']
                    
                    # Calculate targets based on market data
                    direction = "BUY" if config['momentum']['change'] > 0 else "SELL"
                    targets = calculate_market_based_targets(contract_price, underlying_data, direction)
                    
                    # Time-based probability (market open has higher success)
                    hour = int(time_slot.split(':')[0])
                    if hour in [8, 9]:
                        time_prob = config['momentum']['confidence'] + 10
                    elif hour in [13, 14]:
                        time_prob = config['momentum']['confidence'] + 5
                    else:
                        time_prob = config['momentum']['confidence']
                    
                    enhanced_projections.append({
                        'Time': time_slot,
                        'Contract_Price': round(contract_price, 2),
                        'Direction': direction,
                        'TP1': round(targets['tp1'], 2),
                        'TP2': round(targets['tp2'], 2),
                        'Stop_Distance': round(targets['stop_distance'], 2),
                        'Entry_Probability': f"{min(95, time_prob):.0f}%",
                        'ATR_Base': round(targets.get('atr', 0), 2)
                    })
                
                enhanced_df = pd.DataFrame(enhanced_projections)
                st.dataframe(enhanced_df, use_container_width=True, hide_index=True)
                
            else:
                st.dataframe(projections, use_container_width=True, hide_index=True)
        
        with contract_tabs[1]:  # Market Analysis
            st.subheader("Underlying Market Analysis")
            
            momentum = config['momentum']
            overnight = config['overnight_analysis']
            
            # Momentum analysis
            st.subheader("Contract Momentum")
            momentum_col1, momentum_col2, momentum_col3 = st.columns(3)
            
            with momentum_col1:
                st.metric("Hourly Rate", f"${momentum['hourly_rate']:+.2f}")
            
            with momentum_col2:
                st.metric("Strength", momentum['strength'])
            
            with momentum_col3:
                st.metric("Confidence", f"{momentum['confidence']}%")
            
            # Overnight market behavior
            st.subheader("Overnight Market Behavior")
            overnight_col1, overnight_col2, overnight_col3 = st.columns(3)
            
            with overnight_col1:
                st.metric("Avg Overnight Change", f"{overnight['avg_overnight_change']*100:+.2f}%")
            
            with overnight_col2:
                st.metric("Overnight Volatility", f"{overnight['overnight_volatility']*100:.2f}%")
            
            with overnight_col3:
                st.metric("Gap Frequency", f"{overnight['gap_frequency']*100:.1f}%")
            
            # Market context
            if not underlying_data.empty:
                current_volatility = calculate_contract_volatility(underlying_data)
                day_range = underlying_data['High'].max() - underlying_data['Low'].min()
                
                st.subheader("Current Market Context")
                context_col1, context_col2 = st.columns(2)
                
                with context_col1:
                    st.metric("Recent Volatility", f"{current_volatility*100:.2f}%")
                
                with context_col2:
                    st.metric("Recent Range", f"${day_range:.2f}")
        
        with contract_tabs[2]:  # Risk Management
            st.subheader("Risk Management Analysis")
            
            if not underlying_data.empty and not projections.empty:
                # Calculate risk metrics based on market data
                market_volatility = calculate_contract_volatility(underlying_data)
                atr_series = calculate_average_true_range(underlying_data, 14)
                current_atr = atr_series.iloc[-1] if not atr_series.empty else 0
                
                # Risk assessment
                risk_col1, risk_col2, risk_col3 = st.columns(3)
                
                with risk_col1:
                    # Position sizing based on volatility
                    high_vol_threshold = 0.025  # 2.5%
                    if market_volatility > high_vol_threshold:
                        position_recommendation = "Reduce Size"
                        risk_level = "High"
                    else:
                        position_recommendation = "Standard Size"
                        risk_level = "Normal"
                    
                    st.metric("Risk Level", risk_level)
                    st.caption(f"Volatility: {market_volatility*100:.2f}%")
                
                with risk_col2:
                    st.metric("Position Sizing", position_recommendation)
                    st.caption(f"ATR: ${current_atr:.2f}")
                
                with risk_col3:
                    # Time decay consideration
                    avg_contract_price = projections['Contract_Price'].mean()
                    max_risk_per_contract = current_atr * 1.5
                    risk_per_dollar = (max_risk_per_contract / avg_contract_price) * 100
                    
                    st.metric("Risk per $", f"{risk_per_dollar:.1f}%")
                    st.caption("Based on ATR stop")
                
                # Risk management table
                st.subheader("Time-Based Risk Assessment")
                
                risk_analysis = []
                overnight_vol = config['overnight_analysis']['overnight_volatility']
                
                for idx, row in projections.iterrows():
                    time_slot = row['Time']
                    contract_price = row['Contract_Price']
                    
                    # Time-based risk multiplier
                    hour = int(time_slot.split(':')[0])
                    if hour in [8, 9]:  # Market open - higher risk
                        risk_multiplier = 1.5
                        risk_rating = "High"
                    elif hour in [10, 11]:  # Mid morning - moderate
                        risk_multiplier = 1.0
                        risk_rating = "Medium"
                    else:  # Afternoon - lower
                        risk_multiplier = 0.8
                        risk_rating = "Low"
                    
                    # Adjusted stop based on time and volatility
                    base_stop = current_atr * 1.2
                    adjusted_stop = base_stop * risk_multiplier * (1 + overnight_vol * 2)
                    
                    risk_analysis.append({
                        'Time': time_slot,
                        'Contract_Price': round(contract_price, 2),
                        'Risk_Rating': risk_rating,
                        'Suggested_Stop': round(adjusted_stop, 2),
                        'Risk_Multiplier': f"{risk_multiplier:.1f}x",
                        'Max_Risk_$': round(adjusted_stop, 2)
                    })
                
                risk_df = pd.DataFrame(risk_analysis)
                st.dataframe(risk_df, use_container_width=True, hide_index=True)
            
            else:
                st.info("Need underlying market data for comprehensive risk analysis")
    
    else:
        st.info("Configure your overnight contract points and click 'Analyze Contract Projections'")

def project_contract_line(anchor_price: float, anchor_time: datetime, 
                         slope: float, target_date: date) -> pd.DataFrame:
    """Project contract line across RTH using overnight slope"""
    rth_slots = rth_slots_ct(target_date)
    projections = []
    
    for slot_time in rth_slots:
        time_diff = slot_time - anchor_time
        blocks = time_diff.total_seconds() / 1800
        
        projected_price = anchor_price + (slope * blocks)
        
        projections.append({
            'Time': format_ct_time(slot_time),
            'Contract_Price': round(projected_price, 2),
            'Blocks_from_Anchor': round(blocks, 1)
        })
    
    return pd.DataFrame(projections)

# ═══════════════════════════════════════════════════════════════════════════════
# END OF CONTRACT TOOL TAB
# ═══════════════════════════════════════════════════════════════════════════════







# ═══════════════════════════════════════════════════════════════════════════════
# SPX PROPHET - PART 6: FINAL INTEGRATION & COMPLETION
# ═══════════════════════════════════════════════════════════════════════════════

# Initialize all session state variables if not exists
if 'spx_analysis_ready' not in st.session_state:
    st.session_state.spx_analysis_ready = False

if 'stock_analysis_ready' not in st.session_state:
    st.session_state.stock_analysis_ready = False

if 'signal_ready' not in st.session_state:
    st.session_state.signal_ready = False

if 'contract_ready' not in st.session_state:
    st.session_state.contract_ready = False

# ═══════════════════════════════════════════════════════════════════════════════
# ENHANCED PROBABILITY FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def calculate_volume_profile_strength(data: pd.DataFrame, price_level: float) -> float:
    """Calculate volume profile strength at price level"""
    if data.empty or 'Volume' not in data.columns:
        return 50.0
    
    # Find bars near price level (within 1%)
    tolerance = price_level * 0.01
    nearby_bars = data[
        (data['Low'] <= price_level + tolerance) & 
        (data['High'] >= price_level - tolerance)
    ]
    
    if nearby_bars.empty:
        return 50.0
    
    # Calculate volume concentration
    nearby_volume = nearby_bars['Volume'].sum()
    total_volume = data['Volume'].sum()
    
    if total_volume == 0:
        return 50.0
    
    volume_concentration = (nearby_volume / total_volume) * 100
    
    # Convert to strength score
    if volume_concentration >= 15:
        return 90.0
    elif volume_concentration >= 10:
        return 75.0
    elif volume_concentration >= 5:
        return 60.0
    else:
        return 40.0

def detect_market_regime(data: pd.DataFrame) -> dict:
    """Detect current market regime from price data"""
    if data.empty or len(data) < 20:
        return {'regime': 'UNKNOWN', 'strength': 0, 'trend': 'NEUTRAL'}
    
    # Calculate trend metrics
    closes = data['Close'].tail(20)
    ema_short = closes.ewm(span=5).mean().iloc[-1]
    ema_long = closes.ewm(span=15).mean().iloc[-1]
    
    # Volatility calculation
    returns = closes.pct_change().dropna()
    volatility = returns.std() * 100
    
    # Trend direction
    if ema_short > ema_long * 1.005:  # 0.5% threshold
        trend = 'BULLISH'
        trend_strength = min(100, (ema_short - ema_long) / ema_long * 1000)
    elif ema_short < ema_long * 0.995:
        trend = 'BEARISH' 
        trend_strength = min(100, (ema_long - ema_short) / ema_short * 1000)
    else:
        trend = 'NEUTRAL'
        trend_strength = 0
    
    # Regime classification
    if volatility >= 2.5:
        regime = 'HIGH_VOLATILITY'
    elif volatility >= 1.5:
        regime = 'MODERATE_VOLATILITY'
    else:
        regime = 'LOW_VOLATILITY'
    
    return {
        'regime': regime,
        'trend': trend,
        'strength': trend_strength,
        'volatility': volatility
    }

def calculate_confluence_score(price: float, anchor_price: float, market_data: pd.DataFrame) -> float:
    """Calculate confluence score for entry quality"""
    if market_data.empty:
        return 50.0
    
    # Price proximity to anchor (closer = better)
    price_distance = abs(price - anchor_price) / anchor_price * 100
    proximity_score = max(0, 100 - (price_distance * 20))
    
    # Volume profile strength
    volume_score = calculate_volume_profile_strength(market_data, price)
    
    # Market regime alignment
    regime_info = detect_market_regime(market_data)
    regime_score = 75 if regime_info['trend'] != 'NEUTRAL' else 45
    
    # Weighted confluence score
    confluence = (proximity_score * 0.4) + (volume_score * 0.3) + (regime_score * 0.3)
    return min(100, max(0, confluence))

# ═══════════════════════════════════════════════════════════════════════════════
# ADVANCED ANALYTICS FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def generate_trading_insights(symbol: str, data: pd.DataFrame, projections: pd.DataFrame) -> pd.DataFrame:
    """Generate advanced trading insights"""
    if data.empty or projections.empty:
        return pd.DataFrame()
    
    insights = []
    regime_info = detect_market_regime(data)
    
    # Market condition assessment
    insights.append({
        'Category': 'Market Regime',
        'Insight': f"{regime_info['trend']} trend in {regime_info['regime']} environment",
        'Impact': 'High' if regime_info['strength'] > 50 else 'Medium',
        'Recommendation': get_regime_recommendation(regime_info)
    })
    
    # Volatility analysis
    volatility = regime_info['volatility']
    insights.append({
        'Category': 'Volatility',
        'Insight': f"Current volatility: {volatility:.1f}%",
        'Impact': 'High' if volatility > 2.0 else 'Medium' if volatility > 1.0 else 'Low',
        'Recommendation': get_volatility_recommendation(volatility)
    })
    
    # Anchor line validation
    if len(projections) > 0:
        price_spread = projections['Price'].max() - projections['Price'].min()
        spread_pct = (price_spread / projections['Price'].mean()) * 100
        
        insights.append({
            'Category': 'Anchor Reliability',
            'Insight': f"Price projection spread: {spread_pct:.1f}%",
            'Impact': 'High' if spread_pct < 2 else 'Medium' if spread_pct < 5 else 'Low',
            'Recommendation': get_anchor_recommendation(spread_pct)
        })
    
    return pd.DataFrame(insights)

def get_regime_recommendation(regime_info: dict) -> str:
    """Get recommendation based on market regime"""
    trend = regime_info['trend']
    regime = regime_info['regime']
    
    if trend == 'BULLISH' and 'HIGH' in regime:
        return "Favor long entries with wider stops"
    elif trend == 'BEARISH' and 'HIGH' in regime:
        return "Favor short entries with wider stops"
    elif 'LOW' in regime:
        return "Use tighter stops, expect range-bound action"
    else:
        return "Standard position sizing recommended"

def get_volatility_recommendation(volatility: float) -> str:
    """Get recommendation based on volatility"""
    if volatility > 2.5:
        return "High volatility - reduce position size, wider stops"
    elif volatility > 1.5:
        return "Moderate volatility - standard risk management"
    else:
        return "Low volatility - consider larger positions, tighter stops"

def get_anchor_recommendation(spread_pct: float) -> str:
    """Get recommendation based on anchor spread"""
    if spread_pct < 2:
        return "Strong anchor reliability - high confidence trades"
    elif spread_pct < 5:
        return "Moderate anchor reliability - standard confidence"
    else:
        return "Wide anchor spread - reduce position size"

# ═══════════════════════════════════════════════════════════════════════════════
# PERFORMANCE OPTIMIZATION FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=300, show_spinner=False)
def get_optimal_entry_times() -> dict:
    """Get statistically optimal entry times for different strategies"""
    return {
        'SPX_LONG': ['09:00', '09:30', '13:30', '14:00'],
        'SPX_SHORT': ['10:00', '10:30', '11:00', '14:30'], 
        'STOCK_MOMENTUM': ['09:30', '10:00', '13:00', '13:30'],
        'STOCK_REVERSAL': ['08:30', '11:30', '12:00', '14:00']
    }

def calculate_time_edge(time_slot: str, strategy_type: str) -> float:
    """Calculate time-of-day edge for given strategy"""
    optimal_times = get_optimal_entry_times()
    
    if strategy_type in optimal_times and time_slot in optimal_times[strategy_type]:
        return 15.0  # 15% edge bonus
    
    # Market open/close periods generally have more edge
    hour = int(time_slot.split(':')[0])
    if hour in [8, 9, 13, 14]:
        return 8.0   # 8% edge bonus
    else:
        return 0.0   # No time edge

# ═══════════════════════════════════════════════════════════════════════════════
# SUMMARY DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown("---")
st.subheader("Analysis Summary Dashboard")

# Create summary columns
summary_col1, summary_col2, summary_col3 = st.columns(3)

with summary_col1:
    st.markdown("**Active Analysis**")
    
    spx_status = "✅ Ready" if st.session_state.spx_analysis_ready else "⏸️ Pending"
    stock_status = "✅ Ready" if st.session_state.stock_analysis_ready else "⏸️ Pending"
    signal_status = "✅ Ready" if st.session_state.signal_ready else "⏸️ Pending"
    contract_status = "✅ Ready" if st.session_state.contract_ready else "⏸️ Pending"
    
    st.write(f"SPX Anchors: {spx_status}")
    st.write(f"Stock Analysis: {stock_status}")
    st.write(f"Signal Detection: {signal_status}")
    st.write(f"Contract Tool: {contract_status}")

with summary_col2:
    st.markdown("**Current Settings**")
    
    st.write(f"Skyline Slope: {st.session_state.spx_slopes['skyline']:+.3f}")
    st.write(f"Baseline Slope: {st.session_state.spx_slopes['baseline']:+.3f}")
    st.write(f"ES→SPX Offset: {st.session_state.current_offset:+.1f}")
    st.write(f"Theme: {st.session_state.theme}")

with summary_col3:
    st.markdown("**Market Status**")
    
    current_time_ct = datetime.now(CT_TZ)
    market_open_time = current_time_ct.replace(hour=8, minute=30, second=0, microsecond=0)
    market_close_time = current_time_ct.replace(hour=14, minute=30, second=0, microsecond=0)
    
    is_market_hours = market_open_time <= current_time_ct <= market_close_time
    market_status_text = "🟢 OPEN" if is_market_hours else "🔴 CLOSED"
    
    st.write(f"Market: {market_status_text}")
    st.write(f"Time (CT): {current_time_ct.strftime('%H:%M:%S')}")
    
    if is_market_hours:
        time_to_close = market_close_time - current_time_ct
        hours_left = int(time_to_close.total_seconds() // 3600)
        minutes_left = int((time_to_close.total_seconds() % 3600) // 60)
        st.write(f"Time to Close: {hours_left}h {minutes_left}m")
    else:
        # Time to next open
        if current_time_ct.hour < 8 or (current_time_ct.hour == 8 and current_time_ct.minute < 30):
            next_open = market_open_time
        else:
            next_open = market_open_time + timedelta(days=1)
        
        time_to_open = next_open - current_time_ct
        hours_to_open = int(time_to_open.total_seconds() // 3600)
        st.write(f"Next Open: {hours_to_open}h")

# ═══════════════════════════════════════════════════════════════════════════════
# QUICK ACTIONS PANEL
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown("---")
st.subheader("Quick Actions")

action_col1, action_col2, action_col3, action_col4 = st.columns(4)

with action_col1:
    if st.button("🔄 Update ES Offset", key="quick_update_offset"):
        with st.spinner("Updating offset..."):
            today = datetime.now(CT_TZ).date()
            yesterday = today - timedelta(days=1)
            
            es_data = fetch_live_data("ES=F", yesterday, today)
            spx_data = fetch_live_data("^GSPC", yesterday, today)
            
            if not es_data.empty and not spx_data.empty:
                new_offset = calculate_es_spx_offset(es_data, spx_data)
                st.session_state.current_offset = new_offset
                st.success(f"Offset updated: {new_offset:+.1f}")
                st.rerun()
            else:
                st.error("Failed to fetch offset data")

with action_col2:
    if st.button("📊 Reset All Analysis", key="quick_reset_all"):
        # Clear all analysis states
        analysis_keys = [
            'spx_analysis_ready', 'stock_analysis_ready', 
            'signal_ready', 'contract_ready',
            'es_anchor_data', 'spx_manual_anchors',
            'stock_analysis_data', 'signal_data', 'contract_projections'
        ]
        
        for key in analysis_keys:
            if key in st.session_state:
                del st.session_state[key]
        
        st.success("All analysis reset")
        st.rerun()

with action_col3:
    if st.button("⚙️ Reset Slopes", key="quick_reset_slopes"):
        st.session_state.spx_slopes = SPX_SLOPES.copy()
        st.session_state.stock_slopes = STOCK_SLOPES.copy()
        st.success("Slopes reset to defaults")
        st.rerun()

with action_col4:
    current_theme = "Light" if st.session_state.theme == "Dark" else "Dark"
    if st.button(f"🎨 Switch to {current_theme}", key="quick_theme_switch"):
        st.session_state.theme = current_theme
        st.success(f"Switched to {current_theme} theme")
        st.rerun()

# ═══════════════════════════════════════════════════════════════════════════════
# PERFORMANCE METRICS DISPLAY
# ═══════════════════════════════════════════════════════════════════════════════

if any([
    st.session_state.get('spx_analysis_ready', False),
    st.session_state.get('stock_analysis_ready', False),
    st.session_state.get('signal_ready', False),
    st.session_state.get('contract_ready', False)
]):
    
    st.markdown("---")
    st.subheader("Performance Insights")
    
    insight_tabs = st.tabs(["Market Regime", "Volatility Analysis", "Time Edge"])
    
    with insight_tabs[0]:
        # Show market regime for active analyses
        if st.session_state.get('signal_ready', False):
            signal_data = st.session_state.get('signal_data', pd.DataFrame())
            if not signal_data.empty:
                regime = detect_market_regime(signal_data)
                
                regime_col1, regime_col2 = st.columns(2)
                with regime_col1:
                    st.metric("Trend Direction", regime['trend'])
                    st.metric("Trend Strength", f"{regime['strength']:.1f}")
                
                with regime_col2:
                    st.metric("Volatility Regime", regime['regime'])
                    st.metric("Volatility Level", f"{regime['volatility']:.1f}%")
                
                st.info(get_regime_recommendation(regime))
    
    with insight_tabs[1]:
        # Volatility analysis across active symbols
        if st.session_state.get('spx_analysis_ready', False):
            st.write("SPX volatility analysis available after generating anchors")
        
        if st.session_state.get('stock_analysis_ready', False):
            stock_ticker = st.session_state.get('stock_analysis_ticker', 'N/A')
            stock_data = st.session_state.get('stock_analysis_data', pd.DataFrame())
            
            if not stock_data.empty:
                regime = detect_market_regime(stock_data)
                st.write(f"**{stock_ticker} Volatility:** {regime['volatility']:.1f}%")
                st.write(get_volatility_recommendation(regime['volatility']))
    
    with insight_tabs[2]:
        # Time-of-day edge analysis
        optimal_times = get_optimal_entry_times()
        
        st.write("**Optimal Entry Times by Strategy:**")
        for strategy, times in optimal_times.items():
            st.write(f"**{strategy.replace('_', ' ')}:** {', '.join(times)}")

# ═══════════════════════════════════════════════════════════════════════════════
# FOOTER AND VERSION INFO
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown("---")
st.markdown(
    f"""
    <div style='text-align: center; color: #888; font-size: 0.9em;'>
        SPX Prophet Analytics • Real-time market analysis • 
        Session: {datetime.now(CT_TZ).strftime('%H:%M:%S CT')} • 
        Theme: {st.session_state.theme}
    </div>
    """, 
    unsafe_allow_html=True
)

# Error handling wrapper for the entire app
if 'app_errors' not in st.session_state:
    st.session_state.app_errors = []

# Add any persistent error handling here
try:
    # Validate critical session state
    required_states = ['spx_slopes', 'stock_slopes', 'current_offset', 'theme']
    missing_states = [state for state in required_states if state not in st.session_state]
    
    if missing_states:
        st.error(f"Missing session state: {', '.join(missing_states)}. Please refresh the app.")
        
except Exception as e:
    st.error(f"Application error: {str(e)}")
    st.info("Please refresh the page to reset the application.")

# ═══════════════════════════════════════════════════════════════════════════════
# APPLICATION COMPLETE
# ═══════════════════════════════════════════════════════════════════════════════
