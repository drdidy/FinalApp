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
        
        # Calculate targets based on realistic daily range
        # Estimate daily range: SPX typically moves 30-60 points intraday
        estimated_daily_range = anchor_price * 0.008  # Approximately 50 points at 6400 level
        
        if is_skyline:
            # TP1: 30% of daily range, TP2: 50% of daily range
            tp1_distance = estimated_daily_range * 0.30
            tp2_distance = estimated_daily_range * 0.50
            
            entry_price = anchor_price
            tp1_price = anchor_price + tp1_distance  # Bounce up from skyline
            tp2_price = anchor_price + tp2_distance
            direction = "BUY"
            
            # Stop above skyline with retest buffer
            stop_price = anchor_price + (anchor_price * 0.006)
            
        elif is_baseline:
            # TP1: 30% of daily range, TP2: 50% of daily range
            tp1_distance = estimated_daily_range * 0.30
            tp2_distance = estimated_daily_range * 0.50
            
            entry_price = anchor_price
            tp1_price = anchor_price + tp1_distance  # Bounce up from baseline
            tp2_price = anchor_price + tp2_distance
            direction = "BUY"
            
            # Stop below baseline with retest buffer
            stop_price = max(0.01, anchor_price - (anchor_price * 0.006))
            
        else:
            # High/Close/Low anchors with same realistic targets
            tp1_distance = estimated_daily_range * 0.30
            tp2_distance = estimated_daily_range * 0.50
            
            if anchor_type.upper() == 'HIGH':
                entry_price = anchor_price
                tp1_price = anchor_price - tp1_distance
                tp2_price = anchor_price - tp2_distance
                direction = "SELL"
                stop_price = anchor_price + (anchor_price * 0.005)
            else:
                entry_price = anchor_price
                tp1_price = anchor_price + tp1_distance
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
        'SKYLINE': 80.0,
        'BASELINE': 80.0,
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
            key="spx_prev_day"
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
    
    # Check if date changed and auto-fetch data
    if ('last_spx_date' not in st.session_state or 
        st.session_state.last_spx_date != prev_day):
        
        st.session_state.last_spx_date = prev_day
        
        with st.spinner("Loading data for selected date..."):
            # Auto-update offset for selected date
            es_data_for_offset = fetch_live_data("ES=F", prev_day, prev_day)
            spx_data_for_offset = fetch_live_data("^GSPC", prev_day, prev_day)
            
            if not es_data_for_offset.empty and not spx_data_for_offset.empty:
                st.session_state.current_offset = calculate_es_spx_offset(es_data_for_offset, spx_data_for_offset)
            
            # Fetch ES futures data for anchor detection
            es_data = fetch_live_data("ES=F", prev_day, prev_day)
            
            if not es_data.empty:
                # Get anchor window data
                anchor_window = get_session_window(es_data, SPX_ANCHOR_START, SPX_ANCHOR_END)
                if anchor_window.empty:
                    anchor_window = es_data
                
                st.session_state.es_anchor_data = anchor_window
                
                # Get SPX data for High/Close/Low anchors
                spx_data = fetch_live_data("^GSPC", prev_day, prev_day)
                
                if not spx_data.empty:
                    daily_ohlc = get_daily_ohlc(spx_data, prev_day)
                    if daily_ohlc:
                        st.session_state.spx_manual_anchors = daily_ohlc
                else:
                    # Convert ES to SPX equivalent
                    es_daily_ohlc = get_daily_ohlc(anchor_window, prev_day)
                    if es_daily_ohlc:
                        spx_equivalent = {}
                        for key, (es_price, timestamp) in es_daily_ohlc.items():
                            spx_equivalent[key] = (es_price + st.session_state.current_offset, timestamp)
                        st.session_state.spx_manual_anchors = spx_equivalent
                
                st.session_state.spx_analysis_ready = True
    
    # Show current offset
    st.info(f"ES→SPX Offset for {prev_day}: {st.session_state.current_offset:+.1f}")
    st.markdown("---")
    
    # ═══════════════════════════════════════════════════════════════════════════════
    # RESULTS DISPLAY
    # ═══════════════════════════════════════════════════════════════════════════════
    
    if st.session_state.get('spx_analysis_ready', False):
        st.subheader("SPX Anchor Projections")
        
        # Process swing detection for Skyline/Baseline
        es_data = st.session_state.get('es_anchor_data', pd.DataFrame())
        skyline_anchor_spx = None
        baseline_anchor_spx = None
        
        if not es_data.empty:
            es_swings = detect_swings_simple(es_data)
            es_skyline, es_baseline = get_anchor_points(es_swings)
            
            # Convert ES anchors to SPX equivalent
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
            
            # Manual anchors display
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
            
            # Swing anchors display
            with summary_cols[3]:
                if skyline_anchor_spx:
                    price, timestamp = skyline_anchor_spx
                    st.markdown(f"""
                    <div style="text-align: center; padding: 1rem; background: rgba(255,100,100,0.2); border-radius: 10px; border-left: 4px solid #ff4757;">
                        <h4>Skyline</h4>
                        <h3>${price:.2f}</h3>
                        <p>{format_ct_time(timestamp)}</p>
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
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.warning("No Baseline")
        
        st.markdown("---")
        
        # Projection tabs
        projection_tabs = st.tabs(["High", "Close", "Low", "Skyline", "Baseline"])
        
        # Manual anchor projections
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
                    
                    st.subheader("High Anchor SPX Projection")
                    st.dataframe(high_proj, use_container_width=True, hide_index=True)
                    
                    high_analysis = calculate_entry_exit_table(high_proj, "HIGH")
                    st.subheader("Entry/Exit Strategy")
                    st.dataframe(high_analysis, use_container_width=True, hide_index=True)
                else:
                    st.warning("No high anchor data")
            
            # Close Anchor
            with projection_tabs[1]:
                if 'close' in manual_anchors:
                    spx_price, timestamp = manual_anchors['close']
                    anchor_time_ct = timestamp.astimezone(CT_TZ)
                    
                    close_proj = project_anchor_line(
                        spx_price, anchor_time_ct,
                        st.session_state.spx_slopes['close'], proj_day
                    )
                    
                    st.subheader("Close Anchor SPX Projection")
                    st.dataframe(close_proj, use_container_width=True, hide_index=True)
                    
                    close_analysis = calculate_entry_exit_table(close_proj, "CLOSE")
                    st.subheader("Entry/Exit Strategy")
                    st.dataframe(close_analysis, use_container_width=True, hide_index=True)
                else:
                    st.warning("No close anchor data")
            
            # Low Anchor
            with projection_tabs[2]:
                if 'low' in manual_anchors:
                    spx_price, timestamp = manual_anchors['low']
                    anchor_time_ct = timestamp.astimezone(CT_TZ)
                    
                    low_proj = project_anchor_line(
                        spx_price, anchor_time_ct,
                        st.session_state.spx_slopes['low'], proj_day
                    )
                    
                    st.subheader("Low Anchor SPX Projection")
                    st.dataframe(low_proj, use_container_width=True, hide_index=True)
                    
                    low_analysis = calculate_entry_exit_table(low_proj, "LOW")
                    st.subheader("Entry/Exit Strategy")
                    st.dataframe(low_analysis, use_container_width=True, hide_index=True)
                else:
                    st.warning("No low anchor data")
        
        # Swing-based projections using SPX converted values
        with projection_tabs[3]:  # Skyline
            if skyline_anchor_spx:
                spx_sky_price, sky_time = skyline_anchor_spx
                sky_time_ct = sky_time.astimezone(CT_TZ)
                
                skyline_proj = project_anchor_line(
                    spx_sky_price, sky_time_ct,
                    st.session_state.spx_slopes['skyline'], proj_day
                )
                
                st.subheader("Skyline SPX Projection (80% Zone)")
                st.dataframe(skyline_proj, use_container_width=True, hide_index=True)
                
                sky_analysis = calculate_entry_exit_table(skyline_proj, "SKYLINE")
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
                
                st.subheader("Baseline SPX Projection (80% Zone)")
                st.dataframe(baseline_proj, use_container_width=True, hide_index=True)
                
                base_analysis = calculate_entry_exit_table(baseline_proj, "BASELINE")
                st.subheader("Baseline Bounce Strategy")
                st.dataframe(base_analysis, use_container_width=True, hide_index=True)
            else:
                st.warning("No baseline anchor detected")
    
    else:
        st.info("Select a date to automatically load SPX anchor analysis")

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

def calculate_stock_entry_exit_table(projection_df: pd.DataFrame, ticker: str, anchor_type: str, day_name: str) -> pd.DataFrame:
    """Calculate stock entry/exit analysis with weekly probability adjustments"""
    if projection_df.empty:
        return pd.DataFrame()
    
    analysis_rows = []
    stock_volatility = get_stock_volatility_factor(ticker)
    
    # Day-specific probability multipliers
    day_multipliers = {"Wednesday": 1.1, "Thursday": 1.0, "Friday": 0.9}
    day_mult = day_multipliers.get(day_name, 1.0)
    
    # Determine anchor characteristics
    is_skyline = anchor_type.upper() in ['SKYLINE', 'HIGH']
    is_baseline = anchor_type.upper() in ['BASELINE', 'LOW']
    
    for idx, row in projection_df.iterrows():
        time_slot = row['Time']
        anchor_price = row['Price']
        
        # Stock-specific volatility calculations
        base_volatility = anchor_price * stock_volatility * 0.015
        
        if is_skyline:
            # Skyline bounce strategy for stocks
            tp1_distance = base_volatility * 0.7
            tp2_distance = base_volatility * 2.0
            
            entry_price = anchor_price
            tp1_price = anchor_price + tp1_distance  # Bounce up from skyline
            tp2_price = anchor_price + tp2_distance
            direction = "BUY"
            
            # Stop above skyline with retest buffer
            stop_price = anchor_price + (anchor_price * 0.008)
            
        elif is_baseline:
            # Baseline bounce strategy for stocks
            tp1_distance = base_volatility * 0.7
            tp2_distance = base_volatility * 2.0
            
            entry_price = anchor_price
            tp1_price = anchor_price + tp1_distance  # Bounce up from baseline
            tp2_price = anchor_price + tp2_distance
            direction = "BUY"
            
            # Stop below baseline with retest buffer
            stop_price = max(0.01, anchor_price - (anchor_price * 0.008))
            
        else:
            # High/Close/Low stock anchors
            tp1_distance = base_volatility * 0.6
            tp2_distance = base_volatility * 1.6
            
            if anchor_type.upper() == 'HIGH':
                entry_price = anchor_price
                tp1_price = anchor_price - tp1_distance
                tp2_price = anchor_price - tp2_distance
                direction = "SELL"
                stop_price = anchor_price + (anchor_price * 0.006)
            else:
                entry_price = anchor_price
                tp1_price = anchor_price + tp1_distance
                tp2_price = anchor_price + tp2_distance
                direction = "BUY"
                stop_price = anchor_price - (anchor_price * 0.006)
        
        risk_amount = abs(entry_price - stop_price)
        
        # Calculate probabilities with day and ticker adjustments
        entry_prob = calculate_stock_entry_probability(ticker, anchor_type, time_slot) * day_mult
        tp1_prob = calculate_stock_target_probability(ticker, anchor_type, 1) * day_mult
        tp2_prob = calculate_stock_target_probability(ticker, anchor_type, 2) * day_mult
        
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
            'Entry_Prob': f"{min(95, entry_prob):.0f}%",
            'TP1_Prob': f"{min(85, tp1_prob):.0f}%",
            'TP2_Prob': f"{min(75, tp2_prob):.0f}%",
            'Day': day_name
        })
    
    return pd.DataFrame(analysis_rows)

def get_stock_volatility_factor(ticker: str) -> float:
    """Get volatility factor for different stocks"""
    volatility_factors = {
        'TSLA': 1.8, 'NVDA': 1.6, 'META': 1.4, 'NFLX': 1.3,
        'AMZN': 1.2, 'GOOGL': 1.1, 'MSFT': 1.0, 'AAPL': 0.9
    }
    return volatility_factors.get(ticker, 1.2)

def calculate_stock_entry_probability(ticker: str, anchor_type: str, time_slot: str) -> float:
    """Calculate stock entry probability with ticker and time adjustments"""
    base_probs = {
        'SKYLINE': 80.0, 'BASELINE': 80.0,
        'HIGH': 70.0, 'CLOSE': 75.0, 'LOW': 70.0
    }
    
    base_prob = base_probs.get(anchor_type.upper(), 65.0)
    
    # Time adjustments
    hour = int(time_slot.split(':')[0])
    if hour in [9, 10]:  # Market open volatility
        time_adj = 10
    elif hour in [13, 14]:  # End of day momentum
        time_adj = 5
    else:
        time_adj = 0
    
    # Ticker-specific adjustments
    if ticker in ['TSLA', 'NVDA', 'META']:  # High momentum stocks
        ticker_adj = 5
    elif ticker in ['AAPL', 'MSFT']:  # Stable stocks
        ticker_adj = -5
    else:
        ticker_adj = 0
    
    return base_prob + time_adj + ticker_adj

def calculate_stock_target_probability(ticker: str, anchor_type: str, target_num: int) -> float:
    """Calculate stock target probability with volatility adjustments"""
    volatility_factor = get_stock_volatility_factor(ticker)
    
    if anchor_type.upper() in ['SKYLINE', 'BASELINE']:
        base_tp1, base_tp2 = 80.0, 60.0
    else:
        base_tp1, base_tp2 = 70.0, 50.0
    
    # Higher volatility stocks have better target probability
    vol_adj = (volatility_factor - 1) * 8
    
    if target_num == 1:
        return base_tp1 + vol_adj
    else:
        return base_tp2 + vol_adj

def calculate_stock_anchor_analytics(projection_df: pd.DataFrame, ticker: str, anchor_type: str, market_data: pd.DataFrame) -> pd.DataFrame:
    """Calculate stock-specific anchor line analytics"""
    if projection_df.empty:
        return pd.DataFrame()
    
    analytics = []
    volatility_factor = get_stock_volatility_factor(ticker)
    
    # Stock-specific bounce analysis
    if anchor_type.upper() in ['SKYLINE', 'BASELINE']:
        bounce_rate = 80.0
        reliability = 85.0
    else:
        bounce_rate = 70.0
        reliability = 75.0
    
    # Volume analysis for stock
    if not market_data.empty and 'Volume' in market_data.columns:
        recent_volume = market_data['Volume'].tail(20).mean()
        volume_score = min(90, max(40, (recent_volume / 100000) * 5))
    else:
        volume_score = 65.0
    
    for idx, row in projection_df.iterrows():
        time_slot = row['Time']
        
        # Stock momentum at different times
        hour = int(time_slot.split(':')[0])
        momentum_score = 85 if hour in [9, 10] else 75 if hour in [13, 14] else 65
        
        # Volatility-adjusted confidence
        vol_confidence = min(95, 60 + (volatility_factor * 15))
        
        # Overall trade confidence
        trade_confidence = (bounce_rate * 0.25) + (reliability * 0.25) + (volume_score * 0.2) + (momentum_score * 0.15) + (vol_confidence * 0.15)
        
        analytics.append({
            'Time': time_slot,
            'Bounce_Rate': f"{bounce_rate:.0f}%",
            'Reliability': f"{reliability:.0f}%",
            'Volume_Score': f"{volume_score:.0f}%", 
            'Momentum': f"{momentum_score:.0f}%",
            'Vol_Confidence': f"{vol_confidence:.0f}%",
            'Trade_Confidence': f"{trade_confidence:.0f}%",
            'Recommendation': get_stock_recommendation(trade_confidence)
        })
    
    return pd.DataFrame(analytics)

def get_stock_recommendation(confidence: float) -> str:
    """Get trading recommendation based on confidence score"""
    if confidence >= 85:
        return "STRONG BUY"
    elif confidence >= 75:
        return "BUY"
    elif confidence >= 65:
        return "MODERATE"
    else:
        return "WEAK"

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
# AUTO-UPDATE FUNCTIONS FOR SIGNALS
# ═══════════════════════════════════════════════════════════════════════════════

def update_signal_data_for_date():
    """Automatically fetch signal data when date changes"""
    if ('sig_symbol' in st.session_state and 'sig_day' in st.session_state):
        symbol = st.session_state.sig_symbol
        analysis_date = st.session_state.sig_day
        
        # Fetch data for the selected date
        signal_data = fetch_live_data(symbol, analysis_date, analysis_date)
        
        if not signal_data.empty:
            # Filter to RTH session and store
            rth_data = get_session_window(signal_data, RTH_START, RTH_END)
            if not rth_data.empty:
                st.session_state.signal_data = rth_data
                st.session_state.signal_symbol = symbol
                st.session_state.signal_ready = True

# ═══════════════════════════════════════════════════════════════════════════════
# SIGNAL DETECTION FUNCTIONS 
# ═══════════════════════════════════════════════════════════════════════════════

def detect_anchor_touch_signals(price_data: pd.DataFrame, ref_line: pd.DataFrame, anchor_type: str) -> pd.DataFrame:
    """Detect your specific anchor touch patterns for profitability"""
    if price_data.empty or ref_line.empty:
        return pd.DataFrame()
    
    signals = []
    
    # Create reference line lookup
    ref_dict = {}
    for _, row in ref_line.iterrows():
        ref_dict[row['Time']] = row['Price']
    
    for idx, bar in price_data.iterrows():
        bar_time = format_ct_time(idx)
        
        if bar_time not in ref_dict:
            continue
            
        anchor_price = ref_dict[bar_time]
        open_price = bar['Open']
        high_price = bar['High']
        low_price = bar['Low']
        close_price = bar['Close']
        
        # Determine candle type
        is_bearish = close_price < open_price  # Red candle
        is_bullish = close_price > open_price  # Green candle
        
        # Check for anchor touches (exact using your slopes)
        tolerance = anchor_price * 0.0005  # Tight tolerance for precise touches
        touches_anchor = (low_price <= anchor_price + tolerance and 
                         high_price >= anchor_price - tolerance)
        
        if not touches_anchor:
            continue
            
        signal_detected = False
        signal_reason = ""
        entry_direction = ""
        probability = 0
        
        # Your specific strategy patterns
        if anchor_type == "SKYLINE":
            # Red candle touches skyline from above (low touches) and closes above
            if (is_bearish and low_price <= anchor_price + tolerance and 
                close_price > anchor_price):
                signal_detected = True
                signal_reason = "Bearish candle touched skyline from above, closed above"
                entry_direction = "BUY"
                probability = 80.0
            
            # Red candle touches skyline from above and closes below  
            elif (is_bearish and low_price <= anchor_price + tolerance and 
                  close_price < anchor_price):
                signal_detected = True
                signal_reason = "Bearish candle touched skyline, closed below"
                entry_direction = "SELL"
                probability = 75.0
            
            # Green candle touches skyline from above - expect drop to baseline
            elif (is_bullish and low_price <= anchor_price + tolerance):
                signal_detected = True
                signal_reason = "Bullish candle touched skyline - expect drop to baseline"
                entry_direction = "SELL"
                probability = 70.0
        
        elif anchor_type == "BASELINE":
            # Red candle touches baseline from above and closes above
            if (is_bearish and low_price <= anchor_price + tolerance and 
                close_price > anchor_price):
                signal_detected = True
                signal_reason = "Bearish candle touched baseline from above, closed above"
                entry_direction = "BUY"
                probability = 80.0
            
            # Red candle touches baseline from above and closes below
            elif (is_bearish and low_price <= anchor_price + tolerance and 
                  close_price < anchor_price):
                signal_detected = True
                signal_reason = "Bearish candle touched baseline, closed below"
                entry_direction = "SELL" 
                probability = 75.0
            
            # Green candle touches baseline from below and closes below
            elif (is_bullish and high_price >= anchor_price - tolerance and 
                  close_price < anchor_price):
                signal_detected = True
                signal_reason = "Bullish candle touched baseline from below, closed below"
                entry_direction = "SELL"
                probability = 75.0
            
            # Green candle touches baseline from below and closes above  
            elif (is_bullish and high_price >= anchor_price - tolerance and 
                  close_price > anchor_price):
                signal_detected = True
                signal_reason = "Bullish candle touched baseline from below, closed above"
                entry_direction = "BUY"
                probability = 80.0
            
            # Red candle touches baseline from below - expect rise to skyline
            elif (is_bearish and high_price >= anchor_price - tolerance):
                signal_detected = True
                signal_reason = "Bearish candle touched baseline from below - expect rise to skyline"
                entry_direction = "BUY"
                probability = 75.0
        
        if signal_detected:
            # Calculate signal quality metrics
            touch_precision = calculate_touch_precision(bar, anchor_price)
            volume_confirmation = calculate_volume_confirmation(bar, price_data)
            
            # Entry at close of touching candle
            entry_price = close_price
            
            # Calculate stop and targets
            if entry_direction == "BUY":
                stop_buffer = anchor_price * 0.005
                stop_price = anchor_price - stop_buffer if anchor_type == "BASELINE" else anchor_price + stop_buffer
                tp1_price = entry_price + (entry_price * 0.015)
                tp2_price = entry_price + (entry_price * 0.035)
            else:
                stop_buffer = anchor_price * 0.005
                stop_price = anchor_price + stop_buffer if anchor_type == "BASELINE" else anchor_price - stop_buffer
                tp1_price = entry_price - (entry_price * 0.015)
                tp2_price = entry_price - (entry_price * 0.035)
            
            signals.append({
                'Time': bar_time,
                'Anchor_Type': anchor_type,
                'Direction': entry_direction,
                'Entry_Price': round(entry_price, 2),
                'Stop': round(stop_price, 2),
                'TP1': round(tp1_price, 2),
                'TP2': round(tp2_price, 2),
                'Anchor_Price': round(anchor_price, 2),
                'Candle_Type': 'Bearish' if is_bearish else 'Bullish',
                'Touch_Precision': f"{touch_precision:.1f}%",
                'Volume_Conf': f"{volume_confirmation:.1f}%",
                'Probability': f"{probability:.0f}%",
                'Signal_Reason': signal_reason
            })
    
    return pd.DataFrame(signals)

def calculate_touch_precision(bar: pd.Series, anchor_price: float) -> float:
    """Calculate precision of anchor line touch"""
    low_distance = abs(bar['Low'] - anchor_price) / anchor_price
    high_distance = abs(bar['High'] - anchor_price) / anchor_price
    
    closest_distance = min(low_distance, high_distance)
    precision = max(0, 100 - (closest_distance * 2000))
    return min(100, precision)

def calculate_volume_confirmation(bar: pd.Series, data: pd.DataFrame) -> float:
    """Calculate volume confirmation for the touch"""
    if 'Volume' not in data.columns:
        return 70.0
    
    recent_avg_volume = data['Volume'].tail(20).mean()
    current_volume = bar['Volume']
    
    if recent_avg_volume == 0:
        return 70.0
    
    volume_ratio = current_volume / recent_avg_volume
    
    if volume_ratio >= 1.5:
        return 90.0
    elif volume_ratio >= 1.2:
        return 80.0
    elif volume_ratio >= 0.8:
        return 70.0
    else:
        return 50.0

# ═══════════════════════════════════════════════════════════════════════════════
# EMA ANALYSIS FUNCTIONS  
# ═══════════════════════════════════════════════════════════════════════════════

def calculate_ema_crossover_analysis(price_data: pd.DataFrame, ema8: pd.Series, ema21: pd.Series) -> pd.DataFrame:
    """Detect and analyze EMA 8/21 crossovers"""
    if price_data.empty or ema8.empty or ema21.empty:
        return pd.DataFrame()
    
    crossovers = []
    
    for i in range(1, len(ema8)):
        prev_8 = ema8.iloc[i-1]
        prev_21 = ema21.iloc[i-1]
        curr_8 = ema8.iloc[i]
        curr_21 = ema21.iloc[i]
        
        # Detect crossover
        crossover_type = None
        if prev_8 <= prev_21 and curr_8 > curr_21:
            crossover_type = "Bullish Cross"
        elif prev_8 >= prev_21 and curr_8 < curr_21:
            crossover_type = "Bearish Cross"
        
        if crossover_type:
            timestamp = ema8.index[i]
            time_ct = format_ct_time(timestamp)
            
            # Calculate crossover strength
            separation = abs(curr_8 - curr_21) / curr_21 * 100
            strength = "Strong" if separation >= 0.5 else "Moderate" if separation >= 0.2 else "Weak"
            
            crossovers.append({
                'Time': time_ct,
                'Crossover_Type': crossover_type,
                'EMA8': round(curr_8, 2),
                'EMA21': round(curr_21, 2),
                'Current_Price': round(price_data.iloc[i]['Close'], 2),
                'Separation': f"{separation:.3f}%",
                'Strength': strength,
                'Trade_Signal': crossover_type.split()[0].upper()
            })
    
    return pd.DataFrame(crossovers)

def analyze_market_regime(ema8: pd.Series, ema21: pd.Series, vwap: pd.Series, price_data: pd.DataFrame) -> pd.DataFrame:
    """Analyze current market regime for trading context"""
    if ema8.empty or ema21.empty or price_data.empty:
        return pd.DataFrame()
    
    current_8 = ema8.iloc[-1]
    current_21 = ema21.iloc[-1]
    current_price = price_data.iloc[-1]['Close']
    current_vwap = vwap.iloc[-1] if not vwap.empty else current_price
    
    # EMA regime analysis
    if current_8 > current_21:
        ema_regime = "Bullish"
        regime_strength = (current_8 - current_21) / current_21 * 100
    else:
        ema_regime = "Bearish"
        regime_strength = (current_21 - current_8) / current_8 * 100
    
    # VWAP analysis
    vwap_position = "Above VWAP" if current_price > current_vwap else "Below VWAP"
    vwap_distance = abs(current_price - current_vwap) / current_vwap * 100
    
    # Volatility regime
    recent_returns = price_data['Close'].pct_change().tail(20).std() * 100
    if recent_returns >= 2.5:
        vol_regime = "High Volatility"
    elif recent_returns >= 1.5:
        vol_regime = "Moderate Volatility"
    else:
        vol_regime = "Low Volatility"
    
    regime_data = [
        {
            'Component': 'EMA Trend',
            'Status': ema_regime,
            'Strength': f"{regime_strength:.2f}%",
            'Trade_Bias': ema_regime.upper(),
            'Confidence': get_regime_confidence(regime_strength)
        },
        {
            'Component': 'VWAP Position',
            'Status': vwap_position,
            'Strength': f"{vwap_distance:.2f}%",
            'Trade_Bias': 'BULLISH' if vwap_position == 'Above VWAP' else 'BEARISH',
            'Confidence': get_vwap_confidence(vwap_distance)
        },
        {
            'Component': 'Volatility',
            'Status': vol_regime,
            'Strength': f"{recent_returns:.2f}%",
            'Trade_Bias': 'MOMENTUM' if 'High' in vol_regime else 'RANGE',
            'Confidence': get_volatility_confidence(recent_returns)
        }
    ]
    
    return pd.DataFrame(regime_data)

def get_regime_confidence(strength: float) -> str:
    """Get confidence level for regime strength"""
    if strength >= 1.0:
        return "High"
    elif strength >= 0.5:
        return "Moderate"
    else:
        return "Low"

def get_vwap_confidence(distance: float) -> str:
    """Get confidence based on VWAP distance"""
    if distance >= 1.0:
        return "High"
    elif distance >= 0.5:
        return "Moderate"
    else:
        return "Low"

def get_volatility_confidence(volatility: float) -> str:
    """Get confidence based on volatility level"""
    if volatility >= 2.0:
        return "High"
    elif volatility >= 1.0:
        return "Moderate"
    else:
        return "Low"

# ═══════════════════════════════════════════════════════════════════════════════
# ANCHOR LINE STATISTICS
# ═══════════════════════════════════════════════════════════════════════════════

def calculate_anchor_line_interaction_stats(price_data: pd.DataFrame, ref_line: pd.DataFrame) -> pd.DataFrame:
    """Calculate statistical analysis of anchor line interactions for profitability"""
    if price_data.empty or ref_line.empty:
        return pd.DataFrame()
    
    # Create reference line lookup
    ref_dict = {}
    for _, row in ref_line.iterrows():
        ref_dict[row['Time']] = row['Price']
    
    total_touches = 0
    bounces = 0
    penetrations = 0
    bounce_distances = []
    penetration_depths = []
    follow_through_moves = []
    
    for idx, bar in price_data.iterrows():
        bar_time = format_ct_time(idx)
        
        if bar_time not in ref_dict:
            continue
            
        ref_price = ref_dict[bar_time]
        
        # Check for touches
        tolerance = ref_price * 0.002
        touches_line = (bar['Low'] <= ref_price + tolerance and 
                       bar['High'] >= ref_price - tolerance)
        
        if touches_line:
            total_touches += 1
            
            # Analyze bounce vs penetration
            if bar['Close'] > ref_price + tolerance:
                bounces += 1
                bounce_distance = bar['Close'] - ref_price
                bounce_distances.append(bounce_distance)
                
                # Check follow-through in next few bars
                next_bars = price_data.iloc[idx:idx+3] if idx < len(price_data)-3 else pd.DataFrame()
                if not next_bars.empty:
                    max_follow = next_bars['High'].max() - bar['Close']
                    follow_through_moves.append(max_follow)
                    
            elif bar['Close'] < ref_price - tolerance:
                penetrations += 1
                penetration_depth = ref_price - bar['Close']
                penetration_depths.append(penetration_depth)
    
    # Calculate statistics
    bounce_rate = (bounces / total_touches * 100) if total_touches > 0 else 0
    penetration_rate = (penetrations / total_touches * 100) if total_touches > 0 else 0
    avg_bounce = np.mean(bounce_distances) if bounce_distances else 0
    avg_penetration = np.mean(penetration_depths) if penetration_depths else 0
    avg_follow_through = np.mean(follow_through_moves) if follow_through_moves else 0
    
    stats = [
        {
            'Metric': 'Total Line Touches',
            'Value': total_touches,
            'Percentage': '-',
            'Quality': 'High' if total_touches >= 5 else 'Low'
        },
        {
            'Metric': 'Bounce Rate',
            'Value': bounces,
            'Percentage': f"{bounce_rate:.1f}%",
            'Quality': 'High' if bounce_rate >= 70 else 'Moderate' if bounce_rate >= 50 else 'Low'
        },
        {
            'Metric': 'Penetration Rate', 
            'Value': penetrations,
            'Percentage': f"{penetration_rate:.1f}%",
            'Quality': 'Low' if penetration_rate <= 30 else 'Moderate' if penetration_rate <= 50 else 'High'
        },
        {
            'Metric': 'Average Bounce Distance',
            'Value': f"${avg_bounce:.2f}",
            'Percentage': f"{(avg_bounce/ref_dict[list(ref_dict.keys())[0]]*100):.2f}%" if ref_dict else '-',
            'Quality': 'High' if avg_bounce >= 10 else 'Moderate' if avg_bounce >= 5 else 'Low'
        },
        {
            'Metric': 'Average Follow-Through',
            'Value': f"${avg_follow_through:.2f}",
            'Percentage': '-',
            'Quality': 'High' if avg_follow_through >= 15 else 'Moderate' if avg_follow_through >= 8 else 'Low'
        }
    ]
    
    return pd.DataFrame(stats)

with tab3:
    st.subheader("Signal Detection & EMA Analysis")
    st.caption("Real-time anchor touch pattern detection with market regime analysis")
    
    # ═══════════════════════════════════════════════════════════════════════════════
    # INPUT CONTROLS WITH AUTO-UPDATE
    # ═══════════════════════════════════════════════════════════════════════════════
    
    col1, col2 = st.columns(2)
    with col1:
        signal_symbol = st.text_input(
            "Symbol", 
            value="^GSPC",
            key="sig_symbol",
            on_change=update_signal_data_for_date
        )
    
    with col2:
        signal_day = st.date_input(
            "Analysis Day",
            value=datetime.now(CT_TZ).date(),
            key="sig_day",
            on_change=update_signal_data_for_date
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
    
    # Show live data status
    if st.session_state.get('signal_ready', False):
        st.success(f"Live data loaded for {signal_symbol} on {signal_day}")
    else:
        st.info("Enter symbol and date to automatically load signal data")
    
    # ═══════════════════════════════════════════════════════════════════════════════
    # AUTOMATIC RESULTS DISPLAY
    # ═══════════════════════════════════════════════════════════════════════════════
    
    if st.session_state.get('signal_ready', False):
        signal_data = st.session_state.signal_data
        symbol = st.session_state.signal_symbol
        
        st.subheader(f"{symbol} Signal Analysis")
        
        # Create anchor datetime for projection
        anchor_datetime = datetime.combine(signal_day, anchor_time_input)
        anchor_datetime_ct = CT_TZ.localize(anchor_datetime)
        
        # Generate reference line projection
        ref_line_proj = project_anchor_line(
            anchor_price, 
            anchor_datetime_ct,
            ref_slope,
            signal_day
        )
        
        # Calculate indicators
        ema8 = calculate_ema(signal_data['Close'], 8)
        ema21 = calculate_ema(signal_data['Close'], 21)
        vwap = calculate_vwap(signal_data)
        
        # Analysis tabs
        signal_tabs = st.tabs(["Reference Line", "Skyline Patterns", "Baseline Patterns", "EMA Analysis", "Market Regime"])
        
        with signal_tabs[0]:  # Reference Line
            st.subheader("Reference Line Projection")
            st.dataframe(ref_line_proj, use_container_width=True, hide_index=True)
            
            # Anchor line interaction statistics
            line_stats = calculate_anchor_line_interaction_stats(signal_data, ref_line_proj)
            st.subheader("Anchor Line Statistics")
            st.dataframe(line_stats, use_container_width=True, hide_index=True)
        
        with signal_tabs[1]:  # Skyline Patterns
            skyline_signals = detect_anchor_touch_signals(signal_data, ref_line_proj, "SKYLINE")
            if not skyline_signals.empty:
                st.subheader("Skyline Touch Patterns (80% Zone)")
                st.dataframe(skyline_signals, use_container_width=True, hide_index=True)
            else:
                st.info("No skyline touch patterns detected for this day")
        
        with signal_tabs[2]:  # Baseline Patterns
            baseline_signals = detect_anchor_touch_signals(signal_data, ref_line_proj, "BASELINE")
            if not baseline_signals.empty:
                st.subheader("Baseline Touch Patterns (80% Zone)")
                st.dataframe(baseline_signals, use_container_width=True, hide_index=True)
            else:
                st.info("No baseline touch patterns detected for this day")
        
        with signal_tabs[3]:  # EMA Analysis
            ema_analysis = calculate_ema_crossover_analysis(signal_data, ema8, ema21)
            if not ema_analysis.empty:
                st.subheader("EMA 8/21 Crossover Analysis")
                st.dataframe(ema_analysis, use_container_width=True, hide_index=True)
            else:
                st.info("No EMA crossovers detected for this day")
        
        with signal_tabs[4]:  # Market Regime
            regime_analysis = analyze_market_regime(ema8, ema21, vwap, signal_data)
            st.subheader("Market Regime Analysis")
            st.dataframe(regime_analysis, use_container_width=True, hide_index=True)
            
            # Trading recommendations based on regime
            if not regime_analysis.empty:
                st.subheader("Trading Context")
                ema_bias = regime_analysis.iloc[0]['Trade_Bias']
                vwap_bias = regime_analysis.iloc[1]['Trade_Bias'] 
                vol_regime = regime_analysis.iloc[2]['Trade_Bias']
                
                if ema_bias == vwap_bias:
                    st.success(f"Aligned regime: {ema_bias} bias confirmed by both EMA and VWAP")
                else:
                    st.warning(f"Mixed signals: EMA shows {ema_bias}, VWAP shows {vwap_bias}")
                
                st.info(f"Volatility regime: {vol_regime} - adjust position sizing accordingly")

# ═══════════════════════════════════════════════════════════════════════════════
# END OF SIGNALS & EMA TAB
# ═══════════════════════════════════════════════════════════════════════════════







# ═══════════════════════════════════════════════════════════════════════════════
# SPX PROPHET - PART 5: CONTRACT TOOL TAB
# ═══════════════════════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════════════════════
# AUTO-UPDATE FUNCTIONS FOR CONTRACT TOOL
# ═══════════════════════════════════════════════════════════════════════════════

def update_contract_analysis():
    """Automatically update contract analysis when parameters change"""
    if ('ct_p1_price' in st.session_state and 'ct_p2_price' in st.session_state and
        'ct_p1_time' in st.session_state and 'ct_p2_time' in st.session_state and
        'ct_proj_day' in st.session_state):
        
        # Get contract parameters
        p1_price = st.session_state.ct_p1_price
        p2_price = st.session_state.ct_p2_price
        p1_time = st.session_state.ct_p1_time
        p2_time = st.session_state.ct_p2_time
        proj_day = st.session_state.ct_proj_day
        
        # Validate time sequence
        p1_date = proj_day - timedelta(days=1) if p1_time >= time(20, 0) else proj_day
        p2_date = proj_day if p2_time <= time(10, 0) else proj_day
        
        p1_datetime = datetime.combine(p1_date, p1_time)
        p2_datetime = datetime.combine(p2_date, p2_time)
        
        if p2_datetime > p1_datetime:
            # Calculate contract slope and projections
            p1_ct = CT_TZ.localize(p1_datetime)
            time_diff_minutes = (p2_datetime - p1_datetime).total_seconds() / 60
            blocks_between = time_diff_minutes / 30
            
            if blocks_between > 0:
                contract_slope = (p2_price - p1_price) / blocks_between
                
                # Generate projections
                contract_projections = project_contract_line(p1_price, p1_ct, contract_slope, proj_day)
                
                # Store results
                st.session_state.contract_projections = contract_projections
                st.session_state.contract_config = {
                    'p1_price': p1_price,
                    'p1_time': p1_ct,
                    'p2_price': p2_price,
                    'p2_time': CT_TZ.localize(p2_datetime),
                    'slope': contract_slope,
                    'blocks': blocks_between
                }
                st.session_state.contract_ready = True

# ═══════════════════════════════════════════════════════════════════════════════
# CONTRACT PROJECTION FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def project_contract_line(anchor_price: float, anchor_time: datetime, 
                         slope: float, target_date: date) -> pd.DataFrame:
    """Project contract line across RTH using overnight slope"""
    rth_slots = rth_slots_ct(target_date)
    projections = []
    
    for slot_time in rth_slots:
        # Calculate 30-min blocks from anchor to slot
        time_diff = slot_time - anchor_time
        blocks = time_diff.total_seconds() / 1800
        
        projected_price = anchor_price + (slope * blocks)
        
        # Calculate entry probabilities based on overnight momentum
        entry_prob = calculate_contract_entry_probability(slot_time, slope, projected_price)
        
        projections.append({
            'Time': format_ct_time(slot_time),
            'Contract_Price': round(max(0.01, projected_price), 2),
            'Blocks_from_Anchor': round(blocks, 1),
            'Price_Change': round(projected_price - anchor_price, 2),
            'Entry_Probability': f"{entry_prob:.0f}%",
            'Momentum_Quality': get_momentum_quality(slope)
        })
    
    return pd.DataFrame(projections)

def calculate_contract_entry_probability(slot_time: datetime, slope: float, price: float) -> float:
    """Calculate contract entry probability based on overnight momentum and time"""
    # Base probability from overnight momentum
    momentum_strength = min(abs(slope) * 100, 50)  # Scale slope to probability
    base_prob = 60 + momentum_strength
    
    # Time-of-day adjustments
    hour = slot_time.hour
    if hour in [8, 9]:  # Market open high probability
        time_adj = 15
    elif hour in [13, 14]:  # End of day momentum
        time_adj = 8
    else:
        time_adj = 0
    
    # Price level adjustments (higher contract prices = lower probability)
    if price >= 20:
        price_adj = -10
    elif price >= 10:
        price_adj = -5
    else:
        price_adj = 0
    
    return min(85, max(40, base_prob + time_adj + price_adj))

def get_momentum_quality(slope: float) -> str:
    """Assess momentum quality based on slope"""
    abs_slope = abs(slope)
    if abs_slope >= 0.5:
        return "STRONG"
    elif abs_slope >= 0.2:
        return "MODERATE"
    else:
        return "WEAK"

# ═══════════════════════════════════════════════════════════════════════════════
# SPX BASELINE/SKYLINE ANALYSIS FOR CONTRACTS
# ═══════════════════════════════════════════════════════════════════════════════

def analyze_spx_contract_interactions(spx_data: pd.DataFrame, contract_proj: pd.DataFrame) -> pd.DataFrame:
    """Analyze SPX baseline/skyline interactions for contract bounce predictions"""
    if spx_data.empty or contract_proj.empty:
        return pd.DataFrame()
    
    # Detect SPX swings for anchor analysis
    spx_swings = detect_swings_simple(spx_data)
    skyline_anchor, baseline_anchor = get_anchor_points(spx_swings)
    
    interaction_analysis = []
    
    for idx, row in contract_proj.iterrows():
        time_slot = row['Time']
        contract_price = row['Contract_Price']
        
        # Simulate SPX price at this time (use projection or current data)
        current_spx = get_spx_price_estimate(spx_data, time_slot)
        
        analysis_row = {
            'Time': time_slot,
            'Contract_Price': contract_price,
            'SPX_Estimate': round(current_spx, 2) if current_spx else 'N/A'
        }
        
        # Baseline interaction analysis
        if baseline_anchor:
            baseline_price, _ = baseline_anchor
            distance_to_baseline = abs(current_spx - baseline_price) if current_spx else 0
            baseline_pct = (distance_to_baseline / baseline_price * 100) if baseline_price > 0 else 0
            
            if baseline_pct <= 0.5:  # Very close to baseline
                call_bounce_prob = 85
                call_action = "STRONG CALL ENTRY"
            elif baseline_pct <= 1.0:
                call_bounce_prob = 70
                call_action = "CALL ENTRY"
            else:
                call_bounce_prob = 45
                call_action = "MONITOR"
            
            analysis_row.update({
                'Baseline_Price': round(baseline_price, 2),
                'Baseline_Distance': f"{baseline_pct:.2f}%",
                'Call_Bounce_Prob': f"{call_bounce_prob:.0f}%",
                'Call_Action': call_action
            })
        
        # Skyline interaction analysis
        if skyline_anchor:
            skyline_price, _ = skyline_anchor
            distance_to_skyline = abs(current_spx - skyline_price) if current_spx else 0
            skyline_pct = (distance_to_skyline / skyline_price * 100) if skyline_price > 0 else 0
            
            # Check for skyline drops for put opportunities
            if current_spx and current_spx < skyline_price:
                drop_amount = skyline_price - current_spx
                drop_pct = (drop_amount / skyline_price * 100)
                
                if drop_pct >= 1.5:
                    put_entry_prob = 80
                    put_action = "STRONG PUT ENTRY"
                elif drop_pct >= 0.8:
                    put_entry_prob = 65
                    put_action = "PUT ENTRY"
                else:
                    put_entry_prob = 40
                    put_action = "MONITOR"
            else:
                put_entry_prob = 25
                put_action = "WAIT"
            
            analysis_row.update({
                'Skyline_Price': round(skyline_price, 2),
                'Put_Entry_Prob': f"{put_entry_prob:.0f}%",
                'Put_Action': put_action
            })
        
        interaction_analysis.append(analysis_row)
    
    return pd.DataFrame(interaction_analysis)

def get_spx_price_estimate(spx_data: pd.DataFrame, time_slot: str) -> float:
    """Estimate SPX price at specific time slot"""
    try:
        # Use last available close as estimate
        return spx_data.iloc[-1]['Close'] if not spx_data.empty else 0.0
    except:
        return 0.0

# ═══════════════════════════════════════════════════════════════════════════════
# RISK MANAGEMENT FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def calculate_contract_risk_management(projections: pd.DataFrame, config: dict, spx_data: pd.DataFrame) -> pd.DataFrame:
    """Calculate risk management based on real market conditions"""
    if projections.empty:
        return pd.DataFrame()
    
    # Get market volatility from SPX data
    if not spx_data.empty:
        recent_closes = spx_data['Close'].tail(20)
        daily_returns = recent_closes.pct_change().dropna()
        market_vol = daily_returns.std() * 100
        
        # Average daily range
        if len(spx_data) > 0:
            daily_ranges = ((spx_data['High'] - spx_data['Low']) / spx_data['Close'] * 100).tail(10)
            avg_daily_range = daily_ranges.mean()
        else:
            avg_daily_range = 2.0
    else:
        market_vol = 1.5
        avg_daily_range = 2.0
    
    risk_analysis = []
    
    # Contract volatility assessment
    contract_vol = abs(config.get('slope', 0)) * 15
    
    # Volatility regime
    if market_vol >= 2.5:
        vol_regime = "HIGH"
        risk_multiplier = 1.8
    elif market_vol >= 1.5:
        vol_regime = "MODERATE"
        risk_multiplier = 1.2
    else:
        vol_regime = "LOW"
        risk_multiplier = 0.8
    
    # Time-based risk factors
    time_risk_map = {
        "08:30": 2.0, "09:00": 1.8, "09:30": 1.5, "10:00": 1.3,
        "10:30": 1.0, "11:00": 1.0, "11:30": 1.0, "12:00": 1.0,
        "12:30": 1.0, "13:00": 1.2, "13:30": 1.4, "14:00": 1.5, "14:30": 1.3
    }
    
    for idx, row in projections.iterrows():
        time_slot = row['Time']
        entry_price = row['Contract_Price']
        
        time_risk = time_risk_map.get(time_slot, 1.0)
        
        # Dynamic risk calculation
        base_risk_pct = 15  # 15% base risk for contracts
        adjusted_risk_pct = base_risk_pct * risk_multiplier * time_risk
        risk_amount = entry_price * (adjusted_risk_pct / 100)
        
        # TP calculations
        tp1_amount = risk_amount * 1.5
        tp2_amount = risk_amount * 2.5
        
        # Stop price
        stop_price = max(0.01, entry_price - risk_amount)
        tp1_price = entry_price + tp1_amount
        tp2_price = entry_price + tp2_amount
        
        # Market confidence
        confidence = calculate_market_confidence_score(market_vol, contract_vol, time_risk)
        
        risk_analysis.append({
            'Time': time_slot,
            'Entry': round(entry_price, 2),
            'Stop': round(stop_price, 2),
            'TP1': round(tp1_price, 2),
            'TP2': round(tp2_price, 2),
            'Risk_Amount': round(risk_amount, 2),
            'Risk_Pct': f"{adjusted_risk_pct:.1f}%",
            'Vol_Regime': vol_regime,
            'Time_Risk': f"{time_risk:.1f}x",
            'Market_Confidence': f"{confidence:.0f}%",
            'Trade_Quality': get_contract_trade_quality(confidence)
        })
    
    return pd.DataFrame(risk_analysis)

def calculate_market_confidence_score(market_vol: float, contract_vol: float, time_risk: float) -> float:
    """Calculate market confidence for contract entries"""
    base_confidence = 65
    
    # Volatility alignment
    if 0.7 <= contract_vol/market_vol <= 1.3:
        vol_bonus = 15
    else:
        vol_bonus = 0
    
    # Time penalty for high-risk periods
    time_penalty = (time_risk - 1) * 8
    
    confidence = base_confidence + vol_bonus - time_penalty
    return max(25, min(90, confidence))

def get_contract_trade_quality(confidence: float) -> str:
    """Determine contract trade quality"""
    if confidence >= 75:
        return "HIGH"
    elif confidence >= 60:
        return "MODERATE"
    else:
        return "LOW"

with tab4:
    st.subheader("Contract Tool")
    st.caption("Overnight contract price analysis for RTH entry optimization")
    
    # ═══════════════════════════════════════════════════════════════════════════════
    # TWO-POINT INPUT SYSTEM WITH AUTO-UPDATE
    # ═══════════════════════════════════════════════════════════════════════════════
    
    st.markdown("Overnight Contract Price Points")
    
    point_col1, point_col2 = st.columns(2)
    
    with point_col1:
        st.markdown("**Point 1 (Earlier)**")
        
        p1_time = st.time_input(
            "Point 1 Time (CT)",
            value=time(20, 0),
            key="ct_p1_time",
            help="Between 20:00 prev day and 10:00 current day",
            on_change=update_contract_analysis
        )
        
        p1_price = st.number_input(
            "Point 1 Contract Price",
            value=10.0,
            min_value=0.01,
            step=0.01, format="%.2f",
            key="ct_p1_price",
            on_change=update_contract_analysis
        )
    
    with point_col2:
        st.markdown("**Point 2 (Later)**")
        
        p2_time = st.time_input(
            "Point 2 Time (CT)",
            value=time(8, 0),
            key="ct_p2_time",
            help="Between 20:00 prev day and 10:00 current day",
            on_change=update_contract_analysis
        )
        
        p2_price = st.number_input(
            "Point 2 Contract Price",
            value=12.0,
            min_value=0.01,
            step=0.01, format="%.2f",
            key="ct_p2_price",
            on_change=update_contract_analysis
        )
    
    # Projection day
    projection_day = st.date_input(
        "RTH Projection Day",
        value=datetime.now(CT_TZ).date(),
        key="ct_proj_day",
        on_change=update_contract_analysis
    )
    
    # Live slope calculation and validation
    p1_date = projection_day - timedelta(days=1) if p1_time >= time(20, 0) else projection_day
    p2_date = projection_day if p2_time <= time(10, 0) else projection_day
    
    p1_datetime = datetime.combine(p1_date, p1_time)
    p2_datetime = datetime.combine(p2_date, p2_time)
    
    if p2_datetime <= p1_datetime:
        st.error("Point 2 must be after Point 1")
    else:
        # Show live calculations
        time_diff_minutes = (p2_datetime - p1_datetime).total_seconds() / 60
        blocks_between = time_diff_minutes / 30
        contract_slope = (p2_price - p1_price) / blocks_between if blocks_between > 0 else 0
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Time Span", f"{time_diff_minutes/60:.1f} hours")
        with col2:
            st.metric("30-min Blocks", f"{blocks_between:.1f}")
        with col3:
            st.metric("Slope per Block", f"{contract_slope:+.3f}")
    
    st.markdown("---")
    
    # ═══════════════════════════════════════════════════════════════════════════════
    # AUTOMATIC RESULTS DISPLAY
    # ═══════════════════════════════════════════════════════════════════════════════
    
    if st.session_state.get('contract_ready', False):
        st.subheader("Contract Analysis Results")
        
        projections = st.session_state.contract_projections
        config = st.session_state.contract_config
        
        # Fetch SPX data for baseline/skyline analysis
        spx_data = fetch_live_data("^GSPC", projection_day - timedelta(days=1), projection_day)
        
        # Analysis tabs
        contract_tabs = st.tabs(["RTH Projections", "SPX Interactions", "Risk Management"])
        
        with contract_tabs[0]:  # RTH Projections
            st.subheader("RTH Contract Price Projections")
            st.dataframe(projections, use_container_width=True, hide_index=True)
            
            # Key entry levels
            if not projections.empty:
                # Extract probability values for sorting
                prob_values = []
                for prob_str in projections['Entry_Probability']:
                    prob_values.append(float(prob_str.replace('%', '')))
                
                projections_with_prob = projections.copy()
                projections_with_prob['Prob_Value'] = prob_values
                
                # Get top 3 entry opportunities
                top_entries = projections_with_prob.nlargest(3, 'Prob_Value')
                
                st.subheader("Top Entry Opportunities")
                
                key_levels = []
                for idx, row in top_entries.iterrows():
                    key_levels.append({
                        'Time': row['Time'],
                        'Contract_Price': row['Contract_Price'],
                        'Entry_Probability': row['Entry_Probability'],
                        'Momentum': row['Momentum_Quality'],
                        'Recommendation': get_entry_recommendation(row['Prob_Value'])
                    })
                
                key_df = pd.DataFrame(key_levels)
                st.dataframe(key_df, use_container_width=True, hide_index=True)
        
        with contract_tabs[1]:  # SPX Interactions
            if not spx_data.empty:
                spx_interactions = analyze_spx_contract_interactions(spx_data, projections)
                st.subheader("SPX Baseline/Skyline Contract Analysis")
                st.dataframe(spx_interactions, use_container_width=True, hide_index=True)
                
                # Summary insights
                if not spx_interactions.empty:
                    call_opportunities = len(spx_interactions[spx_interactions['Call_Action'].str.contains('CALL', na=False)])
                    put_opportunities = len(spx_interactions[spx_interactions['Put_Action'].str.contains('PUT', na=False)]) if 'Put_Action' in spx_interactions.columns else 0
                    
                    insight_col1, insight_col2 = st.columns(2)
                    with insight_col1:
                        st.metric("Call Opportunities", call_opportunities)
                    with insight_col2:
                        st.metric("Put Opportunities", put_opportunities)
            else:
                st.warning("No SPX data available for interaction analysis")
        
        with contract_tabs[2]:  # Risk Management
            risk_analysis = calculate_contract_risk_management(projections, config, spx_data)
            st.subheader("Contract Risk Management")
            st.dataframe(risk_analysis, use_container_width=True, hide_index=True)
            
            # Risk summary
            if not risk_analysis.empty:
                avg_risk = risk_analysis['Risk_Pct'].str.replace('%', '').astype(float).mean()
                high_confidence_trades = len(risk_analysis[risk_analysis['Trade_Quality'] == 'HIGH'])
                
                risk_col1, risk_col2 = st.columns(2)
                with risk_col1:
                    st.metric("Average Risk", f"{avg_risk:.1f}%")
                with risk_col2:
                    st.metric("High Quality Trades", high_confidence_trades)
    
    else:
        st.info("Configure contract points above to automatically generate analysis")

def get_entry_recommendation(prob_value: float) -> str:
    """Get entry recommendation based on probability"""
    if prob_value >= 75:
        return "STRONG BUY"
    elif prob_value >= 65:
        return "BUY"
    elif prob_value >= 55:
        return "MODERATE"
    else:
        return "WEAK"

# ═══════════════════════════════════════════════════════════════════════════════
# END OF CONTRACT TOOL TAB
# ═══════════════════════════════════════════════════════════════════════════════







# ═══════════════════════════════════════════════════════════════════════════════
# SPX PROPHET - PART 6: FINAL INTEGRATION & PROFITABILITY ENHANCEMENTS
# ═══════════════════════════════════════════════════════════════════════════════

# Initialize all session state variables if not exists
required_states = [
    'spx_analysis_ready', 'stock_analysis_ready', 'signal_ready', 'contract_ready',
    'spx_slopes', 'stock_slopes', 'current_offset', 'theme'
]

for state in required_states:
    if state not in st.session_state:
        if 'slopes' in state:
            st.session_state[state] = SPX_SLOPES.copy() if 'spx' in state else STOCK_SLOPES.copy()
        elif state == 'current_offset':
            st.session_state[state] = 0.0
        elif state == 'theme':
            st.session_state[state] = 'Dark'
        else:
            st.session_state[state] = False

# ═══════════════════════════════════════════════════════════════════════════════
# ADVANCED PROFITABILITY ANALYTICS
# ═══════════════════════════════════════════════════════════════════════════════

def calculate_cross_timeframe_confluence(symbol: str, target_date: date) -> pd.DataFrame:
    """Analyze confluence across multiple timeframes for higher probability entries"""
    confluence_data = []
    
    # Fetch 1-hour data for higher timeframe context
    try:
        ticker = yf.Ticker(symbol)
        hourly_data = ticker.history(start=target_date - timedelta(days=5), 
                                   end=target_date + timedelta(days=1), 
                                   interval="1h")
        
        if not hourly_data.empty:
            # Convert to CT
            if hourly_data.index.tz is None:
                hourly_data.index = hourly_data.index.tz_localize('US/Eastern')
            hourly_data.index = hourly_data.index.tz_convert(CT_TZ)
            
            # Calculate higher timeframe EMAs
            hourly_ema8 = calculate_ema(hourly_data['Close'], 8)
            hourly_ema21 = calculate_ema(hourly_data['Close'], 21)
            
            # Check alignment for RTH hours
            rth_slots = rth_slots_ct(target_date)
            
            for slot_time in rth_slots:
                try:
                    # Find closest hourly bar
                    closest_bar = hourly_data.iloc[hourly_data.index.get_indexer([slot_time], method='nearest')[0]]
                    closest_idx = hourly_data.index.get_loc(closest_bar.name)
                    
                    # Check EMA alignment
                    h_ema8 = hourly_ema8.iloc[closest_idx]
                    h_ema21 = hourly_ema21.iloc[closest_idx]
                    h_price = closest_bar['Close']
                    
                    # Confluence scoring
                    ema_alignment = "Bullish" if h_ema8 > h_ema21 else "Bearish"
                    price_vs_ema = "Above" if h_price > h_ema8 else "Below"
                    
                    # Calculate confluence score
                    confluence_score = 50  # Base score
                    if ema_alignment == "Bullish" and price_vs_ema == "Above":
                        confluence_score += 25
                    elif ema_alignment == "Bearish" and price_vs_ema == "Below":
                        confluence_score += 25
                    
                    confluence_data.append({
                        'Time': format_ct_time(slot_time),
                        'HTF_EMA_Trend': ema_alignment,
                        'Price_vs_EMA8': price_vs_ema,
                        'HTF_Price': round(h_price, 2),
                        'Confluence_Score': confluence_score,
                        'Trade_Setup': get_confluence_setup(confluence_score)
                    })
                    
                except:
                    continue
                    
    except:
        # Return basic structure if data fetch fails
        for slot_time in rth_slots_ct(target_date):
            confluence_data.append({
                'Time': format_ct_time(slot_time),
                'HTF_EMA_Trend': 'Unknown',
                'Price_vs_EMA8': 'Unknown',
                'HTF_Price': 'N/A',
                'Confluence_Score': 50,
                'Trade_Setup': 'NEUTRAL'
            })
    
    return pd.DataFrame(confluence_data)

def get_confluence_setup(score: float) -> str:
    """Determine trade setup quality based on confluence"""
    if score >= 75:
        return "EXCELLENT"
    elif score >= 65:
        return "GOOD"
    elif score >= 55:
        return "FAIR"
    else:
        return "POOR"

def calculate_volume_profile_analysis(data: pd.DataFrame) -> pd.DataFrame:
    """Calculate volume profile for support/resistance levels"""
    if data.empty or 'Volume' not in data.columns:
        return pd.DataFrame()
    
    # Create price buckets
    price_min = data['Low'].min()
    price_max = data['High'].max()
    price_range = price_max - price_min
    bucket_size = price_range / 20  # 20 price levels
    
    volume_profile = []
    
    for i in range(20):
        bucket_low = price_min + (i * bucket_size)
        bucket_high = bucket_low + bucket_size
        bucket_mid = (bucket_low + bucket_high) / 2
        
        # Calculate volume in this price range
        bucket_volume = 0
        for idx, bar in data.iterrows():
            if bucket_low <= bar['Low'] <= bucket_high or bucket_low <= bar['High'] <= bucket_high:
                bucket_volume += bar['Volume']
        
        volume_profile.append({
            'Price_Level': round(bucket_mid, 2),
            'Volume': bucket_volume,
            'Support_Strength': get_volume_strength(bucket_volume, data['Volume'].max())
        })
    
    # Sort by volume to find key levels
    profile_df = pd.DataFrame(volume_profile)
    return profile_df.nlargest(10, 'Volume')

def get_volume_strength(volume: float, max_volume: float) -> str:
    """Determine support/resistance strength based on volume"""
    if max_volume == 0:
        return "Unknown"
    
    ratio = volume / max_volume
    if ratio >= 0.7:
        return "Very Strong"
    elif ratio >= 0.5:
        return "Strong"
    elif ratio >= 0.3:
        return "Moderate"
    else:
        return "Weak"

def generate_daily_trading_plan() -> pd.DataFrame:
    """Generate comprehensive daily trading plan based on all active analyses"""
    trading_plan = []
    
    # Combine insights from all active analyses
    current_time = datetime.now(CT_TZ)
    
    # SPX opportunities
    if st.session_state.get('spx_analysis_ready', False):
        trading_plan.append({
            'Time': current_time.strftime("%H:%M"),
            'Asset': 'SPX',
            'Opportunity': 'Anchor analysis active',
            'Probability': '80%',
            'Action': 'Monitor skyline/baseline touches',
            'Priority': 'HIGH'
        })
    
    # Stock opportunities
    if st.session_state.get('stock_analysis_ready', False):
        ticker = st.session_state.get('stock_analysis_ticker', 'STOCK')
        trading_plan.append({
            'Time': current_time.strftime("%H:%M"),
            'Asset': ticker,
            'Opportunity': 'Weekly projection active',
            'Probability': '75%',
            'Action': 'Monitor anchor line interactions',
            'Priority': 'MEDIUM'
        })
    
    # Contract opportunities
    if st.session_state.get('contract_ready', False):
        trading_plan.append({
            'Time': current_time.strftime("%H:%M"),
            'Asset': 'CONTRACTS',
            'Opportunity': 'Overnight momentum analysis',
            'Probability': '70%',
            'Action': 'Watch for RTH entry points',
            'Priority': 'HIGH'
        })
    
    return pd.DataFrame(trading_plan)

# ═══════════════════════════════════════════════════════════════════════════════
# PROFITABILITY DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown("---")
st.subheader("Trading Profitability Dashboard")

# Real-time trading opportunities
if any([
    st.session_state.get('spx_analysis_ready', False),
    st.session_state.get('stock_analysis_ready', False),
    st.session_state.get('signal_ready', False),
    st.session_state.get('contract_ready', False)
]):
    
    dashboard_tabs = st.tabs(["Trading Plan", "Market Context", "Risk Assessment", "Performance Edge"])
    
    with dashboard_tabs[0]:  # Trading Plan
        daily_plan = generate_daily_trading_plan()
        st.subheader("Today's Trading Plan")
        st.dataframe(daily_plan, use_container_width=True, hide_index=True)
        
        # Priority actions
        if not daily_plan.empty:
            high_priority = daily_plan[daily_plan['Priority'] == 'HIGH']
            if not high_priority.empty:
                st.subheader("High Priority Actions")
                for idx, action in high_priority.iterrows():
                    st.info(f"{action['Asset']}: {action['Action']}")
    
    with dashboard_tabs[1]:  # Market Context
        # Cross-timeframe analysis for active symbol
        if st.session_state.get('signal_ready', False):
            symbol = st.session_state.signal_symbol
            analysis_date = st.session_state.get('sig_day', datetime.now().date())
            
            confluence_analysis = calculate_cross_timeframe_confluence(symbol, analysis_date)
            st.subheader(f"{symbol} Multi-Timeframe Confluence")
            st.dataframe(confluence_analysis, use_container_width=True, hide_index=True)
        
        # Volume profile analysis
        if st.session_state.get('signal_ready', False):
            signal_data = st.session_state.get('signal_data', pd.DataFrame())
            if not signal_data.empty:
                volume_profile = calculate_volume_profile_analysis(signal_data)
                st.subheader("Volume Profile (Key Levels)")
                st.dataframe(volume_profile, use_container_width=True, hide_index=True)
    
    with dashboard_tabs[2]:  # Risk Assessment
        st.subheader("Current Risk Assessment")
        
        # Market volatility status
        current_time_ct = datetime.now(CT_TZ)
        market_open_time = current_time_ct.replace(hour=8, minute=30, second=0, microsecond=0)
        market_close_time = current_time_ct.replace(hour=14, minute=30, second=0, microsecond=0)
        is_weekday = current_time_ct.weekday() < 5
        within_hours = market_open_time <= current_time_ct <= market_close_time
        is_market_open = is_weekday and within_hours
        
        risk_col1, risk_col2 = st.columns(2)
        
        with risk_col1:
            st.markdown("**Market Status**")
            if is_market_open:
                time_to_close = market_close_time - current_time_ct
                hours_left = int(time_to_close.total_seconds() // 3600)
                minutes_left = int((time_to_close.total_seconds() % 3600) // 60)
                st.success(f"Market Open - {hours_left}h {minutes_left}m remaining")
                
                # Time-based risk assessment
                if hours_left <= 1:
                    st.warning("End of day - increased volatility risk")
                elif current_time_ct.hour == 8:
                    st.warning("Market open - high volatility period")
                else:
                    st.info("Normal trading hours - standard risk")
            else:
                st.error("Market Closed")
        
        with risk_col2:
            st.markdown("**Active Analysis Risk**")
            
            total_active = sum([
                st.session_state.get('spx_analysis_ready', False),
                st.session_state.get('stock_analysis_ready', False),
                st.session_state.get('contract_ready', False)
            ])
            
            if total_active >= 3:
                st.warning("High concentration - multiple active analyses")
            elif total_active >= 2:
                st.info("Moderate activity - good diversification")
            elif total_active == 1:
                st.success("Single focus - concentrated analysis")
            else:
                st.info("No active analyses")
    
    with dashboard_tabs[3]:  # Performance Edge
        st.subheader("Performance Optimization")
        
        # Time-of-day edge analysis
        current_hour = datetime.now(CT_TZ).hour
        
        if 8 <= current_hour <= 14:  # During RTH
            edge_analysis = []
            
            # Optimal entry times based on your strategy
            optimal_times = {
                'SPX Momentum': ['09:00', '09:30', '13:30'],
                'Stock Breakouts': ['09:30', '10:00', '14:00'],
                'Contract Entries': ['08:30', '09:00', '13:00'],
                'Anchor Bounces': ['09:00', '10:30', '13:30']
            }
            
            current_time_str = f"{current_hour:02d}:{datetime.now(CT_TZ).minute//30*30:02d}"
            
            for strategy, times in optimal_times.items():
                is_optimal = current_time_str in times
                edge_score = 85 if is_optimal else 60
                
                edge_analysis.append({
                    'Strategy': strategy,
                    'Current_Edge': f"{edge_score}%",
                    'Optimal_Times': ', '.join(times),
                    'Status': 'OPTIMAL' if is_optimal else 'STANDARD'
                })
            
            edge_df = pd.DataFrame(edge_analysis)
            st.dataframe(edge_df, use_container_width=True, hide_index=True)
        
        # Anchor line reliability assessment
        if st.session_state.get('spx_analysis_ready', False):
            st.subheader("Anchor Line Reliability")
            
            reliability_data = []
            for anchor_type in ['HIGH', 'CLOSE', 'LOW', 'SKYLINE', 'BASELINE']:
                if anchor_type in ['SKYLINE', 'BASELINE']:
                    reliability = 80
                    edge = "Primary anchor - highest probability"
                else:
                    reliability = 70
                    edge = "Secondary anchor - good probability"
                
                reliability_data.append({
                    'Anchor_Type': anchor_type,
                    'Reliability': f"{reliability}%",
                    'Trading_Edge': edge,
                    'Recommended_Use': 'Primary' if reliability >= 80 else 'Secondary'
                })
            
            reliability_df = pd.DataFrame(reliability_data)
            st.dataframe(reliability_df, use_container_width=True, hide_index=True)

else:
    st.info("No active analyses. Use the tabs above to start market analysis.")

# ═══════════════════════════════════════════════════════════════════════════════
# QUICK ACTIONS PANEL
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown("---")
st.subheader("Quick Actions")

action_col1, action_col2, action_col3, action_col4 = st.columns(4)

with action_col1:
    if st.button("Update All Offsets", key="quick_update_all_offsets"):
        # Update SPX offset
        today = datetime.now(CT_TZ).date()
        yesterday = today - timedelta(days=1)
        
        es_data = fetch_live_data("ES=F", yesterday, today)
        spx_data = fetch_live_data("^GSPC", yesterday, today)
        
        if not es_data.empty and not spx_data.empty:
            new_offset = calculate_es_spx_offset(es_data, spx_data)
            st.session_state.current_offset = new_offset
            st.success(f"Offset updated: {new_offset:+.1f}")
        else:
            st.error("Failed to update offset")

with action_col2:
    if st.button("Reset All Analysis", key="quick_reset_all"):
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
    if st.button("Reset All Slopes", key="quick_reset_slopes"):
        st.session_state.spx_slopes = SPX_SLOPES.copy()
        st.session_state.stock_slopes = STOCK_SLOPES.copy()
        st.success("Slopes reset to defaults")
        st.rerun()

with action_col4:
    # Quick market status check
    current_time_ct = datetime.now(CT_TZ)
    is_weekday = current_time_ct.weekday() < 5
    within_hours = time(8, 30) <= current_time_ct.time() <= time(14, 30)
    
    if is_weekday and within_hours:
        st.success("Market OPEN")
    elif is_weekday:
        st.warning("Market CLOSED")
    else:
        st.info("WEEKEND")

# ═══════════════════════════════════════════════════════════════════════════════
# LIVE TRADING ALERTS
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown("---")
st.subheader("Live Trading Context")

# Show current market conditions
current_time = datetime.now(CT_TZ)
context_col1, context_col2, context_col3 = st.columns(3)

with context_col1:
    st.markdown("**Current Session**")
    if current_time.weekday() < 5 and time(8, 30) <= current_time.time() <= time(14, 30):
        session_progress = ((current_time.hour - 8) * 60 + (current_time.minute - 30)) / 360 * 100
        st.progress(session_progress / 100)
        st.caption(f"Session {session_progress:.0f}% complete")
    else:
        st.caption("Market closed")

with context_col2:
    st.markdown("**Analysis Status**")
    status_count = sum([
        st.session_state.get('spx_analysis_ready', False),
        st.session_state.get('stock_analysis_ready', False),
        st.session_state.get('signal_ready', False),
        st.session_state.get('contract_ready', False)
    ])
    
    st.metric("Active Analyses", f"{status_count}/4")
    if status_count >= 3:
        st.success("Comprehensive analysis active")
    elif status_count >= 2:
        st.info("Good analysis coverage")
    else:
        st.warning("Limited analysis - consider more setups")

with context_col3:
    st.markdown("**Strategy Focus**")
    
    # Determine primary strategy based on active analyses
    if st.session_state.get('contract_ready', False):
        primary_strategy = "Contract Analysis"
        focus_color = "blue"
    elif st.session_state.get('spx_analysis_ready', False):
        primary_strategy = "SPX Anchors"
        focus_color = "green"
    elif st.session_state.get('stock_analysis_ready', False):
        primary_strategy = "Stock Analysis"
        focus_color = "orange"
    else:
        primary_strategy = "Setup Required"
        focus_color = "gray"
    
    st.markdown(f"""
    <div style="text-align: center; padding: 1rem; background: rgba(255,255,255,0.1); border-radius: 10px; border-left: 4px solid {focus_color};">
        <h4>{primary_strategy}</h4>
    </div>
    """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# FOOTER
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown("---")
st.markdown(
    f"""
    <div style='text-align: center; color: #888; font-size: 0.9em; padding: 1rem;'>
        SPX Prophet Analytics • Real-time Market Analysis • 
        Session: {datetime.now(CT_TZ).strftime('%H:%M:%S CT')} • 
        Theme: {st.session_state.theme} • 
        Offset: {st.session_state.current_offset:+.1f}
    </div>
    """, 
    unsafe_allow_html=True
)

# Error handling for the entire application
try:
    # Validate critical session state
    missing_states = [state for state in required_states if state not in st.session_state]
    
    if missing_states:
        st.error(f"Missing session state: {', '.join(missing_states)}. Please refresh the app.")
        
except Exception as e:
    st.error(f"Application error: {str(e)}")
    st.info("Please refresh the page to reset the application.")

# ═══════════════════════════════════════════════════════════════════════════════
# APPLICATION COMPLETE
# ═══════════════════════════════════════════════════════════════════════════════




