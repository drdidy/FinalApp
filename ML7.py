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
        'SKYLINE': 80.0,  # Corrected to 80%
        'BASELINE': 80.0, # Corrected to 80%
        'HIGH': 75.0,
        'CLOSE': 80.0,
        'LOW': 75.0
    }
    
    base_prob = base_probs.get(anchor_type.upper(), 70.0)
    
    # Time adjustments for volatility periods
    hour = int(time_slot.split(':')[0])
    if hour in [8, 9]:  # Market open volatility
        time_adj = 8
    elif hour in [13, 14]:  # End of day momentum
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

def calculate_anchor_line_analytics(projection_df: pd.DataFrame, anchor_type: str, market_data: pd.DataFrame) -> pd.DataFrame:
    """Calculate statistical analysis of anchor line interactions for profitability"""
    if projection_df.empty:
        return pd.DataFrame()
    
    analytics = []
    
    # Bounce rate analysis
    bounce_strength = 80.0 if anchor_type in ['SKYLINE', 'BASELINE'] else 70.0
    penetration_risk = 20.0 if anchor_type in ['SKYLINE', 'BASELINE'] else 30.0
    
    # Volume profile strength
    if not market_data.empty and 'Volume' in market_data.columns:
        avg_volume = market_data['Volume'].mean()
        volume_strength = min(95, max(50, (avg_volume / 1000000) * 10))  # Scale volume
    else:
        volume_strength = 70.0
    
    # Time-based edge analysis
    for idx, row in projection_df.iterrows():
        time_slot = row['Time']
        price = row['Price']
        
        # Calculate time edge
        hour = int(time_slot.split(':')[0])
        time_edge = 15 if hour in [8, 9, 13, 14] else 5  # Higher edge during volatile periods
        
        # Momentum alignment score
        if anchor_type in ['SKYLINE', 'BASELINE']:
            momentum_score = 85  # Strong momentum at key levels
        else:
            momentum_score = 70  # Moderate momentum at other anchors
        
        # Overall confluence score
        confluence = (bounce_strength * 0.3) + (volume_strength * 0.2) + (momentum_score * 0.3) + (time_edge * 0.2)
        
        analytics.append({
            'Time': time_slot,
            'Bounce_Rate': f"{bounce_strength:.0f}%",
            'Penetration_Risk': f"{penetration_risk:.0f}%", 
            'Volume_Strength': f"{volume_strength:.0f}%",
            'Time_Edge': f"{time_edge:.0f}%",
            'Momentum_Score': f"{momentum_score:.0f}%",
            'Confluence': f"{confluence:.0f}%",
            'Trade_Quality': get_trade_quality(confluence)
        })
    
    return pd.DataFrame(analytics)

def get_trade_quality(confluence_score: float) -> str:
    """Determine trade quality based on confluence score"""
    if confluence_score >= 85:
        return "EXCELLENT"
    elif confluence_score >= 75:
        return "GOOD"
    elif confluence_score >= 65:
        return "MODERATE"
    else:
        return "WEAK"

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
                    st.subheader("Skyline SPX Projection (80% Zone)")
                    st.info("Strategy: Bearish candle touches from above + closes above = BUY signal")
                    st.dataframe(skyline_proj, use_container_width=True, hide_index=True)
                    
                    # Add probability analytics
                    skyline_analytics = calculate_anchor_line_analytics(skyline_proj, "SKYLINE", es_data)
                    st.subheader("Anchor Line Analytics")
                    st.dataframe(skyline_analytics, use_container_width=True, hide_index=True)
                    
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
                    st.subheader("Baseline SPX Projection (80% Zone)")
                    st.info("Strategy: Bearish candle touches from above + closes above = BUY signal")
                    st.dataframe(baseline_proj, use_container_width=True, hide_index=True)
                    
                    # Add probability analytics
                    baseline_analytics = calculate_anchor_line_analytics(baseline_proj, "BASELINE", es_data)
                    st.subheader("Anchor Line Analytics")
                    st.dataframe(baseline_analytics, use_container_width=True, hide_index=True)
                    
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











