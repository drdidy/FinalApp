# ═══════════════════════════════════════════════════════════════════════════════
# SPX PROPHET - PART 1: FOUNDATION & DATA HANDLING
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

# ============================================================================
# CORE CONFIGURATION
# ============================================================================
CT_TZ = pytz.timezone('America/Chicago')
UTC_TZ = pytz.UTC

# Trading parameters
SWING_K = 2  # Fixed swing detection parameter
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

# ============================================================================
# STREAMLIT CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="SPX Prophet Analytics",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Basic styling
st.markdown("""
<style>
    .main { background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%); }
    .metric-container { 
        background: rgba(255,255,255,0.05);
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid rgba(255,255,255,0.1);
    }
    .stTab { background: rgba(255,255,255,0.02); }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# DATA FETCHING FUNCTIONS
# ============================================================================
@st.cache_data(ttl=60)  # Live data cache 60s
def fetch_live_data(symbol: str, start_date: date, end_date: date) -> pd.DataFrame:
    """Fetch live yfinance data and normalize to CT timezone"""
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(start=start_date, end=end_date + timedelta(days=1), 
                           interval="30m", prepost=True)
        
        if df.empty:
            return pd.DataFrame()
            
        # Flatten MultiIndex if present
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
            
        # Convert to CT timezone
        df.index = df.index.tz_convert(CT_TZ)
        
        # Validate data quality
        if not price_range_ok(df):
            st.warning(f"Data quality check failed for {symbol}")
            return pd.DataFrame()
            
        return df
        
    except Exception as e:
        st.error(f"Failed to fetch {symbol}: {str(e)}")
        return pd.DataFrame()

@st.cache_data(ttl=300)  # Historical data cache 300s
def fetch_historical_data(symbol: str, days: int = 60) -> pd.DataFrame:
    """Fetch historical data for backtesting and analysis"""
    try:
        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=days)
        return fetch_live_data(symbol, start_date, end_date)
    except Exception as e:
        st.error(f"Historical fetch failed for {symbol}: {str(e)}")
        return pd.DataFrame()

def price_range_ok(df: pd.DataFrame) -> bool:
    """Validate data quality"""
    if df.empty or 'Close' not in df.columns:
        return False
    
    close_prices = df['Close'].dropna()
    if close_prices.empty or (close_prices <= 0).any():
        return False
        
    # Check for reasonable high/low ratio
    if 'High' in df.columns and 'Low' in df.columns:
        ratios = df['High'] / df['Low']
        if (ratios > 5).any():  # Suspicious data
            return False
            
    return True

# ============================================================================
# TIME HANDLING FUNCTIONS
# ============================================================================
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

# ============================================================================
# SWING DETECTION FUNCTIONS
# ============================================================================
def detect_swings(df: pd.DataFrame, k: int = SWING_K) -> pd.DataFrame:
    """Detect swing highs and lows using CLOSE prices only with k=2"""
    if df.empty or len(df) < (2*k + 1):
        return df.copy()
    
    df_swings = df.copy()
    df_swings['swing_high'] = False
    df_swings['swing_low'] = False
    
    closes = df_swings['Close'].values
    
    # Detect swing highs (k bars on each side must be lower)
    for i in range(k, len(closes) - k):
        is_high = True
        for j in range(i - k, i + k + 1):
            if j != i and closes[j] >= closes[i]:
                is_high = False
                break
        df_swings.iloc[i, df_swings.columns.get_loc('swing_high')] = is_high
    
    # Detect swing lows (k bars on each side must be higher)
    for i in range(k, len(closes) - k):
        is_low = True
        for j in range(i - k, i + k + 1):
            if j != i and closes[j] <= closes[i]:
                is_low = False
                break
        df_swings.iloc[i, df_swings.columns.get_loc('swing_low')] = is_low
    
    return df_swings

def get_anchor_points(df_swings: pd.DataFrame) -> Tuple[Optional[Tuple], Optional[Tuple]]:
    """Extract skyline (highest swing high) and baseline (lowest swing low)"""
    swing_highs = df_swings[df_swings['swing_high'] == True]
    swing_lows = df_swings[df_swings['swing_low'] == True]
    
    skyline = None
    baseline = None
    
    if not swing_highs.empty:
        # Highest swing high, break ties by volume
        max_price = swing_highs['Close'].max()
        candidates = swing_highs[swing_highs['Close'] == max_price]
        if 'Volume' in candidates.columns:
            best = candidates.loc[candidates['Volume'].idxmax()]
        else:
            best = candidates.iloc[0]
        skyline = (best['Close'], best.name)
    
    if not swing_lows.empty:
        # Lowest swing low, break ties by volume
        min_price = swing_lows['Close'].min()
        candidates = swing_lows[swing_lows['Close'] == min_price]
        if 'Volume' in candidates.columns:
            best = candidates.loc[candidates['Volume'].idxmax()]
        else:
            best = candidates.iloc[0]
        baseline = (best['Close'], best.name)
    
    return skyline, baseline

# ============================================================================
# PROJECTION FUNCTIONS
# ============================================================================
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
            'Blocks': round(blocks, 1)
        })
    
    return pd.DataFrame(projections)

# ============================================================================
# INDICATORS & ANALYSIS
# ============================================================================
def calculate_ema(series: pd.Series, span: int) -> pd.Series:
    """Calculate EMA for given span"""
    return series.ewm(span=span).mean()

def calculate_vwap(df: pd.DataFrame) -> pd.Series:
    """Calculate intraday VWAP"""
    if df.empty or 'Volume' not in df.columns:
        return pd.Series(index=df.index)
    
    # Reset VWAP at start of each day
    df_copy = df.copy()
    df_copy['date'] = df_copy.index.date
    
    vwap_values = []
    for date_group in df_copy.groupby('date'):
        day_data = date_group[1]
        typical_price = (day_data['High'] + day_data['Low'] + day_data['Close']) / 3
        cum_vol = day_data['Volume'].cumsum()
        cum_vol_price = (typical_price * day_data['Volume']).cumsum()
        day_vwap = cum_vol_price / cum_vol
        vwap_values.extend(day_vwap.tolist())
    
    return pd.Series(vwap_values, index=df.index)

def calculate_es_spx_offset(es_data: pd.DataFrame, spx_data: pd.DataFrame) -> float:
    """Calculate ES to SPX offset using last available close"""
    try:
        # Get last common timestamp
        es_last = es_data.iloc[-1]['Close'] if not es_data.empty else 0
        spx_last = spx_data.iloc[-1]['Close'] if not spx_data.empty else 0
        return spx_last - es_last
    except:
        return 0.0

# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================
if 'theme' not in st.session_state:
    st.session_state.theme = 'Dark'

if 'spx_slopes' not in st.session_state:
    st.session_state.spx_slopes = SPX_SLOPES.copy()

if 'current_offset' not in st.session_state:
    st.session_state.current_offset = 0.0

# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR CONTROLS
# ═══════════════════════════════════════════════════════════════════════════════
st.sidebar.title("SPX Prophet Analytics")

# Theme selector
theme = st.sidebar.radio("Theme", ["Dark", "Light"], key="ui_theme")
st.session_state.theme = theme

st.sidebar.markdown("---")

# SPX slope controls
st.sidebar.subheader("SPX Slopes (per 30-min)")
high_slope = st.sidebar.number_input(
    "High Slope", 
    value=st.session_state.spx_slopes['high'],
    step=0.001, format="%.4f", key="sb_spx_high"
)
close_slope = st.sidebar.number_input(
    "Close Slope", 
    value=st.session_state.spx_slopes['close'],
    step=0.001, format="%.4f", key="sb_spx_close"
)
low_slope = st.sidebar.number_input(
    "Low Slope", 
    value=st.session_state.spx_slopes['low'],
    step=0.001, format="%.4f", key="sb_spx_low"
)
skyline_slope = st.sidebar.number_input(
    "Skyline Slope", 
    value=st.session_state.spx_slopes['skyline'],
    step=0.001, format="%.3f", key="sb_spx_sky"
)
baseline_slope = st.sidebar.number_input(
    "Baseline Slope", 
    value=st.session_state.spx_slopes['baseline'], 
    step=0.001, format="%.3f", key="sb_spx_base"
)

st.session_state.spx_slopes['high'] = high_slope
st.session_state.spx_slopes['close'] = close_slope
st.session_state.spx_slopes['low'] = low_slope
st.session_state.spx_slopes['skyline'] = skyline_slope
st.session_state.spx_slopes['baseline'] = baseline_slope

st.sidebar.caption("Stock slopes use ± magnitudes")

# ============================================================================
# MAIN APP HEADER
# ============================================================================
st.title("📈 SPX Prophet Analytics")
st.caption("Advanced trading analytics with live data integration")

# Display current market info
col1, col2, col3 = st.columns(3)
with col1:
    current_time_ct = datetime.now(CT_TZ)
    st.metric("Current Time (CT)", current_time_ct.strftime("%H:%M:%S"))

with col2:
    market_open = current_time_ct.replace(hour=8, minute=30, second=0, microsecond=0)
    market_close = current_time_ct.replace(hour=14, minute=30, second=0, microsecond=0)
    is_rth = market_open <= current_time_ct <= market_close
    st.metric("Market Status", "OPEN" if is_rth else "CLOSED")

with col3:
    st.metric("ES→SPX Offset", f"{st.session_state.current_offset:.1f}")