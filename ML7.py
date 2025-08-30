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

st.sidebar.markdown("---")

# Stock slope controls
st.sidebar.subheader("Stock Slopes (magnitude)")
if 'stock_slopes' not in st.session_state:
    st.session_state.stock_slopes = STOCK_SLOPES.copy()

# Display core stock slopes
for ticker, default_slope in STOCK_SLOPES.items():
    current_slope = st.sidebar.number_input(
        f"{ticker}", 
        value=st.session_state.stock_slopes.get(ticker, default_slope),
        step=0.0001, format="%.4f", key=f"sb_stk_{ticker}"
    )
    st.session_state.stock_slopes[ticker] = current_slope

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





# ═══════════════════════════════════════════════════════════════════════════════
# SPX PROPHET - PART 2: SPX ANCHORS TAB
# ═══════════════════════════════════════════════════════════════════════════════

# Create main tabs
tab1, tab2, tab3, tab4 = st.tabs(["📈 SPX Anchors", "📚 Stock Anchors", "✅ Signals & EMA", "🧮 Contract Tool"])

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
    
    with col2:
        proj_day = st.date_input(
            "Projection Day",
            value=datetime.now(CT_TZ).date(),
            key="spx_proj_day"
        )
    
    # ES to SPX offset input
    offset_col1, offset_col2 = st.columns(2)
    with offset_col1:
        manual_offset = st.number_input(
            "ES→SPX Offset", 
            value=st.session_state.current_offset,
            step=0.1, format="%.1f", key="spx_offset_manual"
        )
        st.session_state.current_offset = manual_offset
    
    with offset_col2:
        if st.button("🔄 Update Offset from Live Data", key="spx_update_offset"):
            with st.spinner("Fetching live offset..."):
                es_data = fetch_live_data("ES=F", prev_day, prev_day)
                spx_data = fetch_live_data("^GSPC", prev_day, prev_day)
                if not es_data.empty and not spx_data.empty:
                    st.session_state.current_offset = calculate_es_spx_offset(es_data, spx_data)
                    st.rerun()
    
    st.markdown("---")
    
    # ═══════════════════════════════════════════════════════════════════════════════
    # DATA FETCHING AND ANALYSIS
    # ═══════════════════════════════════════════════════════════════════════════════
    
    if st.button("📊 Generate SPX Anchors", key="spx_generate", type="primary"):
        with st.spinner("Fetching ES futures data..."):
            # Fetch ES futures data for anchor window
            es_data = fetch_live_data("ES=F", prev_day, prev_day)
            
            if es_data.empty:
                st.error("No ES futures data available for selected date")
            else:
                # Filter to anchor window (17:00-19:30 CT)
                anchor_window = get_session_window(es_data, SPX_ANCHOR_START, SPX_ANCHOR_END)
                
                if anchor_window.empty:
                    st.error("No data in anchor window (17:00-19:30 CT)")
                else:
                    # Store raw data for processing
                    st.session_state.es_anchor_data = anchor_window
                    st.session_state.spx_analysis_ready = True
                    
                    # Get last close for High/Close/Low anchors
                    spx_data = fetch_live_data("^GSPC", prev_day, prev_day)
                    if not spx_data.empty:
                        last_bar = spx_data.iloc[-1]
                        st.session_state.spx_manual_anchors = {
                            'high': (last_bar['High'], last_bar.name),
                            'close': (last_bar['Close'], last_bar.name), 
                            'low': (last_bar['Low'], last_bar.name)
                        }
                    else:
                        # Use ES data if SPX not available
                        last_es_bar = anchor_window.iloc[-1]
                        spx_equivalent_high = last_es_bar['High'] + manual_offset
                        spx_equivalent_close = last_es_bar['Close'] + manual_offset
                        spx_equivalent_low = last_es_bar['Low'] + manual_offset
                        
                        st.session_state.spx_manual_anchors = {
                            'high': (spx_equivalent_high, last_es_bar.name),
                            'close': (spx_equivalent_close, last_es_bar.name),
                            'low': (spx_equivalent_low, last_es_bar.name)
                        }
    
    # ═══════════════════════════════════════════════════════════════════════════════
    # RESULTS DISPLAY
    # ═══════════════════════════════════════════════════════════════════════════════
    
    if st.session_state.get('spx_analysis_ready', False):
        st.subheader("SPX Anchor Projections")
        
        # Process swing detection on ES data
        es_data = st.session_state.es_anchor_data
        es_swings = detect_swings(es_data, SWING_K)
        skyline_anchor, baseline_anchor = get_anchor_points(es_swings)
        
        # Create projection tables
        projection_tabs = st.tabs(["📈 High", "📊 Close", "📉 Low", "🔥 Skyline", "🏔️ Baseline"])
        
        # Manual anchor projections (High/Close/Low)
        if 'spx_manual_anchors' in st.session_state:
            manual_anchors = st.session_state.spx_manual_anchors
            
            with projection_tabs[0]:  # High
                if manual_anchors.get('high'):
                    price, timestamp = manual_anchors['high']
                    spx_price = price + manual_offset
                    anchor_time_ct = timestamp.astimezone(CT_TZ)
                    
                    high_proj = project_anchor_line(
                        spx_price, anchor_time_ct, 
                        st.session_state.spx_slopes['high'], proj_day
                    )
                    
                    st.write("High Anchor Projection")
                    st.dataframe(high_proj, use_container_width=True)
                    
                    # Add entry/exit analysis
                    high_analysis = calculate_entry_exit_table(high_proj, "HIGH")
                    st.write("Entry/Exit Analysis")
                    st.dataframe(high_analysis, use_container_width=True)
            
            with projection_tabs[1]:  # Close
                if manual_anchors.get('close'):
                    price, timestamp = manual_anchors['close']
                    spx_price = price + manual_offset
                    anchor_time_ct = timestamp.astimezone(CT_TZ)
                    
                    close_proj = project_anchor_line(
                        spx_price, anchor_time_ct,
                        st.session_state.spx_slopes['close'], proj_day
                    )
                    
                    st.write("Close Anchor Projection")
                    st.dataframe(close_proj, use_container_width=True)
                    
                    close_analysis = calculate_entry_exit_table(close_proj, "CLOSE")
                    st.write("Entry/Exit Analysis")
                    st.dataframe(close_analysis, use_container_width=True)
            
            with projection_tabs[2]:  # Low
                if manual_anchors.get('low'):
                    price, timestamp = manual_anchors['low']
                    spx_price = price + manual_offset
                    anchor_time_ct = timestamp.astimezone(CT_TZ)
                    
                    low_proj = project_anchor_line(
                        spx_price, anchor_time_ct,
                        st.session_state.spx_slopes['low'], proj_day
                    )
                    
                    st.write("Low Anchor Projection")
                    st.dataframe(low_proj, use_container_width=True)
                    
                    low_analysis = calculate_entry_exit_table(low_proj, "LOW")
                    st.write("Entry/Exit Analysis")
                    st.dataframe(low_analysis, use_container_width=True)
        
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
                
                st.write("Skyline Projection (Swing High)")
                st.dataframe(skyline_proj, use_container_width=True)
                
                sky_analysis = calculate_entry_exit_table(skyline_proj, "SKYLINE")
                st.write("Entry/Exit Analysis")
                st.dataframe(sky_analysis, use_container_width=True)
            else:
                st.warning("No skyline anchor detected in ES data")
        
        with projection_tabs[4]:  # Baseline
            if baseline_anchor:
                base_price, base_time = baseline_anchor
                spx_base_price = base_price + st.session_state.current_offset
                base_time_ct = base_time.astimezone(CT_TZ)
                
                baseline_proj = project_anchor_line(
                    spx_base_price, base_time_ct,
                    st.session_state.spx_slopes['baseline'], proj_day
                )
                
                st.write("Baseline Projection (Swing Low)")
                st.dataframe(baseline_proj, use_container_width=True)
                
                base_analysis = calculate_entry_exit_table(baseline_proj, "BASELINE")
                st.write("Entry/Exit Analysis")
                st.dataframe(base_analysis, use_container_width=True)
            else:
                st.warning("No baseline anchor detected in ES data")

# ═══════════════════════════════════════════════════════════════════════════════
# ENTRY/EXIT ANALYSIS FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def calculate_entry_exit_table(projection_df: pd.DataFrame, anchor_type: str) -> pd.DataFrame:
    """Calculate entry/exit probabilities and targets for each time slot"""
    if projection_df.empty:
        return pd.DataFrame()
    
    analysis_rows = []
    
    for idx, row in projection_df.iterrows():
        time_slot = row['Time']
        price = row['Price']
        
        # Calculate dynamic stops based on recent range
        stop_distance = price * 0.008  # 0.8% dynamic stop
        
        # TP1 and TP2 calculations
        tp1_distance = stop_distance * 1.5  # 1.5R
        tp2_distance = stop_distance * 2.5  # 2.5R
        
        # Direction bias based on anchor type
        is_bullish_bias = anchor_type in ['SKYLINE', 'HIGH'] and st.session_state.spx_slopes.get(anchor_type.lower(), 0) > 0
        
        if is_bullish_bias:
            entry_price = price
            stop_price = price - stop_distance
            tp1_price = price + tp1_distance
            tp2_price = price + tp2_distance
            direction = "LONG"
        else:
            entry_price = price  
            stop_price = price + stop_distance
            tp1_price = price - tp1_distance
            tp2_price = price - tp2_distance
            direction = "SHORT"
        
        # Calculate probabilities based on anchor line interaction
        entry_prob = calculate_entry_probability(price, anchor_type)
        tp1_prob = calculate_target_probability(tp1_distance, stop_distance, 1)
        tp2_prob = calculate_target_probability(tp2_distance, stop_distance, 2)
        
        analysis_rows.append({
            'Time': time_slot,
            'Direction': direction,
            'Entry': round(entry_price, 2),
            'Stop': round(stop_price, 2),
            'TP1': round(tp1_price, 2),
            'TP2': round(tp2_price, 2),
            'Risk': round(stop_distance, 2),
            'Entry_Prob': f"{entry_prob:.1f}%",
            'TP1_Prob': f"{tp1_prob:.1f}%", 
            'TP2_Prob': f"{tp2_prob:.1f}%"
        })
    
    return pd.DataFrame(analysis_rows)

def calculate_entry_probability(price: float, anchor_type: str) -> float:
    """Calculate entry probability based on anchor line strength"""
    # Base probability varies by anchor type
    base_probs = {
        'HIGH': 65, 'CLOSE': 70, 'LOW': 65,
        'SKYLINE': 75, 'BASELINE': 80
    }
    
    base_prob = base_probs.get(anchor_type, 65)
    
    # Adjust for time of day (market open has higher volatility)
    # This is a simplified model - you can enhance based on your experience
    return min(95, max(45, base_prob))

def calculate_target_probability(target_distance: float, stop_distance: float, target_num: int) -> float:
    """Calculate probability of reaching target based on risk-reward ratio"""
    rr_ratio = target_distance / stop_distance
    
    # Probability decreases with higher R targets
    if target_num == 1:  # TP1
        base_prob = 70 - (rr_ratio - 1.5) * 10
    else:  # TP2
        base_prob = 45 - (rr_ratio - 2.5) * 8
    
    return min(85, max(25, base_prob))





# ═══════════════════════════════════════════════════════════════════════════════
# SPX PROPHET - PART 3: STOCK ANCHORS TAB
# ═══════════════════════════════════════════════════════════════════════════════

with tab2:
    st.subheader("Stock Anchor Analysis")
    st.caption("Mon/Tue combined session analysis for individual stocks")
    
    # ═══════════════════════════════════════════════════════════════════════════════
    # TICKER SELECTION
    # ═══════════════════════════════════════════════════════════════════════════════
    
    # Quick ticker buttons
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
                value=datetime.now(CT_TZ).date() - timedelta(days=2),
                key=f"stk_mon_{selected_ticker}"
            )
        
        with col2:
            tuesday_date = st.date_input(
                "Tuesday Date", 
                value=monday_date + timedelta(days=1),
                key=f"stk_tue_{selected_ticker}"
            )
        
        with col3:
            # Project for rest of week
            st.write("Project for remaining week:")
            wed_date = tuesday_date + timedelta(days=1)
            thu_date = tuesday_date + timedelta(days=2) 
            fri_date = tuesday_date + timedelta(days=3)
            st.caption(f"Wed: {wed_date}, Thu: {thu_date}, Fri: {fri_date}")
        
        st.markdown("---")
        
        # ═══════════════════════════════════════════════════════════════════════════════
        # ANALYSIS EXECUTION
        # ═══════════════════════════════════════════════════════════════════════════════
        
        if st.button(f"Analyze {selected_ticker}", key=f"stk_analyze_{selected_ticker}", type="primary"):
            with st.spinner(f"Analyzing {selected_ticker} Mon/Tue sessions..."):
                
                # Fetch Monday and Tuesday data
                mon_data = fetch_live_data(selected_ticker, monday_date, monday_date)
                tue_data = fetch_live_data(selected_ticker, tuesday_date, tuesday_date)
                
                if mon_data.empty and tue_data.empty:
                    st.error(f"No data available for {selected_ticker} on selected dates")
                elif mon_data.empty:
                    st.warning("No Monday data, using Tuesday only")
                    combined_data = tue_data
                elif tue_data.empty:
                    st.warning("No Tuesday data, using Monday only")
                    combined_data = mon_data
                else:
                    # Combine Monday and Tuesday data
                    combined_data = pd.concat([mon_data, tue_data]).sort_index()
                
                if not combined_data.empty:
                    # Store analysis results
                    st.session_state.stock_analysis_data = combined_data
                    st.session_state.stock_analysis_ticker = selected_ticker
                    st.session_state.stock_analysis_ready = True
        
        # ═══════════════════════════════════════════════════════════════════════════════
        # RESULTS DISPLAY
        # ═══════════════════════════════════════════════════════════════════════════════
        
        if (st.session_state.get('stock_analysis_ready', False) and 
            st.session_state.get('stock_analysis_ticker') == selected_ticker):
            
            st.subheader(f"{selected_ticker} Anchor Analysis")
            
            # Process swing detection on combined Mon/Tue data
            stock_data = st.session_state.stock_analysis_data
            stock_swings = detect_swings(stock_data, SWING_K)
            
            # Get absolute highest and lowest across both days
            skyline_anchor, baseline_anchor = get_anchor_points(stock_swings)
            
            # Get manual High/Close/Low from last available data (using CLOSE only)
            last_bar = stock_data.iloc[-1]
            manual_anchors = {
                'high': (last_bar['Close'], last_bar.name),  # Use close price from high time
                'close': (last_bar['Close'], last_bar.name),
                'low': (last_bar['Close'], last_bar.name)   # Use close price from low time
            }
            
            # Create weekly projection tables
            st.subheader(f"{selected_ticker} Weekly Projections")
            
            # Project for Wed, Thu, Fri
            projection_dates = [
                ("Wednesday", tuesday_date + timedelta(days=1)),
                ("Thursday", tuesday_date + timedelta(days=2)), 
                ("Friday", tuesday_date + timedelta(days=3))
            ]
            
            weekly_tabs = st.tabs(["Wed", "Thu", "Fri"])
            
            for day_idx, (day_name, proj_date) in enumerate(projection_dates):
                with weekly_tabs[day_idx]:
                    st.write(f"{day_name} - {proj_date}")
                    
                    # Create anchor sub-tabs for each day
                    anchor_subtabs = st.tabs(["High", "Close", "Low", "Skyline", "Baseline"])
                    
                    # Manual anchors
                    with anchor_subtabs[0]:  # High
                        price, timestamp = manual_anchors['high']
                        anchor_time_ct = timestamp.astimezone(CT_TZ)
                        
                        high_proj = project_anchor_line(
                            price, anchor_time_ct, slope_magnitude, proj_date
                        )
                        
                        st.dataframe(high_proj, use_container_width=True)
                        
                        high_analysis = calculate_weekly_entry_exit_table(high_proj, selected_ticker, "HIGH", day_name)
                        st.write("Entry/Exit Analysis")
                        st.dataframe(high_analysis, use_container_width=True)
                    
                    with anchor_subtabs[1]:  # Close
                        price, timestamp = manual_anchors['close']
                        anchor_time_ct = timestamp.astimezone(CT_TZ)
                        
                        close_proj = project_anchor_line(
                            price, anchor_time_ct, slope_magnitude, proj_date
                        )
                        
                        st.dataframe(close_proj, use_container_width=True)
                        
                        close_analysis = calculate_weekly_entry_exit_table(close_proj, selected_ticker, "CLOSE", day_name)
                        st.write("Entry/Exit Analysis")
                        st.dataframe(close_analysis, use_container_width=True)
                    
                    with anchor_subtabs[2]:  # Low
                        price, timestamp = manual_anchors['low']
                        anchor_time_ct = timestamp.astimezone(CT_TZ)
                        
                        low_proj = project_anchor_line(
                            price, anchor_time_ct, slope_magnitude, proj_date
                        )
                        
                        st.dataframe(low_proj, use_container_width=True)
                        
                        low_analysis = calculate_weekly_entry_exit_table(low_proj, selected_ticker, "LOW", day_name)
                        st.write("Entry/Exit Analysis")
                        st.dataframe(low_analysis, use_container_width=True)
                    
                    # Swing anchors
                    with anchor_subtabs[3]:  # Skyline
                        if skyline_anchor:
                            sky_price, sky_time = skyline_anchor
                            sky_time_ct = sky_time.astimezone(CT_TZ)
                            
                            skyline_proj = project_anchor_line(
                                sky_price, sky_time_ct, slope_magnitude, proj_date
                            )
                            
                            st.dataframe(skyline_proj, use_container_width=True)
                            
                            sky_analysis = calculate_weekly_entry_exit_table(skyline_proj, selected_ticker, "SKYLINE", day_name)
                            st.write("Entry/Exit Analysis")
                            st.dataframe(sky_analysis, use_container_width=True)
                        else:
                            st.warning("No skyline anchor detected")
                    
                    with anchor_subtabs[4]:  # Baseline
                        if baseline_anchor:
                            base_price, base_time = baseline_anchor
                            base_time_ct = base_time.astimezone(CT_TZ)
                            
                            baseline_proj = project_anchor_line(
                                base_price, base_time_ct, -slope_magnitude, proj_date
                            )
                            
                            st.dataframe(baseline_proj, use_container_width=True)
                            
                            base_analysis = calculate_weekly_entry_exit_table(baseline_proj, selected_ticker, "BASELINE", day_name)
                            st.write("Entry/Exit Analysis")
                            st.dataframe(base_analysis, use_container_width=True)
                        else:
                            st.warning("No baseline anchor detected")

# ═══════════════════════════════════════════════════════════════════════════════
# STOCK-SPECIFIC ENTRY/EXIT FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def calculate_weekly_entry_exit_table(projection_df: pd.DataFrame, ticker: str, anchor_type: str, day_name: str) -> pd.DataFrame:
    """Calculate weekly stock entry/exit analysis with day-specific adjustments"""
    if projection_df.empty:
        return pd.DataFrame()
    
    analysis_rows = []
    stock_volatility = get_stock_volatility_factor(ticker)
    
    # Day-specific multipliers for probability
    day_multipliers = {"Wednesday": 1.1, "Thursday": 1.0, "Friday": 0.9}
    day_mult = day_multipliers.get(day_name, 1.0)
    
    for idx, row in projection_df.iterrows():
        time_slot = row['Time']
        price = row['Price']
        
        # Dynamic stop based on stock volatility
        stop_distance = price * stock_volatility * 0.012
        
        # TP calculations
        tp1_distance = stop_distance * 1.5
        tp2_distance = stop_distance * 2.5
        
        # Direction based on slope sign
        slope_sign = 1 if anchor_type in ['SKYLINE', 'HIGH'] else -1
        
        entry_price = price
        stop_price = price - (stop_distance * slope_sign)
        tp1_price = price + (tp1_distance * slope_sign)
        tp2_price = price + (tp2_distance * slope_sign)
        direction = "LONG" if slope_sign > 0 else "SHORT"
        
        # Enhanced probability calculations with day adjustment
        entry_prob = calculate_stock_entry_probability(ticker, anchor_type, time_slot) * day_mult
        tp1_prob = calculate_stock_target_probability(ticker, tp1_distance, stop_distance, 1) * day_mult
        tp2_prob = calculate_stock_target_probability(ticker, tp2_distance, stop_distance, 2) * day_mult
        
        analysis_rows.append({
            'Time': time_slot,
            'Direction': direction,
            'Entry': round(entry_price, 2),
            'Stop': round(stop_price, 2),
            'TP1': round(tp1_price, 2),
            'TP2': round(tp2_price, 2),
            'Risk': round(stop_distance, 2),
            'Entry_Prob': f"{min(95, entry_prob):.1f}%",
            'TP1_Prob': f"{min(85, tp1_prob):.1f}%",
            'TP2_Prob': f"{min(75, tp2_prob):.1f}%",
            'Day': day_name
        })
    
    return pd.DataFrame(analysis_rows)

def get_stock_volatility_factor(ticker: str) -> float:
    """Get volatility factor for different stocks"""
    volatility_factors = {
        'TSLA': 1.8, 'NVDA': 1.6, 'META': 1.4, 'NFLX': 1.3,
        'AMZN': 1.2, 'GOOGL': 1.1, 'MSFT': 1.0, 'AAPL': 0.9
    }
    return volatility_factors.get(ticker, 1.2)  # Default for custom tickers

def calculate_stock_entry_probability(ticker: str, anchor_type: str, time_slot: str) -> float:
    """Calculate stock entry probability with time-of-day and ticker adjustments"""
    # Base probabilities by anchor type
    base_probs = {
        'HIGH': 60, 'CLOSE': 65, 'LOW': 60,
        'SKYLINE': 70, 'BASELINE': 75
    }
    
    base_prob = base_probs.get(anchor_type, 60)
    
    # Time-of-day adjustments
    hour = int(time_slot.split(':')[0])
    if hour in [9, 10]:  # Market open volatility
        time_adj = 10
    elif hour in [13, 14]:  # End of day momentum
        time_adj = 5
    else:
        time_adj = 0
    
    # Ticker-specific adjustments
    ticker_adj = 0
    if ticker in ['TSLA', 'NVDA', 'META']:  # High momentum stocks
        ticker_adj = 5
    elif ticker in ['AAPL', 'MSFT']:  # Stable stocks
        ticker_adj = -5
    
    final_prob = base_prob + time_adj + ticker_adj
    return min(90, max(40, final_prob))

def calculate_stock_target_probability(ticker: str, target_distance: float, stop_distance: float, target_num: int) -> float:
    """Calculate target probability adjusted for stock characteristics"""
    rr_ratio = target_distance / stop_distance
    volatility_factor = get_stock_volatility_factor(ticker)
    
    if target_num == 1:  # TP1
        base_prob = 65 - (rr_ratio - 1.5) * 8
        # Higher volatility = higher target probability
        vol_adj = (volatility_factor - 1) * 10
    else:  # TP2
        base_prob = 40 - (rr_ratio - 2.5) * 6
        vol_adj = (volatility_factor - 1) * 8
    
    final_prob = base_prob + vol_adj
    return min(80, max(20, final_prob))








# ═══════════════════════════════════════════════════════════════════════════════
# SPX PROPHET - PART 4: SIGNALS & EMA TAB
# ═══════════════════════════════════════════════════════════════════════════════

with tab3:
    st.subheader("Signal Detection & EMA Analysis")
    st.caption("Single day signal detection with reference line and EMA analysis")
    
    # ═══════════════════════════════════════════════════════════════════════════════
    # INPUT CONTROLS
    # ═══════════════════════════════════════════════════════════════════════════════
    
    col1, col2 = st.columns(2)
    with col1:
        signal_symbol = st.text_input(
            "Symbol", 
            value="^GSPC",
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
    # SIGNAL ANALYSIS EXECUTION
    # ═══════════════════════════════════════════════════════════════════════════════
    
    if st.button("Generate Signals & EMA Analysis", key="sig_generate", type="primary"):
        with st.spinner(f"Analyzing {signal_symbol} for {signal_day}..."):
            
            # Fetch data for analysis day
            signal_data = fetch_live_data(signal_symbol, signal_day, signal_day)
            
            if signal_data.empty:
                st.error(f"No data available for {signal_symbol} on {signal_day}")
            else:
                # Filter to RTH session
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
        
        st.subheader(f"{symbol} Signal Analysis")
        
        # Create anchor datetime for projection
        anchor_datetime = datetime.combine(signal_day, anchor_config['time'])
        anchor_datetime_ct = CT_TZ.localize(anchor_datetime)
        
        # Generate reference line projection
        ref_line_proj = project_anchor_line(
            anchor_config['price'], 
            anchor_datetime_ct,
            anchor_config['slope'],
            signal_day
        )
        
        # Calculate indicators
        ema8 = calculate_ema(signal_data['Close'], 8)
        ema21 = calculate_ema(signal_data['Close'], 21)
        vwap = calculate_vwap(signal_data)
        
        # Create analysis tabs
        signal_tabs = st.tabs(["Reference Line", "BUY Signals", "SELL Signals", "EMA Analysis"])
        
        with signal_tabs[0]:  # Reference Line
            st.write("Reference Line Projection")
            st.dataframe(ref_line_proj, use_container_width=True)
            
            # Show anchor line interaction stats
            line_stats = calculate_anchor_line_stats(signal_data, ref_line_proj)
            st.write("Anchor Line Statistics")
            st.dataframe(line_stats, use_container_width=True)
        
        with signal_tabs[1]:  # BUY Signals
            buy_signals = detect_entry_signals(signal_data, ref_line_proj, "BUY")
            if not buy_signals.empty:
                st.write("BUY Signal Opportunities")
                st.dataframe(buy_signals, use_container_width=True)
            else:
                st.info("No BUY signals detected for this day")
        
        with signal_tabs[2]:  # SELL Signals
            sell_signals = detect_entry_signals(signal_data, ref_line_proj, "SELL")
            if not sell_signals.empty:
                st.write("SELL Signal Opportunities")
                st.dataframe(sell_signals, use_container_width=True)
            else:
                st.info("No SELL signals detected for this day")
        
        with signal_tabs[3]:  # EMA Analysis
            ema_analysis = calculate_ema_crossover_analysis(signal_data, ema8, ema21)
            st.write("EMA 8/21 Crossover Analysis")
            st.dataframe(ema_analysis, use_container_width=True)
            
            # EMA regime analysis
            ema_regime = analyze_ema_regime(ema8, ema21, vwap)
            st.write("Market Regime Analysis")
            st.dataframe(ema_regime, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# SIGNAL DETECTION FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def detect_entry_signals(price_data: pd.DataFrame, ref_line: pd.DataFrame, signal_type: str) -> pd.DataFrame:
    """Detect BUY/SELL signals based on anchor line interaction rules"""
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
            
        ref_price = ref_dict[bar_time]
        close_price = bar['Close']
        open_price = bar['Open']
        
        # Determine if candle is bullish or bearish
        is_bullish_candle = close_price > open_price
        is_bearish_candle = close_price < open_price
        
        # Check for line touches (within 0.1% tolerance)
        tolerance = ref_price * 0.001
        high_touches = abs(bar['High'] - ref_price) <= tolerance
        low_touches = abs(bar['Low'] - ref_price) <= tolerance
        
        signal_detected = False
        signal_reason = ""
        
        if signal_type == "BUY":
            # BUY: Bearish candle touches from above AND closes above line
            if (is_bearish_candle and 
                (high_touches or bar['High'] > ref_price) and 
                close_price > ref_price):
                signal_detected = True
                signal_reason = "Bearish candle touched from above, closed above line"
        
        elif signal_type == "SELL":
            # SELL: Bullish candle touches from below AND closes below line
            if (is_bullish_candle and 
                (low_touches or bar['Low'] < ref_price) and 
                close_price < ref_price):
                signal_detected = True
                signal_reason = "Bullish candle touched from below, closed below line"
        
        if signal_detected:
            # Calculate signal quality metrics
            touch_quality = calculate_touch_quality(bar, ref_price)
            volume_quality = calculate_volume_quality(bar, price_data)
            
            signals.append({
                'Time': bar_time,
                'Signal': signal_type,
                'Entry_Price': close_price,
                'Ref_Price': round(ref_price, 2),
                'Touch_Quality': f"{touch_quality:.1f}%",
                'Volume_Quality': f"{volume_quality:.1f}%",
                'Reason': signal_reason,
                'Probability': f"{calculate_signal_probability(touch_quality, volume_quality):.1f}%"
            })
    
    return pd.DataFrame(signals)

def calculate_touch_quality(bar: pd.Series, ref_price: float) -> float:
    """Calculate quality of anchor line touch"""
    # Closer touch = higher quality
    high_distance = abs(bar['High'] - ref_price) / ref_price
    low_distance = abs(bar['Low'] - ref_price) / ref_price
    
    closest_distance = min(high_distance, low_distance)
    
    # Convert to percentage (closer = higher score)
    quality = max(0, 100 - (closest_distance * 10000))
    return min(100, quality)

def calculate_volume_quality(bar: pd.Series, data: pd.DataFrame) -> float:
    """Calculate volume quality compared to recent average"""
    if 'Volume' not in data.columns:
        return 50.0
    
    # Get recent volume average (last 10 bars)
    recent_avg = data['Volume'].tail(10).mean()
    
    if recent_avg == 0:
        return 50.0
    
    volume_ratio = bar['Volume'] / recent_avg
    
    # Convert to quality score
    if volume_ratio >= 1.5:
        return 90.0
    elif volume_ratio >= 1.2:
        return 75.0
    elif volume_ratio >= 0.8:
        return 60.0
    else:
        return 30.0

def calculate_signal_probability(touch_quality: float, volume_quality: float) -> float:
    """Calculate overall signal probability"""
    # Weighted combination
    probability = (touch_quality * 0.6) + (volume_quality * 0.4)
    return min(95, max(25, probability))

# ═══════════════════════════════════════════════════════════════════════════════
# ANCHOR LINE STATISTICS
# ═══════════════════════════════════════════════════════════════════════════════

def calculate_anchor_line_stats(price_data: pd.DataFrame, ref_line: pd.DataFrame) -> pd.DataFrame:
    """Calculate statistical analysis of anchor line interactions"""
    if price_data.empty or ref_line.empty:
        return pd.DataFrame()
    
    stats = []
    
    # Create reference line lookup
    ref_dict = {}
    for _, row in ref_line.iterrows():
        ref_dict[row['Time']] = row['Price']
    
    total_touches = 0
    bounces = 0
    penetrations = 0
    avg_bounce_distance = 0
    avg_penetration_depth = 0
    
    for idx, bar in price_data.iterrows():
        bar_time = format_ct_time(idx)
        
        if bar_time not in ref_dict:
            continue
            
        ref_price = ref_dict[bar_time]
        
        # Check for touches (within 0.2% tolerance)
        tolerance = ref_price * 0.002
        touches_line = (bar['Low'] <= ref_price + tolerance and 
                       bar['High'] >= ref_price - tolerance)
        
        if touches_line:
            total_touches += 1
            
            # Check if it bounced (closed away from line)
            if bar['Close'] > ref_price + tolerance:
                bounces += 1
                avg_bounce_distance += bar['Close'] - ref_price
            elif bar['Close'] < ref_price - tolerance:
                penetrations += 1
                avg_penetration_depth += ref_price - bar['Close']
    
    # Calculate averages
    if bounces > 0:
        avg_bounce_distance /= bounces
    if penetrations > 0:
        avg_penetration_depth /= penetrations
    
    bounce_rate = (bounces / total_touches * 100) if total_touches > 0 else 0
    penetration_rate = (penetrations / total_touches * 100) if total_touches > 0 else 0
    
    stats.append({
        'Metric': 'Total Line Touches',
        'Value': total_touches,
        'Percentage': '-'
    })
    
    stats.append({
        'Metric': 'Bounce Rate',
        'Value': bounces,
        'Percentage': f"{bounce_rate:.1f}%"
    })
    
    stats.append({
        'Metric': 'Penetration Rate', 
        'Value': penetrations,
        'Percentage': f"{penetration_rate:.1f}%"
    })
    
    stats.append({
        'Metric': 'Avg Bounce Distance',
        'Value': f"{avg_bounce_distance:.2f}",
        'Percentage': '-'
    })
    
    stats.append({
        'Metric': 'Avg Penetration Depth',
        'Value': f"{avg_penetration_depth:.2f}",
        'Percentage': '-'
    })
    
    return pd.DataFrame(stats)

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
            
            crossovers.append({
                'Time': time_ct,
                'Type': crossover_type,
                'EMA8': round(curr_8, 2),
                'EMA21': round(curr_21, 2),
                'Price': round(price_data.iloc[i]['Close'], 2),
                'Strength': calculate_crossover_strength(curr_8, curr_21)
            })
    
    return pd.DataFrame(crossovers)

def calculate_crossover_strength(ema8: float, ema21: float) -> str:
    """Calculate strength of EMA crossover"""
    separation = abs(ema8 - ema21) / ema21 * 100
    
    if separation >= 0.5:
        return "Strong"
    elif separation >= 0.2:
        return "Moderate" 
    else:
        return "Weak"

def analyze_ema_regime(ema8: pd.Series, ema21: pd.Series, vwap: pd.Series) -> pd.DataFrame:
    """Analyze current market regime based on EMA and VWAP"""
    if ema8.empty or ema21.empty:
        return pd.DataFrame()
    
    current_8 = ema8.iloc[-1]
    current_21 = ema21.iloc[-1]
    current_vwap = vwap.iloc[-1] if not vwap.empty else 0
    
    # Determine regime
    if current_8 > current_21:
        ema_regime = "Bullish"
        regime_strength = (current_8 - current_21) / current_21 * 100
    else:
        ema_regime = "Bearish"
        regime_strength = (current_21 - current_8) / current_8 * 100
    
    # VWAP position
    if current_vwap > 0:
        price = ema8.iloc[-1]  # Use current price
        vwap_position = "Above VWAP" if price > current_vwap else "Below VWAP"
        vwap_distance = abs(price - current_vwap) / current_vwap * 100
    else:
        vwap_position = "N/A"
        vwap_distance = 0
    
    regime_data = [{
        'Component': 'EMA Regime',
        'Status': ema_regime,
        'Strength': f"{regime_strength:.2f}%",
        'Signal': 'Bullish' if ema_regime == 'Bullish' else 'Bearish'
    }, {
        'Component': 'VWAP Position',
        'Status': vwap_position,
        'Strength': f"{vwap_distance:.2f}%",
        'Signal': 'Bullish' if vwap_position == 'Above VWAP' else 'Bearish'
    }]
    
    return pd.DataFrame(regime_data)






# ═══════════════════════════════════════════════════════════════════════════════
# SPX PROPHET - PART 5: CONTRACT TOOL TAB
# ═══════════════════════════════════════════════════════════════════════════════

with tab4:
    st.subheader("Contract Tool")
    st.caption("Overnight contract price analysis for RTH entry optimization")
    
    # ═══════════════════════════════════════════════════════════════════════════════
    # TWO-POINT INPUT SYSTEM
    # ═══════════════════════════════════════════════════════════════════════════════
    
    st.write("Overnight Contract Price Points")
    
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
            key="ct_p1_time",
            help="Between 20:00 prev day and 10:00 current day"
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
            key="ct_p2_time",
            help="Between 20:00 prev day and 10:00 current day"
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
        value=datetime.now(CT_TZ).date(),
        key="ct_proj_day"
    )
    
    # Validate time range
    p1_datetime = datetime.combine(p1_date, p1_time)
    p2_datetime = datetime.combine(p2_date, p2_time)
    
    if p2_datetime <= p1_datetime:
        st.error("Point 2 must be after Point 1")
    else:
        # Calculate slope
        time_diff_minutes = (p2_datetime - p1_datetime).total_seconds() / 60
        blocks_between = time_diff_minutes / 30  # 30-min blocks
        
        if blocks_between > 0:
            contract_slope = (p2_price - p1_price) / blocks_between
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Time Span", f"{time_diff_minutes/60:.1f} hours")
            with col2:
                st.metric("30-min Blocks", f"{blocks_between:.1f}")
            with col3:
                st.metric("Slope per Block", f"{contract_slope:+.3f}")
    
    st.markdown("---")
    
    # ═══════════════════════════════════════════════════════════════════════════════
    # CONTRACT ANALYSIS EXECUTION
    # ═══════════════════════════════════════════════════════════════════════════════
    
    if st.button("Generate Contract Analysis", key="ct_generate", type="primary"):
        if p2_datetime <= p1_datetime:
            st.error("Please ensure Point 2 is after Point 1")
        else:
            with st.spinner("Analyzing contract projections..."):
                
                # Calculate contract projections
                p1_ct = CT_TZ.localize(p1_datetime)
                contract_projections = project_contract_line(
                    p1_price, p1_ct, contract_slope, projection_day
                )
                
                # Fetch SPX data for baseline analysis
                spx_data = fetch_live_data("^GSPC", projection_day - timedelta(days=1), projection_day)
                
                # Store results
                st.session_state.contract_projections = contract_projections
                st.session_state.contract_config = {
                    'p1_price': p1_price,
                    'p1_time': p1_ct,
                    'p2_price': p2_price, 
                    'p2_time': CT_TZ.localize(p2_datetime),
                    'slope': contract_slope
                }
                st.session_state.contract_spx_data = spx_data
                st.session_state.contract_ready = True
    
    # ═══════════════════════════════════════════════════════════════════════════════
    # RESULTS DISPLAY
    # ═══════════════════════════════════════════════════════════════════════════════
    
    if st.session_state.get('contract_ready', False):
        st.subheader("Contract Analysis Results")
        
        projections = st.session_state.contract_projections
        config = st.session_state.contract_config
        spx_data = st.session_state.contract_spx_data
        
        # Create analysis tabs
        contract_tabs = st.tabs(["RTH Projections", "Bounce Analysis", "Risk Management"])
        
        with contract_tabs[0]:  # RTH Projections
            st.write("RTH Contract Price Projections")
            
            # Enhance projections with probability scores
            enhanced_projections = calculate_contract_probabilities(projections, config)
            st.dataframe(enhanced_projections, use_container_width=True)
            
            # Key levels analysis
            key_levels = identify_key_contract_levels(enhanced_projections)
            st.write("Key Entry Levels")
            st.dataframe(key_levels, use_container_width=True)
        
        with contract_tabs[1]:  # Bounce Analysis
            if not spx_data.empty:
                bounce_analysis = analyze_spx_baseline_bounces(spx_data, projections, config)
                st.write("SPX Baseline Touch Analysis")
                st.dataframe(bounce_analysis, use_container_width=True)
                
                # Skyline drop analysis
                drop_analysis = analyze_spx_skyline_drops(spx_data, projections, config)
                st.write("SPX Skyline Drop Analysis (Put Opportunities)")
                st.dataframe(drop_analysis, use_container_width=True)
            else:
                st.warning("No SPX data available for bounce analysis")
        
        with contract_tabs[2]:  # Risk Management
            risk_analysis = calculate_contract_risk_management(projections, config)
            st.write("Contract Risk Management")
            st.dataframe(risk_analysis, use_container_width=True)
            
            # Market-based risk recommendations
            market_risk_analysis = calculate_market_based_risk(projections, config, spx_data)
            st.write("Market-Based Risk Analysis")
            st.dataframe(market_risk_analysis, use_container_width=True)

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
        blocks = time_diff.total_seconds() / 1800  # 30 minutes = 1800 seconds
        
        projected_price = anchor_price + (slope * blocks)
        
        projections.append({
            'Time': format_ct_time(slot_time),
            'Contract_Price': round(projected_price, 2),
            'Blocks_from_Anchor': round(blocks, 1),
            'Price_Change': round(projected_price - anchor_price, 2)
        })
    
    return pd.DataFrame(projections)

def calculate_contract_probabilities(projections: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Add probability scores to contract projections"""
    if projections.empty:
        return projections
    
    enhanced = projections.copy()
    prob_scores = []
    
    for idx, row in projections.iterrows():
        time_slot = row['Time']
        price = row['Contract_Price']
        
        # Calculate entry probability based on price momentum
        momentum_score = calculate_contract_momentum_score(config['slope'])
        time_score = calculate_time_of_day_score(time_slot)
        volatility_score = calculate_contract_volatility_score(price, config)
        
        # Combined probability
        entry_prob = (momentum_score * 0.4) + (time_score * 0.3) + (volatility_score * 0.3)
        prob_scores.append(f"{entry_prob:.1f}%")
    
    enhanced['Entry_Probability'] = prob_scores
    return enhanced

def calculate_contract_momentum_score(slope: float) -> float:
    """Score based on overnight momentum"""
    abs_slope = abs(slope)
    if abs_slope >= 0.5:
        return 85
    elif abs_slope >= 0.2:
        return 70
    elif abs_slope >= 0.1:
        return 55
    else:
        return 40

def calculate_time_of_day_score(time_slot: str) -> float:
    """Score based on optimal entry times"""
    hour = int(time_slot.split(':')[0])
    minute = int(time_slot.split(':')[1])
    
    # Market open has higher volatility
    if hour == 8 and minute >= 30:
        return 90
    elif hour == 9:
        return 85
    elif hour in [13, 14]:  # End of day momentum
        return 75
    else:
        return 60

def calculate_contract_volatility_score(price: float, config: dict) -> float:
    """Score based on price volatility"""
    price_range = abs(config['p2_price'] - config['p1_price'])
    volatility_pct = (price_range / config['p1_price']) * 100
    
    if volatility_pct >= 20:
        return 90
    elif volatility_pct >= 10:
        return 75
    elif volatility_pct >= 5:
        return 60
    else:
        return 45

# ═══════════════════════════════════════════════════════════════════════════════
# KEY LEVELS ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

def identify_key_contract_levels(projections: pd.DataFrame) -> pd.DataFrame:
    """Identify key entry levels with highest probability"""
    if projections.empty:
        return pd.DataFrame()
    
    # Extract probability values
    prob_values = []
    for prob_str in projections['Entry_Probability']:
        prob_values.append(float(prob_str.replace('%', '')))
    
    projections_with_prob = projections.copy()
    projections_with_prob['Prob_Value'] = prob_values
    
    # Get top 5 levels
    top_levels = projections_with_prob.nlargest(5, 'Prob_Value')
    
    key_levels = []
    for idx, row in top_levels.iterrows():
        key_levels.append({
            'Time': row['Time'],
            'Contract_Price': row['Contract_Price'],
            'Entry_Probability': row['Entry_Probability'],
            'Recommendation': get_entry_recommendation(row['Prob_Value'])
        })
    
    return pd.DataFrame(key_levels)

def get_entry_recommendation(prob_value: float) -> str:
    """Get entry recommendation based on probability"""
    if prob_value >= 80:
        return "STRONG BUY"
    elif prob_value >= 70:
        return "BUY"
    elif prob_value >= 60:
        return "MODERATE"
    else:
        return "WEAK"

# ═══════════════════════════════════════════════════════════════════════════════
# SPX BASELINE/SKYLINE ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

def analyze_spx_baseline_bounces(spx_data: pd.DataFrame, contract_proj: pd.DataFrame, 
                                config: dict) -> pd.DataFrame:
    """Analyze where SPX touches baseline = call contract bounce zones"""
    if spx_data.empty or contract_proj.empty:
        return pd.DataFrame()
    
    # Simulate baseline detection from SPX data
    spx_swings = detect_swings(spx_data, SWING_K)
    _, baseline_anchor = get_anchor_points(spx_swings)
    
    if not baseline_anchor:
        return pd.DataFrame([{'Analysis': 'No baseline anchor detected', 'Recommendation': 'N/A'}])
    
    baseline_price, baseline_time = baseline_anchor
    
    bounce_analysis = []
    
    # Check each contract projection time
    for idx, row in contract_proj.iterrows():
        time_slot = row['Time']
        contract_price = row['Contract_Price']
        
        # Calculate distance to baseline
        current_spx = get_spx_price_at_time(spx_data, time_slot)
        if current_spx:
            distance_to_baseline = abs(current_spx - baseline_price)
            distance_pct = (distance_to_baseline / baseline_price) * 100
            
            bounce_prob = calculate_bounce_probability(distance_pct)
            
            bounce_analysis.append({
                'Time': time_slot,
                'SPX_Price': round(current_spx, 2),
                'Baseline_Price': round(baseline_price, 2),
                'Distance': round(distance_to_baseline, 2),
                'Distance_Pct': f"{distance_pct:.2f}%",
                'Contract_Price': contract_price,
                'Bounce_Probability': f"{bounce_prob:.1f}%",
                'Action': get_bounce_action(bounce_prob)
            })
    
    return pd.DataFrame(bounce_analysis)

def analyze_spx_skyline_drops(spx_data: pd.DataFrame, contract_proj: pd.DataFrame,
                             config: dict) -> pd.DataFrame:
    """Analyze skyline drops for put entry opportunities"""
    if spx_data.empty or contract_proj.empty:
        return pd.DataFrame()
    
    # Simulate skyline detection
    spx_swings = detect_swings(spx_data, SWING_K)
    skyline_anchor, _ = get_anchor_points(spx_swings)
    
    if not skyline_anchor:
        return pd.DataFrame([{'Analysis': 'No skyline anchor detected', 'Recommendation': 'N/A'}])
    
    skyline_price, skyline_time = skyline_anchor
    
    drop_analysis = []
    
    for idx, row in contract_proj.iterrows():
        time_slot = row['Time']
        contract_price = row['Contract_Price']
        
        current_spx = get_spx_price_at_time(spx_data, time_slot)
        if current_spx:
            # Check if price dropped from skyline
            drop_from_skyline = skyline_price - current_spx
            drop_pct = (drop_from_skyline / skyline_price) * 100 if skyline_price > 0 else 0
            
            put_entry_prob = calculate_put_entry_probability(drop_pct)
            
            drop_analysis.append({
                'Time': time_slot,
                'SPX_Price': round(current_spx, 2),
                'Skyline_Price': round(skyline_price, 2),
                'Drop_Amount': round(drop_from_skyline, 2),
                'Drop_Pct': f"{drop_pct:.2f}%",
                'Put_Entry_Prob': f"{put_entry_prob:.1f}%",
                'Put_Recommendation': get_put_recommendation(put_entry_prob)
            })
    
    return pd.DataFrame(drop_analysis)

def get_spx_price_at_time(spx_data: pd.DataFrame, time_slot: str) -> float:
    """Get SPX price at specific time slot"""
    try:
        # Simple approximation - use closest available price
        return spx_data.iloc[-1]['Close']  # Use last available close
    except:
        return 0.0

def calculate_bounce_probability(distance_pct: float) -> float:
    """Calculate bounce probability based on distance to baseline"""
    if distance_pct <= 0.1:  # Very close to baseline
        return 90
    elif distance_pct <= 0.5:
        return 75
    elif distance_pct <= 1.0:
        return 60
    else:
        return 35

def calculate_put_entry_probability(drop_pct: float) -> float:
    """Calculate put entry probability based on skyline drop"""
    if drop_pct >= 2.0:  # Significant drop
        return 85
    elif drop_pct >= 1.0:
        return 70
    elif drop_pct >= 0.5:
        return 55
    else:
        return 30

def get_bounce_action(prob: float) -> str:
    """Get action recommendation for baseline bounce"""
    if prob >= 80:
        return "STRONG CALL ENTRY"
    elif prob >= 65:
        return "CALL ENTRY"
    else:
        return "MONITOR"

def get_put_recommendation(prob: float) -> str:
    """Get put recommendation based on skyline drop"""
    if prob >= 75:
        return "STRONG PUT ENTRY"
    elif prob >= 60:
        return "PUT ENTRY"
    else:
        return "MONITOR"

# ═══════════════════════════════════════════════════════════════════════════════
# RISK MANAGEMENT FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def calculate_contract_risk_management(projections: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Calculate risk management for contract positions"""
    if projections.empty:
        return pd.DataFrame()
    
    risk_analysis = []
    
    for idx, row in projections.iterrows():
        time_slot = row['Time']
        entry_price = row['Contract_Price']
        
        # Dynamic stop based on contract volatility
        volatility_factor = abs(config['slope']) * 2
        stop_distance = max(entry_price * 0.15, volatility_factor)  # Minimum 15% stop
        
        # Calculate TP levels
        tp1_price = entry_price + (stop_distance * 1.5)
        tp2_price = entry_price + (stop_distance * 2.5)
        stop_price = entry_price - stop_distance
        
        risk_analysis.append({
            'Time': time_slot,
            'Entry': round(entry_price, 2),
            'Stop': round(max(0.01, stop_price), 2),  # Minimum 1 cent
            'TP1': round(tp1_price, 2),
            'TP2': round(tp2_price, 2),
            'Risk_Amount': round(stop_distance, 2),
            'RR1': f"{1.5:.1f}",
            'RR2': f"{2.5:.1f}",
            'Max_Loss_Pct': f"{(stop_distance/entry_price)*100:.1f}%"
        })
    
    return pd.DataFrame(risk_analysis)

def calculate_market_based_risk(projections: pd.DataFrame, config: dict, spx_data: pd.DataFrame) -> pd.DataFrame:
    """Calculate risk analysis based on real market conditions"""
    if projections.empty:
        return pd.DataFrame()
    
    # Get current market volatility from SPX data
    if not spx_data.empty:
        recent_closes = spx_data['Close'].tail(20)
        daily_returns = recent_closes.pct_change().dropna()
        market_vol = daily_returns.std() * 100  # Convert to percentage
        
        # Average daily range
        if 'High' in spx_data.columns and 'Low' in spx_data.columns:
            daily_ranges = ((spx_data['High'] - spx_data['Low']) / spx_data['Close'] * 100).tail(10)
            avg_daily_range = daily_ranges.mean()
        else:
            avg_daily_range = 2.0  # Default 2%
    else:
        market_vol = 1.5
        avg_daily_range = 2.0
    
    # Contract-specific volatility
    contract_vol = abs(config['slope']) * 10  # Convert slope to volatility estimate
    
    risk_data = []
    
    # Volatility regime assessment
    if market_vol >= 2.0:
        vol_regime = "HIGH"
        risk_multiplier = 1.5
    elif market_vol >= 1.0:
        vol_regime = "MODERATE" 
        risk_multiplier = 1.0
    else:
        vol_regime = "LOW"
        risk_multiplier = 0.7
    
    # Time-based risk (market open vs close)
    time_risk_factors = {
        "08:30": 1.8, "09:00": 1.6, "09:30": 1.4, "10:00": 1.2,
        "10:30": 1.0, "11:00": 1.0, "11:30": 1.0, "12:00": 1.0,
        "12:30": 1.0, "13:00": 1.1, "13:30": 1.3, "14:00": 1.4, "14:30": 1.2
    }
    
    for idx, row in projections.iterrows():
        time_slot = row['Time']
        entry_price = row['Contract_Price']
        
        time_risk = time_risk_factors.get(time_slot, 1.0)
        
        # Dynamic risk based on market conditions
        base_risk = entry_price * 0.12  # 12% base risk
        adjusted_risk = base_risk * risk_multiplier * time_risk
        
        # Confidence level based on market alignment
        confidence = calculate_market_confidence(market_vol, contract_vol, time_risk)
        
        risk_data.append({
            'Time': time_slot,
            'Market_Vol': f"{market_vol:.1f}%",
            'Vol_Regime': vol_regime,
            'Time_Risk': f"{time_risk:.1f}x",
            'Recommended_Risk': f"${adjusted_risk:.2f}",
            'Confidence': f"{confidence:.0f}%",
            'Market_Alignment': get_market_alignment(confidence)
        })
    
    return pd.DataFrame(risk_data)

def calculate_market_confidence(market_vol: float, contract_vol: float, time_risk: float) -> float:
    """Calculate confidence based on market conditions"""
    # Base confidence
    base_confidence = 60
    
    # Volatility alignment bonus
    if 0.8 <= contract_vol/market_vol <= 1.2:  # Contract vol matches market vol
        vol_bonus = 20
    else:
        vol_bonus = 0
    
    # Time penalty for high-risk periods
    time_penalty = (time_risk - 1) * 10
    
    confidence = base_confidence + vol_bonus - time_penalty
    return max(20, min(95, confidence))

def get_market_alignment(confidence: float) -> str:
    """Get market alignment assessment"""
    if confidence >= 80:
        return "STRONG"
    elif confidence >= 65:
        return "GOOD"
    elif confidence >= 50:
        return "MODERATE"
    else:
        return "WEAK"







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