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
                        last_close = spx_data.iloc[-1]
                        st.session_state.spx_manual_anchors = {
                            'high': (last_close['High'], last_close.name),
                            'close': (last_close['Close'], last_close.name), 
                            'low': (last_close['Low'], last_close.name)
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
                spx_sky_price = sky_price + manual_offset
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
                spx_base_price = base_price + manual_offset
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