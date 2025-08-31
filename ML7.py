# ═══════════════════════════════════════════════════════════════════════════════
# 🔮 SPX PROPHET - PART 1: FOUNDATION & DATA HANDLING 📊 (fixed)
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
RTH_START = "08:30"   # RTH start in CT
RTH_END   = "14:30"   # RTH end in CT  (use this everywhere)
SPX_ANCHOR_START = "17:00"  # SPX anchor window start CT
SPX_ANCHOR_END   = "19:30"  # SPX anchor window end CT

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

# Added for Part 3 compatibility
SWING_K = 2
def detect_swings(df: pd.DataFrame, k: int = SWING_K) -> pd.DataFrame:
    # Keep your approach; thin wrapper so Part 3 works
    return detect_swings_simple(df)

# ═══════════════════════════════════════════════════════════════════════════════
# 🎨 STREAMLIT CONFIGURATION & STYLING
# ═══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="🔮 SPX Prophet Analytics",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

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
        padding: 1rem; border-radius: 10px; border-left: 5px solid #00ff88;
    }
    .warning-box {
        background: linear-gradient(135deg, #ffbb33, #ff8800);
        padding: 1rem; border-radius: 10px; border-left: 5px solid #ffff00;
    }
    .info-box {
        background: linear-gradient(135deg, #33b5e5, #0099cc);
        padding: 1rem; border-radius: 10px; border-left: 5px solid #00ddff;
    }
    .stDataFrame { background: rgba(255,255,255,0.95); border-radius: 10px; overflow: hidden; }
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# 📊 DATA FETCHING FUNCTIONS (ROBUST)
# ═══════════════════════════════════════════════════════════════════════════════

def _infer_period_interval(start_date: date, end_date: date) -> Tuple[str, str]:
    days = (end_date - start_date).days + 1
    if days <= 7:   return "7d", "30m"
    if days <= 30:  return "30d", "30m"
    return "60d", "30m"  # Max reliable for 30m

def _to_ct_index(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty: return df
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    return df.tz_convert(CT_TZ)

@st.cache_data(ttl=60)
def fetch_live_data(symbol: str, start_date: date, end_date: date) -> pd.DataFrame:
    """
    Robust intraday fetch via yf.download(period, interval); fallback to SPY for ^GSPC.
    Strategy unchanged.
    """
    try:
        period, interval = _infer_period_interval(start_date, end_date)

        def _dl(sym: str) -> pd.DataFrame:
            df = yf.download(
                tickers=sym,
                period=period,
                interval=interval,
                auto_adjust=False,
                back_adjust=False,
                threads=False,
                progress=False,
            )
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [c[0] for c in df.columns]
            return df

        df = _dl(symbol)
        if df.empty:
            df = _dl(symbol)  # retry once

        # index intraday can be flaky → fallback to SPY
        if df.empty and symbol.upper() == "^GSPC":
            df = _dl("SPY")

        if df.empty:
            return pd.DataFrame()

        df = _to_ct_index(df)

        # Filter to requested calendar dates
        start_dt = CT_TZ.localize(datetime.combine(start_date, time(0, 0)))
        end_dt   = CT_TZ.localize(datetime.combine(end_date,   time(23, 59)))
        df = df.loc[(df.index >= start_dt) & (df.index <= end_dt)]

        # Ensure columns
        required_cols = ['Open','High','Low','Close','Volume']
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0] for c in df.columns]
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            return pd.DataFrame()

        # Basic sanity
        if not validate_ohlc_data(df):
            # Keep returning data but warn at call-sites if needed
            pass
        return df

    except Exception as e:
        st.error(f"❌ Error fetching {symbol}: {e}")
        return pd.DataFrame()

def validate_ohlc_data(df: pd.DataFrame) -> bool:
    if df.empty: return False
    if not all(c in df.columns for c in ['Open','High','Low','Close']): return False
    invalid = (
        (df['High'] < df['Low']) |
        (df['High'] < df['Open']) | (df['High'] < df['Close']) |
        (df['Low']  > df['Open']) | (df['Low']  > df['Close']) |
        (df['Close'] <= 0) | (df['High'] <= 0)
    )
    return not invalid.any()

@st.cache_data(ttl=300)
def fetch_historical_data(symbol: str, days_back: int = 30) -> pd.DataFrame:
    end_date = datetime.now(CT_TZ).date()
    start_date = end_date - timedelta(days=days_back)
    return fetch_live_data(symbol, start_date, end_date)

# ═══════════════════════════════════════════════════════════════════════════════
# ⏰ TIME HANDLING FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def rth_slots_ct(target_date: date) -> List[datetime]:
    start_dt = datetime.combine(target_date, time(8, 30))
    start_ct = CT_TZ.localize(start_dt)
    slots = []
    current = start_ct
    end_ct = CT_TZ.localize(datetime.combine(target_date, time(int(RTH_END.split(':')[0]), int(RTH_END.split(':')[1]))))
    while current <= end_ct:
        slots.append(current)
        current += timedelta(minutes=30)
    return slots

def format_ct_time(dt: datetime) -> str:
    if dt.tzinfo is None:
        dt = CT_TZ.localize(dt)
    elif dt.tzinfo != CT_TZ:
        dt = dt.astimezone(CT_TZ)
    return dt.strftime("%H:%M")

def get_session_window(df: pd.DataFrame, start_time: str, end_time: str) -> pd.DataFrame:
    if df.empty: return df
    return df.between_time(start_time, end_time)

def get_daily_ohlc(df: pd.DataFrame, target_date: date) -> Dict:
    if df.empty: return {}
    start_dt = CT_TZ.localize(datetime.combine(target_date, time(0, 0)))
    end_dt   = CT_TZ.localize(datetime.combine(target_date, time(23, 59)))
    day_data = df.loc[start_dt:end_dt]
    if day_data.empty: return {}
    day_open = day_data.iloc[0]['Open']
    day_high = day_data['High'].max()
    day_low  = day_data['Low'].min()
    day_close = day_data.iloc[-1]['Close']
    high_time = day_data[day_data['High'] == day_high].index[0]
    low_time  = day_data[day_data['Low']  == day_low].index[0]
    open_time = day_data.index[0]
    close_time = day_data.index[-1]
    return {
        'open': (day_open, open_time),
        'high': (day_high, high_time),
        'low': (day_low, low_time),
        'close': (day_close, close_time)
    }

# ═══════════════════════════════════════════════════════════════════════════════
# 📈 SWING DETECTION FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def detect_swings_simple(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or len(df) < 2:
        return df.copy()
    df_swings = df.copy()
    df_swings['swing_high'] = False
    df_swings['swing_low'] = False
    if 'Close' in df_swings.columns:
        max_close_idx = df_swings['Close'].idxmax()
        min_close_idx = df_swings['Close'].idxmin()
        df_swings.loc[max_close_idx, 'swing_high'] = True
        df_swings.loc[min_close_idx, 'swing_low'] = True
    return df_swings

def get_anchor_points(df_swings: pd.DataFrame) -> Tuple[Optional[Tuple], Optional[Tuple]]:
    skyline = None
    baseline = None
    if df_swings.empty or 'Close' not in df_swings.columns:
        return skyline, baseline
    swing_highs = df_swings[df_swings.get('swing_high', False) == True]
    swing_lows  = df_swings[df_swings.get('swing_low',  False) == True]
    if not swing_highs.empty:
        best_high = swing_highs.loc[swing_highs['Close'].idxmax()]
        skyline = (best_high['Close'], best_high.name)
    if not swing_lows.empty:
        best_low = swing_lows.loc[swing_lows['Close'].idxmin()]
        baseline = (best_low['Close'], best_low.name)
    return skyline, baseline

# ═══════════════════════════════════════════════════════════════════════════════
# 📊 PROJECTION & INDICATORS
# ═══════════════════════════════════════════════════════════════════════════════

def project_anchor_line(anchor_price: float, anchor_time: datetime,
                        slope: float, target_date: date) -> pd.DataFrame:
    rth_slots = rth_slots_ct(target_date)
    projections = []
    for slot_time in rth_slots:
        blocks = (slot_time - anchor_time).total_seconds() / 1800.0
        projected_price = anchor_price + (slope * blocks)
        projections.append({
            'Time': format_ct_time(slot_time),
            'Price': round(projected_price, 2),
            'Blocks': round(blocks, 1),
            'Anchor_Price': round(anchor_price, 2),
            'Slope': slope
        })
    return pd.DataFrame(projections)

def calculate_ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span).mean()

def calculate_vwap(df: pd.DataFrame) -> pd.Series:
    if df.empty or 'Volume' not in df.columns:
        return pd.Series(index=df.index, dtype=float)
    typical_price = (df['High'] + df['Low'] + df['Close']) / 3
    cum_volume = df['Volume'].cumsum()
    cum_vol_price = (typical_price * df['Volume']).cumsum()
    vwap = cum_vol_price / cum_volume
    vwap = vwap.fillna(method='ffill').fillna(typical_price)
    return vwap

def calculate_es_spx_offset(es_data: pd.DataFrame, spx_data: pd.DataFrame) -> float:
    try:
        if es_data.empty or spx_data.empty:
            return 0.0
        es_rth  = get_session_window(es_data, RTH_START, RTH_END)
        spx_rth = get_session_window(spx_data, RTH_START, RTH_END)
        if es_rth.empty or spx_rth.empty:
            es_close  = es_data.iloc[-1]['Close']
            spx_close = spx_data.iloc[-1]['Close']
        else:
            es_close  = es_rth.iloc[-1]['Close']
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
# 🎛️ SIDEBAR CONTROLS (Part 1 UI)
# ═══════════════════════════════════════════════════════════════════════════════

st.sidebar.title("🔮 SPX Prophet Analytics")
st.sidebar.markdown("---")

theme = st.sidebar.radio("🎨 Theme", ["Dark", "Light"], key="ui_theme")
st.session_state.theme = theme

st.sidebar.markdown("---")
st.sidebar.markdown("### SPX Slopes (per 30-min)")
st.sidebar.caption("Adjust projection slopes for each anchor type")

with st.sidebar.expander("SPX Slope Settings", expanded=False):
    for slope_name, default_value in SPX_SLOPES.items():
        icon_map = {'high': 'High', 'close': 'Close', 'low': 'Low', 'skyline': 'Skyline', 'baseline': 'Baseline'}
        display_name = icon_map.get(slope_name, slope_name.title())
        slope_value = st.number_input(
            display_name,
            value=st.session_state.spx_slopes[slope_name],
            step=0.0001, format="%.4f",
            key=f"sb_spx_{slope_name}"
        )
        st.session_state.spx_slopes[slope_name] = slope_value

st.sidebar.markdown("---")
st.sidebar.markdown("### 🏢 Stock Slopes (magnitude)")
st.sidebar.caption("📊 Individual stock projection parameters")

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
# 🏠 MAIN APP HEADER (Part 1)
# ═══════════════════════════════════════════════════════════════════════════════

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
    is_weekday = current_time_ct.weekday() < 5
    market_open = current_time_ct.replace(hour=8, minute=30, second=0, microsecond=0)
    market_close = current_time_ct.replace(hour=int(RTH_END.split(':')[0]), minute=int(RTH_END.split(':')[1]), second=0, microsecond=0)
    within_hours = market_open <= current_time_ct <= market_close
    is_rth = is_weekday and within_hours
    if is_weekday:
        if is_rth:
            status_color = "#00ff88"; status_text = "MARKET OPEN"
        else:
            status_color = "#ffbb33"; status_text = "MARKET CLOSED"
    else:
        status_color = "#ff6b6b"; status_text = "WEEKEND"
    st.markdown(f"""
    <div class="metric-container">
        <h3>Market Status</h3>
        <h2 style="color: {status_color};">{status_text}</h2>
        <p>RTH: {RTH_START} - {RTH_END} CT</p>
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

if st.button("Test Data Connection", key="test_connection"):
    with st.spinner("Testing market data connection..."):
        test_data = fetch_live_data("^GSPC", datetime.now().date() - timedelta(days=1), datetime.now().date())
        if not test_data.empty:
            st.success("Market data connection successful!")
            st.info(f"Retrieved {len(test_data)} data points for SPX (or SPY fallback)")
        else:
            st.error("Market data connection failed!")

st.markdown("---")

# ═══════════════════════════════════════════════════════════════════════════════
# ✅ PART 1 COMPLETE - ENHANCED FOUNDATION
# ═══════════════════════════════════════════════════════════════════════════════


# ═══════════════════════════════════════════════════════════════════════════════
# SPX PROPHET - PART 2: SPX ANCHORS TAB (fixed)
# ═══════════════════════════════════════════════════════════════════════════════

def update_offset_for_date():
    """Automatically update offset when date changes"""
    if 'spx_prev_day' in st.session_state:
        selected_date = st.session_state.spx_prev_day
        es_data = fetch_live_data("ES=F", selected_date, selected_date)
        spx_data = fetch_live_data("^GSPC", selected_date, selected_date)
        if not es_data.empty and not spx_data.empty:
            new_offset = calculate_es_spx_offset(es_data, spx_data)
            st.session_state.current_offset = new_offset

# Create main tabs (will be reused in later parts)
tab1, tab2, tab3, tab4 = st.tabs(["SPX Anchors", "Stock Anchors", "Signals & EMA", "Contract Tool"])

with tab1:
    st.subheader("SPX Anchor Analysis")
    st.caption("Live ES futures data for anchor detection and SPX projections")

    col1, col2 = st.columns(2)
    with col1:
        prev_day = st.date_input(
            "Previous Trading Day",
            value=datetime.now(CT_TZ).date() - timedelta(days=1),
            key="spx_prev_day",
            on_change=update_offset_for_date
        )
        st.caption(f"Selected: {prev_day.strftime('%A')}")
    with col2:
        proj_day = st.date_input(
            "Projection Day",
            value=prev_day + timedelta(days=1),
            key="spx_proj_day"
        )
        st.caption(f"Projecting for: {proj_day.strftime('%A')}")

    st.markdown("---")

    # Manual price override section
    st.subheader("Price Override (Optional)")
    st.caption("Override Yahoo Finance data with your exact prices for accurate projections")

    use_manual = st.checkbox("Use Manual Prices", key="use_manual_prices")
    if use_manual:
        override_col1, override_col2, override_col3 = st.columns(3)
        with override_col1:
            manual_high = st.number_input(
                "Manual High Price", value=0.0, step=0.1, format="%.1f", key="manual_high_price"
            )
        with override_col2:
            manual_close = st.number_input(
                "Manual Close Price", value=0.0, step=0.1, format="%.1f", key="manual_close_price"
            )
        with override_col3:
            manual_low = st.number_input(
                "Manual Low Price", value=0.0, step=0.1, format="%.1f", key="manual_low_price"
            )

    st.markdown("---")

    # DATA ANALYSIS WITH AUTO OFFSET
    if st.button("Generate SPX Anchors", key="spx_generate", type="primary"):
        with st.spinner("Analyzing market data..."):
            try:
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
                        anchor_window = es_data
                    st.session_state.es_anchor_data = anchor_window

                    # Get SPX data for High/Close/Low anchors (or SPY fallback via fetch_live_data)
                    spx_data = fetch_live_data("^GSPC", prev_day, prev_day)
                    if not spx_data.empty:
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

    # RESULTS DISPLAY
    if st.session_state.get('spx_analysis_ready', False):
        st.subheader("SPX Anchor Projections")
        es_data = st.session_state.get('es_anchor_data', pd.DataFrame())
        skyline_anchor_spx = None
        baseline_anchor_spx = None

        if not es_data.empty:
            es_swings = detect_swings_simple(es_data)
            es_skyline, es_baseline = get_anchor_points(es_swings)
            current_offset = st.session_state.current_offset
            if es_skyline:
                es_price, es_time = es_skyline
                skyline_anchor_spx = (es_price + current_offset, es_time)
            if es_baseline:
                es_price, es_time = es_baseline
                baseline_anchor_spx = (es_price + current_offset, es_time)

        # Detected SPX Anchors
        if st.session_state.get('spx_manual_anchors'):
            manual_anchors = st.session_state.spx_manual_anchors
            st.subheader("Detected SPX Anchors")
            summary_cols = st.columns(5)
            anchor_info = [('high', 'High', '#ff6b6b'),
                           ('close', 'Close', '#f9ca24'),
                           ('low', 'Low', '#4ecdc4')]
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

            with projection_tabs[0]:  # High
                if 'high' in manual_anchors:
                    spx_price, timestamp = manual_anchors['high']
                    high_proj = project_anchor_line(spx_price, timestamp.astimezone(CT_TZ),
                                                    st.session_state.spx_slopes['high'], proj_day)
                    if not high_proj.empty:
                        st.subheader("High Anchor SPX Projection")
                        st.dataframe(high_proj, use_container_width=True, hide_index=True)
                        high_analysis = calculate_entry_exit_table(high_proj, "HIGH")
                        if not high_analysis.empty:
                            st.subheader("Entry/Exit Strategy")
                            st.dataframe(high_analysis, use_container_width=True, hide_index=True)
                else:
                    st.warning("No high anchor data available")

            with projection_tabs[1]:  # Close
                if 'close' in manual_anchors:
                    spx_price, timestamp = manual_anchors['close']
                    close_proj = project_anchor_line(spx_price, timestamp.astimezone(CT_TZ),
                                                     st.session_state.spx_slopes['close'], proj_day)
                    if not close_proj.empty:
                        st.subheader("Close Anchor SPX Projection")
                        st.dataframe(close_proj, use_container_width=True, hide_index=True)
                        close_analysis = calculate_entry_exit_table(close_proj, "CLOSE")
                        if not close_analysis.empty:
                            st.subheader("Entry/Exit Strategy")
                            st.dataframe(close_analysis, use_container_width=True, hide_index=True)
                else:
                    st.warning("No close anchor data available")

            with projection_tabs[2]:  # Low
                if 'low' in manual_anchors:
                    spx_price, timestamp = manual_anchors['low']
                    low_proj = project_anchor_line(spx_price, timestamp.astimezone(CT_TZ),
                                                   st.session_state.spx_slopes['low'], proj_day)
                    if not low_proj.empty:
                        st.subheader("Low Anchor SPX Projection")
                        st.dataframe(low_proj, use_container_width=True, hide_index=True)
                        low_analysis = calculate_entry_exit_table(low_proj, "LOW")
                        if not low_analysis.empty:
                            st.subheader("Entry/Exit Strategy")
                            st.dataframe(low_analysis, use_container_width=True, hide_index=True)
                else:
                    st.warning("No low anchor data available")

        with projection_tabs[3]:  # Skyline
            if skyline_anchor_spx:
                spx_sky_price, sky_time = skyline_anchor_spx
                skyline_proj = project_anchor_line(spx_sky_price, sky_time.astimezone(CT_TZ),
                                                   st.session_state.spx_slopes['skyline'], proj_day)
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
                baseline_proj = project_anchor_line(spx_base_price, base_time.astimezone(CT_TZ),
                                                    st.session_state.spx_slopes['baseline'], proj_day)
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

# ===== Part 2 helpers preserved =====

def calculate_entry_exit_table(projection_df: pd.DataFrame, anchor_type: str) -> pd.DataFrame:
    if projection_df.empty: return pd.DataFrame()
    analysis_rows = []
    is_skyline = anchor_type.upper() in ['SKYLINE','HIGH']
    is_baseline = anchor_type.upper() in ['BASELINE','LOW']
    for _, row in projection_df.iterrows():
        time_slot = row['Time']; anchor_price = row['Price']
        if is_skyline:
            volatility_factor = anchor_price * 0.012
            tp1_distance = volatility_factor * 0.8
            tp2_distance = volatility_factor * 2.2
            entry_price = anchor_price
            tp1_price = anchor_price + tp1_distance
            tp2_price = anchor_price + tp2_distance
            direction = "BUY"
            stop_price = anchor_price + (anchor_price * 0.006)
        elif is_baseline:
            volatility_factor = anchor_price * 0.012
            tp1_distance = volatility_factor * 0.8
            tp2_distance = volatility_factor * 2.2
            entry_price = anchor_price
            tp1_price = anchor_price + tp1_distance
            tp2_price = anchor_price + tp2_distance
            direction = "BUY"
            stop_price = max(0.01, anchor_price - (anchor_price * 0.006))
        else:
            volatility_factor = anchor_price * 0.010
            tp1_distance = volatility_factor * 0.7
            tp2_distance = volatility_factor * 1.8
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
        entry_prob = calculate_anchor_entry_probability(anchor_type, time_slot)
        tp1_prob = calculate_anchor_target_probability(anchor_type, 1)
        tp2_prob = calculate_anchor_target_probability(anchor_type, 2)
        rr1 = abs(tp1_price - entry_price) / risk_amount if risk_amount > 0 else 0
        rr2 = abs(tp2_price - entry_price) / risk_amount if risk_amount > 0 else 0
        analysis_rows.append({
            'Time': time_slot, 'Direction': direction,
            'Entry': round(entry_price,2), 'Stop': round(stop_price,2),
            'TP1': round(tp1_price,2), 'TP2': round(tp2_price,2),
            'Risk': round(risk_amount,2), 'RR1': f"{rr1:.1f}", 'RR2': f"{rr2:.1f}",
            'Entry_Prob': f"{entry_prob:.0f}%", 'TP1_Prob': f"{tp1_prob:.0f}%", 'TP2_Prob': f"{tp2_prob:.0f}%"
        })
    return pd.DataFrame(analysis_rows)

def calculate_anchor_entry_probability(anchor_type: str, time_slot: str) -> float:
    base_probs = {'SKYLINE': 90.0, 'BASELINE': 90.0, 'HIGH': 75.0, 'CLOSE': 80.0, 'LOW': 75.0}
    base_prob = base_probs.get(anchor_type.upper(), 70.0)
    hour = int(time_slot.split(':')[0])
    if hour in [8,9]: time_adj = 8
    elif hour in [13,14]: time_adj = 5
    else: time_adj = 0
    return min(95, base_prob + time_adj)

def calculate_anchor_target_probability(anchor_type: str, target_num: int) -> float:
    if anchor_type.upper() in ['SKYLINE','BASELINE']:
        return 85.0 if target_num == 1 else 68.0
    else:
        return 75.0 if target_num == 1 else 55.0


# ═══════════════════════════════════════════════════════════════════════════════
# SPX PROPHET - PART 3: STOCK ANCHORS TAB (fixed)
# ═══════════════════════════════════════════════════════════════════════════════

with tab2:
    st.subheader("Stock Anchor Analysis")
    st.caption("Mon/Tue combined session analysis for individual stocks")

    st.write("Core Tickers:")
    ticker_cols = st.columns(4)
    selected_ticker = None
    core_tickers = ['TSLA','NVDA','AAPL','MSFT','AMZN','GOOGL','META','NFLX']
    for i, ticker in enumerate(core_tickers):
        with ticker_cols[i % 4]:
            if st.button(f"{ticker}", key=f"stk_btn_{ticker}"):
                selected_ticker = ticker
                st.session_state.selected_stock = ticker

    st.markdown("---")
    custom_ticker = st.text_input("Custom Symbol", placeholder="Enter any ticker symbol", key="stk_custom_input")
    if custom_ticker:
        selected_ticker = custom_ticker.upper()
        st.session_state.selected_stock = selected_ticker
    if not selected_ticker and 'selected_stock' in st.session_state:
        selected_ticker = st.session_state.selected_stock

    if selected_ticker:
        st.info(f"Selected: {selected_ticker}")
        default_slope = STOCK_SLOPES.get(selected_ticker, 0.0150)
        current_slope = st.session_state.stock_slopes.get(selected_ticker, default_slope)
        slope_magnitude = st.number_input(
            f"{selected_ticker} Slope Magnitude",
            value=current_slope, step=0.0001, format="%.4f",
            key=f"stk_slope_{selected_ticker}",
            help="Used as +magnitude for Skyline, -magnitude for Baseline"
        )
        st.session_state.stock_slopes[selected_ticker] = slope_magnitude

        col1, col2, col3 = st.columns(3)
        with col1:
            monday_date = st.date_input("Monday Date",
                                        value=datetime.now(CT_TZ).date() - timedelta(days=2),
                                        key=f"stk_mon_{selected_ticker}")
        with col2:
            tuesday_date = st.date_input("Tuesday Date",
                                         value=monday_date + timedelta(days=1),
                                         key=f"stk_tue_{selected_ticker}")
        with col3:
            st.write("Project for remaining week:")
            wed_date = tuesday_date + timedelta(days=1)
            thu_date = tuesday_date + timedelta(days=2)
            fri_date = tuesday_date + timedelta(days=3)
            st.caption(f"Wed: {wed_date}, Thu: {thu_date}, Fri: {fri_date}")

        st.markdown("---")

        if st.button(f"Analyze {selected_ticker}", key=f"stk_analyze_{selected_ticker}", type="primary"):
            with st.spinner(f"Analyzing {selected_ticker} Mon/Tue sessions..."):
                mon_data = fetch_live_data(selected_ticker, monday_date, monday_date)
                tue_data = fetch_live_data(selected_ticker, tuesday_date, tuesday_date)
                if mon_data.empty and tue_data.empty:
                    st.error(f"No data available for {selected_ticker} on selected dates")
                elif mon_data.empty:
                    st.warning("No Monday data, using Tuesday only"); combined_data = tue_data
                elif tue_data.empty:
                    st.warning("No Tuesday data, using Monday only"); combined_data = mon_data
                else:
                    combined_data = pd.concat([mon_data, tue_data]).sort_index()
                if not combined_data.empty:
                    st.session_state.stock_analysis_data = combined_data
                    st.session_state.stock_analysis_ticker = selected_ticker
                    st.session_state.stock_analysis_ready = True

        if (st.session_state.get('stock_analysis_ready', False) and
            st.session_state.get('stock_analysis_ticker') == selected_ticker):

            st.subheader(f"{selected_ticker} Anchor Analysis")
            stock_data = st.session_state.stock_analysis_data
            stock_swings = detect_swings(stock_data, SWING_K)
            skyline_anchor, baseline_anchor = get_anchor_points(stock_swings)

            last_bar = stock_data.iloc[-1]
            manual_anchors = {
                'high': (last_bar['Close'], last_bar.name),
                'close': (last_bar['Close'], last_bar.name),
                'low':  (last_bar['Close'], last_bar.name)
            }

            st.subheader(f"{selected_ticker} Weekly Projections")
            projection_dates = [("Wednesday", tuesday_date + timedelta(days=1)),
                                ("Thursday",  tuesday_date + timedelta(days=2)),
                                ("Friday",    tuesday_date + timedelta(days=3))]
            weekly_tabs = st.tabs(["Wed","Thu","Fri"])

            for day_idx, (day_name, proj_date) in enumerate(projection_dates):
                with weekly_tabs[day_idx]:
                    st.write(f"{day_name} - {proj_date}")
                    anchor_subtabs = st.tabs(["High","Close","Low","Skyline","Baseline"])

                    with anchor_subtabs[0]:
                        price, timestamp = manual_anchors['high']
                        high_proj = project_anchor_line(price, timestamp.astimezone(CT_TZ), slope_magnitude, proj_date)
                        st.dataframe(high_proj, use_container_width=True)
                        high_analysis = calculate_weekly_entry_exit_table(high_proj, selected_ticker, "HIGH", day_name)
                        st.write("Entry/Exit Analysis"); st.dataframe(high_analysis, use_container_width=True)

                    with anchor_subtabs[1]:
                        price, timestamp = manual_anchors['close']
                        close_proj = project_anchor_line(price, timestamp.astimezone(CT_TZ), slope_magnitude, proj_date)
                        st.dataframe(close_proj, use_container_width=True)
                        close_analysis = calculate_weekly_entry_exit_table(close_proj, selected_ticker, "CLOSE", day_name)
                        st.write("Entry/Exit Analysis"); st.dataframe(close_analysis, use_container_width=True)

                    with anchor_subtabs[2]:
                        price, timestamp = manual_anchors['low']
                        low_proj = project_anchor_line(price, timestamp.astimezone(CT_TZ), slope_magnitude, proj_date)
                        st.dataframe(low_proj, use_container_width=True)
                        low_analysis = calculate_weekly_entry_exit_table(low_proj, selected_ticker, "LOW", day_name)
                        st.write("Entry/Exit Analysis"); st.dataframe(low_analysis, use_container_width=True)

                    with anchor_subtabs[3]:
                        if skyline_anchor:
                            sky_price, sky_time = skyline_anchor
                            skyline_proj = project_anchor_line(sky_price, sky_time.astimezone(CT_TZ), slope_magnitude, proj_date)
                            st.dataframe(skyline_proj, use_container_width=True)
                            sky_analysis = calculate_weekly_entry_exit_table(skyline_proj, selected_ticker, "SKYLINE", day_name)
                            st.write("Entry/Exit Analysis"); st.dataframe(sky_analysis, use_container_width=True)
                        else:
                            st.warning("No skyline anchor detected")

                    with anchor_subtabs[4]:
                        if baseline_anchor:
                            base_price, base_time = baseline_anchor
                            baseline_proj = project_anchor_line(base_price, base_time.astimezone(CT_TZ), -slope_magnitude, proj_date)
                            st.dataframe(baseline_proj, use_container_width=True)
                            base_analysis = calculate_weekly_entry_exit_table(baseline_proj, selected_ticker, "BASELINE", day_name)
                            st.write("Entry/Exit Analysis"); st.dataframe(base_analysis, use_container_width=True)
                        else:
                            st.warning("No baseline anchor detected")

# Stock-specific entry/exit helpers unchanged
def calculate_weekly_entry_exit_table(projection_df: pd.DataFrame, ticker: str, anchor_type: str, day_name: str) -> pd.DataFrame:
    if projection_df.empty: return pd.DataFrame()
    analysis_rows = []; stock_volatility = get_stock_volatility_factor(ticker)
    day_multipliers = {"Wednesday": 1.1, "Thursday": 1.0, "Friday": 0.9}
    day_mult = day_multipliers.get(day_name, 1.0)
    for _, row in projection_df.iterrows():
        time_slot = row['Time']; price = row['Price']
        stop_distance = price * stock_volatility * 0.012
        tp1_distance = stop_distance * 1.5
        tp2_distance = stop_distance * 2.5
        slope_sign = 1 if anchor_type in ['SKYLINE','HIGH'] else -1
        entry_price = price
        stop_price = price - (stop_distance * slope_sign)
        tp1_price = price + (tp1_distance * slope_sign)
        tp2_price = price + (tp2_distance * slope_sign)
        direction = "LONG" if slope_sign > 0 else "SHORT"
        entry_prob = calculate_stock_entry_probability(ticker, anchor_type, time_slot) * day_mult
        tp1_prob = calculate_stock_target_probability(ticker, tp1_distance, stop_distance, 1) * day_mult
        tp2_prob = calculate_stock_target_probability(ticker, tp2_distance, stop_distance, 2) * day_mult
        analysis_rows.append({
            'Time': time_slot, 'Direction': direction,
            'Entry': round(entry_price,2), 'Stop': round(stop_price,2),
            'TP1': round(tp1_price,2), 'TP2': round(tp2_price,2),
            'Risk': round(stop_distance,2),
            'Entry_Prob': f"{min(95, entry_prob):.1f}%",
            'TP1_Prob': f"{min(85, tp1_prob):.1f}%",
            'TP2_Prob': f"{min(75, tp2_prob):.1f}%",
            'Day': day_name
        })
    return pd.DataFrame(analysis_rows)

def get_stock_volatility_factor(ticker: str) -> float:
    volatility_factors = {'TSLA': 1.8, 'NVDA': 1.6, 'META': 1.4, 'NFLX': 1.3,
                          'AMZN': 1.2, 'GOOGL': 1.1, 'MSFT': 1.0, 'AAPL': 0.9}
    return volatility_factors.get(ticker, 1.2)

def calculate_stock_entry_probability(ticker: str, anchor_type: str, time_slot: str) -> float:
    base_probs = {'HIGH':60,'CLOSE':65,'LOW':60,'SKYLINE':70,'BASELINE':75}
    base_prob = base_probs.get(anchor_type, 60)
    hour = int(time_slot.split(':')[0])
    if hour in [9,10]: time_adj = 10
    elif hour in [13,14]: time_adj = 5
    else: time_adj = 0
    ticker_adj = 5 if ticker in ['TSLA','NVDA','META'] else (-5 if ticker in ['AAPL','MSFT'] else 0)
    final_prob = base_prob + time_adj + ticker_adj
    return min(90, max(40, final_prob))

def calculate_stock_target_probability(ticker: str, target_distance: float, stop_distance: float, target_num: int) -> float:
    rr_ratio = target_distance / stop_distance if stop_distance > 0 else 0
    volatility_factor = get_stock_volatility_factor(ticker)
    if target_num == 1:
        base_prob = 65 - (rr_ratio - 1.5) * 8; vol_adj = (volatility_factor - 1) * 10
    else:
        base_prob = 40 - (rr_ratio - 2.5) * 6; vol_adj = (volatility_factor - 1) * 8
    final_prob = base_prob + vol_adj
    return min(80, max(20, final_prob))


# ═══════════════════════════════════════════════════════════════════════════════
# SPX PROPHET - PART 4: SIGNALS & EMA TAB (fixed)
# ═══════════════════════════════════════════════════════════════════════════════

def calculate_market_volatility(data: pd.DataFrame) -> float:
    if data.empty or len(data) < 2: return 1.5
    returns = data['Close'].pct_change().dropna()
    if returns.empty: return 1.5
    volatility = returns.std() * np.sqrt(390)
    return volatility * 100

def calculate_average_true_range(data: pd.DataFrame, periods: int = 14) -> pd.Series:
    if data.empty or len(data) < periods:
        return pd.Series(index=data.index, dtype=float)
    high_low = data['High'] - data['Low']
    high_close = np.abs(data['High'] - data['Close'].shift())
    low_close = np.abs(data['Low'] - data['Close'].shift())
    true_range = np.maximum(high_low, np.maximum(high_close, low_close))
    atr = true_range.rolling(window=periods).mean()
    return atr

def detect_anchor_touches(price_data: pd.DataFrame, anchor_line: pd.DataFrame) -> pd.DataFrame:
    if price_data.empty or anchor_line.empty: return pd.DataFrame()
    anchor_dict = {row['Time']: row['Price'] for _, row in anchor_line.iterrows()}
    touches = []
    for idx, bar in price_data.iterrows():
        bar_time = format_ct_time(idx)
        if bar_time not in anchor_dict: continue
        anchor_price = anchor_dict[bar_time]
        low_distance = abs(bar['Low'] - anchor_price)
        high_distance = abs(bar['High'] - anchor_price)
        recent_atr = calculate_average_true_range(price_data.tail(20), 14)
        tolerance = (recent_atr.iloc[-1] * 0.3) if not recent_atr.empty else (anchor_price * 0.002)
        touches_anchor = (bar['Low'] <= anchor_price + tolerance and bar['High'] >= anchor_price - tolerance)
        if touches_anchor:
            is_bearish = bar['Close'] < bar['Open']; is_bullish = bar['Close'] > bar['Open']
            closest_distance = min(low_distance, high_distance)
            touch_quality = max(0, 100 - (closest_distance / tolerance * 100)) if tolerance > 0 else 0
            volume_ma = price_data['Volume'].rolling(20).mean().iloc[-1] if 'Volume' in price_data.columns else 0
            volume_ratio = bar['Volume'] / volume_ma if volume_ma > 0 else 1.0
            volume_strength = min(100, volume_ratio * 50)
            touches.append({
                'Time': bar_time, 'Anchor_Price': round(anchor_price,2),
                'Touch_Price': round(bar['Low'] if low_distance < high_distance else bar['High'], 2),
                'Candle_Type': 'Bearish' if is_bearish else 'Bullish' if is_bullish else 'Doji',
                'Open': round(bar['Open'],2), 'High': round(bar['High'],2),
                'Low': round(bar['Low'],2), 'Close': round(bar['Close'],2),
                'Volume': int(bar['Volume']) if 'Volume' in bar else 0,
                'Touch_Quality': round(touch_quality,1), 'Volume_Strength': round(volume_strength,1),
                'ATR_Tolerance': round(tolerance,2)
            })
    return pd.DataFrame(touches)

def analyze_anchor_line_interaction(price_data: pd.DataFrame, anchor_line: pd.DataFrame) -> pd.DataFrame:
    if price_data.empty or anchor_line.empty: return pd.DataFrame()
    anchor_dict = {row['Time']: row['Price'] for _, row in anchor_line.iterrows()}
    interactions = []
    for idx, bar in price_data.iterrows():
        bar_time = format_ct_time(idx)
        if bar_time not in anchor_dict: continue
        anchor_price = anchor_dict[bar_time]
        price_above = bar['Close'] > anchor_price
        touched = (bar['Low'] <= anchor_price <= bar['High'])
        distance_from_anchor = bar['Close'] - anchor_price
        distance_pct = (distance_from_anchor / anchor_price) * 100 if anchor_price else 0
        interactions.append({
            'Time': bar_time, 'Close_Price': round(bar['Close'],2),
            'Anchor_Price': round(anchor_price,2), 'Distance': round(distance_from_anchor,2),
            'Distance_Pct': round(distance_pct,2), 'Touched': 'Yes' if touched else 'No',
            'Position': 'Above' if price_above else 'Below'
        })
    return pd.DataFrame(interactions)

def calculate_ema_crossover_signals(price_data: pd.DataFrame) -> pd.DataFrame:
    if price_data.empty or len(price_data) < 21: return pd.DataFrame()
    ema8 = calculate_ema(price_data['Close'], 8)
    ema21 = calculate_ema(price_data['Close'], 21)
    crossovers = []
    for i in range(1, len(price_data)):
        current_time = format_ct_time(price_data.index[i])
        prev_8, prev_21 = ema8.iloc[i-1], ema21.iloc[i-1]
        curr_8, curr_21 = ema8.iloc[i],   ema21.iloc[i]
        current_price = price_data.iloc[i]['Close']
        crossover_type = None
        if prev_8 <= prev_21 and curr_8 > curr_21: crossover_type = "Bullish Cross"
        elif prev_8 >= prev_21 and curr_8 < curr_21: crossover_type = "Bearish Cross"
        strength = abs(curr_8 - curr_21) / curr_21 * 100 if curr_21 else 0
        ema_regime = "Bullish" if curr_8 > curr_21 else "Bearish"
        crossovers.append({
            'Time': current_time, 'Price': round(current_price,2),
            'EMA8': round(curr_8,2), 'EMA21': round(curr_21,2),
            'Separation': round(strength,3), 'Regime': ema_regime,
            'Crossover': crossover_type if crossover_type else 'None',
            'Signal_Strength': 'Strong' if strength > 0.5 else 'Moderate' if strength > 0.2 else 'Weak'
        })
    return pd.DataFrame(crossovers)

with tab3:
    st.subheader("Signal Detection & Market Analysis")
    st.caption("Real-time anchor touch detection with market-derived analytics")

    col1, col2 = st.columns(2)
    with col1:
        signal_symbol = st.selectbox("Analysis Symbol", ["^GSPC","ES=F","SPY"], index=0, key="sig_symbol")
    with col2:
        signal_day = st.date_input("Analysis Day", value=datetime.now(CT_TZ).date(), key="sig_day")

    st.markdown("Reference Line Configuration")
    ref_col1, ref_col2, ref_col3 = st.columns(3)
    with ref_col1:
        anchor_price = st.number_input("Anchor Price", value=6000.0, step=0.1, format="%.2f", key="sig_anchor_price")
    with ref_col2:
        anchor_time_input = st.time_input("Anchor Time (CT)", value=time(17,0), key="sig_anchor_time")
    with ref_col3:
        ref_slope = st.number_input("Slope per 30min", value=0.268, step=0.001, format="%.3f", key="sig_ref_slope")

    st.markdown("---")

    if st.button("Analyze Market Signals", key="sig_generate", type="primary"):
        with st.spinner("Analyzing market data..."):
            signal_data = fetch_live_data(signal_symbol, signal_day, signal_day)
            if signal_data.empty:
                st.error(f"No data available for {signal_symbol} on {signal_day}")
            else:
                rth_data = get_session_window(signal_data, RTH_START, RTH_END)
                if rth_data.empty:
                    st.error("No RTH data available for selected day")
                else:
                    st.session_state.signal_data = rth_data
                    st.session_state.signal_anchor = {'price': anchor_price, 'time': anchor_time_input, 'slope': ref_slope}
                    st.session_state.signal_symbol = signal_symbol
                    st.session_state.signal_ready = True

    if st.session_state.get('signal_ready', False):
        signal_data = st.session_state.signal_data
        anchor_config = st.session_state.signal_anchor
        symbol = st.session_state.signal_symbol
        st.subheader(f"{symbol} Market Analysis Results")
        anchor_datetime = CT_TZ.localize(datetime.combine(signal_day, anchor_config['time']))
        ref_line_proj = project_anchor_line(anchor_config['price'], anchor_datetime, anchor_config['slope'], signal_day)
        volatility = calculate_market_volatility(signal_data)
        atr_series = calculate_average_true_range(signal_data, 14)
        vwap_series = calculate_vwap(signal_data)

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
        signal_tabs = st.tabs(["Reference Line","Anchor Touches","Line Interaction","EMA Analysis"])

        with signal_tabs[0]:
            st.subheader("Reference Line Projection")
            st.dataframe(ref_line_proj, use_container_width=True, hide_index=True)
            if not ref_line_proj.empty:
                price_range = ref_line_proj['Price'].max() - ref_line_proj['Price'].min()
                avg_price = ref_line_proj['Price'].mean()
                range_pct = (price_range / avg_price) * 100 if avg_price else 0
                st.info(f"Projection range: ${price_range:.2f} ({range_pct:.1f}% of average price)")

        with signal_tabs[1]:
            anchor_touches = detect_anchor_touches(signal_data, ref_line_proj)
            if not anchor_touches.empty:
                st.subheader("Detected Anchor Touches")
                st.dataframe(anchor_touches, use_container_width=True, hide_index=True)
                total_touches = len(anchor_touches)
                avg_touch_quality = anchor_touches['Touch_Quality'].mean()
                avg_volume_strength = anchor_touches['Volume_Strength'].mean()
                c1, c2, c3 = st.columns(3)
                with c1: st.metric("Total Touches", total_touches)
                with c2: st.metric("Avg Touch Quality", f"{avg_touch_quality:.1f}%")
                with c3: st.metric("Avg Volume Strength", f"{avg_volume_strength:.1f}%")
            else:
                st.info("No anchor line touches detected for this day")

        with signal_tabs[2]:
            line_interaction = analyze_anchor_line_interaction(signal_data, ref_line_proj)
            if not line_interaction.empty:
                st.subheader("Price-Anchor Line Interaction")
                st.dataframe(line_interaction, use_container_width=True, hide_index=True)
                touches = line_interaction[line_interaction['Touched'] == 'Yes']
                above_line = line_interaction[line_interaction['Position'] == 'Above']
                c1, c2, c3 = st.columns(3)
                with c1: st.metric("Touch Points", len(touches))
                with c2:
                    above_pct = (len(above_line) / len(line_interaction)) * 100 if len(line_interaction) else 0
                    st.metric("Time Above Line", f"{above_pct:.1f}%")
                with c3:
                    avg_distance = abs(line_interaction['Distance']).mean() if not line_interaction.empty else 0
                    st.metric("Avg Distance", f"${avg_distance:.2f}")
            else:
                st.info("No line interaction data available")

        with signal_tabs[3]:
            ema_analysis = calculate_ema_crossover_signals(signal_data)
            if not ema_analysis.empty:
                st.subheader("EMA 8/21 Analysis")
                st.dataframe(ema_analysis, use_container_width=True, hide_index=True)
                crossovers = ema_analysis[ema_analysis['Crossover'] != 'None']
                current_regime = ema_analysis.iloc[-1]['Regime'] if not ema_analysis.empty else 'Unknown'
                current_separation = ema_analysis.iloc[-1]['Separation'] if not ema_analysis.empty else 0
                c1, c2, c3 = st.columns(3)
                with c1: st.metric("Crossovers", len(crossovers))
                with c2: st.metric("Current Regime", current_regime)
                with c3: st.metric("EMA Separation", f"{current_separation:.3f}%")
            else:
                st.info("Insufficient data for EMA analysis")
    else:
        st.info("Configure your parameters and click 'Analyze Market Signals' to begin")


# ═══════════════════════════════════════════════════════════════════════════════
# SPX PROPHET - PART 5: CONTRACT TOOL TAB (fixed)
# ═══════════════════════════════════════════════════════════════════════════════

def calculate_contract_volatility(price_data: pd.DataFrame, window: int = 20) -> float:
    if price_data.empty or len(price_data) < window: return 0.15
    returns = price_data['Close'].pct_change().dropna()
    if returns.empty: return 0.15
    volatility = returns.rolling(window=window).std().iloc[-1]
    return volatility if not np.isnan(volatility) else 0.15

def calculate_price_momentum(p1_price: float, p2_price: float, time_hours: float) -> dict:
    price_change = p2_price - p1_price
    price_change_pct = (price_change / p1_price) * 100 if p1_price > 0 else 0
    hourly_rate = price_change / time_hours if time_hours > 0 else 0
    abs_change_pct = abs(price_change_pct)
    if abs_change_pct >= 15: strength, confidence = "Very Strong", 95
    elif abs_change_pct >= 8: strength, confidence = "Strong", 85
    elif abs_change_pct >= 3: strength, confidence = "Moderate", 70
    else: strength, confidence = "Weak", 50
    return {'change': price_change, 'change_pct': price_change_pct,
            'hourly_rate': hourly_rate, 'strength': strength, 'confidence': confidence}

def calculate_market_based_targets(entry_price: float, market_data: pd.DataFrame, direction: str) -> dict:
    if market_data.empty:
        base_move = entry_price * 0.02
        return {'tp1': entry_price + base_move if direction=="BUY" else entry_price - base_move,
                'tp2': entry_price + (base_move*2.5) if direction=="BUY" else entry_price - (base_move*2.5),
                'stop_distance': base_move*0.6}
    atr_series = calculate_average_true_range(market_data, 14)
    current_atr = atr_series.iloc[-1] if not atr_series.empty else entry_price * 0.015
    recent_range = market_data['High'].tail(10).max() - market_data['Low'].tail(10).min()
    volatility_factor = recent_range / market_data['Close'].tail(10).mean()
    base_target = current_atr * 1.2
    extended_target = current_atr * 3.0
    base_target *= (1 + volatility_factor * 0.5)
    extended_target *= (1 + volatility_factor * 0.3)
    if direction == "BUY":
        tp1, tp2 = entry_price + base_target, entry_price + extended_target
    else:
        tp1, tp2 = entry_price - base_target, entry_price - extended_target
    stop_distance = current_atr * 0.8
    return {'tp1': tp1, 'tp2': tp2, 'stop_distance': stop_distance, 'atr': current_atr, 'volatility_factor': volatility_factor}

def analyze_overnight_market_behavior(symbol: str, start_date: date, end_date: date) -> dict:
    overnight_data = fetch_live_data(symbol, start_date - timedelta(days=5), end_date)
    if overnight_data.empty:
        return {'avg_overnight_change': 0, 'overnight_volatility': 0.02, 'gap_frequency': 0, 'mean_reversion_rate': 0.6}
    overnight_moves = []; gap_moves = []
    for _, daily_data in overnight_data.groupby(overnight_data.index.date):
        if len(daily_data) < 2: continue
        day_close = daily_data.iloc[-1]['Close']
        next_open = daily_data.iloc[0]['Open'] if len(daily_data) > 0 else day_close
        overnight_change = (next_open - day_close) / day_close if day_close > 0 else 0
        overnight_moves.append(overnight_change)
        gap_size = abs(overnight_change)
        if gap_size > 0.005: gap_moves.append(gap_size)
    if not overnight_moves:
        return {'avg_overnight_change': 0, 'overnight_volatility': 0.02, 'gap_frequency': 0, 'mean_reversion_rate': 0.6}
    avg_overnight_change = np.mean(overnight_moves)
    overnight_volatility = np.std(overnight_moves)
    gap_frequency = len(gap_moves) / len(overnight_moves) if overnight_moves else 0
    reversion_count = sum(1 for move in overnight_moves if abs(move) < np.std(overnight_moves))
    mean_reversion_rate = reversion_count / len(overnight_moves) if overnight_moves else 0.6
    return {'avg_overnight_change': avg_overnight_change, 'overnight_volatility': overnight_volatility,
            'gap_frequency': gap_frequency, 'mean_reversion_rate': mean_reversion_rate}

with tab4:
    st.subheader("Contract Tool")
    st.caption("Overnight contract analysis for RTH entry optimization")

    st.subheader("Overnight Contract Price Points")
    point_col1, point_col2 = st.columns(2)
    with point_col1:
        st.markdown("**Point 1 (Earlier)**")
        p1_date = st.date_input("Point 1 Date", value=datetime.now(CT_TZ).date() - timedelta(days=1), key="ct_p1_date")
        p1_time = st.time_input("Point 1 Time (CT)", value=time(20,0), key="ct_p1_time")
        p1_price = st.number_input("Point 1 Contract Price", value=10.0, min_value=0.01, step=0.01, format="%.2f", key="ct_p1_price")
    with point_col2:
        st.markdown("**Point 2 (Later)**")
        p2_date = st.date_input("Point 2 Date", value=datetime.now(CT_TZ).date(), key="ct_p2_date")
        p2_time = st.time_input("Point 2 Time (CT)", value=time(8,0), key="ct_p2_time")
        p2_price = st.number_input("Point 2 Contract Price", value=12.0, min_value=0.01, step=0.01, format="%.2f", key="ct_p2_price")

    projection_day = st.date_input("RTH Projection Day", value=p2_date, key="ct_proj_day")

    p1_datetime = datetime.combine(p1_date, p1_time)
    p2_datetime = datetime.combine(p2_date, p2_time)

    if p2_datetime <= p1_datetime:
        st.error("Point 2 must be after Point 1")
    else:
        time_diff_hours = (p2_datetime - p1_datetime).total_seconds() / 3600
        momentum_metrics = calculate_price_momentum(p1_price, p2_price, time_diff_hours)
        metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
        with metric_col1: st.metric("Time Span", f"{time_diff_hours:.1f} hours")
        with metric_col2: st.metric("Price Change", f"{momentum_metrics['change']:+.2f}")
        with metric_col3: st.metric("Change %", f"{momentum_metrics['change_pct']:+.1f}%")
        with metric_col4: st.metric("Momentum", momentum_metrics['strength'])

    st.markdown("---")

    if st.button("Analyze Contract Projections", key="ct_generate", type="primary"):
        if p2_datetime <= p1_datetime:
            st.error("Please ensure Point 2 is after Point 1")
        else:
            with st.spinner("Analyzing contract and market data..."):
                try:
                    time_diff_minutes = (p2_datetime - p1_datetime).total_seconds() / 60
                    blocks_between = time_diff_minutes / 30
                    contract_slope = (p2_price - p1_price) / blocks_between if blocks_between > 0 else 0
                    underlying_data = fetch_live_data("^GSPC", projection_day - timedelta(days=10), projection_day)
                    overnight_analysis = analyze_overnight_market_behavior("^GSPC", projection_day - timedelta(days=10), projection_day)
                    p1_ct = CT_TZ.localize(p1_datetime)
                    contract_projections = project_contract_line(p1_price, p1_ct, contract_slope, projection_day)
                    st.session_state.contract_projections = contract_projections
                    st.session_state.contract_config = {
                        'p1_price': p1_price, 'p1_time': p1_ct, 'p2_price': p2_price,
                        'p2_time': CT_TZ.localize(p2_datetime), 'slope': contract_slope,
                        'momentum': momentum_metrics, 'overnight_analysis': overnight_analysis
                    }
                    st.session_state.underlying_data = underlying_data
                    st.session_state.contract_ready = True
                except Exception as e:
                    st.error(f"Analysis error: {str(e)}")

    if st.session_state.get('contract_ready', False):
        st.subheader("Contract Analysis Results")
        projections = st.session_state.contract_projections
        config = st.session_state.contract_config
        underlying_data = st.session_state.get('underlying_data', pd.DataFrame())
        contract_tabs = st.tabs(["RTH Projections","Market Analysis","Risk Management"])
        with contract_tabs[0]:
            st.subheader("RTH Contract Price Projections")
            if not projections.empty and not underlying_data.empty:
                enhanced = []
                for _, row in projections.iterrows():
                    time_slot = row['Time']; contract_price = row['Contract_Price']
                    direction = "BUY" if config['momentum']['change'] > 0 else "SELL"
                    targets = calculate_market_based_targets(contract_price, underlying_data, direction)
                    hour = int(time_slot.split(':')[0])
                    if hour in [8,9]: time_prob = config['momentum']['confidence'] + 10
                    elif hour in [13,14]: time_prob = config['momentum']['confidence'] + 5
                    else: time_prob = config['momentum']['confidence']
                    enhanced.append({
                        'Time': time_slot, 'Contract_Price': round(contract_price,2),
                        'Direction': direction, 'TP1': round(targets['tp1'],2),
                        'TP2': round(targets['tp2'],2), 'Stop_Distance': round(targets['stop_distance'],2),
                        'Entry_Probability': f"{min(95, time_prob):.0f}%", 'ATR_Base': round(targets.get('atr',0),2)
                    })
                st.dataframe(pd.DataFrame(enhanced), use_container_width=True, hide_index=True)
            else:
                st.dataframe(projections, use_container_width=True, hide_index=True)

        with contract_tabs[1]:
            st.subheader("Underlying Market Analysis")
            momentum = config['momentum']; overnight = config['overnight_analysis']
            c1, c2, c3 = st.columns(3)
            with c1: st.metric("Hourly Rate", f"${momentum['hourly_rate']:+.2f}")
            with c2: st.metric("Strength", momentum['strength'])
            with c3: st.metric("Confidence", f"{momentum['confidence']}%")
            st.subheader("Overnight Market Behavior")
            c1, c2, c3 = st.columns(3)
            with c1: st.metric("Avg Overnight Change", f"{overnight['avg_overnight_change']*100:+.2f}%")
            with c2: st.metric("Overnight Volatility", f"{overnight['overnight_volatility']*100:.2f}%")
            with c3: st.metric("Gap Frequency", f"{overnight['gap_frequency']*100:.1f}%")
            if not underlying_data.empty:
                current_volatility = calculate_contract_volatility(underlying_data)
                day_range = underlying_data['High'].max() - underlying_data['Low'].min()
                cc1, cc2 = st.columns(2)
                with cc1: st.metric("Recent Volatility", f"{current_volatility*100:.2f}%")
                with cc2: st.metric("Recent Range", f"${day_range:.2f}")

        with contract_tabs[2]:
            st.subheader("Risk Management Analysis")
            if not underlying_data.empty and not projections.empty:
                market_volatility = calculate_contract_volatility(underlying_data)
                atr_series = calculate_average_true_range(underlying_data, 14)
                current_atr = atr_series.iloc[-1] if not atr_series.empty else 0
                c1, c2, c3 = st.columns(3)
                with c1:
                    high_vol_threshold = 0.025
                    if market_volatility > high_vol_threshold:
                        position_recommendation = "Reduce Size"; risk_level = "High"
                    else:
                        position_recommendation = "Standard Size"; risk_level = "Normal"
                    st.metric("Risk Level", risk_level); st.caption(f"Volatility: {market_volatility*100:.2f}%")
                with c2:
                    st.metric("Position Sizing", position_recommendation); st.caption(f"ATR: ${current_atr:.2f}")
                with c3:
                    avg_contract_price = projections['Contract_Price'].mean()
                    max_risk_per_contract = current_atr * 1.5
                    risk_per_dollar = (max_risk_per_contract / avg_contract_price) * 100 if avg_contract_price else 0
                    st.metric("Risk per $", f"{risk_per_dollar:.1f}%"); st.caption("Based on ATR stop")
                st.subheader("Time-Based Risk Assessment")
                risk_analysis = []
                overnight_vol = config['overnight_analysis']['overnight_volatility']
                for _, row in projections.iterrows():
                    time_slot = row['Time']; contract_price = row['Contract_Price']
                    hour = int(time_slot.split(':')[0])
                    if hour in [8,9]: risk_multiplier, risk_rating = 1.5, "High"
                    elif hour in [10,11]: risk_multiplier, risk_rating = 1.0, "Medium"
                    else: risk_multiplier, risk_rating = 0.8, "Low"
                    base_stop = current_atr * 1.2
                    adjusted_stop = base_stop * risk_multiplier * (1 + overnight_vol * 2)
                    risk_analysis.append({
                        'Time': time_slot, 'Contract_Price': round(contract_price,2),
                        'Risk_Rating': risk_rating, 'Suggested_Stop': round(adjusted_stop,2),
                        'Risk_Multiplier': f"{risk_multiplier:.1f}x", 'Max_Risk_$': round(adjusted_stop,2)
                    })
                st.dataframe(pd.DataFrame(risk_analysis), use_container_width=True, hide_index=True)
            else:
                st.info("Need underlying market data for comprehensive risk analysis")

def project_contract_line(anchor_price: float, anchor_time: datetime,
                          slope: float, target_date: date) -> pd.DataFrame:
    rth_slots = rth_slots_ct(target_date)
    projections = []
    for slot_time in rth_slots:
        blocks = (slot_time - anchor_time).total_seconds() / 1800
        projected_price = anchor_price + (slope * blocks)
        projections.append({
            'Time': format_ct_time(slot_time),
            'Contract_Price': round(projected_price,2),
            'Blocks_from_Anchor': round(blocks,1)
        })
    return pd.DataFrame(projections)


# ═══════════════════════════════════════════════════════════════════════════════
# SPX PROPHET - PART 6: FINAL INTEGRATION & COMPLETION
# ═══════════════════════════════════════════════════════════════════════════════

# NOTE: This part assumes Parts 1–5 are already executed in the same Streamlit
# app session, so imports like numpy/pandas/streamlit and helpers such as:
#  - CT_TZ, RTH_START/RTH_END, SPX_SLOPES, STOCK_SLOPES
#  - fetch_live_data, calculate_vwap, calculate_ema, calculate_average_true_range
#  - calculate_es_spx_offset, rth_slots_ct, format_ct_time
# have been defined earlier.

# ───────────────────────────────────────────────────────────────────────────────
# MARKET-DERIVED PROBABILITY / STRUCTURE FUNCTIONS
# ───────────────────────────────────────────────────────────────────────────────

def calculate_volume_profile_strength(data: pd.DataFrame, price_level: float) -> float:
    """Volume concentration score (0–100) around a level using actual volume."""
    if data.empty or 'Volume' not in data.columns:
        return 50.0

    tol = max(0.01, price_level * 0.005)  # ~0.5% tolerance, min safeguard
    nearby = data[(data['Low'] <= price_level + tol) & (data['High'] >= price_level - tol)]

    if nearby.empty:
        return 40.0

    level_vol = float(nearby['Volume'].sum())
    total_vol = float(data['Volume'].sum() or 0)
    if total_vol <= 0:
        return 50.0

    conc = (level_vol / total_vol) * 100.0
    if conc >= 20: return 95.0
    if conc >= 15: return 85.0
    if conc >= 10: return 75.0
    if conc >= 5:  return 60.0
    return 45.0

def detect_market_regime(data: pd.DataFrame) -> dict:
    """Trend/volatility regime from last ~20 bars."""
    if data.empty or len(data) < 20:
        return {'regime': 'INSUFFICIENT_DATA', 'trend': 'NEUTRAL', 'strength': 0.0, 'volatility': 1.5, 'price_change': 0.0}

    closes = data['Close'].tail(20)
    price_change = float((closes.iloc[-1] - closes.iloc[0]) / max(1e-9, closes.iloc[0]) * 100.0)

    returns = closes.pct_change().dropna()
    # annualized-ish intraday vol proxy: std * sqrt(390)
    volatility = float(returns.std() * np.sqrt(390) * 100.0) if not returns.empty else 0.0

    if price_change > 1.0:
        trend = 'BULLISH'
    elif price_change < -1.0:
        trend = 'BEARISH'
    else:
        trend = 'NEUTRAL'

    strength = min(100.0, abs(price_change) * 10.0)

    if volatility >= 3.0:
        regime = 'HIGH_VOLATILITY'
    elif volatility >= 1.8:
        regime = 'MODERATE_VOLATILITY'
    else:
        regime = 'LOW_VOLATILITY'

    return {'regime': regime, 'trend': trend, 'strength': strength, 'volatility': volatility, 'price_change': price_change}

def calculate_support_resistance_strength(data: pd.DataFrame, price_level: float) -> float:
    """Pivot touch count → level strength score (0–100)."""
    if data.empty or len(data) < 10:
        return 50.0

    tol = max(0.01, price_level * 0.008)
    highs = data['High']
    lows  = data['Low']

    high_touches = int((highs - price_level).abs().le(tol).sum())
    low_touches  = int((lows  - price_level).abs().le(tol).sum())
    total = high_touches + low_touches

    if total >= 4: return 90.0
    if total >= 3: return 80.0
    if total >= 2: return 70.0
    if total >= 1: return 60.0
    return 45.0

def calculate_confluence_score(price: float, anchor_price: float, market_data: pd.DataFrame) -> float:
    """Weighted confluence of proximity/volume/regime/S-R (0–100)."""
    if market_data.empty or anchor_price <= 0:
        return 50.0

    dist_pct = abs(price - anchor_price) / anchor_price * 100.0
    proximity = max(0.0, 100.0 - dist_pct * 15.0)

    volume_score  = calculate_volume_profile_strength(market_data, price)
    regime_info   = detect_market_regime(market_data)
    regime_score  = 80.0 if regime_info['trend'] != 'NEUTRAL' else 55.0
    sr_score      = calculate_support_resistance_strength(market_data, price)

    confluence = (proximity * 0.25 +
                  volume_score * 0.25 +
                  regime_score * 0.25 +
                  sr_score * 0.25)
    return float(min(100.0, max(0.0, confluence)))

# ───────────────────────────────────────────────────────────────────────────────
# TIME-OF-DAY ANALYSIS
# ───────────────────────────────────────────────────────────────────────────────

def calculate_time_edge_from_data(symbol: str, lookback_days: int = 30) -> dict:
    """Simple time-slot performance over a lookback window."""
    try:
        end_date = datetime.now(CT_TZ).date()
        start_date = end_date - timedelta(days=lookback_days)
        hist = fetch_live_data(symbol, start_date, end_date)
        if hist.empty:
            return {}

        slots = ['08:30', '09:00', '09:30', '10:00', '10:30', '11:00',
                 '11:30', '12:00', '12:30', '13:00', '13:30', '14:00', '14:30']

        results = {}
        for hhmm in slots:
            slot_data = hist.between_time(hhmm, hhmm)  # same-minute bars across days
            if slot_data.empty:
                continue
            slot_ret = slot_data['Close'].pct_change().dropna()
            if len(slot_ret) <= 5:
                continue

            volatility   = float(slot_ret.std() * 100.0)
            avg_move     = float(slot_ret.abs().mean() * 100.0)
            upward_bias  = float((slot_ret > 0).mean() * 100.0)

            if volatility > 0.8 and abs(upward_bias - 50.0) > 10.0:
                edge_score = min(20.0, volatility * 15.0 + abs(upward_bias - 50.0))
            else:
                edge_score = volatility * 10.0

            results[hhmm] = {
                'volatility': volatility,
                'avg_move': avg_move,
                'upward_bias': upward_bias,
                'edge_score': float(edge_score),
            }
        return results
    except Exception:
        return {}

# ───────────────────────────────────────────────────────────────────────────────
# MARKET HOURS / STATUS
# ───────────────────────────────────────────────────────────────────────────────

def get_market_hours_status() -> dict:
    """Rough US market session status in CT."""
    now = datetime.now(CT_TZ)
    weekday = now.weekday() < 5

    market_open  = now.replace(hour=8,  minute=30, second=0, microsecond=0)
    market_close = now.replace(hour=14, minute=30, second=0, microsecond=0)
    pre_start    = now.replace(hour=7,  minute=0,  second=0, microsecond=0)
    ah_end       = now.replace(hour=17, minute=0,  second=0, microsecond=0)

    if not weekday:
        status, session = "WEEKEND", "Closed"
    elif market_open <= now <= market_close:
        status, session = "RTH_OPEN", "Regular Hours"
    elif pre_start <= now < market_open:
        status, session = "PREMARKET", "Pre-Market"
    elif market_close < now <= ah_end:
        status, session = "AFTERHOURS", "After Hours"
    else:
        status, session = "CLOSED", "Closed"

    if status == "CLOSED" and weekday:
        next_open = market_open + timedelta(days=1)
    elif status == "WEEKEND":
        days_until_mon = (7 - now.weekday()) % 7
        days_until_mon = 1 if days_until_mon == 0 else days_until_mon
        next_open = now.replace(hour=8, minute=30, second=0, microsecond=0) + timedelta(days=days_until_mon)
    else:
        next_open = None

    return {
        'status': status,
        'session': session,
        'current_time': now,
        'is_trading_day': weekday,
        'next_open': next_open
    }

# ───────────────────────────────────────────────────────────────────────────────
# SESSION STATE SAFETY (ensures keys exist if user jumps straight here)
# ───────────────────────────────────────────────────────────────────────────────

if 'spx_analysis_ready' not in st.session_state:
    st.session_state.spx_analysis_ready = False
if 'stock_analysis_ready' not in st.session_state:
    st.session_state.stock_analysis_ready = False
if 'signal_ready' not in st.session_state:
    st.session_state.signal_ready = False
if 'contract_ready' not in st.session_state:
    st.session_state.contract_ready = False
if 'current_offset' not in st.session_state:
    st.session_state.current_offset = 0.0
if 'spx_slopes' not in st.session_state:
    st.session_state.spx_slopes = SPX_SLOPES.copy()
if 'stock_slopes' not in st.session_state:
    st.session_state.stock_slopes = STOCK_SLOPES.copy()

# ───────────────────────────────────────────────────────────────────────────────
# MAIN DASHBOARD & SUMMARY
# ───────────────────────────────────────────────────────────────────────────────

st.markdown("---")
st.subheader("Analysis Summary Dashboard")

market_status = get_market_hours_status()

col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("**Analysis Status**")
    st.write(f"SPX Anchors: {'Ready' if st.session_state.spx_analysis_ready else 'Pending'}")
    st.write(f"Stock Analysis: {'Ready' if st.session_state.stock_analysis_ready else 'Pending'}")
    st.write(f"Signal Detection: {'Ready' if st.session_state.signal_ready else 'Pending'}")
    st.write(f"Contract Tool: {'Ready' if st.session_state.contract_ready else 'Pending'}")

with col2:
    st.markdown("**Current Settings**")
    try:
        st.write(f"Skyline Slope: {st.session_state.spx_slopes['skyline']:+.3f}")
        st.write(f"Baseline Slope: {st.session_state.spx_slopes['baseline']:+.3f}")
        st.write(f"High/Close/Low: {st.session_state.spx_slopes['high']:+.3f}")
    except Exception:
        st.write("Slopes unavailable")
    st.write(f"ES→SPX Offset: {st.session_state.current_offset:+.1f}")

with col3:
    st.markdown("**Market Status**")
    st.write(f"Market: {market_status['session']}")
    st.write(f"Time (CT): {market_status['current_time'].strftime('%H:%M:%S')}")
    if market_status['next_open'] is not None:
        delta = market_status['next_open'] - market_status['current_time']
        hours_to_open = int(delta.total_seconds() // 3600)
        st.write(f"Next Open: {hours_to_open}h")
    else:
        st.write("Session Active")

# ───────────────────────────────────────────────────────────────────────────────
# QUICK ACTIONS
# ───────────────────────────────────────────────────────────────────────────────

st.markdown("---")
st.subheader("Quick Actions")

qa1, qa2, qa3, qa4 = st.columns(4)

with qa1:
    if st.button("Update ES Offset", key="quick_update_offset"):
        with st.spinner("Updating offset from market data..."):
            today = datetime.now(CT_TZ).date()
            yesterday = today - timedelta(days=1)
            es_data  = fetch_live_data("ES=F",  yesterday, today)
            spx_data = fetch_live_data("^GSPC", yesterday, today)
            if not es_data.empty and not spx_data.empty:
                st.session_state.current_offset = calculate_es_spx_offset(es_data, spx_data)
                st.success(f"Offset updated: {st.session_state.current_offset:+.1f}")
                st.rerun()
            else:
                st.error("Failed to fetch offset data")

with qa2:
    if st.button("Reset All Analysis", key="quick_reset_all"):
        keys = [
            'spx_analysis_ready', 'stock_analysis_ready', 'signal_ready', 'contract_ready',
            'es_anchor_data', 'spx_manual_anchors', 'stock_analysis_data',
            'signal_data', 'contract_projections', 'contract_config', 'underlying_data'
        ]
        for k in keys:
            if k in st.session_state:
                del st.session_state[k]
        st.success("All analysis reset")
        st.rerun()

with qa3:
    if st.button("Reset Slopes", key="quick_reset_slopes"):
        st.session_state.spx_slopes = SPX_SLOPES.copy()
        st.session_state.stock_slopes = STOCK_SLOPES.copy()
        st.success("Slopes reset to defaults")
        st.rerun()

with qa4:
    if st.button("Test Connection", key="quick_test"):
        with st.spinner("Testing market connection..."):
            test = fetch_live_data("^GSPC", datetime.now().date() - timedelta(days=1), datetime.now().date())
            if not test.empty:
                st.success("Connection successful")
            else:
                st.error("Connection failed")

# ───────────────────────────────────────────────────────────────────────────────
# MARKET PERFORMANCE INSIGHTS
# ───────────────────────────────────────────────────────────────────────────────

if any([
    st.session_state.get('spx_analysis_ready', False),
    st.session_state.get('stock_analysis_ready', False),
    st.session_state.get('signal_ready', False),
    st.session_state.get('contract_ready', False),
]):
    st.markdown("---")
    st.subheader("Market Performance Insights")

    itab1, itab2, itab3 = st.tabs(["Market Regime", "Time-of-Day Edge", "Volume Analysis"])

    with itab1:
        # Use signal_data if available (RTH of chosen symbol/day from Part 4)
        if st.session_state.get('signal_ready', False):
            sig = st.session_state.get('signal_data', pd.DataFrame())
            if not sig.empty:
                regime = detect_market_regime(sig)
                c1, c2 = st.columns(2)
                with c1:
                    st.metric("Market Trend", regime['trend'])
                    st.metric("Trend Strength", f"{regime['strength']:.1f}")
                with c2:
                    st.metric("Volatility Regime", regime['regime'])
                    st.metric("Volatility Level", f"{regime['volatility']:.1f}%")

                if regime['trend'] == 'BULLISH' and regime['volatility'] < 2.5:
                    ctx = "Stable uptrend — consider long setups with standard stops."
                elif regime['trend'] == 'BEARISH' and regime['volatility'] < 2.5:
                    ctx = "Stable downtrend — consider short setups with standard stops."
                elif regime['volatility'] > 3.0:
                    ctx = "High volatility — widen stops and reduce size."
                else:
                    ctx = "Neutral/ranging — mean reversion and fade setups may work."
                st.info(ctx)
            else:
                st.info("No signal dataset available.")
        else:
            st.info("Generate signal analysis to see market regime data.")

    with itab2:
        if st.button("Calculate Time Edge", key="calc_time_edge"):
            with st.spinner("Analyzing time-of-day patterns..."):
                ted = calculate_time_edge_from_data("^GSPC", 30)
                if ted:
                    rows = [{
                        'Time': t,
                        'Volatility': f"{v['volatility']:.2f}%",
                        'Avg Move': f"{v['avg_move']:.2f}%",
                        'Upward Bias': f"{v['upward_bias']:.1f}%",
                        'Edge Score': f"{v['edge_score']:.1f}",
                    } for t, v in ted.items()]
                    df_edge = pd.DataFrame(rows).sort_values('Time')
                    st.dataframe(df_edge, use_container_width=True, hide_index=True)
                else:
                    st.error("Could not calculate time edge data.")
        else:
            st.info("Click 'Calculate Time Edge' to analyze historical time-of-day patterns.")

    with itab3:
        if st.session_state.get('signal_ready', False):
            sig = st.session_state.get('signal_data', pd.DataFrame())
            if not sig.empty and 'Volume' in sig.columns:
                avg_vol = float(sig['Volume'].mean())
                std_vol = float(sig['Volume'].std())
                thresh = avg_vol + std_vol
                high_vol = sig[sig['Volume'] > thresh]

                vc1, vc2, vc3 = st.columns(3)
                with vc1:
                    st.metric("Average Volume", f"{avg_vol:,.0f}")
                with vc2:
                    st.metric("Volume Std Dev", f"{std_vol:,.0f}")
                with vc3:
                    st.metric("High Volume Bars", len(high_vol))

                if not high_vol.empty:
                    hv_move = float((high_vol['Close'] - high_vol['Open']).abs().mean())
                    norm = sig[sig['Volume'] <= thresh]
                    nv_move = float((norm['Close'] - norm['Open']).abs().mean()) if not norm.empty else 0.0
                    st.info(f"High-volume bars avg move: ${hv_move:.2f}")
                    st.info(f"Normal-volume bars avg move: ${nv_move:.2f}")
            else:
                st.info("No volume data available for analysis.")
        else:
            st.info("Run signal analysis first to see volume insights.")

# ───────────────────────────────────────────────────────────────────────────────
# FOOTER / DATA STATUS
# ───────────────────────────────────────────────────────────────────────────────

st.markdown("---")
d1, d2, d3 = st.columns(3)

with d1:
    try:
        spx_t = fetch_live_data("^GSPC", datetime.now().date() - timedelta(days=1), datetime.now().date())
        status = "Active" if not spx_t.empty else "Issue"
        lastup = spx_t.index[-1].strftime("%H:%M CT") if not spx_t.empty else "N/A"
    except Exception:
        status, lastup = "Error", "N/A"
    st.write("**SPX Data**")
    st.write(f"Status: {status}")
    st.write(f"Last Update: {lastup}")

with d2:
    try:
        es_t = fetch_live_data("ES=F", datetime.now().date() - timedelta(days=1), datetime.now().date())
        status = "Active" if not es_t.empty else "Issue"
        lastup = es_t.index[-1].strftime("%H:%M CT") if not es_t.empty else "N/A"
    except Exception:
        status, lastup = "Error", "N/A"
    st.write("**ES Futures**")
    st.write(f"Status: {status}")
    st.write(f"Last Update: {lastup}")

with d3:
    st.write("**Current Session**")
    st.write(f"Offset Status: {'Live' if st.session_state.current_offset != 0 else 'Default'}")
    st.write(f"Session: {market_status['session']}")

st.markdown("---")
st.markdown(
    f"""
    <div style='text-align:center; color:#888; font-size:0.9rem;'>
      SPX Prophet Analytics • Market Data Integration •
      Session: {datetime.now(CT_TZ).strftime('%H:%M:%S CT')} •
      Status: {market_status['session']}
    </div>
    """,
    unsafe_allow_html=True
)

# ═══════════════════════════════════════════════════════════════════════════════
# END • PART 6
# ═══════════════════════════════════════════════════════════════════════════════

