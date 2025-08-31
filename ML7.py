# ═══════════════════════════════════════════════════════════════════════════════
# 🔮 SPX PROPHET — UNIFIED APP (Parts 1–6)
# Strategy preserved; wiring & runtime issues fixed
# ═══════════════════════════════════════════════════════════════════════════════

# ───────────────────────────────────────────────────────────────────────────────
# Imports & setup (Part 1)
# ───────────────────────────────────────────────────────────────────────────────
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import pytz
from datetime import datetime, date, time, timedelta
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# ───────────────────────────────────────────────────────────────────────────────
# Core configuration (Part 1)
# ───────────────────────────────────────────────────────────────────────────────
CT_TZ = pytz.timezone('America/Chicago')
UTC_TZ = pytz.UTC

# Trading parameters
RTH_START = "08:30"  # RTH start in CT
RTH_END = "14:30"    # RTH end in CT
SPX_ANCHOR_START = "17:00"  # SPX anchor window start CT
SPX_ANCHOR_END = "19:30"    # SPX anchor window end CT

# Default slopes per 30-min block (Part 1)
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

# ───────────────────────────────────────────────────────────────────────────────
# Streamlit config & styling (Part 1)
# ───────────────────────────────────────────────────────────────────────────────
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
    .success-box { background: linear-gradient(135deg, #00C851, #007E33); padding: 1rem; border-radius: 10px; border-left: 5px solid #00ff88; }
    .warning-box { background: linear-gradient(135deg, #ffbb33, #ff8800); padding: 1rem; border-radius: 10px; border-left: 5px solid #ffff00; }
    .info-box { background: linear-gradient(135deg, #33b5e5, #0099cc); padding: 1rem; border-radius: 10px; border-left: 5px solid #00ddff; }
    .stDataFrame { background: rgba(255,255,255,0.95); border-radius: 10px; overflow: hidden; }
</style>
""", unsafe_allow_html=True)

# ───────────────────────────────────────────────────────────────────────────────
# Data fetching utilities (Part 1)
# ───────────────────────────────────────────────────────────────────────────────
@st.cache_data(ttl=60)
def fetch_live_data(symbol: str, start_date: date, end_date: date) -> pd.DataFrame:
    """Fetch yfinance data (30m) with OHLC extraction & CT index."""
    try:
        buffer_start = start_date - timedelta(days=5)
        buffer_end = end_date + timedelta(days=2)

        ticker = yf.Ticker(symbol)
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

        # Flatten potential MultiIndex columns
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0] if isinstance(c, tuple) else str(c) for c in df.columns]

        # Ensure required columns exist
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            st.error(f"❌ Missing columns for {symbol}: {missing}")
            return pd.DataFrame()

        # Timezone normalize → convert to CT
        if df.index.tz is None:
            # yfinance typically returns US/Eastern tz-aware index at intraday, but safeguard:
            df.index = df.index.tz_localize('US/Eastern')
        df.index = df.index.tz_convert(CT_TZ)

        # Clip to requested window (local CT)
        start_dt = CT_TZ.localize(datetime.combine(start_date, time(0, 0)))
        end_dt = CT_TZ.localize(datetime.combine(end_date, time(23, 59)))
        df = df.loc[start_dt:end_dt]

        # Validate OHLC basics
        if not validate_ohlc_data(df):
            st.warning(f"⚠️ Data quality issues detected for {symbol}")

        return df

    except Exception as e:
        st.error(f"❌ Error fetching {symbol}: {str(e)}")
        return pd.DataFrame()

def validate_ohlc_data(df: pd.DataFrame) -> bool:
    if df.empty:
        return False
    for c in ['Open','High','Low','Close']:
        if c not in df.columns:
            return False
    invalid = (
        (df['High'] < df['Low']) |
        (df['High'] < df['Open']) |
        (df['High'] < df['Close']) |
        (df['Low'] > df['Open']) |
        (df['Low'] > df['Close']) |
        (df['Close'] <= 0) |
        (df['High'] <= 0)
    )
    return not invalid.any()

@st.cache_data(ttl=300)
def fetch_historical_data(symbol: str, days_back: int = 30) -> pd.DataFrame:
    end_date = datetime.now(CT_TZ).date()
    start_date = end_date - timedelta(days=days_back)
    return fetch_live_data(symbol, start_date, end_date)

# ───────────────────────────────────────────────────────────────────────────────
# Time & session helpers (Part 1)
# ───────────────────────────────────────────────────────────────────────────────
def rth_slots_ct(target_date: date) -> List[datetime]:
    start_ct = CT_TZ.localize(datetime.combine(target_date, time(8,30)))
    end_ct = CT_TZ.localize(datetime.combine(target_date, time(14,30)))
    slots, cur = [], start_ct
    while cur <= end_ct:
        slots.append(cur)
        cur += timedelta(minutes=30)
    return slots

def format_ct_time(dt_: datetime) -> str:
    if dt_.tzinfo is None:
        dt_ = CT_TZ.localize(dt_)
    elif dt_.tzinfo != CT_TZ:
        dt_ = dt_.astimezone(CT_TZ)
    return dt_.strftime("%H:%M")

def get_session_window(df: pd.DataFrame, start_time: str, end_time: str) -> pd.DataFrame:
    if df.empty:
        return df
    return df.between_time(start_time, end_time)

def get_daily_ohlc(df: pd.DataFrame, target_date: date) -> Dict:
    if df.empty:
        return {}
    start_dt = CT_TZ.localize(datetime.combine(target_date, time(0,0)))
    end_dt = CT_TZ.localize(datetime.combine(target_date, time(23,59)))
    day = df.loc[start_dt:end_dt]
    if day.empty:
        return {}
    day_open = day.iloc[0]['Open']
    day_high = day['High'].max()
    day_low = day['Low'].min()
    day_close = day.iloc[-1]['Close']
    high_time = day[day['High'] == day_high].index[0]
    low_time = day[day['Low'] == day_low].index[0]
    open_time = day.index[0]
    close_time = day.index[-1]
    return {
        'open': (day_open, open_time),
        'high': (day_high, high_time),
        'low': (day_low, low_time),
        'close': (day_close, close_time),
    }

# ───────────────────────────────────────────────────────────────────────────────
# Swing detection & anchors (Part 1)
# ───────────────────────────────────────────────────────────────────────────────
def detect_swings_simple(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or len(df) < 2:
        return df.copy()
    out = df.copy()
    out['swing_high'] = False
    out['swing_low'] = False
    if 'Close' in out.columns:
        max_idx = out['Close'].idxmax()
        min_idx = out['Close'].idxmin()
        out.loc[max_idx, 'swing_high'] = True
        out.loc[min_idx, 'swing_low'] = True
    return out

def get_anchor_points(df_swings: pd.DataFrame) -> Tuple[Optional[Tuple], Optional[Tuple]]:
    skyline, baseline = None, None
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

# ───────────────────────────────────────────────────────────────────────────────
# Projection & indicators (Part 1)
# ───────────────────────────────────────────────────────────────────────────────
def project_anchor_line(anchor_price: float, anchor_time: datetime, slope: float, target_date: date) -> pd.DataFrame:
    rth = rth_slots_ct(target_date)
    rows = []
    for slot in rth:
        blocks = (slot - anchor_time).total_seconds() / 1800.0
        rows.append({
            'Time': format_ct_time(slot),
            'Price': round(anchor_price + (slope * blocks), 2),
            'Blocks': round(blocks, 1),
            'Anchor_Price': round(anchor_price, 2),
            'Slope': slope
        })
    return pd.DataFrame(rows)

def calculate_ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span).mean()

def calculate_vwap(df: pd.DataFrame) -> pd.Series:
    if df.empty or 'Volume' not in df.columns:
        return pd.Series(index=df.index, dtype=float)
    typical = (df['High'] + df['Low'] + df['Close']) / 3
    cum_vol = df['Volume'].cumsum()
    cum_vp = (typical * df['Volume']).cumsum()
    vwap = cum_vp / cum_vol
    return vwap.fillna(method='ffill').fillna(typical)

def calculate_es_spx_offset(es_data: pd.DataFrame, spx_data: pd.DataFrame) -> float:
    try:
        if es_data.empty or spx_data.empty:
            return 0.0
        es_rth = get_session_window(es_data, "08:30", "15:00")
        spx_rth = get_session_window(spx_data, "08:30", "15:00")
        if es_rth.empty or spx_rth.empty:
            es_close = es_data.iloc[-1]['Close']
            spx_close = spx_data.iloc[-1]['Close']
        else:
            es_close = es_rth.iloc[-1]['Close']
            spx_close = spx_rth.iloc[-1]['Close']
        return round(spx_close - es_close, 1)
    except Exception as e:
        st.warning(f"Offset calculation error: {str(e)}")
        return 0.0

# ───────────────────────────────────────────────────────────────────────────────
# Session state init (Part 1)
# ───────────────────────────────────────────────────────────────────────────────
if 'theme' not in st.session_state:
    st.session_state.theme = 'Dark'
if 'spx_slopes' not in st.session_state:
    st.session_state.spx_slopes = SPX_SLOPES.copy()
if 'stock_slopes' not in st.session_state:
    st.session_state.stock_slopes = STOCK_SLOPES.copy()
if 'current_offset' not in st.session_state:
    st.session_state.current_offset = 0.0

# ───────────────────────────────────────────────────────────────────────────────
# Sidebar controls (Part 1)
# ───────────────────────────────────────────────────────────────────────────────
st.sidebar.title("🔮 SPX Prophet Analytics")
st.sidebar.markdown("---")
theme = st.sidebar.radio("🎨 Theme", ["Dark", "Light"], key="ui_theme")
st.session_state.theme = theme
st.sidebar.markdown("---")

st.sidebar.markdown("### SPX Slopes (per 30-min)")
st.sidebar.caption("Adjust projection slopes for each anchor type")
with st.sidebar.expander("SPX Slope Settings", expanded=False):
    for slope_name, default_value in SPX_SLOPES.items():
        icon_map = {'high':'High','close':'Close','low':'Low','skyline':'Skyline','baseline':'Baseline'}
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

# ───────────────────────────────────────────────────────────────────────────────
# Main header & top metrics (Part 1)
# ───────────────────────────────────────────────────────────────────────────────
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
    market_close = current_time_ct.replace(hour=14, minute=30, second=0, microsecond=0)
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
# PART 2 — SPX Anchors Tab (fixed wiring, strategy unchanged)
# ═══════════════════════════════════════════════════════════════════════════════
def update_offset_for_date():
    """Automatically update ES→SPX offset when prev-day date changes."""
    if 'spx_prev_day' in st.session_state:
        selected_date = st.session_state.spx_prev_day
        es_data = fetch_live_data("ES=F", selected_date, selected_date)
        spx_data = fetch_live_data("^GSPC", selected_date, selected_date)
        if not es_data.empty and not spx_data.empty:
            st.session_state.current_offset = calculate_es_spx_offset(es_data, spx_data)

def calculate_anchor_entry_probability(anchor_type: str, time_slot: str) -> float:
    base_probs = {'SKYLINE': 90.0, 'BASELINE': 90.0, 'HIGH': 75.0, 'CLOSE': 80.0, 'LOW': 75.0}
    base = base_probs.get(anchor_type.upper(), 70.0)
    hour = int(time_slot.split(':')[0])
    time_adj = 8 if hour in [8, 9] else (5 if hour in [13,14] else 0)
    return min(95, base + time_adj)

def calculate_anchor_target_probability(anchor_type: str, target_num: int) -> float:
    if anchor_type.upper() in ['SKYLINE','BASELINE']:
        return 85.0 if target_num == 1 else 68.0
    return 75.0 if target_num == 1 else 55.0

def calculate_entry_exit_table(projection_df: pd.DataFrame, anchor_type: str) -> pd.DataFrame:
    if projection_df.empty:
        return pd.DataFrame()
    rows = []
    for _, row in projection_df.iterrows():
        tslot = row['Time']
        anchor_price = row['Price']
        at = anchor_type.upper()

        if at == 'HIGH':
            vol = anchor_price * 0.010
            tp1d, tp2d = vol * 0.7, vol * 1.8
            entry = anchor_price
            tp1 = anchor_price - tp1d
            tp2 = anchor_price - tp2d
            direction = "SELL"
            stop = anchor_price + (anchor_price * 0.005)

        elif at in ['LOW','CLOSE']:
            vol = anchor_price * 0.010
            tp1d, tp2d = vol * 0.7, vol * 1.8
            entry = anchor_price
            tp1 = anchor_price + tp1d
            tp2 = anchor_price + tp2d
            direction = "BUY"
            stop = anchor_price - (anchor_price * 0.005)

        elif at in ['SKYLINE']:
            vol = anchor_price * 0.012
            tp1d, tp2d = vol * 0.8, vol * 2.2
            entry = anchor_price
            tp1 = anchor_price + tp1d
            tp2 = anchor_price + tp2d
            direction = "BUY"
            stop = anchor_price + (anchor_price * 0.006)

        elif at in ['BASELINE']:
            vol = anchor_price * 0.012
            tp1d, tp2d = vol * 0.8, vol * 2.2
            entry = anchor_price
            tp1 = anchor_price + tp1d
            tp2 = anchor_price + tp2d
            direction = "BUY"
            stop = max(0.01, anchor_price - (anchor_price * 0.006))
        else:
            continue

        risk_amt = abs(entry - stop)
        rr1 = abs(tp1 - entry) / risk_amt if risk_amt > 0 else 0
        rr2 = abs(tp2 - entry) / risk_amt if risk_amt > 0 else 0

        rows.append({
            'Time': tslot,
            'Direction': direction,
            'Entry': round(entry, 2),
            'Stop': round(stop, 2),
            'TP1': round(tp1, 2),
            'TP2': round(tp2, 2),
            'Risk': round(risk_amt, 2),
            'RR1': f"{rr1:.1f}",
            'RR2': f"{rr2:.1f}",
            'Entry_Prob': f"{calculate_anchor_entry_probability(at, tslot):.0f}%",
            'TP1_Prob': f"{calculate_anchor_target_probability(at, 1):.0f}%",
            'TP2_Prob': f"{calculate_anchor_target_probability(at, 2):.0f}%"
        })
    return pd.DataFrame(rows)

# Create tabs (Part 2 owns the tab creation; Parts 3–5 use tab2–tab4)
tab1, tab2, tab3, tab4 = st.tabs(["SPX Anchors", "Stock Anchors", "Signals & EMA", "Contract Tool"])

with tab1:
    st.subheader("SPX Anchor Analysis")
    st.caption("Live ES futures data for anchor detection and SPX projections")

    c1, c2 = st.columns(2)
    with c1:
        prev_day = st.date_input(
            "Previous Trading Day",
            value=datetime.now(CT_TZ).date() - timedelta(days=1),
            key="spx_prev_day",
            on_change=update_offset_for_date
        )
        st.caption(f"Selected: {prev_day.strftime('%A')}")
    with c2:
        proj_day = st.date_input(
            "Projection Day",
            value=prev_day + timedelta(days=1),
            key="spx_proj_day"
        )
        st.caption(f"Projecting for: {proj_day.strftime('%A')}")

    st.markdown("---")
    st.subheader("Price Override (Optional)")
    st.caption("Override Yahoo Finance data with your exact prices for accurate projections")
    use_manual = st.checkbox("Use Manual Prices", key="use_manual_prices")
    if use_manual:
        oc1, oc2, oc3 = st.columns(3)
        with oc1:
            manual_high = st.number_input("Manual High Price", value=0.0, step=0.1, format="%.1f", key="manual_high_price")
        with oc2:
            manual_close = st.number_input("Manual Close Price", value=0.0, step=0.1, format="%.1f", key="manual_close_price")
        with oc3:
            manual_low = st.number_input("Manual Low Price", value=0.0, step=0.1, format="%.1f", key="manual_low_price")

    st.markdown("---")

    if st.button("Generate SPX Anchors", key="spx_generate", type="primary"):
        with st.spinner("Analyzing market data..."):
            try:
                es_for_off = fetch_live_data("ES=F", prev_day, prev_day)
                spx_for_off = fetch_live_data("^GSPC", prev_day, prev_day)
                if not es_for_off.empty and not spx_for_off.empty:
                    st.session_state.current_offset = calculate_es_spx_offset(es_for_off, spx_for_off)

                es_data = fetch_live_data("ES=F", prev_day, prev_day)
                if es_data.empty:
                    st.error(f"No ES futures data for {prev_day}")
                else:
                    anchor_window = get_session_window(es_data, SPX_ANCHOR_START, SPX_ANCHOR_END)
                    if anchor_window.empty:
                        anchor_window = es_data
                    st.session_state.es_anchor_data = anchor_window

                    spx_data = fetch_live_data("^GSPC", prev_day, prev_day)
                    if not spx_data.empty:
                        daily_ohlc = get_daily_ohlc(spx_data, prev_day)
                        if daily_ohlc:
                            st.session_state.spx_manual_anchors = daily_ohlc
                        else:
                            st.warning("Could not extract SPX OHLC data")
                    else:
                        es_daily = get_daily_ohlc(anchor_window, prev_day)
                        if es_daily:
                            off = st.session_state.current_offset
                            st.session_state.spx_manual_anchors = {k: (p + off, ts) for k, (p, ts) in es_daily.items()}
                st.session_state.spx_analysis_ready = True
            except Exception as e:
                st.error(f"Analysis error: {str(e)}")

    if st.session_state.get('spx_analysis_ready', False):
        st.subheader("SPX Anchor Projections")
        es_data = st.session_state.get('es_anchor_data', pd.DataFrame())
        skyline_anchor_spx, baseline_anchor_spx = None, None
        if not es_data.empty:
            es_swings = detect_swings_simple(es_data)
            es_sky, es_base = get_anchor_points(es_swings)
            off = st.session_state.current_offset
            if es_sky:
                skyline_anchor_spx = (es_sky[0] + off, es_sky[1])
            if es_base:
                baseline_anchor_spx = (es_base[0] + off, es_base[1])

        if st.session_state.get('spx_manual_anchors'):
            manual = st.session_state.spx_manual_anchors
            st.subheader("Detected SPX Anchors")
            summary_cols = st.columns(5)

            for i, (name, label, color) in enumerate([('high','High','#ff6b6b'),
                                                      ('close','Close','#f9ca24'),
                                                      ('low','Low','#4ecdc4')]):
                if name in manual:
                    price, ts = manual[name]
                    with summary_cols[i]:
                        st.markdown(f"""
                        <div style="text-align: center; padding: 1rem; background: rgba(255,255,255,0.1);
                        border-radius: 10px; border-left: 4px solid {color};">
                            <h4>{label}</h4>
                            <h3>${price:.2f}</h3>
                            <p>{format_ct_time(ts)}</p>
                        </div>
                        """, unsafe_allow_html=True)

            with summary_cols[3]:
                if skyline_anchor_spx:
                    price, ts = skyline_anchor_spx
                    st.markdown(f"""
                    <div style="text-align: center; padding: 1rem; background: rgba(255,100,100,0.2);
                    border-radius: 10px; border-left: 4px solid #ff4757;">
                        <h4>Skyline</h4>
                        <h3>${price:.2f}</h3>
                        <p>{format_ct_time(ts)}</p>
                        <small>SPX Equivalent</small>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.warning("No Skyline")

            with summary_cols[4]:
                if baseline_anchor_spx:
                    price, ts = baseline_anchor_spx
                    st.markdown(f"""
                    <div style="text-align: center; padding: 1rem; background: rgba(100,100,255,0.2);
                    border-radius: 10px; border-left: 4px solid #3742fa;">
                        <h4>Baseline</h4>
                        <h3>${price:.2f}</h3>
                        <p>{format_ct_time(ts)}</p>
                        <small>SPX Equivalent</small>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.warning("No Baseline")

        st.markdown("---")
        proj_tabs = st.tabs(["High", "Close", "Low", "Skyline", "Baseline"])

        if st.session_state.get('spx_manual_anchors'):
            manual = st.session_state.spx_manual_anchors

            with proj_tabs[0]:
                if 'high' in manual:
                    price, ts = manual['high']; ts_ct = ts.astimezone(CT_TZ)
                    high_proj = project_anchor_line(price, ts_ct, st.session_state.spx_slopes['high'], proj_day)
                    if not high_proj.empty:
                        st.subheader("High Anchor SPX Projection")
                        st.dataframe(high_proj, use_container_width=True, hide_index=True)
                        high_analysis = calculate_entry_exit_table(high_proj, "HIGH")
                        if not high_analysis.empty:
                            st.subheader("Entry/Exit Strategy")
                            st.dataframe(high_analysis, use_container_width=True, hide_index=True)
                else:
                    st.warning("No high anchor data available")

            with proj_tabs[1]:
                if 'close' in manual:
                    price, ts = manual['close']; ts_ct = ts.astimezone(CT_TZ)
                    close_proj = project_anchor_line(price, ts_ct, st.session_state.spx_slopes['close'], proj_day)
                    if not close_proj.empty:
                        st.subheader("Close Anchor SPX Projection")
                        st.dataframe(close_proj, use_container_width=True, hide_index=True)
                        close_analysis = calculate_entry_exit_table(close_proj, "CLOSE")
                        if not close_analysis.empty:
                            st.subheader("Entry/Exit Strategy")
                            st.dataframe(close_analysis, use_container_width=True, hide_index=True)
                else:
                    st.warning("No close anchor data available")

            with proj_tabs[2]:
                if 'low' in manual:
                    price, ts = manual['low']; ts_ct = ts.astimezone(CT_TZ)
                    low_proj = project_anchor_line(price, ts_ct, st.session_state.spx_slopes['low'], proj_day)
                    if not low_proj.empty:
                        st.subheader("Low Anchor SPX Projection")
                        st.dataframe(low_proj, use_container_width=True, hide_index=True)
                        low_analysis = calculate_entry_exit_table(low_proj, "LOW")
                        if not low_analysis.empty:
                            st.subheader("Entry/Exit Strategy")
                            st.dataframe(low_analysis, use_container_width=True, hide_index=True)
                else:
                    st.warning("No low anchor data available")

        with proj_tabs[3]:
            if skyline_anchor_spx:
                s_price, s_time = skyline_anchor_spx; s_time_ct = s_time.astimezone(CT_TZ)
                skyline_proj = project_anchor_line(s_price, s_time_ct, st.session_state.spx_slopes['skyline'], proj_day)
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

        with proj_tabs[4]:
            if baseline_anchor_spx:
                b_price, b_time = baseline_anchor_spx; b_time_ct = b_time.astimezone(CT_TZ)
                baseline_proj = project_anchor_line(b_price, b_time_ct, st.session_state.spx_slopes['baseline'], proj_day)
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
# PART 3 — Stock Anchors Tab (uses tab2)
# ═══════════════════════════════════════════════════════════════════════════════
def calculate_weekly_entry_exit_table(projection_df: pd.DataFrame, ticker: str, anchor_type: str, day_name: str) -> pd.DataFrame:
    if projection_df.empty:
        return pd.DataFrame()
    analysis_rows = []
    stock_volatility = get_stock_volatility_factor(ticker)
    day_multipliers = {"Wednesday": 1.1, "Thursday": 1.0, "Friday": 0.9}
    day_mult = day_multipliers.get(day_name, 1.0)

    for _, row in projection_df.iterrows():
        time_slot = row['Time']
        price = row['Price']
        stop_distance = price * stock_volatility * 0.012
        tp1_distance = stop_distance * 1.5
        tp2_distance = stop_distance * 2.5

        slope_sign = 1 if anchor_type in ['SKYLINE', 'HIGH'] else -1
        entry_price = price
        stop_price = price - (stop_distance * slope_sign)
        tp1_price = price + (tp1_distance * slope_sign)
        tp2_price = price + (tp2_distance * slope_sign)
        direction = "LONG" if slope_sign > 0 else "SHORT"

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
    volatility_factors = {
        'TSLA': 1.8, 'NVDA': 1.6, 'META': 1.4, 'NFLX': 1.3,
        'AMZN': 1.2, 'GOOGL': 1.1, 'MSFT': 1.0, 'AAPL': 0.9
    }
    return volatility_factors.get(ticker, 1.2)

def calculate_stock_entry_probability(ticker: str, anchor_type: str, time_slot: str) -> float:
    base_probs = {'HIGH': 60, 'CLOSE': 65, 'LOW': 60, 'SKYLINE': 70, 'BASELINE': 75}
    base_prob = base_probs.get(anchor_type, 60)
    hour = int(time_slot.split(':')[0])
    time_adj = 10 if hour in [9, 10] else (5 if hour in [13, 14] else 0)
    ticker_adj = 5 if ticker in ['TSLA','NVDA','META'] else (-5 if ticker in ['AAPL','MSFT'] else 0)
    final = base_prob + time_adj + ticker_adj
    return min(90, max(40, final))

def calculate_stock_target_probability(ticker: str, target_distance: float, stop_distance: float, target_num: int) -> float:
    rr_ratio = target_distance / stop_distance if stop_distance > 0 else 1.0
    vol = get_stock_volatility_factor(ticker)
    if target_num == 1:
        base = 65 - (rr_ratio - 1.5) * 8
        vol_adj = (vol - 1) * 10
    else:
        base = 40 - (rr_ratio - 2.5) * 6
        vol_adj = (vol - 1) * 8
    return min(80, max(20, base + vol_adj))

with tab2:
    st.subheader("Stock Anchor Analysis")
    st.caption("Mon/Tue combined session analysis for individual stocks")

    st.write("Core Tickers:")
    ticker_cols = st.columns(4)
    selected_ticker = None
    core_tickers = ['TSLA', 'NVDA', 'AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META', 'NFLX']
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
            value=current_slope,
            step=0.0001, format="%.4f",
            key=f"stk_slope_{selected_ticker}",
            help="Used as +magnitude for Skyline, -magnitude for Baseline"
        )
        st.session_state.stock_slopes[selected_ticker] = slope_magnitude

        c1, c2, c3 = st.columns(3)
        with c1:
            monday_date = st.date_input(
                "Monday Date",
                value=datetime.now(CT_TZ).date() - timedelta(days=2),
                key=f"stk_mon_{selected_ticker}"
            )
        with c2:
            tuesday_date = st.date_input(
                "Tuesday Date",
                value=monday_date + timedelta(days=1),
                key=f"stk_tue_{selected_ticker}"
            )
        with c3:
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
                    st.warning("No Monday data, using Tuesday only")
                    combined_data = tue_data
                elif tue_data.empty:
                    st.warning("No Tuesday data, using Monday only")
                    combined_data = mon_data
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

            # Use the same approach as Part 1: detect_swings_simple (fix for undefined detect_swings/SWING_K)
            stock_swings = detect_swings_simple(stock_data)
            skyline_anchor, baseline_anchor = get_anchor_points(stock_swings)

            # Use last bar close as manual anchors placeholder (as you did)
            last_bar = stock_data.iloc[-1]
            manual_anchors = {
                'high': (last_bar['Close'], last_bar.name),
                'close': (last_bar['Close'], last_bar.name),
                'low': (last_bar['Close'], last_bar.name)
            }

            st.subheader(f"{selected_ticker} Weekly Projections")
            projection_dates = [("Wednesday", tuesday_date + timedelta(days=1)),
                                ("Thursday", tuesday_date + timedelta(days=2)),
                                ("Friday", tuesday_date + timedelta(days=3))]
            weekly_tabs = st.tabs(["Wed", "Thu", "Fri"])

            for day_idx, (day_name, proj_date) in enumerate(projection_dates):
                with weekly_tabs[day_idx]:
                    st.write(f"{day_name} - {proj_date}")
                    anchor_subtabs = st.tabs(["High", "Close", "Low", "Skyline", "Baseline"])

                    with anchor_subtabs[0]:
                        price, ts = manual_anchors['high']; ts_ct = ts.astimezone(CT_TZ)
                        high_proj = project_anchor_line(price, ts_ct, slope_magnitude, proj_date)
                        st.dataframe(high_proj, use_container_width=True)
                        high_analysis = calculate_weekly_entry_exit_table(high_proj, selected_ticker, "HIGH", day_name)
                        st.write("Entry/Exit Analysis")
                        st.dataframe(high_analysis, use_container_width=True)

                    with anchor_subtabs[1]:
                        price, ts = manual_anchors['close']; ts_ct = ts.astimezone(CT_TZ)
                        close_proj = project_anchor_line(price, ts_ct, slope_magnitude, proj_date)
                        st.dataframe(close_proj, use_container_width=True)
                        close_analysis = calculate_weekly_entry_exit_table(close_proj, selected_ticker, "CLOSE", day_name)
                        st.write("Entry/Exit Analysis")
                        st.dataframe(close_analysis, use_container_width=True)

                    with anchor_subtabs[2]:
                        price, ts = manual_anchors['low']; ts_ct = ts.astimezone(CT_TZ)
                        low_proj = project_anchor_line(price, ts_ct, slope_magnitude, proj_date)
                        st.dataframe(low_proj, use_container_width=True)
                        low_analysis = calculate_weekly_entry_exit_table(low_proj, selected_ticker, "LOW", day_name)
                        st.write("Entry/Exit Analysis")
                        st.dataframe(low_analysis, use_container_width=True)

                    with anchor_subtabs[3]:
                        if skyline_anchor:
                            s_price, s_time = skyline_anchor; s_time_ct = s_time.astimezone(CT_TZ)
                            skyline_proj = project_anchor_line(s_price, s_time_ct, slope_magnitude, proj_date)
                            st.dataframe(skyline_proj, use_container_width=True)
                            sky_analysis = calculate_weekly_entry_exit_table(skyline_proj, selected_ticker, "SKYLINE", day_name)
                            st.write("Entry/Exit Analysis")
                            st.dataframe(sky_analysis, use_container_width=True)
                        else:
                            st.warning("No skyline anchor detected")

                    with anchor_subtabs[4]:
                        if baseline_anchor:
                            b_price, b_time = baseline_anchor; b_time_ct = b_time.astimezone(CT_TZ)
                            baseline_proj = project_anchor_line(b_price, b_time_ct, -slope_magnitude, proj_date)
                            st.dataframe(baseline_proj, use_container_width=True)
                            base_analysis = calculate_weekly_entry_exit_table(baseline_proj, selected_ticker, "BASELINE", day_name)
                            st.write("Entry/Exit Analysis")
                            st.dataframe(base_analysis, use_container_width=True)
                        else:
                            st.warning("No baseline anchor detected")

# ═══════════════════════════════════════════════════════════════════════════════
# PART 4 — Signals & EMA Tab (uses tab3)
# ═══════════════════════════════════════════════════════════════════════════════
def calculate_market_volatility(data: pd.DataFrame) -> float:
    if data.empty or len(data) < 2:
        return 1.5
    returns = data['Close'].pct_change().dropna()
    if returns.empty:
        return 1.5
    volatility = returns.std() * np.sqrt(390)
    return volatility * 100

def calculate_average_true_range(data: pd.DataFrame, periods: int = 14) -> pd.Series:
    if data.empty or len(data) < periods:
        return pd.Series(index=data.index, dtype=float)
    high_low = data['High'] - data['Low']
    high_close = np.abs(data['High'] - data['Close'].shift())
    low_close = np.abs(data['Low'] - data['Close'].shift())
    tr = np.maximum(high_low, np.maximum(high_close, low_close))
    return tr.rolling(window=periods).mean()

def detect_anchor_touches(price_data: pd.DataFrame, anchor_line: pd.DataFrame) -> pd.DataFrame:
    if price_data.empty or anchor_line.empty:
        return pd.DataFrame()
    anchor_dict = {row['Time']: row['Price'] for _, row in anchor_line.iterrows()}
    touches = []
    for idx, bar in price_data.iterrows():
        bar_time = format_ct_time(idx)
        if bar_time not in anchor_dict:
            continue
        anchor_price = anchor_dict[bar_time]
        low_distance = abs(bar['Low'] - anchor_price)
        high_distance = abs(bar['High'] - anchor_price)
        recent_atr = calculate_average_true_range(price_data.tail(20), 14)
        if not recent_atr.empty:
            tolerance = recent_atr.iloc[-1] * 0.3
        else:
            tolerance = anchor_price * 0.002
        touches_anchor = (bar['Low'] <= anchor_price + tolerance and bar['High'] >= anchor_price - tolerance)
        if touches_anchor:
            is_bearish = bar['Close'] < bar['Open']
            is_bullish = bar['Close'] > bar['Open']
            closest_distance = min(low_distance, high_distance)
            touch_quality = max(0, 100 - (closest_distance / tolerance * 100)) if tolerance > 0 else 0
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
                'Volume': int(bar['Volume']) if 'Volume' in price_data.columns else 0,
                'Touch_Quality': round(touch_quality, 1),
                'Volume_Strength': round(volume_strength, 1),
                'ATR_Tolerance': round(float(tolerance), 2)
            })
    return pd.DataFrame(touches)

def analyze_anchor_line_interaction(price_data: pd.DataFrame, anchor_line: pd.DataFrame) -> pd.DataFrame:
    if price_data.empty or anchor_line.empty:
        return pd.DataFrame()
    anchor_dict = {row['Time']: row['Price'] for _, row in anchor_line.iterrows()}
    interactions = []
    for idx, bar in price_data.iterrows():
        bar_time = format_ct_time(idx)
        if bar_time not in anchor_dict:
            continue
        anchor_price = anchor_dict[bar_time]
        price_above = bar['Close'] > anchor_price
        touched = (bar['Low'] <= anchor_price <= bar['High'])
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
    if price_data.empty or len(price_data) < 21:
        return pd.DataFrame()
    ema8 = calculate_ema(price_data['Close'], 8)
    ema21 = calculate_ema(price_data['Close'], 21)
    crossovers = []
    for i in range(1, len(price_data)):
        current_time = format_ct_time(price_data.index[i])
        prev_8, prev_21 = ema8.iloc[i-1], ema21.iloc[i-1]
        curr_8, curr_21 = ema8.iloc[i], ema21.iloc[i]
        current_price = price_data.iloc[i]['Close']
        crossover_type = None
        strength = abs(curr_8 - curr_21) / curr_21 * 100 if curr_21 != 0 else 0
        if prev_8 <= prev_21 and curr_8 > curr_21:
            crossover_type = "Bullish Cross"
        elif prev_8 >= prev_21 and curr_8 < curr_21:
            crossover_type = "Bearish Cross"
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

with tab3:
    st.subheader("Signal Detection & Market Analysis")
    st.caption("Real-time anchor touch detection with market-derived analytics")

    c1, c2 = st.columns(2)
    with c1:
        signal_symbol = st.selectbox("Analysis Symbol", ["^GSPC", "ES=F", "SPY"], index=0, key="sig_symbol")
    with c2:
        signal_day = st.date_input("Analysis Day", value=datetime.now(CT_TZ).date(), key="sig_day")

    st.markdown("Reference Line Configuration")
    rc1, rc2, rc3 = st.columns(3)
    with rc1:
        anchor_price = st.number_input("Anchor Price", value=6000.0, step=0.1, format="%.2f", key="sig_anchor_price")
    with rc2:
        anchor_time_input = st.time_input("Anchor Time (CT)", value=time(17, 0), key="sig_anchor_time")
    with rc3:
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
        oc1, oc2, oc3 = st.columns(3)
        with oc1:
            day_range = signal_data['High'].max() - signal_data['Low'].min()
            st.metric("Day Range", f"${day_range:.2f}")
        with oc2:
            st.metric("Volatility", f"{volatility:.2f}%")
        with oc3:
            current_atr = atr_series.iloc[-1] if not atr_series.empty else 0
            st.metric("Current ATR", f"${current_atr:.2f}")

        st.markdown("---")
        signal_tabs = st.tabs(["Reference Line", "Anchor Touches", "Line Interaction", "EMA Analysis"])

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
                tc1, tc2, tc3 = st.columns(3)
                with tc1: st.metric("Total Touches", total_touches)
                with tc2: st.metric("Avg Touch Quality", f"{avg_touch_quality:.1f}%")
                with tc3: st.metric("Avg Volume Strength", f"{avg_volume_strength:.1f}%")
            else:
                st.info("No anchor line touches detected for this day")

        with signal_tabs[2]:
            line_interaction = analyze_anchor_line_interaction(signal_data, ref_line_proj)
            if not line_interaction.empty:
                st.subheader("Price-Anchor Line Interaction")
                st.dataframe(line_interaction, use_container_width=True, hide_index=True)
                touches = line_interaction[line_interaction['Touched'] == 'Yes']
                above_line = line_interaction[line_interaction['Position'] == 'Above']
                ic1, ic2, ic3 = st.columns(3)
                with ic1: st.metric("Touch Points", len(touches))
                with ic2:
                    above_pct = (len(above_line) / len(line_interaction)) * 100 if len(line_interaction) else 0
                    st.metric("Time Above Line", f"{above_pct:.1f}%")
                with ic3:
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
                ec1, ec2, ec3 = st.columns(3)
                with ec1: st.metric("Crossovers", len(crossovers))
                with ec2: st.metric("Current Regime", current_regime)
                with ec3: st.metric("EMA Separation", f"{current_separation:.3f}%")
            else:
                st.info("Insufficient data for EMA analysis")
    else:
        st.info("Configure your parameters and click 'Analyze Market Signals' to begin")

# ═══════════════════════════════════════════════════════════════════════════════
# PART 5 — Contract Tool Tab (uses tab4)
# ═══════════════════════════════════════════════════════════════════════════════
def calculate_contract_volatility(price_data: pd.DataFrame, window: int = 20) -> float:
    if price_data.empty or len(price_data) < window:
        return 0.15
    returns = price_data['Close'].pct_change().dropna()
    if returns.empty:
        return 0.15
    vol = returns.rolling(window=window).std().iloc[-1]
    return vol if not np.isnan(vol) else 0.15

def calculate_price_momentum(p1_price: float, p2_price: float, time_hours: float) -> dict:
    change = p2_price - p1_price
    change_pct = (change / p1_price) * 100 if p1_price > 0 else 0
    hourly = change / time_hours if time_hours > 0 else 0
    abs_pct = abs(change_pct)
    if abs_pct >= 15: strength, confidence = "Very Strong", 95
    elif abs_pct >= 8: strength, confidence = "Strong", 85
    elif abs_pct >= 3: strength, confidence = "Moderate", 70
    else: strength, confidence = "Weak", 50
    return {'change': change, 'change_pct': change_pct, 'hourly_rate': hourly, 'strength': strength, 'confidence': confidence}

def calculate_market_based_targets(entry_price: float, market_data: pd.DataFrame, direction: str) -> dict:
    if market_data.empty:
        base_move = entry_price * 0.02
        return {
            'tp1': entry_price + base_move if direction == "BUY" else entry_price - base_move,
            'tp2': entry_price + (base_move * 2.5) if direction == "BUY" else entry_price - (base_move * 2.5),
            'stop_distance': base_move * 0.6
        }
    atr_series = calculate_average_true_range(market_data, 14)
    current_atr = atr_series.iloc[-1] if not atr_series.empty else entry_price * 0.015
    recent_range = market_data['High'].tail(10).max() - market_data['Low'].tail(10).min()
    volatility_factor = recent_range / market_data['Close'].tail(10).mean()
    base_target = current_atr * 1.2 * (1 + volatility_factor * 0.5)
    extended_target = current_atr * 3.0 * (1 + volatility_factor * 0.3)
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
    overnight_moves, gap_moves = [], []
    for _, daily_data in overnight_data.groupby(overnight_data.index.date):
        if len(daily_data) < 2:
            continue
        day_close = daily_data.iloc[-1]['Close']
        next_open = daily_data.iloc[0]['Open'] if len(daily_data) > 0 else day_close
        overnight_change = (next_open - day_close) / day_close if day_close > 0 else 0
        overnight_moves.append(overnight_change)
        gap_size = abs(overnight_change)
        if gap_size > 0.005:
            gap_moves.append(gap_size)
    if not overnight_moves:
        return {'avg_overnight_change': 0, 'overnight_volatility': 0.02, 'gap_frequency': 0, 'mean_reversion_rate': 0.6}
    avg_overnight_change = float(np.mean(overnight_moves))
    overnight_volatility = float(np.std(overnight_moves))
    gap_frequency = len(gap_moves) / len(overnight_moves) if overnight_moves else 0
    reversion_count = sum(1 for move in overnight_moves if abs(move) < np.std(overnight_moves))
    mean_reversion_rate = reversion_count / len(overnight_moves) if overnight_moves else 0.6
    return {'avg_overnight_change': avg_overnight_change, 'overnight_volatility': overnight_volatility, 'gap_frequency': gap_frequency, 'mean_reversion_rate': mean_reversion_rate}

def project_contract_line(anchor_price: float, anchor_time: datetime, slope: float, target_date: date) -> pd.DataFrame:
    rth = rth_slots_ct(target_date)
    rows = []
    for slot in rth:
        blocks = (slot - anchor_time).total_seconds() / 1800.0
        rows.append({'Time': format_ct_time(slot), 'Contract_Price': round(anchor_price + (slope * blocks), 2), 'Blocks_from_Anchor': round(blocks, 1)})
    return pd.DataFrame(rows)

with tab4:
    st.subheader("Contract Tool")
    st.caption("Overnight contract analysis for RTH entry optimization")

    st.subheader("Overnight Contract Price Points")
    pcol1, pcol2 = st.columns(2)
    with pcol1:
        st.markdown("**Point 1 (Earlier)**")
        p1_date = st.date_input("Point 1 Date", value=datetime.now(CT_TZ).date() - timedelta(days=1), key="ct_p1_date")
        p1_time = st.time_input("Point 1 Time (CT)", value=time(20, 0), key="ct_p1_time")
        p1_price = st.number_input("Point 1 Contract Price", value=10.0, min_value=0.01, step=0.01, format="%.2f", key="ct_p1_price")
    with pcol2:
        st.markdown("**Point 2 (Later)**")
        p2_date = st.date_input("Point 2 Date", value=datetime.now(CT_TZ).date(), key="ct_p2_date")
        p2_time = st.time_input("Point 2 Time (CT)", value=time(8, 0), key="ct_p2_time")
        p2_price = st.number_input("Point 2 Contract Price", value=12.0, min_value=0.01, step=0.01, format="%.2f", key="ct_p2_price")

    projection_day = st.date_input("RTH Projection Day", value=p2_date, key="ct_proj_day")
    p1_datetime = datetime.combine(p1_date, p1_time)
    p2_datetime = datetime.combine(p2_date, p2_time)

    if p2_datetime <= p1_datetime:
        st.error("Point 2 must be after Point 1")
    else:
        time_diff_hours = (p2_datetime - p1_datetime).total_seconds() / 3600
        momentum_metrics = calculate_price_momentum(p1_price, p2_price, time_diff_hours)
        mc1, mc2, mc3, mc4 = st.columns(4)
        with mc1: st.metric("Time Span", f"{time_diff_hours:.1f} hours")
        with mc2: st.metric("Price Change", f"{momentum_metrics['change']:+.2f}")
        with mc3: st.metric("Change %", f"{momentum_metrics['change_pct']:+.1f}%")
        with mc4: st.metric("Momentum", momentum_metrics['strength'])

    st.markdown("---")
    if st.button("Analyze Contract Projections", key="ct_generate", type="primary"):
        if p2_datetime <= p1_datetime:
            st.error("Please ensure Point 2 is after Point 1")
        else:
            with st.spinner("Analyzing contract and market data..."):
                try:
                    blocks_between = (p2_datetime - p1_datetime).total_seconds() / 1800.0
                    contract_slope = (p2_price - p1_price) / blocks_between if blocks_between > 0 else 0
                    underlying_data = fetch_live_data("^GSPC", projection_day - timedelta(days=10), projection_day)
                    overnight_analysis = analyze_overnight_market_behavior("^GSPC", projection_day - timedelta(days=10), projection_day)
                    p1_ct = CT_TZ.localize(p1_datetime)
                    contract_projections = project_contract_line(p1_price, p1_ct, contract_slope, projection_day)

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

    if st.session_state.get('contract_ready', False):
        st.subheader("Contract Analysis Results")
        projections = st.session_state.contract_projections
        config = st.session_state.contract_config
        underlying_data = st.session_state.get('underlying_data', pd.DataFrame())

        contract_tabs = st.tabs(["RTH Projections", "Market Analysis", "Risk Management"])
        with contract_tabs[0]:
            st.subheader("RTH Contract Price Projections")
            if not projections.empty and not underlying_data.empty:
                enhanced = []
                for _, row in projections.iterrows():
                    time_slot = row['Time']
                    price_here = row['Contract_Price']
                    direction = "BUY" if config['momentum']['change'] > 0 else "SELL"
                    targets = calculate_market_based_targets(price_here, underlying_data, direction)
                    hour = int(time_slot.split(':')[0])
                    if hour in [8,9]: time_prob = config['momentum']['confidence'] + 10
                    elif hour in [13,14]: time_prob = config['momentum']['confidence'] + 5
                    else: time_prob = config['momentum']['confidence']
                    enhanced.append({
                        'Time': time_slot,
                        'Contract_Price': round(price_here, 2),
                        'Direction': direction,
                        'TP1': round(targets['tp1'], 2),
                        'TP2': round(targets['tp2'], 2),
                        'Stop_Distance': round(targets['stop_distance'], 2),
                        'Entry_Probability': f"{min(95, time_prob):.0f}%",
                        'ATR_Base': round(targets.get('atr', 0), 2)
                    })
                st.dataframe(pd.DataFrame(enhanced), use_container_width=True, hide_index=True)
            else:
                st.dataframe(projections, use_container_width=True, hide_index=True)

        with contract_tabs[1]:
            st.subheader("Underlying Market Analysis")
            momentum = config['momentum']; overnight = config['overnight_analysis']
            sc1, sc2, sc3 = st.columns(3)
            with sc1: st.metric("Hourly Rate", f"${momentum['hourly_rate']:+.2f}")
            with sc2: st.metric("Strength", momentum['strength'])
            with sc3: st.metric("Confidence", f"{momentum['confidence']}%")
            st.subheader("Overnight Market Behavior")
            oc1, oc2, oc3 = st.columns(3)
            with oc1: st.metric("Avg Overnight Change", f"{overnight['avg_overnight_change']*100:+.2f}%")
            with oc2: st.metric("Overnight Volatility", f"{overnight['overnight_volatility']*100:.2f}%")
            with oc3: st.metric("Gap Frequency", f"{overnight['gap_frequency']*100:.1f}%")
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
                rc1, rc2, rc3 = st.columns(3)
                high_vol_threshold = 0.025
                risk_level = "High" if market_volatility > high_vol_threshold else "Normal"
                position_recommendation = "Reduce Size" if risk_level == "High" else "Standard Size"
                with rc1:
                    st.metric("Risk Level", risk_level)
                    st.caption(f"Volatility: {market_volatility*100:.2f}%")
                with rc2:
                    st.metric("Position Sizing", position_recommendation)
                    st.caption(f"ATR: ${current_atr:.2f}")
                with rc3:
                    avg_contract_price = projections['Contract_Price'].mean()
                    max_risk_per_contract = current_atr * 1.5
                    risk_per_dollar = ((max_risk_per_contract / avg_contract_price) * 100) if avg_contract_price else 0
                    st.metric("Risk per $", f"{risk_per_dollar:.1f}%")
                    st.caption("Based on ATR stop")
                st.subheader("Time-Based Risk Assessment")
                risk_analysis = []
                overnight_vol = config['overnight_analysis']['overnight_volatility']
                for _, row in projections.iterrows():
                    time_slot = row['Time']; price_here = row['Contract_Price']
                    hour = int(time_slot.split(':')[0])
                    if hour in [8,9]: risk_multiplier, risk_rating = 1.5, "High"
                    elif hour in [10,11]: risk_multiplier, risk_rating = 1.0, "Medium"
                    else: risk_multiplier, risk_rating = 0.8, "Low"
                    base_stop = current_atr * 1.2
                    adjusted_stop = base_stop * risk_multiplier * (1 + overnight_vol * 2)
                    risk_analysis.append({
                        'Time': time_slot,
                        'Contract_Price': round(price_here, 2),
                        'Risk_Rating': risk_rating,
                        'Suggested_Stop': round(adjusted_stop, 2),
                        'Risk_Multiplier': f"{risk_multiplier:.1f}x",
                        'Max_Risk_$': round(adjusted_stop, 2)
                    })
                st.dataframe(pd.DataFrame(risk_analysis), use_container_width=True, hide_index=True)
            else:
                st.info("Need underlying market data for comprehensive risk analysis")

    else:
        st.info("Configure your overnight contract points and click 'Analyze Contract Projections'")

# ═══════════════════════════════════════════════════════════════════════════════
# PART 6 — Final Integration & Summary (outside tabs)
# ═══════════════════════════════════════════════════════════════════════════════
def calculate_volume_profile_strength(data: pd.DataFrame, price_level: float) -> float:
    if data.empty or 'Volume' not in data.columns:
        return 50.0
    tolerance = price_level * 0.005
    nearby = data[(data['Low'] <= price_level + tolerance) & (data['High'] >= price_level - tolerance)]
    if nearby.empty:
        return 40.0
    level_volume = nearby['Volume'].sum()
    total_volume = data['Volume'].sum()
    if total_volume == 0:
        return 50.0
    concentration = (level_volume / total_volume) * 100
    if concentration >= 20: return 95.0
    if concentration >= 15: return 85.0
    if concentration >= 10: return 75.0
    if concentration >= 5:  return 60.0
    return 45.0

def detect_market_regime(data: pd.DataFrame) -> dict:
    if data.empty or len(data) < 20:
        return {'regime':'INSUFFICIENT_DATA','trend':'NEUTRAL','strength':0,'volatility':1.5}
    closes = data['Close'].tail(20)
    price_change = (closes.iloc[-1] - closes.iloc[0]) / closes.iloc[0] * 100
    returns = closes.pct_change().dropna()
    volatility = returns.std() * np.sqrt(390) * 100
    if price_change > 1.0: trend, strength = 'BULLISH', min(100, abs(price_change)*10)
    elif price_change < -1.0: trend, strength = 'BEARISH', min(100, abs(price_change)*10)
    else: trend, strength = 'NEUTRAL', abs(price_change)*10
    if volatility >= 3.0: regime = 'HIGH_VOLATILITY'
    elif volatility >= 1.8: regime = 'MODERATE_VOLATILITY'
    else: regime = 'LOW_VOLATILITY'
    return {'regime':regime,'trend':trend,'strength':strength,'volatility':volatility,'price_change':price_change}

def calculate_support_resistance_strength(data: pd.DataFrame, price_level: float) -> float:
    if data.empty or len(data) < 10:
        return 50.0
    highs = data['High']; lows = data['Low']
    tolerance = price_level * 0.008
    high_touches = sum(1 for v in highs if abs(v - price_level) <= tolerance)
    low_touches = sum(1 for v in lows if abs(v - price_level) <= tolerance)
    total = high_touches + low_touches
    if total >= 4: return 90.0
    if total >= 3: return 80.0
    if total >= 2: return 70.0
    if total >= 1: return 60.0
    return 45.0

def calculate_confluence_score(price: float, anchor_price: float, market_data: pd.DataFrame) -> float:
    if market_data.empty:
        return 50.0
    price_distance = abs(price - anchor_price) / anchor_price * 100 if anchor_price else 0
    proximity_score = max(0, 100 - (price_distance * 15))
    volume_score = calculate_volume_profile_strength(market_data, price)
    regime_info = detect_market_regime(market_data)
    regime_score = 80 if regime_info['trend'] != 'NEUTRAL' else 55
    sr_score = calculate_support_resistance_strength(market_data, price)
    confluence = (proximity_score*0.25 + volume_score*0.25 + regime_score*0.25 + sr_score*0.25)
    return float(min(100, max(0, confluence)))

def calculate_time_edge_from_data(symbol: str, lookback_days: int = 30) -> dict:
    try:
        end_date = datetime.now(CT_TZ).date()
        start_date = end_date - timedelta(days=lookback_days)
        historical = fetch_live_data(symbol, start_date, end_date)
        if historical.empty:
            return {}
        time_performance = {}
        for time_slot in ['08:30','09:00','09:30','10:00','10:30','11:00',
                          '11:30','12:00','12:30','13:00','13:30','14:00','14:30']:
            slot_data = historical.between_time(time_slot, time_slot)
            if slot_data.empty:
                continue
            slot_returns = slot_data['Close'].pct_change().dropna()
            if len(slot_returns) > 5:
                volatility = slot_returns.std() * 100
                avg_move = abs(slot_returns).mean() * 100
                upward_bias = (slot_returns > 0).mean() * 100
                if volatility > 0.8 and abs(upward_bias - 50) > 10:
                    edge_score = min(20, volatility * 15 + abs(upward_bias - 50))
                else:
                    edge_score = volatility * 10
                time_performance[time_slot] = {
                    'volatility': volatility,
                    'avg_move': avg_move,
                    'upward_bias': upward_bias,
                    'edge_score': edge_score
                }
        return time_performance
    except Exception:
        return {}

def get_market_hours_status() -> dict:
    current_time_ct = datetime.now(CT_TZ)
    is_weekday = current_time_ct.weekday() < 5
    market_open = current_time_ct.replace(hour=8, minute=30, second=0, microsecond=0)
    market_close = current_time_ct.replace(hour=14, minute=30, second=0, microsecond=0)
    premarket_start = current_time_ct.replace(hour=7, minute=0, second=0, microsecond=0)
    afterhours_end = current_time_ct.replace(hour=17, minute=0, second=0, microsecond=0)
    if not is_weekday:
        status, session = "WEEKEND", "Closed"
    elif market_open <= current_time_ct <= market_close:
        status, session = "RTH_OPEN", "Regular Hours"
    elif premarket_start <= current_time_ct < market_open:
        status, session = "PREMARKET", "Pre-Market"
    elif market_close < current_time_ct <= afterhours_end:
        status, session = "AFTERHOURS", "After Hours"
    else:
        status, session = "CLOSED", "Closed"
    if status == "CLOSED" and is_weekday:
        next_open = market_open + timedelta(days=1)
    elif status == "WEEKEND":
        days_until_monday = 7 - current_time_ct.weekday()
        next_open = current_time_ct.replace(hour=8, minute=30, second=0, microsecond=0) + timedelta(days=days_until_monday)
    else:
        next_open = None
    return {'status':status,'session':session,'current_time':current_time_ct,'is_trading_day':is_weekday,'next_open':next_open}

# Flags
for key in ['spx_analysis_ready','stock_analysis_ready','signal_ready','contract_ready']:
    if key not in st.session_state:
        st.session_state[key] = False

st.markdown("---")
st.subheader("Analysis Summary Dashboard")
market_status = get_market_hours_status()
sum1, sum2, sum3 = st.columns(3)
with sum1:
    st.markdown("**Analysis Status**")
    st.write(f"SPX Anchors: {'Ready' if st.session_state.spx_analysis_ready else 'Pending'}")
    st.write(f"Stock Analysis: {'Ready' if st.session_state.stock_analysis_ready else 'Pending'}")
    st.write(f"Signal Detection: {'Ready' if st.session_state.signal_ready else 'Pending'}")
    st.write(f"Contract Tool: {'Ready' if st.session_state.contract_ready else 'Pending'}")
with sum2:
    st.markdown("**Current Settings**")
    st.write(f"Skyline Slope: {st.session_state.spx_slopes['skyline']:+.3f}")
    st.write(f"Baseline Slope: {st.session_state.spx_slopes['baseline']:+.3f}")
    st.write(f"ES→SPX Offset: {st.session_state.current_offset:+.1f}")
    st.write(f"High/Close/Low: {st.session_state.spx_slopes['high']:+.3f}")
with sum3:
    st.markdown("**Market Status**")
    st.write(f"Market: {market_status['session']}")
    st.write(f"Time (CT): {market_status['current_time'].strftime('%H:%M:%S')}")
    if market_status['next_open']:
        time_to_open = market_status['next_open'] - market_status['current_time']
        hours_to_open = int(time_to_open.total_seconds() // 3600)
        st.write(f"Next Open: {hours_to_open}h")
    else:
        st.write("Session Active")

st.markdown("---")
st.subheader("Quick Actions")
qa1, qa2, qa3, qa4 = st.columns(4)
with qa1:
    if st.button("Update ES Offset", key="quick_update_offset"):
        with st.spinner("Updating offset from market data..."):
            today = datetime.now(CT_TZ).date()
            yesterday = today - timedelta(days=1)
            es_data = fetch_live_data("ES=F", yesterday, today)
            spx_data = fetch_live_data("^GSPC", yesterday, today)
            if not es_data.empty and not spx_data.empty:
                new_offset = calculate_es_spx_offset(es_data, spx_data)
                st.session_state.current_offset = new_offset
                st.success(f"Offset updated: {new_offset:+.1f}")
                st.experimental_rerun()
            else:
                st.error("Failed to fetch offset data")
with qa2:
    if st.button("Reset All Analysis", key="quick_reset_all"):
        for key in ['spx_analysis_ready','stock_analysis_ready','signal_ready','contract_ready',
                    'es_anchor_data','spx_manual_anchors','stock_analysis_data','signal_data','contract_projections']:
            if key in st.session_state:
                del st.session_state[key]
        st.success("All analysis reset")
        st.experimental_rerun()
with qa3:
    if st.button("Reset Slopes", key="quick_reset_slopes"):
        st.session_state.spx_slopes = SPX_SLOPES.copy()
        st.session_state.stock_slopes = STOCK_SLOPES.copy()
        st.success("Slopes reset to defaults")
        st.experimental_rerun()
with qa4:
    if st.button("Test Connection", key="quick_test"):
        with st.spinner("Testing market connection..."):
            test_data = fetch_live_data("^GSPC", datetime.now().date() - timedelta(days=1), datetime.now().date())
            st.success("Connection successful" if not test_data.empty else "Connection failed")

if any([st.session_state.get('spx_analysis_ready', False),
        st.session_state.get('stock_analysis_ready', False),
        st.session_state.get('signal_ready', False),
        st.session_state.get('contract_ready', False)]):

    st.markdown("---")
    st.subheader("Market Performance Insights")
    insight_tabs = st.tabs(["Market Regime", "Time-of-Day Edge", "Volume Analysis"])

    with insight_tabs[0]:
        if st.session_state.get('signal_ready', False):
            signal_data = st.session_state.get('signal_data', pd.DataFrame())
            if not signal_data.empty:
                regime = detect_market_regime(signal_data)
                rc1, rc2 = st.columns(2)
                with rc1:
                    st.metric("Market Trend", regime['trend'])
                    st.metric("Trend Strength", f"{regime['strength']:.1f}")
                with rc2:
                    st.metric("Volatility Regime", regime['regime'])
                    st.metric("Volatility Level", f"{regime['volatility']:.1f}%")
                if regime['trend'] == 'BULLISH' and regime['volatility'] < 2.5:
                    context = "Stable uptrend - favorable for long positions"
                elif regime['trend'] == 'BEARISH' and regime['volatility'] < 2.5:
                    context = "Stable downtrend - favorable for short positions"
                elif regime['volatility'] > 3.0:
                    context = "High volatility environment - use wider stops"
                else:
                    context = "Neutral/ranging market - focus on mean reversion"
                st.info(context)
        else:
            st.info("Generate signal analysis to see market regime data")

    with insight_tabs[1]:
        if st.button("Calculate Time Edge", key="calc_time_edge"):
            with st.spinner("Analyzing time-of-day patterns..."):
                time_edge_data = calculate_time_edge_from_data("^GSPC", 30)
                if time_edge_data:
                    edge_rows = [{
                        'Time': ts,
                        'Volatility': f"{m['volatility']:.2f}%",
                        'Avg Move': f"{m['avg_move']:.2f}%",
                        'Upward Bias': f"{m['upward_bias']:.1f}%",
                        'Edge Score': f"{m['edge_score']:.1f}"
                    } for ts, m in time_edge_data.items()]
                    st.dataframe(pd.DataFrame(edge_rows), use_container_width=True, hide_index=True)
                else:
                    st.error("Could not calculate time edge data")
        else:
            st.info("Click 'Calculate Time Edge' to analyze historical time-of-day patterns")

    with insight_tabs[2]:
        if st.session_state.get('signal_ready', False):
            signal_data = st.session_state.get('signal_data', pd.DataFrame())
            if not signal_data.empty and 'Volume' in signal_data.columns:
                avg_volume = signal_data['Volume'].mean()
                volume_std = signal_data['Volume'].std()
                high_thr = avg_volume + volume_std
                high_vol_bars = signal_data[signal_data['Volume'] > high_thr]
                vc1, vc2, vc3 = st.columns(3)
                with vc1: st.metric("Average Volume", f"{avg_volume:,.0f}")
                with vc2: st.metric("Volume Std Dev", f"{volume_std:,.0f}")
                with vc3: st.metric("High Volume Bars", len(high_vol_bars))
                if not high_vol_bars.empty:
                    high_vol_avg_move = abs(high_vol_bars['Close'] - high_vol_bars['Open']).mean()
                    normal = signal_data[signal_data['Volume'] <= high_thr]
                    normal_avg_move = abs(normal['Close'] - normal['Open']).mean() if not normal.empty else 0
                    st.info(f"High volume bars average move: ${high_vol_avg_move:.2f}")
                    st.info(f"Normal volume bars average move: ${normal_avg_move:.2f}")
            else:
                st.info("No volume data available for analysis")

st.markdown("---")
data_status_col1, data_status_col2, data_status_col3 = st.columns(3)
with data_status_col1:
    try:
        spx_test = fetch_live_data("^GSPC", datetime.now().date() - timedelta(days=1), datetime.now().date())
        spx_status = "Active" if not spx_test.empty else "Issue"
        spx_last_update = spx_test.index[-1].strftime("%H:%M CT") if not spx_test.empty else "N/A"
    except:
        spx_status = "Error"; spx_last_update = "N/A"
    st.write("**SPX Data**")
    st.write(f"Status: {spx_status}")
    st.write(f"Last Update: {spx_last_update}")

with data_status_col2:
    try:
        es_test = fetch_live_data("ES=F", datetime.now().date() - timedelta(days=1), datetime.now().date())
        es_status = "Active" if not es_test.empty else "Issue"
        es_last_update = es_test.index[-1].strftime("%H:%M CT") if not es_test.empty else "N/A"
    except:
        es_status = "Error"; es_last_update = "N/A"
    st.write("**ES Futures**")
    st.write(f"Status: {es_status}")
    st.write(f"Last Update: {es_last_update}")

with data_status_col3:
    current_offset_age = "Live" if st.session_state.current_offset != 0 else "Default"
    st.write("**Current Session**")
    st.write(f"Offset Status: {current_offset_age}")
    st.write(f"Session: {market_status['session']}")

st.markdown("---")
st.markdown(
    f"""
    <div style='text-align: center; color: #888; font-size: 0.9rem;'>
        SPX Prophet Analytics • Market Data Integration • 
        Session: {datetime.now(CT_TZ).strftime('%H:%M:%S CT')} • 
        Status: {market_status['session']}
    </div>
    """,
    unsafe_allow_html=True
)
