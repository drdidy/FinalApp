# ═══════════════════════════════════════════════════════════════════════════════
# 🔮 SPX PROPHET — Unified App (Parts 1–6, fixed plumbing; strategy unchanged)
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

# ───────────────────────────────────────────────────────────────────────────────
# 🌎 CORE CONFIGURATION
# ───────────────────────────────────────────────────────────────────────────────
CT_TZ = pytz.timezone('America/Chicago')
UTC_TZ = pytz.UTC

# Trading parameters
RTH_START = "08:30"  # RTH start in CT
RTH_END   = "14:30"  # RTH end in CT
SPX_ANCHOR_START = "17:00"
SPX_ANCHOR_END   = "19:30"

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

# ───────────────────────────────────────────────────────────────────────────────
# 🎨 STREAMLIT CONFIGURATION & STYLING
# ───────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="🔮 SPX Prophet Analytics", page_icon="📈", layout="wide", initial_sidebar_state="expanded")
st.markdown("""
<style>
    .main { background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%); color: white; }
    .metric-container { background: linear-gradient(135deg, rgba(255,255,255,0.1), rgba(255,255,255,0.05));
        padding: 1.2rem; border-radius: 15px; border: 2px solid rgba(255,255,255,0.2); backdrop-filter: blur(10px);
        box-shadow: 0 8px 32px rgba(0,0,0,0.1); transition: all 0.3s ease; }
    .metric-container:hover { transform: translateY(-5px); box-shadow: 0 15px 35px rgba(0,0,0,0.2); }
    .stDataFrame { background: rgba(255,255,255,0.95); border-radius: 10px; overflow: hidden; }
</style>
""", unsafe_allow_html=True)

# ───────────────────────────────────────────────────────────────────────────────
# 📊 DATA FETCHING (ROBUST intraday; keeps your signature)
# ───────────────────────────────────────────────────────────────────────────────

def _infer_period_interval(days: int) -> Tuple[str, str]:
    if days <= 7: return "7d", "30m"
    if days <= 30: return "30d", "30m"
    return "60d", "30m"

def _to_ct_index(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty: return df
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    return df.tz_convert(CT_TZ)

@st.cache_data(ttl=60)
def fetch_live_data(symbol: str, start_date: date, end_date: date) -> pd.DataFrame:
    """Robust yfinance intraday using period/interval; SPY fallback for ^GSPC."""
    try:
        days = (end_date - start_date).days + 1
        period, interval = _infer_period_interval(days)

        def _dl(sym: str) -> pd.DataFrame:
            df = yf.download(
                tickers=sym, period=period, interval=interval,
                auto_adjust=False, back_adjust=False,
                threads=False, progress=False,
            )
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [c[0] for c in df.columns]
            return df

        df = _dl(symbol)
        if df.empty:  # retry once
            df = _dl(symbol)

        if df.empty and symbol.upper() == "^GSPC":
            df = _dl("SPY")  # fallback

        if df.empty:
            return pd.DataFrame()

        df = _to_ct_index(df)

        start_dt = CT_TZ.localize(datetime.combine(start_date, time(0, 0)))
        end_dt   = CT_TZ.localize(datetime.combine(end_date, time(23, 59)))
        df = df.loc[(df.index >= start_dt) & (df.index <= end_dt)]
        return df

    except Exception:
        return pd.DataFrame()

def validate_ohlc_data(df: pd.DataFrame) -> bool:
    if df.empty: return False
    if not all(c in df.columns for c in ['Open','High','Low','Close']): return False
    bad = (
        (df['High'] < df['Low']) |
        (df['High'] < df['Open']) | (df['High'] < df['Close']) |
        (df['Low']  > df['Open']) | (df['Low']  > df['Close']) |
        (df['Close'] <= 0) | (df['High'] <= 0)
    )
    return not bad.any()

# ───────────────────────────────────────────────────────────────────────────────
# ⏰ TIME HANDLING
# ───────────────────────────────────────────────────────────────────────────────

def rth_slots_ct(target_date: date) -> List[datetime]:
    start_ct = CT_TZ.localize(datetime.combine(target_date, time(8,30)))
    end_ct   = CT_TZ.localize(datetime.combine(target_date, time(14,30)))
    slots, current = [], start_ct
    while current <= end_ct:
        slots.append(current)
        current += timedelta(minutes=30)
    return slots

def format_ct_time(dt: datetime) -> str:
    if dt.tzinfo is None:
        dt = CT_TZ.localize(dt)
    else:
        dt = dt.astimezone(CT_TZ)
    return dt.strftime("%H:%M")

def get_session_window(df: pd.DataFrame, start_time: str, end_time: str) -> pd.DataFrame:
    if df.empty: return df
    return df.between_time(start_time, end_time)

def get_daily_ohlc(df: pd.DataFrame, target_date: date) -> Dict:
    if df.empty: return {}
    start_dt = CT_TZ.localize(datetime.combine(target_date, time(0,0)))
    end_dt   = CT_TZ.localize(datetime.combine(target_date, time(23,59)))
    day_data = df.loc[start_dt:end_dt]
    if day_data.empty: return {}
    day_open = day_data.iloc[0]['Open']
    day_high = day_data['High'].max()
    day_low  = day_data['Low'].min()
    day_close= day_data.iloc[-1]['Close']
    high_time = day_data[day_data['High']==day_high].index[0]
    low_time  = day_data[day_data['Low']==day_low].index[0]
    return {'open': (day_open, day_data.index[0]),
            'high': (day_high, high_time),
            'low':  (day_low,  low_time),
            'close':(day_close,day_data.index[-1])}

# ───────────────────────────────────────────────────────────────────────────────
# 📈 SWINGS & PROJECTIONS
# ───────────────────────────────────────────────────────────────────────────────

def detect_swings_simple(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or len(df) < 2: return df.copy()
    out = df.copy()
    out['swing_high'] = False
    out['swing_low']  = False
    if 'Close' in out.columns:
        out.loc[out['Close'].idxmax(), 'swing_high'] = True
        out.loc[out['Close'].idxmin(), 'swing_low']  = True
    return out

# Compatibility wrapper for Part 3 (strategy unchanged)
SWING_K = 2
def detect_swings(df: pd.DataFrame, k: Optional[int] = None) -> pd.DataFrame:
    return detect_swings_simple(df)

def get_anchor_points(df_swings: pd.DataFrame) -> Tuple[Optional[Tuple], Optional[Tuple]]:
    skyline = baseline = None
    if df_swings.empty or 'Close' not in df_swings.columns: return None, None
    highs = df_swings[df_swings.get('swing_high', False) == True]
    lows  = df_swings[df_swings.get('swing_low',  False) == True]
    if not highs.empty:
        bh = highs.loc[highs['Close'].idxmax()]
        skyline = (bh['Close'], bh.name)
    if not lows.empty:
        bl = lows.loc[lows['Close'].idxmin()]
        baseline = (bl['Close'], bl.name)
    return skyline, baseline

def project_anchor_line(anchor_price: float, anchor_time: datetime, slope: float, target_date: date) -> pd.DataFrame:
    rth_slots = rth_slots_ct(target_date)
    rows = []
    for slot_time in rth_slots:
        blocks = (slot_time - anchor_time).total_seconds() / 1800.0
        rows.append({
            'Time': format_ct_time(slot_time),
            'Price': round(anchor_price + slope * blocks, 2),
            'Blocks': round(blocks, 1),
            'Anchor_Price': round(anchor_price, 2),
            'Slope': slope
        })
    return pd.DataFrame(rows)

# ───────────────────────────────────────────────────────────────────────────────
# 📊 INDICATORS
# ───────────────────────────────────────────────────────────────────────────────

def calculate_ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span).mean()

def calculate_vwap(df: pd.DataFrame) -> pd.Series:
    if df.empty or 'Volume' not in df.columns:
        return pd.Series(index=df.index, dtype=float)
    typical = (df['High'] + df['Low'] + df['Close']) / 3.0
    cum_vol = df['Volume'].cumsum()
    vwap = (typical * df['Volume']).cumsum() / cum_vol
    return vwap.fillna(method='ffill').fillna(typical)

def calculate_es_spx_offset(es_data: pd.DataFrame, spx_data: pd.DataFrame) -> float:
    try:
        if es_data.empty or spx_data.empty: return 0.0
        es_rth  = get_session_window(es_data,  "08:30", "15:00")
        spx_rth = get_session_window(spx_data, "08:30", "15:00")
        if es_rth.empty or spx_rth.empty:
            es_close  = es_data.iloc[-1]['Close']
            spx_close = spx_data.iloc[-1]['Close']
        else:
            es_close  = es_rth.iloc[-1]['Close']
            spx_close = spx_rth.iloc[-1]['Close']
        return round(spx_close - es_close, 1)
    except Exception:
        return 0.0

# ───────────────────────────────────────────────────────────────────────────────
# 🔧 SESSION STATE INIT
# ───────────────────────────────────────────────────────────────────────────────
for key, default in [
    ('theme', 'Dark'),
    ('spx_slopes', SPX_SLOPES.copy()),
    ('stock_slopes', STOCK_SLOPES.copy()),
    ('current_offset', 0.0),
    ('spx_analysis_ready', False),
    ('stock_analysis_ready', False),
    ('signal_ready', False),
    ('contract_ready', False),
]:
    if key not in st.session_state: st.session_state[key] = default

# ───────────────────────────────────────────────────────────────────────────────
# 🎛️ SIDEBAR CONTROLS (unchanged)
# ───────────────────────────────────────────────────────────────────────────────
st.sidebar.title("🔮 SPX Prophet Analytics")
st.sidebar.markdown("---")
st.session_state.theme = st.sidebar.radio("🎨 Theme", ["Dark", "Light"], key="ui_theme")
st.sidebar.markdown("---")
st.sidebar.markdown("### SPX Slopes (per 30-min)")
with st.sidebar.expander("SPX Slope Settings", expanded=False):
    for slope_name, default_value in SPX_SLOPES.items():
        icon_map = {'high':'High','close':'Close','low':'Low','skyline':'Skyline','baseline':'Baseline'}
        display_name = icon_map.get(slope_name, slope_name.title())
        st.session_state.spx_slopes[slope_name] = st.number_input(
            display_name, value=st.session_state.spx_slopes[slope_name],
            step=0.0001, format="%.4f", key=f"sb_spx_{slope_name}"
        )
st.sidebar.markdown("---")
st.sidebar.markdown("### 🏢 Stock Slopes (magnitude)")
with st.sidebar.expander("🔧 Stock Slope Settings", expanded=False):
    for ticker, default_slope in STOCK_SLOPES.items():
        st.session_state.stock_slopes[ticker] = st.number_input(
            f"📈 {ticker}", value=st.session_state.stock_slopes.get(ticker, default_slope),
            step=0.0001, format="%.4f", key=f"sb_stk_{ticker}"
        )

# ───────────────────────────────────────────────────────────────────────────────
# 🏠 MAIN HEADER + METRICS
# ───────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center; padding:2rem; background: linear-gradient(135deg, rgba(255,255,255,0.1), rgba(255,255,255,0.05)); border-radius:20px; margin:1rem 0; border:2px solid rgba(255,255,255,0.2);">
  <h1 style="font-size:3rem; margin:0; background:linear-gradient(135deg,#ff6b6b,#4ecdc4,#45b7d1,#f9ca24); -webkit-background-clip:text; -webkit-text-fill-color:transparent;">SPX Prophet Analytics</h1>
  <p style="font-size:1.3rem; margin:1rem 0; opacity:0.9;">Advanced Trading Analytics with Live Market Data Integration</p>
</div>
""", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)
with col1:
    now = datetime.now(CT_TZ)
    st.markdown(f"""
    <div class="metric-container"><h3>⏰ Current Time (CT)</h3>
    <h2>{now.strftime("%H:%M:%S")}</h2><p>{now.strftime("%A, %B %d")}</p></div>
    """, unsafe_allow_html=True)
with col2:
    is_weekday = now.weekday() < 5
    market_open = now.replace(hour=8, minute=30, second=0, microsecond=0)
    market_close= now.replace(hour=14, minute=30, second=0, microsecond=0)
    within = is_weekday and (market_open <= now <= market_close)
    color, text = ("#00ff88","MARKET OPEN") if within else (("#ffbb33","MARKET CLOSED") if is_weekday else ("#ff6b6b","WEEKEND"))
    st.markdown(f"""
    <div class="metric-container"><h3>Market Status</h3>
    <h2 style="color:{color};">{text}</h2><p>RTH: 08:30 - 14:30 CT • Mon-Fri</p></div>
    """, unsafe_allow_html=True)
with col3:
    st.markdown(f"""
    <div class="metric-container"><h3>🔄 ES→SPX Offset</h3>
    <h2>{st.session_state.current_offset:+.1f}</h2><p>Live market differential</p></div>
    """, unsafe_allow_html=True)

st.markdown("---")

# ───────────────────────────────────────────────────────────────────────────────
# Connection test (more robust)
# ───────────────────────────────────────────────────────────────────────────────
if st.button("Test Data Connection", key="test_connection"):
    with st.spinner("Testing market data connection..."):
        # Try ES=F first
        es_test = fetch_live_data("ES=F", datetime.now().date()-timedelta(days=3), datetime.now().date())
        spx_test = fetch_live_data("^GSPC", datetime.now().date()-timedelta(days=3), datetime.now().date())
        if not es_test.empty or not spx_test.empty:
            st.success("Market data connection successful!")
            st.info(f"ES bars: {len(es_test)} • SPX/ETF bars: {len(spx_test)}")
        else:
            st.error("Market data connection failed (Yahoo may be blocked or rate-limited).")

st.markdown("---")

# ═══════════════════════════════════════════════════════════════════════════════
# PART 2 — SPX ANCHORS TAB
# ═══════════════════════════════════════════════════════════════════════════════

def calculate_anchor_entry_probability(anchor_type: str, time_slot: str) -> float:
    base_probs = {'SKYLINE':90.0,'BASELINE':90.0,'HIGH':75.0,'CLOSE':80.0,'LOW':75.0}
    base = base_probs.get(anchor_type.upper(), 70.0)
    hour = int(time_slot.split(':')[0])
    time_adj = 8 if hour in [8,9] else (5 if hour in [13,14] else 0)
    return min(95, base + time_adj)

def calculate_anchor_target_probability(anchor_type: str, target_num: int) -> float:
    if anchor_type.upper() in ['SKYLINE','BASELINE']:
        return 85.0 if target_num==1 else 68.0
    return 75.0 if target_num==1 else 55.0

def calculate_entry_exit_table(projection_df: pd.DataFrame, anchor_type: str) -> pd.DataFrame:
    if projection_df.empty: return pd.DataFrame()
    rows = []
    is_sky = anchor_type.upper() in ['SKYLINE','HIGH']
    is_base= anchor_type.upper() in ['BASELINE','LOW']
    for _, row in projection_df.iterrows():
        t = row['Time']; anchor_price = row['Price']
        if anchor_type.upper() == 'HIGH':
            vol = anchor_price*0.010; tp1d=vol*0.7; tp2d=vol*1.8
            entry=anchor_price; tp1=anchor_price - tp1d; tp2=anchor_price - tp2d; dirn="SELL"; stop=anchor_price + (anchor_price*0.005)
        elif anchor_type.upper() in ['LOW','CLOSE']:
            vol = anchor_price*0.010; tp1d=vol*0.7; tp2d=vol*1.8
            entry=anchor_price; tp1=anchor_price + tp1d; tp2=anchor_price + tp2d; dirn="BUY";  stop=anchor_price - (anchor_price*0.005)
        elif is_sky:
            vol = anchor_price*0.012; tp1d=vol*0.8; tp2d=vol*2.2
            entry=anchor_price; tp1=anchor_price + tp1d; tp2=anchor_price + tp2d; dirn="BUY";  stop=anchor_price + (anchor_price*0.006)
        elif is_base:
            vol = anchor_price*0.012; tp1d=vol*0.8; tp2d=vol*2.2
            entry=anchor_price; tp1=anchor_price + tp1d; tp2=anchor_price + tp2d; dirn="BUY";  stop=max(0.01, anchor_price - (anchor_price*0.006))
        risk = abs(entry - stop)
        rr1 = abs(tp1 - entry)/risk if risk>0 else 0
        rr2 = abs(tp2 - entry)/risk if risk>0 else 0
        rows.append({
            'Time': t, 'Direction': dirn, 'Entry': round(entry,2), 'Stop': round(stop,2),
            'TP1': round(tp1,2), 'TP2': round(tp2,2), 'Risk': round(risk,2),
            'RR1': f"{rr1:.1f}", 'RR2': f"{rr2:.1f}",
            'Entry_Prob': f"{calculate_anchor_entry_probability(anchor_type,t):.0f}%",
            'TP1_Prob': f"{calculate_anchor_target_probability(anchor_type,1):.0f}%",
            'TP2_Prob': f"{calculate_anchor_target_probability(anchor_type,2):.0f}%"
        })
    return pd.DataFrame(rows)

def update_offset_for_date():
    if 'spx_prev_day' in st.session_state:
        d = st.session_state.spx_prev_day
        es = fetch_live_data("ES=F", d, d)
        spx= fetch_live_data("^GSPC", d, d)
        if not es.empty and not spx.empty:
            st.session_state.current_offset = calculate_es_spx_offset(es, spx)

tab1, tab2, tab3, tab4 = st.tabs(["SPX Anchors", "Stock Anchors", "Signals & EMA", "Contract Tool"])

with tab1:
    st.subheader("SPX Anchor Analysis")
    st.caption("Live ES futures data for anchor detection and SPX projections")

    c1, c2 = st.columns(2)
    with c1:
        prev_day = st.date_input(
            "Previous Trading Day", value=datetime.now(CT_TZ).date()-timedelta(days=1),
            key="spx_prev_day", on_change=update_offset_for_date
        )
        st.caption(f"Selected: {prev_day.strftime('%A')}")
    with c2:
        proj_day = st.date_input("Projection Day", value=prev_day+timedelta(days=1), key="spx_proj_day")
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
                es_off = fetch_live_data("ES=F", prev_day, prev_day)
                spx_off = fetch_live_data("^GSPC", prev_day, prev_day)
                if not es_off.empty and not spx_off.empty:
                    st.session_state.current_offset = calculate_es_spx_offset(es_off, spx_off)

                es_data = fetch_live_data("ES=F", prev_day, prev_day)
                if es_data.empty:
                    st.error(f"No ES futures data for {prev_day}")
                else:
                    anchor_window = get_session_window(es_data, SPX_ANCHOR_START, SPX_ANCHOR_END)
                    if anchor_window.empty: anchor_window = es_data
                    st.session_state.es_anchor_data = anchor_window

                    spx_data = fetch_live_data("^GSPC", prev_day, prev_day)
                    if not spx_data.empty:
                        daily_ohlc = get_daily_ohlc(spx_data, prev_day)
                        if daily_ohlc: st.session_state.spx_manual_anchors = daily_ohlc
                        else: st.warning("Could not extract SPX OHLC data")
                    else:
                        es_daily = get_daily_ohlc(anchor_window, prev_day)
                        if es_daily:
                            spx_equiv={}
                            for k,(p,ts) in es_daily.items():
                                spx_equiv[k]=(p+st.session_state.current_offset, ts)
                            st.session_state.spx_manual_anchors = spx_equiv
                    st.session_state.spx_analysis_ready = True
            except Exception as e:
                st.error(f"Analysis error: {str(e)}")

    # Results
    if st.session_state.get('spx_analysis_ready', False):
        st.subheader("SPX Anchor Projections")
        es_data = st.session_state.get('es_anchor_data', pd.DataFrame())
        skyline_anchor_spx = baseline_anchor_spx = None
        if not es_data.empty:
            es_swings = detect_swings_simple(es_data)
            es_sky, es_base = get_anchor_points(es_swings)
            offset = st.session_state.current_offset
            if es_sky:  skyline_anchor_spx = (es_sky[0]+offset,  es_sky[1])
            if es_base: baseline_anchor_spx = (es_base[0]+offset, es_base[1])

        if st.session_state.get('spx_manual_anchors'):
            manual = st.session_state.spx_manual_anchors
            st.subheader("Detected SPX Anchors")
            scols = st.columns(5)
            info = [('high','High','#ff6b6b'),('close','Close','#f9ca24'),('low','Low','#4ecdc4')]
            for i,(key,label,color) in enumerate(info):
                if key in manual:
                    price, ts = manual[key]
                    with scols[i]:
                        st.markdown(f"""
                        <div style="text-align:center; padding:1rem; background:rgba(255,255,255,0.1);
                        border-radius:10px; border-left:4px solid {color};">
                        <h4>{label}</h4><h3>${price:.2f}</h3><p>{format_ct_time(ts)}</p></div>
                        """, unsafe_allow_html=True)
            with scols[3]:
                if skyline_anchor_spx:
                    price, ts = skyline_anchor_spx
                    st.markdown(f"""
                    <div style="text-align:center; padding:1rem; background:rgba(255,100,100,0.2);
                    border-radius:10px; border-left:4px solid #ff4757;">
                    <h4>Skyline</h4><h3>${price:.2f}</h3><p>{format_ct_time(ts)}</p><small>SPX Equivalent</small></div>
                    """, unsafe_allow_html=True)
                else:
                    st.warning("No Skyline")
            with scols[4]:
                if baseline_anchor_spx:
                    price, ts = baseline_anchor_spx
                    st.markdown(f"""
                    <div style="text-align:center; padding:1rem; background:rgba(100,100,255,0.2);
                    border-radius:10px; border-left:4px solid #3742fa;">
                    <h4>Baseline</h4><h3>${price:.2f}</h3><p>{format_ct_time(ts)}</p><small>SPX Equivalent</small></div>
                    """, unsafe_allow_html=True)
                else:
                    st.warning("No Baseline")

        st.markdown("---")
        ptabs = st.tabs(["High","Close","Low","Skyline","Baseline"])

        if st.session_state.get('spx_manual_anchors'):
            manual = st.session_state.spx_manual_anchors
            # High
            with ptabs[0]:
                if 'high' in manual:
                    spx_price, ts = manual['high']; ts_ct = ts.astimezone(CT_TZ)
                    high_proj = project_anchor_line(spx_price, ts_ct, st.session_state.spx_slopes['high'], proj_day)
                    if not high_proj.empty:
                        st.subheader("High Anchor SPX Projection")
                        st.dataframe(high_proj, use_container_width=True, hide_index=True)
                        high_analysis = calculate_entry_exit_table(high_proj, "HIGH")
                        if not high_analysis.empty:
                            st.subheader("Entry/Exit Strategy")
                            st.dataframe(high_analysis, use_container_width=True, hide_index=True)
                else: st.warning("No high anchor data available")
            # Close
            with ptabs[1]:
                if 'close' in manual:
                    spx_price, ts = manual['close']; ts_ct = ts.astimezone(CT_TZ)
                    close_proj = project_anchor_line(spx_price, ts_ct, st.session_state.spx_slopes['close'], proj_day)
                    if not close_proj.empty:
                        st.subheader("Close Anchor SPX Projection")
                        st.dataframe(close_proj, use_container_width=True, hide_index=True)
                        close_analysis = calculate_entry_exit_table(close_proj, "CLOSE")
                        if not close_analysis.empty:
                            st.subheader("Entry/Exit Strategy")
                            st.dataframe(close_analysis, use_container_width=True, hide_index=True)
                else: st.warning("No close anchor data available")
            # Low
            with ptabs[2]:
                if 'low' in manual:
                    spx_price, ts = manual['low']; ts_ct = ts.astimezone(CT_TZ)
                    low_proj = project_anchor_line(spx_price, ts_ct, st.session_state.spx_slopes['low'], proj_day)
                    if not low_proj.empty:
                        st.subheader("Low Anchor SPX Projection")
                        st.dataframe(low_proj, use_container_width=True, hide_index=True)
                        low_analysis = calculate_entry_exit_table(low_proj, "LOW")
                        if not low_analysis.empty:
                            st.subheader("Entry/Exit Strategy")
                            st.dataframe(low_analysis, use_container_width=True, hide_index=True)
                else: st.warning("No low anchor data available")

        with ptabs[3]:
            if skyline_anchor_spx:
                price, ts = skyline_anchor_spx; ts_ct = ts.astimezone(CT_TZ)
                skyline_proj = project_anchor_line(price, ts_ct, st.session_state.spx_slopes['skyline'], proj_day)
                if not skyline_proj.empty:
                    st.subheader("Skyline SPX Projection (90% Zone)")
                    st.info("Strategy: Bearish candle touches from above + closes above = BUY signal")
                    st.dataframe(skyline_proj, use_container_width=True, hide_index=True)
                    sky_analysis = calculate_entry_exit_table(skyline_proj, "SKYLINE")
                    if not sky_analysis.empty:
                        st.subheader("Skyline Bounce Strategy")
                        st.dataframe(sky_analysis, use_container_width=True, hide_index=True)
            else: st.warning("No skyline anchor detected")

        with ptabs[4]:
            if baseline_anchor_spx:
                price, ts = baseline_anchor_spx; ts_ct = ts.astimezone(CT_TZ)
                baseline_proj = project_anchor_line(price, ts_ct, st.session_state.spx_slopes['baseline'], proj_day)
                if not baseline_proj.empty:
                    st.subheader("Baseline SPX Projection (90% Zone)")
                    st.info("Strategy: Bearish candle touches from above + closes above = BUY signal")
                    st.dataframe(baseline_proj, use_container_width=True, hide_index=True)
                    base_analysis = calculate_entry_exit_table(baseline_proj, "BASELINE")
                    if not base_analysis.empty:
                        st.subheader("Baseline Bounce Strategy")
                        st.dataframe(base_analysis, use_container_width=True, hide_index=True)
            else: st.warning("No baseline anchor detected")
    else:
        st.info("Configure your dates and click 'Generate SPX Anchors' to begin analysis")

# ═══════════════════════════════════════════════════════════════════════════════
# PART 3 — STOCK ANCHORS TAB (strategy preserved; plumbing fixed)
# ═══════════════════════════════════════════════════════════════════════════════
def calculate_weekly_entry_exit_table(projection_df: pd.DataFrame, ticker: str, anchor_type: str, day_name: str) -> pd.DataFrame:
    if projection_df.empty: return pd.DataFrame()
    rows = []
    stock_volatility = get_stock_volatility_factor(ticker)
    day_multipliers = {"Wednesday":1.1,"Thursday":1.0,"Friday":0.9}
    day_mult = day_multipliers.get(day_name,1.0)
    for _, r in projection_df.iterrows():
        time_slot = r['Time']; price=r['Price']
        stop_distance = price*stock_volatility*0.012
        tp1_distance = stop_distance*1.5
        tp2_distance = stop_distance*2.5
        slope_sign = 1 if anchor_type in ['SKYLINE','HIGH'] else -1
        entry=price; stop=price - (stop_distance*slope_sign)
        tp1=price + (tp1_distance*slope_sign); tp2=price + (tp2_distance*slope_sign)
        direction = "LONG" if slope_sign>0 else "SHORT"
        entry_prob = calculate_stock_entry_probability(ticker, anchor_type, time_slot)*day_mult
        tp1_prob   = calculate_stock_target_probability(ticker, tp1_distance, stop_distance, 1)*day_mult
        tp2_prob   = calculate_stock_target_probability(ticker, tp2_distance, stop_distance, 2)*day_mult
        rows.append({
            'Time':time_slot,'Direction':direction,'Entry':round(entry,2),'Stop':round(stop,2),
            'TP1':round(tp1,2),'TP2':round(tp2,2),'Risk':round(stop_distance,2),
            'Entry_Prob':f"{min(95,entry_prob):.1f}%",'TP1_Prob':f"{min(85,tp1_prob):.1f}%",
            'TP2_Prob':f"{min(75,tp2_prob):.1f}%",'Day':day_name
        })
    return pd.DataFrame(rows)

def get_stock_volatility_factor(ticker: str) -> float:
    vf = {'TSLA':1.8,'NVDA':1.6,'META':1.4,'NFLX':1.3,'AMZN':1.2,'GOOGL':1.1,'MSFT':1.0,'AAPL':0.9}
    return vf.get(ticker, 1.2)

def calculate_stock_entry_probability(ticker: str, anchor_type: str, time_slot: str) -> float:
    base={'HIGH':60,'CLOSE':65,'LOW':60,'SKYLINE':70,'BASELINE':75}.get(anchor_type,60)
    hour=int(time_slot.split(':')[0]); time_adj=10 if hour in [9,10] else (5 if hour in [13,14] else 0)
    ticker_adj = 5 if ticker in ['TSLA','NVDA','META'] else (-5 if ticker in ['AAPL','MSFT'] else 0)
    return min(90, max(40, base+time_adj+ticker_adj))

def calculate_stock_target_probability(ticker: str, target_distance: float, stop_distance: float, target_num: int) -> float:
    rr = target_distance/max(1e-9, stop_distance); vol = get_stock_volatility_factor(ticker)
    if target_num==1:
        base = 65 - (rr-1.5)*8; vol_adj=(vol-1)*10
    else:
        base = 40 - (rr-2.5)*6; vol_adj=(vol-1)*8
    return min(80, max(20, base+vol_adj))

with tab2:
    st.subheader("Stock Anchor Analysis")
    st.caption("Mon/Tue combined session analysis for individual stocks")

    st.write("Core Tickers:")
    tcols = st.columns(4)
    selected_ticker = None
    core_tickers = ['TSLA','NVDA','AAPL','MSFT','AMZN','GOOGL','META','NFLX']
    for i,ticker in enumerate(core_tickers):
        with tcols[i % 4]:
            if st.button(f"{ticker}", key=f"stk_btn_{ticker}"):
                selected_ticker = ticker; st.session_state.selected_stock = ticker

    st.markdown("---")
    custom_ticker = st.text_input("Custom Symbol", placeholder="Enter any ticker symbol", key="stk_custom_input")
    if custom_ticker:
        selected_ticker = custom_ticker.upper(); st.session_state.selected_stock = selected_ticker
    if not selected_ticker and 'selected_stock' in st.session_state:
        selected_ticker = st.session_state.selected_stock

    if selected_ticker:
        st.info(f"Selected: {selected_ticker}")
        default_slope = STOCK_SLOPES.get(selected_ticker, 0.0150)
        current_slope = st.session_state.stock_slopes.get(selected_ticker, default_slope)
        slope_magnitude = st.number_input(
            f"{selected_ticker} Slope Magnitude", value=current_slope,
            step=0.0001, format="%.4f", key=f"stk_slope_{selected_ticker}",
            help="Used as +magnitude for Skyline, -magnitude for Baseline"
        )
        st.session_state.stock_slopes[selected_ticker] = slope_magnitude

        c1,c2,c3 = st.columns(3)
        with c1:
            monday_date = st.date_input("Monday Date", value=datetime.now(CT_TZ).date()-timedelta(days=2), key=f"stk_mon_{selected_ticker}")
        with c2:
            tuesday_date= st.date_input("Tuesday Date", value=monday_date+timedelta(days=1), key=f"stk_tue_{selected_ticker}")
        with c3:
            wed_date=tuesday_date+timedelta(days=1); thu_date=tuesday_date+timedelta(days=2); fri_date=tuesday_date+timedelta(days=3)
            st.write("Project for remaining week:"); st.caption(f"Wed: {wed_date}, Thu: {thu_date}, Fri: {fri_date}")
        st.markdown("---")

        if st.button(f"Analyze {selected_ticker}", key=f"stk_analyze_{selected_ticker}", type="primary"):
            with st.spinner(f"Analyzing {selected_ticker} Mon/Tue sessions..."):
                mon = fetch_live_data(selected_ticker, monday_date, monday_date)
                tue = fetch_live_data(selected_ticker, tuesday_date, tuesday_date)
                if mon.empty and tue.empty:
                    st.error(f"No data available for {selected_ticker} on selected dates")
                else:
                    if mon.empty: st.warning("No Monday data, using Tuesday only"); combined=tue
                    elif tue.empty: st.warning("No Tuesday data, using Monday only"); combined=mon
                    else: combined = pd.concat([mon,tue]).sort_index()
                    st.session_state.stock_analysis_data = combined
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
                'low': (last_bar['Close'], last_bar.name)
            }

            st.subheader(f"{selected_ticker} Weekly Projections")
            projection_dates = [("Wednesday", tuesday_date+timedelta(days=1)),
                                ("Thursday", tuesday_date+timedelta(days=2)),
                                ("Friday",   tuesday_date+timedelta(days=3))]
            wtabs = st.tabs(["Wed","Thu","Fri"])
            for i,(day_name, proj_date) in enumerate(projection_dates):
                with wtabs[i]:
                    st.write(f"{day_name} - {proj_date}")
                    atabs = st.tabs(["High","Close","Low","Skyline","Baseline"])
                    # High
                    with atabs[0]:
                        price, ts = manual_anchors['high']; ts_ct = ts.astimezone(CT_TZ)
                        high_proj = project_anchor_line(price, ts_ct, slope_magnitude, proj_date)
                        st.dataframe(high_proj, use_container_width=True)
                        st.write("Entry/Exit Analysis")
                        st.dataframe(calculate_weekly_entry_exit_table(high_proj, selected_ticker, "HIGH", day_name), use_container_width=True)
                    # Close
                    with atabs[1]:
                        price, ts = manual_anchors['close']; ts_ct = ts.astimezone(CT_TZ)
                        close_proj = project_anchor_line(price, ts_ct, slope_magnitude, proj_date)
                        st.dataframe(close_proj, use_container_width=True)
                        st.write("Entry/Exit Analysis")
                        st.dataframe(calculate_weekly_entry_exit_table(close_proj, selected_ticker, "CLOSE", day_name), use_container_width=True)
                    # Low
                    with atabs[2]:
                        price, ts = manual_anchors['low']; ts_ct = ts.astimezone(CT_TZ)
                        low_proj = project_anchor_line(price, ts_ct, slope_magnitude, proj_date)
                        st.dataframe(low_proj, use_container_width=True)
                        st.write("Entry/Exit Analysis")
                        st.dataframe(calculate_weekly_entry_exit_table(low_proj, selected_ticker, "LOW", day_name), use_container_width=True)
                    # Skyline
                    with atabs[3]:
                        if skyline_anchor:
                            sky_price, sky_time = skyline_anchor; sky_time_ct = sky_time.astimezone(CT_TZ)
                            skyline_proj = project_anchor_line(sky_price, sky_time_ct, slope_magnitude, proj_date)
                            st.dataframe(skyline_proj, use_container_width=True)
                            st.write("Entry/Exit Analysis")
                            st.dataframe(calculate_weekly_entry_exit_table(skyline_proj, selected_ticker, "SKYLINE", day_name), use_container_width=True)
                        else: st.warning("No skyline anchor detected")
                    # Baseline
                    with atabs[4]:
                        if baseline_anchor:
                            base_price, base_time = baseline_anchor; base_time_ct = base_time.astimezone(CT_TZ)
                            baseline_proj = project_anchor_line(base_price, base_time_ct, -slope_magnitude, proj_date)
                            st.dataframe(baseline_proj, use_container_width=True)
                            st.write("Entry/Exit Analysis")
                            st.dataframe(calculate_weekly_entry_exit_table(baseline_proj, selected_ticker, "BASELINE", day_name), use_container_width=True)
                        else: st.warning("No baseline anchor detected")

# ═══════════════════════════════════════════════════════════════════════════════
# PART 4 — SIGNALS & EMA TAB (unchanged logic)
# ═══════════════════════════════════════════════════════════════════════════════
def calculate_market_volatility(data: pd.DataFrame) -> float:
    if data.empty or len(data) < 2: return 1.5
    returns = data['Close'].pct_change().dropna()
    if returns.empty: return 1.5
    return float(returns.std() * np.sqrt(390) * 100)

def calculate_average_true_range(data: pd.DataFrame, periods: int = 14) -> pd.Series:
    if data.empty or len(data) < periods: return pd.Series(index=data.index, dtype=float)
    hl = data['High'] - data['Low']
    hc = (data['High'] - data['Close'].shift()).abs()
    lc = (data['Low']  - data['Close'].shift()).abs()
    tr = np.maximum(hl, np.maximum(hc, lc))
    return tr.rolling(window=periods).mean()

def detect_anchor_touches(price_data: pd.DataFrame, anchor_line: pd.DataFrame) -> pd.DataFrame:
    if price_data.empty or anchor_line.empty: return pd.DataFrame()
    anchor_dict = {row['Time']: row['Price'] for _, row in anchor_line.iterrows()}
    touches=[]; atr_series = calculate_average_true_range(price_data.tail(20), 14)
    for idx, bar in price_data.iterrows():
        t = format_ct_time(idx)
        if t not in anchor_dict: continue
        anchor_price = anchor_dict[t]
        low_d = abs(bar['Low'] - anchor_price); high_d=abs(bar['High'] - anchor_price)
        tol = (atr_series.iloc[-1]*0.3) if not atr_series.empty else (anchor_price*0.002)
        hits = (bar['Low'] <= anchor_price + tol and bar['High'] >= anchor_price - tol)
        if hits:
            is_bear = bar['Close'] < bar['Open']; is_bull = bar['Close'] > bar['Open']
            closest = min(low_d, high_d)
            quality = max(0, 100 - (closest / max(1e-9, tol) * 100))
            vol_ma = price_data['Volume'].rolling(20).mean().iloc[-1] if 'Volume' in price_data.columns else 0
            vol_ratio = bar['Volume'] / vol_ma if vol_ma>0 else 1.0
            vol_strength = min(100, vol_ratio*50)
            touches.append({
                'Time': t, 'Anchor_Price': round(anchor_price,2),
                'Touch_Price': round(bar['Low'] if low_d < high_d else bar['High'],2),
                'Candle_Type': 'Bearish' if is_bear else ('Bullish' if is_bull else 'Doji'),
                'Open': round(bar['Open'],2), 'High': round(bar['High'],2), 'Low': round(bar['Low'],2), 'Close': round(bar['Close'],2),
                'Volume': int(bar['Volume']) if 'Volume' in bar else 0,
                'Touch_Quality': round(quality,1), 'Volume_Strength': round(vol_strength,1),
                'ATR_Tolerance': round(tol,2)
            })
    return pd.DataFrame(touches)

def analyze_anchor_line_interaction(price_data: pd.DataFrame, anchor_line: pd.DataFrame) -> pd.DataFrame:
    if price_data.empty or anchor_line.empty: return pd.DataFrame()
    anchor_dict = {row['Time']: row['Price'] for _, row in anchor_line.iterrows()}
    inter=[]
    for idx, bar in price_data.iterrows():
        t = format_ct_time(idx)
        if t not in anchor_dict: continue
        anchor_price = anchor_dict[t]
        price_above = bar['Close'] > anchor_price
        touched = (bar['Low'] <= anchor_price <= bar['High'])
        dist = bar['Close'] - anchor_price; dist_pct = (dist/anchor_price)*100
        inter.append({
            'Time': t, 'Close_Price': round(bar['Close'],2), 'Anchor_Price': round(anchor_price,2),
            'Distance': round(dist,2), 'Distance_Pct': round(dist_pct,2),
            'Touched': 'Yes' if touched else 'No', 'Position': 'Above' if price_above else 'Below'
        })
    return pd.DataFrame(inter)

def calculate_ema_crossover_signals(price_data: pd.DataFrame) -> pd.DataFrame:
    if price_data.empty or len(price_data) < 21: return pd.DataFrame()
    ema8  = calculate_ema(price_data['Close'], 8)
    ema21 = calculate_ema(price_data['Close'], 21)
    out=[]
    for i in range(1, len(price_data)):
        t = format_ct_time(price_data.index[i])
        prev8, prev21 = ema8.iloc[i-1], ema21.iloc[i-1]
        cur8,  cur21  = ema8.iloc[i],   ema21.iloc[i]
        price = price_data.iloc[i]['Close']
        x_type=None; sep = abs(cur8 - cur21)/max(1e-9, cur21)*100
        if prev8 <= prev21 and cur8 > cur21: x_type="Bullish Cross"
        elif prev8 >= prev21 and cur8 < cur21: x_type="Bearish Cross"
        regime = "Bullish" if cur8 > cur21 else "Bearish"
        out.append({'Time':t,'Price':round(price,2),'EMA8':round(cur8,2),'EMA21':round(cur21,2),
                    'Separation':round(sep,3),'Regime':regime,'Crossover':x_type or 'None',
                    'Signal_Strength':'Strong' if sep>0.5 else ('Moderate' if sep>0.2 else 'Weak')})
    return pd.DataFrame(out)

with tab3:
    st.subheader("Signal Detection & Market Analysis")
    st.caption("Real-time anchor touch detection with market-derived analytics")

    c1,c2 = st.columns(2)
    with c1:
        signal_symbol = st.selectbox("Analysis Symbol", ["^GSPC","ES=F","SPY"], index=0, key="sig_symbol")
    with c2:
        signal_day = st.date_input("Analysis Day", value=datetime.now(CT_TZ).date(), key="sig_day")

    st.markdown("Reference Line Configuration")
    r1,r2,r3 = st.columns(3)
    with r1:
        anchor_price = st.number_input("Anchor Price", value=6000.0, step=0.1, format="%.2f", key="sig_anchor_price")
    with r2:
        anchor_time_input = st.time_input("Anchor Time (CT)", value=time(17,0), key="sig_anchor_time")
    with r3:
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
                    st.session_state.signal_anchor = {'price':anchor_price,'time':anchor_time_input,'slope':ref_slope}
                    st.session_state.signal_symbol = signal_symbol
                    st.session_state.signal_ready  = True

    if st.session_state.get('signal_ready', False):
        signal_data = st.session_state.signal_data
        anchor_cfg  = st.session_state.signal_anchor
        symbol      = st.session_state.signal_symbol
        st.subheader(f"{symbol} Market Analysis Results")
        anchor_datetime_ct = CT_TZ.localize(datetime.combine(signal_day, anchor_cfg['time']))
        ref_line_proj = project_anchor_line(anchor_cfg['price'], anchor_datetime_ct, anchor_cfg['slope'], signal_day)
        volatility = calculate_market_volatility(signal_data)
        atr_series = calculate_average_true_range(signal_data, 14)
        vwap_series= calculate_vwap(signal_data)

        st.subheader("Market Overview")
        oc1,oc2,oc3 = st.columns(3)
        with oc1:
            day_range = signal_data['High'].max() - signal_data['Low'].min()
            st.metric("Day Range", f"${day_range:.2f}")
        with oc2:
            st.metric("Volatility", f"{volatility:.2f}%")
        with oc3:
            current_atr = atr_series.iloc[-1] if not atr_series.empty else 0
            st.metric("Current ATR", f"${current_atr:.2f}")

        st.markdown("---")
        stabs = st.tabs(["Reference Line","Anchor Touches","Line Interaction","EMA Analysis"])

        with stabs[0]:
            st.subheader("Reference Line Projection")
            st.dataframe(ref_line_proj, use_container_width=True, hide_index=True)
            if not ref_line_proj.empty:
                pr = ref_line_proj['Price'].max() - ref_line_proj['Price'].min()
                avg = ref_line_proj['Price'].mean(); rng_pct = (pr/avg)*100
                st.info(f"Projection range: ${pr:.2f} ({rng_pct:.1f}% of average price)")

        with stabs[1]:
            anchor_touches = detect_anchor_touches(signal_data, ref_line_proj)
            if not anchor_touches.empty:
                st.subheader("Detected Anchor Touches")
                st.dataframe(anchor_touches, use_container_width=True, hide_index=True)
                total = len(anchor_touches); avg_q = anchor_touches['Touch_Quality'].mean(); avg_vs = anchor_touches['Volume_Strength'].mean()
                tc1,tc2,tc3 = st.columns(3)
                tc1.metric("Total Touches", total); tc2.metric("Avg Touch Quality", f"{avg_q:.1f}%"); tc3.metric("Avg Volume Strength", f"{avg_vs:.1f}%")
            else:
                st.info("No anchor line touches detected for this day")

        with stabs[2]:
            line_interaction = analyze_anchor_line_interaction(signal_data, ref_line_proj)
            if not line_interaction.empty:
                st.subheader("Price-Anchor Line Interaction")
                st.dataframe(line_interaction, use_container_width=True, hide_index=True)
                touches = line_interaction[line_interaction['Touched']=='Yes']
                above = line_interaction[line_interaction['Position']=='Above']
                ic1,ic2,ic3=st.columns(3)
                ic1.metric("Touch Points", len(touches))
                ic2.metric("Time Above Line", f"{(len(above)/len(line_interaction))*100:.1f}%")
                ic3.metric("Avg Distance", f"${abs(line_interaction['Distance']).mean():.2f}")
            else:
                st.info("No line interaction data available")

        with stabs[3]:
            ema_analysis = calculate_ema_crossover_signals(signal_data)
            if not ema_analysis.empty:
                st.subheader("EMA 8/21 Analysis")
                st.dataframe(ema_analysis, use_container_width=True, hide_index=True)
                crossovers = ema_analysis[ema_analysis['Crossover']!='None']
                current_regime = ema_analysis.iloc[-1]['Regime'] if not ema_analysis.empty else 'Unknown'
                current_sep    = ema_analysis.iloc[-1]['Separation'] if not ema_analysis.empty else 0
                ec1,ec2,ec3 = st.columns(3)
                ec1.metric("Crossovers", len(crossovers))
                ec2.metric("Current Regime", current_regime)
                ec3.metric("EMA Separation", f"{current_sep:.3f}%")
            else:
                st.info("Insufficient data for EMA analysis")
    else:
        st.info("Configure your parameters and click 'Analyze Market Signals' to begin")

# ═══════════════════════════════════════════════════════════════════════════════
# PART 5 — CONTRACT TOOL TAB (logic unchanged)
# ═══════════════════════════════════════════════════════════════════════════════
def calculate_contract_volatility(price_data: pd.DataFrame, window: int = 20) -> float:
    if price_data.empty or len(price_data) < window: return 0.15
    returns = price_data['Close'].pct_change().dropna()
    if returns.empty: return 0.15
    vol = returns.rolling(window=window).std().iloc[-1]
    return float(vol) if not np.isnan(vol) else 0.15

def calculate_price_momentum(p1_price: float, p2_price: float, time_hours: float) -> dict:
    change = p2_price - p1_price
    change_pct = (change / p1_price)*100 if p1_price>0 else 0
    hourly = change / time_hours if time_hours>0 else 0
    abs_pct = abs(change_pct)
    if abs_pct>=15: strength, conf = "Very Strong", 95
    elif abs_pct>=8: strength, conf = "Strong", 85
    elif abs_pct>=3: strength, conf = "Moderate", 70
    else: strength, conf = "Weak", 50
    return {'change':change,'change_pct':change_pct,'hourly_rate':hourly,'strength':strength,'confidence':conf}

def calculate_market_based_targets(entry_price: float, market_data: pd.DataFrame, direction: str) -> dict:
    if market_data.empty:
        base = entry_price*0.02
        return {'tp1': entry_price + base if direction=="BUY" else entry_price - base,
                'tp2': entry_price + base*2.5 if direction=="BUY" else entry_price - base*2.5,
                'stop_distance': base*0.6}
    atr_series = calculate_average_true_range(market_data, 14)
    cur_atr = atr_series.iloc[-1] if not atr_series.empty else entry_price*0.015
    recent_range = market_data['High'].tail(10).max() - market_data['Low'].tail(10).min()
    vol_factor = recent_range / max(1e-9, market_data['Close'].tail(10).mean())
    base_t = cur_atr*1.2*(1+vol_factor*0.5)
    ext_t  = cur_atr*3.0*(1+vol_factor*0.3)
    if direction=="BUY":
        tp1, tp2 = entry_price + base_t, entry_price + ext_t
    else:
        tp1, tp2 = entry_price - base_t, entry_price - ext_t
    stop = cur_atr*0.8
    return {'tp1':tp1,'tp2':tp2,'stop_distance':stop,'atr':cur_atr,'volatility_factor':vol_factor}

def analyze_overnight_market_behavior(symbol: str, start_date: date, end_date: date) -> dict:
    overnight_data = fetch_live_data(symbol, start_date - timedelta(days=5), end_date)
    if overnight_data.empty:
        return {'avg_overnight_change':0,'overnight_volatility':0.02,'gap_frequency':0,'mean_reversion_rate':0.6}
    overnight_moves=[]; gap_moves=[]
    for date_group in overnight_data.groupby(overnight_data.index.date):
        daily = date_group[1]
        if len(daily) < 2: continue
        day_close = daily.iloc[-1]['Close']
        next_open = daily.iloc[0]['Open'] if len(daily)>0 else day_close
        chg = (next_open - day_close)/day_close if day_close>0 else 0
        overnight_moves.append(chg)
        if abs(chg) > 0.005: gap_moves.append(abs(chg))
    if not overnight_moves:
        return {'avg_overnight_change':0,'overnight_volatility':0.02,'gap_frequency':0,'mean_reversion_rate':0.6}
    avg = float(np.mean(overnight_moves))
    vol = float(np.std(overnight_moves))
    gap_freq = len(gap_moves)/len(overnight_moves) if overnight_moves else 0
    rev_ct = sum(1 for m in overnight_moves if abs(m) < np.std(overnight_moves))
    mr = rev_ct/len(overnight_moves) if overnight_moves else 0.6
    return {'avg_overnight_change':avg,'overnight_volatility':vol,'gap_frequency':gap_freq,'mean_reversion_rate':mr}

def project_contract_line(anchor_price: float, anchor_time: datetime, slope: float, target_date: date) -> pd.DataFrame:
    rth_slots = rth_slots_ct(target_date)
    rows=[]
    for slot_time in rth_slots:
        blocks = (slot_time - anchor_time).total_seconds()/1800
        rows.append({'Time':format_ct_time(slot_time),'Contract_Price':round(anchor_price + slope*blocks,2),'Blocks_from_Anchor':round(blocks,1)})
    return pd.DataFrame(rows)

with tab4:
    st.subheader("Contract Tool")
    st.caption("Overnight contract analysis for RTH entry optimization")

    st.subheader("Overnight Contract Price Points")
    p1c, p2c = st.columns(2)
    with p1c:
        p1_date = st.date_input("Point 1 Date", value=datetime.now(CT_TZ).date()-timedelta(days=1), key="ct_p1_date")
        p1_time = st.time_input("Point 1 Time (CT)", value=time(20,0), key="ct_p1_time")
        p1_price= st.number_input("Point 1 Contract Price", value=10.0, min_value=0.01, step=0.01, format="%.2f", key="ct_p1_price")
    with p2c:
        p2_date = st.date_input("Point 2 Date", value=datetime.now(CT_TZ).date(), key="ct_p2_date")
        p2_time = st.time_input("Point 2 Time (CT)", value=time(8,0), key="ct_p2_time")
        p2_price= st.number_input("Point 2 Contract Price", value=12.0, min_value=0.01, step=0.01, format="%.2f", key="ct_p2_price")

    projection_day = st.date_input("RTH Projection Day", value=p2_date, key="ct_proj_day")
    p1_dt = datetime.combine(p1_date, p1_time); p2_dt = datetime.combine(p2_date, p2_time)

    if p2_dt <= p1_dt:
        st.error("Point 2 must be after Point 1")
    else:
        hours = (p2_dt - p1_dt).total_seconds()/3600
        momentum = calculate_price_momentum(p1_price, p2_price, hours)
        mc1,mc2,mc3,mc4 = st.columns(4)
        mc1.metric("Time Span", f"{hours:.1f} hours")
        mc2.metric("Price Change", f"{momentum['change']:+.2f}")
        mc3.metric("Change %", f"{momentum['change_pct']:+.1f}%")
        mc4.metric("Momentum", momentum['strength'])

    st.markdown("---")
    if st.button("Analyze Contract Projections", key="ct_generate", type="primary"):
        if p2_dt <= p1_dt:
            st.error("Please ensure Point 2 is after Point 1")
        else:
            with st.spinner("Analyzing contract and market data..."):
                try:
                    minutes = (p2_dt - p1_dt).total_seconds()/60
                    blocks_between = minutes/30
                    contract_slope = (p2_price - p1_price)/blocks_between if blocks_between>0 else 0
                    underlying = fetch_live_data("^GSPC", projection_day - timedelta(days=10), projection_day)
                    overnight = analyze_overnight_market_behavior("^GSPC", projection_day - timedelta(days=10), projection_day)
                    p1_ct = CT_TZ.localize(p1_dt)
                    contract_proj = project_contract_line(p1_price, p1_ct, contract_slope, projection_day)
                    st.session_state.contract_projections = contract_proj
                    st.session_state.contract_config = {
                        'p1_price': p1_price, 'p1_time': p1_ct, 'p2_price': p2_price, 'p2_time': CT_TZ.localize(p2_dt),
                        'slope': contract_slope, 'momentum': momentum, 'overnight_analysis': overnight
                    }
                    st.session_state.underlying_data = underlying
                    st.session_state.contract_ready = True
                except Exception as e:
                    st.error(f"Analysis error: {str(e)}")

    if st.session_state.get('contract_ready', False):
        st.subheader("Contract Analysis Results")
        projections = st.session_state.contract_projections
        config = st.session_state.contract_config
        underlying_data = st.session_state.get('underlying_data', pd.DataFrame())

        ctabs = st.tabs(["RTH Projections","Market Analysis","Risk Management"])
        with ctabs[0]:
            st.subheader("RTH Contract Price Projections")
            if not projections.empty and not underlying_data.empty:
                rows=[]
                for _, r in projections.iterrows():
                    t = r['Time']; cp = r['Contract_Price']
                    direction = "BUY" if config['momentum']['change']>0 else "SELL"
                    targets = calculate_market_based_targets(cp, underlying_data, direction)
                    hour = int(t.split(':')[0])
                    if hour in [8,9]: time_prob = config['momentum']['confidence'] + 10
                    elif hour in [13,14]: time_prob = config['momentum']['confidence'] + 5
                    else: time_prob = config['momentum']['confidence']
                    rows.append({'Time':t,'Contract_Price':round(cp,2),'Direction':direction,
                                 'TP1':round(targets['tp1'],2),'TP2':round(targets['tp2'],2),
                                 'Stop_Distance':round(targets['stop_distance'],2),
                                 'Entry_Probability':f"{min(95,time_prob):.0f}%",'ATR_Base':round(targets.get('atr',0),2)})
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
            else:
                st.dataframe(projections, use_container_width=True, hide_index=True)

        with ctabs[1]:
            st.subheader("Underlying Market Analysis")
            momentum = config['momentum']; overnight = config['overnight_analysis']
            sc1,sc2,sc3 = st.columns(3)
            sc1.metric("Hourly Rate", f"{momentum['hourly_rate']:+.2f}")
            sc2.metric("Strength", momentum['strength'])
            sc3.metric("Confidence", f"{momentum['confidence']}%")
            st.subheader("Overnight Market Behavior")
            oc1,oc2,oc3 = st.columns(3)
            oc1.metric("Avg Overnight Change", f"{overnight['avg_overnight_change']*100:+.2f}%")
            oc2.metric("Overnight Volatility", f"{overnight['overnight_volatility']*100:.2f}%")
            oc3.metric("Gap Frequency", f"{overnight['gap_frequency']*100:.1f}%")
            if not underlying_data.empty:
                current_vol = calculate_contract_volatility(underlying_data)
                dr = underlying_data['High'].max() - underlying_data['Low'].min()
                cc1,cc2=st.columns(2)
                cc1.metric("Recent Volatility", f"{current_vol*100:.2f}%")
                cc2.metric("Recent Range", f"${dr:.2f}")

        with ctabs[2]:
            st.subheader("Risk Management Analysis")
            if not underlying_data.empty and not projections.empty:
                mvol = calculate_contract_volatility(underlying_data)
                atr_series = calculate_average_true_range(underlying_data, 14)
                cur_atr = atr_series.iloc[-1] if not atr_series.empty else 0
                rc1,rc2,rc3=st.columns(3)
                if mvol > 0.025: pos_rec, risk_level = "Reduce Size", "High"
                else: pos_rec, risk_level = "Standard Size", "Normal"
                rc1.metric("Risk Level", risk_level); rc1.caption(f"Volatility: {mvol*100:.2f}%")
                rc2.metric("Position Sizing", pos_rec); rc2.caption(f"ATR: ${cur_atr:.2f}")
                avg_cp = projections['Contract_Price'].mean()
                max_risk = cur_atr*1.5; risk_per_dollar = (max_risk/max(1e-9,avg_cp))*100
                rc3.metric("Risk per $", f"{risk_per_dollar:.1f}%"); rc3.caption("Based on ATR stop")
                st.subheader("Time-Based Risk Assessment")
                risk_rows=[]; overnight_vol = config['overnight_analysis']['overnight_volatility']
                for _, r in projections.iterrows():
                    t=r['Time']; cp=r['Contract_Price']; hour=int(t.split(':')[0])
                    if hour in [8,9]: mult, rating = 1.5, "High"
                    elif hour in [10,11]: mult, rating = 1.0, "Medium"
                    else: mult, rating = 0.8, "Low"
                    base_stop = cur_atr*1.2; adj_stop = base_stop*mult*(1+overnight_vol*2)
                    risk_rows.append({'Time':t,'Contract_Price':round(cp,2),'Risk_Rating':rating,
                                      'Suggested_Stop':round(adj_stop,2),'Risk_Multiplier':f"{mult:.1f}x",'Max_Risk_$':round(adj_stop,2)})
                st.dataframe(pd.DataFrame(risk_rows), use_container_width=True, hide_index=True)
            else:
                st.info("Need underlying market data for comprehensive risk analysis")

# ═══════════════════════════════════════════════════════════════════════════════
# PART 6 — FINAL INTEGRATION & SUMMARY (plumbing-only fixes)
# ═══════════════════════════════════════════════════════════════════════════════
def calculate_volume_profile_strength(data: pd.DataFrame, price_level: float) -> float:
    if data.empty or 'Volume' not in data.columns: return 50.0
    tol = max(0.01, price_level*0.005)
    nearby = data[(data['Low'] <= price_level + tol) & (data['High'] >= price_level - tol)]
    if nearby.empty: return 40.0
    level_vol = float(nearby['Volume'].sum()); total_vol = float(data['Volume'].sum() or 0)
    if total_vol<=0: return 50.0
    conc = (level_vol/total_vol)*100
    return 95.0 if conc>=20 else 85.0 if conc>=15 else 75.0 if conc>=10 else 60.0 if conc>=5 else 45.0

def detect_market_regime(data: pd.DataFrame) -> dict:
    if data.empty or len(data)<20:
        return {'regime':'INSUFFICIENT_DATA','trend':'NEUTRAL','strength':0,'volatility':1.5,'price_change':0}
    closes = data['Close'].tail(20)
    price_change = float((closes.iloc[-1]-closes.iloc[0])/max(1e-9,closes.iloc[0])*100)
    returns = closes.pct_change().dropna()
    volatility = float(returns.std()*np.sqrt(390)*100) if not returns.empty else 0.0
    trend = 'BULLISH' if price_change>1.0 else ('BEARISH' if price_change<-1.0 else 'NEUTRAL')
    strength = min(100.0, abs(price_change)*10.0)
    regime = 'HIGH_VOLATILITY' if volatility>=3.0 else ('MODERATE_VOLATILITY' if volatility>=1.8 else 'LOW_VOLATILITY')
    return {'regime':regime,'trend':trend,'strength':strength,'volatility':volatility,'price_change':price_change}

def calculate_support_resistance_strength(data: pd.DataFrame, price_level: float) -> float:
    if data.empty or len(data)<10: return 50.0
    tol = max(0.01, price_level*0.008)
    highs,lows = data['High'], data['Low']
    total = int((highs - price_level).abs().le(tol).sum() + (lows - price_level).abs().le(tol).sum())
    return 90.0 if total>=4 else 80.0 if total>=3 else 70.0 if total>=2 else 60.0 if total>=1 else 45.0

def calculate_confluence_score(price: float, anchor_price: float, market_data: pd.DataFrame) -> float:
    if market_data.empty or anchor_price<=0: return 50.0
    dist_pct = abs(price-anchor_price)/anchor_price*100
    proximity = max(0.0, 100.0 - dist_pct*15.0)
    volume_score = calculate_volume_profile_strength(market_data, price)
    regime_info  = detect_market_regime(market_data)
    regime_score = 80.0 if regime_info['trend']!='NEUTRAL' else 55.0
    sr_score = calculate_support_resistance_strength(market_data, price)
    return float(min(100.0, max(0.0, (proximity+volume_score+regime_score+sr_score)/4.0)))

def calculate_time_edge_from_data(symbol: str, lookback_days: int = 30) -> dict:
    try:
        end_date = datetime.now(CT_TZ).date(); start_date = end_date - timedelta(days=lookback_days)
        hist = fetch_live_data(symbol, start_date, end_date)
        if hist.empty: return {}
        slots = ['08:30','09:00','09:30','10:00','10:30','11:00','11:30','12:00','12:30','13:00','13:30','14:00','14:30']
        out={}
        for s in slots:
            sd = hist.between_time(s, s)
            if sd.empty: continue
            ret = sd['Close'].pct_change().dropna()
            if len(ret)<=5: continue
            vol=float(ret.std()*100); am=float(ret.abs().mean()*100); bias=float((ret>0).mean()*100)
            edge = min(20.0, vol*15.0 + abs(bias-50.0)) if (vol>0.8 and abs(bias-50.0)>10.0) else vol*10.0
            out[s]={'volatility':vol,'avg_move':am,'upward_bias':bias,'edge_score':float(edge)}
        return out
    except Exception:
        return {}

def get_market_hours_status() -> dict:
    now = datetime.now(CT_TZ); weekday = now.weekday()<5
    market_open  = now.replace(hour=8, minute=30, second=0, microsecond=0)
    market_close = now.replace(hour=14,minute=30, second=0, microsecond=0)
    pre_start    = now.replace(hour=7, minute=0,  second=0, microsecond=0)
    ah_end       = now.replace(hour=17,minute=0,  second=0, microsecond=0)
    if not weekday: status, session = "WEEKEND","Closed"
    elif market_open <= now <= market_close: status, session = "RTH_OPEN","Regular Hours"
    elif pre_start <= now < market_open: status, session = "PREMARKET","Pre-Market"
    elif market_close < now <= ah_end: status, session = "AFTERHOURS","After Hours"
    else: status, session = "CLOSED","Closed"
    if status=="CLOSED" and weekday: next_open = market_open + timedelta(days=1)
    elif status=="WEEKEND": 
        days_until_mon = (7 - now.weekday()) % 7; days_until_mon = 1 if days_until_mon==0 else days_until_mon
        next_open = now.replace(hour=8, minute=30, second=0, microsecond=0) + timedelta(days=days_until_mon)
    else: next_open=None
    return {'status':status,'session':session,'current_time':now,'is_trading_day':weekday,'next_open':next_open}

st.markdown("---")
st.subheader("Analysis Summary Dashboard")
market_status = get_market_hours_status()
s1,s2,s3 = st.columns(3)
with s1:
    st.markdown("**Analysis Status**")
    st.write(f"SPX Anchors: {'Ready' if st.session_state.spx_analysis_ready else 'Pending'}")
    st.write(f"Stock Analysis: {'Ready' if st.session_state.stock_analysis_ready else 'Pending'}")
    st.write(f"Signal Detection: {'Ready' if st.session_state.signal_ready else 'Pending'}")
    st.write(f"Contract Tool: {'Ready' if st.session_state.contract_ready else 'Pending'}")
with s2:
    st.markdown("**Current Settings**")
    st.write(f"Skyline Slope: {st.session_state.spx_slopes['skyline']:+.3f}")
    st.write(f"Baseline Slope: {st.session_state.spx_slopes['baseline']:+.3f}")
    st.write(f"ES→SPX Offset: {st.session_state.current_offset:+.1f}")
    st.write(f"High/Close/Low: {st.session_state.spx_slopes['high']:+.3f}")
with s3:
    st.markdown("**Market Status**")
    st.write(f"Market: {market_status['session']}")
    st.write(f"Time (CT): {market_status['current_time'].strftime('%H:%M:%S')}")
    if market_status['next_open']:
        delta = market_status['next_open'] - market_status['current_time']
        st.write(f"Next Open: {int(delta.total_seconds()//3600)}h")
    else:
        st.write("Session Active")

st.markdown("---")
st.subheader("Quick Actions")
qa1,qa2,qa3,qa4 = st.columns(4)
with qa1:
    if st.button("Update ES Offset", key="quick_update_offset"):
        with st.spinner("Updating offset from market data..."):
            today=datetime.now(CT_TZ).date(); y=today - timedelta(days=1)
            es=fetch_live_data("ES=F", y, today); spx=fetch_live_data("^GSPC", y, today)
            if not es.empty and not spx.empty:
                st.session_state.current_offset = calculate_es_spx_offset(es, spx)
                st.success(f"Offset updated: {st.session_state.current_offset:+.1f}"); st.rerun()
            else: st.error("Failed to fetch offset data")
with qa2:
    if st.button("Reset All Analysis", key="quick_reset_all"):
        for k in ['spx_analysis_ready','stock_analysis_ready','signal_ready','contract_ready',
                  'es_anchor_data','spx_manual_anchors','stock_analysis_data','signal_data',
                  'contract_projections','contract_config','underlying_data']:
            if k in st.session_state: del st.session_state[k]
        st.success("All analysis reset"); st.rerun()
with qa3:
    if st.button("Reset Slopes", key="quick_reset_slopes"):
        st.session_state.spx_slopes=SPX_SLOPES.copy(); st.session_state.stock_slopes=STOCK_SLOPES.copy()
        st.success("Slopes reset to defaults"); st.rerun()
with qa4:
    if st.button("Test Connection", key="quick_test"):
        with st.spinner("Testing market connection..."):
            t = fetch_live_data("^GSPC", datetime.now().date()-timedelta(days=1), datetime.now().date())
            st.success("Connection successful") if not t.empty else st.error("Connection failed")

if any([st.session_state.get('spx_analysis_ready',False),
        st.session_state.get('stock_analysis_ready',False),
        st.session_state.get('signal_ready',False),
        st.session_state.get('contract_ready',False)]):
    st.markdown("---")
    st.subheader("Market Performance Insights")
    it1,it2,it3=st.tabs(["Market Regime","Time-of-Day Edge","Volume Analysis"])
    with it1:
        if st.session_state.get('signal_ready',False):
            sig = st.session_state.get('signal_data', pd.DataFrame())
            if not sig.empty:
                regime = detect_market_regime(sig)
                c1,c2=st.columns(2)
                with c1:
                    st.metric("Market Trend", regime['trend'])
                    st.metric("Trend Strength", f"{regime['strength']:.1f}")
                with c2:
                    st.metric("Volatility Regime", regime['regime'])
                    st.metric("Volatility Level", f"{regime['volatility']:.1f}%")
                if regime['trend']=='BULLISH' and regime['volatility']<2.5: ctx="Stable uptrend — favorable for long positions"
                elif regime['trend']=='BEARISH' and regime['volatility']<2.5: ctx="Stable downtrend — favorable for short positions"
                elif regime['volatility']>3.0: ctx="High volatility — use wider stops"
                else: ctx="Neutral/ranging — mean reversion focus"
                st.info(ctx)
        else: st.info("Generate signal analysis to see market regime data")
    with it2:
        if st.button("Calculate Time Edge", key="calc_time_edge"):
            with st.spinner("Analyzing time-of-day patterns..."):
                ted = calculate_time_edge_from_data("^GSPC", 30)
                if ted:
                    rows=[{'Time':t,'Volatility':f"{v['volatility']:.2f}%",'Avg Move':f"{v['avg_move']:.2f}%",
                           'Upward Bias':f"{v['upward_bias']:.1f}%",'Edge Score':f"{v['edge_score']:.1f}"} for t,v in ted.items()]
                    st.dataframe(pd.DataFrame(rows).sort_values('Time'), use_container_width=True, hide_index=True)
                else: st.error("Could not calculate time edge data")
        else:
            st.info("Click 'Calculate Time Edge' to analyze historical time-of-day patterns")
    with it3:
        if st.session_state.get('signal_ready',False):
            sig = st.session_state.get('signal_data', pd.DataFrame())
            if not sig.empty and 'Volume' in sig.columns:
                avg = float(sig['Volume'].mean()); std=float(sig['Volume'].std()); thr=avg+std
                high = sig[sig['Volume']>thr]
                v1,v2,v3=st.columns(3)
                v1.metric("Average Volume", f"{avg:,.0f}"); v2.metric("Volume Std Dev", f"{std:,.0f}"); v3.metric("High Volume Bars", len(high))
                if not high.empty:
                    hv = float((high['Close']-high['Open']).abs().mean())
                    normal = sig[sig['Volume']<=thr]
                    nv = float((normal['Close']-normal['Open']).abs().mean()) if not normal.empty else 0.0
                    st.info(f"High volume bars average move: ${hv:.2f}")
                    st.info(f"Normal volume bars average move: ${nv:.2f}")
            else: st.info("No volume data available for analysis")
        else: st.info("Run signal analysis first to see volume insights")

st.markdown("---")
d1,d2,d3=st.columns(3)
with d1:
    try:
        spx_t = fetch_live_data("^GSPC", datetime.now().date()-timedelta(days=1), datetime.now().date())
        status = "Active" if not spx_t.empty else "Issue"; lastup = spx_t.index[-1].strftime("%H:%M CT") if not spx_t.empty else "N/A"
    except: status,lastup="Error","N/A"
    st.write("**SPX Data**"); st.write(f"Status: {status}"); st.write(f"Last Update: {lastup}")
with d2:
    try:
        es_t = fetch_live_data("ES=F", datetime.now().date()-timedelta(days=1), datetime.now().date())
        status = "Active" if not es_t.empty else "Issue"; lastup = es_t.index[-1].strftime("%H:%M CT") if not es_t.empty else "N/A"
    except: status,lastup="Error","N/A"
    st.write("**ES Futures**"); st.write(f"Status: {status}"); st.write(f"Last Update: {lastup}")
with d3:
    st.write("**Current Session**")
    st.write(f"Offset Status: {'Live' if st.session_state.current_offset != 0 else 'Default'}")
    st.write(f"Session: {market_status['session']}")
st.markdown("---")
st.markdown(f"""
<div style='text-align:center; color:#888; font-size:0.9rem;'>
    SPX Prophet Analytics • Market Data Integration •
    Session: {datetime.now(CT_TZ).strftime('%H:%M:%S CT')} •
    Status: {market_status['session']}
</div>
""", unsafe_allow_html=True)
