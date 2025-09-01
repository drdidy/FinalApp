# app.py
# ═══════════════════════════════════════════════════════════════════════════════
# 🔮 SPX PROPHET — ENTERPRISE EDITION (All 4 tabs, unified)
# Core: Close-anchor fan from previous day’s 3:00 PM CT close, slope ±0.277/30m
# UI: Light/glass, concise tables, no debug/dev text
# ═══════════════════════════════════════════════════════════════════════════════

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import pytz
from datetime import datetime, date, time, timedelta
from typing import Dict, List, Optional, Tuple

# ───────────────────────────────────────────────────────────────────────────────
# CORE SETTINGS
# ───────────────────────────────────────────────────────────────────────────────
CT_TZ = pytz.timezone('America/Chicago')

# Regular Trading Hours (CT)
RTH_START = "08:30"
RTH_END   = "14:30"

# Default slope magnitude per 30-min block (your calibrated value)
DEFAULT_SLOPE = 0.277  # Top = +slope, Bottom = −slope

# Stock slope presets (kept for Stock tab input defaults)
STOCK_SLOPES_DEFAULT = {
    "TSLA": 0.0285, "NVDA": 0.086, "AAPL": 0.0155, "MSFT": 0.0541,
    "AMZN": 0.0139, "GOOGL": 0.0122, "META": 0.0674, "NFLX": 0.0230
}

# ───────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG + THEME
# ───────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="🔮 SPX Prophet Analytics",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Light / glass (neumorphism) styling
st.markdown("""
<style>
    :root {
        --glass-bg: rgba(255,255,255,0.65);
        --glass-border: rgba(255,255,255,0.35);
        --card-shadow: 0 10px 35px rgba(0,0,0,0.10);
        --primary: #2563eb;  /* blue-600 */
        --accent: #10b981;   /* emerald-500 */
        --danger: #ef4444;   /* red-500 */
        --warning: #f59e0b;  /* amber-500 */
        --muted: #6b7280;    /* gray-500 */
    }
    .block-container { padding-top: 1.5rem; }
    body { background: linear-gradient(135deg,#f8fafc 0%,#eef2ff 50%,#f0fdf4 100%); }

    .glass {
        background: var(--glass-bg);
        border: 1px solid var(--glass-border);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        box-shadow: var(--card-shadow);
        border-radius: 18px;
        padding: 1.0rem 1.0rem;
    }
    .metricCard {
        background: var(--glass-bg);
        border: 1px solid var(--glass-border);
        border-radius: 18px;
        padding: 1.1rem 1.2rem;
        box-shadow: var(--card-shadow);
    }
    .metricTitle { color: var(--muted); font-weight: 600; margin-bottom: .25rem; }
    .metricValue { font-size: 1.75rem; font-weight: 800; }
    .good { color: var(--accent); }
    .warn { color: var(--warning); }
    .bad  { color: var(--danger); }

    /* Buttons */
    .stButton>button {
        border-radius: 12px !important;
        padding: .6rem 1.0rem !important;
        border: 1px solid rgba(0,0,0,0.06) !important;
        box-shadow: 0 6px 18px rgba(37,99,235,0.15) !important;
        font-weight: 700 !important;
    }
    /* Tables */
    .stDataFrame { background: rgba(255,255,255,0.92); border-radius: 14px; }
    /* Hide default Streamlit header space */
    header { visibility: hidden; height: 0; }
</style>
""", unsafe_allow_html=True)

# ───────────────────────────────────────────────────────────────────────────────
# UTILITIES
# ───────────────────────────────────────────────────────────────────────────────
def rth_slots_ct(target_date: date) -> List[datetime]:
    """Generate 30-min RTH time slots in CT for the projection day."""
    start_dt = CT_TZ.localize(datetime.combine(target_date, time(8,30)))
    end_dt   = CT_TZ.localize(datetime.combine(target_date, time(14,30)))
    slots = []
    cur = start_dt
    while cur <= end_dt:
        slots.append(cur)
        cur += timedelta(minutes=30)
    return slots

def format_ct_time(dt: datetime) -> str:
    if dt.tzinfo is None:
        dt = CT_TZ.localize(dt)
    else:
        dt = dt.astimezone(CT_TZ)
    return dt.strftime("%H:%M")

def validate_ohlc(df: pd.DataFrame) -> bool:
    if df.empty: return False
    req = ['Open','High','Low','Close','Volume']
    if any(c not in df.columns for c in req): return False
    invalid = (
        (df['High'] < df['Low']) |
        (df['High'] < df['Open']) | (df['High'] < df['Close']) |
        (df['Low']  > df['Open']) | (df['Low']  > df['Close']) |
        (df['High'] <= 0) | (df['Close'] <= 0)
    )
    return not invalid.any()

@st.cache_data(ttl=60, show_spinner=False)
def fetch_live_data(symbol: str, start_date: date, end_date: date) -> pd.DataFrame:
    """Robust 30-min intraday fetch for Yahoo Finance; CT index."""
    def _normalize(df: pd.DataFrame) -> pd.DataFrame:
        if df.empty: return df
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0] if isinstance(c, tuple) else str(c) for c in df.columns]
        if df.index.tz is None:
            df.index = df.index.tz_localize('US/Eastern')
        df.index = df.index.tz_convert(CT_TZ)
        sdt = CT_TZ.localize(datetime.combine(start_date, time(0,0)))
        edt = CT_TZ.localize(datetime.combine(end_date,   time(23,59)))
        return df.loc[sdt:edt]

    try:
        t = yf.Ticker(symbol)
        df = t.history(
            start=(start_date - timedelta(days=3)).strftime('%Y-%m-%d'),
            end=(end_date + timedelta(days=2)).strftime('%Y-%m-%d'),
            interval="30m",
            prepost=True, auto_adjust=False, back_adjust=False
        )
        df = _normalize(df)
        if df.empty:
            # fallback by period
            days = max(7, (end_date - start_date).days + 7)
            df2 = t.history(period=f"{days}d", interval="30m",
                            prepost=True, auto_adjust=False, back_adjust=False)
            df = _normalize(df2)
        if not validate_ohlc(df):
            return pd.DataFrame()
        return df
    except Exception:
        return pd.DataFrame()

def get_session_window(df: pd.DataFrame, start_time: str, end_time: str) -> pd.DataFrame:
    if df.empty: return df
    return df.between_time(start_time, end_time)

def get_daily_ohlc(df: pd.DataFrame, target_date: date) -> Dict:
    if df.empty: return {}
    sdt = CT_TZ.localize(datetime.combine(target_date, time(0,0)))
    edt = CT_TZ.localize(datetime.combine(target_date, time(23,59)))
    day = df.loc[sdt:edt]
    if day.empty: return {}
    open_ = day.iloc[0]['Open']
    high  = day['High'].max(); t_high = day[day['High']==high].index[0]
    low   = day['Low'].min();  t_low  = day[day['Low']==low].index[0]
    close = day.loc[day.index == CT_TZ.localize(datetime.combine(target_date, time(15,0))), 'Close']
    if close.empty:
        # fallback: nearest bar in [14:30, 15:00]
        late = day.between_time("14:30","15:00")
        close_val = late.iloc[-1]['Close'] if not late.empty else day.iloc[-1]['Close']
        t_close   = late.index[-1] if not late.empty else day.index[-1]
    else:
        close_val = float(close.iloc[0]); t_close = CT_TZ.localize(datetime.combine(target_date, time(15,0)))
    return {'open':(open_, day.index[0]),
            'high':(high, t_high),
            'low': (low,  t_low),
            'close':(close_val, t_close)}

# ───────────────────────────────────────────────────────────────────────────────
# BLOCK MATH (3:00 PM CT anchor → next-day RTH slots), skipping 1h maintenance
# ───────────────────────────────────────────────────────────────────────────────
def blocks_from_3pm_to(time_label: str) -> int:
    """
    Returns the number of 30-min blocks from 3:00 PM CT (previous trading day)
    to a given next-day RTH time label (HH:MM).
    Maintenance (3:30–4:30 PM) = 2 blocks removed.
    Mapping: 08:30 → 33, 09:00 → 34, ..., 14:30 → 45.
    """
    base = 33  # 3:00 → 08:30 next day
    hh, mm = map(int, time_label.split(":"))
    steps_after_830 = ((hh - 8) * 60 + (mm - 30)) // 30
    return base + max(0, steps_after_830)

# ───────────────────────────────────────────────────────────────────────────────
# FAN PROJECTIONS
# ───────────────────────────────────────────────────────────────────────────────
def project_close_fan(close_price: float, target_date: date, slope: float) -> pd.DataFrame:
    rows=[]
    for t in rth_slots_ct(target_date):
        label = format_ct_time(t)
        n = blocks_from_3pm_to(label)
        top = close_price + slope * n
        bot = close_price - slope * n
        rows.append({
            "Time": label,
            "Top": round(top,2),
            "Bottom": round(bot,2),
            "Width": round(top - bot, 2)
        })
    return pd.DataFrame(rows)

def project_line_from_anchor(anchor_price: float, anchor_time_label: str, target_date: date, slope: float, col_name:str) -> pd.DataFrame:
    """
    Generic line with the same slope rule: price + slope * blocks_from_3pm_to(time)
    Uses the time-of-day label of the anchor (e.g. '15:00', '14:10' rounded logic not needed since anchor is 15:00 for OHLC).
    For High (+slope) / Low (−slope) we still use the same block count function.
    """
    anchor_blocks = blocks_from_3pm_to(anchor_time_label)  # 15:00 → 33 base for next day; here anchor is 15:00, so 33 at 08:30, etc.
    rows=[]
    for t in rth_slots_ct(target_date):
        label = format_ct_time(t)
        n = blocks_from_3pm_to(label) - anchor_blocks
        price = anchor_price + slope * n
        rows.append({"Time": label, col_name: round(price,2)})
    return pd.DataFrame(rows)

def df_lookup(df: pd.DataFrame, price_col: str) -> Dict[str,float]:
    return {row['Time']: row[price_col] for _,row in df[['Time',price_col]].iterrows()}

def build_strategy_table(spx_rth: pd.DataFrame, fan_df: pd.DataFrame,
                         high_up_df: pd.DataFrame, low_dn_df: pd.DataFrame,
                         close_anchor: float) -> pd.DataFrame:
    if spx_rth.empty or fan_df.empty:
        return pd.DataFrame()

    price_at = {format_ct_time(ix): spx_rth.loc[ix, 'Close'] for ix in spx_rth.index}
    top_at   = df_lookup(fan_df, 'Top')
    bot_at   = df_lookup(fan_df, 'Bottom')
    high_up  = df_lookup(high_up_df, 'High_Asc') if not high_up_df.empty else {}
    low_dn   = df_lookup(low_dn_df, 'Low_Desc')   if not low_dn_df.empty else {}

    rows=[]
    for t in fan_df['Time']:
        if t not in price_at: 
            continue

        p   = price_at[t]
        top = top_at.get(t, np.nan)
        bot = bot_at.get(t, np.nan)
        width = (top - bot) if (pd.notna(top) and pd.notna(bot)) else np.nan

        # Bias by fan: above = UP, below = DOWN, within = RANGE
        if pd.isna(top) or pd.isna(bot):
            bias = "—"
        elif p > top:
            bias = "UP"
        elif p < bot:
            bias = "DOWN"
        else:
            bias = "RANGE"

        # Strategy per your rules (concise)
        entry_side = ""; entry = tp1 = tp2 = np.nan; note = ""

        if bias == "RANGE":
            # Within fan → trade back to opposite edge
            if p - ((top+bot)/2) >= 0:  # leaning high
                entry_side = "SELL"; entry = top; tp1 = bot; tp2 = bot; note = "Within fan → sell top → exit bottom"
            else:
                entry_side = "BUY";  entry = bot; tp1 = top; tp2 = top; note = "Within fan → buy bottom → exit top"

        elif bias == "UP":
            # Above fan → entry = Top; TP2 = Top - width; TP1 = ascending High line
            entry_side = "SELL"; entry = top
            tp2 = (top - width) if pd.notna(width) else np.nan
            tp1 = high_up.get(t, np.nan)
            note = "Above fan → entry at Top; TP1=High+; TP2=Top−Width"

        elif bias == "DOWN":
            # Below fan → entry = Bottom; TP2 = Bottom - width; TP1 = descending Low line
            entry_side = "SELL"; entry = bot
            tp2 = (bot - width) if pd.notna(width) else np.nan
            tp1 = low_dn.get(t, np.nan)
            note = "Below fan → entry at Bottom; TP1=Low−; TP2=Bottom−Width"

        rows.append({
            "Time": t,
            "Price": round(p,2),
            "Bias": bias,
            "EntrySide": entry_side,
            "Entry": round(entry,2) if pd.notna(entry) else np.nan,
            "TP1": round(tp1,2) if pd.notna(tp1) else np.nan,
            "TP2": round(tp2,2) if pd.notna(tp2) else np.nan,
            "Top": round(top,2) if pd.notna(top) else np.nan,
            "Bottom": round(bot,2) if pd.notna(bot) else np.nan,
            "Width": round(width,2) if pd.notna(width) else np.nan,
            "Note": note
        })
    return pd.DataFrame(rows)

# Indicators used in Signals tab
def calculate_ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span).mean()

def calculate_average_true_range(data: pd.DataFrame, periods: int = 14) -> pd.Series:
    if data.empty or len(data) < periods: return pd.Series(index=data.index, dtype=float)
    hl = data['High'] - data['Low']
    hc = np.abs(data['High'] - data['Close'].shift())
    lc = np.abs(data['Low']  - data['Close'].shift())
    tr = np.maximum(hl, np.maximum(hc, lc))
    return tr.rolling(window=periods).mean()

def calculate_market_volatility(data: pd.DataFrame) -> float:
    if data.empty or len(data) < 2: return 0.0
    r = data['Close'].pct_change().dropna()
    if r.empty: return 0.0
    return float(r.std() * np.sqrt(390) * 100)

# ───────────────────────────────────────────────────────────────────────────────
# SIDEBAR — GLOBAL CONTROLS
# ───────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 📊 SPX Prophet Controls")
    today_ct = datetime.now(CT_TZ).date()

    prev_day = st.date_input("Previous Trading Day", value=today_ct - timedelta(days=1), key="prev_day")
    proj_day = st.date_input("Projection Day", value=prev_day + timedelta(days=1), key="proj_day")

    slope = st.number_input("Slope per 30-min block", value=DEFAULT_SLOPE, step=0.001, format="%.3f", help="Top = +slope × blocks; Bottom = −slope × blocks")

    st.markdown("---")
    st.markdown("### 🧪 Slope Calibrator (optional)")
    st.caption("Use Yahoo 3:00 PM CT close + a desired time/price to compute a tuned slope.")
    calib_close = st.number_input("Yahoo 3:00 PM CT Close", value=0.0, step=0.01, format="%.2f", key="calib_close")
    calib_time  = st.selectbox("Target Time (CT)", ["08:30","09:00","09:30","10:00","10:30","11:00","11:30","12:00","12:30","13:00","13:30","14:00","14:30"], index=0)
    calib_price = st.number_input("Desired Price @ Target Time", value=0.0, step=0.01, format="%.2f", key="calib_price")
    if st.button("Compute Tuned Slope"):
        if calib_close > 0 and calib_price > 0:
            n = blocks_from_3pm_to(calib_time)
            tuned = (calib_price - calib_close) / n
            st.success(f"Suggested slope: {tuned:+.3f} per 30-min")
        else:
            st.info("Enter positive values for close and target price.")

# ───────────────────────────────────────────────────────────────────────────────
# HEADER METRICS
# ───────────────────────────────────────────────────────────────────────────────
now = datetime.now(CT_TZ)
colA, colB, colC = st.columns(3)
with colA:
    st.markdown(f"""
    <div class="metricCard">
      <div class="metricTitle">Current Time (CT)</div>
      <div class="metricValue">🕒 {now.strftime("%H:%M:%S")}</div>
      <div style="color:#6b7280;">{now.strftime("%A, %B %d, %Y")}</div>
    </div>
    """, unsafe_allow_html=True)
with colB:
    is_weekday = now.weekday() < 5
    rth_open = now.replace(hour=8,minute=30,second=0,microsecond=0)
    rth_close= now.replace(hour=14,minute=30,second=0,microsecond=0)
    open_now = is_weekday and (rth_open <= now <= rth_close)
    status = "📗 MARKET OPEN" if open_now else ("📙 MARKET CLOSED" if is_weekday else "📕 WEEKEND")
    color  = "good" if open_now else ("warn" if is_weekday else "bad")
    st.markdown(f"""
    <div class="metricCard">
      <div class="metricTitle">Status</div>
      <div class="metricValue {color}">{status}</div>
      <div style="color:#6b7280;">RTH: 08:30 – 14:30 CT</div>
    </div>
    """, unsafe_allow_html=True)
with colC:
    st.markdown(f"""
    <div class="metricCard">
      <div class="metricTitle">Slope per 30-min</div>
      <div class="metricValue">📐 {slope:+.3f}</div>
      <div style="color:#6b7280;">Top = +slope • Bottom = −slope</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<div class='glass'></div>", unsafe_allow_html=True)

# ───────────────────────────────────────────────────────────────────────────────
# TABS
# ───────────────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["🔰 SPX Anchors", "🏢 Stock Anchors", "📶 Signals & EMA", "🎯 Contract Tool"])

# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║ TAB 1: SPX ANCHORS — Close-Anchor Fan + Strategy                           ║
# ╚═════════════════════════════════════════════════════════════════════════════╝
with tab1:
    st.subheader("SPX Close-Anchor Fan")
    st.caption("Anchored at the previous trading day’s 3:00 PM CT close. Maintenance hour excluded (3:30–4:30 PM).")

    # Pull SPX (^GSPC) for prev & proj days (SPX-only display)
    spx_prev = fetch_live_data("^GSPC", prev_day, prev_day)
    spx_proj = fetch_live_data("^GSPC", proj_day, proj_day)

    if spx_prev.empty or spx_proj.empty:
        st.error("Live market data not available for the selected dates.")
    else:
        # RTH slice for projection day
        spx_proj_rth = get_session_window(spx_proj, RTH_START, RTH_END)

        # Extract precise 3:00 PM CT close for prev day
        dprev = get_daily_ohlc(spx_prev, prev_day)
        if not dprev or 'close' not in dprev:
            st.error("Could not determine the 3:00 PM CT close for the previous day.")
        else:
            close_px, close_t = dprev['close']
            high_px, high_t = dprev['high']
            low_px,  low_t  = dprev['low']

            # Build fan (+/- slope * blocks_from_3pm)
            fan_df = project_close_fan(close_px, proj_day, slope)

            # High ascending (TP1 reference when above fan)
            high_up_df = project_line_from_anchor(high_px, "15:00", proj_day, +slope, "High_Asc")

            # Low descending (TP1 reference when below fan)
            low_dn_df  = project_line_from_anchor(low_px,  "15:00", proj_day, -slope, "Low_Desc")

            # Strategy table (concise)
            strat_df = build_strategy_table(spx_proj_rth, fan_df, high_up_df, low_dn_df, close_px)

            # Display
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("#### 🧭 Fan Lines")
                st.dataframe(fan_df, use_container_width=True, hide_index=True)
            with c2:
                st.markdown("#### 🎯 Strategy Summary")
                st.dataframe(strat_df, use_container_width=True, hide_index=True)

            # Quick bullets for clarity
            st.markdown("""
            <div class="glass">
                <b>How to read:</b>
                <ul>
                    <li><b>Bias</b>: Above Top → <b>UP</b>, Below Bottom → <b>DOWN</b>, inside → <b>RANGE</b>.</li>
                    <li><b>Within fan</b>: trade to the opposite edge (buy bottom → exit top; sell top → exit bottom).</li>
                    <li><b>Above fan</b>: entry at Top; <b>TP1</b>=High ascending, <b>TP2</b>=Top−Width.</li>
                    <li><b>Below fan</b>: entry at Bottom; <b>TP1</b>=Low descending, <b>TP2</b>=Bottom−Width.</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║ TAB 2: STOCK ANCHORS — Simple Mon/Tue combine & preview                    ║
# ╚═════════════════════════════════════════════════════════════════════════════╝
with tab2:
    st.subheader("Stock Anchors (Mon/Tue)")
    st.caption("Quick pull & combined preview (30-min).")

    # Ticker pickers
    core_tickers = list(STOCK_SLOPES_DEFAULT.keys())
    tcols = st.columns(4)
    chosen = st.session_state.get("stock_ticker", None)
    for i, tkr in enumerate(core_tickers):
        if tcols[i%4].button(f"📈 {tkr}", key=f"stkbtn_{tkr}"):
            chosen = tkr
            st.session_state["stock_ticker"] = chosen

    custom = st.text_input("Custom Symbol", placeholder="Enter any ticker symbol (e.g. AMD)")
    if custom:
        chosen = custom.upper()
        st.session_state["stock_ticker"] = chosen

    if chosen:
        # Defaults for slope controls (not used for SPX fan)
        st.caption(f"Default slope for {chosen}: {STOCK_SLOPES_DEFAULT.get(chosen, 0.0150):.4f}")

        left, mid, right = st.columns(3)
        with left:
            monday  = st.date_input("Monday Date", value=(now.date() - timedelta(days=2)))
        with mid:
            tuesday = st.date_input("Tuesday Date", value=(monday + timedelta(days=1)))
        with right:
            st.caption(f"Wed: {tuesday+timedelta(days=1)} · Thu: {tuesday+timedelta(days=2)} · Fri: {tuesday+timedelta(days=3)}")

        if st.button(f"Analyze {chosen}", type="primary"):
            mon = fetch_live_data(chosen, monday, monday)
            tue = fetch_live_data(chosen, tuesday, tuesday)
            if mon.empty and tue.empty:
                st.error("No data for selected dates.")
            else:
                combined = mon if tue.empty else (tue if mon.empty else pd.concat([mon, tue]).sort_index())
                st.dataframe(combined.tail(48), use_container_width=True)

# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║ TAB 3: SIGNALS & EMA (RTH, EMA 8/21, concise)                              ║
# ╚═════════════════════════════════════════════════════════════════════════════╝
with tab3:
    st.subheader("Signals & EMA (RTH)")
    c1, c2 = st.columns(2)
    with c1:
        sig_symbol = st.selectbox("Symbol", ["^GSPC","ES=F","SPY"], index=0)
    with c2:
        sig_day = st.date_input("Analysis Day", value=now.date())

    if st.button("Analyze Signals", type="primary"):
        data = fetch_live_data(sig_symbol, sig_day, sig_day)
        rth  = get_session_window(data, RTH_START, RTH_END)
        if rth.empty:
            st.error("No RTH data for selected day.")
        else:
            ema8  = calculate_ema(rth['Close'], 8)
            ema21 = calculate_ema(rth['Close'], 21)
            out=[]
            for i in range(1,len(rth)):
                t  = format_ct_time(rth.index[i])
                p  = rth['Close'].iloc[i]
                p8 = ema8.iloc[i]; p21 = ema21.iloc[i]
                p8p= ema8.iloc[i-1]; p21p=ema21.iloc[i-1]
                cross = "Bullish Cross" if (p8p<=p21p and p8>p21) else ("Bearish Cross" if (p8p>=p21p and p8<p21) else "None")
                sep = abs(p8-p21)/p21*100 if p21!=0 else 0
                regime = "Bullish" if p8>p21 else "Bearish"
                out.append({"Time":t,"Price":round(p,2),"EMA8":round(p8,2),"EMA21":round(p21,2),
                            "Separation_%":round(sep,3),"Regime":regime,"Crossover":cross})
            st.dataframe(pd.DataFrame(out), use_container_width=True, hide_index=True)
            vol = calculate_market_volatility(rth)
            atr = calculate_average_true_range(rth,14)
            m1, m2 = st.columns(2)
            with m1: st.metric("Volatility", f"{vol:.2f}%")
            with m2: st.metric("ATR (14)", f"{(atr.iloc[-1] if not atr.empty else 0):.2f}")

# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║ TAB 4: CONTRACT TOOL (Overnight → RTH projection)                           ║
# ╚═════════════════════════════════════════════════════════════════════════════╝
with tab4:
    st.subheader("Contract Tool")
    c1, c2 = st.columns(2)
    with c1:
        p1_date = st.date_input("Point 1 Date", value=(now.date()-timedelta(days=1)))
        p1_time = st.time_input("Point 1 Time (CT)", value=time(20,0))
        p1_price= st.number_input("Point 1 Price", value=10.00, min_value=0.01, step=0.01, format="%.2f")
    with c2:
        p2_date = st.date_input("Point 2 Date", value=now.date())
        p2_time = st.time_input("Point 2 Time (CT)", value=time(8,0))
        p2_price= st.number_input("Point 2 Price", value=12.00, min_value=0.01, step=0.01, format="%.2f")

    proj_day_ct = st.date_input("RTH Projection Day", value=p2_date)
    p1_dt = CT_TZ.localize(datetime.combine(p1_date, p1_time))
    p2_dt = CT_TZ.localize(datetime.combine(p2_date, p2_time))

    if p2_dt <= p1_dt:
        st.error("Point 2 must be after Point 1.")
    else:
        blocks = (p2_dt - p1_dt).total_seconds() / 1800
        slope_ct = (p2_price - p1_price) / blocks if blocks>0 else 0.0
        if st.button("Project Contract Path", type="primary"):
            rows=[]
            for t in rth_slots_ct(proj_day_ct):
                b = (t - p1_dt).total_seconds()/1800
                price = p1_price + slope_ct * b
                rows.append({"Time": format_ct_time(t), "Contract_Price": round(price,2)})
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)