# app.py
# ═══════════════════════════════════════════════════════════════════════════════
# 🔮 SPX PROPHET — FULL APP (Weekends & Daily Maintenance Aware)
#  • Fan anchored at previous-day 3:00 PM CT close
#  • Slope ± per 30-min block (default 0.277, adjustable in sidebar)
#  • Weekend-aware half-hour block counting:
#       - Closed daily 16:00–17:00 CT
#       - Closed Fri 16:00 → Sun 17:00 CT
#  • SPX tables only (SPY fallback) to show the fan & strategy
#  • Light-mode, glass-neomorphism UI (scoped, safe)
# ═══════════════════════════════════════════════════════════════════════════════

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import pytz
from datetime import datetime, date, time, timedelta
from typing import Dict, List, Optional

# ───────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG (must be first Streamlit call)
# ───────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="🔮 SPX Prophet Analytics",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ───────────────────────────────────────────────────────────────────────────────
# SAFE GLASS-NEOMORPHISM (scoped styles)
# ───────────────────────────────────────────────────────────────────────────────
def inject_styles():
    st.markdown("""
    <style>
    :root {
      --glass-bg: rgba(255,255,255,0.70);
      --glass-stroke: rgba(0,0,0,0.06);
      --glass-shadow: 0 16px 40px rgba(0,0,0,0.10);
      --ink: #0f172a;
      --ok: #16a34a;
      --warn: #f59e0b;
      --bad: #ef4444;
      --violet: #7c3aed;
      --cyan: #06b6d4;
      --amber: #f59e0b;
    }
    .app-shell { padding: 0.4rem; }
    .glass-card {
      background: var(--glass-bg);
      border: 1px solid var(--glass-stroke);
      border-radius: 16px;
      backdrop-filter: blur(10px);
      -webkit-backdrop-filter: blur(10px);
      box-shadow: var(--glass-shadow);
      padding: 1rem 1.25rem;
      margin-bottom: 0.75rem;
      color: var(--ink);
    }
    .hero {
      background: linear-gradient(135deg, rgba(124,58,237,0.10), rgba(6,182,212,0.10));
      border: 1px solid rgba(124,58,237,0.25);
      border-radius: 20px;
      padding: 1.25rem 1.4rem;
    }
    .hero h1 {
      font-size: 2.1rem;
      margin: 0;
      background: linear-gradient(135deg, var(--violet), var(--cyan), var(--amber));
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
      background-clip: text;
    }
    .metric-tile { display: grid; gap: 0.25rem; }
    .metric-title { font-size: 0.85rem; opacity: 0.8; }
    .metric-value { font-size: 1.5rem; font-weight: 700; }
    .metric-ok { color: var(--ok); }
    .metric-warn { color: var(--warn); }
    .metric-bad { color: var(--bad); }
    .glass-table .stDataFrame {
      background: rgba(255,255,255,0.9);
      border-radius: 12px;
      border: 1px solid rgba(0,0,0,0.06);
      box-shadow: 0 4px 16px rgba(0,0,0,0.06);
    }
    .tab-pad { padding-top: 0.35rem; }
    section[data-testid="stSidebar"] .glass-card { background: rgba(255,255,255,0.80); }
    .btn-row { margin-top: 0.25rem; }
    </style>
    """, unsafe_allow_html=True)

inject_styles()

# ───────────────────────────────────────────────────────────────────────────────
# CORE CONFIG
# ───────────────────────────────────────────────────────────────────────────────
CT_TZ = pytz.timezone('America/Chicago')
RTH_START = "08:30"  # CT
RTH_END   = "14:30"  # CT
DEFAULT_SLOPE_PER_BLOCK = 0.277  # + upward, - downward (user adjustable in sidebar)

# ───────────────────────────────────────────────────────────────────────────────
# UTILITIES: time/session helpers
# ───────────────────────────────────────────────────────────────────────────────
def format_ct_time(dt: datetime) -> str:
    if dt.tzinfo is None:
        dt = CT_TZ.localize(dt)
    else:
        dt = dt.astimezone(CT_TZ)
    return dt.strftime("%H:%M")

def rth_slots_ct(target_date: date) -> List[datetime]:
    """All 30m timestamps for RTH in CT for target_date"""
    start_dt = CT_TZ.localize(datetime.combine(target_date, time(8, 30)))
    end_dt   = CT_TZ.localize(datetime.combine(target_date, time(14, 30)))
    slots = []
    cur = start_dt
    while cur <= end_dt:
        slots.append(cur)
        cur += timedelta(minutes=30)
    return slots

def is_es_like_open(dt: datetime) -> bool:
    """
    Returns True if ES-like session is open at dt (CT):
    - Closed daily 16:00–17:00 CT
    - Friday close at 16:00 CT → Reopen Sunday 17:00 CT
    """
    dt = dt.astimezone(CT_TZ)
    wd = dt.weekday()  # Mon=0..Sun=6
    h  = dt.hour + dt.minute/60.0

    # Saturday fully closed
    if wd == 5:
        return False
    # Sunday: open from 17:00
    if wd == 6:
        return h >= 17.0
    # Friday: closed at/after 16:00
    if wd == 4:
        return h < 16.0
    # Mon–Thu: closed 16:00–17:00
    return not (16.0 <= h < 17.0)

def halfhour_blocks_between_trading(anchor_time: datetime, target_time: datetime) -> int:
    """
    Count 30-min blocks between anchor_time (inclusive) and target_time (exclusive),
    counting only intervals that are within ES-like open times.
    """
    if target_time <= anchor_time:
        return 0
    cur = anchor_time.astimezone(CT_TZ)
    tgt = target_time.astimezone(CT_TZ)
    blocks = 0
    while cur < tgt:
        nxt = cur + timedelta(minutes=30)
        # Count this block if interval start is during open session
        if is_es_like_open(cur):
            blocks += 1
        cur = nxt
    return blocks

# ───────────────────────────────────────────────────────────────────────────────
# DATA FETCH & VALIDATION
# ───────────────────────────────────────────────────────────────────────────────
def _normalize_history(df: pd.DataFrame, start_date: date, end_date: date) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    # Flatten MultiIndex columns if needed
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] if isinstance(c, tuple) else str(c) for c in df.columns]
    required = ['Open','High','Low','Close','Volume']
    if any(c not in df.columns for c in required):
        return pd.DataFrame()
    # Yahoo returns in US/Eastern by default → convert to CT
    if df.index.tz is None:
        df.index = df.index.tz_localize('US/Eastern')
    df.index = df.index.tz_convert(CT_TZ)
    sdt = CT_TZ.localize(datetime.combine(start_date, time(0,0)))
    edt = CT_TZ.localize(datetime.combine(end_date,   time(23,59)))
    return df.loc[sdt:edt]

@st.cache_data(ttl=60)
def fetch_live_data(symbol: str, start_date: date, end_date: date) -> pd.DataFrame:
    """Reliable 30m intraday fetch with start/end, period fallback, CT index."""
    try:
        t = yf.Ticker(symbol)
        # Try buffered start/end first
        df = t.history(
            start=(start_date - timedelta(days=5)).strftime('%Y-%m-%d'),
            end=(end_date + timedelta(days=2)).strftime('%Y-%m-%d'),
            interval="30m", prepost=True, auto_adjust=False, back_adjust=False
        )
        df = _normalize_history(df, start_date, end_date)
        if df.empty:
            # Fallback to period
            days = max(7, (end_date - start_date).days + 7)
            df2 = t.history(period=f"{days}d", interval="30m", prepost=True, auto_adjust=False, back_adjust=False)
            df = _normalize_history(df2, start_date, end_date)
        return df
    except Exception:
        return pd.DataFrame()

def get_daily_ohlc(df: pd.DataFrame, target_date: date) -> Dict:
    if df.empty: return {}
    sdt = CT_TZ.localize(datetime.combine(target_date, time(0,0)))
    edt = CT_TZ.localize(datetime.combine(target_date, time(23,59)))
    day = df.loc[sdt:edt]
    if day.empty: return {}
    o = day.iloc[0]['Open']
    h = day['High'].max(); ht = day[day['High']==h].index[0]
    l = day['Low'].min();  lt = day[day['Low']==l].index[0]
    c = day.iloc[-1]['Close']; ct = day.index[-1]
    return {'open':(o,day.index[0]), 'high':(h,ht), 'low':(l,lt), 'close':(c,ct)}

def get_3pm_close(df: pd.DataFrame, target_date: date) -> Optional[float]:
    """
    Return the 3:00 PM CT close for target_date.
    If exact 15:00 bar is missing, use the nearest within ±30 minutes.
    """
    if df.empty: return None
    sdt = CT_TZ.localize(datetime.combine(target_date, time(0,0)))
    edt = CT_TZ.localize(datetime.combine(target_date, time(23,59)))
    day = df.loc[sdt:edt]
    if day.empty: return None

    # Try exact 15:00 CT
    mask_exact = (day.index.hour == 15) & (day.index.minute == 0)
    if mask_exact.any():
        return float(day.loc[mask_exact, 'Close'].iloc[0])

    # Nearest within ±30 minutes
    target_dt = CT_TZ.localize(datetime.combine(target_date, time(15,0)))
    deltas = (day.index - target_dt).to_series().abs()
    nearest = deltas.idxmin()
    if abs((nearest - target_dt).total_seconds()) <= 30*60:
        return float(day.loc[nearest, 'Close'])
    return None

# ───────────────────────────────────────────────────────────────────────────────
# PROJECTIONS (Fan + helper using trading-aware blocks)
# ───────────────────────────────────────────────────────────────────────────────
def project_line_trading(anchor_price: float, anchor_time: datetime, slope_per_block: float,
                         target_date: date, col_name: str) -> pd.DataFrame:
    rows=[]
    for t in rth_slots_ct(target_date):
        blocks = halfhour_blocks_between_trading(anchor_time, t)
        price = anchor_price + slope_per_block * blocks
        rows.append({'Time':format_ct_time(t), col_name: round(price,2), 'Blocks': int(blocks)})
    return pd.DataFrame(rows)

def project_close_fan(anchor_close: float, anchor_time: datetime, target_date: date,
                      slope_per_block: float) -> pd.DataFrame:
    top = project_line_trading(anchor_close, anchor_time, +slope_per_block, target_date, 'Top')
    bot = project_line_trading(anchor_close, anchor_time, -slope_per_block, target_date, 'Bottom')
    df = pd.merge(top[['Time','Top']], bot[['Time','Bottom']], on='Time', how='inner')
    df['Fan_Width'] = (df['Top'] - df['Bottom']).round(2)
    df['Mid'] = round(anchor_close,2)
    return df

# ───────────────────────────────────────────────────────────────────────────────
# STRATEGY TABLE (your rules)
# ───────────────────────────────────────────────────────────────────────────────
def build_strategy_from_fan(
    rth_prices: pd.DataFrame,      # RTH 30m prices (Close)
    fan_df: pd.DataFrame,          # from project_close_fan
    prev_high_df: Optional[pd.DataFrame],  # ascending from previous-day High
    prev_low_df: Optional[pd.DataFrame],   # descending from previous-day Low
    anchor_close_price: float
) -> pd.DataFrame:
    if rth_prices.empty or fan_df.empty:
        return pd.DataFrame()

    # Lookups by HH:MM
    price_lookup = {format_ct_time(ix): rth_prices.loc[ix, 'Close'] for ix in rth_prices.index}
    top_lookup   = {r['Time']: r['Top'] for _, r in fan_df.iterrows()}
    bot_lookup   = {r['Time']: r['Bottom'] for _, r in fan_df.iterrows()}
    high_up_lu   = {r['Time']: r['High_Asc'] for _, r in (prev_high_df or pd.DataFrame()).iterrows()} if prev_high_df is not None else {}
    low_dn_lu    = {r['Time']: r['Low_Desc']  for _, r in (prev_low_df  or pd.DataFrame()).iterrows()} if prev_low_df  is not None else {}

    rows=[]
    for t in fan_df['Time']:
        if t not in price_lookup: 
            continue
        p    = price_lookup[t]
        top  = top_lookup.get(t, np.nan)
        bot  = bot_lookup.get(t, np.nan)
        width= top - bot if pd.notna(top) and pd.notna(bot) else np.nan
        bias = "UP" if p >= anchor_close_price else "DOWN"

        direction=""; entry=np.nan; tp1=np.nan; tp2=np.nan; note=""
        within_fan = (pd.notna(top) and pd.notna(bot) and bot <= p <= top)
        above_fan  = (pd.notna(top) and p > top)
        below_fan  = (pd.notna(bot) and p < bot)

        if within_fan:
            if bias=="UP":
                direction="BUY";  entry = bot; tp1 = top; tp2 = top
                note  = "Within fan; bias UP → buy bottom → exit top"
            else:
                direction="SELL"; entry = top; tp1 = bot; tp2 = bot
                note  = "Within fan; bias DOWN → sell top → exit bottom"

        elif above_fan:
            direction="SELL"; entry = top
            tp2   = max(bot, entry - width) if pd.notna(width) else bot
            tp1   = high_up_lu.get(t, np.nan)  # ascending from prev high
            note  = "Above fan; entry=Top; TP2=Top-width; TP1=High ascending"

        elif below_fan:
            direction="SELL"; entry = bot
            tp2   = entry - width if pd.notna(width) else np.nan
            tp1   = low_dn_lu.get(t, np.nan)   # descending from prev low
            note  = "Below fan; entry=Bottom; TP2=Bottom-width; TP1=Low descending"

        rows.append({
            'Time': t,
            'Price': round(p,2),
            'Bias': bias,
            'EntrySide': direction,
            'Entry': round(entry,2) if pd.notna(entry) else np.nan,
            'TP1': round(tp1,2) if pd.notna(tp1) else np.nan,
            'TP2': round(tp2,2) if pd.notna(tp2) else np.nan,
            'Top': round(top,2) if pd.notna(top) else np.nan,
            'Bottom': round(bot,2) if pd.notna(bot) else np.nan,
            'Fan_Width': round(width,2) if pd.notna(width) else np.nan,
            'Note': note
        })
    return pd.DataFrame(rows)

# ───────────────────────────────────────────────────────────────────────────────
# EXTRA (EMA / Volatility) — concise versions
# ───────────────────────────────────────────────────────────────────────────────
def calculate_ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span).mean()

def calculate_average_true_range(data: pd.DataFrame, periods: int = 14) -> pd.Series:
    if data.empty or len(data) < periods: return pd.Series(index=data.index, dtype=float)
    hl = data['High']-data['Low']
    hc = np.abs(data['High']-data['Close'].shift())
    lc = np.abs(data['Low'] -data['Close'].shift())
    tr = np.maximum(hl, np.maximum(hc, lc))
    return tr.rolling(window=periods).mean()

def calculate_market_volatility(data: pd.DataFrame) -> float:
    if data.empty or len(data) < 2: return 0.0
    r = data['Close'].pct_change().dropna()
    if r.empty: return 0.0
    vol = r.std()*np.sqrt(390)
    return vol*100

# ───────────────────────────────────────────────────────────────────────────────
# SIDEBAR — Inputs (single form with one button)
# ───────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("### 📅 Inputs")
    with st.form("dates_form", clear_on_submit=False):
        today_ct = datetime.now(CT_TZ).date()
        prev_day = st.date_input("Previous Trading Day", value=today_ct - timedelta(days=1), key="prev_day")
        proj_day = st.date_input("Projection Day", value=prev_day + timedelta(days=1), key="proj_day")
        slope_input = st.number_input("Slope per 30m (±)", value=DEFAULT_SLOPE_PER_BLOCK, step=0.001, format="%.3f", key="slope")
        submit = st.form_submit_button("Apply & Generate")
    st.markdown('</div>', unsafe_allow_html=True)

# ───────────────────────────────────────────────────────────────────────────────
# HEADER — Hero + Metrics
# ───────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="app-shell">', unsafe_allow_html=True)
st.markdown('<div class="glass-card hero">', unsafe_allow_html=True)
st.markdown("""
<h1>SPX Prophet Analytics</h1>
<p>Close-anchor fan with weekend-aware block counting and ± slope per 30-min. SPX tables only (SPY fallback).</p>
""", unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

c1, c2, c3 = st.columns(3)
now_ct = datetime.now(CT_TZ)
with c1:
    st.markdown('<div class="glass-card metric-tile">', unsafe_allow_html=True)
    st.markdown('<div class="metric-title">Current Time (CT)</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="metric-value">{now_ct.strftime("%H:%M:%S")}</div>', unsafe_allow_html=True)
    st.caption(now_ct.strftime("%A, %B %d"))
    st.markdown('</div>', unsafe_allow_html=True)

with c2:
    is_weekday = now_ct.weekday() < 5
    market_open = now_ct.replace(hour=8, minute=30, second=0, microsecond=0)
    market_close = now_ct.replace(hour=14, minute=30, second=0, microsecond=0)
    is_rth = is_weekday and (market_open <= now_ct <= market_close)
    status_class = "metric-ok" if is_rth else ("metric-warn" if is_weekday else "metric-bad")
    st.markdown('<div class="glass-card metric-tile">', unsafe_allow_html=True)
    st.markdown('<div class="metric-title">Market Status</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="metric-value {status_class}">{"OPEN" if is_rth else ("CLOSED" if is_weekday else "WEEKEND")}</div>', unsafe_allow_html=True)
    st.caption("RTH: 08:30–14:30 CT, Mon–Fri")
    st.markdown('</div>', unsafe_allow_html=True)

with c3:
    st.markdown('<div class="glass-card metric-tile">', unsafe_allow_html=True)
    st.markdown('<div class="metric-title">Slope per 30m</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="metric-value">{slope_input:+.3f}</div>', unsafe_allow_html=True)
    st.caption("Ascending (+) / Descending (−)")
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown('---')

# ───────────────────────────────────────────────────────────────────────────────
# TABS
# ───────────────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["SPX Anchors", "Stock Anchors", "Signals & EMA", "Contract Tool"])

# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║ TAB 1: SPX ANCHORS — Fan + Strategy (SPX tables only)                       ║
# ╚═════════════════════════════════════════════════════════════════════════════╝
with tab1:
    st.markdown('<div class="tab-pad">', unsafe_allow_html=True)
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("SPX Close-Anchor Fan")
    st.caption("Fan anchored at previous-day 3:00 PM CT close. Weekend & daily maintenance handled in block counts.")
    st.markdown('</div>', unsafe_allow_html=True)

    if submit:
        with st.spinner("Fetching market data & building fan..."):
            # 1) Get previous day SPX (fallback SPY)
            spx_prev = fetch_live_data("^GSPC", prev_day, prev_day)
            if spx_prev.empty:
                spx_prev = fetch_live_data("SPY", prev_day, prev_day)

            spx_proj = fetch_live_data("^GSPC", proj_day, proj_day)
            if spx_proj.empty:
                spx_proj = fetch_live_data("SPY", proj_day, proj_day)

            if spx_prev.empty:
                st.error("No previous day data (SPX/SPY).")
            elif spx_proj.empty:
                st.error("No projection day data (SPX/SPY).")
            else:
                # 2) Find 3:00 PM CT close for previous day
                anchor_close = get_3pm_close(spx_prev, prev_day)
                if anchor_close is None:
                    daily = get_daily_ohlc(spx_prev, prev_day)
                    if daily and 'close' in daily:
                        anchor_close = float(daily['close'][0])
                    else:
                        st.error("Could not determine previous-day 3:00 PM CT close.")
                        st.stop()
                # Anchor time is 15:00 CT (regardless of source bar used)
                anchor_time = CT_TZ.localize(datetime.combine(prev_day, time(15,0)))

                # Previous-day High & Low for TP1 reference lines
                daily = get_daily_ohlc(spx_prev, prev_day)
                prev_high_price, prev_high_time = (daily['high'][0], daily['high'][1]) if daily else (anchor_close, anchor_time)
                prev_low_price,  prev_low_time  = (daily['low'][0],  daily['low'][1])  if daily else (anchor_close, anchor_time)

                # 3) Project fan (Top/Bottom) for projection day — trading-aware blocks
                fan_df = project_close_fan(anchor_close, anchor_time, proj_day, slope_input)

                # 4) Project helper TP1 lines
                prev_high_up = project_line_trading(prev_high_price, prev_high_time, +slope_input, proj_day, 'High_Asc')
                prev_low_dn  = project_line_trading(prev_low_price,  prev_low_time,  -slope_input, proj_day, 'Low_Desc')

                # 5) RTH prices for projection day
                rth_data = spx_proj.between_time(RTH_START, RTH_END)

                # 6) Strategy table
                strat_df = build_strategy_from_fan(
                    rth_data, fan_df, prev_high_up, prev_low_dn, anchor_close_price=anchor_close
                )

                # 7) Display
                st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                st.markdown("### Fan Lines")
                st.caption(f"Anchor (3:00 PM CT Close) = **{anchor_close:.2f}**, Slope ±**{slope_input:.3f}**/30m (trading-hours counted)")
                st.dataframe(fan_df, use_container_width=True, hide_index=True)
                st.markdown('</div>', unsafe_allow_html=True)

                st.markdown('<div class="glass-card glass-table">', unsafe_allow_html=True)
                st.markdown("### Strategy Table")
                st.caption("Bias is UP if current price ≥ anchor close; DOWN otherwise. Entries at fan edge, TP1 = previous-day reference line (when outside fan), TP2 = one fan width.")
                if not strat_df.empty:
                    st.dataframe(strat_df, use_container_width=True, hide_index=True)
                else:
                    st.info("No strategy rows were generated (time alignment gap).")
                st.markdown('</div>', unsafe_allow_html=True)

# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║ TAB 2: STOCK ANCHORS — concise preview                                      ║
# ╚═════════════════════════════════════════════════════════════════════════════╝
with tab2:
    st.markdown('<div class="tab-pad">', unsafe_allow_html=True)
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("Stock Anchors (Mon/Tue Combined) — Preview")
    st.caption("Quick view: fetch Mon/Tue 30-min data; preview last 24 bars (no fan, SPX-only fan strategy).")
    core = ['TSLA','NVDA','AAPL','MSFT','AMZN','GOOGL','META','NFLX']
    cols = st.columns(4)
    selected_ticker = st.session_state.get('stock_ticker', None)
    for i,t in enumerate(core):
        if cols[i%4].button(f"📈 {t}", key=f"btn_{t}"):
            selected_ticker = t
            st.session_state['stock_ticker'] = t
    custom = st.text_input("Custom Symbol", placeholder="Enter any ticker symbol")
    if custom:
        selected_ticker = custom.upper()
        st.session_state['stock_ticker'] = selected_ticker
    if selected_ticker:
        # find the Monday preceding/including prev_day
        mon = prev_day - timedelta(days=(prev_day.weekday())%7)
        tue = mon + timedelta(days=1)
        if st.button(f"Fetch {selected_ticker} Mon/Tue"):
            with st.spinner("Fetching..."):
                mon_df = fetch_live_data(selected_ticker, mon, mon)
                tue_df = fetch_live_data(selected_ticker, tue, tue)
                combined = mon_df if tue_df.empty else (tue_df if mon_df.empty else pd.concat([mon_df,tue_df]).sort_index())
                if combined.empty:
                    st.error("No data for selected dates.")
                else:
                    st.dataframe(combined.tail(24), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║ TAB 3: SIGNALS & EMA                                                         ║
# ╚═════════════════════════════════════════════════════════════════════════════╝
with tab3:
    st.markdown('<div class="tab-pad">', unsafe_allow_html=True)
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("Signals & EMA (8/21), RTH")
    c1, c2 = st.columns(2)
    with c1:
        symbol = st.selectbox("Symbol", ["^GSPC","ES=F","SPY"], index=0)
    with c2:
        sday = st.date_input("Analysis Day", value=proj_day)
    if st.button("Analyze Signals"):
        with st.spinner("Analyzing..."):
            data = fetch_live_data(symbol, sday, sday)
            rth = data.between_time(RTH_START, RTH_END)
            if rth.empty:
                st.error("No RTH data for selected day.")
            else:
                ema8  = calculate_ema(rth['Close'], 8)
                ema21 = calculate_ema(rth['Close'], 21)
                out=[]
                for i in range(1,len(rth)):
                    t = format_ct_time(rth.index[i])
                    p = rth['Close'].iloc[i]
                    prev8, prev21 = ema8.iloc[i-1], ema21.iloc[i-1]
                    c8, c21 = ema8.iloc[i], ema21.iloc[i]
                    cross=None
                    if prev8<=prev21 and c8>c21: cross="Bullish Cross"
                    elif prev8>=prev21 and c8<c21: cross="Bearish Cross"
                    sep = abs(c8-c21)/c21*100 if c21!=0 else 0
                    regime = "Bullish" if c8>c21 else "Bearish"
                    out.append({'Time':t,'Price':round(p,2),'EMA8':round(c8,2),'EMA21':round(c21,2),
                                'Separation_%':round(sep,3),'Regime':regime,'Crossover':cross or 'None'})
                st.dataframe(pd.DataFrame(out), use_container_width=True, hide_index=True)
                vol = calculate_market_volatility(rth)
                atr = calculate_average_true_range(rth,14)
                mc1, mc2 = st.columns(2)
                with mc1: st.metric("Volatility", f"{vol:.2f}%")
                with mc2: st.metric("ATR (14)", f"{(atr.iloc[-1] if not atr.empty else 0):.2f}")
    st.markdown('</div>', unsafe_allow_html=True)

# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║ TAB 4: CONTRACT TOOL                                                         ║
# ╚═════════════════════════════════════════════════════════════════════════════╝
with tab4:
    st.markdown('<div class="tab-pad">', unsafe_allow_html=True)
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("Contract Tool (Overnight 2-Point Slope → RTH projection)")
    pc1, pc2 = st.columns(2)
    with pc1:
        p1_date = st.date_input("Point 1 Date", value=prev_day)
        p1_time = st.time_input("Point 1 Time (CT)", value=time(20,0))
        p1_price= st.number_input("Point 1 Contract Price", value=10.0, min_value=0.01, step=0.01, format="%.2f")
    with pc2:
        p2_date = st.date_input("Point 2 Date", value=proj_day)
        p2_time = st.time_input("Point 2 Time (CT)", value=time(8,0))
        p2_price= st.number_input("Point 2 Contract Price", value=12.0, min_value=0.01, step=0.01, format="%.2f")
    proj_day_ct = st.date_input("RTH Projection Day", value=proj_day)

    if st.button("Analyze Contract Projections"):
        p1_dt = CT_TZ.localize(datetime.combine(p1_date, p1_time))
        p2_dt = CT_TZ.localize(datetime.combine(p2_date, p2_time))
        if p2_dt <= p1_dt:
            st.error("Point 2 must be after Point 1")
        else:
            # slope by trading-time blocks
            blocks = halfhour_blocks_between_trading(p1_dt, p2_dt)
            slope = (p2_price - p1_price)/blocks if blocks>0 else 0.0
            rows=[]
            for t in rth_slots_ct(proj_day_ct):
                b = halfhour_blocks_between_trading(p1_dt, t)
                price = p1_price + slope*b
                rows.append({'Time': format_ct_time(t), 'Contract_Price': round(price,2), 'Blocks': int(b)})
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    st.markdown('</div>', unsafe_allow_html=True)

# Footer shell close
st.markdown('</div>', unsafe_allow_html=True)