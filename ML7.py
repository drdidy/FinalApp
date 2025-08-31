# app.py
# ═══════════════════════════════════════════════════════════════════════════════
# 🔮 SPX PROPHET — FULL APP (All 6 Parts, Unified)
# Close-fan anchored strategy with Globex-aware blocks and slope = 0.267
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
# CORE CONFIG
# ───────────────────────────────────────────────────────────────────────────────
CT_TZ = pytz.timezone('America/Chicago')
UTC_TZ = pytz.UTC

RTH_START = "08:30"  # CT
RTH_END   = "14:30"  # CT

# Slope magnitude per 30-min block (ascending +, descending −)
SLOPE_PER_BLOCK = 0.267

SPX_ANCHOR_START = "17:00"  # kept for completeness
SPX_ANCHOR_END   = "19:30"

STOCK_SLOPES_DEFAULT = {
    "TSLA": 0.0285, "NVDA": 0.086, "AAPL": 0.0155,
    "MSFT": 0.0541, "AMZN": 0.0139, "GOOGL": 0.0122,
    "META": 0.0674, "NFLX": 0.0230
}

# ───────────────────────────────────────────────────────────────────────────────
# STREAMLIT PAGE
# ───────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="🔮 SPX Prophet Analytics",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Light mode minimalist styling
st.markdown("""
<style>
:root { --card-bg: #ffffff; --muted:#6b7280; }
.main block-container { padding-top: 1rem; }
.metric-container {
  background: var(--card-bg);
  border: 1px solid #eef2f7;
  border-radius: 14px;
  padding: 1rem 1.2rem;
  box-shadow: 0 2px 8px rgba(15,23,42,0.06);
}
.dataframe { border-radius: 10px; overflow: hidden; }
hr { border: none; height: 1px; background: #e5e7eb; margin: 1.2rem 0; }
</style>
""", unsafe_allow_html=True)

# ───────────────────────────────────────────────────────────────────────────────
# TIME / SESSION HELPERS (Globex-aware)
# ───────────────────────────────────────────────────────────────────────────────
def as_ct(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return CT_TZ.localize(dt)
    return dt.astimezone(CT_TZ)

def is_globex_open(dt_ct: datetime) -> bool:
    """CME ES schedule approximation in CT:
       - Open: Sun 17:00 → Fri 16:00 continuous
       - Daily maintenance close: 16:00–17:00 CT
       - Weekend close: Fri 16:00 → Sun 17:00
    """
    dt_ct = as_ct(dt_ct)
    wd = dt_ct.weekday()  # Mon=0 ... Sun=6
    t = dt_ct.time()

    # Daily maintenance close 16:00–17:00
    if time(16,0) <= t < time(17,0):
        return False

    # Weekend close: Fri 16:00 → Sun 17:00
    # If it's Saturday: closed all day
    if wd == 5:
        return False
    # If it's Friday and after/equal 16:00 → closed
    if wd == 4 and t >= time(16,0):
        return False
    # If it's Sunday and before 17:00 → closed
    if wd == 6 and t < time(17,0):
        return False

    return True

def active_30m_blocks_between(start_ct: datetime, end_ct: datetime) -> int:
    """Count 30-min blocks only during Globex open periods."""
    if end_ct <= start_ct:
        return 0
    cur = as_ct(start_ct)
    end_ct = as_ct(end_ct)
    blocks = 0
    step = timedelta(minutes=30)
    while cur < end_ct:
        nxt = cur + step
        # Count this block if midpoint is within open session
        mid = cur + timedelta(minutes=15)
        if is_globex_open(mid):
            blocks += 1
        cur = nxt
    return blocks

# ───────────────────────────────────────────────────────────────────────────────
# UTILS
# ───────────────────────────────────────────────────────────────────────────────
def rth_slots_ct(target_date: date) -> List[datetime]:
    start_dt = CT_TZ.localize(datetime.combine(target_date, time(8, 30)))
    end_dt   = CT_TZ.localize(datetime.combine(target_date, time(14, 30)))
    slots = []
    cur = start_dt
    while cur <= end_dt:
        slots.append(cur)
        cur += timedelta(minutes=30)
    return slots

def format_ct_time(dt: datetime) -> str:
    dt = as_ct(dt)
    return dt.strftime("%H:%M")

def get_session_window(df: pd.DataFrame, start_time: str, end_time: str) -> pd.DataFrame:
    if df.empty: return df
    return df.between_time(start_time, end_time)

def validate_ohlc_data(df: pd.DataFrame) -> bool:
    if df.empty: return False
    for c in ['Open','High','Low','Close']:
        if c not in df.columns: return False
    invalid = (
        (df['High'] < df['Low']) |
        (df['High'] < df['Open']) |
        (df['High'] < df['Close'])|
        (df['Low']  > df['Open']) |
        (df['Low']  > df['Close'])|
        (df['Close'] <= 0) | (df['High'] <= 0)
    )
    return not invalid.any()

@st.cache_data(ttl=90)
def fetch_live_data(symbol: str, start_date: date, end_date: date) -> pd.DataFrame:
    """Robust intraday fetch w/ fallback, 30-min interval, CT index, no auto-adjust."""
    def _normalize(df):
        if df.empty: return df
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0] if isinstance(c, tuple) else str(c) for c in df.columns]
        req = ['Open','High','Low','Close','Volume']
        if any(c not in df.columns for c in req):
            return pd.DataFrame()
        if df.index.tz is None:
            # Yahoo returns US/Eastern; localize if naive
            df.index = df.index.tz_localize('US/Eastern')
        df.index = df.index.tz_convert(CT_TZ)
        sdt = CT_TZ.localize(datetime.combine(start_date, time(0,0)))
        edt = CT_TZ.localize(datetime.combine(end_date,   time(23,59)))
        return df.loc[sdt:edt]

    try:
        t = yf.Ticker(symbol)
        # Try start/end with buffer
        df = t.history(
            start=(start_date - timedelta(days=5)).strftime('%Y-%m-%d'),
            end=(end_date + timedelta(days=2)).strftime('%Y-%m-%d'),
            interval="30m", prepost=True,
            auto_adjust=False, back_adjust=False
        )
        df = _normalize(df)

        # Fallback: period-based
        if df.empty:
            days = max(7, (end_date - start_date).days + 7)
            df2 = t.history(
                period=f"{days}d", interval="30m", prepost=True,
                auto_adjust=False, back_adjust=False
            )
            df = _normalize(df2)

        if df.empty:
            return pd.DataFrame()

        if not validate_ohlc_data(df):
            st.warning(f"⚠️ Data quality issues detected for {symbol}")
        return df

    except Exception as e:
        st.error(f"❌ Error fetching {symbol}: {e}")
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

def calculate_ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span).mean()

def calculate_vwap(df: pd.DataFrame) -> pd.Series:
    if df.empty or 'Volume' not in df.columns:
        return pd.Series(index=df.index, dtype=float)
    tp = (df['High']+df['Low']+df['Close'])/3
    cv = df['Volume'].cumsum()
    cvp = (tp*df['Volume']).cumsum()
    vwap = (cvp/cv).fillna(method='ffill').fillna(tp)
    return vwap

# ───────────────────────────────────────────────────────────────────────────────
# PROJECTIONS (Core to your logic) — Globex-aware blocks
# ───────────────────────────────────────────────────────────────────────────────
def project_line_globex(anchor_price: float, anchor_time: datetime, slope_per_block: float,
                        target_date: date, col_name: str) -> pd.DataFrame:
    """Project using Globex-aware block counting from anchor_time → each RTH 30m slot."""
    rows=[]
    anchor_ct = as_ct(anchor_time)
    for t in rth_slots_ct(target_date):
        blocks = active_30m_blocks_between(anchor_ct, t)
        price = anchor_price + slope_per_block*blocks
        rows.append({'Time':format_ct_time(t), col_name: round(price,2), 'Blocks': int(blocks)})
    return pd.DataFrame(rows)

def project_close_fan(close_price: float, close_time: datetime, target_date: date) -> pd.DataFrame:
    """Top = +SLOPE_PER_BLOCK from Close; Bottom = −SLOPE_PER_BLOCK from Close, Globex-aware."""
    top = project_line_globex(close_price, close_time, +SLOPE_PER_BLOCK, target_date, 'Top')
    bot = project_line_globex(close_price, close_time, -SLOPE_PER_BLOCK, target_date, 'Bottom')
    df = pd.merge(top[['Time','Top','Blocks']], bot[['Time','Bottom']], on='Time', how='inner')
    df['Fan_Width'] = (df['Top'] - df['Bottom']).round(2)
    df['Anchor_Close'] = round(close_price,2)
    return df

def df_to_time_price_lookup(df: pd.DataFrame, price_col: str) -> Dict[str,float]:
    if df is None or df.empty or price_col not in df.columns:
        return {}
    return {row['Time']: row[price_col] for _,row in df[['Time',price_col]].iterrows()}

def build_strategy_from_fan(
    rth_prices: pd.DataFrame,                # proj day ^GSPC 30m (RTH slice)
    fan_df: pd.DataFrame,                   # from project_close_fan
    high_up_df: Optional[pd.DataFrame],     # ascending line from prev day's High
    low_dn_df:  Optional[pd.DataFrame],     # descending line from prev day's Low
    anchor_close_price: float
) -> pd.DataFrame:
    if rth_prices.empty or fan_df.empty:
        return pd.DataFrame()

    price_lookup = {format_ct_time(ix): rth_prices.loc[ix, 'Close'] for ix in rth_prices.index}
    top_lookup   = df_to_time_price_lookup(fan_df, 'Top')
    bot_lookup   = df_to_time_price_lookup(fan_df, 'Bottom')
    high_up_lu   = df_to_time_price_lookup(high_up_df, 'High_Asc')
    low_dn_lu    = df_to_time_price_lookup(low_dn_df,  'Low_Desc')

    rows=[]
    for t in fan_df['Time']:
        if t not in price_lookup:
            continue
        p    = price_lookup[t]
        top  = top_lookup.get(t, np.nan)
        bot  = bot_lookup.get(t, np.nan)
        width= top - bot if pd.notna(top) and pd.notna(bot) else np.nan
        # Bias from close fan (as requested)
        bias = "UP" if p >= bot and p >= anchor_close_price else ("DOWN" if p <= top and p < anchor_close_price else "RANGE")

        # Simplified entries per your rules
        direction=""; entry=np.nan; tp1=np.nan; tp2=np.nan; note=""
        within_fan = (pd.notna(top) and pd.notna(bot) and bot <= p <= top)
        above_fan  = (pd.notna(top) and p > top)
        below_fan  = (pd.notna(bot) and p < bot)

        if within_fan:
            if p >= anchor_close_price:
                direction="BUY"; entry=bot; tp1=top; tp2=top
                note="Within fan; bias UP → buy bottom → exit top"
            else:
                direction="SELL"; entry=top; tp1=bot; tp2=bot
                note="Within fan; bias DOWN → sell top → exit bottom"
        elif above_fan:
            direction="SELL"
            entry = top
            tp2   = max(bot, entry - width) if pd.notna(width) else bot
            tp1   = high_up_lu.get(t, np.nan)  # ascending from prev-day HIGH
            note  = "Above fan; TP1=High ascending, TP2=Top-width"
        elif below_fan:
            direction="SELL"
            entry = bot
            tp2   = entry - width if pd.notna(width) else np.nan
            tp1   = low_dn_lu.get(t, np.nan)   # descending from prev-day LOW
            note  = "Below fan; TP1=Low descending, TP2=Bottom-width"

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

# Extra analytics used in Signals & Contract Tool (unchanged)
def calculate_average_true_range(data: pd.DataFrame, periods: int = 14) -> pd.Series:
    if data.empty or len(data) < periods: return pd.Series(index=data.index, dtype=float)
    hl = data['High']-data['Low']
    hc = np.abs(data['High']-data['Close'].shift())
    lc = np.abs(data['Low'] -data['Close'].shift())
    tr = np.maximum(hl, np.maximum(hc, lc))
    return tr.rolling(window=periods).mean()

def calculate_market_volatility(data: pd.DataFrame) -> float:
    if data.empty or len(data) < 2: return 1.5
    r = data['Close'].pct_change().dropna()
    if r.empty: return 1.5
    vol = r.std()*np.sqrt(390)
    return vol*100

# ───────────────────────────────────────────────────────────────────────────────
# SESSION STATE INIT
# ───────────────────────────────────────────────────────────────────────────────
if 'stock_slopes' not in st.session_state:
    st.session_state.stock_slopes = STOCK_SLOPES_DEFAULT.copy()

# ───────────────────────────────────────────────────────────────────────────────
# HEADER
# ───────────────────────────────────────────────────────────────────────────────
st.markdown("## 🔮 SPX Prophet Analytics")
c1,c2,c3 = st.columns(3)
with c1:
    now = datetime.now(CT_TZ)
    st.markdown(f"""
    <div class="metric-container">
        <div style="font-size:0.9rem;color:var(--muted)">Current Time (CT)</div>
        <div style="font-size:1.6rem;font-weight:700">{now.strftime("%H:%M:%S")}</div>
        <div style="color:var(--muted)">{now.strftime("%A, %B %d")}</div>
    </div>""", unsafe_allow_html=True)

with c2:
    wkday = now.weekday()<5
    mo = now.replace(hour=8,minute=30,second=0,microsecond=0)
    mc = now.replace(hour=14,minute=30,second=0,microsecond=0)
    is_rth = wkday and (mo <= now <= mc)
    status_text  = "MARKET OPEN" if is_rth else ("MARKET CLOSED" if wkday else "WEEKEND")
    st.markdown(f"""
    <div class="metric-container">
        <div style="font-size:0.9rem;color:var(--muted)">Market Status</div>
        <div style="font-size:1.6rem;font-weight:700">{status_text}</div>
        <div style="color:var(--muted)">RTH 08:30–14:30 CT (Mon–Fri)</div>
    </div>""", unsafe_allow_html=True)

with c3:
    st.markdown(f"""
    <div class="metric-container">
        <div style="font-size:0.9rem;color:var(--muted)">Slope per 30 min</div>
        <div style="font-size:1.6rem;font-weight:700">{SLOPE_PER_BLOCK:+.3f}</div>
        <div style="color:var(--muted)">Globex-aware blocks</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<hr/>", unsafe_allow_html=True)

# ───────────────────────────────────────────────────────────────────────────────
# TABS
# ───────────────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["SPX Anchors", "Stock Anchors", "Signals & EMA", "Contract Tool"])

# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║ TAB 1: SPX ANCHORS — Close Anchor Fan (Core Strategy)                      ║
# ╚═════════════════════════════════════════════════════════════════════════════╝
with tab1:
    st.subheader("SPX Close-Anchor Fan Strategy (Globex-aware)")
    st.caption("Fan from previous day Close: Top=+0.267/30m, Bottom=−0.267/30m (blocks skip 16:00–17:00 and weekend).")

    left, right = st.columns(2)
    with left:
        prev_day = st.date_input("Previous Trading Day", value=(now.date()-timedelta(days=1)), key="spx_prev_day")
        st.caption(prev_day.strftime("%A"))

    with right:
        proj_day = st.date_input("Projection Day", value=(prev_day + timedelta(days=1)), key="spx_proj_day")
        st.caption(f"Projecting for: {proj_day.strftime('%A')}")

    if st.button("Generate SPX Fan & Strategy", type="primary", key="spx_generate_fan"):
        with st.spinner("Fetching market data & building fan..."):
            # Fetch SPX prev & proj (fallback to SPY if ^GSPC empty)
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
                spx_proj_rth = get_session_window(spx_proj, RTH_START, RTH_END)
                dprev = get_daily_ohlc(spx_prev, prev_day)
                if not dprev or 'close' not in dprev:
                    st.error("Could not extract previous day OHLC.")
                else:
                    close_price, close_time = dprev['close']
                    high_price,  high_time  = dprev['high']
                    low_price,   low_time   = dprev['low']

                    # Fan from previous day Close (Globex-aware)
                    fan_df = project_close_fan(close_price, close_time, proj_day)
                    # Previous-day High ascending (for TP1 when above fan)
                    high_up = project_line_globex(high_price, high_time, +SLOPE_PER_BLOCK, proj_day, 'High_Asc')
                    # Previous-day Low descending (for TP1 when below fan)
                    low_dn  = project_line_globex(low_price,  low_time,  -SLOPE_PER_BLOCK, proj_day, 'Low_Desc')

                    # Strategy table (SPX-only display)
                    strat_df = build_strategy_from_fan(
                        spx_proj_rth, fan_df, high_up, low_dn, anchor_close_price=close_price
                    )

                    st.markdown("#### Fan Lines")
                    st.dataframe(fan_df, use_container_width=True, hide_index=True)

                    st.markdown("#### Strategy Table")
                    if not strat_df.empty:
                        st.dataframe(strat_df, use_container_width=True, hide_index=True)
                    else:
                        st.info("No strategy rows were generated (possible time alignment gap).")

# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║ TAB 2: STOCK ANCHORS (Mon/Tue) — simplified sample                         ║
# ╚═════════════════════════════════════════════════════════════════════════════╝
with tab2:
    st.subheader("Stock Anchor Analysis")
    st.caption("Mon/Tue combined session analysis for individual stocks (30-min).")

    core = ['TSLA','NVDA','AAPL','MSFT','AMZN','GOOGL','META','NFLX']
    tcols = st.columns(4)
    selected_ticker = None
    for i,tkr in enumerate(core):
        if tcols[i%4].button(tkr, key=f"stkbtn_{tkr}"):
            selected_ticker = tkr
            st.session_state['selected_stock'] = tkr

    custom = st.text_input("Custom Symbol", placeholder="Enter any ticker symbol", key="stk_custom_input")
    if custom:
        selected_ticker = custom.upper()
        st.session_state['selected_stock'] = selected_ticker
    if not selected_ticker and 'selected_stock' in st.session_state:
        selected_ticker = st.session_state['selected_stock']

    if selected_ticker:
        st.info(f"Selected: {selected_ticker}")
        slope_mag = st.number_input(f"{selected_ticker} Slope Magnitude",
                                    value=st.session_state.stock_slopes.get(selected_ticker, 0.0150),
                                    step=0.0001, format="%.4f")

        c1,c2,c3 = st.columns(3)
        with c1:
            mon_date = st.date_input("Monday Date", value=(now.date()-timedelta(days=2)), key=f"stk_mon_{selected_ticker}")
        with c2:
            tue_date = st.date_input("Tuesday Date", value=(mon_date+timedelta(days=1)), key=f"stk_tue_{selected_ticker}")
        with c3:
            st.caption(f"Wed: {tue_date+timedelta(days=1)} | Thu: {tue_date+timedelta(days=2)} | Fri: {tue_date+timedelta(days=3)}")

        if st.button(f"Analyze {selected_ticker}", type="primary", key=f"stk_analyze_{selected_ticker}"):
            with st.spinner("Fetching stock data..."):
                mon = fetch_live_data(selected_ticker, mon_date, mon_date)
                tue = fetch_live_data(selected_ticker, tue_date, tue_date)
                if mon.empty and tue.empty:
                    st.error(f"No data for {selected_ticker} on selected dates")
                else:
                    combined = mon if tue.empty else (tue if mon.empty else pd.concat([mon,tue]).sort_index())
                    st.session_state['stock_analysis_data'] = combined
                    st.session_state['stock_analysis_ticker'] = selected_ticker
                    st.session_state['stock_analysis_ready'] = True

        if st.session_state.get('stock_analysis_ready', False) and st.session_state.get('stock_analysis_ticker')==selected_ticker:
            st.subheader(f"{selected_ticker} Mon/Tue Combined (sample view)")
            st.dataframe(st.session_state['stock_analysis_data'].tail(24), use_container_width=True, hide_index=True)

# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║ TAB 3: SIGNALS & EMA — unchanged mechanics                                 ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
with tab3:
    st.subheader("Signal Detection & Market Analysis (EMA 8/21, RTH)")
    sc1, sc2 = st.columns(2)
    with sc1:
        symbol = st.selectbox("Analysis Symbol", ["^GSPC","ES=F","SPY"], index=0)
    with sc2:
        sday = st.date_input("Analysis Day", value=now.date())

    if st.button("Analyze Market Signals", type="primary", key="sig_generate"):
        with st.spinner("Fetching & analyzing..."):
            data = fetch_live_data(symbol, sday, sday)
            rth = get_session_window(data, RTH_START, RTH_END)
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
                ema_df = pd.DataFrame(out)
                st.dataframe(ema_df, use_container_width=True, hide_index=True)
                vol = calculate_market_volatility(rth)
                atr = calculate_average_true_range(rth,14)
                st.metric("Volatility", f"{vol:.2f}%")
                st.metric("Current ATR", f"{(atr.iloc[-1] if not atr.empty else 0):.2f}")

# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║ TAB 4: CONTRACT TOOL — simplified view                                      ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
with tab4:
    st.subheader("Contract Tool (Overnight points → RTH projection)")
    pc1, pc2 = st.columns(2)
    with pc1:
        p1_date = st.date_input("Point 1 Date", value=(now.date()-timedelta(days=1)))
        p1_time = st.time_input("Point 1 Time (CT)", value=time(20,0))
        p1_price= st.number_input("Point 1 Contract Price", value=10.0, min_value=0.01, step=0.01, format="%.2f")
    with pc2:
        p2_date = st.date_input("Point 2 Date", value=now.date())
        p2_time = st.time_input("Point 2 Time (CT)", value=time(8,0))
        p2_price= st.number_input("Point 2 Contract Price", value=12.0, min_value=0.01, step=0.01, format="%.2f")

    proj_day_ct = st.date_input("RTH Projection Day", value=p2_date)
    p1_dt = datetime.combine(p1_date, p1_time)
    p2_dt = datetime.combine(p2_date, p2_time)
    if p2_dt <= p1_dt:
        st.error("Point 2 must be after Point 1")
    else:
        hours = (p2_dt-p1_dt).total_seconds()/3600
        change = p2_price - p1_price
        st.metric("Time Span", f"{hours:.1f}h"); st.metric("ΔPrice", f"{change:+.2f}")
        blocks = active_30m_blocks_between(as_ct(p1_dt), as_ct(p2_dt))
        slope = (p2_price - p1_price)/blocks if blocks>0 else 0.0

        if st.button("Analyze Contract Projections", type="primary", key="ct_generate"):
            rows=[]
            for t in rth_slots_ct(proj_day_ct):
                b = active_30m_blocks_between(as_ct(p1_dt), t)
                price = p1_price + slope*b
                rows.append({'Time': format_ct_time(t), 'Contract_Price': round(price,2), 'Blocks': int(b)})
            proj_df = pd.DataFrame(rows)
            st.dataframe(proj_df, use_container_width=True, hide_index=True)

# ───────────────────────────────────────────────────────────────────────────────
# FOOTER QUICK TEST
# ───────────────────────────────────────────────────────────────────────────────
st.markdown("<hr/>", unsafe_allow_html=True)
if st.button("Test Data Connection", key="test_connection"):
    with st.spinner("Testing market data connection..."):
        test_data = fetch_live_data("^GSPC", now.date()-timedelta(days=3), now.date())
        if test_data.empty:
            # SPY fallback for connectivity test
            test_data = fetch_live_data("SPY", now.date()-timedelta(days=3), now.date())
        if not test_data.empty:
            st.success(f"Connection OK — received {len(test_data)} bars")
        else:
            st.error("Market data connection failed (empty response). Try again later or adjust dates.")