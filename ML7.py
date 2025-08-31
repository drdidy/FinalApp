# app.py
# ═══════════════════════════════════════════════════════════════════════════════
# 🔮 SPX PROPHET — Unified App (Fan logic + weekend/maintenance fix + touch rules)
# ═══════════════════════════════════════════════════════════════════════════════

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import pytz
from datetime import datetime, date, time, timedelta
from datetime import time as dtime
from typing import Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')

# ───────────────────────────────────────────────────────────────────────────────
# CORE CONFIG
# ───────────────────────────────────────────────────────────────────────────────
CT_TZ = pytz.timezone('America/Chicago')
UTC_TZ = pytz.UTC

RTH_START = "08:30"  # CT
RTH_END   = "14:30"  # CT

# Slope magnitude per 30-min block (after 1h maintenance considered)
SLOPE_PER_BLOCK = 0.377  # ascending +0.377, descending -0.377

SPX_ANCHOR_START = "17:00"
SPX_ANCHOR_END   = "19:30"

# ───────────────────────────────────────────────────────────────────────────────
# PAGE
# ───────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="🔮 SPX Prophet Analytics",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
.main { background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%); color: white;}
.metric-container { background: linear-gradient(135deg, rgba(255,255,255,0.1), rgba(255,255,255,0.05));
  padding: 1.2rem; border-radius: 15px; border: 2px solid rgba(255,255,255,0.2); backdrop-filter: blur(10px);
  box-shadow: 0 8px 32px rgba(0,0,0,0.1); transition: all 0.3s ease;}
.metric-container:hover { transform: translateY(-5px); box-shadow: 0 15px 35px rgba(0,0,0,0.2);}
.stTab { background: linear-gradient(135deg, rgba(255,255,255,0.1), rgba(255,255,255,0.05));
  border-radius: 10px; padding: 10px; margin: 5px;}
.stDataFrame { background: rgba(255,255,255,0.95); border-radius: 10px; overflow: hidden;}
</style>
""", unsafe_allow_html=True)

# ───────────────────────────────────────────────────────────────────────────────
# TIME / SESSIONS
# ───────────────────────────────────────────────────────────────────────────────
def rth_slots_ct(target_date: date) -> List[datetime]:
    s = CT_TZ.localize(datetime.combine(target_date, time(8,30)))
    e = CT_TZ.localize(datetime.combine(target_date, time(14,30)))
    slots = []
    cur = s
    while cur <= e:
        slots.append(cur)
        cur += timedelta(minutes=30)
    return slots

def format_ct_time(dt: datetime) -> str:
    if dt.tzinfo is None:
        dt = CT_TZ.localize(dt)
    else:
        dt = dt.astimezone(CT_TZ)
    return dt.strftime("%H:%M")

def _is_maintenance(dt_ct: datetime) -> bool:
    """Daily maintenance halt 16:00–17:00 CT."""
    t = dt_ct.timetz()
    return dtime(16,0) <= t < dtime(17,0)

def _is_globex_open(dt_ct: datetime) -> bool:
    """
    Globex session rules (CT):
      - Sunday: opens 17:00
      - Mon–Thu: 17:00 previous day → 16:00 current day (maintenance 16:00–17:00)
      - Friday: closes 16:00; weekend halt until Sunday 17:00
    """
    if _is_maintenance(dt_ct):
        return False
    wd = dt_ct.weekday()  # Mon=0..Sun=6
    t  = dt_ct.timetz()
    if wd == 5:  # Saturday
        return False
    if wd == 6:  # Sunday
        return t >= dtime(17,0)
    if wd == 4:  # Friday
        return t < dtime(16,0)
    return True  # Mon–Thu (outside maintenance)

def _ceil_to_next_30m(dt_ct: datetime) -> datetime:
    dt_ct = dt_ct.astimezone(CT_TZ)
    if dt_ct.minute == 0 and dt_ct.second == 0 and dt_ct.microsecond == 0:
        return dt_ct
    if dt_ct.minute < 30:
        return dt_ct.replace(minute=30, second=0, microsecond=0)
    return (dt_ct.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1))

def trading_blocks_since(anchor_time: datetime, slot_time: datetime) -> float:
    """
    Count 30-minute blocks between anchor_time and slot_time,
    including ONLY times where Globex is open; excludes maintenance halt and weekend.
    Assumes both are timezone-aware CT.
    We count a block if the END of that 30m step lands in open trading.
    """
    if slot_time <= anchor_time:
        return 0.0
    step = timedelta(minutes=30)
    t = _ceil_to_next_30m(anchor_time)
    blocks = 0
    while t <= slot_time:
        if _is_globex_open(t):
            blocks += 1
        t += step
    return float(blocks)

# ───────────────────────────────────────────────────────────────────────────────
# DATA
# ───────────────────────────────────────────────────────────────────────────────
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

@st.cache_data(ttl=60)
def fetch_live_data(symbol: str, start_date: date, end_date: date) -> pd.DataFrame:
    """Robust intraday fetch with period fallback, 30m interval, CT index."""
    def _normalize(df):
        if df.empty: return df
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0] if isinstance(c, tuple) else str(c) for c in df.columns]
        req = ['Open','High','Low','Close','Volume']
        if any(c not in df.columns for c in req):
            return pd.DataFrame()
        if df.index.tz is None:
            df.index = df.index.tz_localize('US/Eastern')
        df.index = df.index.tz_convert(CT_TZ)
        sdt = CT_TZ.localize(datetime.combine(start_date, time(0,0)))
        edt = CT_TZ.localize(datetime.combine(end_date,   time(23,59)))
        return df.loc[sdt:edt]

    try:
        t = yf.Ticker(symbol)
        df = t.history(
            start=(start_date - timedelta(days=5)).strftime('%Y-%m-%d'),
            end=(end_date + timedelta(days=2)).strftime('%Y-%m-%d'),
            interval="30m", prepost=True, auto_adjust=False, back_adjust=False
        )
        df = _normalize(df)

        if df.empty:
            days = max(7, (end_date - start_date).days + 7)
            df2 = t.history(period=f"{days}d", interval="30m",
                            prepost=True, auto_adjust=False, back_adjust=False)
            df = _normalize(df2)

        if df.empty:
            return pd.DataFrame()

        if not validate_ohlc_data(df):
            st.warning(f"⚠️ Data quality issues detected for {symbol}")
        return df

    except Exception as e:
        st.error(f"❌ Error fetching {symbol}: {e}")
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
    o = day.iloc[0]['Open']
    h = day['High'].max(); ht = day[day['High']==h].index[0]
    l = day['Low'].min();  lt = day[day['Low']==l].index[0]
    c = day.iloc[-1]['Close']; ct = day.index[-1]
    return {'open':(o,day.index[0]), 'high':(h,ht), 'low':(l,lt), 'close':(c,ct)}

# ───────────────────────────────────────────────────────────────────────────────
# PROJECTIONS
# ───────────────────────────────────────────────────────────────────────────────
def project_line(anchor_price: float, anchor_time: datetime, slope_per_block: float,
                 target_date: date, col_name: str) -> pd.DataFrame:
    """Project using ONLY trading-active 30m blocks between anchor and each RTH slot."""
    rows = []
    for t in rth_slots_ct(target_date):
        b = trading_blocks_since(anchor_time, t)
        price = anchor_price + slope_per_block * b
        rows.append({'Time': format_ct_time(t), col_name: round(price,2), 'Blocks': round(b,1)})
    return pd.DataFrame(rows)

def project_close_fan(close_price: float, close_time: datetime, target_date: date) -> pd.DataFrame:
    top = project_line(close_price, close_time, +SLOPE_PER_BLOCK, target_date, 'Top')
    bot = project_line(close_price, close_time, -SLOPE_PER_BLOCK, target_date, 'Bottom')
    df = pd.merge(top[['Time','Top']], bot[['Time','Bottom']], on='Time', how='inner')
    df['Fan_Width'] = (df['Top'] - df['Bottom']).round(2)
    df['Mid'] = round(close_price,2)
    return df

# ───────────────────────────────────────────────────────────────────────────────
# STRATEGY (Bias + fan entries/targets)
# ───────────────────────────────────────────────────────────────────────────────
def df_to_time_price_lookup(df: pd.DataFrame, price_col: str) -> Dict[str,float]:
    if df is None or df.empty: return {}
    return {row['Time']: row[price_col] for _,row in df[['Time',price_col]].iterrows()}

def build_strategy_from_fan(
    rth_prices: pd.DataFrame, fan_df: pd.DataFrame,
    high_up_df: Optional[pd.DataFrame], low_dn_df: Optional[pd.DataFrame],
    anchor_close_price: float
) -> pd.DataFrame:
    if rth_prices.empty or fan_df.empty: return pd.DataFrame()

    price_lookup = {format_ct_time(ix): rth_prices.loc[ix,'Close'] for ix in rth_prices.index}
    top_lu   = df_to_time_price_lookup(fan_df, 'Top')
    bot_lu   = df_to_time_price_lookup(fan_df, 'Bottom')
    high_up  = df_to_time_price_lookup(high_up_df, 'High_Asc')
    low_dn   = df_to_time_price_lookup(low_dn_df,  'Low_Desc')

    rows=[]
    for t in fan_df['Time']:
        if t not in price_lookup: continue
        p   = price_lookup[t]
        top = top_lu.get(t,np.nan)
        bot = bot_lu.get(t,np.nan)
        width = top - bot if pd.notna(top) and pd.notna(bot) else np.nan
        bias = "UP" if p >= anchor_close_price else "DOWN"

        if pd.isna(width):
            continue

        within = bot <= p <= top
        above  = p > top
        below  = p < bot

        direction=""; entry=np.nan; tp1=np.nan; tp2=np.nan; note=""

        if within:
            if bias=="UP":
                direction="BUY"; entry=bot; tp1=top; tp2=top
                note="Within fan; bias UP → buy bottom → exit top"
            else:
                direction="SELL"; entry=top; tp1=bot; tp2=bot
                note="Within fan; bias DOWN → sell top → exit bottom"

        elif above:
            direction="SELL"; entry=top
            tp2 = max(bot, entry - width)  # drop fan width
            tp1 = high_up.get(t, np.nan)   # prev day high ascending
            note="Above fan; entry=Top; TP2=Top-width; TP1=High ascending"

        elif below:
            direction="SELL"; entry=bot
            tp2 = entry - width            # drop fan width
            tp1 = low_dn.get(t, np.nan)    # prev day low descending
            note="Below fan; entry=Bottom; TP2=Bottom-width; TP1=Low descending"

        rows.append({
            'Time': t, 'Price': round(p,2), 'Bias': bias, 'EntrySide': direction,
            'Entry': round(entry,2) if pd.notna(entry) else np.nan,
            'TP1': round(tp1,2) if pd.notna(tp1) else np.nan,
            'TP2': round(tp2,2) if pd.notna(tp2) else np.nan,
            'Top': round(top,2), 'Bottom': round(bot,2),
            'Fan_Width': round(width,2), 'Note': note
        })

    return pd.DataFrame(rows)

# ───────────────────────────────────────────────────────────────────────────────
# FAN TOUCH SIGNAL RULES (Next-candle expectations)
# ───────────────────────────────────────────────────────────────────────────────
def evaluate_fan_touch_signals(rth_prices: pd.DataFrame, fan_df: pd.DataFrame) -> pd.DataFrame:
    """
    Implements your rules:
    - Bottom touch & bearish candle that closes above bottom → Next bullish → BUY to Top.
    - First RTH bar touches bottom & closes bullish → Next bearish → SELL, break down through fan.
    - Top touch & closes bearish → Next bullish → BUY, break up through fan.
    - Top touch & closes bullish but below top → Next bearish → SELL to Bottom.
    """
    if rth_prices.empty or fan_df.empty:
        return pd.DataFrame()

    # lookups
    top_lu = df_to_time_price_lookup(fan_df, 'Top')
    bot_lu = df_to_time_price_lookup(fan_df, 'Bottom')

    rows=[]
    idxs = list(rth_prices.index)
    for i, ts in enumerate(idxs):
        tstr = format_ct_time(ts)
        if tstr not in top_lu or tstr not in bot_lu:
            continue

        bar = rth_prices.loc[ts]
        o,h,l,c = bar['Open'], bar['High'], bar['Low'], bar['Close']
        top = top_lu[tstr]; bottom = bot_lu[tstr]
        is_bull = c > o
        is_bear = c < o

        touched_bottom = (l <= bottom <= h)
        touched_top    = (l <= top    <= h)

        signal=None; next_dir=None; target=None; rationale=None

        # SPECIAL: first RTH bar override
        if i == 0 and touched_bottom and is_bull:
            signal   = "FirstBar_BottomTouch_BullClose"
            next_dir = "BEARISH_NEXT"  # next candle expected bearish & breakdown
            target   = "BreakDown_Through_Fan"
            rationale= "First RTH bar touched bottom & closed bullish → next candle bearish breakdown."
        else:
            # Bottom touch rule
            if touched_bottom and is_bear and (c > bottom):
                signal   = "BottomTouch_BearCloseAboveBottom"
                next_dir = "BULLISH_NEXT"
                target   = "BUY_to_Top"
                rationale= "Bearish candle touched bottom but closed above → next bullish to Top."

            # Top touch rules
            elif touched_top and is_bear:
                signal   = "TopTouch_BearClose"
                next_dir = "BULLISH_NEXT"
                target   = "BUY_break_Up_Through_Fan"
                rationale= "Touched Top and closed bearish → next bullish breakout up."

            elif touched_top and is_bull and (c < top):
                signal   = "TopTouch_BullClose_BelowTop"
                next_dir = "BEARISH_NEXT"
                target   = "SELL_to_Bottom"
                rationale= "Touched Top; bullish close but below Top → next bearish to Bottom."

        if signal:
            rows.append({
                'Time': tstr, 'Open': round(o,2), 'High': round(h,2), 'Low': round(l,2), 'Close': round(c,2),
                'Top': round(top,2), 'Bottom': round(bottom,2),
                'Signal': signal, 'Next_Expectation': next_dir, 'Suggested_Action_Target': target,
                'Note': rationale
            })

    return pd.DataFrame(rows)

# ───────────────────────────────────────────────────────────────────────────────
# INDICATORS (minimal)
# ───────────────────────────────────────────────────────────────────────────────
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
    return r.std()*np.sqrt(390)*100

# ───────────────────────────────────────────────────────────────────────────────
# HEADER METRICS
# ───────────────────────────────────────────────────────────────────────────────
now = datetime.now(CT_TZ)
c1,c2,c3 = st.columns(3)
with c1:
    st.markdown(f"""
    <div class="metric-container">
        <h3>⏰ Current Time (CT)</h3>
        <h2>{now.strftime("%H:%M:%S")}</h2>
        <p>{now.strftime("%A, %B %d")}</p>
    </div>""", unsafe_allow_html=True)
with c2:
    wkday = now.weekday()<5
    mo = now.replace(hour=8,minute=30,second=0,microsecond=0)
    mc = now.replace(hour=14,minute=30,second=0,microsecond=0)
    is_rth = wkday and (mo <= now <= mc)
    status_color = "#00ff88" if is_rth else ("#ffbb33" if wkday else "#ff6b6b")
    status_text  = "MARKET OPEN" if is_rth else ("MARKET CLOSED" if wkday else "WEEKEND")
    st.markdown(f"""
    <div class="metric-container">
        <h3>Market Status</h3>
        <h2 style="color:{status_color};">{status_text}</h2>
        <p>RTH: 08:30 - 14:30 CT</p>
        <p>Mon-Fri Only</p>
    </div>""", unsafe_allow_html=True)
with c3:
    st.markdown(f"""
    <div class="metric-container">
        <h3>📐 Slope per 30m</h3>
        <h2>{SLOPE_PER_BLOCK:+.3f}</h2>
        <p>Ascending / Descending magnitude</p>
    </div>""", unsafe_allow_html=True)

st.markdown("---")

# ───────────────────────────────────────────────────────────────────────────────
# TABS
# ───────────────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["SPX Anchors", "Stock Anchors", "Signals & EMA", "Contract Tool"])

# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║ TAB 1: SPX ANCHORS                                                          ║
# ╚═════════════════════════════════════════════════════════════════════════════╝
with tab1:
    st.subheader("SPX Close-Anchor Fan Strategy (+ touch signals)")
    left, right = st.columns(2)
    with left:
        prev_day = st.date_input("Previous Trading Day", value=(now.date()-timedelta(days=1)), key="spx_prev_day")
        st.caption(prev_day.strftime("%A"))
    with right:
        proj_day = st.date_input("Projection Day", value=(prev_day + timedelta(days=1)), key="spx_proj_day")
        st.caption(f"Projecting for: {proj_day.strftime('%A')}")

    if st.button("Generate SPX Fan & Strategy", type="primary", key="spx_generate_fan"):
        with st.spinner("Fetching market data & building fan..."):
            # Fetch previous day (^GSPC then SPY fallback)
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
                rth = get_session_window(spx_proj, RTH_START, RTH_END)

                dprev = get_daily_ohlc(spx_prev, prev_day)
                if not dprev or 'close' not in dprev:
                    st.error("Could not extract previous day OHLC.")
                else:
                    close_price, close_time = dprev['close']
                    high_price,  high_time  = dprev['high']
                    low_price,   low_time   = dprev['low']

                    # Fan and anchor lines (weekend/maintenance aware)
                    fan_df = project_close_fan(close_price, close_time, proj_day)
                    high_up = project_line(high_price, high_time, +SLOPE_PER_BLOCK, proj_day, 'High_Asc')
                    low_dn  = project_line(low_price,  low_time,  -SLOPE_PER_BLOCK, proj_day, 'Low_Desc')

                    # Strategy table
                    strat_df = build_strategy_from_fan(rth, fan_df, high_up, low_dn, anchor_close_price=close_price)

                    # Touch signals based on your rules
                    touch_df = evaluate_fan_touch_signals(rth, fan_df)

                    st.markdown("### Fan Lines")
                    st.dataframe(fan_df, use_container_width=True, hide_index=True)

                    # Optional: explicit anchor slope tables
                    show_lines = st.checkbox("Show explicit High/Low/Close slope lines", value=False)
                    if show_lines:
                        lt1, lt2, lt3, lt4 = st.tabs(["Close (+)", "Close (−)", "High (+)", "Low (−)"])
                        with lt1:
                            st.dataframe(project_line(close_price, close_time, +SLOPE_PER_BLOCK, proj_day, 'Close_Asc'), use_container_width=True, hide_index=True)
                        with lt2:
                            st.dataframe(project_line(close_price, close_time, -SLOPE_PER_BLOCK, proj_day, 'Close_Desc'), use_container_width=True, hide_index=True)
                        with lt3:
                            st.dataframe(high_up, use_container_width=True, hide_index=True)
                        with lt4:
                            st.dataframe(low_dn, use_container_width=True, hide_index=True)

                    st.markdown("### Strategy Table")
                    if not strat_df.empty:
                        st.dataframe(strat_df, use_container_width=True, hide_index=True)
                    else:
                        st.info("No strategy rows were generated (possible time alignment gap).")

                    st.markdown("### Fan Touch Signals (Next Candle Expectations)")
                    if not touch_df.empty:
                        st.dataframe(touch_df, use_container_width=True, hide_index=True)
                    else:
                        st.info("No qualifying fan touches in RTH for the rules provided.")

# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║ TAB 2: STOCK ANCHORS (light sample)                                         ║
# ╚═════════════════════════════════════════════════════════════════════════════╝
with tab2:
    st.subheader("Stock Anchors (Mon/Tue) — sample")
    core = ['TSLA','NVDA','AAPL','MSFT','AMZN','GOOGL','META','NFLX']
    ccols = st.columns(4)
    selected = None
    for i,tkr in enumerate(core):
        if ccols[i%4].button(tkr, key=f"stkbtn_{tkr}"):
            selected = tkr
            st.session_state['selected_stock'] = tkr
    custom = st.text_input("Custom Symbol", placeholder="Enter any ticker symbol", key="stk_custom")
    if custom:
        selected = custom.upper()
        st.session_state['selected_stock'] = selected
    if not selected and 'selected_stock' in st.session_state:
        selected = st.session_state['selected_stock']
    if selected:
        mon_date = st.date_input("Monday Date", value=(now.date()-timedelta(days=2)), key=f"mon_{selected}")
        tue_date = st.date_input("Tuesday Date", value=(mon_date+timedelta(days=1)), key=f"tue_{selected}")
        if st.button(f"Analyze {selected}", type="primary", key=f"an_{selected}"):
            mon = fetch_live_data(selected, mon_date, mon_date)
            tue = fetch_live_data(selected, tue_date, tue_date)
            if mon.empty and tue.empty:
                st.error("No data for selected dates.")
            else:
                combined = mon if tue.empty else (tue if mon.empty else pd.concat([mon,tue]).sort_index())
                st.dataframe(combined.tail(24), use_container_width=True)

# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║ TAB 3: SIGNALS & EMA (minimal)                                              ║
# ╚═════════════════════════════════════════════════════════════════════════════╝
with tab3:
    st.subheader("Signals & EMA 8/21 (RTH)")
    sc1, sc2 = st.columns(2)
    with sc1:
        symbol = st.selectbox("Symbol", ["^GSPC","ES=F","SPY"], index=0)
    with sc2:
        sday = st.date_input("Day", value=now.date())
    if st.button("Analyze", type="primary", key="sig"):
        data = fetch_live_data(symbol, sday, sday)
        rth = get_session_window(data, RTH_START, RTH_END)
        if rth.empty:
            st.error("No RTH data.")
        else:
            ema8  = rth['Close'].ewm(span=8).mean()
            ema21 = rth['Close'].ewm(span=21).mean()
            out=[]
            for i in range(1,len(rth)):
                t = format_ct_time(rth.index[i]); p=rth['Close'].iloc[i]
                prev8, prev21 = ema8.iloc[i-1], ema21.iloc[i-1]
                c8, c21 = ema8.iloc[i], ema21.iloc[i]
                cross=None
                if prev8<=prev21 and c8>c21: cross="Bullish Cross"
                elif prev8>=prev21 and c8<c21: cross="Bearish Cross"
                sep = abs(c8-c21)/c21*100 if c21!=0 else 0
                out.append({'Time':t,'Price':round(p,2),'EMA8':round(c8,2),'EMA21':round(c21,2),
                            'Separation_%':round(sep,3),'Regime':("Bullish" if c8>c21 else "Bearish"),
                            'Crossover': cross or 'None'})
            st.dataframe(pd.DataFrame(out), use_container_width=True, hide_index=True)

# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║ TAB 4: CONTRACT TOOL (simple)                                               ║
# ╚═════════════════════════════════════════════════════════════════════════════╝
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
    p1_dt = CT_TZ.localize(datetime.combine(p1_date, p1_time))
    p2_dt = CT_TZ.localize(datetime.combine(p2_date, p2_time))
    if p2_dt <= p1_dt:
        st.error("Point 2 must be after Point 1")
    else:
        hours = (p2_dt-p1_dt).total_seconds()/3600
        st.metric("Time Span", f"{hours:.1f}h"); st.metric("ΔPrice", f"{(p2_price-p1_price):+.2f}")
        blocks = trading_blocks_since(p1_dt, p2_dt)
        slope = (p2_price - p1_price)/blocks if blocks>0 else 0.0

        if st.button("Analyze Contract Projections", type="primary", key="ct_generate"):
            rows=[]
            for t in rth_slots_ct(proj_day_ct):
                b = trading_blocks_since(p1_dt, t)
                price = p1_price + slope*b
                rows.append({'Time': format_ct_time(t), 'Contract_Price': round(price,2), 'Blocks': round(b,1)})
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

# ───────────────────────────────────────────────────────────────────────────────
# FOOTER: Connectivity test
# ───────────────────────────────────────────────────────────────────────────────
st.markdown("---")
if st.button("Test Data Connection", key="test_connection"):
    with st.spinner("Testing market data connection..."):
        test_data = fetch_live_data("SPY", now.date()-timedelta(days=5), now.date())
        if not test_data.empty:
            st.success(f"Connection OK — received {len(test_data)} bars for SPY")
        else:
            st.error("Market data connection failed (empty response). Try again later or adjust dates.")