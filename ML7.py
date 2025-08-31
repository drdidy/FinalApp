 ═══════════════════════════════════════════════════════════════════════════════
# 🔮 SPX PROPHET — FULL APP (All 6 Parts, Unified)
# Implements your fan logic: Close anchor + slope ±0.377 per 30-min block
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

# Slope magnitude per 30-min block (after 1h maintenance considered)
SLOPE_PER_BLOCK = 0.377  # ascending +0.377, descending -0.377

SPX_ANCHOR_START = "17:00"  # (kept for completeness)
SPX_ANCHOR_END   = "19:30"

STOCK_SLOPES_DEFAULT = {
    'AAPL': 0.0155, 'MSFT': 0.0541, 'NVDA': 0.0086, 'AMZN': 0.0139,
    'GOOGL': 0.0122, 'TSLA': 0.0285, 'META': 0.0674, 'NFLX': 0.0230
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
    if dt.tzinfo is None:
        dt = CT_TZ.localize(dt)
    elif dt.tzinfo != CT_TZ:
        dt = dt.astimezone(CT_TZ)
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

@st.cache_data(ttl=60)
def fetch_live_data(symbol: str, start_date: date, end_date: date) -> pd.DataFrame:
    """Robust intraday fetch with period fallback, 30-min interval, CT index."""
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
        # try start/end (buffered)
        df = t.history(
            start=(start_date - timedelta(days=5)).strftime('%Y-%m-%d'),
            end=(end_date + timedelta(days=2)).strftime('%Y-%m-%d'),
            interval="30m", prepost=True, auto_adjust=False, back_adjust=False
        )
        df = _normalize(df)

        # fallback: period= (more reliable for intraday)
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

# Simple swings (kept)
def detect_swings_simple(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or len(df)<2: return df.copy()
    d = df.copy(); d['swing_high']=False; d['swing_low']=False
    if 'Close' in d.columns:
        d.loc[d['Close'].idxmax(),'swing_high']=True
        d.loc[d['Close'].idxmin(),'swing_low'] =True
    return d

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
# PROJECTIONS (Core to your logic)
# ───────────────────────────────────────────────────────────────────────────────
def blocks_between(anchor_time: datetime, slot_time: datetime) -> float:
    return (slot_time - anchor_time).total_seconds()/1800.0  # 30m blocks

def project_line(anchor_price: float, anchor_time: datetime, slope_per_block: float,
                 target_date: date, col_name: str) -> pd.DataFrame:
    rows=[]
    for t in rth_slots_ct(target_date):
        b = blocks_between(anchor_time, t)
        price = anchor_price + slope_per_block*b
        rows.append({'Time':format_ct_time(t), col_name: round(price,2), 'Blocks': round(b,1)})
    return pd.DataFrame(rows)

def project_close_fan(close_price: float, close_time: datetime, target_date: date) -> pd.DataFrame:
    """Top = +SLOPE_PER_BLOCK from Close; Bottom = -SLOPE_PER_BLOCK from Close."""
    top = project_line(close_price, close_time, +SLOPE_PER_BLOCK, target_date, 'Top')
    bot = project_line(close_price, close_time, -SLOPE_PER_BLOCK, target_date, 'Bottom')
    df = pd.merge(top[['Time','Top']], bot[['Time','Bottom']], on='Time', how='inner')
    df['Fan_Width'] = (df['Top'] - df['Bottom']).round(2)
    df['Mid'] = round(close_price,2)  # mid is the close anchor price
    return df

# ───────────────────────────────────────────────────────────────────────────────
# STRATEGY TABLE (Your exact entry/exit rules)
# ───────────────────────────────────────────────────────────────────────────────
def df_to_time_price_lookup(df: pd.DataFrame, price_col: str) -> Dict[str,float]:
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

    # Lookups by "HH:MM"
    price_lookup = {format_ct_time(ix): rth_prices.loc[ix, 'Close'] for ix in rth_prices.index}
    top_lookup   = df_to_time_price_lookup(fan_df,'Top')
    bot_lookup   = df_to_time_price_lookup(fan_df,'Bottom')
    high_up_lu   = df_to_time_price_lookup(high_up_df, 'High_Asc') if high_up_df is not None and not high_up_df.empty else {}
    low_dn_lu    = df_to_time_price_lookup(low_dn_df,  'Low_Desc') if low_dn_df  is not None and not low_dn_df.empty  else {}

    rows=[]
    for t in fan_df['Time']:
        if t not in price_lookup: 
            continue
        p    = price_lookup[t]
        top  = top_lookup.get(t, np.nan)
        bot  = bot_lookup.get(t, np.nan)
        width= top - bot if pd.notna(top) and pd.notna(bot) else np.nan
        bias = "UP" if p >= anchor_close_price else "DOWN"

        # default placeholders
        direction=""; entry=np.nan; tp1=np.nan; tp2=np.nan; note=""

        within_fan = (pd.notna(top) and pd.notna(bot) and bot <= p <= top)
        above_fan  = (pd.notna(top) and p > top)
        below_fan  = (pd.notna(bot) and p < bot)

        if within_fan:
            if bias=="UP":
                direction="BUY"
                entry = bot
                tp1   = top  # exit at top slope
                tp2   = top
                note  = "Within fan; bias UP → buy bottom → exit top"
            else:
                direction="SELL"
                entry = top
                tp1   = bot
                tp2   = bot
                note  = "Within fan; bias DOWN → sell top → exit bottom"

        elif above_fan:
            # Outside above fan
            direction="SELL"
            entry = top
            tp2   = max(bot, entry - width) if pd.notna(width) else bot
            # TP1 uses ascending slope of previous day HIGH at this time
            tp1   = high_up_lu.get(t, np.nan)
            note  = "Above fan; entry=Top; TP2=Top-width; TP1=High ascending"

        elif below_fan:
            # Outside below fan
            direction="SELL"   # bias DOWN consistent
            entry = bot
            tp2   = entry - width if pd.notna(width) else np.nan  # drop fan width
            # TP1 uses descending slope of previous day LOW at this time
            tp1   = low_dn_lu.get(t, np.nan)
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
# EXTRA ANALYTICS (Signals/EMA/Contract Tool reuse)
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
    vol = r.std()*np.sqrt(390)
    return vol*100

def detect_market_regime(data: pd.DataFrame) -> dict:
    if data.empty or len(data)<20:
        return {'regime':'INSUFFICIENT_DATA','trend':'NEUTRAL','strength':0,'volatility':1.5,'price_change':0}
    closes = data['Close'].tail(20)
    pc = (closes.iloc[-1]-closes.iloc[0])/closes.iloc[0]*100
    ret = closes.pct_change().dropna()
    vol = ret.std()*np.sqrt(390)*100
    if pc>1.0: trend='BULLISH'; strength=min(100,abs(pc)*10)
    elif pc<-1.0: trend='BEARISH'; strength=min(100,abs(pc)*10)
    else: trend='NEUTRAL'; strength=abs(pc)*10
    regime = 'HIGH_VOLATILITY' if vol>=3.0 else 'MODERATE_VOLATILITY' if vol>=1.8 else 'LOW_VOLATILITY'
    return {'regime':regime,'trend':trend,'strength':strength,'volatility':vol,'price_change':pc}

# ───────────────────────────────────────────────────────────────────────────────
# SESSION STATE INIT
# ───────────────────────────────────────────────────────────────────────────────
if 'stock_slopes' not in st.session_state:
    st.session_state.stock_slopes = STOCK_SLOPES_DEFAULT.copy()

# ───────────────────────────────────────────────────────────────────────────────
# HEADER
# ───────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center; padding: 2rem; background: linear-gradient(135deg, rgba(255,255,255,0.1), rgba(255,255,255,0.05)); border-radius: 20px; margin: 1rem 0; border: 2px solid rgba(255,255,255,0.2);">
    <h1 style="font-size: 3rem; margin: 0; background: linear-gradient(135deg, #ff6b6b, #4ecdc4, #45b7d1, #f9ca24); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;">
        SPX Prophet Analytics
    </h1>
    <p style="font-size: 1.3rem; margin: 1rem 0; opacity: 0.9;">
        Close-anchor fan strategy with ±0.377 per 30-min slope, market-integrated
    </p>
</div>
""", unsafe_allow_html=True)

c1,c2,c3 = st.columns(3)
with c1:
    now = datetime.now(CT_TZ)
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
# ║ TAB 1: SPX ANCHORS — Close Anchor Fan (Core Strategy)                      ║
# ╚═════════════════════════════════════════════════════════════════════════════╝
with tab1:
    st.subheader("SPX Close-Anchor Fan Strategy")
    st.caption("Fan from previous day Close: Top=+0.377/30m, Bottom=−0.377/30m. Bias from Close price.")

    left, right = st.columns(2)
    with left:
        prev_day = st.date_input("Previous Trading Day", value=(now.date()-timedelta(days=1)), key="spx_prev_day")
        st.caption(prev_day.strftime("%A"))

    with right:
        proj_day = st.date_input("Projection Day", value=(prev_day + timedelta(days=1)), key="spx_proj_day")
        st.caption(f"Projecting for: {proj_day.strftime('%A')}")

    if st.button("Generate SPX Fan & Strategy", type="primary", key="spx_generate_fan"):
        with st.spinner("Fetching market data & building fan..."):
            # Fetch SPX prev & proj
            spx_prev = fetch_live_data("^GSPC", prev_day, prev_day)
            if spx_prev.empty:
                # fallback to SPY for prev_day
                spx_prev = fetch_live_data("SPY", prev_day, prev_day)

            spx_proj = fetch_live_data("^GSPC", proj_day, proj_day)
            if spx_proj.empty:
                spx_proj = fetch_live_data("SPY", proj_day, proj_day)

            if spx_prev.empty:
                st.error("No previous day data (SPX/SPY).")
            elif spx_proj.empty:
                st.error("No projection day data (SPX/SPY).")
            else:
                # RTH slice for proj day
                spx_proj_rth = get_session_window(spx_proj, RTH_START, RTH_END)

                # Get prev day OHLC & anchor times
                dprev = get_daily_ohlc(spx_prev, prev_day)
                if not dprev or 'close' not in dprev:
                    st.error("Could not extract previous day OHLC.")
                else:
                    close_price, close_time = dprev['close']
                    high_price,  high_time  = dprev['high']
                    low_price,   low_time   = dprev['low']

                    # Project fan from previous day Close
                    fan_df = project_close_fan(close_price, close_time, proj_day)

                    # High ascending (TP1 when above fan)
                    high_up = project_line(high_price, high_time, +SLOPE_PER_BLOCK, proj_day, 'High_Asc')

                    # Low descending (TP1 when below fan)
                    low_dn  = project_line(low_price,  low_time,  -SLOPE_PER_BLOCK, proj_day, 'Low_Desc')

                    # Strategy table
                    strat_df = build_strategy_from_fan(
                        spx_proj_rth, fan_df, high_up, low_dn, anchor_close_price=close_price
                    )

                    # Display
                    st.markdown("### Fan Lines")
                    st.dataframe(fan_df, use_container_width=True, hide_index=True)

                    st.markdown("### Strategy Table")
                    if not strat_df.empty:
                        st.dataframe(strat_df, use_container_width=True, hide_index=True)
                    else:
                        st.info("No strategy rows were generated (possible time alignment gap).")

# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║ TAB 2: STOCK ANCHORS (Mon/Tue) — unchanged analytics, uses robust fetch     ║
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
        slope_mag = st.number_input(f"{selected_ticker} Slope Magnitude (fan base, optional features)",
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
            st.dataframe(st.session_state['stock_analysis_data'].tail(24), use_container_width=True)

# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║ TAB 3: SIGNALS & EMA — unchanged mechanics                                 ║
# ╚═════════════════════════════════════════════════════════════════════════════╝
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
# ║ TAB 4: CONTRACT TOOL — simplified view (kept functional)                    ║
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
    p1_dt = datetime.combine(p1_date, p1_time)
    p2_dt = datetime.combine(p2_date, p2_time)
    if p2_dt <= p1_dt:
        st.error("Point 2 must be after Point 1")
    else:
        hours = (p2_dt-p1_dt).total_seconds()/3600
        change = p2_price - p1_price
        st.metric("Time Span", f"{hours:.1f}h"); st.metric("ΔPrice", f"{change:+.2f}")
        blocks = (p2_dt - p1_dt).total_seconds()/1800
        slope = (p2_price - p1_price)/blocks if blocks>0 else 0.0

        if st.button("Analyze Contract Projections", type="primary", key="ct_generate"):
            rows=[]
            for t in rth_slots_ct(proj_day_ct):
                b = blocks_between(CT_TZ.localize(p1_dt), t)
                price = p1_price + slope*b
                rows.append({'Time': format_ct_time(t), 'Contract_Price': round(price,2), 'Blocks': round(b,1)})
            proj_df = pd.DataFrame(rows)
            st.dataframe(proj_df, use_container_width=True, hide_index=True)

# ───────────────────────────────────────────────────────────────────────────────
# FOOTER QUICK TEST
# ───────────────────────────────────────────────────────────────────────────────
st.markdown("---")
if st.button("Test Data Connection", key="test_connection"):
    with st.spinner("Testing market data connection..."):
        # SPY is more reliable for 30m intraday connectivity test
        test_data = fetch_live_data("SPY", now.date()-timedelta(days=3), now.date())
        if not test_data.empty:
            st.success(f"Connection OK — received {len(test_data)} bars for SPY")
        else:
            st.error("Market data connection failed (empty response). Try again later or adjust dates.")