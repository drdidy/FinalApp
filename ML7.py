# app.py
# ═══════════════════════════════════════════════════════════════════════════════
# 🔮 SPX PROPHET — Unified App (ES engine → SPX display), Light Theme, v1.0
# ═══════════════════════════════════════════════════════════════════════════════

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import pytz
from datetime import datetime, date, time, timedelta

# ───────────────────────────────────────────────────────────────────────────────
# GLOBALS & CONSTANTS
# ───────────────────────────────────────────────────────────────────────────────
CT = pytz.timezone("America/Chicago")
ET = pytz.timezone("US/Eastern")

RTH_START = time(8, 30)   # 08:30 CT
RTH_END   = time(14, 30)  # 14:30 CT

# Fan slope for SPX projections (per 30-min "trading block")
SPX_FAN_SLOPE = 0.333

# Your per-stock slopes (parallel channel; not fan)
STOCK_SLOPES = {
    "TSLA": 0.0285, "NVDA": 0.086, "AAPL": 0.0155,
    "MSFT": 0.0541, "AMZN": 0.0139, "GOOGL": 0.0122,
    "META": 0.0674, "NFLX": 0.0230
}

# ───────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG & STYLES (Light Mode, polished UI)
# ───────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="🔮 SPX Prophet (ES→SPX)",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

LIGHT_CSS = """
<style>
:root { --card-bg: #ffffff; --soft: rgba(0,0,0,0.06); }
.block-container { padding-top: 1.2rem !important; }
.main > div { background: linear-gradient(180deg,#f7f9fc 0%, #eef2f7 100%); }
h1,h2,h3,h4 { color: #0f172a; }
.card {
    background: var(--card-bg); border: 1px solid #e5e7eb; border-radius: 16px;
    padding: 18px 18px; box-shadow: 0 8px 24px rgba(15, 23, 42, 0.06);
}
.metric {
    background: #fff; border: 1px solid #e5e7eb; border-radius: 14px; padding: 12px 16px;
}
.badge { display:inline-block; padding:4px 10px; border-radius:999px; font-size:12px; border:1px solid #e2e8f0; }
.badge.green { background:#ecfdf5; color:#065f46; border-color:#a7f3d0; }
.badge.yellow{ background:#fffbeb; color:#92400e; border-color:#fde68a; }
.badge.red   { background:#fef2f2; color:#991b1b; border-color:#fecaca; }
.table-note { color:#475569; font-size:12px; }
hr { border: none; border-top: 1px solid #e5e7eb; margin: 1rem 0; }
</style>
"""
st.markdown(LIGHT_CSS, unsafe_allow_html=True)

# ───────────────────────────────────────────────────────────────────────────────
# UTILITIES
# ───────────────────────────────────────────────────────────────────────────────
def now_ct():
    return datetime.now(CT)

def localize_to_ct(idx):
    if idx.tz is None:
        return idx.tz_localize(ET).tz_convert(CT)
    return idx.tz_convert(CT)

def between_ct(df, start_str, end_str):
    if df.empty:
        return df
    return df.between_time(start_str, end_str)

def rth_slots(dt_date: date):
    start_dt = CT.localize(datetime.combine(dt_date, RTH_START))
    end_dt   = CT.localize(datetime.combine(dt_date, RTH_END))
    slots = []
    cur = start_dt
    while cur <= end_dt:
        slots.append(cur)
        cur += timedelta(minutes=30)
    return slots

def is_maintenance(dt_ct: datetime) -> bool:
    t = dt_ct.time()
    return t >= time(16,0) and t < time(17,0)

def is_weekend_halt(dt_ct: datetime) -> bool:
    wd = dt_ct.weekday()  # Mon=0 ... Sun=6
    if wd == 4 and dt_ct.time() >= time(16,0):  # Fri after 16:00
        return True
    if wd == 5:  # Saturday all
        return True
    if wd == 6 and dt_ct.time() < time(17,0):  # Sunday before 17:00
        return True
    return False

def count_trading_blocks(start_ct: datetime, end_ct: datetime) -> int:
    """Count 30-min blocks excluding maintenance and weekend halt."""
    if end_ct <= start_ct:
        return 0
    cur = start_ct
    blocks = 0
    while cur < end_ct:
        if not is_maintenance(cur) and not is_weekend_halt(cur):
            blocks += 1
        cur += timedelta(minutes=30)
    return blocks

# ───────────────────────────────────────────────────────────────────────────────
# DATA FETCH
# ───────────────────────────────────────────────────────────────────────────────
@st.cache_data(ttl=120)
def fetch_bars(symbol: str, start_date: date, end_date: date, interval: str) -> pd.DataFrame:
    try:
        start_str = (start_date - timedelta(days=2)).strftime("%Y-%m-%d")
        end_str   = (end_date + timedelta(days=2)).strftime("%Y-%m-%d")
        df = yf.Ticker(symbol).history(
            start=start_str, end=end_str, interval=interval, prepost=True, auto_adjust=False, back_adjust=False
        )
        if df is None or df.empty:
            return pd.DataFrame()
        # Standardize columns
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0] for c in df.columns]
        if "Open" not in df.columns:
            return pd.DataFrame()
        df.index = localize_to_ct(df.index)
        # Filter to requested date(s)
        sdt = CT.localize(datetime.combine(start_date, time(0,0)))
        edt = CT.localize(datetime.combine(end_date, time(23,59)))
        df = df.loc[(df.index >= sdt) & (df.index <= edt)]
        return df
    except Exception:
        return pd.DataFrame()

@st.cache_data(ttl=300)
def es30(prev_day: date, proj_day: date) -> pd.DataFrame:
    return fetch_bars("ES=F", prev_day - timedelta(days=1), proj_day, "30m")

@st.cache_data(ttl=300)
def es1m(day: date) -> pd.DataFrame:
    return fetch_bars("ES=F", day, day, "1m")

@st.cache_data(ttl=300)
def spx30(prev_day: date, proj_day: date) -> pd.DataFrame:
    return fetch_bars("^GSPC", prev_day, proj_day, "30m")

@st.cache_data(ttl=300)
def spy30(prev_day: date, proj_day: date) -> pd.DataFrame:
    return fetch_bars("SPY", prev_day, proj_day, "30m")

@st.cache_data(ttl=300)
def stock30(symbol: str, start_day: date, end_day: date) -> pd.DataFrame:
    return fetch_bars(symbol, start_day, end_day, "30m")

# ───────────────────────────────────────────────────────────────────────────────
# OFFSET & ATR
# ───────────────────────────────────────────────────────────────────────────────
def compute_es_spx_offset(prev_or_proj_day: date) -> float:
    """ES→SPX offset from last RTH close on given day; fallback SPY."""
    es = fetch_bars("ES=F", prev_or_proj_day, prev_or_proj_day, "30m")
    es_rth = between_ct(es, "08:30", "14:30")
    if es_rth.empty:
        return 0.0
    es_close = es_rth.iloc[-1]["Close"]

    spx = spx30(prev_or_proj_day, prev_or_proj_day)
    spx_rth = between_ct(spx, "08:30", "14:30")
    if not spx_rth.empty:
        spx_close = spx_rth.iloc[-1]["Close"]
        return float(spx_close - es_close)

    spy = spy30(prev_or_proj_day, prev_or_proj_day)
    spy_rth = between_ct(spy, "08:30", "14:30")
    if not spy_rth.empty:
        spy_close = spy_rth.iloc[-1]["Close"]
        # SPY ~ (SPX/10) roughly; scale back to SPX (approx). If you prefer strict, set 0.
        # We'll keep conservative: don't scale; better show 0 than mislead.
        return 0.0

    return 0.0

def atr(series_high, series_low, series_close, n=14):
    tr1 = series_high - series_low
    tr2 = (series_high - series_close.shift()).abs()
    tr3 = (series_low  - series_close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(n).mean()

def ema(s, span):
    return s.ewm(span=span, adjust=False).mean()

# ───────────────────────────────────────────────────────────────────────────────
# ANCHOR EXTRACTION (ES space)
# ───────────────────────────────────────────────────────────────────────────────
def prev_day_es_close(df30_prev: pd.DataFrame, prev_day: date):
    rth = between_ct(df30_prev, "08:30", "14:30")
    if rth.empty:
        return None, None
    row = rth.iloc[-1]
    return float(row["Close"]), rth.index[-1]

def prev_day_high_low(df30_prev_full: pd.DataFrame, prev_day: date):
    day_s = CT.localize(datetime.combine(prev_day, time(0,0)))
    day_e = CT.localize(datetime.combine(prev_day, time(23,59)))
    day = df30_prev_full.loc[(df30_prev_full.index>=day_s)&(df30_prev_full.index<=day_e)]
    if day.empty:
        return (None,None,None,None)
    hi_idx = day["High"].idxmax()
    lo_idx = day["Low"].idxmin()
    return float(day.loc[hi_idx,"High"]), hi_idx, float(day.loc[lo_idx,"Low"]), lo_idx

def asian_skyline_baseline(df30_wide: pd.DataFrame, prev_day: date, proj_day: date):
    start = CT.localize(datetime.combine(prev_day, time(17,0)))
    end   = CT.localize(datetime.combine(proj_day, time(7,0)))
    win = df30_wide.loc[(df30_wide.index>=start)&(df30_wide.index<=end)]
    if win.empty:
        return (None,None,None,None)
    hi_idx = win["High"].idxmax()
    lo_idx = win["Low"].idxmin()
    return float(win.loc[hi_idx,"High"]), hi_idx, float(win.loc[lo_idx,"Low"]), lo_idx

# ───────────────────────────────────────────────────────────────────────────────
# PROJECTIONS (ES → SPX via offset)
# ───────────────────────────────────────────────────────────────────────────────
def project_line_from_anchor_es(anchor_price: float, anchor_time: datetime, slope: float, slots: list[datetime]):
    rows = []
    for t in slots:
        blocks = count_trading_blocks(anchor_time, t)
        price_es = anchor_price + slope * blocks
        rows.append({"Time": t, "ES_Price": price_es, "Blocks": blocks})
    return pd.DataFrame(rows)

def build_spx_fan_tables(prev_day: date, proj_day: date, skybas_on=True):
    df30 = es30(prev_day, proj_day)
    if df30.empty:
        return None

    # Anchors
    close_price, close_time = prev_day_es_close(df30, prev_day)
    hi, hi_t, lo, lo_t = prev_day_high_low(df30, prev_day)
    sky, sky_t, base, base_t = asian_skyline_baseline(df30, prev_day, proj_day) if skybas_on else (None,None,None,None)

    # Guard
    if close_price is None or hi is None or lo is None:
        return None

    # Slots for projection day
    slots = rth_slots(proj_day)

    # ES projections
    close_top_es  = project_line_from_anchor_es(close_price, close_time, +SPX_FAN_SLOPE, slots)
    close_bot_es  = project_line_from_anchor_es(close_price, close_time, -SPX_FAN_SLOPE, slots)
    high_line_es  = project_line_from_anchor_es(hi,          hi_t,        +SPX_FAN_SLOPE, slots)
    low_line_es   = project_line_from_anchor_es(lo,          lo_t,        -SPX_FAN_SLOPE, slots)
    sky_line_es   = project_line_from_anchor_es(sky,         sky_t,       +SPX_FAN_SLOPE, slots) if sky is not None else None
    base_line_es  = project_line_from_anchor_es(base,        base_t,      -SPX_FAN_SLOPE, slots) if base is not None else None

    # ES→SPX offset from prev_day RTH
    offset = compute_es_spx_offset(prev_day)

    # Merge & convert to SPX
    df = close_top_es.merge(close_bot_es, on=["Time","Blocks"], suffixes=("_TopES","_BotES"))
    df["SPX_Top"] = df["ES_Price_TopES"] + offset
    df["SPX_Bot"] = df["ES_Price_BotES"] + offset
    df["SPX_Mid"] = (df["SPX_Top"] + df["SPX_Bot"]) / 2.0
    df["SPX_FanWidth"] = (df["SPX_Top"] - df["SPX_Bot"]).abs()

    # Add anchors (SPX)
    df["SPX_High+"] = (high_line_es["ES_Price"] + offset).values
    df["SPX_Low−"]  = (low_line_es["ES_Price"]  + offset).values
    if sky_line_es is not None and base_line_es is not None:
        df["SPX_Skyline"]  = (sky_line_es["ES_Price"]  + offset).values
        df["SPX_Baseline"] = (base_line_es["ES_Price"] + offset).values
    else:
        df["SPX_Skyline"]  = np.nan
        df["SPX_Baseline"] = np.nan

    # Pretty view
    out = pd.DataFrame({
        "Time": [t.strftime("%H:%M") for t in df["Time"]],
        "Top": df["SPX_Top"].round(2),
        "Bottom": df["SPX_Bot"].round(2),
        "Fan_Width": df["SPX_FanWidth"].round(2),
        "Mid(Prev Close)": df["SPX_Mid"].round(2),
        "High(+0.333)": df["SPX_High+"].round(2),
        "Low(−0.333)": df["SPX_Low−"].round(2),
        "Skyline(Asian)": df["SPX_Skyline"].round(2),
        "Baseline(Asian)": df["SPX_Baseline"].round(2),
    })
    meta = {
        "offset": offset,
        "close_time": close_time,
        "hi_time": hi_t,
        "lo_time": lo_t,
        "sky_time": sky_t,
        "base_time": base_t
    }
    return out, df, meta

# ───────────────────────────────────────────────────────────────────────────────
# SIGNALS: TOUCH + SAME-BAR 1-MIN EMA CONFIRMATION (ES engine → SPX)
# ───────────────────────────────────────────────────────────────────────────────
def detect_touch_and_confirmation(prev_day, proj_day, fan_df_internal):
    """fan_df_internal is the ES→SPX merged internal (df), includes ES blocks and SPX levels.
       We use ES 1m for confirmation; display SPX numbers only."""
    # Fetch 30m SPX (for prices to show) via ES+offset: we will use ES 30m but convert closes
    # For touch accuracy, we use the 30m ES bars mapped to SPX via offset
    es30_df = es30(proj_day, proj_day)
    if es30_df.empty:
        return pd.DataFrame()
    es30_rth = between_ct(es30_df, "08:30", "14:30")
    if es30_rth.empty:
        return pd.DataFrame()

    offset_today = compute_es_spx_offset(proj_day)  # for display of SPX price on projection day
    # Tolerance from 30m ATR (ES) mapped to SPX (same difference)
    atr30 = atr(es30_rth["High"], es30_rth["Low"], es30_rth["Close"], 14)
    tol = (atr30.iloc[-1] if not atr30.empty else 0) * 0.30  # 30% ATR
    tol = float(tol) if not np.isnan(tol) else 1.0

    # 1m ES for confirmation
    es1 = es1m(proj_day)
    if es1.empty:
        return pd.DataFrame()
    # 1m EMA
    es1["EMA8"]  = ema(es1["Close"], 8)
    es1["EMA21"] = ema(es1["Close"], 21)

    rows = []
    for i, (t, row) in enumerate(es30_rth.iterrows()):
        time_str = t.strftime("%H:%M")
        spx_price_close = row["Close"] + offset_today
        # Fan lines at this slot
        m = fan_df_internal.loc[fan_df_internal["Time"]==t]
        if m.empty:
            continue
        spx_top = float(m["SPX_Top"])
        spx_bot = float(m["SPX_Bot"])

        # Touch logic on 30m bar (display SPX; compute using ES + offset)
        spx_high = row["High"] + offset_today
        spx_low  = row["Low"]  + offset_today
        spx_open = row["Open"] + offset_today
        spx_close= spx_price_close

        touched_top = spx_high >= spx_top - tol and spx_low <= spx_top + tol
        touched_bot = spx_low  <= spx_bot + tol and spx_high>= spx_bot - tol

        touch_side = None
        if touched_top:
            # Direction of approach
            from_below = spx_open < spx_top
            touch_side = "Top_from_below" if from_below else "Top_from_above"
        elif touched_bot:
            from_above = spx_open > spx_bot
            touch_side = "Bottom_from_above" if from_above else "Bottom_from_below"

        signal = ""
        cross_type = ""
        next_expect = ""
        target = ""
        ema8 = np.nan; ema21 = np.nan

        if touch_side:
            # Same 30m window in 1m
            win = es1.loc[(es1.index >= t) & (es1.index < t + timedelta(minutes=30))]
            if not win.empty and len(win) > 1:
                # Minute-level touch prox check and EMA cross
                # Cross occurs when (EMA8_prev<=EMA21_prev and EMA8>EMA21) or vice versa.
                win["EMA8_prev"] = win["EMA8"].shift()
                win["EMA21_prev"] = win["EMA21"].shift()
                # Map fan to ES by subtracting today's offset (constant); relation preserved either way.
                es_top = spx_top - offset_today
                es_bot = spx_bot - offset_today

                # Check same-bar cross aligned with touch rule
                for idx1m, r1 in win.iloc[1:].iterrows():
                    touched_top_1m = (r1["High"] >= es_top - tol) and (r1["Low"] <= es_top + tol)
                    touched_bot_1m = (r1["Low"]  <= es_bot + tol) and (r1["High"]>= es_bot - tol)
                    bull_cross = (r1["EMA8_prev"] <= r1["EMA21_prev"]) and (r1["EMA8"] > r1["EMA21"])
                    bear_cross = (r1["EMA8_prev"] >= r1["EMA21_prev"]) and (r1["EMA8"] < r1["EMA21"])
                    if touch_side == "Bottom_from_above" and touched_bot_1m and bull_cross:
                        signal     = "BUY (bottom touch + bullish 1m cross)"
                        cross_type = "Bullish"
                        next_expect= "Move to Top"
                        target     = f"{spx_top:.2f}"
                        ema8, ema21= float(r1["EMA8"]), float(r1["EMA21"])
                        break
                    if touch_side == "Bottom_from_below" and touched_bot_1m and bear_cross:
                        signal     = "SELL (bottom touch from below + bearish 1m cross)"
                        cross_type = "Bearish"
                        next_expect= "Continuation lower"
                        target     = f"{spx_bot:.2f}"
                        ema8, ema21= float(r1["EMA8"]), float(r1["EMA21"])
                        break
                    if touch_side == "Top_from_below" and touched_top_1m and bear_cross:
                        signal     = "SELL (top touch + bearish 1m cross)"
                        cross_type = "Bearish"
                        next_expect= "Move to Bottom"
                        target     = f"{spx_bot:.2f}"
                        ema8, ema21= float(r1["EMA8"]), float(r1["EMA21"])
                        break
                    if touch_side == "Top_from_above" and touched_top_1m and bull_cross:
                        signal     = "BUY (top touch from above + bullish 1m cross)"
                        cross_type = "Bullish"
                        next_expect= "Breakout / continuation"
                        target     = f"{spx_top:.2f}"
                        ema8, ema21= float(r1["EMA8"]), float(r1["EMA21"])
                        break

        rows.append({
            "Time": time_str,
            "Price": round(spx_price_close,2),
            "Top": round(spx_top,2), "Bottom": round(spx_bot,2),
            "TouchSide": touch_side or "",
            "1m_EMA8": round(ema8,2) if not np.isnan(ema8) else "",
            "1m_EMA21": round(ema21,2) if not np.isnan(ema21) else "",
            "CrossType": cross_type,
            "Signal": signal,
            "Next_Expectation": next_expect,
            "Target": target
        })
    return pd.DataFrame(rows)

# ───────────────────────────────────────────────────────────────────────────────
# STRATEGY TABLE (SPX display)
# ───────────────────────────────────────────────────────────────────────────────
def build_strategy_table_spx(proj_day: date, fan_pretty: pd.DataFrame, fan_internal: pd.DataFrame):
    # Price per slot from ES 30m mapped to SPX (display only)
    es = es30(proj_day, proj_day)
    if es.empty:
        return pd.DataFrame()
    rth = between_ct(es, "08:30", "14:30")
    if rth.empty:
        return pd.DataFrame()
    offset_today = compute_es_spx_offset(proj_day)

    rows = []
    for t,row in rth.iterrows():
        spx_price = row["Close"] + offset_today
        time_str = t.strftime("%H:%M")
        f = fan_internal.loc[fan_internal["Time"]==t]
        if f.empty:
            continue
        top = float(f["SPX_Top"]); bot = float(f["SPX_Bot"])
        fan_w = float((top - bot))
        if spx_price > top:
            bias = "UP"
            entry_side = "SELL (MR)"
            mr_trigger = float(f["SPX_High+"])
            entry = mr_trigger
            tp1 = top
            tp2 = top - fan_w
            note = "Above fan → mean reversion via High(+)"
        elif spx_price < bot:
            bias = "DOWN"
            entry_side = "BUY (MR)"
            mr_trigger = float(f["SPX_Low−"])
            entry = mr_trigger
            tp1 = bot
            tp2 = bot + fan_w
            note = "Below fan → mean reversion via Low(−)"
        else:
            bias = "RANGE"
            # inside fan: fade edges (direction chooses edge)
            if abs(spx_price - bot) < abs(spx_price - top):
                entry_side = "BUY (fade)"
                mr_trigger = bot
                entry = bot
                tp1 = top
                tp2 = top  # edge-to-edge simple guide
                note = "Inside fan → fade bottom to top"
            else:
                entry_side = "SELL (fade)"
                mr_trigger = top
                entry = top
                tp1 = bot
                tp2 = bot
                note = "Inside fan → fade top to bottom"

        rows.append({
            "Time": time_str, "Price": round(spx_price,2),
            "Bias": bias, "EntrySide": entry_side,
            "Entry": round(entry,2), "MR_Trigger": round(mr_trigger,2),
            "TP1_Mean": round(tp1,2), "TP2_Ext": round(tp2,2),
            "Top": round(top,2), "Bottom": round(bot,2),
            "Fan_Width": round(fan_w,2), "Note": note
        })
    return pd.DataFrame(rows)

# ───────────────────────────────────────────────────────────────────────────────
# STOCK PARALLEL CHANNEL (Mon/Tue) with per-stock slopes
# ───────────────────────────────────────────────────────────────────────────────
def stock_parallel_channel(symbol: str, mon: date, tue: date, slope: float):
    df = stock30(symbol, mon, tue)
    if df.empty:
        return None, None
    # swings over combined window
    hi_idx = df["High"].idxmax()
    lo_idx = df["Low"].idxmin()
    hi_price, hi_time = float(df.loc[hi_idx,"High"]), hi_idx
    lo_price, lo_time = float(df.loc[lo_idx,"Low"]),  lo_idx
    # project forward for rest of week (Wed-Fri)
    start_day = tue + timedelta(days=1)
    slots = rth_slots(start_day) + rth_slots(start_day+timedelta(days=1)) + rth_slots(start_day+timedelta(days=2))
    rows = []
    for t in slots:
        blocks_hi = count_trading_blocks(hi_time, t)
        blocks_lo = count_trading_blocks(lo_time, t)
        hi_line = hi_price + slope * blocks_hi
        lo_line = lo_price + slope * blocks_lo  # same slope → parallel
        rows.append({"Time": t.strftime("%a %H:%M"), "Upper": round(hi_line,2), "Lower": round(lo_line,2)})
    chan = pd.DataFrame(rows)
    info = {"High":(hi_price,hi_time), "Low":(lo_price,lo_time)}
    return chan, info

# ───────────────────────────────────────────────────────────────────────────────
# CONTRACT TOOL (0–30), slope by trading blocks
# ───────────────────────────────────────────────────────────────────────────────
def contract_projection(p1_dt: datetime, p1_price: float, p2_dt: datetime, p2_price: float, proj_day: date):
    if p2_dt <= p1_dt: return None
    if not (0.0 <= p1_price <= 30.0 and 0.0 <= p2_price <= 30.0): return None
    blocks = count_trading_blocks(p1_dt, p2_dt)
    slope = (p2_price - p1_price)/blocks if blocks>0 else 0.0
    rows=[]
    for t in rth_slots(proj_day):
        b = count_trading_blocks(p1_dt, t)
        price = p1_price + slope*b
        rows.append({"Time": t.strftime("%H:%M"), "Contract_Price": round(price,2), "Blocks": b})
    return pd.DataFrame(rows), slope

# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR (controls)
# ═══════════════════════════════════════════════════════════════════════════════
st.sidebar.title("🌞 SPX Prophet")
st.sidebar.caption("ES engine → SPX display (light mode)")

skybas_toggle = st.sidebar.checkbox("Show Skyline/Baseline (Asian session)", value=True)
filters_toggle = st.sidebar.checkbox("Enable probability filters (touch quality, volume, etc.)", value=False)

st.sidebar.markdown("---")
if st.sidebar.button("🔌 Test Data Connection"):
    test = es30(now_ct().date()-timedelta(days=1), now_ct().date())
    if test is None or test.empty:
        st.error("Market data connection failed (ES=F 30m).")
    else:
        st.success(f"Connection OK — ES=F returned {len(test)} bars.")

# ═══════════════════════════════════════════════════════════════════════════════
# HEADER
# ═══════════════════════════════════════════════════════════════════════════════
colA, colB, colC, colD = st.columns(4)
with colA:
    st.markdown('<div class="metric"><b>⏰ CT Time</b><br>' + now_ct().strftime("%a %b %d, %H:%M:%S") + "</div>", unsafe_allow_html=True)
with colB:
    sess = "RTH" if RTH_START <= now_ct().time() <= RTH_END and now_ct().weekday()<5 else "Closed"
    badge = "green" if sess=="RTH" else "yellow"
    st.markdown(f'<div class="metric"><b>📊 Session</b><br><span class="badge {badge}">{sess}</span></div>', unsafe_allow_html=True)
with colC:
    st.markdown(f'<div class="metric"><b>📏 Fan Slope</b><br>±{SPX_FAN_SLOPE:.3f} / 30m</div>', unsafe_allow_html=True)
with colD:
    st.markdown(f'<div class="metric"><b>🎯 Stocks</b><br>Parallel channels (your slopes)</div>', unsafe_allow_html=True)

st.markdown("<hr/>", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TABS
# ═══════════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4 = st.tabs(["🧭 SPX Anchors", "🏢 Stock Anchors", "📶 Signals & EMA", "🧮 Contract Tool"])

# ───────────────────────────────────────────────────────────────────────────────
# TAB 1 — SPX Anchors (ES engine → SPX display)
# ───────────────────────────────────────────────────────────────────────────────
with tab1:
    st.subheader("🧭 SPX Anchors (ES→SPX)")
    c1, c2 = st.columns(2)
    with c1:
        prev_day = st.date_input("Previous Trading Day", value=now_ct().date()-timedelta(days=1))
    with c2:
        proj_day = st.date_input("Projection Day", value=prev_day+timedelta(days=1))

    res = build_spx_fan_tables(prev_day, proj_day, skybas_toggle)
    if res is None:
        st.error("Unable to build fan — missing data.")
    else:
        fan_table, fan_internal, meta = res
        offset_val = meta["close_time"] and compute_es_spx_offset(prev_day) or 0.0

        st.markdown(f'<div class="card"><b>🔁 ES→SPX Offset (prev day RTH close):</b> {offset_val:+.2f}</div>', unsafe_allow_html=True)
        st.markdown("### 🪜 Fan & Anchors (SPX)")
        st.dataframe(fan_table, use_container_width=True, hide_index=True)

        # Strategy table
        st.markdown("### 🧠 Strategy Table (SPX)")
        strat = build_strategy_table_spx(proj_day, fan_table, fan_internal)
        if strat.empty:
            st.info("No RTH data found for projection day.")
        else:
            st.dataframe(strat, use_container_width=True, hide_index=True)
            st.caption("Bias: Above fan → UP, Below fan → DOWN, Inside → RANGE.")

        st.markdown("### ⚡ Touch + Same-bar 1-Minute Confirmation (SPX)")
        sigs = detect_touch_and_confirmation(prev_day, proj_day, fan_internal)
        if sigs.empty:
            st.info("No touch+confirmation events detected for this day.")
        else:
            st.dataframe(sigs, use_container_width=True, hide_index=True)
            st.caption("Touch tolerance = 0.30 × ATR(14) on 30m. Confirmation = 1m EMA(8/21) cross on the same 1m bar within the 30m slot.")

# ───────────────────────────────────────────────────────────────────────────────
# TAB 2 — Stock Anchors (Parallel Channel; no fan; your slopes)
# ───────────────────────────────────────────────────────────────────────────────
with tab2:
    st.subheader("🏢 Stock Anchors (Parallel Channel)")
    core = ["TSLA","NVDA","AAPL","MSFT","AMZN","GOOGL","META","NFLX"]
    tcol = st.columns(8)
    sel = st.text_input("Symbol (or click a button below)", value="")
    for i,tkr in enumerate(core):
        if tcol[i].button(tkr):
            sel = tkr
    if not sel:
        st.info("Pick a symbol or type one above.")
    else:
        slope = STOCK_SLOPES.get(sel.upper(), 0.02)
        st.caption(f"Using slope: {slope:.4f} per 30-min block (parallel channel)")
        c1,c2,c3 = st.columns(3)
        with c1:
            mon = st.date_input("Monday", value=now_ct().date()-timedelta(days=7))
        with c2:
            tue = st.date_input("Tuesday", value=mon+timedelta(days=1))
        with c3:
            st.write("Projection: Wed–Fri (auto)")

        ch, info = stock_parallel_channel(sel.upper(), mon, tue, slope)
        if ch is None:
            st.error("Not enough data for the selected dates.")
        else:
            st.markdown("### 📐 Parallel Channel (Upper=Mon/Tue highest swing, Lower=Mon/Tue lowest swing)")
            st.dataframe(ch, use_container_width=True, hide_index=True)
            st.caption(f"High anchor @ {info['High'][0]:.2f} ({info['High'][1]}), Low anchor @ {info['Low'][0]:.2f} ({info['Low'][1]})")

# ───────────────────────────────────────────────────────────────────────────────
# TAB 3 — Signals & EMA (Quick RTH EMA dashboard)
# ───────────────────────────────────────────────────────────────────────────────
with tab3:
    st.subheader("📶 Signals & EMA Dashboard")
    sym = st.selectbox("Symbol", ["^GSPC (SPX display via ES)", "ES=F (engine)", "SPY", "TSLA","NVDA","AAPL","MSFT","AMZN","GOOGL","META","NFLX"], index=0)
    day = st.date_input("Analysis Day", value=now_ct().date())
    if st.button("Compute EMA 8/21"):
        if "ES=F" in sym or "SPX" in sym:
            df = es30(day, day)
            off = compute_es_spx_offset(day)
            if df.empty:
                st.error("No ES data for selected day.")
            else:
                rth = between_ct(df, "08:30", "14:30")
                if rth.empty:
                    st.info("No RTH bars.")
                else:
                    spx_close = rth["Close"] + off
                    e8  = ema(spx_close, 8)
                    e21 = ema(spx_close, 21)
                    out = pd.DataFrame({
                        "Time":[t.strftime("%H:%M") for t in rth.index],
                        "Price": spx_close.round(2),
                        "EMA8": e8.round(2), "EMA21": e21.round(2),
                        "Separation_%": ((e8-e21)/e21*100).round(3),
                        "Regime": np.where(e8>e21, "Bullish", "Bearish")
                    })
                    prev8  = e8.shift()
                    prev21 = e21.shift()
                    cross = np.where((prev8<=prev21)&(e8>e21),"Bullish Cross",
                            np.where((prev8>=prev21)&(e8<e21),"Bearish Cross","None"))
                    out["Crossover"] = cross
                    st.dataframe(out, use_container_width=True, hide_index=True)
        else:
            df = stock30(sym, day, day)
            if df.empty:
                st.error("No data for selected stock/day.")
            else:
                rth = between_ct(df, "08:30", "14:30")
                if rth.empty:
                    st.info("No RTH bars.")
                else:
                    e8  = ema(rth["Close"], 8)
                    e21 = ema(rth["Close"], 21)
                    out = pd.DataFrame({
                        "Time":[t.strftime("%H:%M") for t in rth.index],
                        "Price": rth["Close"].round(2),
                        "EMA8": e8.round(2), "EMA21": e21.round(2),
                        "Separation_%": ((e8-e21)/e21*100).round(3),
                        "Regime": np.where(e8>e21, "Bullish", "Bearish")
                    })
                    prev8  = e8.shift()
                    prev21 = e21.shift()
                    cross = np.where((prev8<=prev21)&(e8>e21),"Bullish Cross",
                            np.where((prev8>=prev21)&(e8<e21),"Bearish Cross","None"))
                    out["Crossover"] = cross
                    st.dataframe(out, use_container_width=True, hide_index=True)

# ───────────────────────────────────────────────────────────────────────────────
# TAB 4 — Contract Tool (0–30 inputs)
# ───────────────────────────────────────────────────────────────────────────────
with tab4:
    st.subheader("🧮 Contract Tool (0–30)")
    c1,c2,c3 = st.columns(3)
    with c1:
        p1_date = st.date_input("Point 1 Date", value=now_ct().date()-timedelta(days=1))
        p1_time = st.time_input("Point 1 Time (CT)", value=time(20,0))
        p1_price = st.number_input("Point 1 Price", min_value=0.00, max_value=30.00, value=10.00, step=0.01)
    with c2:
        p2_date = st.date_input("Point 2 Date", value=now_ct().date())
        p2_time = st.time_input("Point 2 Time (CT)", value=time(8,0))
        p2_price = st.number_input("Point 2 Price", min_value=0.00, max_value=30.00, value=12.00, step=0.01)
    with c3:
        proj = st.date_input("RTH Projection Day", value=p2_date)

    if st.button("Project Contract"):
        p1_dt = CT.localize(datetime.combine(p1_date, p1_time))
        p2_dt = CT.localize(datetime.combine(p2_date, p2_time))
        out = contract_projection(p1_dt, p1_price, p2_dt, p2_price, proj)
        if out is None:
            st.error("Invalid inputs (order, or values outside 0–30, or no blocks).")
        else:
            table, slope = out
            st.markdown(f'<div class="card"><b>📐 Observed Slope:</b> {slope:+.4f} per 30-min trading block</div>', unsafe_allow_html=True)
            st.dataframe(table, use_container_width=True, hide_index=True)

# ═══════════════════════════════════════════════════════════════════════════════
# FOOTER
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("<hr/>", unsafe_allow_html=True)
st.markdown(
    f"<div style='text-align:center;color:#64748b'>🔮 SPX Prophet • ES engine → SPX display • "
    f"{now_ct().strftime('%Y-%m-%d %H:%M:%S CT')}</div>", unsafe_allow_html=True
)