# app.py
# ═══════════════════════════════════════════════════════════════════════════════
# 🔮 SPX PROPHET — FULL APP (Manual Close Option)
# Slope fixed at ±0.277 per 30-minute block; 3:00 PM CT close as anchor
# Skips maintenance hour (4–5 PM CT) and Fri 5 PM → Sun 5 PM weekend gap
# You can manually input the previous day's 3:00 PM CT close (no Yahoo dependency)
# ═══════════════════════════════════════════════════════════════════════════════

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import pytz
from datetime import datetime, date, time, timedelta
from typing import Dict, List, Optional, Tuple

# ───────────────────────────────────────────────────────────────────────────────
# CORE CONFIG
# ───────────────────────────────────────────────────────────────────────────────
CT_TZ = pytz.timezone("America/Chicago")
SLOPE_PER_BLOCK = 0.277  # per 30-minute block (Top +, Bottom −)
RTH_START = "08:30"
RTH_END = "14:30"

# Stock per-ticker slopes (magnitude) — your values
STOCK_SLOPES = {
    "TSLA": 0.0285, "NVDA": 0.086, "AAPL": 0.0155,
    "MSFT": 0.0541, "AMZN": 0.0139, "GOOGL": 0.0122,
    "META": 0.0674, "NFLX": 0.0230
}

# ───────────────────────────────────────────────────────────────────────────────
# PAGE & THEME (light, enterprise vibe)
# ───────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="🔮 SPX Prophet Analytics",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
:root {
  --brand: #2563eb; --brand-2: #10b981;
  --surface: #ffffff; --muted: #f8fafc;
  --text: #0f172a; --subtext: #475569; --border: #e2e8f0;
  --warn: #f59e0b; --danger: #ef4444; --ok: #22c55e;
}
html, body, [class*="css"]  { background: var(--muted); color: var(--text); }
.block-container { padding-top: 1.2rem; }

.metric-card {
  background: var(--surface); border: 1px solid var(--border);
  border-radius: 16px; padding: 16px;
  box-shadow: 0 10px 24px rgba(2,6,23,0.06);
}
.metric-title { font-size: 0.9rem; color: var(--subtext); margin: 0; }
.metric-value { font-size: 1.8rem; font-weight: 700; margin-top: 6px; display:flex; gap:10px; align-items:center; }
.kicker { font-size: 0.8rem; color: var(--subtext); }

.badge {
  padding: 2px 10px; border-radius: 999px; font-size: 0.8rem; font-weight: 700; border: 1px solid;
}
.badge-open  { color:#065f46; background:#d1fae5; border-color:#99f6e4; }
.badge-closed{ color:#7c2d12; background:#ffedd5; border-color:#fed7aa; }

.section-card{
  background: var(--surface); border: 1px solid var(--border);
  border-radius: 16px; padding: 16px; margin-top: 10px;
  box-shadow: 0 10px 24px rgba(2,6,23,0.06);
}
hr { border-top: 1px solid var(--border); }
.dataframe { background: var(--surface); border-radius: 12px; overflow: hidden; }
</style>
""",
    unsafe_allow_html=True,
)

# ───────────────────────────────────────────────────────────────────────────────
# HELPERS (Time, Maintenance/Weekend, Fetch, Projections)
# ───────────────────────────────────────────────────────────────────────────────
def fmt_ct(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return CT_TZ.localize(dt)
    return dt.astimezone(CT_TZ)

def between_time(df: pd.DataFrame, start_str: str, end_str: str) -> pd.DataFrame:
    return df.between_time(start_str, end_str) if not df.empty else df

def rth_slots_ct(target_date: date) -> List[datetime]:
    start_dt = fmt_ct(datetime.combine(target_date, time(8, 30)))
    end_dt = fmt_ct(datetime.combine(target_date, time(14, 30)))
    slots = []
    cur = start_dt
    while cur <= end_dt:
        slots.append(cur); cur += timedelta(minutes=30)
    return slots

def is_maintenance(dt: datetime) -> bool:
    # 4–5 PM CT is closed
    return dt.hour == 16

def in_weekend_gap(dt: datetime) -> bool:
    wd = dt.weekday()  # Mon=0 ... Sun=6
    if wd == 5: return True  # Saturday
    if wd == 6 and dt.hour < 17: return True  # Sunday before 5 PM
    if wd == 4 and dt.hour >= 17: return True  # Friday at/after 5 PM
    return False

def count_effective_blocks(anchor_time: datetime, target_time: datetime) -> float:
    if target_time <= anchor_time: return 0.0
    t = anchor_time; blocks = 0
    while t < target_time:
        t_next = t + timedelta(minutes=30)
        if not is_maintenance(t_next) and not in_weekend_gap(t_next):
            blocks += 1
        t = t_next
    return float(blocks)

def ensure_ohlc_cols(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty: return df
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] if isinstance(c, tuple) else str(c) for c in df.columns]
    required = ["Open", "High", "Low", "Close", "Volume"]
    if any(c not in df.columns for c in required): return pd.DataFrame()
    return df

@st.cache_data(ttl=120)
def fetch_intraday(symbol: str, start_d: date, end_d: date) -> pd.DataFrame:
    def normalize(df):
        if df.empty: return df
        df = ensure_ohlc_cols(df)
        if df.empty: return df
        if df.index.tz is None:
            df.index = df.index.tz_localize("US/Eastern")
        df.index = df.index.tz_convert(CT_TZ)
        sdt = fmt_ct(datetime.combine(start_d, time(0, 0)))
        edt = fmt_ct(datetime.combine(end_d, time(23, 59)))
        return df.loc[sdt:edt]
    try:
        t = yf.Ticker(symbol)
        df = t.history(
            start=(start_d - timedelta(days=5)).strftime("%Y-%m-%d"),
            end=(end_d + timedelta(days=2)).strftime("%Y-%m-%d"),
            interval="30m", prepost=True, auto_adjust=False, back_adjust=False
        )
        df = normalize(df)
        if df.empty:
            days = max(7, (end_d - start_d).days + 7)
            df2 = t.history(period=f"{days}d", interval="30m",
                            prepost=True, auto_adjust=False, back_adjust=False)
            df = normalize(df2)
        return df
    except Exception:
        return pd.DataFrame()

def get_prev_day_3pm_close(spx_prev: pd.DataFrame, prev_day: date) -> Optional[float]:
    """Kept for fallback when you are not in manual mode."""
    if spx_prev.empty: return None
    day_start = fmt_ct(datetime.combine(prev_day, time(0, 0)))
    day_end = fmt_ct(datetime.combine(prev_day, time(23, 59)))
    d = spx_prev.loc[day_start:day_end]
    if d.empty: return None
    target = fmt_ct(datetime.combine(prev_day, time(15, 0)))
    if target in d.index:
        return float(d.loc[target, "Close"])
    prior = d.loc[:target]
    if not prior.empty:
        return float(prior.iloc[-1]["Close"])
    return None

def project_fan_from_close(close_price: float, anchor_time: datetime, target_day: date) -> pd.DataFrame:
    rows = []
    for slot in rth_slots_ct(target_day):
        blocks = count_effective_blocks(anchor_time, slot)
        top = close_price + SLOPE_PER_BLOCK * blocks
        bot = close_price - SLOPE_PER_BLOCK * blocks
        rows.append({"Time": slot.strftime("%H:%M"), "Top": round(top,2), "Bottom": round(bot,2), "Fan_Width": round(top-bot,2)})
    return pd.DataFrame(rows)

# ───────────────────────────────────────────────────────────────────────────────
# STRATEGY TABLE (SPX — simplified rules)
# ───────────────────────────────────────────────────────────────────────────────
def build_strategy_table(rth_prices: pd.DataFrame, fan_df: pd.DataFrame, anchor_close: float) -> pd.DataFrame:
    if rth_prices.empty or fan_df.empty: return pd.DataFrame()
    price_lu = {dt.strftime("%H:%M"): float(rth_prices.loc[dt, "Close"]) for dt in rth_prices.index}
    rows=[]
    for _, row in fan_df.iterrows():
        t = row["Time"]
        if t not in price_lu: continue
        p = price_lu[t]; top=row["Top"]; bot=row["Bottom"]; width=row["Fan_Width"]
        bias = "UP" if p >= anchor_close else "DOWN"
        direction=""; entry=np.nan; tp1=np.nan; tp2=np.nan

        if bot <= p <= top:
            if bias=="UP":
                direction="BUY"; entry=bot; tp1=top; tp2=top
            else:
                direction="SELL"; entry=top; tp1=bot; tp2=bot
        elif p > top:
            direction="SELL"; entry=top; tp2=top - width
        else:
            direction="SELL"; entry=bot; tp2=bot - width

        rows.append({
            "Time": t, "Price": round(p,2), "Bias": bias, "EntrySide": direction,
            "Entry": round(entry,2) if not np.isnan(entry) else np.nan,
            "TP1": round(tp1,2) if not np.isnan(tp1) else np.nan,
            "TP2": round(tp2,2) if not np.isnan(tp2) else np.nan,
            "Top": round(top,2), "Bottom": round(bot,2)
        })
    return pd.DataFrame(rows)

# ───────────────────────────────────────────────────────────────────────────────
# STOCK ANCHORS (Mon/Tue swings, no fan — just two lines)
# ───────────────────────────────────────────────────────────────────────────────
def get_mon_tue_data(ticker: str, mon: date, tue: date) -> pd.DataFrame:
    mon_df = fetch_intraday(ticker, mon, mon)
    tue_df = fetch_intraday(ticker, tue, tue)
    if mon_df.empty and tue_df.empty: return pd.DataFrame()
    if mon_df.empty: return tue_df
    if tue_df.empty: return mon_df
    return pd.concat([mon_df, tue_df]).sort_index()

def extract_high_low_anchor(data: pd.DataFrame) -> Optional[Tuple[Tuple[float, datetime], Tuple[float, datetime]]]:
    if data.empty: return None
    high_price = data["High"].max(); high_time = data[data["High"]==high_price].index[0]
    low_price  = data["Low"].min();  low_time  = data[data["Low"]==low_price].index[0]
    return ( (float(high_price), high_time), (float(low_price), low_time) )

def project_line_from(price: float, anchor_time: datetime, slope_per_block: float,
                      target_day: date, col_name: str) -> pd.DataFrame:
    rows=[]
    for slot in rth_slots_ct(target_day):
        blocks = count_effective_blocks(anchor_time, slot)
        pr = price + slope_per_block*blocks
        rows.append({"Time": slot.strftime("%H:%M"), col_name: round(pr,2)})
    return pd.DataFrame(rows)

# ───────────────────────────────────────────────────────────────────────────────
# SIGNALS (touch confirmation) & EMA (8/21 on 30m)
# ───────────────────────────────────────────────────────────────────────────────
def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span).mean()

def detect_touches(rth_prices: pd.DataFrame, fan_df: pd.DataFrame) -> pd.DataFrame:
    if rth_prices.empty or fan_df.empty: return pd.DataFrame()
    top_lu = {r["Time"]: r["Top"] for _, r in fan_df.iterrows()}
    bot_lu = {r["Time"]: r["Bottom"] for _, r in fan_df.iterrows()}
    rows=[]
    for dt, row in rth_prices.iterrows():
        t = dt.strftime("%H:%M")
        if t not in top_lu: continue
        top = top_lu[t]; bot = bot_lu[t]
        touched_top = row["High"] >= top >= row["Low"]
        touched_bot = row["High"] >= bot >= row["Low"]
        if touched_top or touched_bot:
            rows.append({
                "Time": t,
                "Touched": "Top" if touched_top else "Bottom",
                "Open": round(row["Open"],2), "High": round(row["High"],2),
                "Low": round(row["Low"],2), "Close": round(row["Close"],2)
            })
    return pd.DataFrame(rows)

def confirmation_from_ema(rth_prices: pd.DataFrame) -> pd.DataFrame:
    if rth_prices.empty or len(rth_prices)<21: return pd.DataFrame()
    e8 = ema(rth_prices["Close"], 8); e21 = ema(rth_prices["Close"], 21)
    rows=[]
    for i in range(1, len(rth_prices)):
        dt = rth_prices.index[i]
        t = dt.strftime("%H:%M"); p=float(rth_prices.iloc[i]["Close"])
        prev_cross = (e8.iloc[i-1] - e21.iloc[i-1])
        curr_cross = (e8.iloc[i]   - e21.iloc[i])
        cross=None
        if prev_cross <= 0 and curr_cross > 0: cross="Bullish Cross"
        elif prev_cross >= 0 and curr_cross < 0: cross="Bearish Cross"
        regime = "Bullish" if curr_cross>0 else "Bearish"
        rows.append({"Time": t, "Price": round(p,2), "EMA8": round(e8.iloc[i],2),
                     "EMA21": round(e21.iloc[i],2), "Regime": regime, "Crossover": cross or "None"})
    return pd.DataFrame(rows)

# ───────────────────────────────────────────────────────────────────────────────
# SIDEBAR (global controls)
# ───────────────────────────────────────────────────────────────────────────────
st.sidebar.title("🧭 Navigation & Controls")
today_ct = datetime.now(CT_TZ).date()

# SPX dates + manual anchor
prev_day = st.sidebar.date_input("SPX • Previous Trading Day", value=today_ct - timedelta(days=1), key="spx_prev")
proj_day = st.sidebar.date_input("SPX • Projection Day", value=prev_day + timedelta(days=1), key="spx_proj")
st.sidebar.caption("SPX fan anchored at **3:00 PM CT** of the previous trading day.")

use_manual_close = st.sidebar.checkbox("Use manual 3:00 PM CT close", value=False)
manual_close_val = None
if use_manual_close:
    manual_close_val = st.sidebar.number_input("Enter manual close (3:00 PM CT)", value=6400.00, step=0.10, format="%.2f")

st.sidebar.markdown("---")
st.sidebar.subheader("Stocks (Mon/Tue)")
mon_date = st.sidebar.date_input("Monday Date", value=today_ct - timedelta(days=2), key="mon_date")
tue_date = st.sidebar.date_input("Tuesday Date", value=mon_date + timedelta(days=1), key="tue_date")
stock_choice = st.sidebar.selectbox("Ticker", ["TSLA","NVDA","AAPL","MSFT","AMZN","GOOGL","META","NFLX","(custom)"], index=0)
custom_ticker = st.sidebar.text_input("Custom symbol", value="", key="custom_tkr") if stock_choice=="(custom)" else ""
st.sidebar.markdown("---")

st.sidebar.subheader("Signals & EMA")
sig_symbol = st.sidebar.selectbox("Symbol", ["^GSPC","ES=F","SPY"], index=0, key="sig_symbol")
sig_day    = st.sidebar.date_input("Analysis Day", value=today_ct, key="sig_day")

st.sidebar.markdown("---")
st.sidebar.subheader("Contract Tool")
p1_date = st.sidebar.date_input("Point 1 Date", value=today_ct - timedelta(days=1), key="ct_p1d")
p1_time = st.sidebar.time_input("Point 1 Time (CT)", value=time(20,0), key="ct_p1t")
p1_price= st.sidebar.number_input("Point 1 Contract Price", value=10.0, min_value=0.01, step=0.01, format="%.2f", key="ct_p1p")
p2_date = st.sidebar.date_input("Point 2 Date", value=today_ct, key="ct_p2d")
p2_time = st.sidebar.time_input("Point 2 Time (CT)", value=time(8,0), key="ct_p2t")
p2_price= st.sidebar.number_input("Point 2 Contract Price", value=12.0, min_value=0.01, step=0.01, format="%.2f", key="ct_p2p")
ct_proj  = st.sidebar.date_input("RTH Projection Day", value=p2_date, key="ct_proj")

st.sidebar.markdown("---")
btn_spx     = st.sidebar.button("🔮 Generate SPX Fan & Strategy", type="primary")
btn_stocks  = st.sidebar.button("🏢 Analyze Stock Anchors")
btn_signals = st.sidebar.button("📡 Analyze Signals & EMA")
btn_ct      = st.sidebar.button("📈 Analyze Contract Projection")

# ───────────────────────────────────────────────────────────────────────────────
# HEADER METRICS
# ───────────────────────────────────────────────────────────────────────────────
c1, c2, c3 = st.columns(3)
now = datetime.now(CT_TZ)
with c1:
    st.markdown(
        f"""
<div class="metric-card">
  <p class="metric-title">Current Time (CT)</p>
  <div class="metric-value">🕒 {now.strftime("%H:%M:%S")}</div>
  <div class="kicker">{now.strftime("%A, %B %d, %Y")}</div>
</div>""", unsafe_allow_html=True)
with c2:
    is_wkday = now.weekday() < 5
    open_dt = now.replace(hour=8, minute=30, second=0, microsecond=0)
    close_dt = now.replace(hour=14, minute=30, second=0, microsecond=0)
    is_open = is_wkday and (open_dt <= now <= close_dt)
    badge = "badge-open" if is_open else "badge-closed"
    text = "Market Open" if is_open else "Closed"
    st.markdown(
        f"""
<div class="metric-card">
  <p class="metric-title">Market Status</p>
  <div class="metric-value">📊 <span class="badge {badge}">{text}</span></div>
  <div class="kicker">RTH: 08:30–14:30 CT • Mon–Fri</div>
</div>""", unsafe_allow_html=True)
with c3:
    st.markdown(
        f"""
<div class="metric-card">
  <p class="metric-title">Slope / 30-min Block</p>
  <div class="metric-value">📐 ±{SLOPE_PER_BLOCK:.3f}</div>
  <div class="kicker">Top = +slope • Bottom = −slope</div>
</div>""", unsafe_allow_html=True)

st.markdown("---")

# ───────────────────────────────────────────────────────────────────────────────
# TABS
# ───────────────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["SPX Anchors", "Stock Anchors", "Signals & EMA", "Contract Tool"])

# ╔═ TAB 1: SPX ANCHORS ════════════════════════════════════════════════════════╗
with tab1:
    st.subheader("SPX Close-Anchor Fan (3:00 PM CT)")

    if btn_spx:
        with st.spinner("Building SPX fan & strategy…"):
            # If manual close is used: we don't need prev-day fetch
            anchor_close = None
            anchor_time = fmt_ct(datetime.combine(prev_day, time(15, 0)))

            if use_manual_close and manual_close_val is not None:
                anchor_close = float(manual_close_val)
            else:
                # fallback to fetch previous day close if manual is OFF
                spx_prev = fetch_intraday("^GSPC", prev_day, prev_day)
                if spx_prev.empty:
                    spx_prev = fetch_intraday("SPY", prev_day, prev_day)
                anchor_close = get_prev_day_3pm_close(spx_prev, prev_day) if not spx_prev.empty else None

            spx_proj_df = fetch_intraday("^GSPC", proj_day, proj_day)
            if spx_proj_df.empty:
                spx_proj_df = fetch_intraday("SPY", proj_day, proj_day)

            if anchor_close is None or spx_proj_df.empty:
                st.error("❌ Missing anchor close or projection-day data.")
            else:
                fan_df = project_fan_from_close(anchor_close, anchor_time, proj_day)

                rth_data = between_time(spx_proj_df, RTH_START, RTH_END)
                strat_df = build_strategy_table(rth_data, fan_df, anchor_close) if not rth_data.empty else pd.DataFrame()

                st.markdown(f"**Anchor (Prev Day 3:00 PM CT) Close:** {anchor_close:.2f}")
                st.markdown("### 🎯 Fan Lines")
                st.dataframe(fan_df, use_container_width=True, hide_index=True)

                st.markdown("### 📋 Strategy Table")
                if not strat_df.empty:
                    st.dataframe(strat_df, use_container_width=True, hide_index=True)
                else:
                    st.info("No RTH alignment for the projection day (check session times).")
    else:
        st.info("Pick dates (and optionally a manual close) in the sidebar, then click **Generate SPX Fan & Strategy**.")

# ╔═ TAB 2: STOCK ANCHORS ══════════════════════════════════════════════════════╗
with tab2:
    st.subheader("Stock Anchors (Mon/Tue Swings → Parallel Lines)")

    if btn_stocks:
        ticker = custom_ticker.upper() if stock_choice=="(custom)" and custom_ticker else stock_choice
        with st.spinner(f"Analyzing {ticker} Mon/Tue swings…"):
            combo = get_mon_tue_data(ticker, mon_date, tue_date)
            if combo.empty:
                st.error(f"No 30-min data for {ticker} on Mon/Tue.")
            else:
                anchors = extract_high_low_anchor(combo)
                if anchors is None:
                    st.error("Could not extract swing anchors.")
                else:
                    (hi_price, hi_time), (lo_price, lo_time) = anchors
                    slope_mag = STOCK_SLOPES.get(ticker, list(STOCK_SLOPES.values())[0])

                    hi_asc = project_line_from(hi_price, hi_time, +slope_mag, tue_date + timedelta(days=1), "High_Asc")
                    lo_desc= project_line_from(lo_price, lo_time, -slope_mag, tue_date + timedelta(days=1), "Low_Desc")

                    st.markdown(f"**{ticker} Anchors:** High={hi_price:.2f} @ {hi_time.strftime('%Y-%m-%d %H:%M')}, "
                                f"Low={lo_price:.2f} @ {lo_time.strftime('%Y-%m-%d %H:%M')}")
                    out = hi_asc.merge(lo_desc, on="Time", how="outer").sort_values("Time")
                    st.dataframe(out, use_container_width=True, hide_index=True)
    else:
        st.info("Choose Mon/Tue in sidebar, select ticker, then click **Analyze Stock Anchors**.")

# ╔═ TAB 3: SIGNALS & EMA ══════════════════════════════════════════════════════╗
with tab3:
    st.subheader("Signals: Touch-Then-Confirmation + EMA (8/21 on 30-min)")

    if btn_signals:
        with st.spinner("Computing signals…"):
            day_df = fetch_intraday(sig_symbol, sig_day, sig_day)
            if day_df.empty:
                st.error("No intraday data for chosen symbol/day.")
            else:
                rth = between_time(day_df, RTH_START, RTH_END)
                # Prior day anchor for the analysis day
                prior_day = sig_day - timedelta(days=1)
                if use_manual_close and manual_close_val is not None and sig_symbol in ["^GSPC","SPY"]:
                    prev_close = float(manual_close_val)  # allow reuse if you're syncing SPX logic
                else:
                    prev_for_sig = fetch_intraday(sig_symbol, prior_day, prior_day)
                    prev_close = get_prev_day_3pm_close(prev_for_sig, prior_day) if not prev_for_sig.empty else None

                if rth.empty or prev_close is None:
                    st.error("Missing RTH or previous-day 3:00 PM CT anchor for signals.")
                else:
                    anchor_time = fmt_ct(datetime.combine(prior_day, time(15,0)))
                    fan_df = project_fan_from_close(prev_close, anchor_time, sig_day)

                    touches = detect_touches(rth, fan_df)
                    ema_df  = confirmation_from_ema(rth)

                    st.markdown("### Fan Lines (for the analysis day)")
                    st.dataframe(fan_df, use_container_width=True, hide_index=True)

                    st.markdown("### Anchor Touches (Top/Bottom)")
                    if not touches.empty:
                        st.dataframe(touches, use_container_width=True, hide_index=True)
                    else:
                        st.info("No anchor touches detected on 30-min bars.")

                    st.markdown("### EMA 8/21 (30-min)")
                    if not ema_df.empty:
                        st.dataframe(ema_df, use_container_width=True, hide_index=True)
                    else:
                        st.info("Not enough bars to compute EMA(8/21).")
    else:
        st.info("Pick symbol/day in sidebar, then click **Analyze Signals & EMA**.")

# ╔═ TAB 4: CONTRACT TOOL ══════════════════════════════════════════════════════╗
with tab4:
    st.subheader("Contract Tool (Overnight → RTH Projection)")

    if btn_ct:
        p1_dt = datetime.combine(p1_date, p1_time)
        p2_dt = datetime.combine(p2_date, p2_time)
        if p2_dt <= p1_dt:
            st.error("Point 2 must be after Point 1.")
        else:
            with st.spinner("Projecting contract path…"):
                blocks = (p2_dt - p1_dt).total_seconds()/1800.0
                slope = (p2_price - p1_price)/blocks if blocks>0 else 0.0
                rows=[]
                for slot in rth_slots_ct(ct_proj):
                    b = count_effective_blocks(fmt_ct(p1_dt), slot)
                    price = p1_price + slope*b
                    rows.append({"Time": slot.strftime("%H:%M"), "Contract_Price": round(price,2), "Blocks": round(b,1)})
                proj_df = pd.DataFrame(rows)
                st.markdown(f"**Derived slope:** {slope:+.4f} per 30-min block • Typical prices 0–30")
                st.dataframe(proj_df, use_container_width=True, hide_index=True)
    else:
        st.info("Fill the two points in the sidebar, then click **Analyze Contract Projection**.")

st.markdown("---")

# Connectivity test (quiet)
colA, colB = st.columns([1, 2])
with colA:
    if st.button("🔌 Test Data Connection"):
        td = fetch_intraday("^GSPC", today_ct - timedelta(days=3), today_ct)
        if td.empty:
            td = fetch_intraday("SPY", today_ct - timedelta(days=3), today_ct)
        if not td.empty:
            st.success(f"OK — received {len(td)} bars.")
        else:
            st.error("Data fetch failed — try different dates later.")
with colB:
    st.caption("Tip: Turn on **manual 3:00 PM CT close** in the sidebar to remove any data-source variance.")