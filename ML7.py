# app.py
# ═══════════════════════════════════════════════════════════════════════════════
# 🔮 SPX PROPHET — Full App (SPX Anchors • Stock Anchors • Signals & EMA • Contract Tool)
# - SPX: previous day's exact 3:00 PM CT close as anchor (manual override supported)
# - Fan: Top=+slope per 30m, Bottom=−slope per 30m (default ±0.260), skip 4–5 PM CT + Fri 5 PM → Sun 5 PM
# - Stocks: Mon/Tue swing high/low → two parallel anchor lines by your per-ticker slopes
# - Signals: fan touch + same-bar EMA 8/21 confirmation (1m if recent; otherwise 5m/30m fallback)
# - Contract Tool: two points (0–30 price) → slope → RTH projection
# - Clean light UI, icons, and compact tables
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
RTH_START = "08:30"
RTH_END   = "14:30"

# Default slope per 30-minute block (Top +, Bottom −)
SLOPE_PER_BLOCK_DEFAULT = 0.260

# Per-ticker slope magnitudes (your latest set)
STOCK_SLOPES = {
    "TSLA": 0.0285, "NVDA": 0.0860, "AAPL": 0.0155, "MSFT": 0.0541,
    "AMZN": 0.0139, "GOOGL": 0.0122, "META": 0.0674, "NFLX": 0.0230,
}

# ───────────────────────────────────────────────────────────────────────────────
# PAGE & THEME (light, enterprise vibe with glassy cards)
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
  --brand: #2563eb;      /* blue-600 */
  --brand-2: #10b981;    /* emerald-500 */
  --surface: #ffffff;
  --muted: #f8fafc;      /* slate-50 */
  --text: #0f172a;       /* slate-900 */
  --subtext: #475569;    /* slate-600 */
  --border: #e2e8f0;     /* slate-200 */
  --warn: #f59e0b;       /* amber-500 */
  --danger: #ef4444;     /* red-500 */
}

html, body, [class*="css"]  {
  background: var(--muted);
  color: var(--text);
}

.block-container { padding-top: 1.1rem; }

h1, h2, h3 { color: var(--text); }

.card, .metric-card {
  background: rgba(255,255,255,0.9);
  border: 1px solid var(--border);
  border-radius: 16px;
  padding: 16px;
  box-shadow: 0 12px 32px rgba(2,6,23,0.07);
  backdrop-filter: blur(8px);
}

.metric-title { font-size: 0.9rem; color: var(--subtext); margin: 0; }
.metric-value { font-size: 1.8rem; font-weight: 700; margin-top: 6px; }
.kicker { font-size: 0.8rem; color: var(--subtext); }

.badge-open {
  color: #065f46; background: #d1fae5; border: 1px solid #99f6e4;
  padding: 2px 8px; border-radius: 999px; font-size: 0.8rem; font-weight: 600;
}
.badge-closed {
  color: #7c2d12; background: #ffedd5; border: 1px solid #fed7aa;
  padding: 2px 8px; border-radius: 999px; font-size: 0.8rem; font-weight: 600;
}

hr { border-top: 1px solid var(--border); }

.dataframe { background: var(--surface); border-radius: 12px; overflow: hidden; }
.small-note { color: var(--subtext); font-size: 0.85rem; }

.override-tag {
  font-size: 0.75rem; color: #334155; background: #e2e8f0; border: 1px solid #cbd5e1;
  padding: 2px 8px; border-radius: 999px; display:inline-block; margin-top:6px;
}
</style>
""",
    unsafe_allow_html=True,
)

# ───────────────────────────────────────────────────────────────────────────────
# HELPERS
# ───────────────────────────────────────────────────────────────────────────────
def fmt_ct(dt: datetime) -> datetime:
    """Ensure timezone-aware CT."""
    if dt.tzinfo is None:
        return CT_TZ.localize(dt)
    return dt.astimezone(CT_TZ)

def between_time(df: pd.DataFrame, start_str: str, end_str: str) -> pd.DataFrame:
    return df.between_time(start_str, end_str) if not df.empty else df

def rth_slots_ct(target_date: date) -> List[datetime]:
    start_dt = fmt_ct(datetime.combine(target_date, time(8, 30)))
    end_dt   = fmt_ct(datetime.combine(target_date, time(14, 30)))
    slots = []
    cur = start_dt
    while cur <= end_dt:
        slots.append(cur)
        cur += timedelta(minutes=30)
    return slots

def is_maintenance(dt: datetime) -> bool:
    """True for 4–5 PM CT maintenance hour."""
    return dt.hour == 16

def in_weekend_gap(dt: datetime) -> bool:
    """
    True for Fri >= 17:00 → Sun < 17:00 CT (no overnight session counted).
    """
    wd = dt.weekday()  # Mon=0 ... Sun=6
    if wd == 5:
        return True           # Saturday
    if wd == 6 and dt.hour < 17:
        return True           # Sunday before 5pm CT
    if wd == 4 and dt.hour >= 17:
        return True           # Friday from 5pm CT onward
    return False

def count_effective_blocks(anchor_time: datetime, target_time: datetime) -> float:
    """
    Count 30-min blocks from anchor_time → target_time,
    skipping maintenance (4–5 PM) and weekend gap.
    Count a block if the *end* time of that block is not in forbidden windows.
    """
    if target_time <= anchor_time:
        return 0.0
    t = anchor_time
    blocks = 0
    while t < target_time:
        t_next = t + timedelta(minutes=30)
        if not is_maintenance(t_next) and not in_weekend_gap(t_next):
            blocks += 1
        t = t_next
    return float(blocks)

def ensure_ohlc_cols(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] if isinstance(c, tuple) else str(c) for c in df.columns]
    required = ["Open", "High", "Low", "Close", "Volume"]
    if any(c not in df.columns for c in required):
        return pd.DataFrame()
    return df

def normalize_to_ct(df: pd.DataFrame, start_d: date, end_d: date) -> pd.DataFrame:
    if df.empty:
        return df
    df = ensure_ohlc_cols(df)
    if df.empty:
        return df
    if df.index.tz is None:
        df.index = df.index.tz_localize("US/Eastern")
    df.index = df.index.tz_convert(CT_TZ)
    sdt = fmt_ct(datetime.combine(start_d, time(0, 0)))
    edt = fmt_ct(datetime.combine(end_d, time(23, 59)))
    return df.loc[sdt:edt]

@st.cache_data(ttl=120)
def fetch_intraday(symbol: str, start_d: date, end_d: date) -> pd.DataFrame:
    """
    Robust intraday fetch, 30m, CT index, auto_adjust=False for accurate closes.
    Falls back to period-based fetch if start/end returns empty.
    """
    try:
        t = yf.Ticker(symbol)
        df = t.history(
            start=(start_d - timedelta(days=5)).strftime("%Y-%m-%d"),
            end=(end_d + timedelta(days=2)).strftime("%Y-%m-%d"),
            interval="30m", prepost=True, auto_adjust=False, back_adjust=False,
        )
        df = normalize_to_ct(df, start_d, end_d)
        if df.empty:
            days = max(7, (end_d - start_d).days + 7)
            df2 = t.history(
                period=f"{days}d", interval="30m",
                prepost=True, auto_adjust=False, back_adjust=False,
            )
            df = normalize_to_ct(df2, start_d, end_d)
        return df
    except Exception:
        return pd.DataFrame()

@st.cache_data(ttl=120)
def fetch_intraday_interval(symbol: str, start_d: date, end_d: date, interval: str) -> pd.DataFrame:
    """
    Interval-aware fetch. For 1m data, yfinance requires recent periods (<=7d).
    We automatically choose a period window for 1m/5m; otherwise use start/end.
    """
    try:
        t = yf.Ticker(symbol)
        if interval in ["1m", "2m", "5m", "15m"]:
            # choose a period window that safely covers [start_d, end_d]
            days = max(1, min(7, (end_d - start_d).days + 2))
            df = t.history(period=f"{days}d", interval=interval,
                           prepost=True, auto_adjust=False, back_adjust=False)
            # trim to the exact date range after tz normalization
            # pick a broad range for normalization
            df = normalize_to_ct(df, start_d - timedelta(days=1), end_d + timedelta(days=1))
            sdt = fmt_ct(datetime.combine(start_d, time(0, 0)))
            edt = fmt_ct(datetime.combine(end_d, time(23, 59)))
            df = df.loc[sdt:edt]
        else:
            df = t.history(
                start=(start_d - timedelta(days=5)).strftime("%Y-%m-%d"),
                end=(end_d + timedelta(days=2)).strftime("%Y-%m-%d"),
                interval=interval, prepost=True, auto_adjust=False, back_adjust=False,
            )
            df = normalize_to_ct(df, start_d, end_d)
        return df
    except Exception:
        return pd.DataFrame()

def get_prev_day_3pm_close(spx_prev: pd.DataFrame, prev_day: date) -> Optional[float]:
    """
    Get the **3:00 PM CT exact bar close** for prev_day.
    If exact 15:00 not present, use the last bar <= 15:00 within prev_day.
    """
    if spx_prev.empty:
        return None
    day_start = fmt_ct(datetime.combine(prev_day, time(0, 0)))
    day_end   = fmt_ct(datetime.combine(prev_day, time(23, 59)))
    d = spx_prev.loc[day_start:day_end].copy()
    if d.empty:
        return None
    target = fmt_ct(datetime.combine(prev_day, time(15, 0)))
    if target in d.index:
        return float(d.loc[target, "Close"])
    prior = d.loc[:target]
    if not prior.empty:
        return float(prior.iloc[-1]["Close"])
    return None

# ───── Slope state
def current_slope() -> float:
    return float(st.session_state.get("slope_per_block", SLOPE_PER_BLOCK_DEFAULT))

# ───── SPX Fan Projection
def project_fan_from_close(close_price: float, anchor_time: datetime, target_day: date) -> pd.DataFrame:
    slope = current_slope()
    rows = []
    for slot in rth_slots_ct(target_day):
        blocks = count_effective_blocks(anchor_time, slot)
        top = close_price + slope * blocks
        bot = close_price - slope * blocks
        rows.append({"Time": slot.strftime("%H:%M"),
                     "Top": round(top, 2),
                     "Bottom": round(bot, 2),
                     "Fan_Width": round(top - bot, 2)})
    return pd.DataFrame(rows)

# ───── SPX Strategy
def build_spx_strategy(rth_prices: pd.DataFrame, fan_df: pd.DataFrame, anchor_close: float) -> pd.DataFrame:
    """
    - Bias: "UP" if price ≥ anchor_close; else "DOWN".
    - Within fan:
        - Bias UP → BUY bottom → TP1/TP2 at top
        - Bias DOWN → SELL top → TP1/TP2 at bottom
    - Above fan: SELL at top, TP2 = top - width
    - Below fan: SELL at bottom, TP2 = bottom - width
    """
    if rth_prices.empty or fan_df.empty:
        return pd.DataFrame()

    price_lu = {dt.strftime("%H:%M"): float(rth_prices.loc[dt, "Close"]) for dt in rth_prices.index}
    rows = []
    for _, row in fan_df.iterrows():
        t = row["Time"]
        if t not in price_lu:
            continue
        p = price_lu[t]
        top, bot, width = row["Top"], row["Bottom"], row["Fan_Width"]
        bias = "UP" if p >= anchor_close else "DOWN"

        if bot <= p <= top:
            if bias == "UP":
                direction = "BUY"; entry = bot; tp1 = top; tp2 = top; note = "Within fan; bias UP"
            else:
                direction = "SELL"; entry = top; tp1 = bot; tp2 = bot; note = "Within fan; bias DOWN"
        elif p > top:
            direction = "SELL"; entry = top; tp1 = np.nan; tp2 = top - width; note = "Above fan"
        else:  # p < bottom
            direction = "SELL"; entry = bot; tp1 = np.nan; tp2 = bot - width; note = "Below fan"

        rows.append({
            "Time": t, "Price": round(p, 2), "Bias": bias, "EntrySide": direction,
            "Entry": round(entry, 2), "TP1": (round(tp1, 2) if not pd.isna(tp1) else np.nan),
            "TP2": (round(tp2, 2) if not pd.isna(tp2) else np.nan),
            "Top": round(top, 2), "Bottom": round(bot, 2)
        })
    return pd.DataFrame(rows)

# ───── Stocks: swings & two-line projection
def detect_absolute_swings(df: pd.DataFrame) -> Tuple[Optional[Tuple[float, datetime]], Optional[Tuple[float, datetime]]]:
    """Return (highest_high, its_time), (lowest_low, its_time)"""
    if df.empty:
        return None, None
    hi = df['High'].idxmax() if 'High' in df else None
    lo = df['Low'].idxmin() if 'Low' in df else None
    high = (float(df.loc[hi, 'High']), hi) if hi is not None else None
    low  = (float(df.loc[lo, 'Low']),  lo) if lo is not None else None
    return high, low

def project_two_stock_lines(high_price: float, high_time: datetime,
                            low_price: float, low_time: datetime,
                            slope_mag: float, target_day: date) -> pd.DataFrame:
    """Ascending from swing high (+slope_mag) and descending from swing low (−slope_mag)."""
    rows = []
    for slot in rth_slots_ct(target_day):
        b_high = count_effective_blocks(high_time, slot)
        b_low  = count_effective_blocks(low_time,  slot)
        high_asc = high_price + slope_mag * b_high
        low_desc = low_price  - slope_mag * b_low
        rows.append({"Time": slot.strftime("%H:%M"),
                     "High_Asc": round(high_asc, 2),
                     "Low_Desc": round(low_desc, 2)})
    return pd.DataFrame(rows)

# ───── EMA utils
def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()

def compute_ema_cross_df(df: pd.DataFrame) -> pd.DataFrame:
    """Return df with EMA8/EMA21 and crossover label for same-bar check."""
    if df.empty or 'Close' not in df:
        return pd.DataFrame()
    out = df.copy()
    out['EMA8'] = ema(out['Close'], 8)
    out['EMA21'] = ema(out['Close'], 21)
    # same-bar "state"
    out['Crossover'] = np.where(out['EMA8'] > out['EMA21'], 'Bullish', np.where(out['EMA8'] < out['EMA21'], 'Bearish', 'None'))
    return out

# ───────────────────────────────────────────────────────────────────────────────
# SIDEBAR — Global controls (SPX panel)
# ───────────────────────────────────────────────────────────────────────────────
st.sidebar.title("🔧 Controls")

today_ct = datetime.now(CT_TZ).date()
prev_day = st.sidebar.date_input("Previous Trading Day", value=today_ct - timedelta(days=1))
proj_day = st.sidebar.date_input("Projection Day", value=prev_day + timedelta(days=1))
st.sidebar.caption("Anchor at **3:00 PM CT** on the previous trading day.")

st.sidebar.markdown("---")
st.sidebar.subheader("✍️ Manual Close (optional)")
use_manual_close = st.sidebar.checkbox("Enter 3:00 PM CT Close Manually", value=False)
manual_close_val = st.sidebar.number_input(
    "Manual 3:00 PM Close",
    value=6400.00,
    step=0.01,
    format="%.2f",
    disabled=not use_manual_close,
    help="If enabled, this value overrides the fetched close for the SPX anchor."
)

st.sidebar.markdown("---")
with st.sidebar.expander("⚙️ Advanced (optional)", expanded=False):
    st.caption("Adjust per-30m slope (applies to SPX and stock projections).")
    enable_slope = st.checkbox("Enable slope override", value=("slope_per_block" in st.session_state))
    slope_val = st.number_input(
        "Slope per 30m (Top +, Bottom −)",
        value=float(st.session_state.get("slope_per_block", SLOPE_PER_BLOCK_DEFAULT)),
        step=0.001, format="%.3f",
        help="Example: 0.260 means Top = anchor + 0.260 per 30 minutes."
    )
    col_adv_a, col_adv_b = st.columns(2)
    with col_adv_a:
        if st.button("Apply slope", use_container_width=True, key="apply_slope"):
            if enable_slope:
                st.session_state["slope_per_block"] = float(slope_val)
                st.success(f"Slope set to ±{slope_val:.3f}")
            else:
                if "slope_per_block" in st.session_state:
                    del st.session_state["slope_per_block"]
                st.info("Slope override disabled (using default).")
    with col_adv_b:
        if st.button("Reset slope", use_container_width=True, key="reset_slope"):
            if "slope_per_block" in st.session_state:
                del st.session_state["slope_per_block"]
            st.success(f"Reset slope to default ±{SLOPE_PER_BLOCK_DEFAULT:.3f}")

st.sidebar.markdown("---")
go_spx = st.sidebar.button("🔮 Generate SPX Fan & Strategy", type="primary", use_container_width=True)

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
</div>
""",
        unsafe_allow_html=True,
    )
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
  <div class="metric-value">📊 <span class="{badge}">{text}</span></div>
  <div class="kicker">RTH: 08:30–14:30 CT • Mon–Fri</div>
</div>
""",
        unsafe_allow_html=True,
    )
with c3:
    slope_disp = current_slope()
    st.markdown(
        f"""
<div class="metric-card">
  <p class="metric-title">Slope / 30-min Block</p>
  <div class="metric-value">📐 ±{slope_disp:.3f}</div>
  <div class="kicker">Top = +slope • Bottom = −slope</div>
  {"<div class='override-tag'>Override active</div>" if "slope_per_block" in st.session_state else ""}
</div>
""",
        unsafe_allow_html=True,
    )

st.markdown("---")

# ───────────────────────────────────────────────────────────────────────────────
# TABS
# ───────────────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["SPX Anchors", "Stock Anchors", "Signals & EMA", "Contract Tool"])

# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║ TAB 1: SPX ANCHORS                                                          ║
# ╚═════════════════════════════════════════════════════════════════════════════╝
with tab1:
    st.subheader("SPX Close-Anchor Fan (3:00 PM CT)")

    if go_spx:
        with st.spinner("Building SPX fan & strategy…"):
            # Fetch prev/proj for ^GSPC (fallback to SPY if ^GSPC empty)
            spx_prev = fetch_intraday("^GSPC", prev_day, prev_day)
            if spx_prev.empty:
                spx_prev = fetch_intraday("SPY", prev_day, prev_day)

            spx_proj = fetch_intraday("^GSPC", proj_day, proj_day)
            if spx_proj.empty:
                spx_proj = fetch_intraday("SPY", proj_day, proj_day)

            if spx_prev.empty or spx_proj.empty:
                st.error("❌ Market data connection failed for the selected dates.")
            else:
                # Anchor close
                if use_manual_close:
                    anchor_close = float(manual_close_val)
                    anchor_time  = fmt_ct(datetime.combine(prev_day, time(15, 0)))
                    st.success(f"Using manual 3:00 PM CT close: **{anchor_close:.2f}**")
                else:
                    prev_3pm_close = get_prev_day_3pm_close(spx_prev, prev_day)
                    if prev_3pm_close is None:
                        st.error("Could not find a 3:00 PM CT close for the previous day.")
                        st.stop()
                    anchor_close = float(prev_3pm_close)
                    anchor_time  = fmt_ct(datetime.combine(prev_day, time(15, 0)))
                    st.success(f"Anchor (Prev Day 3:00 PM CT) Close: **{anchor_close:.2f}**")

                # Fan
                fan_df = project_fan_from_close(anchor_close, anchor_time, proj_day)

                # Strategy
                spx_proj_rth = between_time(spx_proj, RTH_START, RTH_END)
                if spx_proj_rth.empty:
                    st.error("No RTH data available for the projection day.")
                else:
                    strat_df = build_spx_strategy(spx_proj_rth, fan_df, anchor_close)

                    st.markdown("### 🎯 Fan Lines (Top / Bottom @ 30-min)")
                    st.dataframe(fan_df, use_container_width=True, hide_index=True)

                    st.markdown("### 📋 Strategy Table")
                    st.caption("Bias from close: **UP** if price ≥ 3:00 PM close; **DOWN** otherwise.")
                    st.dataframe(
                        strat_df[["Time","Price","Bias","EntrySide","Entry","TP1","TP2","Top","Bottom"]],
                        use_container_width=True, hide_index=True
                    )
    else:
        st.info("Use the **sidebar** to pick dates (and optional manual close), then click **Generate SPX Fan & Strategy**.")

# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║ TAB 2: STOCK ANCHORS                                                        ║
# ╚═════════════════════════════════════════════════════════════════════════════╝
with tab2:
    st.subheader("Stock Anchor Lines (Mon/Tue swings → two lines)")
    st.caption("We project an ascending line from the highest swing high and a descending line from the lowest swing low (Mon+Tue combined), using your per-ticker slope magnitude.")

    # Controls
    core = list(STOCK_SLOPES.keys())
    cc1, cc2, cc3 = st.columns([1.4,1,1])
    with cc1:
        ticker = st.selectbox("Ticker", core + ["Custom…"], index=0, key="stk_ticker")
        custom_ticker = ""
        if ticker == "Custom…":
            custom_ticker = st.text_input("Custom Symbol", value="", placeholder="e.g., AMD")
    with cc2:
        monday_date = st.date_input("Monday Date", value=today_ct - timedelta(days=max(1, (today_ct.weekday()+6)%7 + 1)))
    with cc3:
        tuesday_date = st.date_input("Tuesday Date", value=monday_date + timedelta(days=1))

    # Slope picker (uses your presets; can be changed per run)
    slope_mag_default = STOCK_SLOPES.get(ticker, 0.0150) if ticker != "Custom…" else 0.0150
    slope_mag = st.number_input("Slope Magnitude (per 30m)", value=float(slope_mag_default), step=0.0001, format="%.4f")

    proj_day_stock = st.date_input("Projection Day", value=tuesday_date + timedelta(days=1))
    run_stock = st.button("📈 Analyze Stock Anchors", type="primary")

    if run_stock:
        with st.spinner("Fetching and projecting…"):
            sym = custom_ticker.upper() if ticker == "Custom…" and custom_ticker else (ticker if ticker != "Custom…" else None)
            if not sym:
                st.error("Please enter a custom symbol.")
                st.stop()

            mon = fetch_intraday(sym, monday_date, monday_date)
            tue = fetch_intraday(sym, tuesday_date, tuesday_date)
            if mon.empty and tue.empty:
                st.error("No data for selected dates.")
                st.stop()

            combined = mon if tue.empty else (tue if mon.empty else pd.concat([mon, tue]).sort_index())

            hi, lo = detect_absolute_swings(combined)
            if not hi or not lo:
                st.error("Could not detect swings.")
                st.stop()

            (high_price, high_time) = hi
            (low_price,  low_time)  = lo
            high_time = fmt_ct(high_time); low_time = fmt_ct(low_time)

            proj_df = project_two_stock_lines(high_price, high_time, low_price, low_time, slope_mag, proj_day_stock)

            cA, cB = st.columns(2)
            with cA:
                st.markdown("**Swing High (Mon/Tue):**")
                st.write(f"📈 {sym} — High: **{high_price:.2f}** at {high_time.strftime('%Y-%m-%d %H:%M CT')}")
            with cB:
                st.markdown("**Swing Low (Mon/Tue):**")
                st.write(f"📉 {sym} — Low: **{low_price:.2f}** at {low_time.strftime('%Y-%m-%d %H:%M CT')}")

            st.markdown("### 🔧 Projection (RTH)")
            st.dataframe(proj_df, use_container_width=True, hide_index=True)

# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║ TAB 3: SIGNALS & EMA                                                        ║
# ╚═════════════════════════════════════════════════════════════════════════════╝
with tab3:
    st.subheader("Signals: Fan Touch + Same-Bar EMA 8/21 Confirmation")

    colS1, colS2, colS3 = st.columns([1.2,1,1])
    with colS1:
        sig_symbol = st.selectbox("Symbol", ["^GSPC", "SPY", "ES=F"], index=0)
    with colS2:
        sig_day = st.date_input("Analysis Day", value=today_ct)
    with colS3:
        interval_pref = st.selectbox("Interval preference", ["1m", "5m", "30m"], index=0, help="1m requires recent dates (≤7 days)")

    run_sig = st.button("🔎 Analyze Signals", type="primary")

    if run_sig:
        with st.spinner("Computing signals…"):
            # Build fan for the day (needs previous day close at 3:00 PM CT)
            prev_day_sig = sig_day - timedelta(days=1)
            spx_prev = fetch_intraday("^GSPC", prev_day_sig, prev_day_sig)
            if spx_prev.empty and sig_symbol in ["^GSPC", "SPY"]:
                spx_prev = fetch_intraday("SPY", prev_day_sig, prev_day_sig)

            if spx_prev.empty:
                st.error("Could not build fan (prev day data missing).")
                st.stop()

            prev_3pm_close = get_prev_day_3pm_close(spx_prev, prev_day_sig)
            if prev_3pm_close is None:
                st.error("Prev day 3:00 PM close not found.")
                st.stop()

            anchor_close = float(prev_3pm_close)
            anchor_time  = fmt_ct(datetime.combine(prev_day_sig, time(15, 0)))
            fan_df = project_fan_from_close(anchor_close, anchor_time, sig_day)

            # Intraday data for signals: try requested interval, fallback gracefully
            intraday = fetch_intraday_interval(sig_symbol, sig_day, sig_day, interval_pref)
            if intraday.empty and interval_pref != "5m":
                intraday = fetch_intraday_interval(sig_symbol, sig_day, sig_day, "5m")
            if intraday.empty and interval_pref != "30m":
                intraday = fetch_intraday_interval(sig_symbol, sig_day, sig_day, "30m")

            if intraday.empty:
                st.error("No intraday data available for the analysis day.")
                st.stop()

            # Map fan prices to nearest matching times by "HH:MM"
            fan_lu_top = {r['Time']: r['Top'] for _, r in fan_df.iterrows()}
            fan_lu_bot = {r['Time']: r['Bottom'] for _, r in fan_df.iterrows()}

            # Prepare EMA cross dataframe on same intraday interval
            ema_df = compute_ema_cross_df(intraday)

            signals = []
            for ts, bar in ema_df.iterrows():
                tstr = ts.strftime("%H:%M")
                if tstr not in fan_lu_top or tstr not in fan_lu_bot:
                    continue
                top = fan_lu_top[tstr]; bot = fan_lu_bot[tstr]
                low, high, close, open_ = bar.get('Low', np.nan), bar.get('High', np.nan), bar.get('Close', np.nan), bar.get('Open', np.nan)

                touched_bottom = (not pd.isna(low) and not pd.isna(high) and (low <= bot <= high))
                touched_top    = (not pd.isna(low) and not pd.isna(high) and (low <= top <= high))

                confirmation = bar['Crossover']  # 'Bullish' / 'Bearish' / 'None'
                action = ""
                rationale = ""

                if touched_bottom:
                    if confirmation == 'Bullish':
                        action = "BUY → target Top"
                        rationale = "Touched Bottom & same-bar EMA8>EMA21"
                    elif confirmation == 'Bearish':
                        action = "SELL → potential breakdown"
                        rationale = "Touched Bottom & same-bar EMA8<EMA21"
                elif touched_top:
                    if confirmation == 'Bearish':
                        action = "SELL → target Bottom"
                        rationale = "Touched Top & same-bar EMA8<EMA21"
                    elif confirmation == 'Bullish':
                        action = "BUY → potential breakout"
                        rationale = "Touched Top & same-bar EMA8>EMA21"

                if touched_bottom or touched_top:
                    signals.append({
                        "Time": tstr,
                        "Open": round(open_,2), "High": round(high,2), "Low": round(low,2), "Close": round(close,2),
                        "Top": round(top,2), "Bottom": round(bot,2),
                        "EMA8": round(bar['EMA8'],2), "EMA21": round(bar['EMA21'],2),
                        "Cross": confirmation,
                        "Signal": action if action else "Touch w/o confirm",
                        "Note": rationale if rationale else "Confirmation not aligned"
                    })

            st.markdown("### 📡 Fan Touch + EMA Confirmation")
            if signals:
                st.dataframe(pd.DataFrame(signals), use_container_width=True, hide_index=True)
            else:
                st.info("No touch+confirmation signals found for that day/interval.")

# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║ TAB 4: CONTRACT TOOL                                                        ║
# ╚═════════════════════════════════════════════════════════════════════════════╝
with tab4:
    st.subheader("Contract Tool (Overnight → RTH Projection)")

    point_col1, point_col2 = st.columns(2)
    with point_col1:
        p1_date = st.date_input("Point 1 Date", value=today_ct - timedelta(days=1))
        p1_time = st.time_input("Point 1 Time (CT)", value=time(20, 0))
        p1_price = st.number_input("Point 1 Contract Price", value=10.00, min_value=0.01, step=0.01, format="%.2f")
    with point_col2:
        p2_date = st.date_input("Point 2 Date", value=today_ct)
        p2_time = st.time_input("Point 2 Time (CT)", value=time(8, 0))
        p2_price = st.number_input("Point 2 Contract Price", value=12.00, min_value=0.01, step=0.01, format="%.2f")

    proj_day_ct = st.date_input("RTH Projection Day", value=p2_date)
    run_ct = st.button("🧮 Analyze Contract Projections", type="primary")

    if run_ct:
        p1_dt = fmt_ct(datetime.combine(p1_date, p1_time))
        p2_dt = fmt_ct(datetime.combine(p2_date, p2_time))
        if p2_dt <= p1_dt:
            st.error("Point 2 must be after Point 1")
            st.stop()

        # slope per 30m blocks (skip maintenance & weekend)
        blocks = count_effective_blocks(p1_dt, p2_dt)
        slope_ct = (p2_price - p1_price) / blocks if blocks > 0 else 0.0

        # project across RTH
        rows = []
        for slot in rth_slots_ct(proj_day_ct):
            b = count_effective_blocks(p1_dt, slot)
            price = p1_price + slope_ct * b
            rows.append({"Time": slot.strftime("%H:%M"),
                         "Contract_Price": round(price, 2),
                         "Blocks": round(b, 1)})
        proj_df = pd.DataFrame(rows)

        mc1, mc2, mc3, mc4 = st.columns(4)
        with mc1: st.markdown(f"<div class='metric-card'><p class='metric-title'>Time Span</p><div class='metric-value'>⏱ {(p2_dt-p1_dt).total_seconds()/3600:.1f}h</div></div>", unsafe_allow_html=True)
        with mc2: st.markdown(f"<div class='metric-card'><p class='metric-title'>Δ Price</p><div class='metric-value'>↕ {p2_price - p1_price:+.2f}</div></div>", unsafe_allow_html=True)
        with mc3: st.markdown(f"<div class='metric-card'><p class='metric-title'>Blocks Counted</p><div class='metric-value'>🧩 {blocks:.1f}</div></div>", unsafe_allow_html=True)
        with mc4: st.markdown(f"<div class='metric-card'><p class='metric-title'>Slope / 30m</p><div class='metric-value'>📐 {slope_ct:+.3f}</div></div>", unsafe_allow_html=True)

        st.markdown("### 📊 RTH Projection")
        st.dataframe(proj_df, use_container_width=True, hide_index=True)

# ───────────────────────────────────────────────────────────────────────────────
# FOOTER UTILITIES
# ───────────────────────────────────────────────────────────────────────────────
st.markdown("---")
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
    st.caption("Times normalized to **CT**. SPX uses **3:00 PM CT** close. Slope applies to all projections. Manual close available on SPX tab via sidebar.")
