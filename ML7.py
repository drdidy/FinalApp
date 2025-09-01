# ═══════════════════════════════════════════════════════════════════════════════
# 🔮 SPX PROPHET — FULL APP (Unified, Enterprise UI)
# Close-anchor fan; default slope 0.277 / 30m; maintenance hour excluded
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
UTC_TZ = pytz.UTC

RTH_START = "08:30"  # CT
RTH_END   = "14:30"  # CT

# Maintenance hour (excluded from block count)
MAINT_START_CT = time(15, 30)
MAINT_END_CT   = time(16, 30)

# Default slope magnitude (user can change in sidebar)
DEFAULT_SLOPE = 0.277  # per 30-min block

# ───────────────────────────────────────────────────────────────────────────────
# PAGE & THEME (Light + Glassmorphism)
# ───────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="🔮 SPX Prophet",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
:root{
  --bg:#f7f7fb; --fg:#0f172a; --card:#ffffffcc; --muted:#64748b; --accent:#7c3aed;
  --green:#10b981; --red:#ef4444; --amber:#f59e0b; --blue:#3b82f6; --border:#e5e7eb;
}
html, body, [class*="css"]  { background: var(--bg); color: var(--fg); }
section.main .block-container { padding-top: 1.2rem; }
.glass {
  background: var(--card);
  backdrop-filter: blur(10px);
  -webkit-backdrop-filter: blur(10px);
  border: 1px solid var(--border);
  box-shadow: 0 12px 32px rgba(15, 23, 42, .08);
  border-radius: 18px; padding: 18px;
}
.kpi {
  background: var(--card); border:1px solid var(--border); border-radius:18px;
  padding:16px 18px; box-shadow: 0 8px 24px rgba(15, 23, 42, .05);
}
.kpi h3 { margin:0; font-weight:700; font-size:14px; color: var(--muted);}
.kpi .big { font-size:28px; font-weight:800; margin-top:6px;}
.kpi .sub { color: var(--muted); font-size:12px; margin-top:6px;}
hr { border: none; height: 1px; background: var(--border); margin: 12px 0 8px;}
.stTabs [data-baseweb="tab-list"] { gap: 8px; }
.stTabs [data-baseweb="tab"] { background: var(--card); padding: 10px 14px; border-radius: 12px; }
.stDataFrame { border-radius: 12px; overflow: hidden; border: 1px solid var(--border); }
button[kind="primary"] { border-radius: 12px !important; }
</style>
""", unsafe_allow_html=True)

# ───────────────────────────────────────────────────────────────────────────────
# HELPERS
# ───────────────────────────────────────────────────────────────────────────────
def to_ct(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return CT_TZ.localize(dt)
    return dt.astimezone(CT_TZ)

def rth_slots_ct(target_date: date) -> List[datetime]:
    start_dt = to_ct(datetime.combine(target_date, time(8,30)))
    end_dt   = to_ct(datetime.combine(target_date, time(14,30)))
    slots = []
    cur = start_dt
    while cur <= end_dt:
        slots.append(cur)
        cur += timedelta(minutes=30)
    return slots

def format_ct_time(dt: datetime) -> str:
    return to_ct(dt).strftime("%H:%M")

def get_session_window(df: pd.DataFrame, start_hhmm: str, end_hhmm: str) -> pd.DataFrame:
    if df.empty: return df
    return df.between_time(start_hhmm, end_hhmm)

def _normalize_df(df: pd.DataFrame, start_date: date, end_date: date) -> pd.DataFrame:
    if df.empty: return df
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] if isinstance(c, tuple) else str(c) for c in df.columns]
    needed = {"Open","High","Low","Close","Volume"}
    if not needed.issubset(set(df.columns)): return pd.DataFrame()
    # yfinance intraday is tz-aware in exchange tz; make CT
    if df.index.tz is None:
        df.index = df.index.tz_localize("US/Eastern")
    df.index = df.index.tz_convert(CT_TZ)
    sdt = to_ct(datetime.combine(start_date, time(0,0)))
    edt = to_ct(datetime.combine(end_date,   time(23,59)))
    return df.loc[sdt:edt]

@st.cache_data(ttl=90, show_spinner=False)
def fetch_live_data(symbol: str, start_date: date, end_date: date) -> pd.DataFrame:
    """Robust 30m intraday fetch with period fallback; CT index."""
    try:
        t = yf.Ticker(symbol)
        # Try date range first
        df = t.history(
            start=(start_date - timedelta(days=5)).strftime("%Y-%m-%d"),
            end=(end_date + timedelta(days=2)).strftime("%Y-%m-%d"),
            interval="30m", prepost=True, auto_adjust=False, back_adjust=False
        )
        df = _normalize_df(df, start_date, end_date)
        # Fallback to period
        if df.empty:
            span = max(7, (end_date - start_date).days + 7)
            df2 = t.history(period=f"{span}d", interval="30m",
                            prepost=True, auto_adjust=False, back_adjust=False)
            df = _normalize_df(df2, start_date, end_date)
        return df
    except Exception:
        return pd.DataFrame()

def get_3pm_close(df: pd.DataFrame, target_date: date) -> Optional[Tuple[float, datetime]]:
    """
    Get the bar labeled exactly 15:00 CT for the target_date (30m data).
    If missing, fallback to the latest bar <= 15:00.
    """
    if df.empty: return None
    sdt = to_ct(datetime.combine(target_date, time(0,0)))
    edt = to_ct(datetime.combine(target_date, time(23,59)))
    day = df.loc[sdt:edt]
    if day.empty: return None
    # Try exact 15:00 index
    mask_exact = day.index.hour.eq(15) & day.index.minute.eq(0)
    if mask_exact.any():
        row = day.loc[mask_exact].iloc[0]
        ts = day.loc[mask_exact].index[0]
        return float(row["Close"]), ts
    # Fallback: nearest before 15:00
    before = day[day.index <= to_ct(datetime.combine(target_date, time(15,0)))]
    if before.empty: return None
    row = before.iloc[-1]
    ts = before.index[-1]
    return float(row["Close"]), ts

def day_has_maintenance_cut(anchor_dt: datetime, target_dt: datetime) -> bool:
    """Return True if the timespan crosses the 15:30–16:30 CT maintenance window on the anchor date."""
    a = to_ct(anchor_dt)
    t = to_ct(target_dt)
    if a.date() != t.date():
        # spans beyond the anchor date; if the anchor date's window is after anchor time, it's crossed
        if (a.time() <= MAINT_START_CT):
            return True
        # If anchor after 16:30, not crossing that day's window
        return False
    # Same calendar date
    win_start = to_ct(datetime.combine(a.date(), MAINT_START_CT))
    win_end   = to_ct(datetime.combine(a.date(), MAINT_END_CT))
    return (a <= win_start) and (t >= win_end)

def count_30m_blocks_ex_maintenance(anchor_dt: datetime, target_dt: datetime) -> float:
    """
    Count 30-min blocks from anchor_dt -> target_dt, subtracting 2 blocks if the span
    crosses the anchor day maintenance window (15:30–16:30 CT).
    """
    a = to_ct(anchor_dt)
    t = to_ct(target_dt)
    blocks = (t - a).total_seconds() / 1800.0
    # If it crossed the maintenance hour on the anchor date, subtract 2 blocks
    if day_has_maintenance_cut(a, t):
        blocks -= 2.0
    return blocks

def project_line(anchor_price: float, anchor_time: datetime, slope_per_block: float,
                 target_date: date, col_name: str) -> pd.DataFrame:
    rows = []
    for slot in rth_slots_ct(target_date):
        blocks = count_30m_blocks_ex_maintenance(anchor_time, slot)
        price  = anchor_price + slope_per_block * blocks
        rows.append({"Time": format_ct_time(slot), col_name: round(price, 2), "Blocks": round(blocks, 1)})
    return pd.DataFrame(rows)

def project_fan_from_close(close_price: float, close_time: datetime, target_date: date,
                           slope: float) -> pd.DataFrame:
    top = project_line(close_price, close_time, +slope, target_date, "Top")
    bot = project_line(close_price, close_time, -slope, target_date, "Bottom")
    df  = pd.merge(top[["Time","Top"]], bot[["Time","Bottom"]], on="Time", how="inner")
    df["Fan_Width"] = (df["Top"] - df["Bottom"]).round(2)
    df["Close_Anchor"] = round(close_price, 2)
    return df

def project_line_from_time(anchor_price: float, anchor_hhmm: str, anchor_date: date,
                           slope: float, target_date: date, col_name: str) -> pd.DataFrame:
    a_dt = to_ct(datetime.combine(anchor_date, datetime.strptime(anchor_hhmm, "%H:%M").time()))
    return project_line(anchor_price, a_dt, slope, target_date, col_name)

def df_to_lookup(df: pd.DataFrame, price_col: str) -> Dict[str, float]:
    if df is None or df.empty: return {}
    return {row["Time"]: row[price_col] for _, row in df[["Time", price_col]].iterrows()}

def build_strategy_table(rth_prices: pd.DataFrame, fan_df: pd.DataFrame,
                         high_up_df: pd.DataFrame, low_dn_df: pd.DataFrame,
                         close_anchor_price: float) -> pd.DataFrame:
    if rth_prices.empty or fan_df.empty: return pd.DataFrame()
    px_lu  = {format_ct_time(ix): rth_prices.loc[ix, "Close"] for ix in rth_prices.index}
    top_lu = df_to_lookup(fan_df, "Top")
    bot_lu = df_to_lookup(fan_df, "Bottom")
    hup_lu = df_to_lookup(high_up_df, "High_Asc") if high_up_df is not None else {}
    ldn_lu = df_to_lookup(low_dn_df, "Low_Desc")  if low_dn_df  is not None else {}

    rows=[]
    for t in fan_df["Time"]:
        if t not in px_lu: continue
        p   = px_lu[t]
        top = top_lu.get(t, np.nan)
        bot = bot_lu.get(t, np.nan)
        if np.isnan(top) or np.isnan(bot): continue
        width = top - bot
        # Bias from fan: above=UP, below=DOWN, inside=RANGE
        if p > top: bias = "UP"
        elif p < bot: bias = "DOWN"
        else: bias = "RANGE"

        direction=""; entry=np.nan; tp1=np.nan; tp2=np.nan; note=""
        if bias == "RANGE":
            # Trade inside fan to opposite edge
            if p - (bot + width/2) >= 0:
                # Closer to top → SELL down
                direction="SELL"; entry=top; tp1=bot; tp2=bot; note="Inside fan: sell top → exit bottom"
            else:
                direction="BUY"; entry=bot; tp1=top; tp2=top; note="Inside fan: buy bottom → exit top"
        elif bias == "UP":  # price above top → short back inside
            direction="SELL"; entry=top
            tp2 = top - width  # mean reversion across fan width
            tp1 = hup_lu.get(t, np.nan)  # High ascending (prev day)
            note = "Above fan: entry @Top; TP1=High↑; TP2=Top−Width"
        else:  # bias == "DOWN": price below bottom → continue down
            direction="SELL"; entry=bot
            tp2 = bot - width
            tp1 = ldn_lu.get(t, np.nan)  # Low descending (prev day)
            note = "Below fan: entry @Bottom; TP1=Low↓; TP2=Bottom−Width"

        rows.append({
            "Time": t,
            "Price": round(p,2),
            "Bias": bias,
            "EntrySide": direction,
            "Entry": round(entry,2),
            "TP1": round(tp1,2) if not np.isnan(tp1) else np.nan,
            "TP2": round(tp2,2) if not np.isnan(tp2) else np.nan,
            "Top": round(top,2),
            "Bottom": round(bot,2),
            "Width": round(width,2),
            "Note": note
        })
    return pd.DataFrame(rows)

# ───────────────────────────────────────────────────────────────────────────────
# SIDEBAR — Controls
# ───────────────────────────────────────────────────────────────────────────────
st.sidebar.markdown("### ⚙️ Controls")
today_ct = datetime.now(CT_TZ).date()

# Remember last picks
prev_day_sb = st.sidebar.date_input("Previous Trading Day", value=st.session_state.get("prev_day_sb", today_ct - timedelta(days=1)))
proj_day_sb = st.sidebar.date_input("Projection Day", value=st.session_state.get("proj_day_sb", today_ct))
slope_sb    = st.sidebar.number_input("Slope per 30m (±)", value=float(st.session_state.get("slope_sb", DEFAULT_SLOPE)), step=0.001, format="%.3f")

st.sidebar.markdown("---")
sb_run = st.sidebar.button("▶️ Run with these settings", use_container_width=True, key="sb_run")

# keep for other tabs too
st.session_state["prev_day_sb"] = prev_day_sb
st.session_state["proj_day_sb"] = proj_day_sb
st.session_state["slope_sb"]    = slope_sb

# ───────────────────────────────────────────────────────────────────────────────
# HEADER KPIs
# ───────────────────────────────────────────────────────────────────────────────
c1,c2,c3,c4 = st.columns(4)
with c1:
    now = datetime.now(CT_TZ)
    st.markdown(f'<div class="kpi"><h3>Current Time (CT)</h3><div class="big">{now:%H:%M:%S}</div><div class="sub">{now:%A, %b %d}</div></div>', unsafe_allow_html=True)
with c2:
    is_wk = now.weekday() < 5
    mo = now.replace(hour=8,minute=30,second=0,microsecond=0)
    mc = now.replace(hour=14,minute=30,second=0,microsecond=0)
    is_open = is_wk and (mo <= now <= mc)
    label = "MARKET OPEN" if is_open else ("MARKET CLOSED" if is_wk else "WEEKEND")
    color = "#10b981" if is_open else ("#f59e0b" if is_wk else "#ef4444")
    st.markdown(f'<div class="kpi"><h3>Market</h3><div class="big" style="color:{color}">{label}</div><div class="sub">RTH 08:30–14:30 CT</div></div>', unsafe_allow_html=True)
with c3:
    st.markdown(f'<div class="kpi"><h3>Fan Slope</h3><div class="big">{slope_sb:+.3f} / 30m</div><div class="sub">Maintenance hour excluded</div></div>', unsafe_allow_html=True)
with c4:
    st.markdown(f'<div class="kpi"><h3>Prev / Proj</h3><div class="big">{prev_day_sb} → {proj_day_sb}</div><div class="sub">SPX fan day pair</div></div>', unsafe_allow_html=True)

st.markdown("<hr>", unsafe_allow_html=True)

# ───────────────────────────────────────────────────────────────────────────────
# TABS
# ───────────────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["SPX Anchors", "Stock Anchors", "Signals & EMA", "Contract Tool"])

# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║ TAB 1: SPX Anchors                                                          ║
# ╚═════════════════════════════════════════════════════════════════════════════╝
with tab1:
    st.markdown("#### 🧭 SPX Close-Anchor Fan (3:00 PM CT)")
    st.caption("Top = +slope from 3:00 PM close; Bottom = −slope; 30-min blocks minus the 1-hour maintenance window (15:30–16:30 CT).")

    with st.form("spx_form"):
        colx, coly, colz = st.columns([1,1,1])
        with colx:
            prev_day_local = st.date_input("Previous Trading Day", value=prev_day_sb, key="t1_prev")
        with coly:
            proj_day_local = st.date_input("Projection Day", value=proj_day_sb, key="t1_proj")
        with colz:
            slope_local = st.number_input("Slope / 30m", value=float(slope_sb), step=0.001, format="%.3f", key="t1_slope")
        submitted = st.form_submit_button("🚀 Generate Fan & Strategy")

    if submitted or sb_run:
        spx_prev = fetch_live_data("^GSPC", prev_day_local, prev_day_local)
        spx_proj = fetch_live_data("^GSPC", proj_day_local, proj_day_local)

        if spx_prev.empty or spx_proj.empty:
            st.error("Live market data not available for the selected dates (Yahoo returned empty).")
        else:
            rth_proj = get_session_window(spx_proj, RTH_START, RTH_END)

            close_info = get_3pm_close(spx_prev, prev_day_local)
            if close_info is None:
                st.error("Could not locate the 3:00 PM CT bar for the previous day.")
            else:
                close_px, close_ts = close_info

                # Previous day High/Low (for TP1 lines)
                day = spx_prev.loc[
                    to_ct(datetime.combine(prev_day_local, time(0,0))):
                    to_ct(datetime.combine(prev_day_local, time(23,59)))
                ]
                high_px = float(day["High"].max()); high_ts = day["High"].idxmax()
                low_px  = float(day["Low"].min());  low_ts  = day["Low"].idxmin()

                fan_df = project_fan_from_close(close_px, close_ts, proj_day_local, slope_local)
                high_up_df = project_line(high_px, high_ts, +slope_local, proj_day_local, "High_Asc")
                low_dn_df  = project_line(low_px,  low_ts,  -slope_local, proj_day_local, "Low_Desc")

                strat_df = build_strategy_table(rth_proj, fan_df, high_up_df, low_dn_df, close_px)

                cA, cB = st.columns(2)
                with cA:
                    st.markdown("##### 📐 Fan Lines")
                    st.dataframe(fan_df, use_container_width=True, hide_index=True)
                with cB:
                    st.markdown("##### 🎯 Strategy Table")
                    st.dataframe(strat_df, use_container_width=True, hide_index=True)

                st.markdown("""
<div class="glass">
<b>How to use</b>
<ul>
  <li><b>Bias</b>: Above <i>Top</i> → <b>UP</b>; Below <i>Bottom</i> → <b>DOWN</b>; Inside → <b>RANGE</b>.</li>
  <li><b>Inside fan</b>: Trade to opposite edge (buy Bottom → exit Top; sell Top → exit Bottom).</li>
  <li><b>Above fan</b>: Entry=Top; <b>TP1</b> = High ascending; <b>TP2</b> = Top − Width.</li>
  <li><b>Below fan</b>: Entry=Bottom; <b>TP1</b> = Low descending; <b>TP2</b> = Bottom − Width.</li>
</ul>
</div>
                """, unsafe_allow_html=True)
    else:
        st.info("Pick dates and click **Generate Fan & Strategy** (or use the sidebar **Run**).")

# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║ TAB 2: Stock Anchors (kept simple, no fan — just data preview)              ║
# ╚═════════════════════════════════════════════════════════════════════════════╝
with tab2:
    st.markdown("#### 🏢 Stock Anchors (Mon/Tue preview)")
    tickers = ["TSLA","NVDA","AAPL","MSFT","AMZN","GOOGL","META","NFLX"]
    tcols = st.columns(4); chosen=None
    for i,t in enumerate(tickers):
        if tcols[i%4].button(t): chosen=t
    custom = st.text_input("Custom symbol", placeholder="e.g., AMD")
    if custom: chosen = custom.upper()
    if chosen:
        c1,c2 = st.columns(2)
        with c1: mon = st.date_input("Monday", value=today_ct - timedelta(days=2))
        with c2: tue = st.date_input("Tuesday", value=today_ct - timedelta(days=1))
        if st.button(f"Fetch {chosen} Mon/Tue"):
            d1 = fetch_live_data(chosen, mon, mon)
            d2 = fetch_live_data(chosen, tue, tue)
            df = pd.concat([d1,d2]).sort_index() if not d1.empty or not d2.empty else pd.DataFrame()
            if df.empty:
                st.error("No intraday data returned.")
            else:
                st.dataframe(df, use_container_width=True)

# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║ TAB 3: Signals & EMA (RTH)                                                  ║
# ╚═════════════════════════════════════════════════════════════════════════════╝
def calculate_ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span).mean()

def calculate_average_true_range(data: pd.DataFrame, periods: int = 14) -> pd.Series:
    if data.empty or len(data) < periods: return pd.Series(index=data.index, dtype=float)
    hl = data["High"] - data["Low"]
    hc = (data["High"] - data["Close"].shift()).abs()
    lc = (data["Low"]  - data["Close"].shift()).abs()
    tr = np.maximum(hl, np.maximum(hc, lc))
    return tr.rolling(window=periods).mean()

def calculate_volatility(data: pd.DataFrame) -> float:
    if data.empty or len(data) < 2: return 0.0
    r = data["Close"].pct_change().dropna()
    if r.empty: return 0.0
    return float(r.std() * np.sqrt(390) * 100)

with tab3:
    st.markdown("#### 🔎 Signals & EMA (8/21)")
    with st.form("sig_form"):
        col1, col2 = st.columns(2)
        with col1:
            sym = st.selectbox("Symbol", ["^GSPC","ES=F","SPY"], index=0)
        with col2:
            sday = st.date_input("Analysis Day", value=today_ct)
        sgo = st.form_submit_button("Analyze")
    if sgo:
        d = fetch_live_data(sym, sday, sday)
        rth = get_session_window(d, RTH_START, RTH_END)
        if rth.empty:
            st.error("No RTH data for that day.")
        else:
            ema8  = calculate_ema(rth["Close"], 8)
            ema21 = calculate_ema(rth["Close"], 21)
            out=[]
            for i in range(1, len(rth)):
                t = format_ct_time(rth.index[i])
                p = float(rth["Close"].iloc[i])
                p8_prev, p21_prev = float(ema8.iloc[i-1]), float(ema21.iloc[i-1])
                p8, p21 = float(ema8.iloc[i]), float(ema21.iloc[i])
                cross = "None"
                if p8_prev <= p21_prev and p8 > p21: cross="Bullish Cross"
                if p8_prev >= p21_prev and p8 < p21: cross="Bearish Cross"
                sep = (abs(p8 - p21)/p21*100) if p21 != 0 else 0
                regime = "Bullish" if p8 > p21 else "Bearish"
                out.append({"Time":t,"Price":round(p,2),"EMA8":round(p8,2),"EMA21":round(p21,2),
                            "Separation_%":round(sep,3),"Regime":regime,"Crossover":cross})
            st.dataframe(pd.DataFrame(out), use_container_width=True, hide_index=True)
            st.markdown("<hr>", unsafe_allow_html=True)
            vol = calculate_volatility(rth)
            atr = calculate_average_true_range(rth, 14)
            c1,c2 = st.columns(2)
            with c1: st.markdown(f'<div class="kpi"><h3>Volatility</h3><div class="big">{vol:.2f}%</div></div>', unsafe_allow_html=True)
            with c2: st.markdown(f'<div class="kpi"><h3>Current ATR</h3><div class="big">{(atr.iloc[-1] if not atr.empty else 0):.2f}</div></div>', unsafe_allow_html=True)

# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║ TAB 4: Contract Tool (simple)                                               ║
# ╚═════════════════════════════════════════════════════════════════════════════╝
with tab4:
    st.markdown("#### 📦 Contract Tool (Overnight → RTH Projection)")
    pc1, pc2 = st.columns(2)
    with pc1:
        p1_date = st.date_input("Point 1 Date", value=today_ct - timedelta(days=1))
        p1_time = st.time_input("Point 1 Time (CT)", value=time(20,0))
        p1_px   = st.number_input("Point 1 Price", value=10.00, min_value=0.01, step=0.01, format="%.2f")
    with pc2:
        p2_date = st.date_input("Point 2 Date", value=today_ct)
        p2_time = st.time_input("Point 2 Time (CT)", value=time(8,0))
        p2_px   = st.number_input("Point 2 Price", value=12.00, min_value=0.01, step=0.01, format="%.2f")
    proj_day_ct = st.date_input("Projection Day", value=p2_date)
    if st.button("Project Contract"):
        p1_dt = to_ct(datetime.combine(p1_date, p1_time))
        p2_dt = to_ct(datetime.combine(p2_date, p2_time))
        if p2_dt <= p1_dt:
            st.error("Point 2 must be after Point 1.")
        else:
            blocks = (p2_dt - p1_dt).total_seconds() / 1800.0
            slope  = (p2_px - p1_px) / blocks if blocks > 0 else 0.0
            rows=[]
            for slot in rth_slots_ct(proj_day_ct):
                b = (slot - p1_dt).total_seconds()/1800.0
                rows.append({"Time": format_ct_time(slot), "Contract_Price": round(p1_px + slope*b, 2)})
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

# ───────────────────────────────────────────────────────────────────────────────
# FOOTER / QUICK TEST
# ───────────────────────────────────────────────────────────────────────────────
st.markdown("<hr>", unsafe_allow_html=True)
if st.button("🔌 Test Data Connection"):
    test = fetch_live_data("^GSPC", today_ct - timedelta(days=5), today_ct)
    if not test.empty:
        st.success(f"Connection OK — {len(test)} intraday bars received.")
    else:
        st.error("Market data connection failed (empty from Yahoo).")