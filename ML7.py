# ============================================================
# MarketLens Pro v5 — Part 1 of N
# Core bootstrap: layout, theming, router, safe utilities
# Append-only build: later parts are pasted BELOW this block.
# ============================================================

# (optional) pip:
#   pip install streamlit pandas numpy yfinance pytz tzdata

from __future__ import annotations
import os
from datetime import datetime
from typing import Dict

import pandas as pd
import numpy as np
import streamlit as st

# ---- Timezones (prefer zoneinfo; fallback to pytz) ----
try:
    from zoneinfo import ZoneInfo
    CT = ZoneInfo("America/Chicago")
    UTC = ZoneInfo("UTC")
except Exception:
    import pytz  # type: ignore
    CT = pytz.timezone("America/Chicago")
    UTC = pytz.UTC

# ---- Optional provider check (no calls yet in Part 1) ----
try:
    import yfinance as yf  # noqa: F401
    HAS_YF = True
except Exception:
    HAS_YF = False


# -----------------------------
# App metadata & configuration
# -----------------------------
APP_NAME = "MarketLens Pro v5"
APP_TAGLINE = "Professional SPX & Equities Analysis"
APP_VERSION = "5.0.0-P1"

st.set_page_config(
    page_title=APP_NAME,
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)


# -----------------------------
# Hidden model constants (YOUR LATEST)
# -----------------------------
# Default Slopes per 30-minute block (SPX)
SPX_SLOPES = {
    "skyline": 0.268,   # positive
    "baseline": -0.235  # negative
}

# Stock slope magnitudes (used as +/- for Skyline/Baseline)
STOCK_SLOPES = {
    "AAPL": 0.0155,
    "MSFT": 0.0541,
    "NVDA": 0.0086,
    "AMZN": 0.0139,
    "GOOGL": 0.0122,
    "TSLA": 0.0285,
    "META": 0.0674,
    "NFLX": 0.0230,
}

# Core symbols list (used later)
CORE_SYMBOLS = list(STOCK_SLOPES.keys())


# -----------------------------
# Utilities
# -----------------------------
def get_env_flag(name: str, default: bool = False) -> bool:
    v = str(os.environ.get(name, "")).strip().lower()
    if v in ("1", "true", "yes", "y", "on"):
        return True
    if v in ("0", "false", "no", "n", "off"):
        return False
    return default


def now_ct() -> datetime:
    """Current time in America/Chicago (DST-aware)."""
    return datetime.now(tz=CT)


def to_ct(dt: datetime) -> datetime:
    """Convert any aware/naive datetime to CT."""
    if getattr(dt, "tzinfo", None) is None:
        return dt.replace(tzinfo=CT)  # assume CT if naive
    try:
        return dt.astimezone(CT)
    except Exception:
        return dt.astimezone(CT)


def to_utc(dt: datetime) -> datetime:
    """Convert any aware/naive datetime to UTC."""
    if getattr(dt, "tzinfo", None) is None:
        return dt.replace(tzinfo=UTC)
    try:
        return dt.astimezone(UTC)
    except Exception:
        return dt.astimezone(UTC)


def ensure_dtindex_utc(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure a UTC DatetimeIndex (robust across pandas versions)."""
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("DataFrame index must be a pandas.DatetimeIndex")
    tz = getattr(df.index, "tz", None)
    if tz is None:
        df.index = df.index.tz_localize(UTC)
    elif tz != UTC:
        df.index = df.index.tz_convert(UTC)
    return df


# -----------------------------
# Session state initialization
# -----------------------------
DEFAULT_THEME = {
    "bg_gradient": (
        "radial-gradient(1200px 800px at 10% -10%, rgba(255,255,255,0.06), transparent), "
        "linear-gradient(135deg, #0e0f13 0%, #0a0b10 40%, #0b111b 100%)"
    ),
    "card_bg": "rgba(255,255,255,0.06)",
    "card_border": "rgba(255,255,255,0.10)",
    "accent": "#7dd3fc",      # sky-300
    "accent_hi": "#22d3ee",   # cyan-400
    "text": "rgba(255,255,255,0.92)",
    "muted": "rgba(255,255,255,0.65)",
    "danger": "#ef4444",
    "success": "#22c55e",
    "warning": "#f59e0b",
}

def init_state() -> None:
    ss = st.session_state
    ss.setdefault("theme", DEFAULT_THEME.copy())
    ss.setdefault("spx_slopes", SPX_SLOPES.copy())     # hidden
    ss.setdefault("stock_slopes", STOCK_SLOPES.copy()) # hidden
    ss.setdefault("settings", {
        "timezone": "America/Chicago",
        "debug": get_env_flag("MLPRO_DEBUG", False),
    })
    ss.setdefault("nav", "Dashboard")

init_state()


# -----------------------------
# Theming (CSS injection)
# -----------------------------
def inject_css(theme: Dict[str, str]) -> None:
    bg = theme["bg_gradient"]
    card = theme["card_bg"]
    border = theme["card_border"]
    text = theme["text"]
    muted = theme["muted"]
    accent = theme["accent"]

    css = f"""
    <style>
      /* App background */
      [data-testid="stAppViewContainer"] {{
        background: {bg};
      }}

      /* Subtle star sparkle */
      @keyframes floatSparkles {{
        0% {{ background-position: 0px 0px, 0px 0px; }}
        100% {{ background-position: 1000px 1000px, -800px 600px; }}
      }}
      [data-testid="stAppViewContainer"]::before {{
        content: "";
        position: fixed; inset: 0; pointer-events: none;
        background-image:
          radial-gradient(2px 2px at 20% 30%, rgba(255,255,255,0.12), transparent 40%),
          radial-gradient(1px 1px at 70% 10%, rgba(255,255,255,0.10), transparent 45%);
        animation: floatSparkles 60s linear infinite;
      }}

      /* Global type color */
      html, body, [data-testid="stAppViewContainer"] * {{ color: {text}; }}

      /* Sidebar (future-proof) */
      section[data-testid="stSidebar"] > div {{
        background: {card};
        backdrop-filter: blur(22px);
        border-right: 1px solid {border};
      }}
      section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
      section[data-testid="stSidebar"] label,
      section[data-testid="stSidebar"] span {{ color: {muted}; }}

      /* Glass cards */
      .ml-card {{
        background: {card};
        border: 1px solid {border};
        border-radius: 16px;
        padding: 18px 18px;
        box-shadow: 0 8px 30px rgba(0,0,0,0.18);
        transition: transform .25s ease, box-shadow .25s ease, border-color .25s ease;
      }}
      .ml-card:hover {{
        transform: translateY(-2px);
        box-shadow: 0 16px 54px rgba(0,0,0,0.25);
        border-color: {accent};
      }}

      .pill {{
        display:inline-block; padding:2px 10px; border-radius:999px;
        border:1px solid {border}; background: rgba(255,255,255,0.04);
        color:{muted}; font-size:12px;
      }}
      .brand-title {{ font-weight:700; letter-spacing:.3px; font-size:20px; }}
      .muted {{ color:{muted}; }}

      .stButton > button {{
        border-radius: 12px; border:1px solid {border};
        background: rgba(255,255,255,0.05);
        transition: box-shadow .2s ease, transform .2s ease, border-color .2s ease;
      }}
      .stButton > button:hover {{
        border-color:{accent}; transform: translateY(-1px);
        box-shadow:0 10px 30px rgba(0,0,0,0.25);
      }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

inject_css(st.session_state["theme"])


# -----------------------------
# Page stubs (real logic added later via append-only upgrades)
# -----------------------------
def page_dashboard():
    st.markdown("### Overview")
    n_consts = len(st.session_state["spx_slopes"]) + len(st.session_state["stock_slopes"])
    st.markdown(
        f"""
        <div class="ml-card">
          <div style="font-size:14px" class="muted">
            Part 1 (layout, theme, router, utilities). Later parts will add data fetchers,
            SPX Skyline/Baseline, Stock anchors, Signals/EMA, Analytics/Backtest, and Contract Tool —
            while keeping slopes hidden from the UI.
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            """
            <div class="ml-card">
              <div class="muted">Timezone</div>
              <div style="font-size:22px; font-weight:700;">America/Chicago</div>
              <div class="muted" style="margin-top:6px;">DST-aware</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f"""
            <div class="ml-card">
              <div class="muted">Hidden Model Vars</div>
              <div style="font-size:22px; font-weight:700;">{n_consts} constants</div>
              <div class="muted" style="margin-top:6px;">Kept out of UI</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            """
            <div class="ml-card">
              <div class="muted">Theme</div>
              <div style="font-size:22px; font-weight:700;">Glassmorphism</div>
              <div class="muted" style="margin-top:6px;">Cosmic gradient</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def page_spx_skyline():
    st.markdown("### SPX Skyline (placeholder)")
    st.info("This tab will host SPX Skyline projections and channel logic in Part 2+.")


def page_spx_baseline():
    st.markdown("### SPX Baseline (placeholder)")
    st.info("This tab will host SPX Baseline projections and channel logic in Part 2+.")


def page_stocks_skyline():
    st.markdown("### Stocks Skyline (placeholder)")
    st.info("This tab will host per-ticker Skyline projections in Part 3+.")


def page_stocks_baseline():
    st.markdown("### Stocks Baseline (placeholder)")
    st.info("This tab will host per-ticker Baseline projections in Part 3+.")


def page_settings():
    st.markdown("### Settings")
    st.caption("Non-sensitive preferences only; model constants remain internal.")
    debug = st.toggle("Debug mode", value=st.session_state["settings"]["debug"])
    st.session_state["settings"]["debug"] = debug
    st.success("Settings saved.")


def page_about():
    st.markdown("### About")
    st.write(
        f"""
        **{APP_NAME}** — {APP_TAGLINE}  
        Version: {APP_VERSION}

        • Timezone: America/Chicago (CT)  
        • Data: yfinance {'available' if HAS_YF else 'not detected'}  
        • UI: Dark glassmorphism with cosmic gradient
        """
    )


# Router map (we will update entries later in append-only Parts)
PAGES = {
    "Dashboard": page_dashboard,
    "SPX • Skyline": page_spx_skyline,
    "SPX • Baseline": page_spx_baseline,
    "Stocks • Skyline": page_stocks_skyline,
    "Stocks • Baseline": page_stocks_baseline,
    "Settings": page_settings,
    "About": page_about,
}


# -----------------------------
# Sidebar (router)
# -----------------------------
with st.sidebar:
    st.markdown("#### Navigation")
    choice = st.radio(
        "Choose a section",
        list(PAGES.keys()),
        index=list(PAGES.keys()).index(st.session_state["nav"]),
        label_visibility="collapsed",
    )
    st.session_state["nav"] = choice

    st.divider()
    st.markdown("#### Quick Info")
    st.caption(f"Local time (CT): {now_ct():%Y-%m-%d %I:%M %p}")


# -----------------------------
# Main render (deferred)
# -----------------------------
def __MLP_RENDER__():
    # Top bar
    t = now_ct().strftime("%a, %b %d • %I:%M %p CT")
    cols = st.columns([1.2, 1, 1, 1])
    with cols[0]:
        st.markdown(
            f"""
            <div class="ml-card" style="padding:14px 16px; display:flex; align-items:center; gap:12px;">
              <div class="pill">v{APP_VERSION}</div>
              <div class="brand-title">{APP_NAME}</div>
              <div class="muted" style="margin-left:8px;">{APP_TAGLINE}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with cols[1]:
        st.markdown(
            f"""
            <div class="ml-card" style="text-align:center">
              <div class="muted">Clock</div>
              <div style="font-size:18px; font-weight:600">{t}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with cols[2]:
        st.markdown(
            f"""
            <div class="ml-card" style="text-align:center">
              <div class="muted">Data Provider</div>
              <div style="font-size:18px; font-weight:600">{'yfinance ✓' if HAS_YF else 'yfinance —'}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with cols[3]:
        st.markdown(
            f"""
            <div class="ml-card" style="text-align:center">
              <div class="muted">Mode</div>
              <div style="font-size:18px; font-weight:600">
                {'Debug' if st.session_state['settings']['debug'] else 'Standard'}
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Page body
    PAGES[st.session_state["nav"]]()








# ======================================================================
# Part 2 — SPX Skyline foundation:
# - yfinance fetchers
# - ES Asian-session anchor window (17:00–19:30 CT, previous trading day)
# - Swing selectivity (k), ES→SPX offset
# - Skyline/Baseline projections for 08:30–14:30 CT
# - Append-only: swaps router entry, then calls deferred render
# ======================================================================

from datetime import date, time, timedelta

# -----------------------------
# Trading-day helpers & time grids
# -----------------------------
RTH_START = time(8, 30)   # 08:30 CT
RTH_END   = time(14, 30)  # 14:30 CT

def previous_weekday(d: date) -> date:
    while d.weekday() >= 5:
        d = d - timedelta(days=1)
    return d

def next_weekday(d: date) -> date:
    while d.weekday() >= 5:
        d = d + timedelta(days=1)
    return d

def prev_trading_day_base(now_ct_val: datetime) -> date:
    return previous_weekday((now_ct_val - timedelta(days=1)).date())

def projection_day_default(now_ct_val: datetime) -> date:
    d = now_ct_val.date()
    return d if d.weekday() < 5 else next_weekday(d)

def generate_rth_times_ct(day: date):
    slots = []
    t = datetime.combine(day, RTH_START, tzinfo=CT)
    end_dt = datetime.combine(day, RTH_END, tzinfo=CT)
    while t <= end_dt:
        slots.append(t)
        t += timedelta(minutes=30)
    return slots

def floor_to_30min(dt_ct: datetime) -> datetime:
    return dt_ct.replace(minute=(dt_ct.minute // 30) * 30, second=0, microsecond=0)

# -----------------------------
# Data fetchers (yfinance)
# -----------------------------
def _yf_download(symbol: str, start_dt_utc: datetime, end_dt_utc: datetime, interval: str = "30m") -> pd.DataFrame:
    if not HAS_YF:
        return pd.DataFrame()
    df = yf.download(
        symbol,
        interval=interval,
        start=start_dt_utc.replace(tzinfo=None),
        end=end_dt_utc.replace(tzinfo=None),
        progress=False,
        auto_adjust=True,
        prepost=False,
        threads=True,
    )
    if df is None or df.empty:
        return pd.DataFrame()
    df.index = pd.to_datetime(df.index)
    return ensure_dtindex_utc(df)

def fetch_es_30m_for_prev_day_window(prev_day: date) -> pd.DataFrame:
    """
    Fetch ES futures (ES=F) for the previous trading day's Asian window 17:00–19:30 CT.
    Pull with buffer and filter in CT.
    """
    start_ct = datetime.combine(prev_day, time(0, 0), tzinfo=CT)
    end_ct   = datetime.combine(prev_day + timedelta(days=1), time(23, 59), tzinfo=CT)
    df = _yf_download("ES=F", to_utc(start_ct), to_utc(end_ct), interval="30m")
    if df.empty:
        return df
    df_ct = df.tz_convert(CT)
    win_start = datetime.combine(prev_day, time(17, 0), tzinfo=CT)
    win_end   = datetime.combine(prev_day, time(19, 30), tzinfo=CT)
    df_ct = df_ct.loc[(df_ct.index >= win_start) & (df_ct.index <= win_end)].copy()
    cols = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df_ct.columns]
    return df_ct[cols]

# -----------------------------
# Anchor selection (k-th extremes by CLOSE)
# -----------------------------
def kth_extreme_by_close(df_ct: pd.DataFrame, k: int = 1):
    """
    Return (kth_high_close_price, timestamp), (kth_low_close_price, timestamp) within df.
    k=1 -> highest/lowest, k=2 -> second highest/lowest, etc.
    """
    if df_ct.empty or "Close" not in df_ct.columns:
        return (None, None), (None, None)

    highs = df_ct.sort_values("Close", ascending=False).iloc[:k]
    lows  = df_ct.sort_values("Close", ascending=True).iloc[:k]
    if len(highs) < k or len(lows) < k:
        return (None, None), (None, None)

    kth_high_row = highs.iloc[-1]
    kth_low_row  = lows.iloc[-1]
    kth_high_price = float(kth_high_row["Close"])
    kth_high_time  = highs.index[-1]
    kth_low_price  = float(kth_low_row["Close"])
    kth_low_time   = lows.index[-1]
    return (kth_high_price, kth_high_time), (kth_low_price, kth_low_time)

# -----------------------------
# Projection builder
# -----------------------------
def build_projection_table(anchor_price: float, anchor_time_ct: datetime, slope_per_block: float, slots_ct: list[datetime]) -> pd.DataFrame:
    """
    Slope units = price change per 30-min block.
    Price(t) = anchor_price + slope * blocks_since_anchor
    """
    rows = []
    for t in slots_ct:
        delta_minutes = (t - anchor_time_ct).total_seconds() / 60.0
        blocks = int(np.floor(delta_minutes / 30.0))
        price = anchor_price + slope_per_block * blocks
        rows.append({"Time_CT": t.strftime("%Y-%m-%d %H:%M"), "Price": round(price, 4)})
    return pd.DataFrame(rows)

# -----------------------------
# UI: SPX Skyline upgraded page
# -----------------------------
def page_spx_skyline_v2():
    st.markdown("### SPX Skyline")

    colA, colB, colC, colD = st.columns([1.2, 1, 1, 1.2])
    with colA:
        prev_day_default = prev_trading_day_base(now_ct())
        prev_day_input = st.date_input(
            "Previous trading day (for ES window)",
            value=prev_day_default,
            help="We use ES=F 30m candles in the 17:00–19:30 CT window of this day."
        )
    with colB:
        k = st.selectbox("Swing selectivity (k)", [1, 2, 3], index=0, help="k=1 highest/lowest close; k=2 second; k=3 third")
    with colC:
        es_to_spx_offset = st.number_input(
            "ES→SPX offset",
            value=0.0,
            step=0.5,
            help="Manual adjustment to convert ES close to an SPX anchor (points)."
        )
    with colD:
        projection_day = st.date_input(
            "Projection day",
            value=projection_day_default(now_ct()),
            help="Projections are generated for 08:30–14:30 CT of this day."
        )

    df_es_win = fetch_es_30m_for_prev_day_window(prev_day_input)
    if df_es_win.empty:
        st.error("No ES data returned for the selected previous day. Try another date or check your connection.")
        return

    (hi_price, hi_time), (lo_price, lo_time) = kth_extreme_by_close(df_es_win, k=k)
    if hi_price is None or lo_price is None:
        st.error("Could not compute k-th extremes from ES window.")
        return

    # Adjust ES anchors → SPX anchors via offset; ensure CT-aware
    hi_price_adj = hi_price + es_to_spx_offset
    lo_price_adj = lo_price + es_to_spx_offset
    hi_time_ct = hi_time if getattr(hi_time, "tzinfo", None) else CT.localize(hi_time)
    lo_time_ct = lo_time if getattr(lo_time, "tzinfo", None) else CT.localize(lo_time)

    slots_ct = generate_rth_times_ct(projection_day)
    spx_skyline_slope  = st.session_state["spx_slopes"]["skyline"]
    spx_baseline_slope = st.session_state["spx_slopes"]["baseline"]

    # Convention:
    # - Skyline projects from k-th HIGH close
    # - Baseline projects from k-th LOW close
    sky_df  = build_projection_table(hi_price_adj, hi_time_ct, spx_skyline_slope,  slots_ct)
    base_df = build_projection_table(lo_price_adj, lo_time_ct, spx_baseline_slope, slots_ct)

    # Persist anchors for later tabs
    st.session_state["spx_anchors"] = {
        "previous_day": prev_day_input,
        "projection_day": projection_day,
        "skyline": {"price": hi_price_adj, "time": hi_time_ct},
        "baseline": {"price": lo_price_adj, "time": lo_time_ct},
        "k": k,
        "offset": es_to_spx_offset,
    }

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Skyline anchor** (k-th high close)")
        st.caption(f"Time: {hi_time_ct.strftime('%Y-%m-%d %H:%M CT')} • Price (ES+offset): {hi_price_adj:.4f} • k={k}")
        st.dataframe(sky_df, hide_index=True, use_container_width=True)
        st.download_button(
            "📥 Download Skyline CSV",
            data=sky_df.to_csv(index=False),
            file_name=f"SPX_Skyline_{projection_day}.csv",
            mime="text/csv",
            key="dl_skyline_csv_p2",
        )
    with c2:
        st.markdown("**Baseline anchor** (k-th low close)")
        st.caption(f"Time: {lo_time_ct.strftime('%Y-%m-%d %H:%M CT')} • Price (ES+offset): {lo_price_adj:.4f} • k={k}")
        st.dataframe(base_df, hide_index=True, use_container_width=True)
        st.download_button(
            "📥 Download Baseline CSV",
            data=base_df.to_csv(index=False),
            file_name=f"SPX_Baseline_{projection_day}.csv",
            mime="text/csv",
            key="dl_baseline_csv_p2",
        )

    with st.expander("ES 30m window (prev day 17:00–19:30 CT)", expanded=False):
        df_show = df_es_win.copy()
        df_show.index = df_show.index.tz_convert(CT)
        st.dataframe(df_show, use_container_width=True)

    st.success("SPX Skyline/Baseline projections generated.")

# Upgrade router entry (append-only)
PAGES["SPX • Skyline"] = page_spx_skyline_v2

# -----------------------------
# Auto-render footer (call once)
# -----------------------------
if not st.session_state.get("_mlp_rendered", False):
    st.session_state["_mlp_rendered"] = True
    __MLP_RENDER__()
