# ============================================================
# MarketLens Pro v5 — PART 1/?
# Single-file app split across parts. Paste ALL parts into one file.
# This part sets up: imports, constants, state, utils, theming, pages (stubs).
# NOTE: Do NOT run yet. The final part will call render_app().
# ============================================================

from __future__ import annotations

# ---------- Standard libs ----------
import os
from datetime import datetime, date, time, timedelta
from typing import Dict, List, Tuple, Optional

# ---------- Third-party ----------
import pandas as pd
import numpy as np
import streamlit as st

# Timezones: prefer zoneinfo; fall back to pytz
try:
    from zoneinfo import ZoneInfo
    CT = ZoneInfo("America/Chicago")
    UTC = ZoneInfo("UTC")
    _HAS_PYTZ = False
except Exception:
    import pytz  # type: ignore
    CT = pytz.timezone("America/Chicago")
    UTC = pytz.UTC
    _HAS_PYTZ = True

# Optional data provider (used in later parts)
try:
    import yfinance as yf  # noqa: F401
    HAS_YF = True
except Exception:
    HAS_YF = False

# ---------- App meta (used later inside render_app) ----------
APP_NAME = "MarketLens Pro v5"
APP_TAGLINE = "Analytics for SPX Entries (No Simulation • Advisory Only)"
APP_VERSION = "6.0.0"

# ============================================================
# HIDDEN MODEL CONSTANTS (your latest slopes)
# ============================================================
# SPX per 30-minute block
SPX_SLOPES = {
    "skyline": 0.268,   # positive
    "baseline": -0.235  # negative
}

# Stock slope magnitudes (± per 30-minute block)
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
CORE_SYMBOLS = list(STOCK_SLOPES.keys())

# ============================================================
# STRATEGY HYPOTHESES (scaffold; analytics only)
# ============================================================
HYPOTHESES = {
    "H1_BASELINE_TOUCH_LONG": dict(
        id="H1_BASELINE_TOUCH_LONG",
        title="Baseline Touch → Long (Call)",
        definition=(
            "If RTH underlying price touches the Baseline anchor (drawn from the previous session window), "
            "probability of an upward reaction increases; analyze historical frequency of bounce vs fail."
        ),
    ),
    "H2_SKYLINE_OVN_DROP_SHORT": dict(
        id="H2_SKYLINE_OVN_DROP_SHORT",
        title="Skyline Touch + Overnight Drop → Short (Put)",
        definition=(
            "If underlying touches Skyline anchor and subsequently trades below it through the overnight, "
            "opening below into RTH, probability of downside continuation increases; analyze follow-through vs reclaim."
        ),
    ),
}

PROB_KEYS = [
    "direction_prob_up",     # P(up over next N blocks) given trigger
    "direction_prob_down",   # P(down over next N blocks) given trigger
    "entry_success_prob",    # P(target before stop) given trigger
    "exit_success_prob",     # P(exit achieves chosen RR/time criteria)
]

# ============================================================
# UTILS (datetime, tz, small helpers)
# ============================================================
def get_env_flag(name: str, default: bool = False) -> bool:
    v = str(os.environ.get(name, "")).strip().lower()
    if v in ("1", "true", "yes", "y", "on"):
        return True
    if v in ("0", "false", "no", "n", "off"):
        return False
    return default

def now_ct() -> datetime:
    return datetime.now(tz=CT)

def to_ct(dt: datetime) -> datetime:
    if getattr(dt, "tzinfo", None) is None:
        return dt.replace(tzinfo=CT) if not _HAS_PYTZ else CT.localize(dt)
    try:
        return dt.astimezone(CT)
    except Exception:
        try:
            return dt.tz_convert(CT)  # pandas Timestamp
        except Exception:
            return dt

def to_utc(dt: datetime) -> datetime:
    if getattr(dt, "tzinfo", None) is None:
        return dt.replace(tzinfo=UTC) if not _HAS_PYTZ else UTC.localize(dt)
    try:
        return dt.astimezone(UTC)
    except Exception:
        try:
            return dt.tz_convert(UTC)
        except Exception:
            return dt

def ensure_dtindex_utc(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("DataFrame index must be a pandas.DatetimeIndex")
    tz = getattr(df.index, "tz", None)
    if tz is None:
        df.index = df.index.tz_localize(UTC)
    elif tz != UTC:
        df.index = df.index.tz_convert(UTC)
    return df

# Trading day helpers (used later)
def previous_weekday(d: date) -> date:
    while d.weekday() >= 5:
        d -= timedelta(days=1)
    return d

def next_weekday(d: date) -> date:
    while d.weekday() >= 5:
        d += timedelta(days=1)
    return d

# ============================================================
# SESSION STATE INIT
# ============================================================
DEFAULT_THEME = {
    "bg_gradient": (
        "radial-gradient(1200px 800px at 10% -10%, rgba(255,255,255,0.06), transparent), "
        "linear-gradient(135deg, #0e0f13 0%, #0a0b10 40%, #0b111b 100%)"
    ),
    "card_bg": "rgba(255,255,255,0.06)",
    "card_border": "rgba(255,255,255,0.10)",
    "accent": "#7dd3fc",
    "accent_hi": "#22d3ee",
    "text": "rgba(255,255,255,0.92)",
    "muted": "rgba(255,255,255,0.65)",
    "danger": "#ef4444",
    "success": "#22c55e",
    "warning": "#f59e0b",
}

def init_state() -> None:
    ss = st.session_state
    ss.setdefault("theme", DEFAULT_THEME.copy())

    # Hidden model constants
    ss.setdefault("spx_slopes", SPX_SLOPES.copy())
    ss.setdefault("stock_slopes", STOCK_SLOPES.copy())

    # Analytics config (used by probability engine later)
    ss.setdefault("analytics", {
        "lookahead_blocks": 4,  # N x 30-min blocks
        "rr_target": 1.0,       # reward:risk for success calc
        "stop_blocks": 2,       # stop horizon in blocks
        "use_close_only": True, # close-to-close computations
    })

    # Contract tool (user will input overnight contract prices)
    ss.setdefault("contracts", {
        "overnight_call_entry_price": None,  # call at swing low
        "overnight_put_entry_price": None,   # put at skyline touch
    })

    ss.setdefault("strategy_hypotheses", HYPOTHESES.copy())
    ss.setdefault("settings", {
        "timezone": "America/Chicago",
        "debug": get_env_flag("MLPRO_DEBUG", False),
    })
    ss.setdefault("nav", "Dashboard")

# ============================================================
# THEMING (brace-safe CSS via Template; no f-strings here)
# ============================================================
from string import Template

def inject_css(theme: Dict[str, str]) -> None:
    bg     = theme["bg_gradient"]
    card   = theme["card_bg"]
    border = theme["card_border"]
    text   = theme["text"]
    muted  = theme["muted"]
    accent = theme["accent"]

    tpl = Template("""
    <style>
      /* App background */
      [data-testid="stAppViewContainer"] {
        background: $bg;
      }
      /* Subtle star sparkle */
      @keyframes floatSparkles {
        0% { background-position: 0px 0px, 0px 0px; }
        100% { background-position: 1000px 1000px, -800px 600px; }
      }
      [data-testid="stAppViewContainer"]::before {
        content: "";
        position: fixed; inset: 0; pointer-events: none;
        background-image:
          radial-gradient(2px 2px at 20% 30%, rgba(255,255,255,0.12), transparent 40%),
          radial-gradient(1px 1px at 70% 10%, rgba(255,255,255,0.10), transparent 45%);
        animation: floatSparkles 60s linear infinite;
      }
      /* Global type color */
      html, body, [data-testid="stAppViewContainer"] * { color: $text; }
      /* Sidebar (future-proof) */
      section[data-testid="stSidebar"] > div {
        background: $card;
        backdrop-filter: blur(22px);
        border-right: 1px solid $border;
      }
      section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
      section[data-testid="stSidebar"] label,
      section[data-testid="stSidebar"] span { color: $muted; }
      /* Glass cards */
      .ml-card {
        background: $card;
        border: 1px solid $border;
        border-radius: 16px;
        padding: 18px 18px;
        box-shadow: 0 8px 30px rgba(0,0,0,0.18);
        transition: transform .25s ease, box-shadow .25s ease, border-color .25s ease;
      }
      .ml-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 16px 54px rgba(0,0,0,0.25);
        border-color: $accent;
      }
      .pill {
        display:inline-block; padding:2px 10px; border-radius:999px;
        border:1px solid $border; background: rgba(255,255,255,0.04);
        color:$muted; font-size:12px;
      }
      .brand-title { font-weight:700; letter-spacing:.3px; font-size:20px; }
      .muted { color:$muted; }
      .stButton > button {
        border-radius: 12px; border:1px solid $border;
        background: rgba(255,255,255,0.05);
        transition: box-shadow .2s ease, transform .2s ease, border-color .2s ease;
      }
      .stButton > button:hover {
        border-color:$accent; transform: translateY(-1px);
        box-shadow:0 10px 30px rgba(0,0,0,0.25);
      }
    </style>
    """)
    css = tpl.substitute(bg=bg, card=card, border=border, text=text, muted=muted, accent=accent)
    st.markdown(css, unsafe_allow_html=True)

# ============================================================
# UI COMPONENTS (topbar)
# ============================================================
def topbar():
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

# ============================================================
# PAGE STUBS (real logic added in later parts)
# ============================================================
def page_dashboard():
    st.markdown("### Overview")
    n_consts = len(st.session_state["spx_slopes"]) + len(st.session_state["stock_slopes"])
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            """
            <div class="ml-card">
              <div class="muted">Strategy Focus</div>
              <div style="font-size:20px; font-weight:700;">Analytics, not Simulation</div>
              <div class="muted" style="margin-top:6px;">
                Probabilities for direction, entry success, and exit success — advisory only.
              </div>
            </div>
            """, unsafe_allow_html=True)
    with c2:
        st.markdown(
            f"""
            <div class="ml-card">
              <div class="muted">Hidden Model Vars</div>
              <div style="font-size:22px; font-weight:700;">{n_consts} constants</div>
              <div class="muted" style="margin-top:6px;">Slopes kept out of UI</div>
            </div>
            """, unsafe_allow_html=True)
    with c3:
        st.markdown(
            """
            <div class="ml-card">
              <div class="muted">Key Hypotheses</div>
              <div style="font-size:14px; margin-top:6px;">
                • Baseline touch → Call bounce<br/>
                • Skyline touch + overnight drop → Put follow-through
              </div>
            </div>
            """, unsafe_allow_html=True)

def page_spx_skyline():  # upgraded in later part
    st.markdown("### SPX Skyline (placeholder)")
    st.info("Will compute Skyline projections and support probability triggers in later parts.")

def page_spx_baseline():  # upgraded in later part
    st.markdown("### SPX Baseline (placeholder)")
    st.info("Will compute Baseline projections and support probability triggers in later parts.")

def page_signals_probabilities():  # added in later part
    st.markdown("### Signals & Probabilities (placeholder)")
    st.caption("Direction/Entry/Exit probabilities — computed from historical analytics (no simulation).")

def page_contract_tool():  # added in later part
    st.markdown("### Contract Tool (placeholder)")
    st.caption("Input contract prices at swing lows/highs (overnight) to analyze RTH entries — advisory only.")

def page_stocks_skyline():  # added in later part
    st.markdown("### Stocks • Skyline (placeholder)")

def page_stocks_baseline():  # added in later part
    st.markdown("### Stocks • Baseline (placeholder)")

def page_settings():
    st.markdown("### Settings")
    st.caption("Non-sensitive preferences only; model constants remain internal.")
    cols = st.columns(3)
    with cols[0]:
        la = st.number_input("Lookahead blocks (30-min each)", min_value=1, max_value=16, value=4, step=1)
    with cols[1]:
        rr = st.number_input("Reward:Risk target", min_value=0.1, max_value=5.0, value=1.0, step=0.1)
    with cols[2]:
        sb = st.number_input("Stop horizon (blocks)", min_value=1, max_value=16, value=2, step=1)
    st.session_state.setdefault("analytics", {})
    st.session_state["analytics"].update({"lookahead_blocks": int(la), "rr_target": float(rr), "stop_blocks": int(sb)})

def page_about():
    st.markdown("### About")
    st.write(
        f"""**{APP_NAME}** — {APP_TAGLINE}
- Version: {APP_VERSION}
- Timezone: America/Chicago (CT)
- Data: yfinance {'available' if HAS_YF else 'not detected'}
- UI: Dark glassmorphism
- Mode: Analytics only (no simulation). Not a trading app."""
    )

# Router (entries may be replaced/extended by later parts)
PAGES = {
    "Dashboard": page_dashboard,
    "SPX • Skyline": page_spx_skyline,
    "SPX • Baseline": page_spx_baseline,
    "Signals • Probabilities": page_signals_probabilities,
    "Contract • Tool": page_contract_tool,
    "Stocks • Skyline": page_stocks_skyline,
    "Stocks • Baseline": page_stocks_baseline,
    "Settings": page_settings,
    "About": page_about,
}

# ============================================================
# APP RENDERER (called ONLY in the final part)
# ============================================================
def render_app() -> None:
    # Must be called before any other Streamlit UI calls in a real run,
    # but we purposely delay calling this until the final part.
    st.set_page_config(
        page_title=APP_NAME,
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Initialize state and theming
    init_state()
    inject_css(st.session_state["theme"])

    # Sidebar nav
    with st.sidebar:
        st.markdown("#### Navigation")
        choice = st.radio(
            "Choose a section",
            list(PAGES.keys()),
            index=list(PAGES.keys()).index(st.session_state.get("nav", "Dashboard")),
            label_visibility="collapsed",
        )
        st.session_state["nav"] = choice
        st.divider()
        st.markdown("#### Quick Info")
        st.caption(f"Local time (CT): {now_ct():%Y-%m-%d %I:%M %p}")

    # Top bar + page body
    topbar()
    PAGES[st.session_state["nav"]]()





# ============================================================
# MarketLens Pro v5 — PART 2/?
# Data helpers, ES window fetch, anchor selection, Skyline page v2
# (This file still won't run until the final part calls render_app().)
# ============================================================

# ---------- RTH & session windows ----------
RTH_START = time(8, 30)   # 08:30 CT
RTH_END   = time(14, 30)  # 14:30 CT

def prev_trading_day_base(now_ct_val: datetime) -> date:
    return previous_weekday((now_ct_val - timedelta(days=1)).date())

def projection_day_default(now_ct_val: datetime) -> date:
    d = now_ct_val.date()
    return d if d.weekday() < 5 else next_weekday(d)

def generate_rth_times_ct(day: date) -> List[datetime]:
    """30-min grid from 08:30 to 14:30 inclusive, in CT."""
    slots: List[datetime] = []
    t = datetime.combine(day, RTH_START, tzinfo=CT)
    end_dt = datetime.combine(day, RTH_END, tzinfo=CT)
    while t <= end_dt:
        slots.append(t)
        t += timedelta(minutes=30)
    return slots

def floor_to_30min(dt_ct: datetime) -> datetime:
    return dt_ct.replace(minute=(dt_ct.minute // 30) * 30, second=0, microsecond=0)

def _as_ct(dt_like) -> datetime:
    """Ensure tz-aware CT datetime from pandas.Timestamp or datetime."""
    if hasattr(dt_like, "to_pydatetime"):
        dt_like = dt_like.to_pydatetime()
    if getattr(dt_like, "tzinfo", None) is None:
        try:
            return dt_like.replace(tzinfo=CT)
        except Exception:
            return CT.localize(dt_like)  # pytz path
    try:
        return dt_like.astimezone(CT)
    except Exception:
        try:
            return dt_like.tz_convert(CT)  # pandas Timestamp
        except Exception:
            return dt_like

# ---------- yfinance fetchers ----------
def _yf_download(symbol: str, start_dt_utc: datetime, end_dt_utc: datetime, interval: str = "30m") -> pd.DataFrame:
    """Download symbol at interval; return UTC-indexed DataFrame with OHLCV columns when available."""
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
    Fetch ES futures (ES=F) for the previous trading day's 'Asian window' 17:00–19:30 CT.
    We pull a full surrounding range in UTC, then convert to CT and filter.
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
    return df_ct[cols] if cols else df_ct

# ---------- Anchor selection (k-th extremes by CLOSE) ----------
def kth_extreme_by_close(df_ct: pd.DataFrame, k: int = 1) -> Tuple[Tuple[Optional[float], Optional[datetime]], Tuple[Optional[float], Optional[datetime]]]:
    """
    Return ((kth_high_close, t_high), (kth_low_close, t_low)) from df.
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
    kth_low_price  = float(kth_low_row["Close"])

    # Use the last index from the slices (already aligned)
    kth_high_time = highs.index[-1]
    kth_low_time  = lows.index[-1]
    return (kth_high_price, kth_high_time), (kth_low_price, kth_low_time)

# ---------- Projection builder ----------
def build_projection_table(anchor_price: float, anchor_time_ct: datetime, slope_per_block: float, slots_ct: List[datetime]) -> pd.DataFrame:
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

# ---------- Skyline Page (v2) ----------
def page_spx_skyline_v2():
    st.markdown("### SPX Skyline")

    if not HAS_YF:
        st.warning("yfinance not available. Install it and rerun the full app: `pip install yfinance`.")
        return

    colA, colB, colC, colD = st.columns([1.2, 1, 1, 1.2])
    with colA:
        prev_day_default = prev_trading_day_base(now_ct())
        prev_day_input = st.date_input(
            "Previous trading day (for ES window)",
            value=prev_day_default,
            help="Uses ES=F 30m candles in the 17:00–19:30 CT window."
        )
    with colB:
        k = st.selectbox("Swing selectivity (k)", [1, 2, 3], index=0, help="k=1 highest/lowest close; k=2 second; k=3 third")
    with colC:
        es_to_spx_offset = st.number_input(
            "ES→SPX offset",
            value=0.0, step=0.5,
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
    hi_time_ct = _as_ct(hi_time)
    lo_time_ct = _as_ct(lo_time)

    slots_ct = generate_rth_times_ct(projection_day)
    spx_skyline_slope  = st.session_state["spx_slopes"]["skyline"]
    spx_baseline_slope = st.session_state["spx_slopes"]["baseline"]

    # Convention:
    # - Skyline projects from k-th HIGH close
    # - Baseline projects from k-th LOW close (shown side-by-side for context)
    sky_df  = build_projection_table(hi_price_adj, hi_time_ct, spx_skyline_slope,  slots_ct)
    base_df = build_projection_table(lo_price_adj, lo_time_ct, spx_baseline_slope, slots_ct)

    # Persist anchors for later tabs/analytics
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

# ---------- Upgrade router entry (append-only) ----------
PAGES["SPX • Skyline"] = page_spx_skyline_v2



# ============================================================
# MarketLens Pro v5 — PART 3/?
# SPX Baseline page v2 (reuses Part 2 helpers)
# ============================================================

def _anchor_summary_card(title: str, when_ct: datetime, price: float, extra: str = ""):
    st.markdown(
        f"""
        <div class="ml-card">
            <div style="font-size:0.9rem; opacity:.7; margin-bottom:.25rem;">{title}</div>
            <div style="font-size:1.2rem; font-weight:700;">{price:.4f}</div>
            <div class="muted" style="margin-top:.25rem;">
                {when_ct.strftime('%Y-%m-%d %H:%M CT')} {extra}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

def page_spx_baseline_v2():
    st.markdown("### SPX Baseline")

    if not HAS_YF:
        st.warning("yfinance not available. Install it before running the full app: `pip install yfinance`.")
        return

    # Reuse Skyline settings if available
    reuse = "spx_anchors" in st.session_state
    if reuse:
        with st.expander("Reuse Skyline settings", expanded=True):
            reuse = st.toggle(
                "Use Skyline k / previous day / projection day / ES→SPX offset",
                value=True
            )

    if reuse and "spx_anchors" in st.session_state:
        sky = st.session_state["spx_anchors"]
        prev_day_input   = sky["previous_day"]
        projection_day   = sky["projection_day"]
        k                = sky["k"]
        es_to_spx_offset = sky["offset"]
        st.caption("Using settings from Skyline.")
    else:
        colA, colB, colC, colD = st.columns([1.2, 1, 1, 1.2])
        with colA:
            prev_day_input = st.date_input(
                "Previous trading day (for ES window)",
                value=prev_trading_day_base(now_ct()),
                help="ES=F 30m candles, 17:00–19:30 CT window."
            )
        with colB:
            k = st.selectbox("Swing selectivity (k)", [1, 2, 3], index=0)
        with colC:
            es_to_spx_offset = st.number_input(
                "ES→SPX offset", value=0.0, step=0.5,
                help="Adjustment to convert ES close to an SPX anchor (points)."
            )
        with colD:
            projection_day = st.date_input(
                "Projection day",
                value=projection_day_default(now_ct()),
                help="08:30–14:30 CT will be projected."
            )

    # Fetch ES window & compute k-th extremes
    df_es_win = fetch_es_30m_for_prev_day_window(prev_day_input)
    if df_es_win.empty:
        st.error("No ES data returned for the selected previous day.")
        return

    (hi_price, hi_time), (lo_price, lo_time) = kth_extreme_by_close(df_es_win, k=k)
    if hi_price is None or lo_price is None:
        st.error("Could not compute k-th extremes from ES window.")
        return

    # Baseline uses LOW-side anchor; show HIGH for context
    lo_price_adj = (lo_price or 0.0) + es_to_spx_offset
    lo_time_ct   = _as_ct(lo_time)

    hi_price_adj = (hi_price or 0.0) + es_to_spx_offset
    hi_time_ct   = _as_ct(hi_time)

    # Build projection for RTH
    slots_ct = generate_rth_times_ct(projection_day)
    spx_baseline_slope = st.session_state["spx_slopes"]["baseline"]
    base_df = build_projection_table(lo_price_adj, lo_time_ct, spx_baseline_slope, slots_ct)

    # Save/merge into session anchors
    st.session_state.setdefault("spx_anchors", {})
    st.session_state["spx_anchors"].update({
        "previous_day": prev_day_input,
        "projection_day": projection_day,
        "k": k,
        "offset": es_to_spx_offset,
        "baseline": {"price": lo_price_adj, "time": lo_time_ct},
        # keep skyline if already present; otherwise store current hi as reference
        "skyline": st.session_state["spx_anchors"].get("skyline", {"price": hi_price_adj, "time": hi_time_ct}),
    })

    # UI
    cTop1, cTop2 = st.columns(2)
    with cTop1:
        _anchor_summary_card("Baseline anchor (k-th LOW close, ES+offset)", lo_time_ct, lo_price_adj, f"• k={k}")
    with cTop2:
        _anchor_summary_card("Skyline anchor (k-th HIGH close, ES+offset)", hi_time_ct, hi_price_adj, f"• k={k}")

    st.markdown("#### Baseline Projection (08:30–14:30 CT)")
    st.dataframe(base_df, hide_index=True, use_container_width=True)
    st.download_button(
        "📥 Download Baseline CSV",
        data=base_df.to_csv(index=False),
        file_name=f"SPX_Baseline_{projection_day}.csv",
        mime="text/csv",
        key="dl_baseline_csv_p3",
    )

    with st.expander("ES 30m window (prev day 17:00–19:30 CT)", expanded=False):
        df_show = df_es_win.copy()
        df_show.index = df_show.index.tz_convert(CT)
        st.dataframe(df_show, use_container_width=True)

    st.success("SPX Baseline projection generated.")

# Upgrade router entry (append-only)
PAGES["SPX • Baseline"] = page_spx_baseline_v2





# ============================================================
# MarketLens Pro v5 — PART 4/?
# Signals & Probabilities (analytics-only, no simulation)
# - Cached yfinance downloads
# - Historical scan of triggers:
#   • H1: Baseline-touch → Long (Call)
#   • H2: Skyline-touch then overnight drop → Short (Put)
# - Probabilities:
#   • direction_prob_up/down
#   • entry_success_prob (target before stop)
#   • exit_success_prob (favorable close at horizon)
# ============================================================

# -------- Cached downloads (UTC index) --------
@st.cache_data(show_spinner=False, ttl=3600, max_entries=64)
def _yf_download_cached(symbol: str, start_dt_utc: datetime, end_dt_utc: datetime, interval: str = "30m") -> pd.DataFrame:
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

def _fetch_symbol_30m_ct_range(symbol: str, start_ct: datetime, end_ct: datetime) -> pd.DataFrame:
    df = _yf_download_cached(symbol, to_utc(start_ct), to_utc(end_ct), interval="30m")
    if df.empty:
        return df
    return df.tz_convert(CT)

def fetch_rth_30m_for_day(symbol: str, day: date) -> pd.DataFrame:
    """Fetch an entire calendar day and filter to 08:30–14:30 CT to avoid boundary gaps."""
    start_ct = datetime.combine(day, time(0, 0), tzinfo=CT)
    end_ct   = datetime.combine(day, time(23, 59), tzinfo=CT)
    df_ct = _fetch_symbol_30m_ct_range(symbol, start_ct, end_ct)
    if df_ct.empty:
        return df_ct
    rth_mask = (df_ct.index.time >= time(8,30)) & (df_ct.index.time <= time(14,30))
    return df_ct.loc[rth_mask].copy()

# -------- Helper math for projected lines --------
def line_value_at(anchor_price: float, anchor_time_ct: datetime, slope_per_block: float, t: datetime) -> float:
    """Value of projected line at time t (30m blocks)."""
    blocks = int(np.floor((t - anchor_time_ct).total_seconds() / 60.0 / 30.0))
    return float(anchor_price + slope_per_block * blocks)

def first_touch_time(rth_df_ct: pd.DataFrame, anchor_price: float, anchor_time_ct: datetime,
                     slope_per_block: float, touch_pad: float = 0.0) -> Optional[datetime]:
    """
    Return the first RTH bar time where the bar's [Low, High] touches the projected line (±touch_pad).
    """
    if rth_df_ct.empty:
        return None
    for t, row in rth_df_ct.iterrows():
        lv = line_value_at(anchor_price, anchor_time_ct, slope_per_block, t)
        lo = float(row.get("Low", np.nan))
        hi = float(row.get("High", np.nan))
        if np.isnan(lo) or np.isnan(hi):
            continue
        if (lo - touch_pad) <= lv <= (hi + touch_pad):
            return _as_ct(t)
    return None

def evaluate_path_long(rth_df_ct: pd.DataFrame, entry_time: datetime, entry_price: float,
                       lookahead_blocks: int, risk_points: float, rr_target: float) -> Tuple[bool, bool, Optional[datetime]]:
    """
    Long logic (Baseline-touch):
      - entry_success: did High reach entry + rr*risk before Low hit entry - risk, within horizon?
      - exit_success:  Close at horizon >= entry (favorable close)
      - returns (entry_success, exit_success, time_of_resolution)
    """
    if rth_df_ct.empty:
        return (False, False, None)
    target = entry_price + rr_target * risk_points
    stop   = entry_price - risk_points

    # Walk forward up to N blocks
    times = [t for t in rth_df_ct.index if t >= entry_time][:lookahead_blocks+1]
    # If entry bar isn't exact index (e.g., touch within bar), include following bars
    if len(times) == 0:
        return (False, False, None)

    for t in times:
        row = rth_df_ct.loc[t]
        hi = float(row.get("High", np.nan))
        lo = float(row.get("Low", np.nan))
        if not np.isnan(hi) and hi >= target:
            # Check stop on same bar first — assume worst-case: stop before target if both touched
            if not np.isnan(lo) and lo <= stop:
                # Order ambiguity: treat as fail (conservative)
                return (False, False, t)
            return (True, True, t)  # target met ⇒ exit_success True by definition
        if not np.isnan(lo) and lo <= stop:
            return (False, False, t)

    # No target/stop: exit_success by time (close at horizon ≥ entry)
    horizon_t = times[-1]
    close_h = float(rth_df_ct.loc[horizon_t].get("Close", np.nan))
    exit_success = (not np.isnan(close_h)) and (close_h >= entry_price)
    return (False, exit_success, horizon_t)

def evaluate_path_short(rth_df_ct: pd.DataFrame, entry_time: datetime, entry_price: float,
                        lookahead_blocks: int, risk_points: float, rr_target: float) -> Tuple[bool, bool, Optional[datetime]]:
    """
    Short logic (Skyline overnight-drop):
      - entry_success: did Low reach entry - rr*risk before High hit entry + risk?
      - exit_success:  Close at horizon <= entry
    """
    if rth_df_ct.empty:
        return (False, False, None)
    target = entry_price - rr_target * risk_points
    stop   = entry_price + risk_points

    times = [t for t in rth_df_ct.index if t >= entry_time][:lookahead_blocks+1]
    if len(times) == 0:
        return (False, False, None)

    for t in times:
        row = rth_df_ct.loc[t]
        hi = float(row.get("High", np.nan))
        lo = float(row.get("Low", np.nan))
        if not np.isnan(lo) and lo <= target:
            if not np.isnan(hi) and hi >= stop:
                return (False, False, t)
            return (True, True, t)
        if not np.isnan(hi) and hi >= stop:
            return (False, False, t)

    horizon_t = times[-1]
    close_h = float(rth_df_ct.loc[horizon_t].get("Close", np.nan))
    exit_success = (not np.isnan(close_h)) and (close_h <= entry_price)
    return (False, exit_success, horizon_t)

def direction_move_prob(rth_df_ct: pd.DataFrame, entry_time: datetime, lookahead_blocks: int) -> Tuple[float, float]:
    """
    Direction probability based on close at horizon vs entry close.
    Returns (prob_up, prob_down) as 0/1 for a single path; caller aggregates.
    """
    idx = [t for t in rth_df_ct.index if t <= entry_time]
    if not idx:
        return (0.0, 0.0)
    entry_bar = idx[-1]
    entry_close = float(rth_df_ct.loc[entry_bar].get("Close", np.nan))
    future = [t for t in rth_df_ct.index if t > entry_time][:lookahead_blocks]
    if not future:
        return (0.0, 0.0)
    horizon_t = future[-1]
    horizon_close = float(rth_df_ct.loc[horizon_t].get("Close", np.nan))
    if np.isnan(entry_close) or np.isnan(horizon_close):
        return (0.0, 0.0)
    return (1.0, 0.0) if horizon_close >= entry_close else (0.0, 1.0)

# -------- Historical scanners for each hypothesis --------
def scan_baseline_touch_signals(lookback_days: int, k: int, es_to_spx_offset: float,
                                risk_points: float, lookahead_blocks: int, rr_target: float,
                                symbol_underlying: str = "ES=F",
                                touch_pad: float = 0.0) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    For the last N trading days, compute Baseline-touch long signals and outcomes.
    Returns (events_df, summary_probs)
    """
    events = []
    counted = 0
    dir_up = dir_down = 0.0
    succ_entry = succ_exit = 0.0

    # iterate projection days from yesterday backwards skipping weekends
    day = previous_weekday(now_ct().date())
    processed = 0
    while processed < lookback_days:
        proj_day = day
        prev_day = previous_weekday(proj_day - timedelta(days=1))

        # Anchors from Part 2 function
        df_es_win = fetch_es_30m_for_prev_day_window(prev_day)
        if not df_es_win.empty:
            (_, _), (lo_price, lo_time) = kth_extreme_by_close(df_es_win, k=k)
            if lo_price is not None and lo_time is not None:
                lo_price_adj = lo_price + es_to_spx_offset
                lo_time_ct   = _as_ct(lo_time)

                # RTH data for projection day
                rth = fetch_rth_30m_for_day(symbol_underlying, proj_day)
                if not rth.empty:
                    # Find first touch of Baseline projection (using baseline slope)
                    touch_t = first_touch_time(rth, lo_price_adj, lo_time_ct, st.session_state["spx_slopes"]["baseline"], touch_pad=touch_pad)
                    if touch_t is not None:
                        counted += 1
                        entry_price = line_value_at(lo_price_adj, lo_time_ct, st.session_state["spx_slopes"]["baseline"], touch_t)
                        e_succ, x_succ, t_res = evaluate_path_long(
                            rth, touch_t, entry_price, lookahead_blocks, risk_points, rr_target
                        )
                        up, down = direction_move_prob(rth, touch_t, lookahead_blocks)
                        dir_up  += up
                        dir_down += down
                        succ_entry += 1.0 if e_succ else 0.0
                        succ_exit  += 1.0 if x_succ else 0.0
                        events.append(dict(
                            projection_day=str(proj_day),
                            model="Baseline→Long",
                            k=k,
                            offset=es_to_spx_offset,
                            touch_time=str(touch_t),
                            entry_price=round(entry_price, 4),
                            entry_success=bool(e_succ),
                            exit_success=bool(x_succ),
                            resolved_time=str(t_res) if t_res else None,
                            dir_up=int(up == 1.0),
                            dir_down=int(down == 1.0),
                        ))

        # previous trading day
        day = previous_weekday(proj_day - timedelta(days=1))
        processed += 1

    if counted == 0:
        return pd.DataFrame(), dict(
            direction_prob_up=0.0,
            direction_prob_down=0.0,
            entry_success_prob=0.0,
            exit_success_prob=0.0,
            samples=0,
        )

    summary = dict(
        direction_prob_up=round(dir_up / counted, 4),
        direction_prob_down=round(dir_down / counted, 4),
        entry_success_prob=round(succ_entry / counted, 4),
        exit_success_prob=round(succ_exit / counted, 4),
        samples=int(counted),
    )
    return pd.DataFrame(events), summary

def scan_skyline_overnight_drop_signals(lookback_days: int, k: int, es_to_spx_offset: float,
                                        risk_points: float, lookahead_blocks: int, rr_target: float,
                                        symbol_underlying: str = "ES=F",
                                        touch_pad: float = 0.0) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    For the last N trading days, compute Skyline-touch + overnight drop short signals and outcomes.
    Trigger criterion: 08:30 CT open below Skyline line at 08:30.
    """
    events = []
    counted = 0
    dir_up = dir_down = 0.0
    succ_entry = succ_exit = 0.0

    day = previous_weekday(now_ct().date())
    processed = 0
    while processed < lookback_days:
        proj_day = day
        prev_day = previous_weekday(proj_day - timedelta(days=1))

        df_es_win = fetch_es_30m_for_prev_day_window(prev_day)
        if not df_es_win.empty:
            (hi_price, hi_time), _ = kth_extreme_by_close(df_es_win, k=k)
            if hi_price is not None and hi_time is not None:
                hi_price_adj = hi_price + es_to_spx_offset
                hi_time_ct   = _as_ct(hi_time)

                rth = fetch_rth_30m_for_day(symbol_underlying, proj_day)
                if not rth.empty:
                    # Check 08:30 open relative to Skyline line at 08:30
                    rth_830 = rth.loc[rth.index.time == time(8,30)]
                    if not rth_830.empty:
                        t0 = rth_830.index[0]
                        skyline_at_830 = line_value_at(hi_price_adj, hi_time_ct, st.session_state["spx_slopes"]["skyline"], t0)
                        open_830 = float(rth_830.iloc[0].get("Open", np.nan))
                        if not np.isnan(open_830) and open_830 < (skyline_at_830 - touch_pad):
                            # Trigger: overnight drop below skyline at open
                            counted += 1
                            entry_time = t0
                            entry_price = open_830  # conservative: use open as entry reference
                            e_succ, x_succ, t_res = evaluate_path_short(
                                rth, entry_time, entry_price, lookahead_blocks, risk_points, rr_target
                            )
                            up, down = direction_move_prob(rth, entry_time, lookahead_blocks)
                            dir_up  += up
                            dir_down += down
                            succ_entry += 1.0 if e_succ else 0.0
                            succ_exit  += 1.0 if x_succ else 0.0
                            events.append(dict(
                                projection_day=str(proj_day),
                                model="Skyline→Short (OVN drop)",
                                k=k,
                                offset=es_to_spx_offset,
                                open_time=str(entry_time),
                                skyline_at_open=round(skyline_at_830, 4),
                                entry_price=round(entry_price, 4),
                                entry_success=bool(e_succ),
                                exit_success=bool(x_succ),
                                resolved_time=str(t_res) if t_res else None,
                                dir_up=int(up == 1.0),
                                dir_down=int(down == 1.0),
                            ))

        day = previous_weekday(proj_day - timedelta(days=1))
        processed += 1

    if counted == 0:
        return pd.DataFrame(), dict(
            direction_prob_up=0.0,
            direction_prob_down=0.0,
            entry_success_prob=0.0,
            exit_success_prob=0.0,
            samples=0,
        )

    summary = dict(
        direction_prob_up=round(dir_up / counted, 4),
        direction_prob_down=round(dir_down / counted, 4),
        entry_success_prob=round(succ_entry / counted, 4),
        exit_success_prob=round(succ_exit / counted, 4),
        samples=int(counted),
    )
    return pd.DataFrame(events), summary

# -------- Signals & Probabilities Page (v2) --------
def page_signals_probabilities_v2():
    st.markdown("### Signals & Probabilities — Analytics Only")

    if not HAS_YF:
        st.warning("yfinance not available. Install before running: `pip install yfinance`.")
        return

    # Ensure defaults exist (backward compatible with Part 1)
    st.session_state.setdefault("analytics", {})
    analytics = st.session_state["analytics"]
    analytics.setdefault("lookahead_blocks", 4)
    analytics.setdefault("rr_target", 1.0)
    analytics.setdefault("stop_blocks", 2)
    analytics.setdefault("use_close_only", True)
    analytics.setdefault("risk_points", 5.0)  # new: base risk unit in points for target/stop

    cA, cB, cC, cD = st.columns([1.1, 1, 1, 1.2])
    with cA:
        lookback_days = st.number_input("Lookback trading days", min_value=5, max_value=60, value=40, step=5,
                                        help="Yahoo intraday supports ~60d. 30–40d is a good start.")
    with cB:
        k = st.selectbox("Swing selectivity (k)", [1, 2, 3], index=0)
    with cC:
        es_to_spx_offset = st.number_input("ES→SPX offset (points)", value=0.0, step=0.5)
    with cD:
        risk_points = st.number_input("Risk unit (points)", value=float(analytics["risk_points"]), step=0.5,
                                      help="Used to compute target (RR×risk) and stop (risk).")

    cE, cF = st.columns(2)
    with cE:
        lookahead_blocks = st.number_input("Horizon (30-min blocks)", min_value=1, max_value=16,
                                           value=int(analytics["lookahead_blocks"]), step=1)
    with cF:
        rr_target = st.number_input("Reward:Risk target", min_value=0.1, max_value=5.0,
                                    value=float(analytics["rr_target"]), step=0.1)

    st.session_state["analytics"].update({
        "lookahead_blocks": int(lookahead_blocks),
        "rr_target": float(rr_target),
        "risk_points": float(risk_points),
    })

    st.markdown("#### H1: Baseline-touch → Long (Call)")
    with st.spinner("Scanning history for Baseline-touch long signals..."):
        df1, s1 = scan_baseline_touch_signals(
            lookback_days=int(lookback_days),
            k=int(k),
            es_to_spx_offset=float(es_to_spx_offset),
            risk_points=float(risk_points),
            lookahead_blocks=int(lookahead_blocks),
            rr_target=float(rr_target),
            symbol_underlying="ES=F",
            touch_pad=0.0,
        )
    if df1.empty:
        st.info("No Baseline-touch signals found in the selected lookback.")
    else:
        c1, c2, c3, c4, c5 = st.columns(5)
        with c1: st.metric("Samples", s1["samples"])
        with c2: st.metric("Dir ↑ prob", f'{s1["direction_prob_up"]:.2%}')
        with c3: st.metric("Dir ↓ prob", f'{s1["direction_prob_down"]:.2%}')
        with c4: st.metric("Entry success", f'{s1["entry_success_prob"]:.2%}')
        with c5: st.metric("Exit success", f'{s1["exit_success_prob"]:.2%}')
        st.dataframe(df1.tail(20), use_container_width=True, hide_index=True)

    st.markdown("#### H2: Skyline-touch + Overnight Drop → Short (Put)")
    with st.spinner("Scanning history for Skyline-overnight-drop short signals..."):
        df2, s2 = scan_skyline_overnight_drop_signals(
            lookback_days=int(lookback_days),
            k=int(k),
            es_to_spx_offset=float(es_to_spx_offset),
            risk_points=float(risk_points),
            lookahead_blocks=int(lookahead_blocks),
            rr_target=float(rr_target),
            symbol_underlying="ES=F",
            touch_pad=0.0,
        )
    if df2.empty:
        st.info("No Skyline overnight-drop signals found in the selected lookback.")
    else:
        c1, c2, c3, c4, c5 = st.columns(5)
        with c1: st.metric("Samples", s2["samples"])
        with c2: st.metric("Dir ↑ prob", f'{s2["direction_prob_up"]:.2%}')
        with c3: st.metric("Dir ↓ prob", f'{s2["direction_prob_down"]:.2%}')
        with c4: st.metric("Entry success", f'{s2["entry_success_prob"]:.2%}')
        with c5: st.metric("Exit success", f'{s2["exit_success_prob"]:.2%}')
        st.dataframe(df2.tail(20), use_container_width=True, hide_index=True)

    st.caption("Notes: Analytics only (no simulation). Results are historical frequencies under the stated rules.")

# Upgrade router entry (append-only)
PAGES["Signals • Probabilities"] = page_signals_probabilities_v2


# ============================================================
# MarketLens Pro v5 — PART 5/?
# Contract Tool wiring:
# - Strategy selection (Baseline→Call or Skyline OVN Drop→Put)
# - Detect entry time in RTH (08:30–14:30 CT)
# - Compute underlying Entry / Stop / Target (RR-based)
# - Map to contract prices via elasticity from an overnight reference
# - Show entries table + probability overlay (from Part 4 scanners)
# ============================================================

# -------- Helper: map underlying delta -> contract delta --------
def map_contract_from_anchor(anchor_under: float, anchor_contract: float,
                             entry_under: float, elasticity: float) -> float:
    """
    Linear advisory mapping (no greeks/IV):
      contract_entry ≈ anchor_contract + elasticity * (entry_under - anchor_under)
    elasticity: $ contract change per 1 point of underlying.
    """
    return float(anchor_contract + elasticity * (entry_under - anchor_under))

def contract_levels_from_entry(entry_contract: float, risk_points: float, rr_target: float,
                               side: str, elasticity: float, slippage: float = 0.0) -> Tuple[float, float]:
    """
    Compute (stop_contract, target_contract) from contract entry using the same underlying R:R geometry,
    but scaled with elasticity (advisory). Slippage adjusts pessimistically.
    """
    if side == "long":
        stop_c   = entry_contract - elasticity * risk_points + slippage
        target_c = entry_contract + elasticity * (rr_target * risk_points) - slippage
    else:  # short
        stop_c   = entry_contract + elasticity * risk_points + slippage
        target_c = entry_contract - elasticity * (rr_target * risk_points) - slippage
    return float(round(stop_c, 4)), float(round(target_c, 4))

# -------- Entry detectors (RTH) --------
def detect_entry_baseline_long(rth_df_ct: pd.DataFrame,
                               lo_anchor_price: float, lo_anchor_time_ct: datetime,
                               baseline_slope: float, touch_pad: float = 0.0) -> Optional[Tuple[datetime, float]]:
    """
    Return (entry_time, entry_underlying_price) at first RTH touch of the Baseline projection.
    """
    t = first_touch_time(rth_df_ct, lo_anchor_price, lo_anchor_time_ct, baseline_slope, touch_pad=touch_pad)
    if t is None:
        return None
    price = line_value_at(lo_anchor_price, lo_anchor_time_ct, baseline_slope, t)
    return (t, float(round(price, 4)))

def detect_entry_skyline_short_overnight_drop(rth_df_ct: pd.DataFrame,
                                              hi_anchor_price: float, hi_anchor_time_ct: datetime,
                                              skyline_slope: float, touch_pad: float = 0.0) -> Optional[Tuple[datetime, float]]:
    """
    Trigger: 08:30 CT open < Skyline line at 08:30 (overnight drop).
    Entry reference is 08:30 open.
    """
    rth_830 = rth_df_ct.loc[rth_df_ct.index.time == time(8, 30)]
    if rth_830.empty:
        return None
    t0 = rth_830.index[0]
    skyline_at_830 = line_value_at(hi_anchor_price, hi_anchor_time_ct, skyline_slope, t0)
    open_830 = float(rth_830.iloc[0].get("Open", np.nan))
    if np.isnan(open_830):
        return None
    if open_830 < (skyline_at_830 - touch_pad):
        return (t0, float(round(open_830, 4)))
    return None

# -------- Page: Contract Tool v2 --------
def page_contract_tool_v2():
    st.markdown("### Contract Tool — Entries • Stops • Targets • Probabilities (Advisory Only)")

    if not HAS_YF:
        st.warning("yfinance not available. Install before running the full app: `pip install yfinance`.")
        return

    st.session_state.setdefault("analytics", {})
    st.session_state["analytics"].setdefault("lookahead_blocks", 4)
    st.session_state["analytics"].setdefault("rr_target", 1.0)
    st.session_state["analytics"].setdefault("risk_points", 5.0)

    # ---- Strategy + config ----
    colS1, colS2, colS3, colS4 = st.columns([1.1, 1, 1, 1.2])
    with colS1:
        strat = st.selectbox(
            "Strategy",
            ["Baseline touch → Call (Long)", "Skyline overnight drop → Put (Short)"],
            index=0
        )
    with colS2:
        k = st.selectbox("Swing selectivity (k)", [1, 2, 3], index=0)
    with colS3:
        es_to_spx_offset = st.number_input("ES→SPX offset (points)", value=0.0, step=0.5)
    with colS4:
        lookback_days = st.number_input("Lookback days (for probabilities)", min_value=5, max_value=60, value=40, step=5)

    colP1, colP2, colP3, colP4 = st.columns(4)
    with colP1:
        risk_points = st.number_input("Risk (points)", value=float(st.session_state["analytics"]["risk_points"]), step=0.5)
    with colP2:
        rr_target = st.number_input("Reward:Risk", value=float(st.session_state["analytics"]["rr_target"]), min_value=0.1, max_value=5.0, step=0.1)
    with colP3:
        elasticity = st.number_input("Elasticity ($ per 1 pt underlying)", value=1.00, min_value=0.0, step=0.05,
                                     help="Advisory mapping from underlying points to contract $ change.")
    with colP4:
        slippage = st.number_input("Slippage ($)", value=0.00, min_value=0.0, step=0.05)

    # ---- Anchors / days ----
    # Prefer previously computed anchors from Skyline/Baseline pages; otherwise compute on-the-fly.
    default_prev = previous_weekday(now_ct().date() - timedelta(days=1))
    default_proj = previous_weekday(now_ct().date())  # yesterday by default

    have_anchors = "spx_anchors" in st.session_state
    with st.expander("Anchors & Days", expanded=not have_anchors):
        colA, colB = st.columns(2)
        with colA:
            prev_day = st.date_input("Previous trading day (for ES window)", value=default_prev,
                                     help="17:00–19:30 CT window to pick k-th highs/lows from ES=F.")
        with colB:
            projection_day = st.date_input("Projection day (RTH 08:30–14:30 CT)", value=default_proj)

        # If anchors missing or user wants to refresh, compute them here
        refresh = st.checkbox("Compute/refresh anchors for these days", value=not have_anchors)
        if refresh:
            df_es_win = fetch_es_30m_for_prev_day_window(prev_day)
            if df_es_win.empty:
                st.error("No ES data for the selected previous day.")
                return
            (hi_price, hi_time), (lo_price, lo_time) = kth_extreme_by_close(df_es_win, k=int(k))
            if hi_price is None or lo_price is None:
                st.error("Could not compute k-th extremes from ES window.")
                return
            st.session_state["spx_anchors"] = {
                "previous_day": prev_day,
                "projection_day": projection_day,
                "skyline": {"price": float(hi_price) + es_to_spx_offset, "time": _as_ct(hi_time)},
                "baseline": {"price": float(lo_price) + es_to_spx_offset, "time": _as_ct(lo_time)},
                "k": int(k),
                "offset": float(es_to_spx_offset),
            }
        else:
            # fall back to existing
            if not have_anchors:
                st.warning("No anchors in session; check 'Compute/refresh anchors'.")
                return
            prev_day      = st.session_state["spx_anchors"]["previous_day"]
            projection_day= st.session_state["spx_anchors"]["projection_day"]

    # ---- Contract price inputs (overnight references at anchor) ----
    colC1, colC2 = st.columns(2)
    with colC1:
        call_ref = st.number_input(
            "Overnight Call Contract Price at Baseline Swing Low (anchor)",
            value=st.session_state["contracts"].get("overnight_call_entry_price") or 0.0,
            min_value=0.0, step=0.05, format="%.2f",
            help="Used to map to a call entry price at RTH baseline-touch."
        )
    with colC2:
        put_ref = st.number_input(
            "Overnight Put Contract Price at Skyline Touch (anchor)",
            value=st.session_state["contracts"].get("overnight_put_entry_price") or 0.0,
            min_value=0.0, step=0.05, format="%.2f",
            help="Used to map to a put entry price at RTH overnight-drop case."
        )
    st.session_state["contracts"]["overnight_call_entry_price"] = float(call_ref)
    st.session_state["contracts"]["overnight_put_entry_price"]  = float(put_ref)

    # ---- Fetch RTH for projection day ----
    rth = fetch_rth_30m_for_day("ES=F", projection_day)
    if rth.empty:
        st.error("No RTH data fetched for the projection day.")
        return

    # ---- Build entries table depending on strategy ----
    cols = ["Time_CT","RefLine","Side","Entry_Under","Stop_Under","Target_Under",
            "Entry_Contract","Stop_Contract","Target_Contract","RR"]
    rows = []

    # Probabilities overlay (aggregated from history by the matching hypothesis)
    if "Baseline touch" in strat:
        # Scan historical Baseline-touch long signals
        df_hist, summary = scan_baseline_touch_signals(
            lookback_days=int(lookback_days),
            k=int(k),
            es_to_spx_offset=float(es_to_spx_offset),
            risk_points=float(risk_points),
            lookahead_blocks=int(st.session_state["analytics"]["lookahead_blocks"]),
            rr_target=float(rr_target),
            symbol_underlying="ES=F",
            touch_pad=0.0,
        )
        prob_pack = summary
    else:
        # Scan historical Skyline overnight-drop short signals
        df_hist, summary = scan_skyline_overnight_drop_signals(
            lookback_days=int(lookback_days),
            k=int(k),
            es_to_spx_offset=float(es_to_spx_offset),
            risk_points=float(risk_points),
            lookahead_blocks=int(st.session_state["analytics"]["lookahead_blocks"]),
            rr_target=float(rr_target),
            symbol_underlying="ES=F",
            touch_pad=0.0,
        )
        prob_pack = summary

    # Compute strategy specifics
    base = st.session_state["spx_anchors"]["baseline"]
    sky  = st.session_state["spx_anchors"]["skyline"]
    baseline_slope = st.session_state["spx_slopes"]["baseline"]
    skyline_slope  = st.session_state["spx_slopes"]["skyline"]

    if "Baseline touch" in strat:
        side = "long"
        anchor_price = float(base["price"])
        anchor_time  = _as_ct(base["time"])
        # Fill the time grid (reference line values)
        for t in rth.index:
            ref_line = line_value_at(anchor_price, anchor_time, baseline_slope, t)
            rows.append([t.strftime("%Y-%m-%d %H:%M"), round(ref_line,4), "CALL","","","","","","", f"{rr_target:.2f}"])

        # Detect entry
        det = detect_entry_baseline_long(rth, anchor_price, anchor_time, baseline_slope, touch_pad=0.0)
        if det is None:
            st.info("No Baseline touch detected during RTH. Table shows the Baseline reference line only.")
        else:
            entry_t, entry_under = det
            # Underlying stop/target
            stop_under   = entry_under - float(risk_points)
            target_under = entry_under + float(rr_target) * float(risk_points)
            # Contract mapping from anchor
            entry_contract = map_contract_from_anchor(
                anchor_under=anchor_price,
                anchor_contract=float(call_ref),
                entry_under=entry_under,
                elasticity=float(elasticity),
            )
            stop_c, target_c = contract_levels_from_entry(
                entry_contract=float(entry_contract),
                risk_points=float(risk_points),
                rr_target=float(rr_target),
                side="long",
                elasticity=float(elasticity),
                slippage=float(slippage),
            )
            # Patch the row at entry time
            for r in rows:
                if r[0] == entry_t.strftime("%Y-%m-%d %H:%M"):
                    r[3] = round(entry_under,4)
                    r[4] = round(stop_under,4)
                    r[5] = round(target_under,4)
                    r[6] = round(entry_contract,4)
                    r[7] = stop_c
                    r[8] = target_c
                    break

    else:
        side = "short"
        anchor_price = float(sky["price"])
        anchor_time  = _as_ct(sky["time"])
        # Time grid (reference Skyline line)
        for t in rth.index:
            ref_line = line_value_at(anchor_price, anchor_time, skyline_slope, t)
            rows.append([t.strftime("%Y-%m-%d %H:%M"), round(ref_line,4), "PUT","","","","","","", f"{rr_target:.2f}"])

        det = detect_entry_skyline_short_overnight_drop(rth, anchor_price, anchor_time, skyline_slope, touch_pad=0.0)
        if det is None:
            st.info("No overnight drop below Skyline at 08:30 detected. Table shows Skyline reference line only.")
        else:
            entry_t, entry_under = det
            stop_under   = entry_under + float(risk_points)
            target_under = entry_under - float(rr_target) * float(risk_points)
            entry_contract = map_contract_from_anchor(
                anchor_under=anchor_price,
                anchor_contract=float(put_ref),
                entry_under=entry_under,
                elasticity=float(elasticity),
            )
            stop_c, target_c = contract_levels_from_entry(
                entry_contract=float(entry_contract),
                risk_points=float(risk_points),
                rr_target=float(rr_target),
                side="short",
                elasticity=float(elasticity),
                slippage=float(slippage),
            )
            for r in rows:
                if r[0] == entry_t.strftime("%Y-%m-%d %H:%M"):
                    r[3] = round(entry_under,4)
                    r[4] = round(stop_under,4)
                    r[5] = round(target_under,4)
                    r[6] = round(entry_contract,4)
                    r[7] = stop_c
                    r[8] = target_c
                    break

    df_entries = pd.DataFrame(rows, columns=cols)
    st.markdown("#### Entries Table (08:30–14:30 CT)")
    st.dataframe(df_entries, use_container_width=True, hide_index=True)
    st.download_button(
        "📥 Download Entries CSV",
        data=df_entries.to_csv(index=False),
        file_name=f"Entries_{('CALL' if side=='long' else 'PUT')}_{st.session_state['spx_anchors']['projection_day']}.csv",
        mime="text/csv",
        key="dl_entries_csv_p5",
    )

    # Probability overlay summary
    if prob_pack["samples"] == 0:
        st.info("No historical samples in the chosen lookback to compute probabilities.")
    else:
        st.markdown("#### Probability Overlay (historical frequencies)")
        c1, c2, c3, c4, c5 = st.columns(5)
        with c1: st.metric("Samples", prob_pack["samples"])
        with c2: st.metric("Dir ↑ prob", f'{prob_pack["direction_prob_up"]:.2%}')
        with c3: st.metric("Dir ↓ prob", f'{prob_pack["direction_prob_down"]:.2%}')
        with c4: st.metric("Entry success", f'{prob_pack["entry_success_prob"]:.2%}')
        with c5: st.metric("Exit success", f'{prob_pack["exit_success_prob"]:.2%}')
        st.caption("Advisory analytics only — not trading advice. No simulations used; pure historical frequencies under stated rules.")

# Upgrade router entry (append-only)
PAGES["Contract • Tool"] = page_contract_tool_v2





# ============================================================
# MarketLens Pro v5 — PART 6/?
# Stocks • Skyline / Baseline (per-ticker)
# - Fetch each equity's 30m extended-hours prev-day window (17:00–19:30 CT)
# - k-th extremes for anchors (per ticker)
# - Projection tables 08:30–14:30 CT using STOCK_SLOPES magnitudes:
#     Skyline slope = +magnitude, Baseline slope = -magnitude
# - Detect first RTH entry (touch) and compute Stop/Target (advisory)
# - Optional probability overlay using ES-anchored scanners
# ============================================================

# ---------- yfinance (extended-hours) ----------
def _yf_download_ext(symbol: str, start_dt_utc: datetime, end_dt_utc: datetime,
                     interval: str = "30m", prepost: bool = True) -> pd.DataFrame:
    if not HAS_YF:
        return pd.DataFrame()
    df = yf.download(
        symbol,
        interval=interval,
        start=start_dt_utc.replace(tzinfo=None),
        end=end_dt_utc.replace(tzinfo=None),
        progress=False,
        auto_adjust=True,
        prepost=prepost,
        threads=True,
    )
    if df is None or df.empty:
        return pd.DataFrame()
    df.index = pd.to_datetime(df.index)
    return ensure_dtindex_utc(df)

def fetch_equity_30m_for_prev_day_window(symbol: str, prev_day: date) -> pd.DataFrame:
    """
    For equities, use extended-hours to capture 17:00–19:30 CT.
    """
    start_ct = datetime.combine(prev_day, time(0, 0), tzinfo=CT)
    end_ct   = datetime.combine(prev_day + timedelta(days=1), time(23, 59), tzinfo=CT)
    df = _yf_download_ext(symbol, to_utc(start_ct), to_utc(end_ct), interval="30m", prepost=True)
    if df.empty:
        return df
    df_ct = df.tz_convert(CT)
    win_start = datetime.combine(prev_day, time(17, 0), tzinfo=CT)
    win_end   = datetime.combine(prev_day, time(19, 30), tzinfo=CT)
    df_ct = df_ct.loc[(df_ct.index >= win_start) & (df_ct.index <= win_end)].copy()
    cols = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df_ct.columns]
    return df_ct[cols] if cols else df_ct

# ---------- Shared helpers for stocks ----------
def _stock_slopes(symbol: str) -> Tuple[float, float]:
    """
    Returns (skyline_slope, baseline_slope) for a given stock symbol
    using your STOCK_SLOPES magnitude with +/- sign.
    """
    mag = float(STOCK_SLOPES.get(symbol, 0.0))
    return (mag, -mag)

def _fill_entries_rows(rth_df_ct: pd.DataFrame, refline_fn, side_label: str,
                       rr_target: float, risk_points: float) -> List[List]:
    rows: List[List] = []
    for t in rth_df_ct.index:
        ref_line = float(round(refline_fn(t), 4))
        rows.append([t.strftime("%Y-%m-%d %H:%M"), ref_line, side_label, "", "", "", "", "", "", f"{rr_target:.2f}"])
    return rows

# ---------- Stocks • Baseline (per-ticker) ----------
def page_stocks_baseline_v2():
    st.markdown("### Stocks • Baseline (per-ticker)")

    if not HAS_YF:
        st.warning("yfinance not available. Install before running: `pip install yfinance`.")
        return

    st.session_state.setdefault("analytics", {})
    lookahead_blocks = int(st.session_state["analytics"].get("lookahead_blocks", 4))
    rr_target = float(st.session_state["analytics"].get("rr_target", 1.0))
    risk_points = float(st.session_state["analytics"].get("risk_points", 1.0))

    cTop = st.columns([1.2, 1, 1, 1.2])
    with cTop[0]:
        symbols = st.multiselect("Tickers", CORE_SYMBOLS, default=CORE_SYMBOLS[:4])
    with cTop[1]:
        k = st.selectbox("Swing selectivity (k)", [1, 2, 3], index=0)
    with cTop[2]:
        prev_day = st.date_input("Previous trading day (for anchor window)", value=previous_weekday(now_ct().date()))
    with cTop[3]:
        projection_day = st.date_input("Projection day", value=previous_weekday(now_ct().date()))

    cOpts = st.columns(3)
    with cOpts[0]:
        touch_pad = st.number_input("Touch tolerance (pts)", value=0.0, step=0.1)
    with cOpts[1]:
        risk_points = st.number_input("Risk (pts)", value=risk_points, step=0.5)
    with cOpts[2]:
        rr_target = st.number_input("Reward:Risk", value=rr_target, min_value=0.1, max_value=5.0, step=0.1)

    for sym in symbols:
        st.markdown(f"#### {sym} — Baseline")
        df_win = fetch_equity_30m_for_prev_day_window(sym, prev_day)
        if df_win.empty:
            st.info(f"No data for {sym} in 17:00–19:30 CT window on {prev_day}.")
            continue

        (hi_price, hi_time), (lo_price, lo_time) = kth_extreme_by_close(df_win, k=int(k))
        if lo_price is None or lo_time is None:
            st.info(f"Could not compute k-th LOW for {sym}.")
            continue

        # Slopes for this stock
        sky_slope, base_slope = _stock_slopes(sym)

        # Build projection
        lo_price_adj = float(lo_price)
        lo_time_ct   = _as_ct(lo_time)
        slots_ct     = generate_rth_times_ct(projection_day)
        proj_df      = build_projection_table(lo_price_adj, lo_time_ct, base_slope, slots_ct)

        # Fetch RTH and detect first touch for long
        rth = fetch_rth_30m_for_day(sym, projection_day)
        if rth.empty:
            st.info(f"No RTH data for {sym} on {projection_day}.")
            continue

        entry_det = detect_entry_baseline_long(rth, lo_price_adj, lo_time_ct, base_slope, touch_pad=touch_pad)
        # Pre-fill rows with ref line
        rows = _fill_entries_rows(rth, lambda t: line_value_at(lo_price_adj, lo_time_ct, base_slope, t), "CALL", rr_target, risk_points)

        if entry_det is not None:
            entry_t, entry_under = entry_det
            stop_under   = entry_under - float(risk_points)
            target_under = entry_under + float(rr_target) * float(risk_points)
            for r in rows:
                if r[0] == entry_t.strftime("%Y-%m-%d %H:%M"):
                    r[3] = round(entry_under, 4)
                    r[4] = round(stop_under, 4)
                    r[5] = round(target_under, 4)
                    break
        else:
            st.caption("No Baseline touch detected during RTH; table shows reference line only.")

        df_entries = pd.DataFrame(rows, columns=["Time_CT","RefLine","Side","Entry_Under","Stop_Under","Target_Under",
                                                 "Entry_Contract","Stop_Contract","Target_Contract","RR"])
        cL, cR = st.columns([1.1, 1])
        with cL:
            st.dataframe(df_entries, use_container_width=True, hide_index=True)
        with cR:
            st.download_button(
                "📥 Download Baseline CSV",
                data=df_entries.to_csv(index=False),
                file_name=f"{sym}_Baseline_{projection_day}.csv",
                mime="text/csv",
                key=f"dl_{sym}_baseline_csv_p6",
            )
            st.markdown(
                f"""
                <div class="ml-card">
                  <div class="muted">Anchor</div>
                  <div><b>LOW</b> {lo_price_adj:.4f} @ {lo_time_ct.strftime('%Y-%m-%d %H:%M CT')}</div>
                  <div class="muted" style="margin-top:6px;">Slope: {base_slope:+.4f} / 30m</div>
                </div>
                """, unsafe_allow_html=True
            )

        with st.expander(f"{sym} — Previous-day window (17:00–19:30 CT)", expanded=False):
            df_show = df_win.copy()
            df_show.index = df_show.index.tz_convert(CT)
            st.dataframe(df_show, use_container_width=True)

# ---------- Stocks • Skyline (per-ticker) ----------
def page_stocks_skyline_v2():
    st.markdown("### Stocks • Skyline (per-ticker)")

    if not HAS_YF:
        st.warning("yfinance not available. Install before running: `pip install yfinance`.")
        return

    st.session_state.setdefault("analytics", {})
    lookahead_blocks = int(st.session_state["analytics"].get("lookahead_blocks", 4))
    rr_target = float(st.session_state["analytics"].get("rr_target", 1.0))
    risk_points = float(st.session_state["analytics"].get("risk_points", 1.0))

    cTop = st.columns([1.2, 1, 1, 1.2])
    with cTop[0]:
        symbols = st.multiselect("Tickers", CORE_SYMBOLS, default=CORE_SYMBOLS[:4])
    with cTop[1]:
        k = st.selectbox("Swing selectivity (k)", [1, 2, 3], index=0)
    with cTop[2]:
        prev_day = st.date_input("Previous trading day (for anchor window)", value=previous_weekday(now_ct().date()))
    with cTop[3]:
        projection_day = st.date_input("Projection day", value=previous_weekday(now_ct().date()))

    cOpts = st.columns(3)
    with cOpts[0]:
        touch_pad = st.number_input("Touch tolerance (pts)", value=0.0, step=0.1)
    with cOpts[1]:
        risk_points = st.number_input("Risk (pts)", value=risk_points, step=0.5)
    with cOpts[2]:
        rr_target = st.number_input("Reward:Risk", value=rr_target, min_value=0.1, max_value=5.0, step=0.1)

    for sym in symbols:
        st.markdown(f"#### {sym} — Skyline")
        df_win = fetch_equity_30m_for_prev_day_window(sym, prev_day)
        if df_win.empty:
            st.info(f"No data for {sym} in 17:00–19:30 CT window on {prev_day}.")
            continue

        (hi_price, hi_time), (lo_price, lo_time) = kth_extreme_by_close(df_win, k=int(k))
        if hi_price is None or hi_time is None:
            st.info(f"Could not compute k-th HIGH for {sym}.")
            continue

        sky_slope, base_slope = _stock_slopes(sym)

        # Build projection
        hi_price_adj = float(hi_price)
        hi_time_ct   = _as_ct(hi_time)
        slots_ct     = generate_rth_times_ct(projection_day)
        proj_df      = build_projection_table(hi_price_adj, hi_time_ct, sky_slope, slots_ct)

        # Fetch RTH and detect overnight-drop short (08:30 open < skyline@08:30)
        rth = fetch_rth_30m_for_day(sym, projection_day)
        if rth.empty:
            st.info(f"No RTH data for {sym} on {projection_day}.")
            continue

        det = detect_entry_skyline_short_overnight_drop(rth, hi_price_adj, hi_time_ct, sky_slope, touch_pad=touch_pad)
        rows = _fill_entries_rows(rth, lambda t: line_value_at(hi_price_adj, hi_time_ct, sky_slope, t), "PUT", rr_target, risk_points)

        if det is not None:
            entry_t, entry_under = det
            stop_under   = entry_under + float(risk_points)
            target_under = entry_under - float(rr_target) * float(risk_points)
            for r in rows:
                if r[0] == entry_t.strftime("%Y-%m-%d %H:%M"):
                    r[3] = round(entry_under, 4)
                    r[4] = round(stop_under, 4)
                    r[5] = round(target_under, 4)
                    break
        else:
            st.caption("No 08:30 open below Skyline detected; table shows reference line only.")

        df_entries = pd.DataFrame(rows, columns=["Time_CT","RefLine","Side","Entry_Under","Stop_Under","Target_Under",
                                                 "Entry_Contract","Stop_Contract","Target_Contract","RR"])
        cL, cR = st.columns([1.1, 1])
        with cL:
            st.dataframe(df_entries, use_container_width=True, hide_index=True)
        with cR:
            st.download_button(
                "📥 Download Skyline CSV",
                data=df_entries.to_csv(index=False),
                file_name=f"{sym}_Skyline_{projection_day}.csv",
                mime="text/csv",
                key=f"dl_{sym}_skyline_csv_p6",
            )
            st.markdown(
                f"""
                <div class="ml-card">
                  <div class="muted">Anchor</div>
                  <div><b>HIGH</b> {hi_price_adj:.4f} @ {hi_time_ct.strftime('%Y-%m-%d %H:%M CT')}</div>
                  <div class="muted" style="margin-top:6px;">Slope: {sky_slope:+.4f} / 30m</div>
                </div>
                """, unsafe_allow_html=True
            )

        with st.expander(f"{sym} — Previous-day window (17:00–19:30 CT)", expanded=False):
            df_show = df_win.copy()
            df_show.index = df_show.index.tz_convert(CT)
            st.dataframe(df_show, use_container_width=True)

# ---------- Upgrade router entries ----------
PAGES["Stocks • Baseline"] = page_stocks_baseline_v2
PAGES["Stocks • Skyline"]  = page_stocks_skyline_v2




# ============================================================
# MarketLens Pro v5 — FINAL BOOT (call render_app exactly once)
# ============================================================
if "render_app" in globals():
    if not st.session_state.get("__MLP_BOOTED__", False):
        st.session_state["__MLP_BOOTED__"] = True
    render_app()
else:
    import streamlit as st  # safety in case top imports failed
    st.error("render_app() not found. Make sure Parts 1–6 are pasted above this block.")
