# ============================================================
# MarketLens Pro v5 — Part 1 of N (Analytics-First)
# Core bootstrap + strategy scaffolding for probabilities
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
APP_TAGLINE = "Analytics for SPX Entries (No Simulation. Advisory Only.)"
APP_VERSION = "5.1.0-P1A"

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

CORE_SYMBOLS = list(STOCK_SLOPES.keys())


# -----------------------------
# Strategy hypotheses (scaffold)
# -----------------------------
# H1: Calls tend to bounce when RTH price touches a prior Baseline anchor extended into RTH
HYPOTHESIS_CALL_ON_BASELINE_TOUCH = {
    "id": "H1_BASELINE_TOUCH_LONG",
    "title": "Baseline Touch → Long (Call)",
    "definition": (
        "If the RTH underlying price touches the Baseline anchor (drawn from the previous session window) "
        "the probability of an upward reaction increases; analyze historical frequency of bounce vs. fail."
    ),
}

# H2: Puts tend to work when price touches Skyline anchor then drops through the overnight into RTH
HYPOTHESIS_PUT_ON_SKYLINE_DROP = {
    "id": "H2_SKYLINE_OVN_DROP_SHORT",
    "title": "Skyline Touch + Overnight Drop → Short (Put)",
    "definition": (
        "If the underlying touches the Skyline anchor and subsequently trades below it through the overnight, "
        "opening below into RTH, the probability of downside continuation increases; analyze frequency of follow-through vs. reclaim."
    ),
}

# Probability buckets we will compute later (no simulation, purely analytics)
PROB_KEYS = [
    "direction_prob_up",         # P(move up over next N blocks) given trigger
    "direction_prob_down",       # P(move down over next N blocks) given trigger
    "entry_success_prob",        # P(entry reaches target before stop) given trigger
    "exit_success_prob",         # P(exit achieves desired RR or time exit) given trigger
]


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

    # Hidden model constants
    ss.setdefault("spx_slopes", SPX_SLOPES.copy())       # hidden
    ss.setdefault("stock_slopes", STOCK_SLOPES.copy())   # hidden

    # Analytics settings (RR targets, lookahead blocks, etc.) — used by probability engine later
    ss.setdefault("analytics", {
        "lookahead_blocks": 4,      # N x 30-min blocks to evaluate direction probability
        "rr_target": 1.0,           # reward:risk target for entry/exit success calc
        "stop_blocks": 2,           # default stop horizon in 30-min blocks for success calc
        "use_close_only": True,     # compute using close-to-close blocks
    })

    # Contract tool state (user can supply contract prices at swing lows/highs)
    ss.setdefault("contracts", {
        "overnight_call_entry_price": None,   # call contract price at swing low (user input)
        "overnight_put_entry_price": None,    # put contract price at skyline touch (user input)
    })

    ss.setdefault("strategy_hypotheses", {
        HYPOTHESIS_CALL_ON_BASELINE_TOUCH["id"]: HYPOTHESIS_CALL_ON_BASELINE_TOUCH,
        HYPOTHESIS_PUT_ON_SKYLINE_DROP["id"]: HYPOTHESIS_PUT_ON_SKYLINE_DROP,
    })

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
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            """
            <div class="ml-card">
              <div class="muted">Strategy Focus</div>
              <div style="font-size:20px; font-weight:700;">Analytics, not Simulation</div>
              <div class="muted" style="margin-top:6px;">
                Advisory-only probabilities for direction, entry success, and exit success.
              </div>
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
              <div class="muted" style="margin-top:6px;">Slopes kept out of UI</div>
            </div>
