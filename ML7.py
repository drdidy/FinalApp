# ============================================================
# MarketLens Pro v5 — Part 1 of N (Revised with your slopes)
# Core bootstrap: layout, theming, router, safe utilities
# ============================================================

# (optional) pip:
#   pip install streamlit pandas numpy yfinance pytz tzdata

from __future__ import annotations
import os
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import Dict, Any, Optional

import pandas as pd
import numpy as np

# Timezones (prefer zoneinfo; falls back to pytz if needed)
try:
    from zoneinfo import ZoneInfo
    CT = ZoneInfo("America/Chicago")
    UTC = ZoneInfo("UTC")
except Exception:
    import pytz  # type: ignore
    CT = pytz.timezone("America/Chicago")
    UTC = pytz.UTC

# Optional data providers
try:
    import yfinance as yf  # noqa: F401
    HAS_YF = True
except Exception:
    HAS_YF = False

import streamlit as st


# -----------------------------
# App metadata & configuration
# -----------------------------
APP_NAME = "MarketLens Pro v5"
APP_TAGLINE = "Professional SPX & Equities Analysis"
APP_VERSION = "5.0.0-part1r1"

st.set_page_config(
    page_title=f"{APP_NAME}",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)


# -----------------------------
# Utilities
# -----------------------------
def now_ct() -> datetime:
    """Current time in America/Chicago (handles DST)."""
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
    if getattr(dt, "tzinfo", None) is None:
        return dt.replace(tzinfo=UTC)
    try:
        return dt.astimezone(UTC)
    except Exception:
        return dt.astimezone(UTC)


def ensure_dtindex_utc(df: pd.DataFrame) -> pd.DataFrame:
    """Safely ensure index is a UTC DatetimeIndex."""
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("DataFrame index must be a DatetimeIndex")
    tz = getattr(df.index, "tz", None)
    if tz is None:
        df.index = df.index.tz_localize(UTC)
    elif tz != UTC:
        df.index = df.index.tz_convert(UTC)
    return df


def get_env_flag(name: str, default: bool = False) -> bool:
    v = str(os.environ.get(name, "")).strip().lower()
    if v in ("1", "true", "yes", "y", "on"):
        return True
    if v in ("0", "false", "no", "n", "off"):
        return False
    return default


# -----------------------------
# Hidden model constants (your latest)
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


# -----------------------------
# Session state initialization
# -----------------------------
DEFAULT_THEME = {
    "bg_gradient": "radial-gradient(1200px 800px at 10% -10%, rgba(255,255,255,0.06), transparent), "
                   "linear-gradient(135deg, #0e0f13 0%, #0a0b10 40%, #0b111b 100%)",
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
    if "theme" not in st.session_state:
        st.session_state["theme"] = DEFAULT_THEME.copy()
    # store your latest constants, hidden from UI
    if "spx_slopes" not in st.session_state:
        st.session_state["spx_slopes"] = SPX_SLOPES.copy()
    if "stock_slopes" not in st.session_state:
        st.session_state["stock_slopes"] = STOCK_SLOPES.copy()
    if "settings" not in st.session_state:
        st.session_state["settings"] = {
            "timezone": "America/Chicago",
            "debug": get_env_flag("MLPRO_DEBUG", False),
        }
    if "nav" not in st.session_state:
        st.session_state["nav"] = "Dashboard"


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

      /* Subtle moving stars (pure CSS) */
      @keyframes floatSparkles {{
        0% {{ background-position: 0px 0px, 0px 0px; }}
        100% {{ background-position: 1000px 1000px, -800px 600px; }}
      }}
      [data-testid="stAppViewContainer"]::before {{
        content: "";
        position: fixed;
        inset: 0;
        pointer-events: none;
        background-image:
          radial-gradient(2px 2px at 20% 30%, rgba(255,255,255,0.12), transparent 40%),
          radial-gradient(1px 1px at 70% 10%, rgba(255,255,255,0.10), transparent 45%);
        animation: floatSparkles 60s linear infinite;
      }}

      /* Global typography */
      html, body, [data-testid="stAppViewContainer"] * {{
        color: {text};
      }}

      /* Sidebar (future-proof selector) */
      section[data-testid="stSidebar"] > div {{
        background: {card};
        backdrop-filter: blur(22px);
        border-right: 1px solid {border};
      }}
      section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
      section[data-testid="stSidebar"] label,
      section[data-testid="stSidebar"] span {{
        color: {muted};
      }}

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

      /* Pills */
      .pill {{
        display: inline-block;
        padding: 2px 10px;
        border-radius: 999px;
        border: 1px solid {border};
        background: rgba(255,255,255,0.04);
        color: {muted};
        font-size: 12px;
      }}

      /* Headline */
      .brand-title {{
        font-weight: 700;
        letter-spacing: 0.3px;
        font-size: 20px;
      }}
      .muted {{
        color: {muted};
      }}

      /* Buttons (slight glow on hover) */
      .stButton > button {{
        border-radius: 12px;
        border: 1px solid {border};
        background: rgba(255,255,255,0.05);
        transition: box-shadow .2s ease, transform .2s ease, border-color .2s ease;
      }}
      .stButton > button:hover {{
        border-color: {accent};
        transform: translateY(-1px);
        box-shadow: 0 10px 30px rgba(0,0,0,0.25);
      }}

      /* Tables */
      .blank[data-testid="stTable"] table {{
        border-radius: 12px;
        overflow: hidden;
      }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


inject_css(st.session_state["theme"])


# -----------------------------
# Header / Top bar
# -----------------------------
def topbar() -> None:
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
              <div style="font-size:18px; font-weight:600">{'Debug' if st.session_state['settings']['debug'] else 'Standard'}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# -----------------------------
# Pages (stubbed for Part 1)
# -----------------------------
def page_dashboard():
    st.markdown("### Overview")
    n_consts = len(st.session_state["spx_slopes"]) + len(st.session_state["stock_slopes"])
    st.markdown(
        f"""
        <div class="ml-card">
          <div style="font-size:14px" class="muted">
            Part 1 (layout, theme, router, utilities). Subsequent parts will add data fetchers,
            SPX Skyline/Baseline, stock anchors, projections, and exports — while keeping model
            constants (slopes) hidden from the UI.
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
# Main render
# -----------------------------
topbar()
PAGES[st.session_state["nav"]]()
