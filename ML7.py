# ============================================================
# MarketLens Pro v5 — Part 1 of N (Analytics-First, FIXED)
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
HYPOTHESIS_CALL_ON_BASELINE_TOUCH = {
    "id": "H1_BASELINE_TOUCH_LONG",
    "title": "Baseline Touch → Long (Call)",
    "definition": (
        "If the RTH underlying price touches the Baseline anchor (drawn from the previous session window) "
        "the probability of an upward reaction increases; analyze historical frequency of bounce vs. fail."
    ),
}

HYPOTHESIS_PUT_ON_SKYLINE_DROP = {
    "id": "H2_SKYLINE_OVN_DROP_SHORT",
    "title": "Skyline Touch + Overnight Drop → Short (Put)",
    "definition": (
        "If the underlying touches the Skyline anchor and subsequently trades below it through the overnight, "
        "opening below into RTH, the probability of downside continuation increases; analyze frequency of follow-through vs. reclaim."
    ),
}

# Probability buckets we will compute later (analytics only)
PROB_KEYS = [
    "direction_prob_up",
    "direction_prob_down",
    "entry_success_prob",
    "exit_success_prob",
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

    # Analytics settings — used by probability engine later
    ss.setdefault("analytics", {
        "lookahead_blocks": 4,      # N x 30-min blocks to evaluate direction probability
        "rr_target": 1.0,           # reward:risk target for entry/exit success calc
        "stop_blocks": 2,           # default stop horizon in 30-min blocks for success calc
        "use_close_only": True,     # compute using close-to-close blocks
    })

    # Contract tool state (user supplies contract prices at swing lows/highs)
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
# Theming (CSS injection) — brace-safe (no f-strings)
# -----------------------------
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
            """,
            unsafe_allow_html=True,
        )
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
            """,
            unsafe_allow_html=True,
        )

    st.markdown("#### Roadmap")
    st.markdown(
        """
        <div class="ml-card">
          <ul style="margin:0 0 0 18px;">
            <li>Part 2–3: SPX Skyline/Baseline anchors & projections (k-select, ES→SPX offset).</li>
            <li>Part 4: Signals & Probabilities — analytics engine for direction/entry/exit probabilities (no simulation).</li>
            <li>Part 5: Contract Tool — user inputs contract prices at overnight swing low/high to evaluate entries at RTH.</li>
            <li>Part 6+: Per-ticker Stock anchors using your STOCK_SLOPES + exports.</li>
          </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )


def page_spx_skyline():
    st.markdown("### SPX Skyline (placeholder)")
    st.info("Will compute Skyline projections and support probability triggers in Part 2/4.")


def page_spx_baseline():
    st.markdown("### SPX Baseline (placeholder)")
    st.info("Will compute Baseline projections and support probability triggers in Part 2/4.")


def page_signals_probabilities():
    st.markdown("### Signals & Probabilities (placeholder)")
    st.caption("Direction/Entry/Exit probabilities — computed from historical analytics (no simulation).")
    st.markdown(
        """
        <div class="ml-card">
          <div class="muted">Planned metrics</div>
          <ul style="margin:0 0 0 18px;">
            <li>Direction probability up/down over N x 30-min blocks</li>
            <li>Entry success probability to target before stop (RR-based)</li>
            <li>Exit success probability (time-based or RR-based)</li>
          </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )


def page_contract_tool():
    st.markdown("### Contract Tool (placeholder)")
    st.caption("Input contract prices at swing lows/highs (overnight) to analyze RTH entries — advisory only.")
    c1, c2 = st.columns(2)
    with c1:
        val = st.number_input(
            "Overnight Call Contract Price at Swing Low",
            value=st.session_state["contracts"]["overnight_call_entry_price"] or 0.0,
            min_value=0.0, step=0.05, format="%.2f",
            help="Used with Baseline-touch analysis for RTH call entries."
        )
        st.session_state["contracts"]["overnight_call_entry_price"] = float(val)
    with c2:
        val2 = st.number_input(
            "Overnight Put Contract Price at Skyline Touch",
            value=st.session_state["contracts"]["overnight_put_entry_price"] or 0.0,
            min_value=0.0, step=0.05, format="%.2f",
            help="Used with Skyline-drop analysis for RTH put entries."
        )
        st.session_state["contracts"]["overnight_put_entry_price"] = float(val2)
    st.success("Saved. Analytics wiring will connect in Parts 4–5.")


def page_settings():
    st.markdown("### Settings")
    st.caption("Non-sensitive preferences only; model constants remain internal.")
    col1, col2, col3 = st.columns(3)
    with col1:
        la = st.number_input("Lookahead blocks (30-min each)", min_value=1, max_value=16,
                             value=st.session_state["analytics"]["lookahead_blocks"], step=1)
        st.session_state["analytics"]["lookahead_blocks"] = int(la)
    with col2:
        rr = st.number_input("Reward:Risk target", min_value=0.1, max_value=5.0,
                             value=float(st.session_state["analytics"]["rr_target"]), step=0.1)
        st.session_state["analytics"]["rr_target"] = float(rr)
    with col3:
        sb = st.number_input("Stop horizon (blocks)", min_value=1, max_value=16,
                             value=st.session_state["analytics"]["stop_blocks"], step=1)
        st.session_state["analytics"]["stop_blocks"] = int(sb)

    st.toggle("Use close-only computation", value=st.session_state["analytics"]["use_close_only"],
              key="use_close_only_toggle")
    st.session_state["analytics"]["use_close_only"] = bool(st.session_state["use_close_only_toggle"])

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
        • Mode: Analytics only (no simulation). Not a trading app.
        """
    )


# Router map (will be upgraded later in append-only Parts)
PAGES = {
    "Dashboard": page_dashboard,
    "SPX • Skyline": page_spx_skyline,
    "SPX • Baseline": page_spx_baseline,
    "Signals • Probabilities": page_signals_probabilities,
    "Contract • Tool": page_contract_tool,
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
