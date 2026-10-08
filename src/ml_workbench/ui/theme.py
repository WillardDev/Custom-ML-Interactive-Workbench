"""Theme tokens, global CSS and card/page-header helpers for the mockup design system."""

from __future__ import annotations

import contextlib
import re
from collections.abc import Iterator
from dataclasses import dataclass

import streamlit as st

from ml_workbench.tabs import TabInfo

_FONT_URL = "https://fonts.googleapis.com/css2?family=Figtree:wght@400;500;600;700&display=swap"

_USED_CARD_KEYS: set[str] = set()


@dataclass(frozen=True)
class _Tokens:
    bg: str
    sf: str
    ink: str
    mute: str
    line: str
    acc: str
    soft: str
    ok: str
    bad: str
    on: str


LIGHT = _Tokens(
    bg="#F5F6F8",
    sf="#FFFFFF",
    ink="#192033",
    mute="#667085",
    line="#E4E7EC",
    acc="#3347D9",
    soft="#E8EBFD",
    ok="#1A8F5C",
    bad="#C8412F",
    on="#FFFFFF",
)

DARK = _Tokens(
    bg="#0F1320",
    sf="#171C2D",
    ink="#E8EBF5",
    mute="#98A2B8",
    line="#262D44",
    acc="#8A98FF",
    soft="#232A52",
    ok="#46C28A",
    bad="#F07A67",
    on="#0F1320",
)


def _stylesheet(t: _Tokens) -> str:
    css = f"""
<style>
:root {{
  --bg: {t.bg}; --sf: {t.sf}; --ink: {t.ink}; --mute: {t.mute};
  --line: {t.line}; --acc: {t.acc}; --soft: {t.soft};
  --ok: {t.ok}; --bad: {t.bad}; --on: {t.on};
  --gdg-font-family: "Figtree", ui-sans-serif, system-ui, sans-serif;
  --gdg-text-dark: {t.ink};
  --gdg-text-medium: {t.mute};
  --gdg-border-color: {t.line};
  --gdg-rounding-radius: 8px;
}}
html, body {{ font-family: "Figtree", ui-sans-serif, system-ui, sans-serif; }}
[data-testid="stApp"] {{
  background: var(--bg);
  color: var(--ink);
  font-family: "Figtree", ui-sans-serif, system-ui, sans-serif;
}}
[data-testid="stMarkdownContainer"] h1,
[data-testid="stMarkdownContainer"] h2 {{
  font-size: 28px;
  font-weight: 700;
  letter-spacing: -0.02em;
  line-height: 1.2;
  color: var(--ink);
}}
[data-testid="stMarkdownContainer"] h3,
[data-testid="stMarkdownContainer"] h4 {{
  font-size: 14px;
  font-weight: 600;
  color: var(--ink);
  margin-bottom: 0.25rem;
}}
[data-testid="stApp"] p, [data-testid="stApp"] li {{ color: var(--ink); }}
[data-testid="stApp"] [data-testid="stCaptionContainer"] {{ color: var(--mute); }}

/* Sidebar rail */
[data-testid="stSidebar"] {{
  width: 268px !important;
  background: var(--sf);
  border-right: 1px solid var(--line);
}}
[data-testid="stSidebarContent"] {{ background: var(--sf); }}
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h1 {{
  font-size: 17px;
  display: flex;
  align-items: center;
  gap: 10px;
}}
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h1::before {{
  content: "";
  width: 22px;
  height: 22px;
  border-radius: 7px;
  background: var(--acc);
  flex: 0 0 auto;
}}
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] {{ color: var(--mute); }}
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] li {{ color: var(--mute); }}

/* Nav rail (workflow radio) */
[data-testid="stSidebar"] [data-testid="stRadio"] > div:first-child {{
  color: var(--mute);
  font-size: 11px;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.06em;
}}
[data-testid="stSidebar"] [data-testid="stRadioGroup"] {{
  position: relative;
  gap: 2px !important;
}}
[data-testid="stSidebar"] [data-testid="stRadioGroup"]::before {{
  content: "";
  position: absolute;
  left: 9px;
  top: 20px;
  bottom: 20px;
  width: 2px;
  background: var(--line);
}}
[data-testid="stSidebar"] [data-testid="stRadio"] [role="radio"] {{
  color: var(--mute);
  font-weight: 500;
  font-size: 14px;
  padding: 6px 8px;
  border-radius: 8px;
  align-items: center;
}}
[data-testid="stSidebar"] [data-testid="stRadio"] [role="radio"]:hover {{
  background: var(--soft) !important;
}}
[data-testid="stSidebar"] [data-testid="stRadio"] [role="radio"][aria-checked="true"] {{
  color: var(--ink);
  font-weight: 600;
}}
[data-testid="stSidebar"] [data-testid="stRadio"] [role="radio"]:disabled {{
  opacity: 0.55;
}}
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h2 {{ font-size: 12px; }}

/* Cards (st.container with key="card-...") */
[class*="st-key-card-"] {{
  background: var(--sf);
  border: 1px solid var(--line) !important;
  border-radius: 14px;
  padding: 20px;
}}
[class*="st-key-card-"] > div:first-child {{ padding-top: 0; }}

/* Buttons */
[data-testid="stBaseButton-primary"],
[data-testid="stFormSubmitButton"] {{
  background: var(--acc) !important;
  color: var(--on) !important;
  border: 1px solid var(--acc) !important;
  border-radius: 10px !important;
  font-weight: 600;
  transition: filter 0.15s ease;
}}
[data-testid="stBaseButton-primary"]:hover,
[data-testid="stFormSubmitButton"]:hover {{ filter: brightness(1.08); }}
[data-testid="stBaseButton-secondary"],
[data-testid="stBaseButton-secondaryGhost"] {{
  background: var(--sf) !important;
  color: var(--ink) !important;
  border: 1px solid var(--line) !important;
  border-radius: 10px !important;
  font-weight: 600;
}}
[data-testid="stBaseButton-secondary"]:hover {{ background: var(--soft) !important; }}
[data-testid="stDownloadButton"] {{
  border-radius: 10px !important;
  font-weight: 600;
}}

/* Inputs */
[data-testid="stTextInput"] input,
[data-testid="stNumberInput"] input,
[data-testid="stDateInput"] input,
[data-testid="stTimeInput"] input,
[data-testid="stColorPicker"] [data-testid="stColorPickerSwatch"],
[data-testid="stSelectbox"] [role="combobox"],
[data-testid="stSelectbox"] button,
[data-testid="stMultiSelect"] [data-testid="stTag"],
[data-testid="stSlider"] input[type="range"] {{
  border-radius: 8px;
}}
[data-testid="stTextInput"] input,
[data-testid="stNumberInput"] input,
[data-testid="stDateInput"] input,
[data-testid="stTimeInput"] input,
[data-testid="stSelectbox"] [role="combobox"],
[data-testid="stSelectbox"] button {{
  border: 1px solid var(--line) !important;
  background: var(--sf) !important;
  color: var(--ink) !important;
}}
[data-testid="stSlider"] [role="slider"] {{
  background: var(--acc) !important;
  border-color: var(--acc) !important;
}}
[data-testid="stFileUploader"] section {{
  border: 1.5px dashed var(--line) !important;
  border-radius: 12px !important;
  background: var(--sf) !important;
}}

/* Tables and dataframes */
[data-testid="stTable"] table {{
  border-collapse: collapse;
  width: 100%;
  font-size: 13px;
}}
[data-testid="stTable"] th {{
  text-align: left;
  font-weight: 600;
  color: var(--ink);
  border-bottom: 1px solid var(--line) !important;
  padding: 8px 10px;
}}
[data-testid="stTable"] td {{
  border-bottom: 1px solid var(--line) !important;
  padding: 8px 10px;
  color: var(--ink);
}}
[data-testid="stDataFrame"] table,
[data-testid="stDataFrame"] canvas {{ font-family: inherit; }}

/* Metric */
[data-testid="stMetric"] {{
  background: var(--sf);
  border: 1px solid var(--line);
  border-radius: 14px;
  padding: 16px 18px;
}}
[data-testid="stMetricValue"] {{
  font-size: 30px;
  font-weight: 700;
  color: var(--ink);
  line-height: 1.1;
}}
[data-testid="stMetricLabel"] {{ color: var(--mute); font-weight: 500; }}
[data-testid="stMetricDelta"] [data-testid="stMarkdownContainer"] p {{ font-weight: 600; }}

/* Progress */
[data-testid="stProgress"] > div {{
  background: var(--line);
  border-radius: 99px;
  height: 8px;
}}
[data-testid="stProgress"] [role="progressbar"] {{ background: var(--acc) !important; }}

/* Expander */
[data-testid="stExpander"] {{
  background: var(--sf);
  border: 1px solid var(--line) !important;
  border-radius: 14px;
}}
[data-testid="stExpander"] summary {{ font-weight: 600; }}

/* Alerts */
[data-testid="stAlert"] {{
  border: 1px solid var(--line) !important;
  border-radius: 12px !important;
  background: var(--soft) !important;
}}

/* Tabs */
[data-testid="stTabs"] [role="tablist"] {{ gap: 4px; }}
[data-testid="stTabs"] [role="tab"] {{
  border-radius: 10px 10px 0 0;
  font-weight: 600;
  color: var(--mute);
}}
[data-testid="stTabs"] [role="tab"][aria-selected="true"] {{ color: var(--acc); }}
[data-testid="stTabs"] [data-baseweb="tab-highlight"] {{ background-color: var(--acc) !important; }}

/* Misc */
[data-testid="stDivider"] hr {{ border-color: var(--line); }}
[data-testid="stInfo"] [data-testid="stAlertIcon"] {{ color: var(--acc); }}
[data-testid="stSuccess"] [data-testid="stAlertIcon"] {{ color: var(--ok); }}
[data-testid="stWarning"] [data-testid="stAlertIcon"] {{ color: var(--bad); }}
[data-testid="stError"] [data-testid="stAlertIcon"] {{ color: var(--bad); }}
[data-testid="stVerticalBlock"] > div:has(> [data-testid="stPlotlyChart"]),
[data-testid="stVerticalBlock"] > div:has(> [data-testid="stArrowVegaLiteChart"]) {{
  background: var(--sf);
  border: 1px solid var(--line);
  border-radius: 14px;
  padding: 12px;
}}
</style>
"""
    # markdown raw-HTML blocks end at blank lines; keep the sheet one block
    return "\n".join(line for line in css.splitlines() if line.strip())


def inject_theme() -> None:
    """Inject the active palette stylesheet. Call once at the top of main()."""
    dark = bool(st.session_state.get("dark_theme", False))
    tokens = DARK if dark else LIGHT
    markup = f'<link href="{_FONT_URL}" rel="stylesheet">{_stylesheet(tokens)}'
    st.html(markup)


def begin_run() -> None:
    """Reset per-run card key tracking. Call once at the top of main()."""
    _USED_CARD_KEYS.clear()


def _card_key(title: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9_-]+", "-", title).strip("-").lower() or "card"
    key = f"card-{slug}"
    n = 2
    while key in _USED_CARD_KEYS:
        key = f"card-{slug}-{n}"
        n += 1
    _USED_CARD_KEYS.add(key)
    return key


@contextlib.contextmanager
def card(title: str) -> Iterator[None]:
    """Render a surface card with an st.subheader title (kept for AppTest)."""
    with st.container(border=True, key=_card_key(title)):
        st.subheader(title)
        yield


def page_header(info: TabInfo) -> None:
    """Page chip row + dark toggle + numbered page header + subtitle."""
    left, right = st.columns([4, 1], vertical_alignment="bottom")
    with left:
        st.caption(f"Tab {info.number} of 10")
    with right:
        dark = bool(st.session_state.get("dark_theme", False))
        if st.button(
            "Dark" if not dark else "Light",
            key="theme_toggle",
            help="Toggle dark or light theme",
        ):
            st.session_state["dark_theme"] = not dark
            st.rerun()
    st.header(f"{info.number}. {info.title}")
    st.caption(info.subtitle)
