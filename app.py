"""
AdLens — Ad Intelligence for Meta & LinkedIn
=============================================

Analyzes a brand's live ads from the Meta Ad Library (Facebook, Instagram,
Messenger, Audience Network, Threads) and the LinkedIn Ad Library, and reports:

  * Primary hook           * Longest-running ad      * Engagement / reach signals
  * Offer type             * Platform split          * Dominant creative format

Data source
-----------
SearchApi.io (https://www.searchapi.io) — one API key covers:
  engine=meta_ad_library_page_search   -> resolve brand name to a Meta Page ID
  engine=meta_ad_library               -> the brand's Meta ads
  engine=linkedin_ad_library           -> the brand's LinkedIn ads
  engine=linkedin_ad_library_ad_details-> LinkedIn run dates + impressions (EU-delivered ads)

Optional: a Google Gemini API key adds AI hook/offer labelling and a strategy brief.
Without it the app uses its built-in rule-based classifiers.

Run
---
  pip install "streamlit>=1.50" "pandas>=2.0" "plotly>=5.20" requests google-genai
  streamlit run app.py

API keys: put them in .streamlit/secrets.toml (next to app.py) and they load
automatically — or type them into the sidebar, or set the environment variables
SEARCHAPI_API_KEY and GEMINI_API_KEY.
Tick "Demo mode" in the sidebar to try the full UI without any keys.
"""

from __future__ import annotations

import html
import inspect
import json
import os
import random
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import plotly.express as px
import requests
import streamlit as st

try:  # Optional dependency: only needed for AI analysis
    from google import genai  # type: ignore

    GENAI_AVAILABLE = True
except Exception:  # pragma: no cover
    genai = None
    GENAI_AVAILABLE = False


# ─────────────────────────────────────────────────────────────────────────────
# Page config & styling
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="AdLens · Ad Intelligence", page_icon="🎯", layout="wide")

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, .stApp, .stMarkdown p, .stMarkdown li, label, input, textarea, button p {
    font-family: 'Inter', system-ui, -apple-system, sans-serif;
}
.block-container { padding-top: 1.6rem; padding-bottom: 3rem; max-width: 1400px; }

/* Hero */
.hero {
    background: linear-gradient(120deg, #1E1B4B 0%, #312E81 45%, #0E7490 100%);
    border-radius: 18px; padding: 28px 32px; color: #fff; margin-bottom: 22px;
}
.hero h1 { color: #fff; font-size: 2.0rem; font-weight: 800; margin: 0 0 6px 0; letter-spacing: -0.5px; }
.hero p  { color: #C7D2FE; font-size: 1.0rem; margin: 0; }
.hero .pill {
    display: inline-block; background: rgba(255,255,255,0.14); color: #E0E7FF;
    border: 1px solid rgba(255,255,255,0.25); border-radius: 999px;
    padding: 3px 12px; font-size: 0.78rem; font-weight: 600; margin-top: 14px; margin-right: 6px;
}

/* KPI cards */
.kpi {
    background: #FFFFFF; border: 1px solid #E5E7EB; border-radius: 14px;
    padding: 16px 18px; height: 100%; box-shadow: 0 1px 2px rgba(16,24,40,0.04);
}
.kpi .label { color: #6B7280; font-size: 0.74rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.6px; }
.kpi .value { color: #111827; font-size: 1.25rem; font-weight: 800; margin-top: 4px; line-height: 1.2; }
.kpi .sub   { color: #6B7280; font-size: 0.80rem; margin-top: 4px; }

/* Insight cards */
.insight {
    background: #FFFFFF; border: 1px solid #E5E7EB; border-radius: 16px;
    padding: 20px 22px; margin-bottom: 16px; min-height: 210px;
    box-shadow: 0 1px 3px rgba(16,24,40,0.05); color: #111827;
}
.insight .head { display: flex; align-items: center; gap: 10px; margin-bottom: 10px; }
.insight .icon {
    width: 34px; height: 34px; border-radius: 10px; display: flex; align-items: center;
    justify-content: center; font-size: 1.05rem; background: #EEF2FF;
}
.insight .title { color: #4338CA; font-size: 0.78rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.7px; }
.insight .main  { color: #111827; font-size: 1.25rem; font-weight: 700; margin: 2px 0 8px 0; line-height: 1.3; }
.insight .quote {
    color: #374151; font-size: 0.92rem; border-left: 3px solid #6366F1; background: #F8FAFC;
    padding: 8px 12px; border-radius: 0 8px 8px 0; margin: 8px 0; font-style: italic;
}
.insight .detail { color: #4B5563; font-size: 0.86rem; line-height: 1.55; }
.insight .detail b { color: #111827; }

/* Bars inside insight cards */
.split-row { display: flex; align-items: center; gap: 10px; margin: 6px 0; font-size: 0.84rem; color: #374151; }
.split-row .name { width: 120px; flex-shrink: 0; }
.split-row .track { flex: 1; background: #F1F5F9; border-radius: 999px; height: 9px; overflow: hidden; }
.split-row .fill  { height: 9px; border-radius: 999px; }
.split-row .pct   { width: 46px; text-align: right; font-weight: 600; color: #111827; }

/* Badges */
.badge {
    display: inline-block; border-radius: 999px; padding: 2px 10px; font-size: 0.72rem;
    font-weight: 600; margin: 2px 4px 2px 0; border: 1px solid transparent;
}
.b-meta     { background: #EEF2FF; color: #4338CA; border-color: #C7D2FE; }
.b-linkedin { background: #E0F2FE; color: #0369A1; border-color: #BAE6FD; }
.b-neutral  { background: #F3F4F6; color: #374151; border-color: #E5E7EB; }
.b-green    { background: #ECFDF5; color: #047857; border-color: #A7F3D0; }
.b-amber    { background: #FFFBEB; color: #B45309; border-color: #FDE68A; }

/* Ad gallery cards */
.ad-card {
    background: #FFFFFF; border: 1px solid #E5E7EB; border-radius: 14px; overflow: hidden;
    margin-bottom: 18px; box-shadow: 0 1px 3px rgba(16,24,40,0.05); color: #111827;
}
.ad-card .thumb {
    width: 100%; height: 200px; object-fit: cover; background: #F1F5F9; display: block;
}
.ad-card .noimg {
    width: 100%; height: 200px; background: linear-gradient(135deg,#EEF2FF,#E0F2FE);
    display: flex; align-items: center; justify-content: center; color: #6366F1; font-weight: 700;
}
.ad-card .body { padding: 14px 16px; }
.ad-card .hook { font-weight: 700; font-size: 0.95rem; color: #111827; margin: 6px 0; line-height: 1.35; }
.ad-card .text { color: #4B5563; font-size: 0.84rem; line-height: 1.5; max-height: 6.2em; overflow: hidden; }
.ad-card .meta { color: #6B7280; font-size: 0.78rem; margin-top: 10px; }
.ad-card a { color: #4F46E5; font-weight: 600; text-decoration: none; }

.section-note { color: #6B7280; font-size: 0.86rem; margin: -4px 0 12px 0; }
.brief {
    background: #FFFFFF; border: 1px solid #E5E7EB; border-left: 4px solid #6366F1;
    border-radius: 12px; padding: 18px 22px; color: #111827;
}
div[data-testid="stSidebar"] .stButton button { width: 100%; font-weight: 700; }
</style>
""",
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
SEARCHAPI_URL = "https://www.searchapi.io/api/v1/search"
PALETTE = ["#6366F1", "#0EA5E9", "#10B981", "#F59E0B", "#EF4444", "#8B5CF6",
           "#EC4899", "#14B8A6", "#64748B", "#84CC16"]
PLATFORM_COLORS = {"Meta": "#6366F1", "LinkedIn": "#0EA5E9"}
GEMINI_MODELS = ["gemini-3.5-flash-lite", "gemini-3.8-flash", "gemini-3.7-flash", "gemini-3.6-flash"]

META_COUNTRIES = {
    "All countries": "ALL", "United States": "us", "India": "in", "United Kingdom": "gb",
    "Canada": "ca", "Australia": "au", "Germany": "de", "France": "fr", "Spain": "es",
    "Italy": "it", "Netherlands": "nl", "Brazil": "br", "Mexico": "mx",
    "United Arab Emirates": "ae", "Singapore": "sg", "Japan": "jp",
}
LINKEDIN_COUNTRIES = {k: ("" if v == "ALL" else v) for k, v in META_COUNTRIES.items()}
LINKEDIN_PERIODS = {
    "Last 12 months (all available)": "", "Last 30 days": "last_30_days",
    "This month": "this_month", "This year": "this_year",
}
META_PLACEMENTS = {
    "FACEBOOK": "Facebook", "INSTAGRAM": "Instagram", "MESSENGER": "Messenger",
    "AUDIENCE_NETWORK": "Audience Network", "THREADS": "Threads", "WHATSAPP": "WhatsApp",
}
META_FORMATS = {
    "VIDEO": "Video", "IMAGE": "Image", "CAROUSEL": "Carousel", "MULTI_IMAGES": "Carousel",
    "DPA": "Catalog (DPA)", "TEXT": "Text only", "EVENT": "Event", "MULTI_MEDIA": "Mixed media",
    "PAGE_LIKE": "Page like",
}
LINKEDIN_FORMATS = {
    "image": "Image", "video": "Video", "carousel": "Carousel", "document": "Document",
    "event": "Event", "text": "Text only", "spotlight": "Spotlight", "message": "Message",
    "conversation": "Conversation", "follower": "Follower", "follow_company": "Follower",
    "job": "Job", "article": "Article", "thought_leader": "Thought Leader",
}

HOOK_RULES = [
    ("Question", r"\?"),
    ("Pain point", r"\b(tired of|struggl\w*|sick of|frustrat\w*|hate|stop (wasting|guessing|losing)|"
                   r"no more|without the|problem|mistake|wasting|stressed|overwhelm\w*|can't|cannot)\b"),
    ("Statistic / Number", r"(\d+\s?%|\b\d+x\b|\b\d[\d,.]*\s?(k|m|million|billion|users|customers|"
                           r"people|companies|teams|brands|hours|minutes|days|steps|ways|reasons)\b)"),
    ("Social proof", r"\b(trusted by|loved by|join (over )?\d|rated|reviews?|testimonial|best-?selling|"
                     r"#1|award|customers say|as seen)\b"),
    ("Urgency / FOMO", r"\b(last chance|ends (today|tonight|soon|sunday|monday|friday)|today only|limited|"
                       r"hurry|don'?t miss|only \d+ left|final (hours|days)|deadline|expires|while stocks)\b"),
    ("Offer-led", r"(\d+\s?%\s?off|\bfree\b|\bsave\b|discount|\bsale\b|\bdeal\b|\$\d|₹\s?\d|€\s?\d|£\s?\d)"),
    ("Announcement / New", r"\b(introducing|meet the|new|launch\w*|now available|just dropped|announcing|"
                           r"coming soon|is here|arrived)\b"),
    ("Educational / How-to", r"\b(how to|guide|tips?|learn|ways to|secret|lessons?|playbook|what (is|are))\b"),
]
HOOK_TYPES = [h for h, _ in HOOK_RULES] + ["Benefit statement"]

OFFER_RULES = [
    ("Discount / Sale", r"(\d+\s?%\s?off|\bsale\b|discount|\bsave\s?(up to\s?)?[\$₹€£]?\d|coupon|promo code|"
                        r"\bdeals?\b|flat \d+|clearance|black friday|cyber monday)"),
    ("Free trial", r"(free trial|try (it )?(for )?free|\d+[- ]day (free )?trial|start (for )?free|"
                   r"free for \d+|freemium|no credit card)"),
    ("Free shipping", r"(free shipping|free delivery|ships free)"),
    ("Bundle / Gift", r"(\bbogo\b|buy one|bundle|free gift|gift with purchase|2 for 1|\d\s?\+\s?\d free)"),
    ("Demo / Consultation", r"(\bdemo\b|consultation|book a (call|meeting)|talk to (sales|an expert|us)|"
                            r"free (assessment|audit|quote)|get a quote|schedule a)"),
    ("Event / Webinar", r"(webinar|\bevent\b|summit|conference|register (now|today)|join us (live|on)|"
                        r"live session|masterclass|workshop)"),
    ("Lead magnet (content)", r"(e-?book|\bguide\b|whitepaper|white paper|\breport\b|checklist|template|"
                              r"playbook|download|toolkit|case study)"),
    ("Hiring / Employer brand", r"(we'?re hiring|join our team|careers|apply now|open roles|now hiring)"),
    ("New launch / Drop", r"(new collection|just dropped|introducing|now available|pre-?order|launch\w*|"
                          r"new arrivals?)"),
    ("Direct purchase", r"(shop now|buy now|order now|get offer|add to cart|shop the)"),
]
OFFER_TYPES = [o for o, _ in OFFER_RULES] + ["Brand awareness / Content"]


# ─────────────────────────────────────────────────────────────────────────────
# Small helpers
# ─────────────────────────────────────────────────────────────────────────────
def esc(value) -> str:
    return html.escape("" if value is None else str(value))


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _real_key(value) -> str:
    """Blank out empty values and the untouched template placeholder ("PASTE_YOUR_..._HERE")."""
    value = str(value or "").strip()
    return "" if value.upper().startswith("PASTE_") else value


def _secrets_paths() -> list:
    paths, seen = [], set()
    for p in (Path(__file__).resolve().parent / ".streamlit" / "secrets.toml",
              Path.cwd() / ".streamlit" / "secrets.toml",
              Path.home() / ".streamlit" / "secrets.toml"):
        key = str(p).lower()
        if key not in seen and p.is_file():
            seen.add(key)
            paths.append(p)
    return paths


_KEY_LINE = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*[=:]\s*(.*?)\s*$")
_QUOTES = "\"'\u201c\u201d\u2018\u2019\u00ab\u00bb`"


def _parse_secrets_text(text: str) -> dict:
    """Forgiving KEY = "value" reader. Copes with what Windows editors often produce:
    BOM, curly quotes, missing quotes, spaces, and non-UTF-8 comments."""
    try:
        import tomllib  # Python 3.11+
        data = tomllib.loads(text)
        return {k: v for k, v in data.items() if isinstance(v, str)}
    except Exception:
        pass
    out = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or line.startswith("["):
            continue
        m = _KEY_LINE.match(line)
        if not m:
            continue
        name, raw = m.group(1), m.group(2)
        if raw and raw[0] in _QUOTES:
            closing = max(raw.rfind(q) for q in _QUOTES)
            raw = raw[1:closing] if closing > 0 else raw[1:]
        else:
            raw = raw.split(" #")[0]
        out[name] = raw.strip().strip(_QUOTES).strip()
    return out


@st.cache_data(show_spinner=False)
def _load_local_secrets(stamp: tuple) -> dict:
    merged = {}
    for path_str, _mtime in reversed(stamp):  # project-level file wins over the home-level one
        raw = Path(path_str).read_bytes()
        for enc in ("utf-8-sig", "utf-16", "cp1252", "latin-1"):
            try:
                text = raw.decode(enc)
                if enc == "utf-16" and not raw[:2] in (b"\xff\xfe", b"\xfe\xff"):
                    continue
                break
            except UnicodeDecodeError:
                continue
        merged.update(_parse_secrets_text(text))
    return merged


def local_secrets() -> dict:
    paths = _secrets_paths()
    stamp = tuple((str(p), p.stat().st_mtime) for p in paths)  # re-read automatically when the file is saved
    try:
        return _load_local_secrets(stamp)
    except Exception:
        return {}


def get_secret(name: str) -> str:
    """Environment variable first, then .streamlit/secrets.toml, read by our own forgiving
    parser (st.secrets is never touched, so a malformed file can't break the app)."""
    return _real_key(os.environ.get(name)) or _real_key(local_secrets().get(name))


def to_num(value, default: float = 0.0) -> float:
    """Coerce API values like 12000, "12000", None or "" into a float."""
    try:
        if value is None or value == "" or isinstance(value, bool):
            return default
        return float(str(value).replace(",", ""))
    except (TypeError, ValueError):
        return default


def _full_width_kwargs(fn) -> dict:
    """Streamlit >= 1.50 stretches charts/tables by default (width="stretch");
    older versions need use_container_width=True. Works on either."""
    try:
        param = inspect.signature(fn).parameters.get("width")
        if param is not None and param.default == "stretch":
            return {}
    except (TypeError, ValueError):
        pass
    return {"use_container_width": True}


_DF_KW = _full_width_kwargs(st.dataframe)
_CHART_KW = _full_width_kwargs(st.plotly_chart)


def show_df(data, **kwargs):
    st.dataframe(data, **_DF_KW, **kwargs)


def clean_text(value) -> str:
    """Strip whitespace and dynamic-catalog placeholders like {{product.name}}."""
    if not isinstance(value, str):
        return ""
    text = re.sub(r"\{\{.*?\}\}", "", value).strip()
    return re.sub(r"[ \t]+", " ", text)


def parse_date(value):
    """Accepts ISO strings, YYYY-MM-DD, or epoch seconds. Returns aware datetime or None."""
    if value in (None, "", 0):
        return None
    try:
        if isinstance(value, (int, float)):
            return datetime.fromtimestamp(float(value), tz=timezone.utc)
        ts = pd.to_datetime(str(value), utc=True, errors="coerce")
        return None if pd.isna(ts) else ts.to_pydatetime()
    except Exception:
        return None


def fmt_int(n) -> str:
    try:
        n = float(n)
    except (TypeError, ValueError):
        return "—"
    if pd.isna(n):
        return "—"
    for div, suffix in ((1e9, "B"), (1e6, "M"), (1e3, "K")):
        if abs(n) >= div:
            return f"{n / div:.1f}{suffix}"
    return f"{int(n):,}"


def truncate(text: str, n: int) -> str:
    text = text or ""
    return text if len(text) <= n else text[: n - 1].rstrip() + "…"


def mode_or(series: pd.Series, default="—"):
    s = series.dropna()
    s = s[s.astype(str).str.len() > 0]
    return s.value_counts().idxmax() if not s.empty else default


# ─────────────────────────────────────────────────────────────────────────────
# SearchApi client
# ─────────────────────────────────────────────────────────────────────────────
class APIError(Exception):
    pass


def searchapi_request(params: dict, api_key: str, use_post: bool = False, timeout: int = 90) -> dict:
    headers = {"Authorization": f"Bearer {api_key}"}
    try:
        if use_post:  # SearchApi recommends POST when Meta pagination tokens get large
            resp = requests.post(SEARCHAPI_URL, json=params, headers=headers, timeout=timeout)
        else:
            resp = requests.get(SEARCHAPI_URL, params=params, headers=headers, timeout=timeout)
    except requests.RequestException as exc:
        raise APIError(f"Could not reach SearchApi: {exc}") from exc

    try:
        data = resp.json()
    except ValueError:
        raise APIError(f"SearchApi returned a non-JSON response (HTTP {resp.status_code}).")

    if resp.status_code == 401:
        raise APIError("SearchApi rejected the API key (HTTP 401). Check the key in the sidebar.")
    if resp.status_code == 429:
        raise APIError("SearchApi rate limit or monthly credits exhausted (HTTP 429).")
    if resp.status_code >= 400 or (isinstance(data, dict) and data.get("error")):
        msg = data.get("error") if isinstance(data, dict) else None
        raise APIError(f"SearchApi error (HTTP {resp.status_code}): {msg or 'unknown error'}")
    return data


@st.cache_data(ttl=3600, show_spinner=False)
def meta_page_search(brand: str, country: str, api_key: str) -> list:
    data = searchapi_request(
        {"engine": "meta_ad_library_page_search", "q": brand, "country": country}, api_key
    )
    return data.get("page_results") or []


def pick_best_page(pages: list, brand: str):
    if not pages:
        return None
    b = brand.casefold().strip()
    b_compact = re.sub(r"[^a-z0-9]", "", b)

    def score(p):
        name = str(p.get("name") or "").casefold().strip()
        alias = re.sub(r"[^a-z0-9]", "", str(p.get("page_alias") or "").casefold())
        exact = bool(name == b or (alias != "" and alias == b_compact))
        starts = bool(name.startswith(b))
        verified = str(p.get("verification") or "NOT_VERIFIED").upper() != "NOT_VERIFIED"
        # Every element is a bool or a float, so tuples always compare cleanly.
        return (exact, verified, starts, to_num(p.get("likes")), to_num(p.get("ig_followers")))

    return max(pages, key=score)


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_meta_ads(page_id: str, keyword: str, country: str, active_status: str,
                   max_ads: int, api_key: str):
    params = {"engine": "meta_ad_library", "country": country, "active_status": active_status}
    if page_id:
        params["page_id"] = page_id
    else:
        params["q"] = keyword

    ads, page_info, total, token = [], {}, None, None
    for page_no in range(25):
        req = dict(params)
        if token:
            req["next_page_token"] = token
        data = searchapi_request(req, api_key, use_post=bool(token))
        info = data.get("search_information") or {}
        if page_no == 0:
            page_info = info.get("ad_library_page_info") or {}
            total = info.get("total_results")
        batch = data.get("ads") or []
        ads.extend(batch)
        token = (data.get("pagination") or {}).get("next_page_token")
        if not batch or not token or len(ads) >= max_ads:
            break
    return ads[:max_ads], page_info, total


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_linkedin_ads(advertiser: str, country: str, time_period: str, max_ads: int, api_key: str):
    params = {"engine": "linkedin_ad_library", "advertiser": advertiser}
    if country:
        params["country"] = country
    if time_period:
        params["time_period"] = time_period

    ads, total, token = [], None, None
    for page_no in range(25):
        req = dict(params)
        if token:
            req["next_page_token"] = token
        data = searchapi_request(req, api_key)
        if page_no == 0:
            total = (data.get("search_information") or {}).get("total_results")
        batch = data.get("ads") or []
        ads.extend(batch)
        token = (data.get("pagination") or {}).get("next_page_token")
        if not batch or not token or len(ads) >= max_ads:
            break
    return ads[:max_ads], total


def _linkedin_detail(ad_id: str, api_key: str):
    try:
        data = searchapi_request(
            {"engine": "linkedin_ad_library_ad_details", "ad_id": ad_id}, api_key, timeout=60
        )
        return ad_id, data.get("ad") or {}
    except APIError:
        return ad_id, {}


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_linkedin_details(ad_ids: tuple, api_key: str) -> dict:
    if not ad_ids:
        return {}
    with ThreadPoolExecutor(max_workers=6) as pool:
        results = list(pool.map(lambda i: _linkedin_detail(i, api_key), ad_ids))
    return {ad_id: detail for ad_id, detail in results if detail}


# ─────────────────────────────────────────────────────────────────────────────
# Normalisation → one common schema for both platforms
# ─────────────────────────────────────────────────────────────────────────────
def normalize_meta_ad(ad: dict) -> dict:
    snap = ad.get("snapshot") or {}
    cards = snap.get("cards") or []
    videos = snap.get("videos") or []
    images = snap.get("images") or []

    body = snap.get("body")
    text = clean_text(body.get("text") if isinstance(body, dict) else body)
    if not text:
        text = next((clean_text(c.get("body")) for c in cards if clean_text(c.get("body"))), "")

    headline = clean_text(snap.get("title")) or next(
        (clean_text(c.get("title")) for c in cards if clean_text(c.get("title"))), "")
    cta = clean_text(snap.get("cta_text")) or next(
        (clean_text(c.get("cta_text")) for c in cards if clean_text(c.get("cta_text"))), "")
    link = snap.get("link_url") or next((c.get("link_url") for c in cards if c.get("link_url")), "")

    card_has_video = any(k.startswith("video") and c.get(k) for c in cards for k in c.keys())
    raw_fmt = str(snap.get("display_format") or "").upper()
    if raw_fmt == "DCO":  # dynamic creative: report the underlying media
        fmt = "Video" if (videos or card_has_video) else ("Carousel" if len(cards) > 1 else "Image")
    elif raw_fmt in META_FORMATS:
        fmt = META_FORMATS[raw_fmt]
    elif videos or card_has_video:
        fmt = "Video"
    elif len(cards) > 1:
        fmt = "Carousel"
    elif images or cards:
        fmt = "Image"
    else:
        fmt = "Text only"

    thumb = ""
    for v in videos:
        thumb = v.get("video_preview_image_url") or ""
        if thumb:
            break
    if not thumb:
        for i in images:
            thumb = i.get("resized_image_url") or i.get("original_image_url") or ""
            if thumb:
                break
    if not thumb:
        for c in cards:
            thumb = (c.get("resized_image_url") or c.get("original_image_url")
                     or c.get("video_preview_image_url") or "")
            if thumb:
                break

    start = parse_date(ad.get("start_date"))
    end = parse_date(ad.get("end_date"))
    is_active = bool(ad.get("is_active"))
    days = None
    if start:
        stop = now_utc() if (is_active or end is None) else end
        days = max(1, (stop - start).days + 1)

    imp_index = (ad.get("impressions_with_index") or {}).get("impressions_index")
    placements = [META_PLACEMENTS.get(str(p).upper(), str(p).title())
                  for p in (ad.get("publisher_platform") or [])]
    ad_id = str(ad.get("ad_archive_id") or "")

    return {
        "platform": "Meta",
        "ad_id": ad_id,
        "advertiser": snap.get("page_name") or ad.get("page_name") or "",
        "primary_text": text,
        "headline": headline,
        "cta": cta,
        "landing_url": link or "",
        "format": fmt,
        "placements": placements or ["Facebook"],
        "start_date": start,
        "end_date": None if is_active else end,
        "is_active": is_active,
        "days_running": days,
        "impressions_min": None,
        "impressions_max": None,
        "impressions_index": imp_index if isinstance(imp_index, (int, float)) and imp_index >= 0 else None,
        "variations": int(ad.get("collation_count") or 1),
        "page_likes": snap.get("page_like_count"),
        "thumbnail": thumb,
        "ad_url": f"https://www.facebook.com/ads/library/?id={ad_id}" if ad_id else "",
    }


def normalize_linkedin_ad(ad: dict, detail: dict | None = None) -> dict:
    detail = detail or {}
    content = dict(ad.get("content") or {})
    content.update({k: v for k, v in (detail.get("content") or {}).items() if v})

    ad_type = str(detail.get("ad_type") or ad.get("ad_type") or "").lower()
    if not ad_type and (ad.get("organizer") or ad.get("time")):
        ad_type = "event"
    fmt = LINKEDIN_FORMATS.get(ad_type, ad_type.replace("_", " ").title() or "Other")

    text = clean_text(content.get("headline") or ad.get("headline"))
    items = content.get("items") or []
    headline = clean_text(content.get("title") or content.get("cta") or ad.get("name")) or next(
        (clean_text(i.get("cta")) for i in items if clean_text(i.get("cta"))), "")
    cta = clean_text(content.get("call_to_action"))
    thumb = (content.get("image") or (items[0].get("image") if items else "")
             or ((content.get("pages") or [""])[0]) or ad.get("image") or "")

    advertiser = ((detail.get("advertiser") or ad.get("advertiser") or {}).get("name")
                  or ad.get("organizer") or "")
    first = parse_date(detail.get("first_shown_date"))
    last = parse_date(detail.get("last_shown_date"))
    days = max(1, (last - first).days + 1) if (first and last) else None
    is_active = bool(last and (now_utc() - last).days <= 3)

    ad_id = str(ad.get("id") or detail.get("id") or "")
    return {
        "platform": "LinkedIn",
        "ad_id": ad_id,
        "advertiser": advertiser,
        "primary_text": text,
        "headline": headline,
        "cta": cta,
        "landing_url": detail.get("external_link") or "",
        "format": fmt,
        "placements": ["LinkedIn"],
        "start_date": first,
        "end_date": last,
        "is_active": is_active if last else None,
        "days_running": days,
        "impressions_min": detail.get("total_impressions_min"),
        "impressions_max": detail.get("total_impressions_max"),
        "impressions_index": None,
        "variations": 1,
        "page_likes": None,
        "thumbnail": thumb or "",
        "ad_url": ad.get("link") or (f"https://www.linkedin.com/ad-library/detail/{ad_id}" if ad_id else ""),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Rule-based analysis
# ─────────────────────────────────────────────────────────────────────────────
def extract_hook(text: str, fallback: str = "") -> str:
    """The hook = the opening line / first sentence a viewer sees before 'See more'."""
    source = text or fallback or ""
    first_line = next((ln.strip() for ln in source.splitlines() if ln.strip()), "")
    if not first_line:
        return ""
    m = re.match(r"^(.+?[.!?…])(\s|$)", first_line)
    hook = m.group(1) if (m and len(m.group(1)) >= 15) else first_line
    return truncate(hook, 150)


def classify_hook(hook: str) -> str:
    h = (hook or "").lower()
    if not h:
        return "Benefit statement"
    for name, pattern in HOOK_RULES:
        if re.search(pattern, h):
            return name
    return "Benefit statement"


def classify_offer(text: str, headline: str, cta: str) -> str:
    blob = f"{text} {headline} {cta}".lower()
    for name, pattern in OFFER_RULES:
        if re.search(pattern, blob):
            return name
    return "Brand awareness / Content"


def build_dataframe(records: list) -> pd.DataFrame:
    cols = ["platform", "ad_id", "advertiser", "primary_text", "headline", "cta", "landing_url",
            "format", "placements", "start_date", "end_date", "is_active", "days_running",
            "impressions_min", "impressions_max", "impressions_index", "variations", "page_likes",
            "thumbnail", "ad_url"]
    df = pd.DataFrame(records, columns=cols)
    if df.empty:
        return df
    df["hook"] = [extract_hook(t, h) for t, h in zip(df["primary_text"], df["headline"])]
    df["hook_type"] = df["hook"].map(classify_hook)
    df["offer_type"] = [classify_offer(t, h, c) for t, h, c in
                        zip(df["primary_text"].fillna(""), df["headline"].fillna(""), df["cta"].fillna(""))]
    df["start_date"] = pd.to_datetime(df["start_date"], utc=True, errors="coerce")
    df["end_date"] = pd.to_datetime(df["end_date"], utc=True, errors="coerce")
    for c in ("days_running", "impressions_min", "impressions_max", "impressions_index",
              "variations", "page_likes"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["impressions_mid"] = (df["impressions_min"] + df["impressions_max"]) / 2
    df = df.drop_duplicates(subset=["platform", "ad_id"]).reset_index(drop=True)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Optional Gemini enrichment
# ─────────────────────────────────────────────────────────────────────────────
def gemini_enrich(df: pd.DataFrame, brand: str, api_key: str, model: str):
    """Returns (df, brief_markdown, error_message)."""
    if not GENAI_AVAILABLE:
        return df, "", "The google-genai package isn't installed (pip install google-genai)."

    sample = df.sort_values("days_running", ascending=False, na_position="last").head(80)
    ads_payload = [
        {
            "id": f"{r.platform}:{r.ad_id}",
            "platform": r.platform,
            "format": r.format,
            "days_running": None if pd.isna(r.days_running) else int(r.days_running),
            "cta": r.cta,
            "headline": truncate(r.headline or "", 150),
            "text": truncate(r.primary_text or "", 500),
        }
        for r in sample.itertuples()
    ]
    stats = {
        "total_ads": int(len(df)),
        "platform_split": df["platform"].value_counts().to_dict(),
        "format_split": df["format"].value_counts().to_dict(),
        "median_days_running": None if df["days_running"].dropna().empty
        else float(df["days_running"].median()),
    }
    prompt = f"""You are a senior performance-marketing strategist analysing the live paid ads of the brand "{brand}".

For EVERY ad below, identify:
- "hook": the attention-grabbing opening idea, max 15 words, in the ad's own words where possible
- "hook_type": exactly one of {json.dumps(HOOK_TYPES)}
- "offer_type": exactly one of {json.dumps(OFFER_TYPES)}

Then write "brief": a concise markdown strategy brief (max 220 words) with these bold section labels:
**Primary hook strategy**, **Offer strategy**, **Creative & platform strategy**, **What's likely working** (ads that run longest are usually profitable), **Gaps & opportunities**.
Base every claim only on the data provided.

Account stats: {json.dumps(stats)}
Ads (JSON): {json.dumps(ads_payload, ensure_ascii=False)}

Return ONLY JSON of the form:
{{"ads": [{{"id": "...", "hook": "...", "hook_type": "...", "offer_type": "..."}}], "brief": "..."}}"""

    try:
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model=model, contents=prompt, config={"response_mime_type": "application/json"}
        )
        raw = (response.text or "").strip()
        raw = re.sub(r"^```(?:json)?|```$", "", raw, flags=re.MULTILINE).strip()
        data = json.loads(raw)
    except Exception as exc:
        return df, "", f"Gemini analysis failed ({type(exc).__name__}: {truncate(str(exc), 200)}). " \
                       "Showing rule-based results instead."

    labels = {str(a.get("id")): a for a in (data.get("ads") or []) if isinstance(a, dict)}
    df = df.copy()
    keys = df["platform"] + ":" + df["ad_id"].astype(str)
    for idx, key in keys.items():
        lab = labels.get(key)
        if not lab:
            continue
        if isinstance(lab.get("hook"), str) and lab["hook"].strip():
            df.at[idx, "hook"] = truncate(lab["hook"].strip(), 150)
        if lab.get("hook_type") in HOOK_TYPES:
            df.at[idx, "hook_type"] = lab["hook_type"]
        if lab.get("offer_type") in OFFER_TYPES:
            df.at[idx, "offer_type"] = lab["offer_type"]
    brief = data.get("brief") if isinstance(data.get("brief"), str) else ""
    return df, brief, ""


# ─────────────────────────────────────────────────────────────────────────────
# Demo data (fictional brand, lets the UI run without API keys)
# ─────────────────────────────────────────────────────────────────────────────
def demo_records(brand: str) -> list:
    rng = random.Random(7)
    today = now_utc()
    meta_texts = [
        "Tired of juggling five tools to run one campaign? {b} puts planning, publishing and reporting in one place.",
        "Trusted by 12,000+ marketing teams. See why {b} is the #1 rated workspace for growth.",
        "What if your weekly report wrote itself? {b} turns raw data into insights in minutes.",
        "Last chance: 40% off annual plans ends Sunday. Lock in your price today.",
        "Introducing {b} AI Assist — draft, test and ship ads 3x faster.",
        "Start your 14-day free trial. No credit card needed.",
        "How to cut your reporting time in half: our free guide for lean teams.",
        "Your team deserves better than spreadsheets. Switch to {b} and never look back.",
    ]
    li_texts = [
        "How do top B2B teams forecast pipeline? Download the 2026 Revenue Benchmark Report.",
        "Join our live webinar: Building an AI-ready marketing stack, Oct 14.",
        "Book a 20-minute demo and see {b} on your own data.",
        "87% of CMOs say attribution is their biggest blind spot. Here's how {b} fixes it.",
        "We're hiring! Join the {b} product team in Bengaluru and London.",
    ]
    ctas_meta = ["Learn more", "Sign up", "Shop now", "Get offer", "Download"]
    recs = []
    for i in range(64):
        fmt = rng.choices(["Video", "Image", "Carousel", "Catalog (DPA)"], [45, 30, 18, 7])[0]
        start = today - timedelta(days=rng.choice([3, 8, 15, 22, 36, 51, 74, 96, 133, 188, 240]) + rng.randint(0, 6))
        active = rng.random() > 0.12
        end = None if active else start + timedelta(days=rng.randint(5, 40))
        places = rng.choice([["Facebook", "Instagram"], ["Facebook", "Instagram", "Messenger", "Audience Network"],
                             ["Instagram"], ["Facebook", "Instagram", "Threads"], ["Facebook"]])
        recs.append({
            "platform": "Meta", "ad_id": f"demo-m{i}", "advertiser": brand,
            "primary_text": rng.choice(meta_texts).format(b=brand), "headline": f"{brand} for growth teams",
            "cta": rng.choice(ctas_meta), "landing_url": "https://example.com", "format": fmt,
            "placements": places, "start_date": start, "end_date": end, "is_active": active,
            "days_running": max(1, ((today if active else end) - start).days + 1),
            "impressions_min": None, "impressions_max": None, "impressions_index": None,
            "variations": rng.choice([1, 1, 1, 2, 3, 5]), "page_likes": 184_000, "thumbnail": "",
            "ad_url": "",
        })
    for i in range(22):
        first = today - timedelta(days=rng.randint(4, 150))
        last = min(today, first + timedelta(days=rng.randint(3, 90)))
        lo = rng.choice([1000, 5000, 10000, 20000, 50000])
        recs.append({
            "platform": "LinkedIn", "ad_id": f"demo-l{i}", "advertiser": brand,
            "primary_text": rng.choice(li_texts).format(b=brand), "headline": "",
            "cta": rng.choice(["Learn more", "Register", "Download", "Request demo"]),
            "landing_url": "https://example.com",
            "format": rng.choices(["Image", "Document", "Video", "Carousel", "Event"], [40, 22, 18, 12, 8])[0],
            "placements": ["LinkedIn"], "start_date": first, "end_date": last,
            "is_active": (today - last).days <= 3, "days_running": (last - first).days + 1,
            "impressions_min": lo, "impressions_max": lo * 2 if lo < 50000 else 100000,
            "impressions_index": None, "variations": 1, "page_likes": None, "thumbnail": "", "ad_url": "",
        })
    return recs


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline
# ─────────────────────────────────────────────────────────────────────────────
def run_analysis(cfg: dict) -> dict:
    brand = cfg["brand"]
    notes, records, meta_info = [], [], {}

    if cfg["demo"]:
        notes.append("Demo mode: showing generated sample data for a fictional brand — no API calls were made.")
        df = build_dataframe(demo_records(brand))
        meta_info = {"page_name": brand, "likes": 184_000, "ig_followers": 96_500, "total": 64}
    else:
        key = cfg["searchapi_key"]
        # ── Meta ──
        if "Meta" in cfg["platforms"]:
            page_id = cfg["meta_page_id"].strip()
            chosen = None
            if not page_id:
                with st.spinner("Finding the brand's Meta page…"):
                    pages = meta_page_search(brand, cfg["meta_country"], key)
                chosen = pick_best_page(pages, brand)
                if chosen:
                    page_id = str(chosen.get("page_id") or "")
                    meta_info["candidates"] = [
                        {"Page": p.get("name"), "Page ID": p.get("page_id"), "Likes": p.get("likes"),
                         "Instagram": p.get("ig_username"), "Verified": p.get("verification")}
                        for p in pages[:8]
                    ]
                else:
                    notes.append("No Meta page matched the brand name, so Meta ads were found by keyword "
                                 "(may include other advertisers). Paste the Page ID in the sidebar for precision.")
            with st.spinner("Pulling Meta Ad Library…"):
                raw, page_info, total = fetch_meta_ads(
                    page_id, brand, cfg["meta_country"], cfg["meta_status"], cfg["max_ads"], key)
            meta_info.update({
                "page_id": page_id,
                "page_name": page_info.get("page_name") or (chosen or {}).get("name") or brand,
                "likes": page_info.get("likes") or (chosen or {}).get("likes"),
                "ig_followers": page_info.get("ig_followers") or (chosen or {}).get("ig_followers"),
                "ig_username": page_info.get("ig_username") or (chosen or {}).get("ig_username"),
                "verified": page_info.get("page_verification") or (chosen or {}).get("verification"),
                "total": total,
            })
            records.extend(normalize_meta_ad(a) for a in raw)
            if not raw:
                notes.append("Meta returned no ads for these filters.")

        # ── LinkedIn ──
        if "LinkedIn" in cfg["platforms"]:
            with st.spinner("Pulling LinkedIn Ad Library…"):
                raw_li, _ = fetch_linkedin_ads(brand, cfg["li_country"], cfg["li_period"], cfg["max_ads"], key)
            if cfg["li_strict"] and raw_li:
                b = brand.casefold()
                strict = [a for a in raw_li if b in str(((a.get("advertiser") or {}).get("name")
                                                         or a.get("organizer") or "")).casefold()]
                if strict:
                    raw_li = strict
                else:
                    notes.append("No LinkedIn ads had an advertiser name containing the brand, so all "
                                 "results for the advertiser search are shown.")
            details = {}
            if cfg["li_details"] and raw_li:
                ids = tuple(str(a.get("id")) for a in raw_li
                            if a.get("id") and "/ad-library/detail/" in str(a.get("link") or ""))
                ids = ids[: cfg["li_detail_limit"]]
                with st.spinner(f"Fetching run dates & impressions for {len(ids)} LinkedIn ads…"):
                    details = fetch_linkedin_details(ids, key)
            records.extend(normalize_linkedin_ad(a, details.get(str(a.get("id")))) for a in raw_li)
            if not raw_li:
                notes.append("LinkedIn returned no ads for this advertiser.")

        df = build_dataframe(records)

    ai_brief, ai_error = "", ""
    if cfg["gemini_key"] and not df.empty:
        with st.spinner("Gemini is labelling hooks and offers…"):
            df, ai_brief, ai_error = gemini_enrich(df, brand, cfg["gemini_key"], cfg["gemini_model"])
        if ai_error:
            notes.append(ai_error)

    return {"brand": brand, "df": df, "meta_info": meta_info, "notes": notes,
            "ai_brief": ai_brief, "ai_used": bool(cfg["gemini_key"]) and not ai_error,
            "demo": cfg["demo"], "generated": now_utc().strftime("%d %b %Y, %H:%M UTC")}


# ─────────────────────────────────────────────────────────────────────────────
# Insight computation
# ─────────────────────────────────────────────────────────────────────────────
def compute_insights(df: pd.DataFrame, meta_info: dict) -> dict:
    ins = {}
    # Primary hook
    hook_counts = df["hook_type"].value_counts()
    top_hook_type = hook_counts.idxmax()
    subset = df[df["hook_type"] == top_hook_type]
    common_hook = mode_or(subset["hook"], "")
    longest_in_type = subset.sort_values("days_running", ascending=False, na_position="last")
    rep_hook = common_hook if (subset["hook"] == common_hook).sum() > 1 else (
        longest_in_type["hook"].iloc[0] if not longest_in_type.empty else common_hook)
    ins["hook"] = {"type": top_hook_type, "share": hook_counts.iloc[0] / len(df),
                   "example": rep_hook, "runner_up": hook_counts.index[1] if len(hook_counts) > 1 else None}
    # Offer
    offer_counts = df["offer_type"].value_counts()
    ins["offer"] = {"type": offer_counts.idxmax(), "share": offer_counts.iloc[0] / len(df),
                    "top3": offer_counts.head(3), "top_cta": mode_or(df["cta"])}
    # Longest running
    dated = df.dropna(subset=["days_running"])
    ins["longest"] = dated.sort_values("days_running", ascending=False).iloc[0] if not dated.empty else None
    # Platform split
    ins["platform_split"] = df["platform"].value_counts()
    ins["placement_split"] = df.explode("placements")["placements"].value_counts()
    # Format
    fmt_counts = df["format"].value_counts()
    ins["format"] = {"type": fmt_counts.idxmax(), "share": fmt_counts.iloc[0] / len(df), "counts": fmt_counts}
    # Engagement / reach signals
    li = df[df["platform"] == "LinkedIn"].dropna(subset=["impressions_min"])
    ins["engagement"] = {
        "li_imp_min": li["impressions_min"].sum() if not li.empty else None,
        "li_imp_max": li["impressions_max"].sum() if not li.empty else None,
        "li_ads_with_imp": len(li),
        "li_top": li.sort_values("impressions_mid", ascending=False).iloc[0] if not li.empty else None,
        "meta_likes": meta_info.get("likes"),
        "meta_ig": meta_info.get("ig_followers"),
        "avg_variations": df.loc[df["platform"] == "Meta", "variations"].mean(),
        "median_days": dated["days_running"].median() if not dated.empty else None,
        "evergreen": int((dated["days_running"] >= 30).sum()),
    }
    return ins


# ─────────────────────────────────────────────────────────────────────────────
# UI components
# ─────────────────────────────────────────────────────────────────────────────
def kpi(label: str, value: str, sub: str = "") -> str:
    return (f'<div class="kpi"><div class="label">{esc(label)}</div>'
            f'<div class="value">{esc(value)}</div><div class="sub">{esc(sub)}</div></div>')


def split_bars(series: pd.Series, colors: dict | None = None, limit: int = 5, total: float | None = None) -> str:
    total = total or series.sum() or 1
    rows = []
    for i, (name, count) in enumerate(series.head(limit).items()):
        pct = 100 * count / total
        color = (colors or {}).get(name, PALETTE[i % len(PALETTE)])
        rows.append(
            f'<div class="split-row"><div class="name">{esc(name)}</div>'
            f'<div class="track"><div class="fill" style="width:{pct:.1f}%;background:{color}"></div></div>'
            f'<div class="pct">{pct:.0f}%</div></div>')
    return "".join(rows)


def insight_card(icon: str, title: str, main: str, inner_html: str) -> str:
    return (f'<div class="insight"><div class="head"><div class="icon">{icon}</div>'
            f'<div class="title">{esc(title)}</div></div><div class="main">{esc(main)}</div>{inner_html}</div>')


def style_fig(fig, height=360):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=50, b=10),
                      font=dict(family="Inter, sans-serif"), legend_title_text="",
                      title_font=dict(size=15))
    return fig


def show_chart(fig):
    st.plotly_chart(fig, config={"displayModeBar": False}, **_CHART_KW)


def platform_badge(p: str) -> str:
    cls = "b-meta" if p == "Meta" else "b-linkedin"
    return f'<span class="badge {cls}">{esc(p)}</span>'


def ad_card_html(r) -> str:
    thumb = (f'<img class="thumb" src="{esc(r.thumbnail)}" loading="lazy" referrerpolicy="no-referrer" '
             f'onerror="this.style.display=\'none\'">' if r.thumbnail
             else f'<div class="noimg">{esc(r.format)}</div>')
    status = ""
    if r.is_active is True:
        status = '<span class="badge b-green">Active</span>'
    elif r.is_active is False:
        status = '<span class="badge b-neutral">Ended</span>'
    days = f"{int(r.days_running)} days running" if pd.notna(r.days_running) else "Run length n/a"
    started = r.start_date.strftime("%d %b %Y") if pd.notna(r.start_date) else "—"
    imp = ""
    if pd.notna(r.impressions_min):
        imp = f" · {fmt_int(r.impressions_min)}–{fmt_int(r.impressions_max)} impressions"
    link = f'<a href="{esc(r.ad_url)}" target="_blank">View in Ad Library →</a>' if r.ad_url else ""
    return (
        f'<div class="ad-card">{thumb}<div class="body">'
        f'{platform_badge(r.platform)}<span class="badge b-neutral">{esc(r.format)}</span>'
        f'<span class="badge b-amber">{esc(r.offer_type)}</span>{status}'
        f'<div class="hook">{esc(r.hook or "(no copy)")}</div>'
        f'<div class="text">{esc(truncate(r.primary_text or "", 320))}</div>'
        f'<div class="meta">CTA: <b>{esc(r.cta or "—")}</b> · Started {esc(started)} · {esc(days)}{esc(imp)}</div>'
        f'<div class="meta">{link}</div></div></div>')


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🎯 AdLens")
    st.caption("Ad intelligence for Meta & LinkedIn")

    brand = st.text_input("Brand to analyze", value="HubSpot", help="Company / page name as it appears on Meta or LinkedIn")
    platforms = st.multiselect("Platforms", ["Meta", "LinkedIn"], default=["Meta", "LinkedIn"])
    max_ads = st.slider("Max ads per platform", 10, 300, 60, step=10,
                        help="More ads = better statistics, but more API credits.")

    with st.expander("Meta settings", expanded=False):
        meta_country_label = st.selectbox("Delivery country", list(META_COUNTRIES.keys()), index=0)
        meta_status = st.selectbox("Ad status", ["active", "all", "inactive"], index=0,
                                   help="'active' = ads currently running")
        meta_page_id = st.text_input("Meta Page ID (optional override)",
                                     help="From the Ad Library URL: view_all_page_id=…  Leave blank to auto-match.")

    with st.expander("LinkedIn settings", expanded=False):
        li_country_label = st.selectbox("Country ", list(LINKEDIN_COUNTRIES.keys()), index=0)
        li_period_label = st.selectbox("Time period", list(LINKEDIN_PERIODS.keys()), index=0)
        li_strict = st.checkbox("Only exact advertiser matches", value=True,
                                help="LinkedIn's advertiser search is fuzzy; this keeps only ads whose advertiser name contains the brand.")
        li_details = st.checkbox("Fetch run dates & impressions", value=True,
                                 help="1 extra API credit per ad. LinkedIn publishes these for EU-delivered ads only.")
        li_detail_limit = st.slider("Max ads to enrich", 5, 100, 25, step=5)

    st.markdown("---")
    searchapi_key = st.text_input("SearchApi.io API key 🔑", value=get_secret("SEARCHAPI_API_KEY"), type="password",
                                  help="Get one at searchapi.io — used for both Meta and LinkedIn ad libraries.")
    gemini_key = st.text_input("Gemini API key (optional) ✨", value=get_secret("GEMINI_API_KEY"), type="password",
                               help="Adds AI hook/offer labelling and a strategy brief.")
    gemini_model = st.selectbox("Gemini model", GEMINI_MODELS, index=0)
    demo = st.toggle("Demo mode (sample data, no keys)", value=False)

    analyze = st.button("Analyze ads 🚀", type="primary")

# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(
    """
<div class="hero">
  <h1>Ad Intelligence</h1>
  <p>Decode any brand's live Meta &amp; LinkedIn advertising — hooks, offers, longevity, platform mix, reach and creative formats.</p>
  <span class="pill">Meta Ad Library</span><span class="pill">LinkedIn Ad Library</span><span class="pill">AI analysis</span>
</div>
""",
    unsafe_allow_html=True,
)

if analyze:
    errors = []
    if not brand.strip():
        errors.append("Enter a brand name.")
    if not platforms:
        errors.append("Select at least one platform.")
    if not demo and not searchapi_key.strip():
        errors.append("Enter a SearchApi.io API key, or switch on Demo mode.")
    if errors:
        for e in errors:
            st.error(e)
    else:
        cfg = {
            "brand": brand.strip(), "platforms": platforms, "max_ads": int(max_ads), "demo": demo,
            "searchapi_key": searchapi_key.strip(), "gemini_key": gemini_key.strip(),
            "gemini_model": gemini_model, "meta_country": META_COUNTRIES[meta_country_label],
            "meta_status": meta_status, "meta_page_id": meta_page_id or "",
            "li_country": LINKEDIN_COUNTRIES[li_country_label], "li_period": LINKEDIN_PERIODS[li_period_label],
            "li_strict": li_strict, "li_details": li_details, "li_detail_limit": int(li_detail_limit),
        }
        try:
            st.session_state["result"] = run_analysis(cfg)
        except APIError as exc:
            st.error(str(exc))
        except Exception as exc:  # keep the app alive on anything unexpected
            st.error(f"Something went wrong: {type(exc).__name__}: {exc}")

result = st.session_state.get("result")

if not result:
    c1, c2, c3 = st.columns(3)
    blurbs = [
        ("🔎", "What it finds", "Primary hook, offer type, longest-running ad, platform split, "
                               "engagement & reach signals, and the dominant creative format."),
        ("🔌", "Where the data comes from", "Meta Ad Library (Facebook, Instagram, Messenger, Audience Network, "
                                            "Threads) and LinkedIn Ad Library, via one SearchApi.io key."),
        ("🚀", "Get started", "Enter a brand in the sidebar, add your SearchApi.io key (and optionally a Gemini key), "
                             "then click Analyze. No key yet? Switch on Demo mode."),
    ]
    for col, (icon, title, text) in zip((c1, c2, c3), blurbs):
        col.markdown(insight_card(icon, title, "", f'<div class="detail">{esc(text)}</div>'), unsafe_allow_html=True)
    st.stop()

df: pd.DataFrame = result["df"]
meta_info = result["meta_info"]

for note in result["notes"]:
    st.info(note)

if df.empty:
    st.warning("No ads found for this brand with the current filters. Try 'all' ad status, a different "
               "country, or a Meta Page ID override.")
    st.stop()

ins = compute_insights(df, meta_info)

# Header row
head_l, head_r = st.columns([3, 1])
with head_l:
    st.markdown(f"### {esc(result['brand'])}" + (" · demo data" if result.get("demo") else ""))
    sub = [f"{len(df)} ads analyzed", f"Generated {result['generated']}"]
    if meta_info.get("page_name") and meta_info.get("page_id"):
        sub.insert(0, f"Meta page: {meta_info['page_name']} (ID {meta_info['page_id']})")
    if result["ai_used"]:
        sub.append("AI-labelled with Gemini")
    st.caption(" · ".join(sub))
with head_r:
    if meta_info.get("candidates"):
        with st.popover("Wrong Meta page?"):
            st.caption("Top page matches — copy the right Page ID into the sidebar override.")
            show_df(pd.DataFrame(meta_info["candidates"]), hide_index=True)

# KPI row
active_n = int((df["is_active"] == True).sum())  # noqa: E712
eng = ins["engagement"]
longest = ins["longest"]
kcols = st.columns(6)
kpis = [
    ("Ads analyzed", f"{len(df)}", " · ".join(f"{p}: {n}" for p, n in ins["platform_split"].items())),
    ("Active now", f"{active_n}", f"{active_n / len(df):.0%} of analyzed ads"),
    ("Median run length", f"{eng['median_days']:.0f} days" if eng["median_days"] is not None else "—",
     f"{eng['evergreen']} ads running 30+ days"),
    ("Longest run", f"{int(longest.days_running)} days" if longest is not None else "—",
     longest.platform if longest is not None else ""),
    ("Dominant format", re.sub(r"\s*\(.*?\)", "", ins["format"]["type"]), f"{ins['format']['share']:.0%} of ads"),
    ("Primary offer", re.sub(r"\s*[(/].*$", "", ins["offer"]["type"]).strip(), f"{ins['offer']['share']:.0%} of ads"),
]
for col, (label, value, sub) in zip(kcols, kpis):
    col.markdown(kpi(label, value, sub), unsafe_allow_html=True)
st.write("")

tab_sum, tab_hooks, tab_creative, tab_long, tab_gallery, tab_data = st.tabs(
    ["📋 Summary", "🪝 Hooks & Offers", "🎨 Creative & Platforms", "⏳ Longevity & Reach", "🖼️ Ad Gallery", "📦 Data"]
)

# ── Summary ────────────────────────────────────────────────────────────────
with tab_sum:
    h = ins["hook"]
    hook_html = (f'<div class="quote">“{esc(h["example"])}”</div>' if h["example"] else "") + \
        f'<div class="detail"><b>{h["share"]:.0%}</b> of ads open with a {esc(h["type"].lower())} hook' + \
        (f'; runner-up: <b>{esc(h["runner_up"])}</b>.' if h["runner_up"] else ".") + "</div>"

    o = ins["offer"]
    offer_html = split_bars(o["top3"], total=len(df)) + f'<div class="detail" style="margin-top:8px">Most-used CTA button: ' \
                                         f'<b>{esc(o["top_cta"])}</b></div>'

    if longest is not None:
        started = longest.start_date.strftime("%d %b %Y") if pd.notna(longest.start_date) else "—"
        state = "still running" if longest.is_active is True else "ended"
        link = f' · <a href="{esc(longest.ad_url)}" target="_blank">open ad</a>' if longest.ad_url else ""
        long_main = f"{int(longest.days_running)} days · {longest.platform} {longest.format}"
        long_html = (f'<div class="quote">“{esc(truncate(longest.hook or longest.primary_text or "", 140))}”</div>'
                     f'<div class="detail">Started <b>{esc(started)}</b>, {state}. Offer: <b>{esc(longest.offer_type)}</b>'
                     f'{link}</div>')
    else:
        long_main, long_html = "Not available", '<div class="detail">No run-date data was returned.</div>'

    plat_html = split_bars(ins["platform_split"], PLATFORM_COLORS) + \
        '<div class="detail" style="margin-top:10px"><b>Placements</b></div>' + split_bars(ins["placement_split"], limit=4)

    eng_lines = []
    if eng["li_ads_with_imp"]:
        eng_lines.append(f"LinkedIn impressions: <b>{fmt_int(eng['li_imp_min'])}–{fmt_int(eng['li_imp_max'])}</b> "
                         f"across {eng['li_ads_with_imp']} ads")
    if eng["meta_likes"]:
        eng_lines.append(f"Meta page likes: <b>{fmt_int(eng['meta_likes'])}</b>")
    if eng["meta_ig"]:
        eng_lines.append(f"Instagram followers: <b>{fmt_int(eng['meta_ig'])}</b>")
    if pd.notna(eng["avg_variations"]):
        eng_lines.append(f"Avg creative variations per Meta ad: <b>{eng['avg_variations']:.1f}</b>")
    eng_lines.append(f"Evergreen ads (30+ days): <b>{eng['evergreen']}</b>")
    eng_main = (f"{fmt_int(eng['li_imp_max'])} max impressions" if eng["li_ads_with_imp"]
                else f"{fmt_int(eng['meta_likes'])} page likes" if eng["meta_likes"] else "Reach signals")
    eng_html = '<div class="detail">' + "<br>".join(eng_lines) + \
        '<br><span style="color:#9CA3AF;font-size:0.76rem">Ad libraries don\'t publish likes/comments per ad; ' \
        'these are the public reach signals.</span></div>'

    f = ins["format"]
    fmt_html = split_bars(f["counts"])

    r1 = st.columns(3)
    r1[0].markdown(insight_card("🪝", "Primary hook", h["type"], hook_html), unsafe_allow_html=True)
    r1[1].markdown(insight_card("🏷️", "Offer type", o["type"], offer_html), unsafe_allow_html=True)
    r1[2].markdown(insight_card("⏳", "Longest-running ad", long_main, long_html), unsafe_allow_html=True)
    r2 = st.columns(3)
    r2[0].markdown(insight_card("📊", "Platform split",
                                " / ".join(f"{p} {n / len(df):.0%}" for p, n in ins["platform_split"].items()),
                                plat_html), unsafe_allow_html=True)
    r2[1].markdown(insight_card("📈", "Engagement & reach", eng_main, eng_html), unsafe_allow_html=True)
    r2[2].markdown(insight_card("🎬", "Dominant creative format", f"{f['type']} ({f['share']:.0%})", fmt_html),
                   unsafe_allow_html=True)

    if result["ai_brief"]:
        st.markdown("#### ✨ AI strategy brief")
        st.markdown(result["ai_brief"])
    elif not result["ai_used"]:
        st.caption("Add a Gemini API key in the sidebar for AI-written hook labels and a strategy brief.")

# ── Hooks & Offers ─────────────────────────────────────────────────────────
with tab_hooks:
    c1, c2 = st.columns(2)
    with c1:
        hc = df["hook_type"].value_counts().rename_axis("Hook type").reset_index(name="Ads")
        fig = px.bar(hc, x="Ads", y="Hook type", orientation="h", title="Hook types",
                     color_discrete_sequence=[PALETTE[0]])
        fig.update_yaxes(categoryorder="total ascending")
        show_chart(style_fig(fig))
    with c2:
        oc = df["offer_type"].value_counts().rename_axis("Offer").reset_index(name="Ads")
        fig = px.pie(oc, names="Offer", values="Ads", hole=0.55, title="Offer mix",
                     color_discrete_sequence=PALETTE)
        show_chart(style_fig(fig))

    st.markdown("#### Top hooks")
    st.markdown('<div class="section-note">Grouped by opening line; ranked by how many ads use it and how long they run.</div>',
                unsafe_allow_html=True)
    top_hooks = (df[df["hook"].str.len() > 0]
                 .groupby("hook")
                 .agg(Ads=("ad_id", "count"), Type=("hook_type", "first"), Offer=("offer_type", "first"),
                      Platforms=("platform", lambda s: ", ".join(sorted(set(s)))),
                      **{"Max days running": ("days_running", "max")})
                 .reset_index().rename(columns={"hook": "Hook"})
                 .sort_values(["Ads", "Max days running"], ascending=False).head(20))
    show_df(top_hooks, hide_index=True)

    c3, c4 = st.columns(2)
    with c3:
        cta = df[df["cta"].str.len() > 0]["cta"].str.title().value_counts().head(10)
        cta = cta.rename_axis("CTA").reset_index(name="Ads")
        if not cta.empty:
            fig = px.bar(cta, x="Ads", y="CTA", orientation="h", title="CTA buttons",
                         color_discrete_sequence=[PALETTE[1]])
            fig.update_yaxes(categoryorder="total ascending")
            show_chart(style_fig(fig))
    with c4:
        ho = pd.crosstab(df["hook_type"], df["offer_type"])
        fig = px.imshow(ho, text_auto=True, aspect="auto", color_continuous_scale="Blues",
                        title="Hook × Offer combinations")
        fig.update_layout(coloraxis_showscale=False, xaxis_title="", yaxis_title="")
        show_chart(style_fig(fig, 380))

# ── Creative & Platforms ───────────────────────────────────────────────────
with tab_creative:
    c1, c2 = st.columns(2)
    with c1:
        fp = df.groupby(["format", "platform"]).size().reset_index(name="Ads")
        fig = px.bar(fp, x="format", y="Ads", color="platform", title="Creative formats by platform",
                     color_discrete_map=PLATFORM_COLORS, barmode="stack")
        fig.update_xaxes(title="", categoryorder="total descending")
        show_chart(style_fig(fig))
    with c2:
        pc = ins["placement_split"].rename_axis("Placement").reset_index(name="Ads")
        fig = px.bar(pc, x="Ads", y="Placement", orientation="h", title="Placement coverage",
                     color_discrete_sequence=[PALETTE[2]])
        fig.update_yaxes(categoryorder="total ascending")
        show_chart(style_fig(fig))

    timeline = df.dropna(subset=["start_date"]).copy()
    if not timeline.empty:
        timeline["Week"] = timeline["start_date"].dt.tz_convert(None).dt.to_period("W").dt.start_time
        tl = timeline.groupby(["Week", "platform"]).size().reset_index(name="Ads launched")
        fig = px.bar(tl, x="Week", y="Ads launched", color="platform", title="Launch cadence (ads started per week)",
                     color_discrete_map=PLATFORM_COLORS)
        show_chart(style_fig(fig, 320))

    c3, c4 = st.columns(2)
    with c3:
        fo = df.groupby("format")["days_running"].median().dropna().sort_values(ascending=False)
        if not fo.empty:
            fig = px.bar(fo.rename_axis("Format").reset_index(name="Median days"), x="Format", y="Median days",
                         title="Median run length by format", color_discrete_sequence=[PALETTE[5]])
            show_chart(style_fig(fig, 320))
    with c4:
        pl = df.groupby("platform")["days_running"].median().dropna()
        if not pl.empty:
            fig = px.bar(pl.rename_axis("Platform").reset_index(name="Median days"), x="Platform", y="Median days",
                         title="Median run length by platform", color="Platform",
                         color_discrete_map=PLATFORM_COLORS)
            show_chart(style_fig(fig, 320))

# ── Longevity & Reach ──────────────────────────────────────────────────────
with tab_long:
    st.markdown('<div class="section-note">Ads that keep running are usually the profitable ones — '
                'longevity is the strongest public performance signal in ad libraries.</div>', unsafe_allow_html=True)
    dated = df.dropna(subset=["days_running"])
    if dated.empty:
        st.info("No run-date data available for these ads.")
    else:
        fig = px.histogram(dated, x="days_running", color="platform", nbins=30, barmode="overlay",
                           title="Distribution of days running", color_discrete_map=PLATFORM_COLORS,
                           labels={"days_running": "Days running"})
        show_chart(style_fig(fig, 320))

        st.markdown("#### Longest-running ads")
        top = dated.sort_values("days_running", ascending=False).head(15)
        show_df(
            top[["platform", "format", "hook", "offer_type", "cta", "days_running", "start_date", "is_active", "ad_url"]]
            .assign(start_date=lambda d: d["start_date"].dt.strftime("%Y-%m-%d")),
            hide_index=True,
            column_config={
                "platform": "Platform", "format": "Format", "hook": st.column_config.TextColumn("Hook", width="large"),
                "offer_type": "Offer", "cta": "CTA",
                "days_running": st.column_config.NumberColumn("Days", format="%d"),
                "start_date": "Started", "is_active": "Active",
                "ad_url": st.column_config.LinkColumn("Ad", display_text="Open"),
            },
        )

    li = df[(df["platform"] == "LinkedIn") & df["impressions_mid"].notna()]
    if not li.empty:
        st.markdown("#### LinkedIn impressions (EU-reported)")
        li_plot = li.sort_values("impressions_mid", ascending=False).head(15).copy()
        li_plot["Ad"] = li_plot["hook"].map(lambda t: truncate(t or "(no copy)", 60))
        li_plot["err"] = li_plot["impressions_max"] - li_plot["impressions_mid"]
        fig = px.bar(li_plot, x="impressions_mid", y="Ad", orientation="h", error_x="err",
                     title="Estimated impressions (range midpoint)", color_discrete_sequence=[PLATFORM_COLORS["LinkedIn"]],
                     labels={"impressions_mid": "Impressions"})
        fig.update_yaxes(categoryorder="total ascending", title="")
        show_chart(style_fig(fig, 460))
    elif "LinkedIn" in set(df["platform"]):
        st.caption("LinkedIn only publishes impressions for ads delivered in the EU; none were available here.")

    if meta_info.get("likes") or meta_info.get("ig_followers"):
        m1, m2, m3 = st.columns(3)
        m1.markdown(kpi("Meta page likes", fmt_int(meta_info.get("likes"))), unsafe_allow_html=True)
        m2.markdown(kpi("Instagram followers", fmt_int(meta_info.get("ig_followers")),
                        f"@{meta_info['ig_username']}" if meta_info.get("ig_username") else ""), unsafe_allow_html=True)
        m3.markdown(kpi("Meta ads in library", fmt_int(meta_info.get("total")), "for current filters"),
                    unsafe_allow_html=True)

# ── Ad Gallery ─────────────────────────────────────────────────────────────
with tab_gallery:
    f1, f2, f3, f4 = st.columns(4)
    sel_platform = f1.multiselect("Platform", sorted(df["platform"].unique()), default=sorted(df["platform"].unique()))
    sel_format = f2.multiselect("Format", sorted(df["format"].unique()))
    sel_offer = f3.multiselect("Offer", sorted(df["offer_type"].unique()))
    sort_by = f4.selectbox("Sort by", ["Longest running", "Newest", "Most impressions"])

    view = df[df["platform"].isin(sel_platform)]
    if sel_format:
        view = view[view["format"].isin(sel_format)]
    if sel_offer:
        view = view[view["offer_type"].isin(sel_offer)]
    if sort_by == "Longest running":
        view = view.sort_values("days_running", ascending=False, na_position="last")
    elif sort_by == "Newest":
        view = view.sort_values("start_date", ascending=False, na_position="last")
    else:
        view = view.sort_values("impressions_mid", ascending=False, na_position="last")

    st.caption(f"Showing {min(len(view), 60)} of {len(view)} ads")
    cols = st.columns(3)
    for i, r in enumerate(view.head(60).itertuples()):
        cols[i % 3].markdown(ad_card_html(r), unsafe_allow_html=True)

# ── Data ───────────────────────────────────────────────────────────────────
with tab_data:
    export = df.copy()
    export["placements"] = export["placements"].map(lambda p: ", ".join(p) if isinstance(p, list) else p)
    for c in ("start_date", "end_date"):
        export[c] = export[c].dt.strftime("%Y-%m-%d")
    order = ["platform", "advertiser", "format", "hook", "hook_type", "offer_type", "cta", "primary_text",
             "headline", "placements", "start_date", "end_date", "is_active", "days_running",
             "impressions_min", "impressions_max", "variations", "landing_url", "ad_url", "ad_id"]
    export = export[order]
    show_df(export, hide_index=True, height=520,
                 column_config={"ad_url": st.column_config.LinkColumn("Ad"),
                                "landing_url": st.column_config.LinkColumn("Landing page")})
    safe_name = re.sub(r"[^A-Za-z0-9]+", "_", result["brand"]).strip("_") or "brand"
    st.download_button("⬇️ Download CSV", export.to_csv(index=False).encode("utf-8"),
                       file_name=f"{safe_name}_ad_intelligence.csv", mime="text/csv")
