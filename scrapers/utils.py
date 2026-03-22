"""
Shared utilities for NFL contract scraping.
Handles: session management, rate limiting, local caching, HTML parsing helpers.
"""

import time
import json
import hashlib
import logging
import os
from pathlib import Path
from datetime import datetime, timedelta

import requests
from bs4 import BeautifulSoup
import pandas as pd

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR      = PROJECT_ROOT / "data" / "raw"
PROC_DIR     = PROJECT_ROOT / "data" / "processed"
CACHE_DIR    = PROJECT_ROOT / "data" / ".cache"

for _d in (RAW_DIR, PROC_DIR, CACHE_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ── HTTP session ───────────────────────────────────────────────────────────────
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}


def make_session() -> requests.Session:
    """Return a requests.Session with shared headers and retry logic."""
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry

    session = requests.Session()
    session.headers.update(HEADERS)

    retry = Retry(
        total=4,
        backoff_factor=1.5,          # 0s, 1.5s, 3s, 6s
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://",  adapter)
    return session


SESSION = make_session()


# ── Caching ────────────────────────────────────────────────────────────────────
def _cache_path(url: str) -> Path:
    key = hashlib.md5(url.encode()).hexdigest()
    return CACHE_DIR / f"{key}.html"


def fetch(
    url: str,
    *,
    delay: float = 1.5,
    cache_ttl_days: int = 7,
    force: bool = False,
) -> BeautifulSoup:
    """
    Fetch a URL and return a BeautifulSoup object.

    Results are cached locally for `cache_ttl_days` days so repeat runs
    don't hammer the source sites. Pass force=True to bypass the cache.

    Args:
        url:            Target URL.
        delay:          Seconds to sleep BEFORE each live request (be polite).
        cache_ttl_days: How long to consider a cached response fresh.
        force:          If True, ignore any existing cache.
    """
    cache_file = _cache_path(url)

    # Return cached version if fresh enough
    if not force and cache_file.exists():
        age = datetime.now() - datetime.fromtimestamp(cache_file.stat().st_mtime)
        if age < timedelta(days=cache_ttl_days):
            log.debug("Cache hit  %s", url)
            return BeautifulSoup(cache_file.read_text(encoding="utf-8"), "lxml")

    log.info("Fetching   %s", url)
    time.sleep(delay)

    resp = SESSION.get(url, timeout=30)
    resp.raise_for_status()

    cache_file.write_text(resp.text, encoding="utf-8")
    return BeautifulSoup(resp.text, "lxml")


# ── DataFrame helpers ──────────────────────────────────────────────────────────
def save_raw(df: pd.DataFrame, name: str) -> Path:
    """Save a raw DataFrame to CSV under data/raw/."""
    path = RAW_DIR / f"{name}.csv"
    df.to_csv(path, index=False)
    log.info("Saved %d rows → %s", len(df), path)
    return path


def load_raw(name: str) -> pd.DataFrame:
    path = RAW_DIR / f"{name}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Raw file not found: {path}. Run the scraper first.")
    return pd.read_csv(path)


def save_processed(df: pd.DataFrame, name: str) -> Path:
    path = PROC_DIR / f"{name}.csv"
    df.to_csv(path, index=False)
    log.info("Saved %d rows → %s", len(df), path)
    return path


# ── Normalisation helpers ──────────────────────────────────────────────────────
def clean_money(value: str) -> float | None:
    """'$24,500,000' or '24.5M'  →  24500000.0"""
    if not isinstance(value, str):
        return None
    v = value.strip().replace("$", "").replace(",", "")
    if v.upper().endswith("M"):
        try:
            return float(v[:-1]) * 1_000_000
        except ValueError:
            return None
    try:
        return float(v)
    except ValueError:
        return None


def clean_pct(value: str) -> float | None:
    """'12.5%'  →  0.125"""
    if not isinstance(value, str):
        return None
    try:
        return float(value.strip().replace("%", "")) / 100
    except ValueError:
        return None


def normalise_name(name: str) -> str:
    """
    Light normalisation so names can be joined across sources.
    'Patrick Mahomes II' → 'patrick mahomes ii'
    Suffixes (Jr., Sr., II, III) are kept so we don't collide on common surnames.
    """
    import unicodedata
    name = unicodedata.normalize("NFKD", name)
    name = name.encode("ascii", "ignore").decode("ascii")
    return name.strip().lower()


POSITION_MAP = {
    # Quarterback
    "QB": "QB",
    # Skill
    "WR": "WR", "RB": "RB", "HB": "RB", "FB": "RB", "TE": "TE",
    # OL
    "T": "OT", "OT": "OT", "G": "OG", "OG": "OG", "C": "C",
    "LT": "OT", "RT": "OT", "LG": "OG", "RG": "OG",
}

POSITION_GROUP = {
    "QB": "QB",
    "WR": "SKILL", "RB": "SKILL", "TE": "SKILL",
    "OT": "OL",    "OG": "OL",    "C": "OL",
}


def map_position(raw_pos: str) -> tuple[str | None, str | None]:
    """
    Returns (canonical_position, position_group) or (None, None) if not in scope.
    E.g. 'LT' → ('OT', 'OL')
    """
    pos = POSITION_MAP.get(raw_pos.strip().upper())
    if pos is None:
        return None, None
    return pos, POSITION_GROUP.get(pos)
