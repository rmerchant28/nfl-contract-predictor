"""
OverTheCap scraper
==================
Collects two things:
  1. Player contracts  (position, APY, total value, signing year, contract type)
  2. Historical salary cap figures by year

Source: https://overthecap.com
"""

import re
import logging
from typing import Optional

import pandas as pd
from bs4 import BeautifulSoup, Tag, Tag

from .utils import fetch, save_raw, clean_money, normalise_name, map_position

log = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────
OTC_BASE = "https://overthecap.com"

# Contract HISTORY pages — have actual signing years, not current contracts
# URL pattern: https://overthecap.com/contract-history/{slug}
POSITION_SLUGS = {
    "QB":  "quarterback",
    "WR":  "wide-receiver",
    "RB":  "running-back",
    "TE":  "tight-end",
    "OT":  "left-tackle",
    "OG":  "left-guard",
    "C":   "center",
}

# Minimum APY thresholds — only exclude true minimum/practice squad deals
# Lowered QB threshold to include backup contracts (rookies are filtered out later by missing stats)
ROOKIE_APY_THRESHOLD = {
    "QB":  750_000,
    "WR":  2_000_000,
    "RB":  2_000_000,
    "TE":  2_000_000,
    "OT":  3_000_000,
    "OG":  2_000_000,
    "C":   2_000_000,
}

# Historical cap figures — hardcoded as a reliable fallback.
# OTC no longer has a single dedicated cap history page.
# Source: overthecap.com (2026 base cap confirmed $301.2M)
CAP_HISTORY_FALLBACK = {
    1994: 34_600_000,  2000: 62_172_000,  2001: 67_405_000,
    2002: 71_101_000,  2003: 75_007_000,  2004: 80_582_000,
    2005: 85_500_000,  2006: 102_000_000, 2007: 109_000_000,
    2008: 116_000_000, 2009: 123_000_000, 2010: 102_000_000,
    2011: 120_000_000, 2012: 120_600_000, 2013: 123_000_000,
    2014: 133_000_000, 2015: 143_280_000, 2016: 155_270_000,
    2017: 167_000_000, 2018: 177_200_000, 2019: 188_200_000,
    2020: 198_200_000, 2021: 182_500_000, 2022: 208_200_000,
    2023: 224_800_000, 2024: 255_400_000, 2025: 279_000_000,
    2026: 301_200_000,
}

CAP_HISTORY_URL = "https://overthecap.com/salary-cap-space"


# ── Salary cap history ─────────────────────────────────────────────────────────
def scrape_cap_history() -> pd.DataFrame:
    """
    Return NFL salary cap by year as a DataFrame.
    Uses a hardcoded lookup table (OTC no longer has a single cap history page).
    Values sourced from overthecap.com and cross-checked against nflref.

    Returns a DataFrame with columns:
        year (int), salary_cap (float)
    """
    log.info("Loading salary cap history from built-in table...")
    rows = [{"year": yr, "salary_cap": cap} for yr, cap in CAP_HISTORY_FALLBACK.items()]
    df = pd.DataFrame(rows).sort_values("year").reset_index(drop=True)
    log.info("  Cap figures for %d seasons (%d – %d)",
             len(df), df["year"].min(), df["year"].max())
    save_raw(df, "cap_history_otc")
    return df


# ── Contract pages ─────────────────────────────────────────────────────────────
def _contract_url(position_slug: str, page: int = 1) -> str:
    url = f"{OTC_BASE}/contract-history/{position_slug}"
    if page > 1:
        url += f"?page={page}"
    return url


def _parse_contracts_table(soup: BeautifulSoup, position: str) -> list[dict]:
    """
    Parse one OTC contracts page using pandas.read_html for robustness.
    Falls back to manual BS4 parsing if pandas can't find a table.
    """
    records = []

    # ── Try pandas.read_html first (most reliable) ─────────────────────────
    try:
        # Strip HTML comments (LiteSpeed cache injects them and breaks read_html)
        html_str = re.sub(r'<!--.*?-->', '', str(soup), flags=re.DOTALL)
        tables = pd.read_html(html_str)
        if not tables:
            return records
        # Pick the largest table — that's the contracts table
        df = max(tables, key=len)
        # Normalise column names for matching
        df.columns = [str(c).strip().lower() for c in df.columns]

        log.debug("OTC table columns: %s", df.columns.tolist())

        # ── Map whatever OTC calls these columns to our names ──────────────
        # OTC uses headers like "Player", "Team", "Years", "Total Value",
        # "Avg./Year", "Guaranteed", "At Sign.", "Free Agent", "Inflated Value"
        col_map = {
            "player":           "player_name",
            "name":             "player_name",
            "team":             "team",
            "yearsigned":       "signing_year",
            "year signed":      "signing_year",
            "signed":           "signing_year",
            "years":            "contract_years",
            "yrs":              "contract_years",
            "length":           "contract_years",
            "value":            "total_value",
            "apy":              "apy",
            "avg./year":        "apy",
            "avg/year":         "apy",
            "average":          "apy",
            "per year":         "apy",
            "guaranteed":       "guaranteed",
            "fully gtd":        "guaranteed",
            "apy as % ofcap at signing": "apy_pct_cap_otc",
            "apy as % of cap":  "apy_pct_cap_otc",
            "inflatedvalue":    "inflated_value",
            "inflated value":   "inflated_value",
        }
        # Apply first-match rename
        rename = {}
        for col in df.columns:
            for pattern, target in col_map.items():
                if pattern in col and target not in rename.values():
                    rename[col] = target
                    break
        df = df.rename(columns=rename)

        # ── Extract signing year — direct YearSigned column ───────────────
        if "signing_year" in df.columns:
            df["signing_year"] = (
                df["signing_year"]
                .astype(str)
                .str.extract(r"(20\d{2})", expand=False)
                .pipe(pd.to_numeric, errors="coerce")
            )

        # ── Clean money columns ────────────────────────────────────────────
        for money_col in ["total_value", "apy", "guaranteed", "gtd_at_sign"]:
            if money_col in df.columns:
                df[money_col] = df[money_col].apply(
                    lambda x: clean_money(str(x)) if pd.notna(x) else None
                )

        # ── Clean contract years ───────────────────────────────────────────
        if "contract_years" in df.columns:
            df["contract_years"] = (
                df["contract_years"]
                .astype(str)
                .str.extract(r"(\d+)", expand=False)
                .pipe(pd.to_numeric, errors="coerce")
            )

        # ── Build records ──────────────────────────────────────────────────
        for _, row in df.iterrows():
            player_name = str(row.get("player_name", "")).strip()
            if not player_name or player_name.lower() in ("nan", "player", "name"):
                continue
            canon_pos, pos_group = map_position(
                str(row.get("position_raw", position)).strip()
            )
            records.append({
                "player_name":      player_name,
                "player_name_norm": normalise_name(player_name),
                "team":             row.get("team"),
                "position_raw":     position,
                "position":         canon_pos,
                "position_group":   pos_group,
                "total_value":      row.get("total_value"),
                "apy":              row.get("apy"),
                "guaranteed":       row.get("guaranteed"),
                "contract_years":   row.get("contract_years"),
                "signing_year":     row.get("signing_year"),
                "source_url":       None,
            })
        return records

    except Exception as e:
        log.warning("pandas.read_html failed (%s), falling back to BS4.", e)

    # ── BS4 fallback — contract-history table column order ─────────────────
    # Confirmed headers: Player(0), Team(1), YearSigned(2), Years(3), blank(4),
    # Value(5), APY(6), Guaranteed(7), blank(8), APY%ofCap(9), blank(10),
    # InflatedValue(11) — rows have extra cells for inflated APY/Gtd too
    table = soup.find("table")
    if table is None:
        return records

    for tr in table.find_all("tr")[1:]:
        cells = tr.find_all(["td", "th"])
        if len(cells) < 7:
            continue

        player_name = cells[0].get_text(strip=True)
        if not player_name or player_name.lower() in ("player", "name", ""):
            continue

        team            = cells[1].get_text(strip=True) if len(cells) > 1 else None
        signing_year_str= cells[2].get_text(strip=True) if len(cells) > 2 else None
        contract_yrs_str= cells[3].get_text(strip=True) if len(cells) > 3 else None
        total_value_str = cells[5].get_text(strip=True) if len(cells) > 5 else None
        apy_str         = cells[6].get_text(strip=True) if len(cells) > 6 else None
        gtd_str         = cells[7].get_text(strip=True) if len(cells) > 7 else None

        # YearSigned is a direct 4-digit year
        signing_year = None
        if signing_year_str:
            m = re.search(r"(20\d{2})", signing_year_str)
            if m:
                signing_year = int(m.group(1))

        contract_years = None
        if contract_yrs_str:
            m = re.search(r"(\d+)", contract_yrs_str)
            if m:
                contract_years = int(m.group(1))

        canon_pos, pos_group = map_position(position)
        records.append({
            "player_name":      player_name,
            "player_name_norm": normalise_name(player_name),
            "team":             team,
            "position_raw":     position,
            "position":         canon_pos,
            "position_group":   pos_group,
            "total_value":      clean_money(total_value_str) if total_value_str else None,
            "apy":              clean_money(apy_str)         if apy_str         else None,
            "guaranteed":       clean_money(gtd_str)         if gtd_str         else None,
            "contract_years":   contract_years,
            "signing_year":     signing_year,
            "source_url":       None,
        })

    return records


def _scrape_position_contracts(position: str, slug: str, max_pages: int = 10) -> pd.DataFrame:
    """Paginate through all contract pages for a given position."""
    all_records = []

    for page in range(1, max_pages + 1):
        url  = _contract_url(slug, page)
        soup = fetch(url, delay=2.0)

        records = _parse_contracts_table(soup, position)
        if not records:
            log.info("  No records on page %d for %s — stopping.", page, position)
            break

        all_records.extend(records)
        log.info("  %s page %d: +%d contracts (total %d)",
                 position, page, len(records), len(all_records))

        # Check for a "next" pagination link; stop if none
        next_link = soup.find("a", string=re.compile(r"next|»", re.I))
        if next_link is None:
            break

    return pd.DataFrame(all_records)


def scrape_contracts(positions: Optional[list[str]] = None) -> pd.DataFrame:
    """
    Scrape contracts for all (or specified) positions.

    Args:
        positions: List of position codes e.g. ['QB', 'WR'].
                   Defaults to all positions in POSITION_SLUGS.

    Returns:
        Combined DataFrame of all contracts.
    """
    if positions is None:
        positions = list(POSITION_SLUGS.keys())

    frames = []
    for pos in positions:
        slug = POSITION_SLUGS.get(pos)
        if slug is None:
            log.warning("No OTC slug for position %s — skipping.", pos)
            continue

        log.info("Scraping OTC contracts: %s (%s)", pos, slug)
        df = _scrape_position_contracts(pos, slug)
        if not df.empty:
            frames.append(df)

    if not frames:
        log.warning("No contracts scraped.")
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)

    # Deduplicate: first drop exact duplicates, then keep highest APY per player+position+year
    combined = combined.drop_duplicates(subset=["player_name_norm", "signing_year", "apy"])
    combined = (
        combined
        .sort_values("apy", ascending=False)
        .drop_duplicates(subset=["player_name_norm", "position", "signing_year"], keep="first")
        .reset_index(drop=True)
    )

    # Filter out rookie/minimum contracts — we only want extensions and FA deals
    before = len(combined)
    def is_veteran_deal(row):
        # Use position_raw (always populated from scraper) or fall back to position
        pos = str(row.get("position") or row.get("position_raw") or "")
        threshold = ROOKIE_APY_THRESHOLD.get(pos, 4_000_000)
        apy = row.get("apy")
        if pd.isna(apy) or apy is None:
            return False
        return float(apy) >= threshold

    combined = combined[combined.apply(is_veteran_deal, axis=1)].reset_index(drop=True)
    log.info("Filtered rookie deals: %d → %d contracts", before, len(combined))

    if len(combined) == 0:
        log.warning("Rookie filter removed all contracts — check position parsing. Keeping all.")
        combined = pd.concat(frames, ignore_index=True)
        combined = combined.drop_duplicates(subset=["player_name_norm", "signing_year", "apy"])

    save_raw(combined, "contracts_otc")
    log.info("Total contracts scraped: %d", len(combined))
    return combined


# ── Enrich with cap % ──────────────────────────────────────────────────────────
def add_cap_percentage(contracts: pd.DataFrame, cap_history: pd.DataFrame) -> pd.DataFrame:
    """
    Merge salary cap data onto contracts and compute APY as % of cap.
    This is the target variable for the model.
    """
    contracts["signing_year"] = pd.to_numeric(contracts["signing_year"], errors="coerce").astype("Int64")
    cap_history["year"] = pd.to_numeric(cap_history["year"], errors="coerce").astype("Int64")

    merged = contracts.merge(
        cap_history.rename(columns={"salary_cap": "cap_that_year"}),
        left_on="signing_year",
        right_on="year",
        how="left",
    )
    merged["apy_pct_cap"] = merged["apy"] / merged["cap_that_year"]
    merged["guaranteed_pct_cap"] = merged["guaranteed"] / merged["cap_that_year"]

    missing = merged["apy_pct_cap"].isna().sum()
    if missing:
        log.warning("%d contracts missing cap pct (no cap data for that year).", missing)

    return merged


# ── CLI entry point ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Scrape NFL contracts from OverTheCap")
    parser.add_argument(
        "--positions", nargs="*",
        default=None,
        help="Positions to scrape (e.g. QB WR RB). Defaults to all.",
    )
    parser.add_argument(
        "--cap-only", action="store_true",
        help="Only scrape cap history, skip contracts.",
    )
    args = parser.parse_args()

    cap_df = scrape_cap_history()
    print(cap_df.tail())

    if not args.cap_only:
        contracts_df = scrape_contracts(args.positions)
        enriched_df  = add_cap_percentage(contracts_df, cap_df)
        save_raw(enriched_df, "contracts_with_cap_pct")
        print(enriched_df[["player_name", "position", "signing_year",
                            "apy", "cap_that_year", "apy_pct_cap"]].head(10))