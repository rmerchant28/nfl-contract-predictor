"""
Player stats via nflreadpy
===========================
nflreadpy is the Python port of the R nflreadr package, pulling from the
same nflverse data repository. It includes 2025+ season data and has no
pandas version constraints.

Install: pip install nflreadpy 'polars[rtcompat]'
"""

import logging
import ssl
import urllib.request
from typing import Optional

import numpy as np
import pandas as pd

from .utils import save_raw, normalise_name

log = logging.getLogger(__name__)

MIN_YEAR = 2011
MAX_YEAR = 2025


# ── SSL fix (macOS Python 3 ships without system certs) ───────────────────────
def _fix_ssl():
    try:
        import certifi
        ctx = ssl.create_default_context(cafile=certifi.where())
        opener = urllib.request.build_opener(urllib.request.HTTPSHandler(context=ctx))
        urllib.request.install_opener(opener)
    except ImportError:
        log.warning("certifi not installed. Run: pip install certifi")
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        opener = urllib.request.build_opener(urllib.request.HTTPSHandler(context=ctx))
        urllib.request.install_opener(opener)


def _import_nflreadpy():
    try:
        import nflreadpy
        return nflreadpy
    except ImportError:
        raise ImportError(
            "nflreadpy is not installed. Run: pip install nflreadpy 'polars[rtcompat]'"
        )


def _make_name_key(name: str) -> str:
    """
    Produce a match key that works across full and abbreviated names.
    'Patrick Mahomes' → 'p.mahomes'
    'P.Mahomes'       → 'p.mahomes'
    'P. Mahomes'      → 'p.mahomes'
    """
    if not isinstance(name, str):
        return ""
    import unicodedata
    name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode("ascii")
    name = name.strip().lower()
    parts = name.replace(".", " ").split()
    if len(parts) >= 2:
        return f"{parts[0][0]}.{parts[-1]}"
    return name


def _load_player_stats(years: list[int]) -> pd.DataFrame:
    """
    Fetch regular-season player stats via nflreadpy and return as a pandas
    DataFrame with normalized name columns ready for downstream joins.
    """
    nfl = _import_nflreadpy()
    df = nfl.load_player_stats(seasons=years, summary_level="reg").to_pandas()

    # player_display_name is the full name ("Bo Nix"); player_name is abbreviated ("B.Nix")
    # Drop the abbreviated column first to avoid duplicate column names after rename
    if "player_display_name" in df.columns:
        df = df.drop(columns=["player_name"], errors="ignore")
        df = df.rename(columns={"player_display_name": "player_name"})
    elif "player_name" not in df.columns:
        df["player_name"] = None

    df["player_name_norm"] = df["player_name"].fillna("").apply(normalise_name)
    df["name_key"] = df["player_name"].fillna("").apply(_make_name_key)
    return df


# ── Seasonal stats ─────────────────────────────────────────────────────────────
def _get_games_started(years: list[int]) -> pd.DataFrame:
    """
    Derive games_started and snap_pct from weekly offensive snap counts.

    - games_started: weeks where the player ran >= 50% of offensive snaps
    - snap_pct:      mean offensive snap share across all weeks with snaps

    Returns DataFrame with columns: player_id, season, games_started, snap_pct
    Falls back to empty DataFrame if snap data is unavailable.
    """
    nfl = _import_nflreadpy()
    try:
        snaps = nfl.load_snap_counts(seasons=years).to_pandas()
    except Exception as e:
        log.warning("load_snap_counts() failed — snap features unavailable: %s", e)
        return pd.DataFrame()

    if "offense_pct" not in snaps.columns:
        log.warning("offense_pct missing from snap data — snap features unavailable")
        return pd.DataFrame()

    snap_filter = "offense_snaps" if "offense_snaps" in snaps.columns else "offense_pct"
    snaps = snaps[snaps[snap_filter] > 0].copy()

    if snaps.empty or "player_id" not in snaps.columns:
        return pd.DataFrame()

    agg = (
        snaps.groupby(["player_id", "season"])
        .agg(
            games_started=("offense_pct", lambda x: int((x >= 0.5).sum())),
            snap_pct=("offense_pct", "mean"),
        )
        .reset_index()
    )

    log.info("  Snap count data: %d player-season rows", len(agg))
    return agg


def scrape_passing_seasons(years: Optional[list[int]] = None) -> pd.DataFrame:
    """QB passing stats for a range of seasons via nflreadpy."""
    _fix_ssl()
    if years is None:
        years = list(range(MIN_YEAR, MAX_YEAR + 1))

    log.info("Fetching passing stats %d-%d via nflreadpy...", min(years), max(years))
    df = _load_player_stats(years)
    df = df[df["attempts"] >= 1].copy()

    gs = _get_games_started(years)
    if not gs.empty:
        df = df.merge(gs, on=["player_id", "season"], how="left")

    rename = {
        "passing_yards":             "pass_yards",
        "passing_tds":               "pass_tds",
        "passing_interceptions":     "interceptions",
        "sacks_suffered":            "sacks_taken",
        "passing_air_yards":         "air_yards",
        "passing_yards_after_catch": "yac",
        "passing_first_downs":       "pass_first_downs",
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})

    if "completions" in df.columns and "attempts" in df.columns:
        df["completion_pct"] = df["completions"] / df["attempts"]
    if "pass_tds" in df.columns and "attempts" in df.columns:
        df["td_pct"] = df["pass_tds"] / df["attempts"]
    if "interceptions" in df.columns and "attempts" in df.columns:
        df["int_pct"] = df["interceptions"] / df["attempts"]
    if "attempts" in df.columns and "games" in df.columns:
        df["attempts_per_game"] = df["attempts"] / df["games"].replace(0, 1)
        df["is_starter"] = (df["attempts_per_game"] >= 20).astype(int)

    if "games" in df.columns:
        df["games_missed"] = df.apply(
            lambda r: max(0, (17 if r["season"] >= 2021 else 16) - int(r["games"])),
            axis=1,
        )

    save_raw(df, "pfr_passing")
    named = df["player_name"].notna().sum() if "player_name" in df.columns else 0
    log.info("  Passing rows: %d (named: %d)", len(df), named)
    return df


def scrape_rushing_seasons(years: Optional[list[int]] = None) -> pd.DataFrame:
    """RB/QB rushing stats for a range of seasons via nflreadpy."""
    _fix_ssl()
    if years is None:
        years = list(range(MIN_YEAR, MAX_YEAR + 1))

    log.info("Fetching rushing stats %d-%d via nflreadpy...", min(years), max(years))
    df = _load_player_stats(years)
    df = df[df["carries"] >= 1].copy()

    gs = _get_games_started(years)
    if not gs.empty:
        df = df.merge(gs, on=["player_id", "season"], how="left")

    rename = {
        "carries":             "rush_attempts",
        "rushing_yards":       "rush_yards",
        "rushing_tds":         "rush_tds",
        "rushing_fumbles":     "fumbles",
        "rushing_first_downs": "rush_first_downs",
        "rushing_epa":         "rush_epa",
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})

    if "rush_yards" in df.columns and "rush_attempts" in df.columns:
        df["yards_per_carry"] = df["rush_yards"] / df["rush_attempts"]

    if "games" in df.columns:
        df["games_missed"] = df.apply(
            lambda r: max(0, (17 if r["season"] >= 2021 else 16) - int(r["games"])),
            axis=1,
        )

    save_raw(df, "pfr_rushing")
    log.info("  Rushing rows: %d", len(df))
    return df


def scrape_receiving_seasons(years: Optional[list[int]] = None) -> pd.DataFrame:
    """WR/TE/RB receiving stats for a range of seasons via nflreadpy."""
    _fix_ssl()
    if years is None:
        years = list(range(MIN_YEAR, MAX_YEAR + 1))

    log.info("Fetching receiving stats %d-%d via nflreadpy...", min(years), max(years))
    df = _load_player_stats(years)
    df = df[df["targets"] >= 1].copy()

    gs = _get_games_started(years)
    if not gs.empty:
        df = df.merge(gs, on=["player_id", "season"], how="left")

    rename = {
        "receiving_yards":             "rec_yards",
        "receiving_tds":               "rec_tds",
        "receiving_air_yards":         "air_yards",
        "receiving_yards_after_catch": "yac",
        "receiving_fumbles":           "fumbles",
        "receiving_first_downs":       "rec_first_downs",
        "receiving_epa":               "rec_epa",
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})

    if "receptions" in df.columns and "targets" in df.columns:
        df["catch_rate"] = df["receptions"] / df["targets"]
    if "rec_yards" in df.columns and "receptions" in df.columns:
        df["yards_per_reception"] = df["rec_yards"] / df["receptions"]
    if "rec_yards" in df.columns and "targets" in df.columns:
        df["yards_per_target"] = df["rec_yards"] / df["targets"]
    if "targets" in df.columns and "games" in df.columns:
        df["targets_per_game"] = df["targets"] / df["games"].replace(0, 1)

    if "games" in df.columns:
        df["games_missed"] = df.apply(
            lambda r: max(0, (17 if r["season"] >= 2021 else 16) - int(r["games"])),
            axis=1,
        )

    save_raw(df, "pfr_receiving")
    log.info("  Receiving rows: %d", len(df))
    return df


# ── Multi-season wrapper ───────────────────────────────────────────────────────
def scrape_seasons(
    stat_type: str,
    years: Optional[list[int]] = None,
) -> pd.DataFrame:
    fn = {
        "passing":   scrape_passing_seasons,
        "rushing":   scrape_rushing_seasons,
        "receiving": scrape_receiving_seasons,
    }.get(stat_type)

    if fn is None:
        log.warning("Stat type '%s' not supported. Skipping.", stat_type)
        return pd.DataFrame()

    return fn(years)


# ── Feature builder: rolling 3-season window ──────────────────────────────────
def build_pre_contract_stats(
    stats_df: pd.DataFrame,
    signing_year: int,
    player_name_norm: str,
    window: int = 3,
    numeric_cols: Optional[list[str]] = None,
) -> dict:
    signing_year = int(signing_year)
    seasons = range(signing_year - window, signing_year)

    # Pass 1: exact normalised name match
    subset = stats_df[
        (stats_df["player_name_norm"] == player_name_norm) &
        (stats_df["season"].isin(seasons))
    ].sort_values("season")

    # Pass 2: abbreviated name key fallback (handles 'P.Mahomes' vs 'patrick mahomes')
    if subset.empty and "name_key" in stats_df.columns:
        key = _make_name_key(player_name_norm)
        subset = stats_df[
            (stats_df["name_key"] == key) &
            (stats_df["season"].isin(seasons))
        ].sort_values("season")

    if subset.empty:
        return {}

    # ── Gap Years (Recency penalty) ───────────────────────────────────────────
    last_active_season = int(subset["season"].max())
    features = {"gap_years": int((signing_year - 1) - last_active_season)}

    if numeric_cols is None:
        numeric_cols = subset.select_dtypes(include="number").columns.tolist()
        numeric_cols = [c for c in numeric_cols if c != "season"]
    for col in numeric_cols:
        vals = pd.to_numeric(subset[col], errors="coerce").dropna().values
        if len(vals) == 0:
            continue
        features[f"{col}_mean"]  = float(vals.mean())
        features[f"{col}_last"]  = float(vals[-1])
        features[f"{col}_max"]   = float(vals.max())
        features[f"{col}_games"] = int(len(vals))
        if len(vals) >= 2:
            xs = np.arange(len(vals), dtype=float)
            features[f"{col}_trend"] = float(np.polyfit(xs, vals, 1)[0])
        # Peak decline: fraction fallen from career-window best to most recent season.
        # 0.0 = still at peak, 1.0 = produced nothing last year vs. best year.
        if vals.max() > 0:
            features[f"{col}_peak_decline"] = float(
                (vals.max() - vals[-1]) / vals.max()
            )

    # ── Special Feature: Starter Seasons Count ────────────────────────────────
    # Use rate-based signals (attempts_per_game / is_starter / games_started)
    # so injury-shortened seasons still count if the player was clearly starting.
    last_season = int(subset["season"].max())
    last_row = subset[subset["season"] == last_season]

    if "is_starter" in subset.columns:
        # QB path: is_starter = (attempts_per_game >= 20), set during scrape
        is_starter_col = pd.to_numeric(subset["is_starter"], errors="coerce").fillna(0)
        features["starter_seasons"] = int((is_starter_col >= 1).sum())
        last_is_starter_vals = pd.to_numeric(last_row["is_starter"], errors="coerce").fillna(0)
        last_is_starter = int(last_is_starter_vals.values[0] >= 1) if len(last_is_starter_vals) else 1
    elif "games_started" in subset.columns:
        # Skill position path: started at least 4 games (rate-safe for short seasons)
        features["starter_seasons"] = int((subset["games_started"] >= 4).sum())
        last_gs = pd.to_numeric(last_row["games_started"], errors="coerce").fillna(0)
        last_is_starter = int(last_gs.values[0] >= 4) if len(last_gs) else 1
    elif "games" in subset.columns:
        features["starter_seasons"] = int((subset["games"] >= 8).sum())
        last_g = pd.to_numeric(last_row["games"], errors="coerce").fillna(0)
        last_is_starter = int(last_g.values[0] >= 8) if len(last_g) else 1
    else:
        last_is_starter = 1

    # ── Recent Demotion Flag ──────────────────────────────────────────────────
    # Fires when a player was a starter for 2+ of the last 3 seasons but their
    # most recent season shows they lost the starting job.
    features["recent_demotion"] = int(
        features.get("starter_seasons", 0) >= 2 and last_is_starter == 0
    )

    return features


# ── CLI ────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--stat-types", nargs="+", default=["passing", "rushing", "receiving"])
    parser.add_argument("--start-year", type=int, default=MIN_YEAR)
    parser.add_argument("--end-year",   type=int, default=MAX_YEAR)
    args = parser.parse_args()

    years = list(range(args.start_year, args.end_year + 1))
    for stat_type in args.stat_types:
        df = scrape_seasons(stat_type, years)
        print(df[["player_name", "season"]].head(5).to_string())
