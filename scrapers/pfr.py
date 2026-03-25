"""
Player stats via nfl_data_py
=============================
nfl_data_py pulls from the official NFL data repository — no scraping,
no 403s, and cleaner data than PFR for our purposes.

Install: pip install nfl_data_py certifi

Key notes:
  - import_seasonal_data() has no player_name — only player_id
  - import_ids() provides player_id -> player_display_name mapping
  - import_weekly_data() also has player_display_name if ids join fails
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
MAX_YEAR = 2024


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


def _import_nfl_data_py():
    try:
        import nfl_data_py as nfl
        return nfl
    except ImportError:
        raise ImportError("nfl_data_py is not installed. Run: pip install nfl_data_py")


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


def _get_player_id_map() -> pd.DataFrame:
    """
    Return player_id -> player_name + position via import_ids().
    import_ids() columns confirmed: gsis_id, name, position
    Falls back to empty DataFrame if unavailable.
    """
    nfl = _import_nfl_data_py()
    try:
        ids = nfl.import_ids()

        # Use exact column names confirmed from import_ids() schema
        id_col  = "gsis_id"   if "gsis_id"  in ids.columns else None
        name_col = "name"     if "name"      in ids.columns else None
        pos_col  = "position" if "position"  in ids.columns else None

        # Fallback fuzzy search if exact names not found
        if not id_col:
            id_col = next((c for c in ids.columns if "gsis" in c.lower()), None)
        if not name_col:
            # Prefer "name" over "merge_name" — match exact word only
            name_col = next((c for c in ids.columns if c.lower() == "name"), None)
            name_col = name_col or next((c for c in ids.columns if "display" in c.lower()), None)

        if not id_col or not name_col:
            log.warning("import_ids() columns unexpected: %s", ids.columns.tolist())
            return pd.DataFrame()

        keep = [id_col, name_col] + ([pos_col] if pos_col else [])
        result = ids[keep].dropna(subset=[id_col, name_col]).drop_duplicates(id_col)
        result = result.rename(columns={id_col: "player_id", name_col: "player_name"})
        if pos_col:
            result = result.rename(columns={pos_col: "position"})
        result["player_name_norm"] = result["player_name"].apply(normalise_name)
        result["name_key"] = result["player_name"].apply(_make_name_key)
        log.info("  Loaded %d player id mappings", len(result))
        return result

    except Exception as e:
        log.warning("import_ids() failed: %s", e)
        return pd.DataFrame()


def _add_player_names(df: pd.DataFrame, id_map: pd.DataFrame) -> pd.DataFrame:
    """
    Merge player names onto a seasonal stats DataFrame via player_id.
    Two-pass strategy:
      1. Join on player_id directly (exact)
      2. For any remaining nulls, try matching nfl_data_py's abbreviated
         display name (e.g. 'P.Mahomes') against our name_key index
    """
    if id_map.empty or "player_id" not in df.columns:
        df["player_name"] = None
        df["player_name_norm"] = None
        return df

    cols = ["player_id", "player_name", "player_name_norm", "name_key"]
    cols = [c for c in cols if c in id_map.columns]
    df = df.merge(id_map[cols], on="player_id", how="left")

    # Pass 2: for rows still missing a name, try matching on the abbreviated
    # display_name that nfl_data_py puts in other fields (if present)
    if "player_display_name" in df.columns and "name_key" in id_map.columns:
        missing = df["player_name"].isna()
        if missing.any():
            df.loc[missing, "_abbr_key"] = df.loc[missing, "player_display_name"].apply(_make_name_key)
            key_map = id_map.set_index("name_key")[["player_name", "player_name_norm"]]
            filled = df.loc[missing, "_abbr_key"].map(key_map["player_name"])
            filled_norm = df.loc[missing, "_abbr_key"].map(key_map["player_name_norm"])
            df.loc[missing, "player_name"] = filled
            df.loc[missing, "player_name_norm"] = filled_norm
            df = df.drop(columns=["_abbr_key"], errors="ignore")

    return df


# ── Seasonal stats ─────────────────────────────────────────────────────────────
def _get_games_started(years: list[int], id_map: pd.DataFrame) -> pd.DataFrame:
    """
    Derive games_started and snap_pct from weekly offensive snap counts.

    - games_started: weeks where the player ran >= 50% of offensive snaps
    - snap_pct:      mean offensive snap share across all weeks with snaps

    Returns DataFrame with columns: player_id, season, games_started, snap_pct
    Falls back to empty DataFrame if weekly data is unavailable.
    """
    nfl = _import_nfl_data_py()
    try:
        weekly = nfl.import_weekly_data(years)
    except Exception as e:
        log.warning("import_weekly_data() failed — snap features unavailable: %s", e)
        return pd.DataFrame()

    if "offense_pct" not in weekly.columns:
        log.warning("offense_pct missing from weekly data — snap features unavailable")
        return pd.DataFrame()

    snap_col = "offense_snaps" if "offense_snaps" in weekly.columns else None
    if snap_col:
        weekly = weekly[weekly[snap_col] > 0].copy()
    else:
        weekly = weekly[weekly["offense_pct"] > 0].copy()

    if weekly.empty or "player_id" not in weekly.columns:
        return pd.DataFrame()

    agg = (
        weekly.groupby(["player_id", "season"])
        .agg(
            games_started=("offense_pct", lambda x: int((x >= 0.5).sum())),
            snap_pct=("offense_pct", "mean"),
        )
        .reset_index()
    )

    log.info("  Weekly snap data: %d player-season rows", len(agg))
    return agg


def scrape_passing_seasons(years: Optional[list[int]] = None) -> pd.DataFrame:
    """QB passing stats for a range of seasons via nfl_data_py."""
    _fix_ssl()
    nfl = _import_nfl_data_py()
    if years is None:
        years = list(range(MIN_YEAR, MAX_YEAR + 1))

    log.info("Fetching passing stats %d-%d via nfl_data_py...", min(years), max(years))
    df = nfl.import_seasonal_data(years, s_type="REG")
    df = df[df["attempts"] >= 1].copy()

    id_map = _get_player_id_map()
    df = _add_player_names(df, id_map)
    df["name_key"] = df["player_name_norm"].apply(_make_name_key)

    # Add games_started from weekly data
    gs = _get_games_started(years, id_map)
    if not gs.empty:
        df = df.merge(gs, on=["player_id", "season"], how="left")

    rename = {
        "passing_yards":             "pass_yards",
        "passing_tds":               "pass_tds",
        "sacks":                     "sacks_taken",
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

    # Starter proxy — attempts per game. Starters typically average 25+ per game.
    # This lets the model learn backup vs starter market value.
    if "attempts" in df.columns and "games" in df.columns:
        df["attempts_per_game"] = df["attempts"] / df["games"].replace(0, 1)
        df["is_starter"] = (df["attempts_per_game"] >= 20).astype(int)

    save_raw(df, "pfr_passing")
    named = df["player_name"].notna().sum() if "player_name" in df.columns else 0
    log.info("  Passing rows: %d (named: %d)", len(df), named)
    return df


def scrape_rushing_seasons(years: Optional[list[int]] = None) -> pd.DataFrame:
    """RB/QB rushing stats for a range of seasons via nfl_data_py."""
    _fix_ssl()
    nfl = _import_nfl_data_py()
    if years is None:
        years = list(range(MIN_YEAR, MAX_YEAR + 1))

    log.info("Fetching rushing stats %d-%d via nfl_data_py...", min(years), max(years))
    df = nfl.import_seasonal_data(years, s_type="REG")
    df = df[df["carries"] >= 1].copy()

    id_map = _get_player_id_map()
    df = _add_player_names(df, id_map)
    df["name_key"] = df["player_name_norm"].apply(_make_name_key)

    gs = _get_games_started(years, id_map)
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

    save_raw(df, "pfr_rushing")
    log.info("  Rushing rows: %d", len(df))
    return df


def scrape_receiving_seasons(years: Optional[list[int]] = None) -> pd.DataFrame:
    """WR/TE/RB receiving stats for a range of seasons via nfl_data_py."""
    _fix_ssl()
    nfl = _import_nfl_data_py()
    if years is None:
        years = list(range(MIN_YEAR, MAX_YEAR + 1))

    log.info("Fetching receiving stats %d-%d via nfl_data_py...", min(years), max(years))
    df = nfl.import_seasonal_data(years, s_type="REG")
    df = df[df["targets"] >= 1].copy()

    id_map = _get_player_id_map()
    df = _add_player_names(df, id_map)
    df["name_key"] = df["player_name_norm"].apply(_make_name_key)

    gs = _get_games_started(years, id_map)
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
        "target_share":                "target_share",
        "air_yards_share":             "air_yards_share",
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
    # How many years since the player's last active season in this window?
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
    # explicitly count seasons with starter-level volume
    last_season = int(subset["season"].max())
    last_row = subset[subset["season"] == last_season]

    if "attempts" in subset.columns:
        features["starter_seasons"] = int((subset["attempts"] >= 250).sum())
        last_attempts = pd.to_numeric(last_row["attempts"], errors="coerce").fillna(0)
        last_is_starter = int(last_attempts.values[0] >= 250) if len(last_attempts) else 1
    elif "games_started" in subset.columns:
        features["starter_seasons"] = int((subset["games_started"] >= 8).sum())
        last_gs = pd.to_numeric(last_row["games_started"], errors="coerce").fillna(0)
        last_is_starter = int(last_gs.values[0] >= 8) if len(last_gs) else 1
    elif "games" in subset.columns:
        # Fallback if games_started missing: >8 games played is roughly starter territory
        features["starter_seasons"] = int((subset["games"] >= 8).sum())
        last_g = pd.to_numeric(last_row["games"], errors="coerce").fillna(0)
        last_is_starter = int(last_g.values[0] >= 8) if len(last_g) else 1
    else:
        last_is_starter = 1  # can't determine

    # ── Recent Demotion Flag ──────────────────────────────────────────────────
    # Fires when a player was a starter for 2+ of the last 3 seasons but their
    # most recent season shows they lost the starting job. This is the key signal
    # for players like Kyler Murray who need to take below-market-rate deals.
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
