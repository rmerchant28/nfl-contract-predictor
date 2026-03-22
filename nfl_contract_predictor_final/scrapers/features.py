"""
Feature builder
===============
Joins raw contract data (OTC) with player stats (PFR) to produce a
model-ready dataset. One row = one contract signing.

Target variable:  apy_pct_cap  (APY as % of salary cap in the signing year)

For each contract, stats are aggregated over the 3 seasons prior to signing:
  - mean, last-season value, and trend (slope) for every numeric stat
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .utils import (
    load_raw, save_processed, normalise_name,
    POSITION_GROUP, PROJECT_ROOT,
)
from .pfr import build_pre_contract_stats

log = logging.getLogger(__name__)

WINDOW = 3   # seasons of stats to look back


# ── Helpers ────────────────────────────────────────────────────────────────────
# Columns that must never be coerced to numeric
_KEEP_AS_STR = {"player_name", "player_name_norm", "name_key", "team",
                "position", "position_raw", "position_group"}

def _numeric(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce object columns that look numeric to float, preserving string identity cols."""
    for col in df.select_dtypes(include="object").columns:
        if col in _KEEP_AS_STR:
            continue
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _age_at_signing(contracts: pd.DataFrame) -> pd.DataFrame:
    """
    Compute player age at time of signing if birth year is available.
    PFR player pages carry DOB — for now we leave this as a placeholder
    and populate it after player-page scraping.
    """
    if "birth_year" in contracts.columns:
        contracts["age_at_signing"] = contracts["signing_year"] - contracts["birth_year"]
    else:
        contracts["age_at_signing"] = np.nan
    return contracts


# ── QB feature set ─────────────────────────────────────────────────────────────
QB_STAT_COLS = [
    "completions", "attempts", "completion_pct",
    "pass_yards", "pass_tds", "interceptions",
    "td_pct", "int_pct",
    "sacks_taken", "air_yards", "yac",
    "pass_first_downs", "passing_epa",
    "dakota", "pacr",
    "carries", "rushing_yards", "rushing_tds",
    "games",
]


def build_qb_features(contracts: pd.DataFrame, passing_df: pd.DataFrame) -> pd.DataFrame:
    passing_df = _numeric(passing_df)
    # Only keep stat cols that actually exist in this DataFrame
    available_cols = [c for c in QB_STAT_COLS if c in passing_df.columns]
    rows = []

    for _, contract in contracts[contracts["position_group"] == "QB"].iterrows():
        feats = build_pre_contract_stats(
            passing_df,
            signing_year     = int(contract["signing_year"]),
            player_name_norm = contract["player_name_norm"],
            window           = WINDOW,
            numeric_cols     = available_cols,
        )
        if not feats:
            log.debug("No stats found for QB %s (signing %d)",
                      contract["player_name"], contract["signing_year"])
        row = {**contract.to_dict(), **feats}
        rows.append(row)

    return pd.DataFrame(rows)


# ── Skill position (WR / RB / TE) feature set ─────────────────────────────────
WR_STAT_COLS = [
    "targets", "receptions", "rec_yards",
    "yards_per_reception", "yards_per_target",
    "rec_tds", "catch_rate", "rec_first_downs",
    "target_share", "air_yards_share",
    "receiving_epa", "racr", "wopr_x",
    "games",
]

RB_STAT_COLS = [
    "rush_attempts", "rush_yards", "yards_per_carry",
    "rush_tds", "rush_first_downs", "fumbles",
    "rushing_epa",
    "targets", "receptions", "rec_yards", "rec_tds",
    "games",
]

TE_STAT_COLS = [
    "targets", "receptions", "rec_yards",
    "yards_per_reception", "yards_per_target",
    "rec_tds", "catch_rate",
    "receiving_epa", "target_share",
    "games",
]


def build_skill_features(
    contracts: pd.DataFrame,
    receiving_df: pd.DataFrame,
    rushing_df: pd.DataFrame,
) -> pd.DataFrame:
    receiving_df = _numeric(receiving_df)
    rushing_df   = _numeric(rushing_df)

    avail_recv  = set(receiving_df.columns)
    avail_rush  = set(rushing_df.columns)
    wr_cols     = [c for c in WR_STAT_COLS if c in avail_recv]
    rb_rush_cols= [c for c in RB_STAT_COLS if c in avail_rush]
    rb_recv_cols= [c for c in ["targets","receptions","rec_yards","rec_tds"] if c in avail_recv]
    te_cols     = [c for c in TE_STAT_COLS if c in avail_recv]
    rows = []

    skill_contracts = contracts[contracts["position_group"] == "SKILL"]

    for _, contract in skill_contracts.iterrows():
        pos = contract.get("position", "")

        # Choose stat source and columns based on sub-position
        if pos == "RB":
            rush_feats = build_pre_contract_stats(
                rushing_df, int(contract["signing_year"]),
                contract["player_name_norm"], WINDOW, rb_rush_cols,
            )
            recv_feats = build_pre_contract_stats(
                receiving_df, int(contract["signing_year"]),
                contract["player_name_norm"], WINDOW, rb_recv_cols,
            )
            recv_feats = {f"rcv_{k}": v for k, v in recv_feats.items()}
            feats = {**rush_feats, **recv_feats}

        elif pos == "WR":
            feats = build_pre_contract_stats(
                receiving_df, int(contract["signing_year"]),
                contract["player_name_norm"], WINDOW, wr_cols,
            )

        elif pos == "TE":
            feats = build_pre_contract_stats(
                receiving_df, int(contract["signing_year"]),
                contract["player_name_norm"], WINDOW, te_cols,
            )
        else:
            feats = {}

        if not feats:
            log.debug("No stats for %s %s (signing %d)",
                      pos, contract["player_name"], contract["signing_year"])

        row = {**contract.to_dict(), **feats}
        rows.append(row)

    return pd.DataFrame(rows)


# ── OL feature set ─────────────────────────────────────────────────────────────
OL_STAT_COLS = [
    "games", "games_started",
    "approx_value",
    # penalty count comes from the OL penalty scrape
    "penalty_count",
]


def build_ol_features(
    contracts: pd.DataFrame,
    ol_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    OL features are limited from PFR alone.
    Penalty counts are aggregated per player-season from the raw OL data.
    """
    ol_df = _numeric(ol_df)

    # Aggregate penalty counts per player per season
    if "player_name_norm" in ol_df.columns and "season" in ol_df.columns:
        pen_counts = (
            ol_df.groupby(["player_name_norm", "season"])
            .size()
            .reset_index(name="penalty_count")
        )
    else:
        pen_counts = pd.DataFrame(columns=["player_name_norm", "season", "penalty_count"])

    rows = []
    for _, contract in contracts[contracts["position_group"] == "OL"].iterrows():
        feats = build_pre_contract_stats(
            pen_counts, contract["signing_year"],
            contract["player_name_norm"], WINDOW, ["penalty_count"],
        )
        row = {**contract.to_dict(), **feats}
        rows.append(row)

    return pd.DataFrame(rows)


# ── Market context features ────────────────────────────────────────────────────
def add_market_context(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add positional market context features.
    Skips gracefully if position or signing_year columns are missing/empty.
    """
    if df.empty:
        return df

    # Ensure required columns exist
    for col in ["position", "signing_year", "apy_pct_cap"]:
        if col not in df.columns:
            log.warning("add_market_context: missing column '%s', skipping context features.", col)
            return df

    # Drop rows where groupby keys are null
    df = df.copy()
    valid = df[df["position"].notna() & df["signing_year"].notna()]
    if valid.empty:
        log.warning("add_market_context: no rows with both position and signing_year populated.")
        return df

    out_frames = []
    for (pos, yr), group in valid.groupby(["position", "signing_year"]):
        group = group.copy()

        # Rank within position+year by APY (1 = highest paid)
        group["peer_rank"] = group["apy"].rank(ascending=False, method="min").astype("Int64")

        # Top contract at this position this year
        top_cap_pct = group["apy_pct_cap"].max()
        group["top_contract_pct"] = top_cap_pct

        out_frames.append(group)

    if not out_frames:
        return df

    result = pd.concat(out_frames, ignore_index=True)

    # Years since last record contract per position (rolling max of top_contract_pct)
    result = result.sort_values(["position", "signing_year"])
    result["years_since_reset"] = np.nan

    for pos, grp in result.groupby("position"):
        grp = grp.sort_values("signing_year")
        running_max = 0.0
        last_reset  = None

        for idx, row in grp.iterrows():
            if row["apy_pct_cap"] and row["apy_pct_cap"] > running_max:
                running_max = row["apy_pct_cap"]
                last_reset  = row["signing_year"]
            if last_reset is not None:
                result.loc[idx, "years_since_reset"] = row["signing_year"] - last_reset

    return result


# ── Master builder ─────────────────────────────────────────────────────────────
def build_dataset(
    positions: Optional[list[str]] = None,
    min_year: int = 2013,
    max_year: int = 2024,
) -> pd.DataFrame:
    """
    Load raw data and produce the model-ready feature matrix.

    Args:
        positions: Position groups to include. Defaults to all.
        min_year:  Earliest signing year to include.
        max_year:  Latest signing year to include.

    Returns:
        DataFrame where each row is a contract with full feature set.
    """
    log.info("Loading raw data...")
    contracts = load_raw("contracts_with_cap_pct")
    passing   = load_raw("pfr_passing")   if positions is None or "QB"    in (positions or []) else pd.DataFrame()
    rushing   = load_raw("pfr_rushing")   if positions is None or "SKILL" in (positions or []) else pd.DataFrame()
    receiving = load_raw("pfr_receiving") if positions is None or "SKILL" in (positions or []) else pd.DataFrame()
    ol        = load_raw("pfr_ol")        if positions is None or "OL"    in (positions or []) else pd.DataFrame()

    # Filter contracts
    contracts = contracts[
        contracts["signing_year"].between(min_year, max_year) &
        contracts["apy_pct_cap"].notna() &
        contracts["position_group"].notna()
    ].copy()

    # Deduplicate — keep highest APY row per (player, position, signing_year)
    # Duplicates come from OTC listing restructures/amendments as separate entries
    before = len(contracts)
    contracts = (
        contracts
        .sort_values("apy", ascending=False)
        .drop_duplicates(subset=["player_name_norm", "position", "signing_year"], keep="first")
        .reset_index(drop=True)
    )
    if len(contracts) < before:
        log.info("Deduplicated contracts: %d → %d rows", before, len(contracts))

    if positions:
        contracts = contracts[contracts["position_group"].isin(positions)]

    contracts = _age_at_signing(contracts)
    log.info("Contracts in scope: %d", len(contracts))

    # Build per-group feature sets
    frames = []
    if not passing.empty:
        log.info("Building QB features...")
        frames.append(build_qb_features(contracts, passing))

    if not rushing.empty or not receiving.empty:
        log.info("Building skill position features...")
        frames.append(build_skill_features(contracts, receiving, rushing))

    if not ol.empty:
        log.info("Building OL features...")
        frames.append(build_ol_features(contracts, ol))

    if not frames:
        log.warning("No feature frames built — check that raw data exists.")
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)
    combined = add_market_context(combined)

    # Drop columns unlikely to be useful as model inputs
    drop_cols = ["source_url", "position_raw", "player_name_norm", "year"]
    combined  = combined.drop(columns=[c for c in drop_cols if c in combined.columns])

    save_processed(combined, "model_features")
    log.info("Final dataset: %d rows × %d columns", *combined.shape)
    return combined


# ── CLI ────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    df = build_dataset()
    print(df.info())
    print(df[["player_name", "position", "signing_year",
              "apy_pct_cap", "peer_rank"]].head(10).to_string())
