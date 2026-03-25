#!/usr/bin/env python3
"""
Master runner — NFL Contract Predictor data pipeline
=====================================================
Run this script to execute the full data collection and feature-building pipeline.

Usage:
    # Full pipeline (all positions, all years)
    python run_pipeline.py

    # Only scrape contracts + cap data (skip PFR stats)
    python run_pipeline.py --contracts-only

    # Specific positions and year range
    python run_pipeline.py --positions QB WR --start-year 2015 --end-year 2023

    # Skip scraping and just rebuild the feature dataset from cached raw data
    python run_pipeline.py --features-only
"""

import argparse
import logging
import sys
import time
from pathlib import Path

# Ensure the project root is on the path regardless of where the script is invoked from
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from scrapers.overthecap import scrape_cap_history, scrape_contracts, add_cap_percentage
from scrapers.pfr import scrape_seasons, MIN_YEAR, MAX_YEAR
from scrapers.features import build_dataset
from scrapers.utils import save_raw

log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="NFL contract prediction — data pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--positions", nargs="*",
        default=None,
        help="OTC position codes to scrape (e.g. QB WR RB TE OT OG C). Default: all.",
    )
    parser.add_argument(
        "--start-year", type=int, default=MIN_YEAR,
        help=f"First season year for PFR stats (default: {MIN_YEAR})",
    )
    parser.add_argument(
        "--end-year", type=int, default=MAX_YEAR,
        help=f"Last season year for PFR stats (default: {MAX_YEAR})",
    )
    parser.add_argument(
        "--contracts-only", action="store_true",
        help="Only scrape OTC contracts and cap history; skip PFR stats.",
    )
    parser.add_argument(
        "--features-only", action="store_true",
        help="Skip all scraping; just rebuild features from existing raw CSV files.",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Bypass local page cache and re-fetch all pages from source sites.",
    )
    return parser.parse_args()


def step(label: str):
    log.info("")
    log.info("=" * 60)
    log.info("  %s", label)
    log.info("=" * 60)


def main():
    args = parse_args()
    t0   = time.time()

    # ── Position group mapping for PFR ────────────────────────────────────────
    # OTC positions → PFR stat types needed
    pos_to_stat_type = {
        "QB": ["passing", "rushing"],
        "WR": ["receiving"],
        "RB": ["rushing", "receiving"],
        "TE": ["receiving"],
        "OT": ["ol"],
        "OG": ["ol"],
        "C":  ["ol"],
    }

    positions     = args.positions  # None = all
    # Stats need to go back WINDOW=3 years before the earliest contract signing year
    # so a 2015 contract can find 2012-2014 stats
    years         = list(range(args.start_year - 3, args.end_year + 1))

    if args.features_only:
        step("Building feature dataset from existing raw files")
        df = build_dataset()
        log.info("Done. %d rows, %d features.", *df.shape)
        return

    # ── Step 1: OTC cap history ────────────────────────────────────────────────
    step("1 / 4 — Salary cap history (OverTheCap)")
    cap_df = scrape_cap_history()
    if cap_df.empty:
        log.error("Could not scrape cap history. Aborting.")
        sys.exit(1)

    # ── Step 2: OTC contracts ─────────────────────────────────────────────────
    step("2 / 4 — Player contracts (OverTheCap)")
    contracts_df = scrape_contracts(positions)
    if contracts_df.empty:
        log.error("No contracts scraped. Check your internet connection or OTC page structure.")
        sys.exit(1)

    enriched_df = add_cap_percentage(contracts_df, cap_df)
    save_raw(enriched_df, "contracts_with_cap_pct")

    log.info("Contracts scraped:      %d", len(enriched_df))
    log.info("Contracts with cap %%:  %d", enriched_df["apy_pct_cap"].notna().sum())

    if args.contracts_only:
        log.info("--contracts-only flag set. Stopping after OTC.")
        log.info("Total time: %.0fs", time.time() - t0)
        return

    # ── Step 3: PFR stats ─────────────────────────────────────────────────────
    step("3 / 4 — Player statistics (Pro Football Reference)")

    # Determine which stat types are needed
    if positions is None:
        stat_types_needed = {"passing", "rushing", "receiving", "ol"}
    else:
        stat_types_needed = set()
        for pos in positions:
            stat_types_needed.update(pos_to_stat_type.get(pos, []))

    log.info("Stat types to scrape: %s", ", ".join(sorted(stat_types_needed)))

    for stat_type in sorted(stat_types_needed):
        log.info("--- %s ---", stat_type)
        scrape_seasons(stat_type, years)
        # Extra pause between stat types to be courteous to PFR
        time.sleep(5)

    # ── Step 4: Feature engineering ───────────────────────────────────────────
    step("4 / 4 — Building feature dataset")

    # Map OTC positions to position_group strings
    group_filter = None
    if positions is not None:
        group_map = {"QB": "QB", "WR": "SKILL", "RB": "SKILL", "TE": "SKILL",
                     "OT": "OL", "OG": "OL", "C": "OL"}
        group_filter = list({group_map[p] for p in positions if p in group_map})

    df = build_dataset(
        positions=group_filter,
        min_year=args.start_year,
        max_year=args.end_year,
    )

    elapsed = time.time() - t0
    log.info("")
    log.info("Pipeline complete in %.0fs", elapsed)
    log.info("Final dataset: %d rows × %d columns", *df.shape)
    log.info("Output: data/processed/model_features.csv")
    log.info("")
    log.info("Sample (target variable distribution):")
    if "apy_pct_cap" in df.columns:
        log.info("  mean  = %.3f%%", df["apy_pct_cap"].mean() * 100)
        log.info("  max   = %.3f%%", df["apy_pct_cap"].max()  * 100)
        log.info("  by position:")
        for pos, grp in df.groupby("position_group"):
            log.info("    %s  mean=%.3f%%  n=%d",
                     pos, grp["apy_pct_cap"].mean() * 100, len(grp))


WINDOW = 3  # keep consistent with features.py

if __name__ == "__main__":
    main()
