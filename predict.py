"""
predict.py
==========
Inference module — loads trained models and predicts contract value for a player.
Used by the Streamlit app and can be called directly for ad-hoc predictions.

Usage:
    from predict import predict_contract

    result = predict_contract("Justin Jefferson", position="WR", signing_year=2025)
    print(result)
"""

import json
import logging
import ssl
import urllib.request
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import joblib

log = logging.getLogger(__name__)

ROOT        = Path(__file__).resolve().parent
MODELS_DIR  = ROOT / "models"
CAP_HISTORY = {
    2014: 133_000_000, 2015: 143_280_000, 2016: 155_270_000,
    2017: 167_000_000, 2018: 177_200_000, 2019: 188_200_000,
    2020: 198_200_000, 2021: 182_500_000, 2022: 208_200_000,
    2023: 224_800_000, 2024: 255_400_000, 2025: 279_000_000,
    2026: 301_200_000,
}
WINDOW = 3


# ── SSL fix ────────────────────────────────────────────────────────────────────
def _fix_ssl():
    try:
        import certifi
        ctx = ssl.create_default_context(cafile=certifi.where())
        opener = urllib.request.build_opener(urllib.request.HTTPSHandler(context=ctx))
        urllib.request.install_opener(opener)
    except ImportError:
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


# ── Model loading ──────────────────────────────────────────────────────────────
_model_cache: dict = {}
_manifest_cache: dict = {}


def load_model(position: str) -> tuple:
    """Load the best trained model and its feature manifest for a position."""
    if position in _model_cache:
        return _model_cache[position], _manifest_cache[position]

    # 1. Try to identify the specific best model from results_summary.json
    results_path = MODELS_DIR / "results_summary.json"
    target_model_file = None

    if results_path.exists():
        try:
            with open(results_path) as f:
                summary = json.load(f)
            best_name = summary.get("best_models", {}).get(position)
            if best_name:
                log.info("Best model for %s according to summary: %s", position, best_name)
                target_model_file = MODELS_DIR / f"{position}_{best_name.lower()}.pkl"
        except Exception as e:
            log.warning("Could not read results_summary.json: %s", e)

    # 2. Use specific model if found, else fallback to latest modified file
    if target_model_file and target_model_file.exists():
        model_path = target_model_file
    else:
        model_files = list(MODELS_DIR.glob(f"{position}_*.pkl"))
        if not model_files:
            raise FileNotFoundError(
                f"No model found for {position} in {MODELS_DIR}. "
                "Run notebooks/model.py first."
            )
        # Sort by mtime descending (newest first)
        model_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        model_path = model_files[0]

        if target_model_file:
            log.warning("Expected model %s not found. Falling back to %s.",
                        target_model_file.name, model_path.name)

    log.info("Loading model file: %s", model_path.name)
    manifest_path = MODELS_DIR / f"{model_path.stem}_manifest.json"

    model = joblib.load(model_path)
    with open(manifest_path) as f:
        manifest = json.load(f)

    _model_cache[position]   = model
    _manifest_cache[position] = manifest
    log.info("Loaded %s model for %s", manifest["model_name"], position)
    return model, manifest


def list_available_positions() -> list[str]:
    """Return positions that have trained models available."""
    return sorted({p.stem.split("_")[0] for p in MODELS_DIR.glob("*.pkl")})


# ── Name helpers ───────────────────────────────────────────────────────────────
def _normalise(name: str) -> str:
    import unicodedata
    name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode("ascii")
    return name.strip().lower()


def _name_key(name: str) -> str:
    """'Patrick Mahomes' → 'p.mahomes'"""
    name = _normalise(name).replace(".", " ")
    parts = name.split()
    if len(parts) >= 2:
        return f"{parts[0][0]}.{parts[-1]}"
    return name


# ── nfl_data_py cache (download parquet files once per session) ────────────────
# NOTE: nfl_data_py has install issues on some Python versions.
# To re-enable live stat fetching, uncomment this block and the
# corresponding code in _fetch_player_stats below.
#
# _stats_cache: dict = {}
# _ids_cache = None
#
# def _get_seasonal_data(years: list[int]) -> pd.DataFrame:
#     global _stats_cache
#     key = tuple(sorted(years))
#     if key not in _stats_cache:
#         _fix_ssl()
#         nfl = _import_nfl_data_py()
#         _stats_cache[key] = nfl.import_seasonal_data(list(years), s_type="REG")
#     return _stats_cache[key]
#
# def _get_ids() -> pd.DataFrame:
#     global _ids_cache
#     if _ids_cache is None:
#         _fix_ssl()
#         nfl = _import_nfl_data_py()
#         try:
#             ids = nfl.import_ids()
#             id_col   = next((c for c in ids.columns if "gsis" in c.lower()), None)
#             name_col = next((c for c in ids.columns if "display" in c.lower() or "name" in c.lower()), None)
#             if id_col and name_col:
#                 ids = ids[[id_col, name_col]].dropna().drop_duplicates(id_col)
#                 ids.columns = ["player_id", "player_name"]
#                 ids["player_name_norm"] = ids["player_name"].apply(_normalise)
#                 ids["name_key"]         = ids["player_name"].apply(_name_key)
#                 _ids_cache = ids
#             else:
#                 _ids_cache = pd.DataFrame()
#         except Exception as e:
#             log.warning("Could not load player ids: %s", e)
#             _ids_cache = pd.DataFrame()
#     return _ids_cache


# ── CSV-based stats cache (reads from data/raw/ — no network calls) ───────────
_csv_stats_cache: dict = {}


def _load_stats_csv(stat_type: str) -> pd.DataFrame:
    """Load a stats CSV from data/raw/, cached after first read."""
    global _csv_stats_cache
    if stat_type not in _csv_stats_cache:
        path = ROOT / "data" / "raw" / f"pfr_{stat_type}.csv"
        if path.exists():
            df = pd.read_csv(path)
            df["season"] = pd.to_numeric(df["season"], errors="coerce")
            _csv_stats_cache[stat_type] = df
        else:
            log.warning("Stats file not found: %s — run the pipeline first.", path.name)
            _csv_stats_cache[stat_type] = pd.DataFrame()
    return _csv_stats_cache[stat_type]


def _get_stats_for_position(position: str) -> pd.DataFrame:
    """Return the right stats DataFrame for a given position."""
    if position == "QB":
        return _load_stats_csv("passing")
    elif position in ("WR", "TE"):
        return _load_stats_csv("receiving")
    elif position == "RB":
        rushing   = _load_stats_csv("rushing")
        receiving = _load_stats_csv("receiving")
        if rushing.empty:
            return receiving
        if receiving.empty:
            return rushing
        return pd.concat([rushing, receiving], ignore_index=True)
    return pd.DataFrame()


# ── Stats fetcher ──────────────────────────────────────────────────────────────
def _fetch_player_stats(
    player_name: str,
    signing_year: int,
    position: str,
) -> dict:
    """
    Look up player stats from cached CSV files for the 3 seasons before
    signing_year. Reads from data/raw/pfr_*.csv — no network calls.

    To switch back to live nfl_data_py fetching, uncomment the
    _get_seasonal_data / _get_ids block above and replace this function
    body with the commented-out version below.
    """
    seasons = list(range(signing_year - WINDOW, signing_year))
    player_norm = _normalise(player_name)
    player_key  = _name_key(player_name)

    stats = _get_stats_for_position(position)
    if stats.empty:
        log.warning("No stats CSV for %s — run the pipeline first.", position)
        return {}

    if "player_name_norm" not in stats.columns:
        log.warning("player_name_norm column missing from stats CSV.")
        return {}

    # Match on normalised name or abbreviated name key
    mask = stats["player_name_norm"] == player_norm
    if "name_key" in stats.columns:
        mask = mask | (stats["name_key"] == player_key)

    player_stats = stats[mask & stats["season"].isin(seasons)].copy()

    if player_stats.empty:
        log.warning("No stats found for %s in seasons %s", player_name, seasons)
        return {}

    numeric_cols = player_stats.select_dtypes(include="number").columns.tolist()
    numeric_cols = [c for c in numeric_cols if c != "season"]

    # ── Gap Years ─────────────────────────────────────────────────────────────
    last_active_season = int(player_stats["season"].max())
    features = {"gap_years": int((signing_year - 1) - last_active_season)}

    player_stats = player_stats.sort_values("season")

    for col in numeric_cols:
        vals = pd.to_numeric(player_stats[col], errors="coerce").dropna().values
        if len(vals) == 0:
            continue
        features[f"{col}_mean"]  = float(vals.mean())
        features[f"{col}_last"]  = float(vals[-1])
        features[f"{col}_max"]   = float(vals.max())
        features[f"{col}_games"] = int(len(vals))
        if len(vals) >= 2:
            xs = np.arange(len(vals), dtype=float)
            features[f"{col}_trend"] = float(np.polyfit(xs, vals, 1)[0])
        if vals.max() > 0:
            features[f"{col}_peak_decline"] = float(
                (vals.max() - vals[-1]) / vals.max()
            )

    # ── Starter Seasons Count + Recent Demotion ────────────────────────────────
    # Use rate-based signals so injury-shortened seasons still count.
    last_season = int(player_stats["season"].max())
    last_row = player_stats[player_stats["season"] == last_season]

    if "is_starter" in player_stats.columns:
        # QB path: is_starter = (attempts_per_game >= 20), injury-safe
        is_starter_col = pd.to_numeric(player_stats["is_starter"], errors="coerce").fillna(0)
        features["starter_seasons"] = int((is_starter_col >= 1).sum())
        last_is_starter_vals = pd.to_numeric(last_row["is_starter"], errors="coerce").fillna(0)
        last_is_starter = int(last_is_starter_vals.values[0] >= 1) if len(last_is_starter_vals) else 1
    elif "games_started" in player_stats.columns:
        features["starter_seasons"] = int((player_stats["games_started"] >= 4).sum())
        last_gs = pd.to_numeric(last_row["games_started"], errors="coerce").fillna(0)
        last_is_starter = int(last_gs.values[0] >= 4) if len(last_gs) else 1
    elif "games" in player_stats.columns:
        features["starter_seasons"] = int((player_stats["games"] >= 8).sum())
        last_g = pd.to_numeric(last_row["games"], errors="coerce").fillna(0)
        last_is_starter = int(last_g.values[0] >= 8) if len(last_g) else 1
    else:
        last_is_starter = 1

    features["recent_demotion"] = int(
        features.get("starter_seasons", 0) >= 2 and last_is_starter == 0
    )

    return features

    # ── Live nfl_data_py version (uncomment to re-enable) ──────────────────
    # seasons = list(range(signing_year - WINDOW, signing_year))
    # player_norm = _normalise(player_name)
    # player_key  = _name_key(player_name)
    # try:
    #     stats = _get_seasonal_data(seasons)
    # except Exception as e:
    #     log.warning("Could not fetch stats: %s", e)
    #     return {}
    # ids = _get_ids()
    # if not ids.empty:
    #     stats = stats.merge(ids[["player_id", "player_name_norm", "name_key"]],
    #                         on="player_id", how="left")
    # if "player_name_norm" not in stats.columns:
    #     return {}
    # player_stats = stats[
    #     (stats["player_name_norm"] == player_norm) |
    #     (stats.get("name_key", pd.Series(dtype=str)) == player_key)
    # ].copy()
    # player_stats = player_stats[player_stats["season"].isin(seasons)]
    # if player_stats.empty:
    #     log.warning("No stats found for %s in seasons %s", player_name, seasons)
    #     return {}
    # numeric_cols = [c for c in player_stats.select_dtypes(include="number").columns if c != "season"]
    # features = {}
    # player_stats = player_stats.sort_values("season")
    # for col in numeric_cols:
    #     vals = pd.to_numeric(player_stats[col], errors="coerce").dropna().values
    #     if len(vals) == 0:
    #         continue
    #     features[f"{col}_mean"]  = float(vals.mean())
    #     features[f"{col}_last"]  = float(vals[-1])
    #     features[f"{col}_games"] = int(len(vals))
    #     if len(vals) >= 2:
    #         xs = np.arange(len(vals), dtype=float)
    #         features[f"{col}_trend"] = float(np.polyfit(xs, vals, 1)[0])
    # return features


# ── Market context ─────────────────────────────────────────────────────────────
def _get_market_context(position: str, signing_year: int) -> dict:
    """
    Load historical contracts to compute market context features:
    - top_contract_pct: highest cap % at this position this year
    - years_since_reset: years since the last record contract
    """
    contracts_path = ROOT / "data" / "raw" / "contracts_with_cap_pct.csv"
    if not contracts_path.exists():
        return {}

    contracts = pd.read_csv(contracts_path)
    contracts["signing_year"] = pd.to_numeric(contracts["signing_year"], errors="coerce")

    same_pos_same_year = contracts[
        (contracts["position"] == position) &
        (contracts["signing_year"] == signing_year)
    ]
    top_pct = same_pos_same_year["apy_pct_cap"].max() if not same_pos_same_year.empty else np.nan

    # Fallback: If predicting for a future year (e.g. 2025) with no signed contracts yet,
    # carry forward the max cap % from the most recent active year.
    # This prevents the model from seeing "NaN" and imputing a low median value.
    if pd.isna(top_pct):
        past_contracts = contracts[
            (contracts["position"] == position) &
            (contracts["signing_year"] < signing_year)
        ].sort_values("signing_year", ascending=False)

        if not past_contracts.empty:
            last_year = past_contracts["signing_year"].iloc[0]
            last_year_max = past_contracts[past_contracts["signing_year"] == last_year]["apy_pct_cap"].max()
            top_pct = last_year_max

    # Years since record contract
    historical = contracts[
        (contracts["position"] == position) &
        (contracts["signing_year"] < signing_year)
    ].sort_values("signing_year")

    years_since_reset = np.nan
    if not historical.empty:
        running_max = 0.0
        last_reset = None
        for _, row in historical.iterrows():
            if pd.notna(row["apy_pct_cap"]) and row["apy_pct_cap"] > running_max:
                running_max = row["apy_pct_cap"]
                last_reset = row["signing_year"]
        if last_reset is not None:
            years_since_reset = signing_year - last_reset

    return {
        "top_contract_pct": top_pct,
        "years_since_reset": years_since_reset,
    }


# ── Main prediction function ───────────────────────────────────────────────────
def predict_contract(
    player_name: str,
    position: str,
    signing_year: int,
    extra_features: Optional[dict] = None,
) -> dict:
    """
    Predict the contract value for a player.

    Args:
        player_name:    Full player name e.g. 'Justin Jefferson'
        position:       'QB', 'WR', 'RB', or 'TE'
        signing_year:   Year the contract would be signed
        extra_features: Optional dict of additional feature overrides

    Returns:
        dict with keys:
            predicted_cap_pct   float  — predicted APY as % of cap
            predicted_apy       float  — predicted APY in dollars
            cap_that_year       int    — salary cap for signing_year
            confidence_range    tuple  — (low, high) rough range
            features_used       dict   — feature values used for prediction
            missing_features    list   — features the model wanted but didn't have
    """
    position = position.upper()
    available = list_available_positions()
    if position not in available:
        raise ValueError(
            f"No model available for {position}. "
            f"Available: {available}. Run notebooks/model.py first."
        )

    model, manifest = load_model(position)
    feature_cols    = manifest["feature_cols"]

    # Gather features
    stat_features = _fetch_player_stats(player_name, signing_year, position)
    mkt_features  = _get_market_context(position, signing_year)
    all_features  = {**stat_features, **mkt_features, **(extra_features or {})}

    # Build feature vector in the exact order the model expects
    feature_vector = []
    missing = []
    for col in feature_cols:
        val = all_features.get(col, np.nan)
        feature_vector.append(val)
        if np.isnan(val) if isinstance(val, float) else val != val:
            missing.append(col)

    # Convert to DataFrame to avoid warnings from tree models expecting feature names
    X = pd.DataFrame([feature_vector], columns=feature_cols)

    # Predict
    predicted_cap_pct = float(model.predict(X)[0])
    predicted_cap_pct = max(0.0, predicted_cap_pct)  # floor at 0

    cap = CAP_HISTORY.get(signing_year, CAP_HISTORY[max(CAP_HISTORY)])
    predicted_apy = predicted_cap_pct * cap

    # Rough confidence range (±1 MAE — load from results summary if available)
    results_path = MODELS_DIR / "results_summary.json"
    mae_cap_pct = 0.02  # default ±2%
    if results_path.exists():
        with open(results_path) as f:
            summary = json.load(f)
        model_name = manifest["model_name"]
        if position in summary and model_name in summary[position]:
            mae_cap_pct = summary[position][model_name]["mae"] / 100

    confidence_low  = max(0.0, predicted_cap_pct - mae_cap_pct)
    confidence_high = predicted_cap_pct + mae_cap_pct

    return {
        "player_name":        player_name,
        "position":           position,
        "signing_year":       signing_year,
        "predicted_cap_pct":  round(predicted_cap_pct * 100, 2),
        "predicted_apy":      round(predicted_apy),
        "cap_that_year":      cap,
        "confidence_range": (
            round(confidence_low  * 100, 2),
            round(confidence_high * 100, 2),
        ),
        "features_used":      {k: v for k, v in all_features.items() if k in feature_cols},
        "missing_features":   missing,
        "model_used":         manifest["model_name"],
    }


# ── Comps finder ──────────────────────────────────────────────────────────────
def find_comps(
    position: str,
    predicted_cap_pct: float,
    signing_year: int,
    n: int = 5,
) -> pd.DataFrame:
    """
    Find historical contracts most similar in cap % to the prediction.
    Useful for the Streamlit app's 'comparable contracts' table.
    """
    # Use the committed contracts file (model_features.csv is gitignored)
    contracts_path = ROOT / "data" / "raw" / "contracts_with_cap_pct.csv"
    if not contracts_path.exists():
        return pd.DataFrame()

    df = pd.read_csv(contracts_path)
    df = df[
        (df["position"] == position) &
        (df["apy_pct_cap"].notna()) &
        (df["signing_year"] < signing_year)
    ].copy()

    df["diff"] = (df["apy_pct_cap"] - predicted_cap_pct / 100).abs()
    comps = df.nsmallest(n, "diff")[
        ["player_name", "team", "signing_year", "apy", "apy_pct_cap", "contract_years"]
    ].copy()
    comps["apy_pct_cap"] = (comps["apy_pct_cap"] * 100).round(1)
    comps["apy"] = comps["apy"].apply(lambda x: f"${x/1e6:.1f}M" if pd.notna(x) else "—")
    comps.columns = ["Player", "Team", "Year", "APY", "Cap %", "Years"]
    return comps.reset_index(drop=True)


# ── CLI ────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Predict NFL contract value")
    parser.add_argument("player",       type=str, help="Player name e.g. 'Josh Allen'")
    parser.add_argument("position",     type=str, help="Position: QB, WR, RB, TE")
    parser.add_argument("signing_year", type=int, help="Year of signing e.g. 2025")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    result = predict_contract(args.player, args.position, args.signing_year)

    print(f"\n{'='*50}")
    print(f"  {result['player_name']} ({result['position']}) — {result['signing_year']}")
    print(f"{'='*50}")
    print(f"  Predicted APY:     ${result['predicted_apy']:,.0f}")
    print(f"  Cap %:             {result['predicted_cap_pct']:.1f}%")
    print(f"  Confidence range:  {result['confidence_range'][0]:.1f}% – {result['confidence_range'][1]:.1f}%")
    print(f"  Model:             {result['model_used']}")
    print(f"  Missing features:  {len(result['missing_features'])}")

    comps = find_comps(result["position"], result["predicted_cap_pct"], result["signing_year"])
    if not comps.empty:
        print(f"\n  Comparable contracts:")
        print(comps.to_string(index=False))
