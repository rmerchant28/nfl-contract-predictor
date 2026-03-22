"""
NFL Contract Value Model
========================
Predicts a player's contract APY as % of the salary cap.

Models:       Ridge regression, XGBoost, LightGBM — compared per position
Validation:   Leave-one-year-out (LOYO) cross-validation
Metric:       MAE and RMSE (in cap % points)
Positions:    QB, WR, RB, TE — one model per position
Output:       Serialized models + feature manifest → models/

Run:
    python notebooks/model.py
"""

import sys
import json
import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from sklearn.impute import SimpleImputer
from sklearn.ensemble import HistGradientBoostingRegressor, GradientBoostingRegressor
import joblib

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

try:
    import lightgbm as lgb
    HAS_LGB = True
except Exception:
    log.warning("LightGBM not available — using GradientBoostingRegressor instead.")
    HAS_LGB = False

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT        = Path(__file__).resolve().parent.parent
DATA_PATH   = ROOT / "data" / "processed" / "model_features.csv"
MODELS_DIR  = ROOT / "models"
FIGURES_DIR = ROOT / "notebooks" / "figures"
MODELS_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)

# ── Config ─────────────────────────────────────────────────────────────────────
POSITIONS       = ["QB", "WR", "RB", "TE"]
TARGET          = "apy_pct_cap"
MIN_TRAIN_ROWS  = 10      # skip year if too few training samples
RANDOM_STATE    = 42

# Columns to always drop before modelling
DROP_COLS = [
    "player_name", "player_name_norm", "team", "position_raw",
    "position_group", "total_value", "apy", "guaranteed",
    "contract_years", "cap_that_year", "year", "source_url",
    "signing_year",           # used for LOYO split, not a feature
    "apy_pct_cap_otc",        # OTC's own cap% — data leakage
    "inflated_value",
    "position",               # encoded separately per-model
]


# ── Data loading ───────────────────────────────────────────────────────────────
def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df["signing_year"] = pd.to_numeric(df["signing_year"], errors="coerce").astype("Int64")
    df = df[df[TARGET].notna()].copy()
    log.info("Loaded %d rows × %d cols", *df.shape)
    return df


def position_df(df: pd.DataFrame, position: str) -> pd.DataFrame:
    """Filter to a single position and drop irrelevant columns."""
    pos_df = df[df["position"] == position].copy()

    drop = [c for c in DROP_COLS if c in pos_df.columns]
    pos_df = pos_df.drop(columns=drop)

    # Drop columns with >60% nulls
    null_frac = pos_df.isnull().mean()
    high_null = null_frac[null_frac > 0.6].index.tolist()
    if high_null:
        log.info("  Dropping high-null cols for %s: %s", position, high_null)
    pos_df = pos_df.drop(columns=high_null)

    return pos_df


# ── Feature matrix builder ─────────────────────────────────────────────────────
def get_feature_cols(df: pd.DataFrame) -> list[str]:
    """Return all numeric feature columns (everything except the target)."""
    return [c for c in df.select_dtypes(include="number").columns if c != TARGET]


# ── LOYO cross-validation ──────────────────────────────────────────────────────
def loyo_cv(
    df: pd.DataFrame,
    model_fn,
    feature_cols: list[str],
) -> pd.DataFrame:
    """
    Leave-one-year-out cross-validation.

    For each year Y in the dataset:
      - Train on all years except Y
      - Predict on year Y
      - Record MAE, RMSE, and per-row predictions

    Returns a DataFrame of out-of-fold predictions with columns:
        signing_year, y_true, y_pred
    """
    years = sorted(df["signing_year"].dropna().unique())
    records = []

    for test_year in years:
        train = df[df["signing_year"] != test_year]
        test  = df[df["signing_year"] == test_year]

        if len(train) < MIN_TRAIN_ROWS or len(test) == 0:
            continue

        X_train = train[feature_cols].values
        y_train = train[TARGET].values
        X_test  = test[feature_cols].values
        y_test  = test[TARGET].values

        model = model_fn()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        for yt, yp, yr in zip(y_test, y_pred, test["signing_year"]):
            records.append({"signing_year": yr, "y_true": yt, "y_pred": yp})

    return pd.DataFrame(records)


def cv_metrics(oof: pd.DataFrame) -> dict:
    """Compute MAE and RMSE from out-of-fold predictions (in cap % points)."""
    mae  = mean_absolute_error(oof["y_true"], oof["y_pred"]) * 100
    rmse = root_mean_squared_error(oof["y_true"], oof["y_pred"]) * 100
    return {"mae": round(mae, 3), "rmse": round(rmse, 3), "n": len(oof)}


# ── Model factories ────────────────────────────────────────────────────────────
def make_ridge():
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
        ("model",   Ridge(alpha=10.0)),
    ])


def make_hgb():
    """HistGradientBoostingRegressor — pure sklearn, no native deps, handles NaN natively."""
    return Pipeline([
        ("model", HistGradientBoostingRegressor(
            max_iter=300,
            learning_rate=0.05,
            max_depth=4,
            random_state=RANDOM_STATE,
        )),
    ])


def make_lgb():
    if HAS_LGB:
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", lgb.LGBMRegressor(
                n_estimators=300,
                learning_rate=0.05,
                max_depth=4,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=RANDOM_STATE,
                verbose=-1,
            )),
        ])
    else:
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", GradientBoostingRegressor(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=4,
                random_state=RANDOM_STATE,
            )),
        ])


MODEL_FACTORIES = {
    "Ridge":    make_ridge,
    "HGB":      make_hgb,
    "LightGBM": make_lgb,
}


# ── EDA plots ──────────────────────────────────────────────────────────────────
def plot_target_distribution(df: pd.DataFrame):
    fig, axes = plt.subplots(1, len(POSITIONS), figsize=(14, 4))
    for ax, pos in zip(axes, POSITIONS):
        vals = df[df["position"] == pos][TARGET] * 100
        ax.hist(vals, bins=20, edgecolor="white", linewidth=0.5)
        ax.set_title(pos)
        ax.set_xlabel("APY % of cap")
        ax.set_ylabel("Count")
        ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
    fig.suptitle("Target variable distribution by position", y=1.02)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "target_distribution.png", dpi=150, bbox_inches="tight")
    plt.close()
    log.info("Saved target_distribution.png")


def plot_cap_pct_over_time(df: pd.DataFrame):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()
    for ax, pos in zip(axes, POSITIONS):
        pos_df = df[df["position"] == pos].copy()
        yearly = pos_df.groupby("signing_year")[TARGET].agg(["mean", "max"]).reset_index()
        ax.plot(yearly["signing_year"], yearly["mean"] * 100, label="Mean", marker="o", ms=4)
        ax.plot(yearly["signing_year"], yearly["max"]  * 100, label="Max",  marker="s", ms=4, linestyle="--")
        ax.set_title(pos)
        ax.set_xlabel("Year")
        ax.set_ylabel("APY % of cap")
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
        ax.legend(fontsize=8)
    fig.suptitle("Cap % over time by position", y=1.02)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "cap_pct_over_time.png", dpi=150, bbox_inches="tight")
    plt.close()
    log.info("Saved cap_pct_over_time.png")


def plot_feature_correlations(df: pd.DataFrame, position: str, feature_cols: list[str]):
    pos_df = df[df["position"] == position][feature_cols + [TARGET]].copy()
    corrs = pos_df.corr()[TARGET].drop(TARGET).sort_values(key=abs, ascending=False).head(15)

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#2196F3" if v > 0 else "#EF5350" for v in corrs.values]
    ax.barh(corrs.index[::-1], corrs.values[::-1] * 100, color=colors[::-1])
    ax.set_xlabel("Correlation with cap % (×100)")
    ax.set_title(f"{position} — top feature correlations with target")
    ax.axvline(0, color="black", linewidth=0.8)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f"correlations_{position}.png", dpi=150, bbox_inches="tight")
    plt.close()
    log.info("Saved correlations_%s.png", position)


# ── Feature importance (tree models) ──────────────────────────────────────────
def plot_feature_importance(model_pipeline, feature_cols: list[str], position: str, model_name: str):
    model = model_pipeline.named_steps["model"]
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_)
    else:
        return

    imp_df = pd.DataFrame({"feature": feature_cols, "importance": importances})
    imp_df = imp_df.sort_values("importance", ascending=False).head(20)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(imp_df["feature"][::-1], imp_df["importance"][::-1])
    ax.set_title(f"{position} {model_name} — feature importance (top 20)")
    ax.set_xlabel("Importance")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f"importance_{position}_{model_name}.png", dpi=150, bbox_inches="tight")
    plt.close()


# ── Predicted vs actual plot ───────────────────────────────────────────────────
def plot_pred_vs_actual(oof: pd.DataFrame, position: str, model_name: str):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(oof["y_true"] * 100, oof["y_pred"] * 100, alpha=0.5, s=20)
    lims = [
        min(oof["y_true"].min(), oof["y_pred"].min()) * 100,
        max(oof["y_true"].max(), oof["y_pred"].max()) * 100,
    ]
    ax.plot(lims, lims, "r--", linewidth=1, label="Perfect prediction")
    ax.set_xlabel("Actual cap %")
    ax.set_ylabel("Predicted cap %")
    ax.set_title(f"{position} {model_name} — predicted vs actual (LOYO)")
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f"pred_vs_actual_{position}_{model_name}.png", dpi=150, bbox_inches="tight")
    plt.close()


# ── Model comparison summary ───────────────────────────────────────────────────
def print_results_table(results: dict):
    print("\n" + "=" * 65)
    print(f"{'Position':<8} {'Model':<12} {'MAE':>8} {'RMSE':>8} {'N':>6}")
    print("-" * 65)
    for pos in POSITIONS:
        if pos not in results:
            continue
        for model_name, metrics in results[pos].items():
            print(f"{pos:<8} {model_name:<12} "
                  f"{metrics['mae']:>7.2f}%  {metrics['rmse']:>7.2f}%  {metrics['n']:>6}")
        print()
    print("=" * 65)
    print("MAE / RMSE are in cap percentage points (e.g. 1.5 = 1.5% of cap)")


# ── Final model training (on all data) ────────────────────────────────────────
def train_final_model(
    df: pd.DataFrame,
    position: str,
    model_name: str,
    model_fn,
    feature_cols: list[str],
) -> object:
    """Train the chosen model on all available data for deployment."""
    pos_df = df[df["position"] == position].copy()
    pos_df_clean = pos_df.drop(
        columns=[c for c in DROP_COLS if c in pos_df.columns],
        errors="ignore"
    )
    # Drop high-null cols
    null_frac = pos_df_clean.isnull().mean()
    pos_df_clean = pos_df_clean.drop(columns=null_frac[null_frac > 0.6].index)

    X = pos_df_clean[feature_cols].values
    y = pos_df_clean[TARGET].values

    model = model_fn()
    model.fit(X, y)
    log.info("  Trained final %s model for %s on %d rows", model_name, position, len(X))
    return model


# ── Save model artifacts ───────────────────────────────────────────────────────
def save_artifacts(position: str, model_name: str, model, feature_cols: list[str]):
    model_path = MODELS_DIR / f"{position}_{model_name.lower()}.pkl"
    joblib.dump(model, model_path)

    manifest = {
        "position":     position,
        "model_name":   model_name,
        "feature_cols": feature_cols,
        "target":       TARGET,
    }
    manifest_path = MODELS_DIR / f"{position}_{model_name.lower()}_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    log.info("  Saved model → %s", model_path.name)
    log.info("  Saved manifest → %s", manifest_path.name)


# ── Best model selector ────────────────────────────────────────────────────────
def best_model_name(position_results: dict) -> str:
    """Return the model name with lowest MAE for a position."""
    return min(position_results, key=lambda m: position_results[m]["mae"])


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    df = load_data()

    # Filter to target positions only
    df = df[df["position"].isin(POSITIONS)].copy()
    log.info("Positions in scope: %s  (%d rows)", POSITIONS, len(df))

    # ── EDA ────────────────────────────────────────────────────────────────────
    log.info("Generating EDA plots...")
    plot_target_distribution(df)
    plot_cap_pct_over_time(df)

    # ── LOYO CV per position per model ─────────────────────────────────────────
    results = {}          # results[position][model_name] = metrics dict
    oof_preds = {}        # oof_preds[position][model_name] = oof DataFrame

    for position in POSITIONS:
        log.info("\n── %s ──────────────────────────────────────────", position)
        pos_df = position_df(df, position)

        if len(pos_df) < 20:
            log.warning("  Not enough data for %s (%d rows) — skipping.", position, len(pos_df))
            continue

        feature_cols = get_feature_cols(pos_df)
        log.info("  %d contracts, %d features", len(pos_df), len(feature_cols))

        plot_feature_correlations(df, position, feature_cols)

        results[position]  = {}
        oof_preds[position] = {}

        for model_name, model_fn in MODEL_FACTORIES.items():
            log.info("  Running LOYO CV: %s...", model_name)
            # Re-attach signing_year for the LOYO split
            pos_df_with_year = pos_df.copy()
            pos_df_with_year["signing_year"] = df.loc[pos_df.index, "signing_year"]

            oof = loyo_cv(pos_df_with_year, model_fn, feature_cols)
            if oof.empty:
                log.warning("  No OOF predictions for %s %s", position, model_name)
                continue

            metrics = cv_metrics(oof)
            results[position][model_name] = metrics
            oof_preds[position][model_name] = oof

            log.info("  %s  MAE=%.2f%%  RMSE=%.2f%%  n=%d",
                     model_name, metrics["mae"], metrics["rmse"], metrics["n"])

            plot_pred_vs_actual(oof, position, model_name)

    # ── Print summary ──────────────────────────────────────────────────────────
    print_results_table(results)

    # ── Train final models on all data and save ────────────────────────────────
    log.info("\nTraining final models on full dataset...")
    best_models = {}

    for position in POSITIONS:
        if position not in results:
            continue

        pos_df      = position_df(df, position)
        feature_cols = get_feature_cols(pos_df)

        best_name  = best_model_name(results[position])
        best_fn    = MODEL_FACTORIES[best_name]
        best_models[position] = best_name

        log.info("  %s → best model: %s (MAE=%.2f%%)",
                 position, best_name, results[position][best_name]["mae"])

        final_model = train_final_model(df, position, best_name, best_fn, feature_cols)
        save_artifacts(position, best_name, final_model, feature_cols)

        # Also save feature importance plot for the final model
        plot_feature_importance(final_model, feature_cols, position, best_name)

    # ── Save results summary ───────────────────────────────────────────────────
    summary = {
        pos: {
            model: metrics
            for model, metrics in model_results.items()
        }
        for pos, model_results in results.items()
    }
    summary["best_models"] = best_models

    with open(MODELS_DIR / "results_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # ── Save OOF predictions for best model per position ──────────────────────
    # These are your "test set" — one row per contract, left out during training
    all_oof = []
    for position in POSITIONS:
        if position not in oof_preds:
            continue
        best_name = best_models.get(position)
        if not best_name or best_name not in oof_preds[position]:
            continue

        oof = oof_preds[position][best_name].copy()

        # Pull contract metadata and deduplicate before merging
        pos_contracts = (
            df[df["position"] == position][["player_name", "position", "signing_year", TARGET]]
            .sort_values("signing_year")
            .drop_duplicates(subset=["player_name", "signing_year"], keep="first")
            .reset_index(drop=True)
        )

        # Also deduplicate OOF on (signing_year, y_true) to ensure 1:1 merge
        oof = (
            oof.sort_values("signing_year")
            .drop_duplicates(subset=["signing_year", "y_true"], keep="first")
            .reset_index(drop=True)
        )

        merged = pos_contracts.merge(
            oof[["signing_year", "y_true", "y_pred"]],
            left_on=["signing_year", TARGET],
            right_on=["signing_year", "y_true"],
            how="inner",
        )
        merged["error_pct"]     = (merged["y_pred"] - merged["y_true"]) * 100
        merged["abs_error_pct"] = merged["error_pct"].abs()
        merged["y_true_pct"]    = (merged["y_true"] * 100).round(2)
        merged["y_pred_pct"]    = (merged["y_pred"] * 100).round(2)
        merged["model"]         = best_name
        all_oof.append(merged)

    if all_oof:
        oof_df = pd.concat(all_oof, ignore_index=True)
        oof_path = MODELS_DIR / "oof_predictions.csv"
        oof_df[[
            "player_name", "position", "signing_year",
            "y_true_pct", "y_pred_pct", "error_pct", "abs_error_pct", "model"
        ]].sort_values(["position", "signing_year"]).to_csv(oof_path, index=False)
        log.info("Saved OOF predictions → %s", oof_path.name)

        # Print worst predictions per position for a sanity check
        log.info("\nLargest errors per position (top 3):")
        for pos, grp in oof_df.groupby("position"):
            worst = grp.nlargest(3, "abs_error_pct")[
                ["player_name", "signing_year", "y_true_pct", "y_pred_pct", "error_pct"]
            ]
            log.info("  %s:", pos)
            for _, row in worst.iterrows():
                log.info("    %-22s %d  actual=%.1f%%  pred=%.1f%%  err=%+.1f%%",
                         row["player_name"], int(row["signing_year"]),
                         row["y_true_pct"], row["y_pred_pct"], row["error_pct"])

    log.info("\nAll artifacts saved to %s/", MODELS_DIR)
    log.info("Figures saved to %s/", FIGURES_DIR)
    log.info("\nBest models selected:")
    for pos, name in best_models.items():
        mae = results[pos][name]["mae"]
        log.info("  %s → %s  (MAE %.2f%%)", pos, name, mae)


if __name__ == "__main__":
    main()
