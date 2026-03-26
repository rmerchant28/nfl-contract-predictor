# NFL Contract Predictor

Predicts what an NFL player would earn on a new free agent contract as a **percentage of the salary cap** — normalizing across eras so contracts from different years are directly comparable.

## Quick start

```bash
# Create venv and install all dependencies
make setup

# Run full data pipeline (scrape contracts + stats, build features)
make data

# Train models
make model

# Launch the app
make app

# Run tests
make test
```

## Manual setup

```bash
python3.12 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Workflow

```bash
source venv/bin/activate

# Step 1 — collect data
python run_pipeline.py --positions QB WR RB TE --start-year 2015

# Step 2 — train models
python notebooks/model.py

# Step 3 — run app
streamlit run app.py
```

## How it works

**1. Salary cap history** — hardcoded lookup table (2014–2026). More reliable than scraping since OTC removed their history page.

**2. Contract data (OverTheCap)** — scraped from `overthecap.com/contract-history/{position}`. Uses history pages so signing years are accurate. Rookie deals filtered out by position-specific APY thresholds.

**3. Player statistics (nflreadpy)** — `load_player_stats()` returns per-season stats including 2025+ data. Stats fetched from `start_year - 3` to cover the pre-contract lookback window.

**4. Feature engineering** — for each contract, look back 3 seasons and compute per stat:
- `_mean` — 3-year average
- `_last` — most recent season value
- `_max` — career-window peak
- `_trend` — linear regression slope (improving vs. declining)
- `_peak_decline` — how far the player has fallen from their peak (`(max - last) / max`)

Additional signals:
- `games_missed` — games absent per season (season-length-aware: 17 games from 2021, 16 before). Injury signal.
- `age_at_signing` — player age when the contract is signed, sourced from `nflreadpy.load_players()` birth dates
- `starter_seasons` — how many of the last 3 seasons the player had starter-level volume
- `recent_demotion` — flag for players who were multi-year starters but lost the job last season
- `snap_pct` — mean offensive snap share from weekly snap count data
- `gap_years` — seasons since last active (recency penalty)

**5. Market context** — top contract at position that year, years since a record deal was set.

## Prediction targets

The model trains three outputs per position:

| Target | File suffix | Description |
|--------|-------------|-------------|
| `apy_pct_cap` | `_{model}.pkl` | APY as % of salary cap — primary output |
| `guaranteed_pct_cap` | `_{model}_gtd.pkl` | Guaranteed money as % of cap |
| `contract_years` | `_{model}_years.pkl` | Expected contract length in years |

Quantile regression models (XGBoost `reg:quantileerror` at p10/p90) are trained per position and then **conformalized** using leave-one-year-out residuals (Conformalized Quantile Regression, CQR). This produces confidence intervals with a statistical ~80% coverage guarantee rather than just hoping the raw quantile model lands there.

| File | Description |
|------|-------------|
| `{POS}_quantile_low.pkl` | Raw p10 boundary |
| `{POS}_quantile_high.pkl` | Raw p90 boundary |
| `evaluation.json` → `calibration.cqr_correction` | Per-position shift applied at inference |

At prediction time: `ci_low = raw_p10 − correction`, `ci_high = raw_p90 + correction`. Falls back to ±MAE symmetric intervals if quantile models are unavailable.

## Model evaluation

`make model` writes `models/evaluation.json` with per-position diagnostics:

- **Quantile calibration** — fraction of actual values falling within the raw p10/p90 band, plus the `cqr_correction` value used to shift bounds at inference to achieve ~80% coverage
- **Tier bias** — MAE and mean signed error sliced by APY percentile tier (Bottom 25%, Q25–50%, Q50–75%, Q75–90%, Top 10%)
- **Residual analysis** — out-of-fold error distribution (mean bias, std, skewness, % overpredicted)
- **Contract years MAE** — accuracy of the years prediction in years

## Project structure

```
nfl_contract_predictor/
├── Makefile                 # Single-venv workflow automation
├── requirements.txt         # All dependencies — one file, one venv
├── run_pipeline.py          # Data pipeline
├── notebooks/
│   └── model.py             # Model training (APY, guaranteed, contract years, quantile)
├── app.py                   # Streamlit app — dark mode, hero section, metric cards
├── predict.py               # Inference module used by the app
├── pages/
│   ├── 1_Compare_Players.py # Side-by-side player stats + contract predictions
│   └── 2_Model_Diagnostics.py  # Calibration, tier bias, residuals, worst predictions
├── scrapers/
│   ├── utils.py             # HTTP session, caching, name normalisation
│   ├── overthecap.py        # Contract data + salary cap history
│   ├── pfr.py               # Player stats via nflreadpy
│   └── features.py          # Joins contracts + stats → model-ready CSV
├── tests/
│   ├── conftest.py          # sys.path setup
│   ├── test_utils.py        # clean_money, clean_pct, normalise_name, map_position
│   ├── test_pfr.py          # _make_name_key, build_pre_contract_stats
│   ├── test_overthecap.py   # add_cap_percentage
│   ├── test_features.py     # _age_at_signing
│   └── test_predict.py      # _normalise, _name_key, find_comps
├── data/
│   ├── raw/                 # Scraped CSVs (contracts, stats)
│   ├── processed/           # model_features.csv — feature matrix
│   └── .cache/              # Cached HTML pages (7-day TTL)
└── models/                  # Trained .pkl files + manifests + evaluation.json
```

## Models

One model per position, selected by lowest leave-one-year-out (LOYO) MAE. The same LOYO procedure trains the guaranteed and contract years models.

| Position | Model | APY MAE (approx) |
|----------|-------|-----------------|
| QB | XGBoost | ~1.5% cap |
| WR | LightGBM | ~1.3% cap |
| RB | Random Forest | ~0.9% cap |
| TE | Random Forest | ~1.0% cap |

## Frontend pages

| Page | Description |
|------|-------------|
| **Home** (`app.py`) | Enter a player name, position, and signing year to get a prediction. Shows APY, cap %, estimated guaranteed money, contract length, and a comparable contracts table. |
| **Compare Players** (`pages/1_Compare_Players.py`) | Side-by-side view of two players — career stats, season-by-season tables, and contract predictions on the same chart. |
| **Model Diagnostics** (`pages/2_Model_Diagnostics.py`) | Per-position model health: calibration gauge, tier bias bar chart, residual histogram + scatter, worst prediction table. |

## Tests

86 unit tests, no network calls or trained models required.

```bash
make test           # run all tests
make test-v         # verbose output
```

Coverage:
- `test_utils.py` — money/pct cleaning, name normalisation, position mapping (27 tests)
- `test_pfr.py` — name key generation, rolling stat aggregation, peak_decline, gap_years, starter tracking, demotion flag, window slicing, trend direction (25 tests)
- `test_overthecap.py` — cap percentage computation, guaranteed pct, null cap year (5 tests)
- `test_features.py` — age_at_signing with birth_year column and nflreadpy mocking (5 tests)
- `test_predict.py` — name normalisation, name key, find_comps filtering/ordering/formatting (19 tests)

## Data sources

| Source | What | How |
|--------|------|-----|
| [OverTheCap](https://overthecap.com/contract-history) | Contract APY, guaranteed, signing year, length | Scraped (BS4 + pandas) |
| [nflreadpy](https://github.com/nflverse/nflreadpy) | Passing, rushing, receiving, snap stats, player birth dates (2011–present) | Python library |

## Makefile reference

```
make setup                          Create venv and install dependencies
make data                           Full data pipeline
make data POSITIONS="QB WR" START_YEAR=2018   Custom run
make data-contracts-only            Contracts only, skip stats
make data-features-only             Rebuild features from existing CSVs
make model                          Train all models (APY, guaranteed, years, quantile)
make app                            Launch Streamlit app
make test                           Run test suite
make test-v                         Run tests with verbose output
make all                            data + model in one command
make clean                          Clear cache and processed files
make clean-all                      Remove all generated data and models
make help                           Show all commands
```

## Known limitations

- **OL not included** — no public per-player blocking grades at scale.
- **Name matching** covers ~95% of contracts. The 5% gap is unusual name formats.
- **Separate models per position** — QB market dynamics differ significantly from skill positions.
- **2025 data** — available via nflreadpy. Re-run `make data` each offseason to pull the latest season.
