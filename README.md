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
- `starter_seasons` — how many of the last 3 seasons the player had starter-level volume
- `recent_demotion` — flag for players who were multi-year starters but lost the job last season (key signal for players needing to take below-market deals)
- `snap_pct` — mean offensive snap share from weekly snap count data
- `gap_years` — seasons since last active (recency penalty)

**5. Market context** — top contract at position that year, years since a record deal was set.

## Project structure

```
nfl_contract_predictor/
├── Makefile                 # Single-venv workflow automation
├── requirements.txt         # All dependencies — one file, one venv
├── run_pipeline.py          # Data pipeline
├── notebooks/
│   └── model.py             # Model training
├── app.py                   # Streamlit app
├── predict.py               # Inference module used by the app
├── scrapers/
│   ├── utils.py             # HTTP session, caching, name normalisation
│   ├── overthecap.py        # Contract data + salary cap history
│   ├── pfr.py               # Player stats via nflreadpy
│   └── features.py          # Joins contracts + stats → model-ready CSV
├── data/
│   ├── raw/                 # Scraped CSVs (contracts, stats)
│   ├── processed/           # model_features.csv — feature matrix
│   └── .cache/              # Cached HTML pages (7-day TTL)
└── models/                  # Trained model .pkl + manifest files
```

## Models

One model per position, selected by lowest leave-one-year-out MAE:

| Position | Model | MAE |
|---|---|---|
| QB | XGBoost | ~1.5% cap |
| WR | LightGBM | ~1.3% cap |
| RB | Random Forest | ~0.9% cap |
| TE | Random Forest | ~1.0% cap |

Target variable: `apy_pct_cap = annual average value ÷ salary cap in the signing year`

## Data sources

| Source | What | How |
|---|---|---|
| [OverTheCap](https://overthecap.com/contract-history) | Contract APY, value, signing year | Scraped (BS4) |
| [nflreadpy](https://github.com/nflverse/nflreadpy) | Passing, rushing, receiving, snap stats (2011–present) | Python library |

## Makefile reference

```
make setup                          Create venv and install dependencies
make data                           Full data pipeline
make data POSITIONS="QB WR" START_YEAR=2018   Custom run
make data-contracts-only            Contracts only, skip stats
make data-features-only             Rebuild features from existing CSVs
make model                          Train all models
make app                            Launch Streamlit app
make all                            data + model in one command
make clean                          Clear cache and processed files
make clean-all                      Remove all generated data and models
make help                           Show all commands
```

## Known limitations

- **OL not included** — no public per-player blocking grades at scale.
- **Age at signing** not populated — requires scraping individual player pages.
- **Name matching** covers ~95% of contracts. The 5% gap is unusual name formats.
- **Separate models per position** — QB market dynamics differ significantly from skill positions.
- **2025 data** — available via nflreadpy. Re-run `make data` each offseason to pull the latest season.
