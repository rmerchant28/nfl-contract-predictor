# NFL Contract Predictor

Predicts how much an NFL player will earn on their next contract as a **percentage of the salary cap** — normalizing across eras so contracts from different years are directly comparable.

## Two-venv setup

This project uses two Python environments because `nfl_data_py` (used for stat collection) requires `pandas<2.0`, while the modelling and Streamlit app work best with pandas 2.x.

| Venv | Purpose | Key packages |
|---|---|---|
| `venv_data` | Data pipeline only | nfl_data_py, pandas<2.0 |
| `venv312` | Models + Streamlit app | scikit-learn, lightgbm, streamlit, pandas 2.x |

The CSVs in `data/raw/` are the handoff point — `venv_data` writes them, `venv312` reads them.

## Quick start

```bash
# Create both venvs and install all dependencies
make setup

# Run full data pipeline (scrape contracts + stats, build features)
make data

# Train models
make model

# Launch the app
make app
```

## Manual setup (without make)

```bash
# venv_data — for data collection
python3.12 -m venv venv_data
source venv_data/bin/activate
pip install "pandas<2.0" numpy requests beautifulsoup4 lxml certifi joblib
pip install setuptools==69.5.1
pip install nfl_data_py --no-deps
pip install pyarrow
deactivate

# venv312 — for modelling and app
python3.12 -m venv venv312
source venv312/bin/activate
pip install "pandas>=2.0" numpy scikit-learn lightgbm streamlit plotly requests beautifulsoup4 lxml certifi joblib
```

## Workflow

```bash
# Step 1 — collect data (venv_data)
source venv_data/bin/activate
python run_pipeline.py --positions QB WR RB TE --start-year 2015
deactivate

# Step 2 — train models (venv312)
source venv312/bin/activate
python notebooks/model.py

# Step 3 — run app (venv312)
streamlit run app.py
```

Or with make:
```bash
make data POSITIONS="QB WR RB TE" START_YEAR=2015
make model
make app
```

## How it works

**1. Salary cap history** — hardcoded lookup table (2014–2026). More reliable than scraping since OTC removed their history page.

**2. Contract data (OverTheCap)** — scraped from `overthecap.com/contract-history/{position}`. Uses history pages (not current pages) so signing years are accurate. Rookie deals filtered out by position-specific APY thresholds.

**3. Player statistics (nfl_data_py)** — `import_seasonal_data()` returns per-season stats. Player names are joined via `import_ids()`. A `name_key` (first initial + last name) handles abbreviated names. Stats fetched from `start_year - 3` to cover the pre-contract lookback window.

**4. Feature engineering** — for each contract, look back 3 seasons and compute three features per stat: mean (overall level), last (most recent season), trend (linear regression slope). Market context features added: peer rank, top contract at position that year.

## Project structure

```
nfl_contract_predictor/
├── Makefile                 # Two-venv workflow automation
├── run_pipeline.py          # Data pipeline — run with venv_data
├── notebooks/
│   └── model.py             # Model training — run with venv312
├── app.py                   # Streamlit app — run with venv312
├── predict.py               # Inference module used by the app
├── scrapers/
│   ├── utils.py             # HTTP session, caching, name normalisation
│   ├── overthecap.py        # Contract data + salary cap history
│   ├── pfr.py               # Player stats via nfl_data_py
│   └── features.py          # Joins contracts + stats → model-ready CSV
├── data/
│   ├── raw/                 # Scraped CSVs (contracts, stats)
│   ├── processed/           # model_features.csv — feature matrix
│   └── .cache/              # Cached HTML pages (7-day TTL)
├── models/                  # Trained model .pkl files + manifests
└── notebooks/figures/       # EDA and model accuracy plots
```

## Target variable

`apy_pct_cap = annual average value ÷ salary cap in the signing year`

| Position | Model | MAE | Typical range |
|---|---|---|---|
| QB | LightGBM | 2.01% | 10–24% |
| WR | Ridge | 1.15% | 2–16% |
| RB | HGB | 0.86% | 1–11% |
| TE | HGB | 0.60% | 1–10% |

Validated via leave-one-year-out cross-validation.

## Data sources

| Source | What we get | How |
|---|---|---|
| [OverTheCap](https://overthecap.com/contract-history) | Contract APY, value, signing year | Scraped (BS4) |
| [nfl_data_py](https://github.com/nflverse/nfl_data_py) | Passing, rushing, receiving stats | Python library (venv_data) |

## Makefile reference

```
make setup              Create both venvs
make data               Full data pipeline
make data POSITIONS="QB WR" START_YEAR=2018   Custom run
make data-contracts-only   Contracts only, skip stats
make data-features-only    Rebuild features from existing CSVs
make model              Train all models
make app                Launch Streamlit app
make all                data + model in one command
make clean              Clear cache and processed files
make clean-all          Remove all generated data and models
make help               Show all commands
```

## Known limitations

- **OL not included** — no public per-player blocking grades. PFF has them (paid).
- **Age at signing** not populated — requires scraping individual player pages.
- **Name matching** covers ~95% of contracts. The 5% gap is unusual name formats.
- **Separate models per position** — QB market dynamics differ significantly from skill positions.
- **PFF grades** are the highest-value addition — merge on `player_name_norm + season`.
- **nfl_data_py** requires pandas<2.0, hence the two-venv setup.
