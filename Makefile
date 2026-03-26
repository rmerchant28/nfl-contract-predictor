# NFL Contract Predictor — Makefile
# ==================================
# Single-venv workflow:
#   venv  — Python 3.12, all dependencies (data pipeline + models + app)
#
# Usage:
#   make setup        — create venv and install all dependencies
#   make data         — run the full data pipeline
#   make model        — train all models (APY, guaranteed, contract years, quantile)
#   make app          — launch Streamlit app
#   make test         — run unit tests
#   make all          — data → model (then run 'make app' separately)
#   make clean        — remove cached data (keeps raw CSVs and models)
#   make clean-all    — remove all generated data including CSVs and models

POSITIONS  ?= QB WR RB TE
START_YEAR ?= 2015

VENV   = venv
PYTHON = $(VENV)/bin/python
PIP    = $(VENV)/bin/pip
PYTEST = $(VENV)/bin/pytest

.PHONY: setup data data-contracts-only data-features-only model app test test-v all clean clean-all help

# ── Setup ──────────────────────────────────────────────────────────────────────

setup:
	@echo "Creating venv and installing dependencies..."
	python3.12 -m venv $(VENV)
	$(PIP) install --upgrade pip setuptools wheel
	$(PIP) install -r requirements.txt
	$(PIP) install pytest
	@echo ""
	@echo "Setup complete. Next steps:"
	@echo "  make data     — collect data"
	@echo "  make model    — train models"
	@echo "  make app      — launch app"
	@echo "  make test     — run tests"

# ── Pipeline ───────────────────────────────────────────────────────────────────

data:
	@echo "Running data pipeline..."
	@echo "Positions: $(POSITIONS)  |  Start year: $(START_YEAR)"
	$(PYTHON) run_pipeline.py --positions $(POSITIONS) --start-year $(START_YEAR)
	@echo "Data pipeline complete. CSVs written to data/raw/ and data/processed/"

data-contracts-only:
	@echo "Scraping contracts only (no stats)..."
	$(PYTHON) run_pipeline.py --positions $(POSITIONS) --start-year $(START_YEAR) --contracts-only

data-features-only:
	@echo "Rebuilding features from existing CSVs..."
	$(PYTHON) run_pipeline.py --features-only

# ── Modelling ──────────────────────────────────────────────────────────────────

model:
	@echo "Training models (APY, guaranteed, contract years, quantile + CQR calibration)..."
	$(PYTHON) notebooks/model.py
	@echo "Models saved to models/"
	@echo "Evaluation diagnostics saved to models/evaluation.json (includes CQR corrections)"

# ── App ────────────────────────────────────────────────────────────────────────

app:
	@echo "Starting Streamlit app..."
	$(VENV)/bin/streamlit run app.py

# ── Tests ──────────────────────────────────────────────────────────────────────

test:
	@echo "Running tests..."
	$(PYTEST) tests/

test-v:
	@echo "Running tests (verbose)..."
	$(PYTEST) tests/ -v

# ── Combined ───────────────────────────────────────────────────────────────────

all: data model
	@echo ""
	@echo "Pipeline and training complete. Run 'make app' to launch."

# ── Cleanup ────────────────────────────────────────────────────────────────────

clean:
	@echo "Clearing cache and processed data..."
	rm -rf data/.cache/
	rm -rf data/processed/
	rm -rf __pycache__/ scrapers/__pycache__/ notebooks/__pycache__/ tests/__pycache__/
	rm -rf .pytest_cache/
	@echo "Done. Raw CSVs and models preserved."

clean-all: clean
	@echo "Removing all generated data including raw CSVs and models..."
	rm -rf data/raw/
	rm -rf models/*.pkl models/*.json
	@echo "Done. Re-run 'make data' then 'make model' to regenerate."

# ── Help ───────────────────────────────────────────────────────────────────────

help:
	@echo ""
	@echo "NFL Contract Predictor"
	@echo "======================"
	@echo ""
	@echo "Setup:"
	@echo "  make setup                          Create venv and install all dependencies"
	@echo ""
	@echo "Data pipeline:"
	@echo "  make data                           Full pipeline — contracts + stats + features"
	@echo "  make data POSITIONS='QB WR' START_YEAR=2018   Custom positions/year"
	@echo "  make data-contracts-only            Scrape contracts only, skip stats"
	@echo "  make data-features-only             Rebuild features from existing CSVs"
	@echo ""
	@echo "Modelling:"
	@echo "  make model                          Train all models (APY, guaranteed, years, quantile + CQR)"
	@echo "                                      Saves evaluation.json with CQR corrections + residual diagnostics"
	@echo ""
	@echo "App:"
	@echo "  make app                            Launch Streamlit app at localhost:8501"
	@echo ""
	@echo "Tests:"
	@echo "  make test                           Run all unit tests"
	@echo "  make test-v                         Run tests with verbose output"
	@echo ""
	@echo "Combined:"
	@echo "  make all                            Run data pipeline then train models"
	@echo ""
	@echo "Cleanup:"
	@echo "  make clean                          Clear cache, processed files, and pycache"
	@echo "  make clean-all                      Remove all generated data and model files"
	@echo ""
