# NFL Contract Predictor — Makefile
# ==================================
# Two-venv workflow:
#   venv_data  — Python 3.12 + pandas<2.0 + nfl_data_py (data collection only)
#   venv312    — Python 3.12 + pandas 2.x  (modelling + app)
#
# Usage:
#   make setup        — create both venvs and install dependencies
#   make data         — run the full data pipeline (venv_data)
#   make model        — train models (venv312)
#   make app          — launch Streamlit app (venv312)
#   make all          — data → model → app in one command
#   make clean        — remove cached data (keeps raw CSVs and models)
#   make clean-all    — remove all generated data including CSVs and models

POSITIONS   ?= QB WR RB TE
START_YEAR  ?= 2015

VENV_DATA   = venv_data
VENV_APP    = venv312
PYTHON_DATA = $(VENV_DATA)/bin/python
PYTHON_APP  = $(VENV_APP)/bin/python
PIP_DATA    = $(VENV_DATA)/bin/pip
PIP_APP     = $(VENV_APP)/bin/pip

.PHONY: setup setup-data setup-app data model app all clean clean-all help

# ── Setup ──────────────────────────────────────────────────────────────────────

setup: setup-data setup-app
	@echo ""
	@echo "Both venvs ready."
	@echo "  Data pipeline:  make data"
	@echo "  Train models:   make model"
	@echo "  Run app:        make app"

setup-data:
	@echo "Creating venv_data (nfl_data_py + data pipeline deps)..."
	python3.12 -m venv $(VENV_DATA)
	$(PIP_DATA) install --upgrade pip setuptools==69.5.1 wheel
	$(PIP_DATA) install -r requirements-data.txt
	$(PIP_DATA) install nfl_data_py --no-deps
	@echo "venv_data ready."

setup-app:
	@echo "Creating venv312 (modelling + Streamlit app deps)..."
	python3.12 -m venv $(VENV_APP)
	$(PIP_APP) install --upgrade pip setuptools wheel
	$(PIP_APP) install -r requirements-app.txt
	@echo "venv312 ready."

# ── Pipeline ───────────────────────────────────────────────────────────────────

data:
	@echo "Running data pipeline (venv_data)..."
	@echo "Positions: $(POSITIONS)  |  Start year: $(START_YEAR)"
	$(PYTHON_DATA) run_pipeline.py --positions $(POSITIONS) --start-year $(START_YEAR)
	@echo "Data pipeline complete. CSVs written to data/raw/ and data/processed/"

data-contracts-only:
	@echo "Scraping contracts only (no stats)..."
	$(PYTHON_DATA) run_pipeline.py --positions $(POSITIONS) --start-year $(START_YEAR) --contracts-only

data-features-only:
	@echo "Rebuilding features from existing CSVs..."
	$(PYTHON_DATA) run_pipeline.py --features-only

# ── Modelling ──────────────────────────────────────────────────────────────────

model:
	@echo "Training models (venv312)..."
	$(PYTHON_APP) notebooks/model.py
	@echo "Models saved to models/"

# ── App ────────────────────────────────────────────────────────────────────────

app:
	@echo "Starting Streamlit app (venv312)..."
	$(VENV_APP)/bin/streamlit run app.py

# ── Combined ───────────────────────────────────────────────────────────────────

all: data model
	@echo ""
	@echo "Pipeline and training complete. Run 'make app' to launch."

# ── Cleanup ────────────────────────────────────────────────────────────────────

clean:
	@echo "Clearing cache and processed data..."
	rm -rf data/.cache/
	rm -rf data/processed/
	rm -rf __pycache__/ scrapers/__pycache__/ notebooks/__pycache__/
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
	@echo "  make setup             Create both venvs and install all dependencies"
	@echo "  make setup-data        Create venv_data only (nfl_data_py + old pandas)"
	@echo "  make setup-app         Create venv312 only (streamlit + new pandas)"
	@echo ""
	@echo "Data pipeline (runs in venv_data):"
	@echo "  make data              Full pipeline — scrape contracts + stats + build features"
	@echo "  make data POSITIONS='QB WR' START_YEAR=2018   Custom positions/year"
	@echo "  make data-contracts-only   Scrape contracts only, skip stats"
	@echo "  make data-features-only    Rebuild features from existing CSVs"
	@echo ""
	@echo "Modelling (runs in venv312):"
	@echo "  make model             Train all models, save to models/"
	@echo ""
	@echo "App (runs in venv312):"
	@echo "  make app               Launch Streamlit app at localhost:8501"
	@echo ""
	@echo "Combined:"
	@echo "  make all               Run data pipeline then train models"
	@echo ""
	@echo "Cleanup:"
	@echo "  make clean             Clear cache and processed files"
	@echo "  make clean-all         Remove all generated data and model files"
	@echo ""
