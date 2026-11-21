# Procurement XAI - one-command workflow. All parameters live in config/*.yaml.
PYTHON_BIN ?= python3.11
VENV := .venv
PY := $(VENV)/bin/python
export PYTHONHASHSEED := 42

.PHONY: setup test data rules-demo train optimise explain experiments api dashboard demo clean

setup:            ## Create virtualenv and install pinned dependencies
	test -d $(VENV) || $(PYTHON_BIN) -m venv $(VENV)
	$(PY) -m pip install --quiet --upgrade pip
	$(PY) -m pip install --quiet -r requirements.txt
	$(PY) -m pip install --quiet -e . --no-deps

test:             ## Run the fast pytest suite
	$(PY) -m pytest

data:             ## Raw -> clean -> features -> synthetic outcomes -> temporal split
	$(PY) -m procure.data.build

rules-demo:       ## Print rule-layer reasons for an example request
	$(PY) -m procure.rules.demo

train:            ## Optuna-tune and train the three XGBoost models + baselines, evaluate on 2024
	$(PY) -m procure.models.train
	$(PY) -m procure.models.evaluate

optimise:         ## NSGA-II vs baselines on sampled 2024 requests
	$(PY) -m procure.optimise.benchmark

explain:          ## SHAP global plots + LIME agreement check
	$(PY) -m procure.explain.lime_check

demo:             ## One end-to-end recommendation printed to the terminal
	$(PY) -m procure.pipeline

experiments: setup ## Reproduce every thesis number/table/figure from raw data
	$(PY) experiments/run_all.py

api:              ## Flask REST API on :5000
	$(PY) -m procure.api.app

dashboard:        ## Streamlit dashboard on :8501
	$(VENV)/bin/streamlit run dashboard/streamlit_app.py

clean:
	rm -rf data/interim/* data/processed/* models/* experiments/results/* data/*.db
	touch data/interim/.gitkeep data/processed/.gitkeep models/.gitkeep experiments/results/.gitkeep
