# procure-xai

**An Explainable AI-Based Procurement Optimization Model for Improving Supply Chain Efficiency in the
North-East Zone, Nigeria** (Adamawa, Bauchi, Borno, Gombe, Taraba, Yobe) — M.Sc. Computer Science thesis
software, Modibbo Adama University, Yola.

For a procurement request, the system returns 3–5 ranked supplier options, showing their cost, delivery-time,
quality and reliability trade-offs. Each option comes with plain-English explanations. **A human procurement officer
always makes the final decision**, and every recommendation, explanation, fired rule and decision is written to an
audit log.

## Architecture

| Layer | What it does | Code |
|---|---|---|
| 1. Rules | PPA 2007 compliance filter from `config/rules.yaml` (R1–R6, exclude/flag, reasons logged) | `src/procure/rules/engine.py` |
| 2. Prediction | XGBoost cost (log ₦ 2024), delay (days), failure (classifier) sharing one `ColumnTransformer`; Optuna (50 trials) with expanding-window time-series CV | `src/procure/models/` |
| 3. Optimisation | NSGA-II (pymoo) splits quantity Q across ≤ k suppliers; f1 cost, f2 worst delay, f3 −quality, f4 P(any failure); budget constraint; repair operator | `src/procure/optimise/` |
| 4. Explanation | Lowest-cost / fastest / most-reliable / balanced options; top-3 SHAP reasons per supplier as sentences; LIME agreement check | `src/procure/explain/`, `src/procure/pipeline.py` |
| App | Flask REST API with role-based access; Streamlit dashboard; SQLAlchemy (SQLite dev, MySQL prod) | `src/procure/api/`, `dashboard/`, `src/procure/db/` |

## Quick start

```bash
make setup        # Python 3.11 venv + pinned requirements
make test         # fast pytest suite (~15 s)
make data         # raw -> clean -> features -> synthetic outcomes -> train/val/test parquet
make train        # Optuna-tune + train 3 models, evaluate vs baselines on 2024 (5 seeds)
make optimise     # NSGA-II vs weighted-sum GA, random search, lowest bid (30 requests x 2 settings)
make explain      # SHAP global plots + SHAP-LIME agreement
make demo         # one end-to-end recommendation printed in the terminal
make experiments  # EVERYTHING above + back-test, latency, figures, user-study pack (~15 min)
make api          # REST API on http://127.0.0.1:5000
make dashboard    # Streamlit dashboard on http://localhost:8501
```

Demo accounts (role accounts, no personal names): `officer1`, `logistics1` and `admin1`. Their password is taken from
`PROCURE_DEMO_PASSWORD` (default `change-me`). In production, set `PROCURE_SECRET_KEY` and `PROCURE_ANON_SALT`.

### Data

Put real files in `data/raw/` (they are never committed):

| Pattern | Content |
|---|---|
| `contracts*.csv` / `ocds*.json` | NOCOPO exports (flat CSV, or OCDS release packages) |
| `supplier_registry*.csv` | supplier_name, state, lga, registered, debarred, tax_clearance, cac |
| `acled*.csv` | ACLED events (event_date, admin1, latitude, longitude, fatalities) |
| `macro*.csv` | month, cpi, exchange_rate_ngn_usd, fuel_price_ngn_litre |

**If `data/raw/` is empty, `make data` generates SIMULATED raw data** (`procure.data.demo_raw`, documented in
`config/demo_catalog.yaml`) and prints a warning. Every number produced this way is illustrative only. Re-run
`make experiments` on real data before reporting results in the thesis.

### API

```
POST /login                 {"username","password"} -> {"token","role"}
POST /recommend             {"item","item_category","quantity","delivery_state","delivery_lga","method","budget","date"}
GET  /recommendation/<id>
POST /decision              {"recommendation_id","action":"accept|override|return","option_rank","reason"}
GET  /audit?recommendation_id=&event_type=&limit=      (administrator)
```
Send `Authorization: Bearer <token>`. The permissions for each role live in `config/settings.yaml` under `api.permissions`.

## Reproducibility rules enforced in code
- Temporal split only: train 2019–2022, validation 2023, test 2024. Supplier history features use only earlier
  transactions (outcome-based rates only once the outcome would have been known). Tests in `tests/test_data.py`.
- Seed 42 everywhere. All parameters are in `config/*.yaml`.
- Synthetic-outcome rows are flagged `is_synthetic` and never placed in `test.parquet` (they go to
  `test_synthetic.parquet`). Metrics are reported for the real-only and the full test set.
- NDPA 2023: supplier and buyer names are replaced with salted hashes. Audit logs hold IDs and roles only.

See `DECISIONS.md` for the reasoning behind each design choice, and `docs/data_dictionary.csv` for every column.
