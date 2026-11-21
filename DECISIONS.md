# Design decisions

Format: date — decision — reason.

## Phase 0 – Setup
- 2026-11-21 — Python 3.11 venv; versions pinned in `requirements.txt`; package installed in editable mode via `pyproject.toml`. — Gives reproducible installs, and `import procure` works from tests, scripts and Streamlit.
- 2026-11-21 — Added `src/procure/domain.py` (dataclasses), `store.py` (data access at request time), `service.py` (persistence and audit), `sampling.py`, `backtest.py`, `figures.py`, `reporting.py`, `user_study.py` beyond the listed structure. — These are shared by several layers. Keeping them separate avoids circular imports and duplicated code.

## Phase 1 – Data
- 2026-11-21 — **Simulated raw data generator** (`demo_raw.py`) is used when `data/raw` is empty. — No real NOCOPO/ACLED files were available while building. The generator writes files in the same formats the loader expects and injects realistic messiness (name variants, mixed date formats, USD rows, duplicates, missing values), so the cleaning code is exercised. **Every result in `experiments/results` is illustrative until re-run on real data.**
- 2026-11-21 — LGA coordinates are approximate headquarters positions (6 LGAs per state) in `config/geo.yaml`. — Distance and the 50 km conflict radius need coordinates. Replace them with official NBS/OSGOF centroids and extend to all 112 LGAs for the final run.
- 2026-11-21 — Dates: ISO strings are parsed year-first and everything else day-first (dd/mm/yyyy). — A single `dayfirst=True` parse silently swapped day and month in ISO dates. A unit test caught this.
- 2026-11-21 — supplier_id = salted SHA-256 of (normalised name, canonical state). — This is the "name + state match" in the spec. Hashing anonymises the name (NDPA 2023). Name normalisation maps Limited→ltd, Enterprises→ent, and so on.
- 2026-11-21 — Suppliers missing from the registry are treated as non-compliant. — This is conservative: the rule layer then excludes them *with a stated reason*, rather than letting unknown suppliers through.
- 2026-11-21 — Outcome-based supplier history (on-time rate, failure rate, quality) counts a past contract only after signing date + 30 days + its delay. — A contract signed last week has no known outcome yet. Using it would leak future information even though its signing date is earlier.
- 2026-11-21 — Supplier rates are Bayesian-shrunk toward priors (pseudo-count 5). — New suppliers would otherwise show rates of 0% or 100%.
- 2026-11-21 — Item reference prices (and `past_price_index`, `log_item_ref_price`) come from training years only. — Prevents test-period price information leaking into features.
- 2026-11-21 — Pipeline order: context features → synthetic outcomes → history features. — Synthetic outcomes must be conditioned on season, conflict and distance (context), and history features need the outcomes.
- 2026-11-21 — Rows whose outcomes are synthetic in 2024 go to `test_synthetic.parquet`. "Full" metrics = test + test_synthetic. — This satisfies both "no synthetic rows in test when real outcomes exist" and "report real-only and full".
- 2026-11-21 — Macro variables (fuel price, exchange rate) are loaded but not used as model features. CPI is used to deflate costs to 2024 naira. — The 2023 fuel-subsidy removal and naira float put 2024 values far outside the training range, and tree models cannot extrapolate.

## Phase 2 – Rules
- 2026-11-21 — Rule conditions are expressions evaluated by an AST-whitelist evaluator (no function calls, no private attributes). — Administrators can edit `rules.yaml` without touching code, and rules cannot run arbitrary code.
- 2026-11-21 — Added R6 (zero eligible suppliers → flag). R4 uses `params.min_eligible_suppliers`. — Explains the edge case to the officer and keeps thresholds out of code.
- 2026-11-21 — Method value thresholds in `rules.yaml` are ILLUSTRATIVE. — Verify against the current BPP thresholds circular before the final run.
- 2026-11-21 — Bid value = supplier's quote if supplied, otherwise item reference price × exp(supplier past price index) × Q. — R2 needs a value before predictions run.

## Phase 3 – Prediction
- 2026-11-21 — The cost model is trained on log(unit_cost_real). — Prices span four orders of magnitude, so a log target gives relative errors. Narration turns log contributions into % effects.
- 2026-11-21 — Added feature `log_item_ref_price`. — Trees otherwise have to learn each item's price level from one-hot columns, and linear regression beat XGBoost on cost. This is a legitimate feature, known at request time and taken from training years.
- 2026-11-21 — Failure model: scale_pos_weight = neg/pos, then probabilities corrected with p = s/(s + w(1−s)). — Re-weighting inflates the odds by w. The optimiser's f4 and the Brier score need calibrated probabilities. Logistic regression (class-balanced) gets the same correction, so the comparison is fair.
- 2026-11-21 — Optuna objective: mean MAE (regression) or 1−AUC (classification) over three expanding-window folds (train→2021, →2022, →2023). The preprocessor is refit inside each fold. — Time-ordered CV, with no leakage from the preprocessor.
- 2026-11-21 — MAPE excludes rows with |y| < 1. — Many delays are 0–1 days, and MAPE explodes there. MAE/RMSE remain the primary metrics.
- 2026-11-21 — Added a paired bootstrap (1,000 resamples) of XGBoost-minus-baseline. — Shows whether a "win" or "loss" is statistically meaningful.
- 2026-11-21 — **Exit-check result (simulated data):** XGBoost beats every baseline on cost MAE and delay MAE. On failure AUC it is 0.006 *below* logistic regression, and the bootstrap 95% CI includes 0 (not significant). Explanation: the simulated failure process is close to additive in its drivers (distance, conflict, supplier reliability), so a linear logit model is correctly specified, and the noise ceiling (AUC ≈ 0.72) leaves no room for non-linear gains. Optuna chose depth-2 trees, which are nearly additive. On real data with interactions (e.g. rainy season × remote LGA) the ranking should be re-checked. Do not tune until XGBoost wins.

## Phase 4 – Optimisation
- 2026-11-21 — Predictions are precomputed on a share grid (5%…100% of Q) and linearly interpolated. — Quantity affects unit cost and risk. This avoids running XGBoost inside every NSGA-II evaluation (each run takes about 2 s).
- 2026-11-21 — Repair: keep the top-k shares, drop shares below 5%, renormalise. The initial population contains every single-supplier solution plus random sparse mixes. The random-search baseline uses the same sampler. — Enforces sum = 1 and ≤ k suppliers, and gives every method the same start.
- 2026-11-21 — NSGA-II keeps an **external archive** of all evaluated points. Every method's non-dominated set is truncated by crowding distance to 300 points. — Without the archive, NSGA-II returned only its 100-point final population while random search returned the non-dominated set of all 20,000 samples, which is an unfair comparison. The cap keeps the metrics affordable: an uncapped spread computation needed about 6 GB.
- 2026-11-21 — Hypervolume is computed after min–max normalisation over the union of all methods' fronts per request, with reference point 1.1. Spread = generalised spread (Zhou et al. 2006). Significance = Wilcoxon signed-rank test over requests.
- 2026-11-21 — **Exit-check result (simulated data):** NSGA-II beats lowest-bid and weighted-sum GA by a wide margin (p < 1e-8). In the *standard* setting (≤ 15 candidates, k = 3, about 575 possible supplier subsets) it ties random search (difference −0.001, p = 0.23), because 20,000 random samples already cover that small space densely. In the *large* setting (≤ 60 candidates, k = 5) NSGA-II beats random search on 93% of requests (p ≈ 2e-6). Report both; the advantage grows with the size of the search space.

## Phase 5 – Explanation and application
- 2026-11-21 — SHAP is computed on one-hot columns and summed back to original features. — Officers see "Item category", not "item_category_fuel".
- 2026-11-21 — Cost narration excludes item identity features (item, category, reference price) and compares with the typical price for the item. — These explain the price *level*, not differences between suppliers, and would crowd out useful reasons.
- 2026-11-21 — Effects are shown in natural units: % of price, days, and percentage points of failure risk (computed by removing the feature's log-odds contribution and re-applying the probability correction).
- 2026-11-21 — LIME runs on the same transformed matrix. Its weight × standardised deviation gives a local contribution, summed to original features. — This makes it comparable to SHAP attributions. Agreement is about 0.71–0.76 top-3 overlap.
- 2026-11-21 — Added an `app_user` table and token auth (itsdangerous, already a Flask dependency). — Needed for login and role-based access. Audit rows store user_id + role only.
- 2026-11-21 — Override cannot select a supplier excluded by a compliance rule. Override and return require a reason. Only one decision per recommendation. — The human decides, but cannot bypass PPA 2007 exclusions, and every deviation is justified.
- 2026-11-21 — The dashboard calls the same service layer as the API (not HTTP). — One code path for auditing, and a single process for the demo.

## Phase 6 – Evaluation
- 2026-11-21 — The back-test compares the top-ranked option with the actual award, both scored by the same models. — The counterfactual outcome of a supplier that was not chosen cannot be observed, so savings are *expected* savings. The realised failure rate of actual awards is reported next to them as a calibration check.
- 2026-11-21 — Figures use a validated colour-blind-safe categorical order, a single y-axis per chart, and 300 dpi.
