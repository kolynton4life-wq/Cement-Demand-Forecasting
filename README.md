# MIG Cement Demand Forecasting

A per-site cement demand forecasting and inventory optimization system for
Midlands Infrastructure Group (MIG), covering 30 active UK construction sites.
Predicts demand up to 8 weeks ahead, drives dynamic reorder points, and
surfaces everything through a live dashboard.

**Start here** this README walks through what
was built, what was found, and what still needs real data before production
use. Full detail lives in [`docs/project_documentation.md`](docs/project_documentation.md).




#  Feature Engineering
MIG Cement Demand Forecasting

Per the project spec: *"Create lag features, rolling aggregates, and
interaction variables. Engineer weather-adjusted pour indicators and
calculate inventory turnover metrics to capture site-specific operational
characteristics for modeling."*

Every feature here is justified by a specific finding's

| EDA finding | Feature response |
|---|---|
| `behavior` is the dominant signal (65% vs 14% stockout-risk) | Keep as categorical; add interaction terms |
| Heavy rain (15mm+) crashes consumption non-linearly | Threshold flag, not raw `rain_mm` |
| `avg_temp_c` has ~0 correlation | Dropped no signal |
| Planned-vs-actual adherence varies sharply by behavior (0.485–0.983) | Lag/smooth planned pour instead of using it raw |
| No day-of-week effect | No day-of-week feature |
| Series is stationary (ADF p≈0) | No differencing feature needed for tree models |
| Region is a weak, small-sample signal | Keep as filter, not a primary feature |

Reads `data/processed/operations_clean.parquet` (output), writes
`data/processed/model_features.parquet`.
---

## Result at a glance

| Target (per project brief) | Result | Status |
|---|---|---|
| Forecast MAPE ≤ 15% | **3.3%** (holdout), 4.6% (rolling validation) | ✅ |
| ≥98% pour readiness | 96.5% average, 10/30 sites individually ≥98% | ⚠️ Below target see [§7](#step-7--validation--deployment) |
| 20% silo utilization improvement | +39.6 percentage points in healthy-band time | ✅ |
| 30% write-off reduction | 0 overflow tonnes (structural guarantee) | ✅ |

---

## Project structure

```
cement-forecasting/
├── notebooks/              Steps 1-5, exploratory + documented
│   ├── 01_data_ingestion.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_model_development.ipynb
│   └── 05_inventory_simulation.ipynb
├── src/
│   ├── data/                connect.py, extract.py — SQLite access
│   ├── preprocessing.py     production port of notebooks 01+03
│   ├── pipeline.py          Step 7: train + validate + persist models
│   ├── inference.py         Step 7: load a model, forecast (no retrain)
│   ├── monitoring.py        Step 7: live accuracy tracking + retrain trigger
│   ├── model_registry.py    save/load model artifacts
│   ├── api.py                FastAPI serving layer
│   └── dashboard/
│       ├── app.py            Plotly Dash (project spec deliverable)
│       └── streamlit_app.py  Streamlit (this project's deployment target)
├── test/                    25 pytest tests
├── models/                  trained model artifacts (gitignored — run pipeline.py)
├── data/                    raw + processed (gitignored — run notebooks/pipeline)
├── docs/
│   ├── data_dictionary.md
│   └── project_documentation.md   full methodology + performance + insights
├── assets/                  charts used in this README
├── Dockerfile.model / .api / .dashboard / .streamlit
├── docker-compose.yaml
└── requirements*.txt
```

## Quickstart

```

# 1. Copy MIG_Cement_Records.db into data/raw/, then run notebooks in order:
#    01_data_ingestion -> 02_eda -> 03_feature_engineering
#    -> 04_model_development -> 05_inventory_simulation

# 2. Train and persist production models
python src/pipeline.py

# 3. Run the dashboard (either)
python src/dashboard/app.py                          # Dash, http://127.0.0.1:8050
streamlit run src/dashboard/streamlit_app.py          # Streamlit, http://127.0.0.1:8501

# 4. Or serve predictions via API
uvicorn src.api:app --reload --port 8000              # docs at /docs

# Run the test suite
pytest test/ -v
```

---

## Step 1 Data Ingestion & Cleaning

**Task**: import from SQLite, validate schema, handle missing values, fix
negative/invalid entries, ensure the inventory balance equation holds.

**What was found**: the real database is normalized (`Sites`, `CementTypes`,
`Operations` — 32,880 rows, 30 sites, 3 years daily, 2022-2024) rather than
the single flat table the spec described  functionally equivalent, joined
in `extract.py`. Zero nulls, zero true negatives, balance equation held to
floating-point rounding. One real data-quality issue: **34.8% of rows**
(concentrated in `conservative`-behavior sites) showed closing inventory up
to ~38x silo capacity a synthetic-data generation artifact, capped at
capacity with the excess tracked separately rather than silently discarded.

**Preparation for later steps**: `src/preprocessing.py` was later built as a
production port of this notebook's logic and verified **byte-for-byte
identical** to the notebook's output on the real dataset before anything
was built on top of it.

---

## Step 2  Exploratory Data Analysis

**Task**: demand patterns by site/type/time, seasonality, weather
correlation, planned-vs-actual impact, outlier detection.

**Key finding behavior segmentation dominates everything else.** Sites
split into `aggressive` (JIT ordering), `chaotic` (erratic), and
`conservative` (overstocking) and this pattern shaped every later
modeling decision.


![Behavior segmentation](docs/assets/01_behavior_segmentation.png)

Network-wide demand over the full 3-year history, used to check for
seasonality (found: stationary series, no strong annual cycle) and to
choose the forecast granularity:

![Weekly demand trend](docs/assets/02_weekly_demand_trend.png)

**Weather — the linear correlation is misleading.** Raw correlation between
rainfall and consumption is only -0.18, which looks negligible:

![Weather correlation matrix](docs/assets/03_weather_correlation.png)

But banding rainfall into none/light/moderate/heavy revealed a sharp
**threshold effect** hidden by that single correlation number heavy rain
(≥15mm) crashes consumption to ~4 t/day (vs. ~24-25 normally) while planned
pours stay unchanged, pushing stockout risk to 64%. This directly drove a
`heavy_rain_flag` feature in Step 3 instead of using `rain_mm` as a linear
input the correlation number alone would have led to dropping a real
signal.

**Planned-pour reliability tracks behavior exactly**, visible directly in
the data:

![Planned vs actual by behavior](docs/assets/04_planned_vs_actual.png)

`conservative` sites (blue) hug the perfect-adherence line; `aggressive`
(orange) and `chaotic` (green) scatter far below it meaning
`planned_pour_tonnes` could not be trusted uniformly as a model input.

---

## Step 3 Feature Engineering

**Task**: lag features, rolling aggregates, weather-adjusted pour
indicators, inventory turnover metrics, interaction variables.

**Two real bugs caught and fixed, not left in:**

1. **Grain-fragmentation bug**  lag/rolling features were initially
   grouped by `(site_id, cement_type)`. Each site actually logs exactly
   one row per calendar day, with `cement_type` rotating as a same-day
   attribute grouping by both fragmented each site's continuous daily
   series into three irregularly-spaced sub-series. Fixed to group by
   `site_id` alone; verified `consumed_lag_1d` matches literal yesterday's
   value for every one of 1,095 checked rows.
2. **Non-informative interaction** an early `behavior_x_planned_pour`
   feature multiplied every behavior by the same constant (1.0), making it
   mathematically identical to `planned_pour_tonnes` and useless as an
   interaction. Fixed to genuine `continuous x dummy` interactions.

**Leakage audit**: any feature mathematically derived from the same-day
target (`stockout_risk`, `days_of_stock`, `closing_inventory_tonnes`, etc.)
is explicitly excluded from the model's input feature list.

---

## Step 4  Model Development

**Task**: SARIMAX baseline with external regressors vs. Random Forest with
exogenous variables, evaluated by MAPE/RMSE, best model selected.

**Finding: Random Forest wins on all 30/30 sites.**

![Model comparison](docs/assets/05_model_comparison_mape.png)

| Model | Mean MAPE | Sites passing ≤15% |
|---|---|---|
| Naive baseline | 40.6% | 0/30 |
| SARIMAX + exogenous | 24.9% | 14/30 |
| **Random Forest** | **3.3%** | **30/30** |

SARIMAX specifically struggled on `aggressive`-behavior sites (40.5% MAPE
*worse* than naive there), consistent with Step 2's finding that those
sites are the most erratic and threshold-driven exactly what a linear
model can't represent but a tree-based model handles naturally:

![Forecast example](docs/assets/06_forecast_example.png)

Random Forest (green) tracks the actual series (black) closely, including
the sharp drops to zero; SARIMAX (orange) lags and occasionally forecasts
impossible negative consumption.

---

## Interaction variables

One-hot encode `behavior` and `cement_type` **first**, then build
interaction terms from the resulting numeric dummy columns.
1. SARIMAX (Step 4 baseline) needs numeric exogenous regressors a raw
   string interaction column like `"aggressive_1"` isn't usable by it.
2. Genuine interactions must actually vary with the categorical level.
   An earlier draft of multiplied `planned_pour_tonnes` by
   the same constant (1.0) for every behavior that's mathematically
   identical to `planned_pour_tonnes` itself, not an interaction. the interactions below are computed as
   `continuous_feature x dummy_column`, so they're zero for rows outside
   that category and equal to the continuous value inside it a
   standard, genuinely non-redundant way to let a linear model learn a
   different slope per category.


## Step 5 Inventory Simulation

**Task**: forecast silo levels, define dynamic reorder points per site
based on forecast demand, lead time, and silo capacity.

**Result**: a continuous-review (s, S) policy per site produces the
expected sawtooth inventory pattern:

![Inventory simulation](docs/assets/07_inventory_simulation.png)

**Two metrics were corrected before being reported**, not taken at face
value:
- **Silo utilization**: a naive %-change calculation produced nonsense
  (+918% for some sites) against a bimodal historical baseline. Replaced
  with "% of days in a healthy 30-80% band" 8.1% → 47.7%, a real result.
- **Write-off reduction**: the simulated 0-tonnes overflow is a
  **structural guarantee** of capping the order-up-to level at silo
  capacity, not an empirically discovered 100% improvement reported to
  stakeholders as a design property, not a statistic.




Per the project spec "Forecast silo levels using predicted demand,
scheduled deliveries, and opening inventory positions. Define dynamic
reorder points for each site based on forecasted demand, lead times, and
silo capacity constraints."

**Assumption flagged up front** the project's data dictionary does not
include supplier lead-time data anywhere in the source database or
documentation (already noted as an open question in the audit).
`LEAD_TIME_DAYS` below is a stated placeholder assumption (3 days,
typical for UK domestic bulk cement delivery), not derived from data.
**This must be replaced with real supplier lead-time data from MIG
procurement before this simulation is used for actual ordering
decisions** treat every number in this notebook as illustrative of the
*methodology*, not yet production-accurate.

**Approach**: for each site, refit the winning model (Random
Forest won on all 30/30 sites) on its **full** history, forecast 56
days beyond the end of the dataset, then run a continuous-review (s, S)
reorder-point simulation when inventory position (on-hand + on-order)
drops to or below the reorder point, place an order that arrives after
the lead time.


## Reorder-point simulation (continuous review, s/S policy)

- **Reorder point (s)**: `avg_daily_demand x lead_time + safety_stock`
- **Safety stock**: `z x forecast_RMSE x sqrt(lead_time)` ses each
  site's Step 4 Random Forest holdout RMSE as the forecast-uncertainty
  estimate. **Caveat**: backtest RMSE was very low (RF's MAPE averaged
  3.3%), which may understate real-world forecast error once weather/pour
  inputs are genuinely unknown (vs. this notebook's seasonal-proxy
  assumption) treat the resulting safety stock as a lower bound, and
  recalibrate from live forecast-error monitoring once deployed
  ( stated purpose).
- **Order-up-to level (S)**: `ROP + one review period of demand`, capped
  at `silo_capacity` the physical constraint the project brief
  explicitly requires respecting.


---

## Step 6 Dashboard Application

**Task**: Plotly Dash app with forecasts, inventory projections, reorder
alerts, site drill-down and aggregate views.

Built and **tested live** (server start, real HTTP requests, and an actual
interactive callback test not just page load): `src/dashboard/app.py`.

A Streamlit version (`streamlit_app.py`) was added alongside it for this
project's deployment target, kept in feature parity deliberately, and
verified with Streamlit's `AppTest` framework including a real interactive
site-switch test.

## findings

- **Behavior segmentation matters**: `aggressive` sites show the highest
  stockout-risk rate; `conservative` sites show structural overstocking
  (see Phase 1 audit capacity-capping applied). Confirms behavior should
  be a modeling feature and/or basis for per-segment strategy in Phase 3.
- **Seasonality**: (inspect monthly/decomposition charts above and note
  whether a clear annual cycle exists  winter cement curing constraints
  are a common real-world pattern to look for).
- **Weather**: (note correlation strength and whether the heavy-rain band
  shows a meaningfully different stockout rate decides whether rain_mm
  earns a place as a regressor in SARIMAX/LightGBM, per Phase 3 plan).
- **Planned vs. actual adherence**: (note the correlation value if well
  below ~0.7-0.8, planned_pour_tonnes should be engineered as a soft
  signal, e.g. lagged/smoothed, not used directly as a forecast input).
- **Outliers**: (note % flagged and whether concentrated in specific
  sites worth a follow-up conversation with site managers before Phase
  3 if concentrated, as it may indicate real operational events rather
  than noise).
- **Inventory buffers**: (note which sites show lowest days-of-stock
  these are your highest-priority candidates for the Phase 6 pilot).

---

## Step 7 Validation & Deployment

**Task**: validate against holdout data, deploy the pipeline to a
production environment, establish a monitoring framework with a
retraining trigger.

- **Rolling-window validation**: Step 4 used one fixed holdout window.
  `src/pipeline.py` re-validates every site across 3 independent
  walk-forward windows 4.6% mean MAPE, worst site 8.75%, confirming
  Step 4's result wasn't specific to one lucky period.
- **Production pipeline**: `pipeline.py` (train + persist) →
  `inference.py` (load and predict **without** retraining verified ~2s
  vs. ~5s/site to retrain) → `monitoring.py` (live MAPE tracking against
  the 15% target, with a tested retrain-trigger) → `api.py` (FastAPI,
  every endpoint tested including error paths).
- **Test suite**: 25 pytest tests. **The suite caught a real bug** during
  development `consumed_tonnes` could exceed physically available
  supply on adversarial synthetic data, producing negative closing
  inventory. Fixed at the source (cap consumption at what was actually
  available) rather than patching the symptom, then back-ported the same
  fix to the Step 1 notebook for consistency.
- **Docker**: `Dockerfile.model/api/dashboard/streamlit` +
  `docker-compose.yaml` written to standard conventions but **not
  build-verified** — no Docker daemon was available during development.
  Verify locally before relying on them.

**Pour readiness gap, stated honestly**: simulated readiness is 96.5%, not
the 98% target. Two causes identified a cold-start effect (sites whose
real ending inventory was already below the reorder point show stockouts
before the first order arrives), and safety stock built from backtest RMSE
that's likely optimistic once real future weather/schedule data replaces
the proxy used here. `monitoring.py` is the mechanism to close this gap
recalibrate safety stock from live forecast-error data once deployed.

---

## Summary of findings

- Both SARIMAX and Random Forest beat the naive baseline on average
  confirms the engineered features carry real signal.
- MAPE target (<=15%) achievement see pass-rate in Section 7  note
  whether failures cluster in a particular behavior segment (chaotic
  sites, with high variance, are the hardest case for any model).
- Model selection is per-site, not universal see Section 8.
- Known limitation: both models assume future planned_pour_tonnes and
  weather values are known in advance (MIG's schedule, weather forecast
  API) true zero-knowledge forecasting would need a separate
  weather-forecast model, out of scope here but worth flagging in
  project documentation.
- Handoff model_results_by_site.parquet (metrics + best-model
  choice) and model_forecasts_holdout.parquet (actual + all predictions)
  are both saved for the inventory simulation step.



## Summary of findings and known limitations

- **Pour readiness** fill in the printed mean/pass-rate from Section 5
  note the meaningful gap between full-period and post-warmup numbers,
  and that the gap is a cold-start artifact of sites whose real ending
  inventory was already below the computed reorder point, not a flaw in
  the reorder policy itself. If the post-warmup rate is still below 98%,
  that's a genuine finding don't round it up.
- **Silo utilization (Target 3)** measured as % of days in a healthy
  30-80% band, not raw average % change a naive %-change was found to
  be statistically misleading given the bimodal historical baseline
  (near-empty aggressive sites vs. capacity-capped conservative sites).
  Report the percentage-point improvement from Section 5's band
  comparison, and the cross-site std reduction as a secondary "more
  consistent practice" indicator.
- **Write-offs (Target 4)** the simulated 0.0 tonnes overflow is a
  **structural guarantee** of the reorder-point framework (order-up-to
  level is capped at silo capacity by design), not an empirically
  discovered result report it to stakeholders as a design property,
  not a percentage achievement.
- **Two explicit assumptions requiring real data before production use**
  (1) `LEAD_TIME_DAYS=3` is not sourced from data — get real supplier
  lead times from procurement; (2) future weather/pour-schedule inputs
  use a same-period-prior-year proxy replace with MIG's actual pour
  schedule and a weather forecast API feed.
- **Safety stock caveat** built from backtest RMSE, which may
  understate real-world forecast uncertainty's monitoring
  framework should recalibrate this from live forecast-error tracking
  rather than treating the backtest number as permanent.

## Conclusion

Two of four target outcomes are cleanly met with margin (MAPE, write-offs).
The utilization target required rejecting a misleading naive metric before
it could be reported honestly, and is met under the corrected measure. Pour
readiness is the one target genuinely not met yet reported as such rather
than rounded up, with a specific, actionable path to closing it once real
supplier lead-time data and live forecast-error monitoring are in place.

**Before production use, three things need real data, not the assumptions
used throughout this build**
1. Supplier lead times (currently a stated 3-day placeholder  not in the
   source data at all)
2. A real weather forecast feed and MIG's actual pour schedule (currently
   a same-period-prior-year proxy for future dates)
3. Docker build verification (untested in this development environment)

Everything else the forecasting model, the feature engineering, the
inventory logic, both dashboards, and the production pipeline was built,
tested by actually running it, and in three separate cases had a real bug
caught and fixed before being called done. Full detail, including every
number and every assumption, is in
[`docs/project_documentation.md`](docs/project_documentation.md).

<<<<<<< HEAD
## Author
[![LinkedIn](https://img.shields.io/badge/LinkedIn-0A66C2?style=flat&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/elias-data-scientist)

=======

