# MIG Cement Demand Forecasting Project Documentation

**Deliverable 4 of 4** (per project brief): methodology, model performance, and key insights.

---

## 1. Executive Summary

A per-site forecasting and inventory optimization system was built for MIG's 30 active
sites, covering all 7 project steps data ingestion, EDA, feature engineering, model
development, inventory simulation, a dashboard, and a production deployment pipeline.

**Bottom line against the four target outcomes:**

| Target | Result | Status |
|---|---|---|
| MAPE ≤ 15% | 3.3% (holdout), 4.6% (rolling 3-window validation) | ✅ Met, with margin |
| ≥98% pour readiness | 96.5% average; 10/30 sites individually ≥98% | ⚠️ Not fully met see §5 |
| 20% silo utilization improvement | +39.6 percentage points in healthy-band time (see §4 for why raw % change was rejected as a metric) | ✅ Met |
| 30% write-off reduction | 0 overflow tonnes a structural design property, not a statistical finding | ✅ Met by construction |

Two of four targets are cleanly met. The utilization target required correcting a
statistically misleading calculation before it could be reported honestly. The pour
readiness target is **not** fully met and should not be presented as met the gap and
its cause are explained in §5, along with what real data would fix it.

---

## 2. Methodology

### 2.1 Data
Source: SQLite database (`MIG_Cement_Records.db`), normalized into `Sites` (30 rows),
`CementTypes` (3 rows), `Operations` (32,880 rows one row per site per calendar day,
2022-01-01 to 2024-12-31). `cement_type` rotates as a same-day attribute per site, not
a parallel series — this distinction mattered directly for feature engineering (§2.3).

### 2.2 Data Cleaning (Step 1)
- Grain verified: no duplicate (date, site_id, cement_type) rows
- Balance equation (`closing = opening + deliveries − consumed`) verified/corrected;
  a physical constraint (`consumed` cannot exceed `opening + deliveries`) was added
  during Step 7 hardening after a pytest regression test caught it on adversarial
  synthetic data (never triggers on the real dataset, but the logic is now correct
  regardless of future data quality)
- **Capacity-capping decision**: 34.8% of rows (concentrated in `conservative` behavior
  sites) showed closing inventory up to ~38x silo capacity a synthetic-data
  generation artifact, not physically real. Capped at capacity; excess tracked as
  `overflow_tonnes`. This is a stated modeling assumption, not raw ground truth.

### 2.3 EDA Findings (Step 2) That Shaped Everything Downstream
- **`behavior` (aggressive/chaotic/conservative) is the dominant signal** in the
  dataset aggressive sites show 65% stockout-risk rate vs. 14% for conservative,
  and this pattern held all the way through to final model selection.
- **No day-of-week effect** consumption is flat across all 7 days (~23-24 t/day).
- **Series is stationary** (ADF p≈0.000) informed SARIMAX's `d=0`.
- **Weather has a threshold effect, not a linear one** raw `rain_mm` correlation
  with consumption is only -0.18, which would wrongly suggest dropping it. Banding
  rain into none/light/moderate/heavy revealed heavy rain (≥15mm) crashes
  consumption to ~4 t/day (vs. ~24-25 normally) while planned pours stay unchanged,
  pushing stockout risk to 64%. This directly drove the `heavy_rain_flag` feature
  design in Step 3 rather than using `rain_mm` as a linear regressor.
- **Planned-pour reliability varies sharply by behavior**: correlation with actual
  consumption is 0.485 (aggressive) vs. 0.983 (conservative) meant
  `planned_pour_tonnes` could not be trusted uniformly as a model input.

### 2.4 Feature Engineering (Step 3)
Lag features (1/7/14/28-day), rolling mean/std (7/14/28-day), weather-threshold flags,
a trust-adjusted planned-pour feature (scaled by rolling adherence), inventory turnover
metrics, and behavior×weather/behavior×planned-pour interaction terms.

**Two real bugs were caught and fixed during development, not left in:**
1. A grain-fragmentation bug lag/rolling features were initially grouped by
   `(site_id, cement_type)`, which fragmented each site's continuous daily series
   into three irregularly-spaced sub-series (since cement_type rotates day to day).
   Fixed to group by `site_id` alone; verified `consumed_lag_1d` now matches literal
   yesterday's value for every row.
2. A non-informative interaction variable an early `behavior_x_planned_pour`
   feature multiplied every behavior by the same constant (1.0), making it
   mathematically identical to `planned_pour_tonnes` itself. Fixed to genuine
   `continuous_feature x dummy_column` interactions (zero outside the category,
   real value inside it).

A full leakage audit excludes any feature mathematically derived from the same-day
target (e.g. `stockout_risk`, `days_of_stock`, `closing_inventory_tonnes`) from the
model's input feature list  using them would let the model see the answer.

### 2.5 Model Development (Step 4)
Three approaches compared per site (30 site-level models, not 90  see §2.3 on grain)
naive baseline (trailing 7-day average), SARIMAX(1,0,1) with weather/pour exogenous
regressors, and Random Forest with the same leakage-safe feature set, using recursive
multi-step forecasting for the 8-week (56-day) horizon.

**Result: Random Forest won on all 30/30 sites.** SARIMAX specifically struggled on
`aggressive`behavior sites (40.5% mean MAPE  worse than the naive baseline there),
consistent with those sites' erratic, threshold-driven consumption that a linear
model can't represent. Verified both numerically (MAPE/RMSE table) and visually
(forecast overlay chart) before trusting the result.

### 2.6 Rolling-Window Validation (Step 7)
Step 4 used a single fixed 56-day holdout. To confirm that result wasn't specific to
one period, Step 7's production pipeline re-validates each site across 3 independent,
non-overlapping walk-forward windows. Result 4.6% mean MAPE (worst site 8.75%),
consistent with the single-window result  not a lucky-period artifact.

### 2.7 Inventory Simulation (Step 5)
Continuous-review (s, S) reorder-point policy per site: reorder point = forecasted
demand over lead time + safety stock (98%-service-level z-score × forecast RMSE);
order-up-to level capped at silo capacity. Two assumptions not present in the source
data are stated explicitly (§5).

### 2.8 Dashboard (Step 6)
Two implementations, both reading the same processed data
- **Plotly Dash** (`src/dashboard/app.py`)  the project brief's specified deliverable
- **Streamlit** (`src/dashboard/streamlit_app.py`)  added for this project's specific
  deployment target; kept in feature parity with the Dash app deliberately

Both provide an aggregate Overview (KPI cards, MAPE/readiness charts, alert table) and
a Site Detail drill-down (forecast chart, inventory projection with ROP/capacity
lines, per-site alerts). Both were tested live (server start + real HTTP requests +
interactive callback/selector test), not just written and assumed correct.

### 2.9 Production Pipeline (Step 7)
- `src/preprocessing.py`  verified byte-for-byte identical output to the exploratory
  notebooks on the real dataset before anything was built on top of it
- `src/pipeline.py` orchestrates ingest → preprocess → rolling-window validate →
  train final model on full history → persist to `models/`
- `src/inference.py`  loads a saved model and predicts without retraining (verified
  ~2s per forecast vs. ~5s/site to retrain)
- `src/monitoring.py`  live MAPE tracking against the 15% target; tested both the
  trigger and no-trigger branches explicitly, plus the exact-threshold boundary case
- `src/api.py` (FastAPI) all endpoints tested including error paths (404 unknown
  site, 400 invalid horizon)
- `test/`  25 pytest tests; the suite caught a real bug during development (see §5)
- Docker (`Dockerfile.model/api/dashboard/streamlit`, `docker-compose.yaml`)  written
  to standard conventions but **not build-verified**  no Docker daemon was available
  in the development environment

---

## 3. Model Performance Summary

| Model | Mean MAPE (Step 4 holdout) | Mean MAPE (Step 7 rolling validation) | Sites passing ≤15% |
|---|---|---|---|
| Naive baseline | 40.6% | — | 0/30 |
| SARIMAX + exogenous | 24.9% | — | 14/30 |
| **Random Forest (selected)** | **3.3%** | **4.6%** (worst site 8.75%) | **30/30** |

---

## 4. Key Insight: Two Metrics Were Corrected Before Reporting

Worth documenting explicitly, since the corrected numbers look very different from
the naive first-pass calculation:

- **Silo utilization**: a raw average % change was computed first and produced
  nonsensical results (up to +918% for individual sites) because the historical
  baseline is bimodal  near-empty `aggressive` sites vs. capacity-capped
  `conservative` sites near 100%. Replaced with "% of days in a healthy 30-80%
  utilization band," which moved from 8.1% historically to 47.7% under the new
  policy (+39.6 percentage points) — a real, defensible result.
- **Write-off reduction**: the simulated 0-tonnes overflow is **guaranteed by
  construction** (the order-up-to level is mathematically capped at silo capacity),
  not an empirically discovered 100% improvement. Reported to stakeholders as a
  design property of the reorder-point framework, not a statistic.

---

## 5. Known Limitations and Assumptions (Read Before Production Use)

1. **Lead time (3 days) is not sourced from data**  no supplier lead-time data
   exists anywhere in the source database or documentation. This is a placeholder
   assumption. **Get real figures from MIG procurement before using the reorder
   points for actual ordering decisions.**
2. **Future weather/pour-schedule inputs use a same-calendar-period-prior-year
   proxy** (Jan-Feb 2024 standing in for Jan-Feb 2025)  a simulation convenience,
   not a forecast of real future conditions. Replace with MIG's actual pour
   schedule (known in advance) and a weather forecast API before production use.
3. **Pour readiness (96.5%, not the ≥98% target) is a genuine finding, not
   rounded up.** Root cause: (a) a "cold-start" effect sites whose real ending
   inventory was already below the computed reorder point show stockouts in the
   first `lead_time` days before the first order arrives; (b) safety stock is
   built from backtest RMSE, which is likely optimistic once the weather/pour
   proxy assumption (limitation #2) is factored in for genuinely unknown future
   periods. **Recommended fix**: recalibrate safety stock from live forecast-error
   monitoring (the `monitoring.py` module already built for this purpose) once
   real production data starts flowing, rather than trusting the backtest number
   indefinitely.
4. **Docker deployment files are unverified**  written to standard Dockerfile/
   Compose conventions, but no Docker daemon was available during development to
   run `docker build`/`docker compose up`. Verify locally before relying on them.
5. **This is a synthetic/teaching dataset**  fully complete with no missing dates,
   which is unusual for real operational data. Treat all forecast accuracy numbers
   in this document as an upper bound relative to what a live MIG rollout on
   messier real data would likely achieve.

---

## 6. Recommendations for Production Rollout

1. Source real supplier lead times per site from procurement before trusting any
   reorder point calculated here.
2. Integrate a real weather forecast API and MIG's actual pour-schedule system,
   replacing the prior-year proxy used throughout Steps 5 and 7.
3. Pilot on a small subset of sites first (e.g. one from each `behavior` segment)
   in shadow mode before rolling out reorder alerts network-wide.
4. Schedule `src/pipeline.py` to re-run periodically (e.g. weekly) and wire
   `src/monitoring.py`'s retrain trigger into that schedule so degrading sites are
   caught automatically, not discovered manually.
5. Verify the Docker build locally / in CI before treating the deployment
   scaffolding as production-ready.
6. Revisit the safety-stock formula once ~2-3 months of live forecast-error data
   exists, to close the pour-readiness gap identified in §5.
