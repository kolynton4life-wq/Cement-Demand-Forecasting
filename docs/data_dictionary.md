# MIG Cement Demand Forecasting Data Dictionary & Phase 1 Audit

## Source
`MIG_Cement_Records.db` (SQLite), copied to `data/raw/`

## Schema (as-built differs from original spec doc)

The original project spec described a single flat `Cement_Demand` table.
The actual database is normalized into three tables. This is a **better**
design (no repeated site attributes on every row) extraction code joins
them back into an analysis-ready dataframe in `src/data/extract.py`.

### `Sites` (30 rows)
| Column | Type | Notes |
|---|---|---|
| site_id | TEXT (PK) | SITE_001 – SITE_030 |
| region | TEXT | North (3), South (11), East (12), West (4) unbalanced, note for segmentation |
| silo_capacity | INTEGER | Tonnes, matches Operations.silo_capacity (verified 0 mismatches) |
| behavior | TEXT | aggressive (14), conservative (9), chaotic (7) see Finding 1 below |

### `CementTypes` (3 rows)
| Column | Type | Notes |
|---|---|---|
| cement_type | TEXT (PK) | CEM_I, CEM_II, CEM_III — lookup only, no other attributes |

### `Operations` (32,880 rows) — the fact table
| Column | Type | Notes |
|---|---|---|
| date | TEXT | 2022-01-01 to 2024-12-31, daily, no gaps |
| site_id | TEXT (FK → Sites) | |
| cement_type | TEXT (FK → CementTypes) | |
| planned_pour_tonnes | FLOAT | |
| consumed_tonnes | FLOAT | |
| opening_inventory_tonnes | FLOAT | |
| deliveries_tonnes | FLOAT | |
| closing_inventory_tonnes | FLOAT | See Finding 2 |
| rain_mm | FLOAT | |
| avg_temp_c | FLOAT | |
| silo_capacity | INTEGER | Redundant copy of Sites.silo_capacity — consistent |

**Grain confirmed**: one row per (date, site_id, cement_type). Zero duplicates.

## Phase 1 Audit Findings

### Finding 1 — `behavior` encodes the three business problems from the brief
| Behavior | Sites | Stockout-risk rate¹ | Avg inv. vs. capacity | Interpretation |
|---|---|---|---|---|
| aggressive | 14 | 65.0% | ~5% | JIT ordering → chronic stockout risk |
| chaotic | 7 | 21.8% | ~63% | Erratic/reactive ordering |
| conservative | 9 | 14.3% | ~3,856% (!) | Structural overstocking |

¹ stockout-risk = share of rows where `consumed_tonnes < planned_pour_tonnes`.

This directly maps to the brief's stated pain points (stockouts/idle
resources; overstocking/capital tie-up; reactive ordering). Use this
segmentation in EDA (Phase 2) and consider it as a candidate model
feature or a basis for per-segment modeling strategy (Phase 3).

### Finding 2 Conservative-site inventory exceeds physical silo capacity
34.8% of all rows (11,439 / 32,880) show `closing_inventory_tonnes` above
`silo_capacity`; almost entirely concentrated in `conservative` sites
(98.7% of their rows), averaging ~38x capacity. This is not physically
possible for a real silo and is a **synthetic-data generation artifact**
the simulator did not cap deliveries at capacity for that behavior
class.

**Decision taken**: `src/data/validate.py` clips `closing_inventory_tonnes`
at `silo_capacity` and records the excess as `overflow_tonnes` (proxy for
material write-off/waste). Raw uncapped value preserved as
`closing_inventory_raw_tonnes`. Toggle via `CAP_INVENTORY_AT_CAPACITY` in
`src/config.py`. **This must be stated explicitly in final project
documentation** it's a modeling assumption, not ground truth.

### Finding 3 Balance equation holds
`closing = opening + deliveries - consumed` holds to within floating-point
rounding (max deviation £0.01) across all 32,880 rows. No correction
needed.

### Finding 4 No nulls, no negatives, no missing dates
All 30 sites have exactly 1,096 rows spanning the full date range with no
gaps. Full completeness unusual for real operational data, consistent
with this being a synthetic/teaching dataset rather than a raw ERP export.
Treat forecast performance on this data as an upper bound relative to what
a live MIG rollout on real (messier) data would achieve.

### Finding 5 Region imbalance
Sites are unevenly split across regions (North=3, West=4, South=11,
East=12). Small regional groups (North, West) will have limited
statistical power for any region-level modeling or weather-interaction
analysis flag this for Phase 3 model design, don't build a
region-specific model expecting reliable results from 3-4 sites.

## Open questions for stakeholders (carry into Phase 0 sign-off)
- Confirm `CAP_INVENTORY_AT_CAPACITY` decision is acceptable, or whether
  MIG wants raw values with overflow reported separately in the dashboard.
- Confirm whether `region` should be a modeling feature or just a
  dashboard filter, given the sample-size imbalance.
- Confirm supplier lead times and minimum order quantities are NOT in
  this dataset needed separately for Phase 4 (inventory optimization /
  reorder point calculation) and must be sourced from procurement.
