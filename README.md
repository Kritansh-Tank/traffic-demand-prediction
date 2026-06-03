# Traffic Demand Prediction

**Final Online Score: 90.21 (R²)**

## Problem Statement

Predict passenger travel demand across urban geohash zones using historical data from **Day 48** to forecast demand on **Day 49** (2:15 AM – 1:00 PM).

## Dataset

| File | Description |
|---|---|
| `dataset/train.csv` | Days 48 & 49 (0:00–2:00) demand data |
| `dataset/test.csv` | Day 49 (2:15–13:00) rows to predict |
| `dataset/sample_submission.csv` | Submission format |

**Key columns:** `geohash`, `timestamp`, `day`, `demand`, `RoadType`, `NumberofLanes`, `LargeVehicles`, `Landmarks`, `Weather`, `Temperature`

## Approach

### Core Insight
- **Day 48 → Day 49 Pearson correlation ≈ 0.79** — yesterday's demand is the strongest predictor
- Training includes all Day 48 rows (with `demand_d48 = demand`, intentional leakage that teaches the model to trust historical values)
- Day 49 early data (0:00–2:00) provides geohash-level scaling factors

### Feature Engineering

| Category | Features |
|---|---|
| **Historical demand** | `demand_d48`, hierarchical fill via `gh_slot_mean → p4_slot_mean → slot_mean` |
| **Lag/Lead curve** | `demand_d48_lag1–3`, `demand_d48_lead1–3`, trend, acceleration, smooth |
| **3-way interactions** | `gh_slot_wth` (geohash×slot×weather), `gh_slot_rt` (×roadtype), `gh_slot_lv` (×LargeVehicles), `gh_slot_lm` (×Landmarks) |
| **Hierarchical fills** | NaN → hour-level → geohash-level → slot-level fallbacks |
| **Day49 signals** | `d49e_mean`, `d49_scale` (geohash scaling), `p4_d49_scale` (prefix fallback) |
| **Spatial** | lat/lon from geohash, prefix4/prefix3 neighbor stats |
| **Ratio features** | `d48_vs_wth_fill`, `d48_vs_rt_fill` (demand vs contextual predictions) |

### Model

3-model ensemble with scipy-optimised blend weights:

| Model | Config | OOF R² | Blend Weight |
|---|---|---|---|
| LightGBM | `lr=0.008, depth=7, leaves=63, reg_α=0.2, reg_λ=1.0` | 99.31 | ~5% |
| LightGBM (diverse) | `lr=0.008, depth=7, leaves=63, reg_α=0.2, reg_λ=1.0, seed=123` | 99.31 | ~50% |
| CatBoost | `lr=0.02, depth=7, l2_leaf_reg=5` | 99.30 | ~45% |

**5-Fold KFold CV | Blend OOF R²: 99.33**

### Why OOF ≈ 99 but Online ≈ 90?

The OOF is inflated because training rows where `demand_d48 == demand` are evaluated in-fold. The model essentially memorises Day 48. The online score (90.21) reflects genuine Day 49 generalisation.

The remaining ~10 point gap is **irreducible** — caused by:
- Day-to-day random variation (events, weather changes)
- Training Day 49 data is only 0:00–2:00 (night); test is 2:15–13:00 (daytime)

## Results

| Version | Online R² | Key Change |
|---|---|---|
| v1 | 87.11 | Baseline (LGB + XGB) |
| v5 | 89.81 | 3-way interactions |
| v7 | 89.86 | Hierarchical imputation |
| v9 | 90.06 | Demand lag/lead features |
| **v10 ✅** | **90.21** | Landmarks interaction + p4_d49_scale + diverse ensemble |

## Files

```
├── dataset/
│   ├── train.csv
│   ├── test.csv
│   └── sample_submission.csv
├── traffic_demand_prediction.ipynb  # Full notebook (run this)
├── submission.csv               # Best submission (90.21)
└── README.md
```

## How to Run

1. Install dependencies:
```bash
pip install lightgbm catboost pygeohash scikit-learn scipy pandas numpy
```

2. Open and run the notebook:
```
traffic_demand_prediction.ipynb
```
Run all cells top to bottom. The notebook will output `submission_v10.csv`.

## Dependencies

```
pandas, numpy, lightgbm, catboost, pygeohash, scikit-learn, scipy
```
