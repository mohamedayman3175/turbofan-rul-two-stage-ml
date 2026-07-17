# turbofan-rul-two-stage-ml
Two-stage ML pipeline (PCA + classifier + regressor) for predicting turbofan engine RUL on NASA's C-MAPSS dataset

## Project Overview

Most public solutions to the NASA C-MAPSS dataset train on a single sub-dataset
(commonly FD001) for simplicity. This project deliberately combines all four
sub-datasets (FD001–FD004) — spanning 1 to 6 operating conditions and 1 to 2
fault modes — into a single ~160K-row dataset, to force the model to learn a
degradation pattern that generalizes across operating regimes rather than
overfitting to one clean scenario. This mirrors a real factory setting, where
a monitoring system needs to work across a fleet of machines running under
different conditions, not just one.

### Key design decisions

- **Per-condition scaling, not global scaling.** The same sensor reads
  differently depending on operating condition (e.g. cruise vs. takeoff), so
  all sensors are standard-scaled separately for every `(dataset, operating
  condition)` combination. Scaling globally would blur that condition-driven
  shift into the degradation signal and hurt the model significantly.

- **Outlier removal, not degradation removal.** Rows flagged as statistical
  outliers by more than 12 sensors simultaneously were removed — these are
  data-quality artifacts (sensor glitches), not genuine failure signatures.
  Real degradation trends were left fully intact, since that's the actual
  signal the model needs to learn.

- **Two-stage pipeline instead of one regressor.** Rather than training a
  single model to predict RUL for every engine (including perfectly healthy
  ones, where RUL is essentially unknowable from the sensors alone), the
  problem is split in two:
  1. **Stage 1 — Classifier:** flags each reading as `healthy` or
     `degrading`, using 11 PCA components (94.8% variance retained from 17
     active sensors). Several classifiers were benchmarked, and Random
     Forest was selected and tuned via a systematic hyperparameter search.
  2. **Stage 2 — Regressor:** only for rows flagged `degrading`, predicts the
     actual remaining cycles using the raw scaled sensors. This stage trains
     on a much smaller, focused set (~16K degrading rows out of 160K+), which
     matters because RUL for a healthy engine is far less meaningful than RUL
     for one that's actually approaching failure.

This kind of two-stage early-warning + remaining-life estimate is directly
applicable to predictive maintenance in manufacturing — flagging equipment
that's starting to degrade before it fails, and estimating how much runway
is left to schedule maintenance.

## Results

**Operating condition clustering (KMeans, FD002/FD004):** 6 clusters,
inertia 0.88 — reasonably tight groupings across ~17K rows each.

**PCA compression:** 17 active sensors → 11 components, 94.8% variance retained.

**Stage 1 — Random Forest Classifier**

| Set | Class | Precision | Recall | F1 |
|---|---|---|---|---|
| Validation | healthy | 0.98 | 0.97 | 0.98 |
| Validation | degrading | 0.82 | 0.87 | 0.85 |
| Test | healthy | 0.99 | 0.99 | 0.99 |
| Test | degrading | 0.67 | 0.69 | 0.68 |

**Stage 2 — Random Forest Regressor (degrading rows only)**

| Set | RMSE (cycles) |
|---|---|
| Validation | 5.02 |
| Test | 20.03 |

**Full pipeline (Stage 1 → Stage 2, all 104,897 test rows):** RMSE = 26.58 cycles
