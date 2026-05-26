# STA 9890 Course Prediction Competition

This repository contains the final modeling pipeline for the STA 9890 Course Prediction Competition. The goal is to predict `PERCENT_PROFICIENT` for New York State assessment records at the school, subgroup, and assessment level.

The final selected model is implemented in:

```text
sta9890_competition_model_v20_structural_repaired.py
```

The selected final submission from my run was:

```text
outputs/submissions/submission_final_20260518_001121.csv
```

Final public leaderboard MSE:

```text
31.873
```

Final out-of-fold validation MSE:

```text
53.69
```

## Project summary

The final model is a leakage-safe LightGBM ensemble with structure-aware post-processing. The core idea is:

```text
LightGBM ensemble
-> OOF-optimized blend
-> count-space subgroup constraints
-> residual meta-calibration
-> final PERCENT_PROFICIENT prediction
```

The strongest improvement did not come from simply adding more models. It came from combining a strong tabular model family with the structure of the prediction task. The target is a percentage derived from student counts, and subgroup rows inside the same school-assessment block are related. The final model uses that structure after the machine learning blend.

## Repository structure

Expected structure:

```text
project_folder/
│
├── sta9890_competition_model_v20_structural_repaired.py
├── README.md
│
├── data/
│   └── raw/
│       ├── scores_training.csv
│       ├── scores_test.csv
│       ├── school_covariates.csv
│       └── district_covariates.csv
│
└── outputs/
    ├── submissions/
    └── diagnostics/
```

Important note on folder case:

The script looks for the data folder at `data/raw/`. On Windows, `data/Raw/` will usually work because file paths are case-insensitive. For portability on GitHub or case-sensitive systems, use lowercase `raw`.

## Required data files

Place the four course-provided CSV files in `data/raw/`:

```text
scores_training.csv
scores_test.csv
school_covariates.csv
district_covariates.csv
```

No external data are used. The model is designed to use only the instructor-provided competition files.

## Python environment

The code was developed in Python 3.12. The main packages used are:

```text
numpy
pandas
scikit-learn
lightgbm
```

A simple setup is:

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install numpy pandas scikit-learn lightgbm
```

On macOS or Linux:

```bash
python -m venv .venv
source .venv/bin/activate
pip install numpy pandas scikit-learn lightgbm
```

## How to run the final model

From the main project directory, run:

```powershell
python sta9890_competition_model_v20_structural_repaired.py
```

The full run can take a while because it trains multiple LightGBM models across cross-validation folds and saves diagnostic files.

## Smoke test

To confirm the code runs before launching the full model, use a smaller sample:

```powershell
python sta9890_competition_model_v20_structural_repaired.py --models lgbm_countaware --sample-train-rows 25000 --sample-test-rows 5000 --n-splits 2 --inner-splits 2
```

This is not meant to reproduce the final score. It is only a quick check that the environment and file paths are working.

## Optional run controls

Run only selected models:

```powershell
python sta9890_competition_model_v20_structural_repaired.py --models lgbm_countaware,lgbm_countaware_seed2,lgbm_deep_bagged_xlong
```

Disable residual meta-calibration:

```powershell
python sta9890_competition_model_v20_structural_repaired.py --no-residual-stacking
```

Disable the global count solver:

```powershell
python sta9890_competition_model_v20_structural_repaired.py --no-global-count-solver
```

Disable optional sibling-anchor calibration:

```powershell
python sta9890_competition_model_v20_structural_repaired.py --no-sibling-anchor-calibration
```

Skip diagnostics:

```powershell
python sta9890_competition_model_v20_structural_repaired.py --skip-diagnostics
```

## Output files

The script creates two main output folders:

```text
outputs/submissions/
outputs/diagnostics/
```

### Submission outputs

Important files in `outputs/submissions/` include:

```text
submission_final_<timestamp>.csv
submission_final.csv
submission_raw_blend_<timestamp>.csv
submission_pairwise_constraint_blend_<timestamp>.csv
submission_chosen_constraint_base_<timestamp>.csv
submission_pre_sibling_rescue_<timestamp>.csv
```

The Kaggle-ready file has two columns:

```text
ASSESSMENT_ID
PERCENT_PROFICIENT
```

The final selected submission from my completed run was:

```text
submission_final_20260518_001121.csv
```

### Diagnostic outputs

Important files in `outputs/diagnostics/` include:

```text
cv_model_summary.csv
cv_fold_scores.csv
oof_predictions.csv
overfitting_summary.csv
constraint_weights.json
global_count_solver_diagnostics.json
residual_stacking_diagnostics.json
bias_calibration_diagnostics.json
sibling_anchor_diagnostics.json
```

These diagnostics were used to compare model versions, check overfitting, confirm blend weights, inspect constraint performance, and verify whether calibration layers should be kept.

## Final modeling approach

### 1. Feature engineering

The final feature set included assessment, count, school, district, unsupervised structure, target encoding, peer, and algebra features.

Key feature groups:

```text
Assessment parsing:
- subject
- extracted grade
- grade band
- Regents versus grade 3-8 indicator

School and district context:
- demographics
- funding
- staffing ratios
- class size
- attendance and enrollment
- district outcome features

Count and reliability features:
- N_STUDENTS
- log N_STUDENTS
- sqrt N_STUDENTS
- small group indicators

Unsupervised structure:
- school-level PCA features
- district-level PCA features
- KMeans cluster labels

Leakage-safe target encodings:
- school-assessment history
- district-subgroup history
- county-assessment history
- assessment-subgroup history

Peer and algebra features:
- same school-assessment sibling subgroup information
- inferred proficient counts
- subgroup consistency indicators
```

### 2. LightGBM ensemble

The final raw prediction used six LightGBM models. The blend weights are prediction weights, not feature importances.

Final raw blend weights:

| Model | Weight | Role |
|---|---:|---|
| `lgbm_countaware` | 28.21% | Main count-aware LightGBM model |
| `lgbm_deep_bagged_xlong` | 21.08% | Deep model with bagging and sampling diversity |
| `lgbm_countaware_seed2` | 20.83% | Second count-aware model with different seed |
| `lgbm_deep_xlong_seed2` | 18.33% | Second deep long-run LightGBM with different seed |
| `lgbm_deep_xlong` | 6.06% | Long deep baseline model |
| `lgbm_logit_countaware` | 5.49% | Logit-scale count-aware diversity model |

The count-aware models made up about 54.5% of the raw blend, which shows that `N_STUDENTS` and count reliability were central to the final predictions.

### 3. OOF-optimized blending

The ensemble uses out-of-fold predictions to optimize a nonnegative weighted average. This means each model's contribution is based on validation performance, not on test labels.

For one row, the raw blend is approximately:

```text
raw blend =
0.2821 * lgbm_countaware
+ 0.2108 * lgbm_deep_bagged_xlong
+ 0.2083 * lgbm_countaware_seed2
+ 0.1833 * lgbm_deep_xlong_seed2
+ 0.0606 * lgbm_deep_xlong
+ 0.0549 * lgbm_logit_countaware
```

A weaker standalone model can still receive a small positive blend weight if its errors are different from the stronger models.

### 4. Count-space subgroup constraints

The most important extension was the count-space subgroup constraint layer.

A pure LightGBM model predicts each row independently. However, subgroup rows within the same school-assessment block are related. For example:

```text
All Students ~= Male + Female
All Students ~= Economically Disadvantaged + Not Economically Disadvantaged
```

Instead of correcting percentages directly, the script converts predictions into inferred proficient counts:

```text
predicted proficient count = predicted percent * N_STUDENTS
```

The solver then projects predictions back toward subgroup-consistent percentages.

This layer produced the largest structural gain:

| Stage | OOF MSE |
|---|---:|
| Raw LightGBM blend | 57.65 |
| Pairwise constraint blend | 55.93 |
| Anchored count solver | 53.86 |
| Residual-meta final | 53.69 |

### 5. Residual meta-calibration

After the count-space constraint layer, the model applies a guarded residual meta-calibration layer.

The residual is:

```text
residual = actual PERCENT_PROFICIENT - constrained prediction
```

The residual model learns whether the constrained prediction still has systematic leftover error. The final prediction is:

```text
final prediction = constrained prediction + predicted residual correction
```

This layer was kept because it improved OOF validation performance. More aggressive calibration layers were rejected when they did not generalize well enough.

## Validation and leakage controls

The script was built around out-of-fold validation and fold-pure feature creation.

Important controls:

```text
- Target encodings are fit only on training folds.
- Validation rows do not create their own target-encoded features.
- Blend weights are optimized using OOF predictions.
- Constraint and residual layers are selected using OOF performance.
- Test rows are used only as feature rows.
- No external data are used.
- Predictions are clipped to the valid 0 to 100 range by default.
```

## Final result

Selected final submission:

```text
submission_final_20260518_001121.csv
```

Final public leaderboard MSE:

```text
31.873
```

Final OOF MSE:

```text
53.69
```

The main takeaway is that the final model worked because it combined flexible machine learning with problem-specific structure. LightGBM captured nonlinear tabular patterns, while the count-space subgroup constraint layer corrected relationships that a row-by-row model would not learn perfectly on its own.

## Troubleshooting

### Missing required files

If you see an error about missing files, check that the data files are in:

```text
data/raw/
```

and that the filenames exactly match:

```text
scores_training.csv
scores_test.csv
school_covariates.csv
district_covariates.csv
```

### Folder named Raw instead of raw

On Windows, `data/Raw/` may work. On GitHub, Linux, or other case-sensitive systems, use:

```text
data/raw/
```

### LightGBM is not installed

Install it with:

```powershell
pip install lightgbm
```

### The run is taking a long time

The full run trains multiple LightGBM models across cross-validation folds. For a quick test, use the smoke-test command first.

### Windows joblib physical-core warning

The script sets `LOKY_MAX_CPU_COUNT` to reduce harmless joblib warnings during KMeans. If a warning still appears, it should not affect the model output.

## Academic integrity and data use

This project uses only the course-provided data files. No external data are joined into the modeling pipeline.

The final report includes the required AI usage disclosure and describes how AI was used for coding, debugging, organization, and drafting support, while final modeling decisions and submitted work were reviewed and selected by the student.
