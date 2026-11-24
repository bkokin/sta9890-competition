# STA 9890 Data Mining Competition | First Place

![Competition result: First Place](https://img.shields.io/badge/Competition%20Result-1st%20Place-gold)
![Public leaderboard MSE: 31.873](https://img.shields.io/badge/Public%20MSE-31.873-blue)
![Model: LightGBM Ensemble](https://img.shields.io/badge/Model-LightGBM%20Ensemble-green)

**First-place winning solution for the STA 9890 graduate data mining competition at Baruch College, Spring 2026.**

This repository documents my winning approach to predicting New York State assessment proficiency at the school, subgroup, and assessment level. The final pipeline combines six LightGBM models with different hyperparameters and random seeds, out-of-fold (OOF) optimized blending, count-space subgroup constraints, and residual meta-calibration.

## Competition results

| Metric | Result |
|---|---:|
| Final competition placement | **1st place** |
| Selected submission public leaderboard MSE | **31.873** |
| Final OOF validation MSE | 53.6933 |
| Final OOF validation RMSE | 7.3276 |
| Training observations | 144,921 |
| Test observations | 48,307 |
| Schools | 4,754 |
| Models in final ensemble | 6 |

The public leaderboard score is the score of my selected submission, not a claim about the final private leaderboard MSE. The final competition placement was first overall.

**Final model script:** `sta9890_competition_model_v20_structural_repaired.py`  
**Selected submission:** `submission_final_20260518_001121.csv`

The key lesson from the class presentations was that LightGBM alone was not the differentiator. Several classmates also used LightGBM. My approach combined multiple configurations of the same algorithm to diversify prediction errors, then used the mathematical relationships between subgroup results to improve the blended predictions.

## Project summary

The final model is a LightGBM ensemble with fold-aware feature engineering and structure-aware post-processing. The core idea is:

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

The script looks for the data folder at `data/raw/`. On a typical Windows filesystem, `data/Raw/` also works. Use lowercase `raw` for portability to case-sensitive systems.

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

The original development environment used Python 3.12. The main packages required for the default LightGBM pipeline are:

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
python -m pip install numpy pandas scikit-learn lightgbm
```

On macOS or Linux:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install numpy pandas scikit-learn lightgbm
```

## How to run the final model

From the main project directory, run:

```powershell
python sta9890_competition_model_v20_structural_repaired.py
```

The default run trains all six LightGBM variants using five outer and five inner folds. It is computationally intensive. For a reproduction run, use the full data and default settings. A new run creates a new timestamped submission rather than reusing the original filename. Record package versions using `python -m pip freeze` and consult the original run metadata if available; exact numerical reproducibility can depend on package versions, hardware, and thread settings.

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
blend_weights.json
run_metadata.json
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

The count-aware models made up about 54.5% of the raw blend. This reflects their contribution to the optimized ensemble, although blend weights alone do not measure the importance of individual features.

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
All Students proficient count ~= Male proficient count + Female proficient count
All Students proficient count ~= Economically Disadvantaged proficient count
                                 + Not Economically Disadvantaged proficient count
```

Instead of correcting percentages directly, the script converts predictions into inferred proficient counts:

```text
predicted proficient count = (predicted percent / 100) * N_STUDENTS
```

The solver projects predictions toward subgroup-consistent counts, using observed training subgroup outcomes as anchors where available and model estimates for unknown outcomes. Rounding, missing subgroup records, and differences in reported counts mean that not every relationship can be enforced exactly. The corrected counts are converted back to percentages. In the selected run, direct algebra was available for approximately 56.5% of test rows, while anchored projection covered approximately 92.4%.

This layer produced the largest structural gain:

| Stage | OOF MSE | OOF RMSE |
|---|---:|---:|
| Raw LightGBM blend | 57.6483 | 7.5926 |
| Pairwise constraint blend | 55.9267 | 7.4784 |
| Anchored count solver | 53.8590 | 7.3389 |
| Residual-meta final | **53.6933** | **7.3276** |

The full pipeline reduced OOF MSE by approximately 6.86% relative to the raw ensemble. Most of that reduction came from the subgroup constraints.

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

This layer was retained after it improved OOF MSE. An optional sibling-anchor calibration did not pass its acceptance threshold, and an additional bias correction was not retained. The selected final prediction label was `residual_meta_constraint_blend`.

## Validation and leakage controls

The script uses out-of-fold validation and fold-specific target feature construction.

Important controls:

```text
- Target encodings are fit only on training folds.
- Validation rows do not create their own target-encoded features.
- Blend weights are optimized using OOF predictions.
- Constraint and residual layers are selected using OOF performance.
- No hidden test targets are used; known training outcomes may inform related test rows.
- No external data are used.
- Predictions are clipped to the valid 0 to 100 range by default.
```

The OOF predictions were also used to select blend weights and post-processing settings, so the reported final OOF error is a development estimate rather than a completely independent evaluation of the selected pipeline. Public leaderboard feedback was monitored during iteration. The public MSE and final competition placement should be interpreted separately from local validation results.

## Development workflow and lessons learned

More than 20 model iterations were evaluated throughout the competition. I saved and reviewed diagnostic files after major runs, including individual model scores, fold-level results, blend weights, and constraint and residual diagnostics. This helped identify which changes contributed to performance and which added unnecessary runtime or complexity.

The most valuable improvements came from diversifying the LightGBM ensemble, accounting for the student-count denominator, and exploiting subgroup relationships. The final first-place finish illustrates the value of pairing flexible machine learning with statistical knowledge of the prediction target.

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

On Windows, `data/Raw/` may work. On Linux or other case-sensitive systems, the folder name must match the path in the script. Use:

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

This project uses only the course-provided data files. No external data are joined into the modeling pipeline. Obtain the original files through authorized course access and do not redistribute them without permission. 

The final report includes the required AI usage disclosure and describes how AI was used for coding, debugging, organization, and drafting support, while final modeling decisions and submitted work were reviewed and selected by the student.
