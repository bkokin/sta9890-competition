"""
STA 9890 Competition - single-file modeling pipeline

Purpose:
    Build a strong, reproducible model using only instructor-provided files.

What this script does:
    1. Loads the four course-provided CSV files from data/raw/.
    2. Joins school and district covariates to score records.
    3. Creates assessment, school-composition, staffing, funding, and count features.
    4. Builds leakage-safe target-encoding features inside cross-validation.
    5. Trains several model families when available:
          - Ridge regression with one-hot categoricals
          - HistGradientBoostingRegressor
          - LightGBM, if installed
          - CatBoost, if installed
    6. Blends models using out-of-fold predictions.
    7. Saves Kaggle-ready submissions and diagnostic CSVs.

Important:
    - This script uses no outside data.
    - Validation target encodings are fit only on the training portion of each fold.
    - Final submission file columns are ASSESSMENT_ID and PERCENT_PROFICIENT.

Recommended first full run:
    python sta9890_competition_model.py --models ridge,lgbm

Optional slower run:
    python sta9890_competition_model.py --models ridge,lgbm,cat

Smoke test:
    python sta9890_competition_model.py --models ridge --sample-train-rows 25000 --sample-test-rows 5000
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import re
import sys
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore", category=FutureWarning)
pd.options.mode.copy_on_write = True


TARGET = "PERCENT_PROFICIENT"
ID_COL = "ASSESSMENT_ID"
SEED = 20260511


# These are explicitly course-provided raw files. Do not add outside data.
REQUIRED_FILES = {
    "train": "scores_training.csv",
    "test": "scores_test.csv",
    "school": "school_covariates.csv",
    "district": "district_covariates.csv",
}


# Target-encoding groups. The exact SCHOOL + SUBGROUP + ASSESSMENT interaction is
# deliberately omitted because it usually does not overlap between train and test.
GROUP_SPECS: List[Tuple[str, List[str], float]] = [
    ("te_school_assessment", ["SCHOOL", "ASSESSMENT_NAME"], 20.0),
    ("te_school_subgroup", ["SCHOOL", "SUBGROUP_NAME"], 20.0),
    ("te_school_subject_subgroup", ["SCHOOL", "SUBJECT", "SUBGROUP_NAME"], 25.0),
    ("te_school_subject", ["SCHOOL", "SUBJECT"], 25.0),
    ("te_school", ["SCHOOL"], 50.0),

    ("te_district_assessment_subgroup", ["DISTRICT", "ASSESSMENT_NAME", "SUBGROUP_NAME"], 25.0),
    ("te_district_assessment", ["DISTRICT", "ASSESSMENT_NAME"], 30.0),
    ("te_district_subject_subgroup", ["DISTRICT", "SUBJECT", "SUBGROUP_NAME"], 30.0),
    ("te_district_subgroup", ["DISTRICT", "SUBGROUP_NAME"], 40.0),
    ("te_district", ["DISTRICT"], 80.0),

    ("te_county_assessment_subgroup", ["COUNTY", "ASSESSMENT_NAME", "SUBGROUP_NAME"], 35.0),
    ("te_county_assessment", ["COUNTY", "ASSESSMENT_NAME"], 50.0),
    ("te_county_subject_subgroup", ["COUNTY", "SUBJECT", "SUBGROUP_NAME"], 50.0),
    ("te_county_subgroup", ["COUNTY", "SUBGROUP_NAME"], 70.0),
    ("te_county", ["COUNTY"], 100.0),

    ("te_region_assessment_subgroup", ["REGION", "ASSESSMENT_NAME", "SUBGROUP_NAME"], 40.0),
    ("te_district_type_assessment_subgroup", ["DISTRICT_TYPE", "ASSESSMENT_NAME", "SUBGROUP_NAME"], 40.0),
    ("te_assessment_subgroup", ["ASSESSMENT_NAME", "SUBGROUP_NAME"], 60.0),
    ("te_assessment", ["ASSESSMENT_NAME"], 80.0),
    ("te_subject_subgroup", ["SUBJECT", "SUBGROUP_NAME"], 80.0),
    ("te_family_subgroup", ["ASSESSMENT_FAMILY", "SUBGROUP_NAME"], 80.0),
    ("te_subgroup", ["SUBGROUP_NAME"], 120.0),
]


BASE_CATEGORICAL_COLS = [
    "SCHOOL",
    "DISTRICT",
    "COUNTY",
    "DISTRICT_TYPE",
    "REGION",
    "SUBGROUP_NAME",
    "ASSESSMENT_NAME",
    "SUBJECT",
    "ASSESSMENT_FAMILY",
    "GRADE_BAND",
]


@dataclass
class FeatureSet:
    train_full: pd.DataFrame
    test_full: pd.DataFrame
    folds: List[Tuple[np.ndarray, np.ndarray, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]]
    cat_cols: List[str]
    num_cols: List[str]


def find_project_root() -> Path:
    """
    Find a project root containing data/raw.
    Supports placing this file either in the repo root or inside src/.
    """
    here = Path(__file__).resolve().parent
    candidates = [here, *here.parents]
    for c in candidates:
        if (c / "data" / "raw").exists():
            return c
    return here


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models",
        default="ridge,lgbm",
        help="Comma-separated list from: ridge,lgbm,cat,hgb. Missing optional packages are skipped.",
    )
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--inner-splits", type=int, default=5)
    parser.add_argument("--sample-train-rows", type=int, default=None)
    parser.add_argument("--sample-test-rows", type=int, default=None)
    parser.add_argument("--skip-diagnostics", action="store_true")
    parser.add_argument("--no-clip", action="store_true", help="Do not clip predictions to [0, 100].")
    return parser.parse_args()


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def mse(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    return float(mean_squared_error(y_true, y_pred))


def rmse(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    return math.sqrt(mse(y_true, y_pred))


def clip_pred(pred: np.ndarray, do_clip: bool = True) -> np.ndarray:
    pred = np.asarray(pred, dtype=float)
    if do_clip:
        return np.clip(pred, 0.0, 100.0)
    return pred


def one_hot_encoder() -> OneHotEncoder:
    """
    Compatible with newer and older scikit-learn versions.
    """
    try:
        return OneHotEncoder(handle_unknown="ignore", min_frequency=10, sparse_output=True)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=True)


def load_data(data_dir: Path, sample_train_rows: Optional[int], sample_test_rows: Optional[int]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    paths = {name: data_dir / fname for name, fname in REQUIRED_FILES.items()}
    missing = [str(p) for p in paths.values() if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing required file(s):\n" + "\n".join(missing))

    train = pd.read_csv(paths["train"])
    test = pd.read_csv(paths["test"])
    school = pd.read_csv(paths["school"])
    district = pd.read_csv(paths["district"])

    if sample_train_rows is not None and sample_train_rows < len(train):
        train = train.sample(sample_train_rows, random_state=SEED).reset_index(drop=True)
        log(f"Smoke-test mode: sampled train to {len(train):,} rows.")
    if sample_test_rows is not None and sample_test_rows < len(test):
        test = test.sample(sample_test_rows, random_state=SEED).reset_index(drop=True)
        log(f"Smoke-test mode: sampled test to {len(test):,} rows.")

    required_train = {ID_COL, "SCHOOL", "SUBGROUP_NAME", "ASSESSMENT_NAME", "N_STUDENTS", TARGET}
    required_test = {ID_COL, "SCHOOL", "SUBGROUP_NAME", "ASSESSMENT_NAME", "N_STUDENTS"}
    if not required_train.issubset(train.columns):
        raise ValueError(f"Training file missing columns: {sorted(required_train - set(train.columns))}")
    if not required_test.issubset(test.columns):
        raise ValueError(f"Test file missing columns: {sorted(required_test - set(test.columns))}")
    if "SCHOOL" not in school.columns or "DISTRICT" not in school.columns:
        raise ValueError("school_covariates.csv must contain SCHOOL and DISTRICT.")
    if "DISTRICT" not in district.columns:
        raise ValueError("district_covariates.csv must contain DISTRICT.")

    return train, test, school, district


def infer_subject(name: object) -> str:
    s = str(name).upper()
    if any(x in s for x in ["MATH", "ALGEBRA", "GEOMETRY"]):
        return "MATH"
    if any(x in s for x in ["ELA", "ENGLISH"]):
        return "ELA"
    if any(x in s for x in ["SCI", "BIO", "CHEM", "PHYS", "EARTH", "LIVING ENV"]):
        return "SCIENCE"
    if any(x in s for x in ["HIST", "GLOBAL", "GOV", "GEOGRAPHY"]):
        return "HISTORY"
    return "OTHER"


def extract_grade(name: object) -> float:
    s = str(name).upper()
    # Handles ELA4, MATH6, Grade 5, etc. Regents assessments will usually be NaN.
    match = re.search(r"(?:ELA|MATH|GRADE\s*)(\d{1,2})", s)
    if not match:
        match = re.search(r"\b(\d{1,2})\b", s)
    if match:
        val = int(match.group(1))
        if 1 <= val <= 12:
            return float(val)
    return np.nan


def make_grade_band(grade: object, assessment_name: object) -> str:
    s = str(assessment_name).upper()
    if "REGENTS" in s:
        return "REGENTS"
    try:
        g = float(grade)
    except Exception:
        return "UNKNOWN"
    if np.isnan(g):
        return "UNKNOWN"
    if g <= 5:
        return "ELEMENTARY"
    if g <= 8:
        return "MIDDLE"
    return "HIGH"


def safe_divide(num: pd.Series, den: pd.Series) -> pd.Series:
    den2 = den.replace({0: np.nan})
    return num / den2


def add_joined_covariates(df: pd.DataFrame, school: pd.DataFrame, district: pd.DataFrame) -> pd.DataFrame:
    out = df.merge(school, on="SCHOOL", how="left", validate="many_to_one")
    out = out.merge(district, on="DISTRICT", how="left", validate="many_to_one")
    return out


def add_base_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # Assessment parsing.
    out["SUBJECT"] = out["ASSESSMENT_NAME"].map(infer_subject)
    out["GRADE_EXTRACTED"] = out["ASSESSMENT_NAME"].map(extract_grade)
    out["ASSESSMENT_FAMILY"] = np.where(
        out["ASSESSMENT_NAME"].astype(str).str.upper().str.contains("REGENTS", na=False),
        "REGENTS",
        "GRADE_3_8",
    )
    out["GRADE_BAND"] = [
        make_grade_band(g, a) for g, a in zip(out["GRADE_EXTRACTED"], out["ASSESSMENT_NAME"])
    ]
    out["IS_REGENTS"] = (out["ASSESSMENT_FAMILY"] == "REGENTS").astype(int)
    out["IS_STEM"] = out["SUBJECT"].isin(["MATH", "SCIENCE"]).astype(int)

    # Test-row group size/noise features.
    out["LOG_N_STUDENTS"] = np.log1p(out["N_STUDENTS"])
    out["SQRT_N_STUDENTS"] = np.sqrt(out["N_STUDENTS"].clip(lower=0))
    out["INV_SQRT_N_STUDENTS"] = 1.0 / np.sqrt(out["N_STUDENTS"].clip(lower=1))

    # Funding and staff intensity.
    if {"FEDERAL_FUNDING_PER_PUPIL", "LOCAL_FUNDING_PER_PUPIL"}.issubset(out.columns):
        out["TOTAL_FUNDING_PER_PUPIL"] = out["FEDERAL_FUNDING_PER_PUPIL"] + out["LOCAL_FUNDING_PER_PUPIL"]
        out["LOG_TOTAL_FUNDING_PER_PUPIL"] = np.log1p(out["TOTAL_FUNDING_PER_PUPIL"].clip(lower=0))
        out["FEDERAL_FUNDING_SHARE"] = safe_divide(
            out["FEDERAL_FUNDING_PER_PUPIL"], out["TOTAL_FUNDING_PER_PUPIL"]
        )

    if {"NUMBER_OF_TEACHERS", "N_PUPILS"}.issubset(out.columns):
        out["TEACHERS_PER_100_PUPILS"] = 100.0 * safe_divide(out["NUMBER_OF_TEACHERS"], out["N_PUPILS"])
    if {"NUMBER_OF_COUNSELORS", "N_PUPILS"}.issubset(out.columns):
        out["COUNSELORS_PER_1000_PUPILS"] = 1000.0 * safe_divide(out["NUMBER_OF_COUNSELORS"], out["N_PUPILS"])
    if {"NUMBER_OF_SOCIAL_WORKERS", "N_PUPILS"}.issubset(out.columns):
        out["SOCIAL_WORKERS_PER_1000_PUPILS"] = 1000.0 * safe_divide(out["NUMBER_OF_SOCIAL_WORKERS"], out["N_PUPILS"])

    if {"PERCENT_FREE_LUNCH", "PERCENT_REDUCED_LUNCH"}.issubset(out.columns):
        out["PERCENT_FREE_OR_REDUCED_LUNCH"] = out["PERCENT_FREE_LUNCH"] + out["PERCENT_REDUCED_LUNCH"]

    # Grade composition.
    grade_cols = [c for c in out.columns if re.fullmatch(r"(PRE_K|K|GRADE_\d{2})", c)]
    if grade_cols and "N_PUPILS" in out.columns:
        elem_cols = [c for c in ["PRE_K", "K", "GRADE_01", "GRADE_02", "GRADE_03", "GRADE_04", "GRADE_05"] if c in out.columns]
        mid_cols = [c for c in ["GRADE_06", "GRADE_07", "GRADE_08"] if c in out.columns]
        high_cols = [c for c in ["GRADE_09", "GRADE_10", "GRADE_11", "GRADE_12"] if c in out.columns]
        out["ELEM_ENROLLMENT"] = out[elem_cols].sum(axis=1) if elem_cols else 0
        out["MIDDLE_ENROLLMENT"] = out[mid_cols].sum(axis=1) if mid_cols else 0
        out["HIGH_ENROLLMENT"] = out[high_cols].sum(axis=1) if high_cols else 0
        out["ELEM_SHARE"] = safe_divide(out["ELEM_ENROLLMENT"], out["N_PUPILS"])
        out["MIDDLE_SHARE"] = safe_divide(out["MIDDLE_ENROLLMENT"], out["N_PUPILS"])
        out["HIGH_SHARE"] = safe_divide(out["HIGH_ENROLLMENT"], out["N_PUPILS"])

    # Subject-matched class-size feature.
    out["SUBJECT_MATCHED_CLASS_SIZE"] = np.nan
    class_size_map = {
        "ELA": "LANGUAGE_ARTS_AVERAGE_CLASS_SIZE",
        "MATH": "MATHEMATICS_AVERAGE_CLASS_SIZE",
        "SCIENCE": "SCIENCE_AVERAGE_CLASS_SIZE",
        "HISTORY": "HISTORY_GOVERNMENT_AND_GEOGRAPHY_AVERAGE_CLASS_SIZE",
    }
    for subject, col in class_size_map.items():
        if col in out.columns:
            out.loc[out["SUBJECT"] == subject, "SUBJECT_MATCHED_CLASS_SIZE"] = out.loc[out["SUBJECT"] == subject, col]

    # Missingness indicators for numeric columns. This often helps tree and linear models.
    numeric_cols = out.select_dtypes(include=[np.number]).columns.tolist()
    for c in numeric_cols:
        if c not in [TARGET] and out[c].isna().any():
            out[f"missing_{c}"] = out[c].isna().astype(int)

    return out


def add_count_features(train_base: pd.DataFrame, test_base: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Adds non-target count/frequency features. Counts are not target leakage.
    Training-history counts are computed from train only.
    Combined row counts are computed from train + test because they use only feature availability.
    """
    train = train_base.copy()
    test = test_base.copy()
    combined = pd.concat([train.assign(__part="train"), test.assign(__part="test")], axis=0, ignore_index=True)

    count_groups = {
        "cnt_school_all_rows": ["SCHOOL"],
        "cnt_district_all_rows": ["DISTRICT"],
        "cnt_assessment_all_rows": ["ASSESSMENT_NAME"],
        "cnt_school_assessment_all_rows": ["SCHOOL", "ASSESSMENT_NAME"],
        "cnt_school_subgroup_all_rows": ["SCHOOL", "SUBGROUP_NAME"],
        "cnt_district_assessment_all_rows": ["DISTRICT", "ASSESSMENT_NAME"],
        "cnt_county_assessment_subgroup_all_rows": ["COUNTY", "ASSESSMENT_NAME", "SUBGROUP_NAME"],
    }

    for new_col, cols in count_groups.items():
        if all(c in combined.columns for c in cols):
            counts = combined.groupby(cols).size().rename(new_col).reset_index()
            combined = combined.merge(counts, on=cols, how="left")

    combined = combined.drop(columns=["__part"])
    train_out = combined.iloc[: len(train)].reset_index(drop=True)
    test_out = combined.iloc[len(train) :].reset_index(drop=True)
    return train_out, test_out


def prepare_base_tables(train: pd.DataFrame, test: pd.DataFrame, school: pd.DataFrame, district: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_joined = add_joined_covariates(train, school, district)
    test_joined = add_joined_covariates(test, school, district)

    train_base = add_base_features(train_joined).reset_index(drop=True)
    test_base = add_base_features(test_joined).reset_index(drop=True)

    # Fill categorical missing values early so groupby keys are stable.
    all_cat_cols = sorted(set(BASE_CATEGORICAL_COLS + ["GRADE_BAND", "SUBJECT", "ASSESSMENT_FAMILY"]))
    for c in all_cat_cols:
        if c in train_base.columns:
            train_base[c] = train_base[c].astype("string").fillna("UNKNOWN").astype(str)
        if c in test_base.columns:
            test_base[c] = test_base[c].astype("string").fillna("UNKNOWN").astype(str)

    train_base, test_base = add_count_features(train_base, test_base)
    return train_base, test_base


def fit_smooth_mapping(fit_df: pd.DataFrame, cols: List[str], alpha: float, global_mean: float) -> pd.DataFrame:
    stats = fit_df.groupby(cols, dropna=False)[TARGET].agg(["count", "mean"]).reset_index()
    stats["__te_value"] = (stats["count"] * stats["mean"] + alpha * global_mean) / (stats["count"] + alpha)
    return stats[cols + ["__te_value"]]


def apply_smooth_mapping(dest_df: pd.DataFrame, mapping: pd.DataFrame, cols: List[str], name: str, global_mean: float) -> pd.Series:
    tmp = dest_df[cols].merge(mapping, on=cols, how="left")["__te_value"]
    return tmp.fillna(global_mean).astype(float)


def add_target_encodings_from_fit(fit_df: pd.DataFrame, dest_df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds target-encoding columns to dest_df using only fit_df target values.
    """
    out = dest_df.copy()
    global_mean = float(fit_df[TARGET].mean())
    out["te_global"] = global_mean

    for name, cols, alpha in GROUP_SPECS:
        if all(c in fit_df.columns for c in cols) and all(c in out.columns for c in cols):
            mapping = fit_smooth_mapping(fit_df, cols, alpha, global_mean)
            out[name] = apply_smooth_mapping(out, mapping, cols, name, global_mean)
        else:
            out[name] = global_mean

    return out


def make_oof_target_encodings(train_base_with_y: pd.DataFrame, n_splits: int) -> pd.DataFrame:
    """
    For every training row, creates target encodings from other folds only.
    This prevents each row's own target from being used in its encoded features.
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    te_cols = ["te_global"] + [name for name, _, _ in GROUP_SPECS]
    out = pd.DataFrame(index=train_base_with_y.index, columns=te_cols, dtype=float)

    for inner_i, (fit_idx, val_idx) in enumerate(kf.split(train_base_with_y), start=1):
        fit_df = train_base_with_y.iloc[fit_idx].copy()
        val_df = train_base_with_y.iloc[val_idx].drop(columns=[TARGET], errors="ignore").copy()
        encoded = add_target_encodings_from_fit(fit_df, val_df)
        out.iloc[val_idx, :] = encoded[te_cols].to_numpy()
    return out.astype(float)


def add_te_columns(base_df: pd.DataFrame, te_df: pd.DataFrame) -> pd.DataFrame:
    out = base_df.drop(columns=[TARGET], errors="ignore").copy()
    for c in te_df.columns:
        out[c] = te_df[c].to_numpy(dtype=float)
    return out


def choose_feature_columns(X: pd.DataFrame) -> Tuple[List[str], List[str]]:
    drop_cols = {ID_COL, TARGET}
    cat_cols = [c for c in BASE_CATEGORICAL_COLS if c in X.columns and c not in drop_cols]

    # Keep grade extracted as numeric. IDs and raw string columns not in BASE_CATEGORICAL_COLS are dropped.
    candidate_cols = [c for c in X.columns if c not in drop_cols]
    numeric_cols = [
        c for c in candidate_cols
        if c not in cat_cols and pd.api.types.is_numeric_dtype(X[c])
    ]

    # Keep only categoricals and numerics.
    return cat_cols, numeric_cols


def subset_features(X: pd.DataFrame, cat_cols: List[str], num_cols: List[str]) -> pd.DataFrame:
    cols = cat_cols + num_cols
    out = X[cols].copy()
    for c in cat_cols:
        out[c] = out[c].astype("string").fillna("UNKNOWN").astype(str)
    return out


def build_feature_set(
    train_base: pd.DataFrame,
    test_base: pd.DataFrame,
    n_splits: int,
    inner_splits: int,
) -> FeatureSet:
    """
    Builds final train/test features and fold-specific validation features.

    For model evaluation:
      - Each outer validation fold is encoded using only the outer training fold.
      - The outer training fold's target encodings are inner-OOF encoded.

    For final test prediction:
      - Full training rows are OOF encoded.
      - Test rows are encoded from all training rows.
    """
    log("Building leakage-safe target encodings for final train/test features...")
    train_with_y = train_base.copy()
    if TARGET not in train_with_y.columns:
        raise ValueError("train_base must include target.")

    y = train_with_y[TARGET].astype(float).reset_index(drop=True)

    full_train_te = make_oof_target_encodings(train_with_y, n_splits=inner_splits)
    full_fit_df = train_with_y.copy()
    test_encoded = add_target_encodings_from_fit(full_fit_df, test_base)

    X_train_full = add_te_columns(train_base, full_train_te).reset_index(drop=True)
    X_test_full = test_encoded.drop(columns=[TARGET], errors="ignore").reset_index(drop=True)

    cat_cols, num_cols = choose_feature_columns(X_train_full)
    X_train_full = subset_features(X_train_full, cat_cols, num_cols)
    X_test_full = subset_features(X_test_full, cat_cols, num_cols)

    log(f"Selected {len(num_cols):,} numeric features and {len(cat_cols):,} categorical features.")

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    folds = []
    for fold_num, (tr_idx, val_idx) in enumerate(kf.split(train_base), start=1):
        log(f"Preparing fold {fold_num}/{n_splits} features...")
        tr_base = train_base.iloc[tr_idx].reset_index(drop=True)
        val_base = train_base.iloc[val_idx].reset_index(drop=True)
        y_tr = tr_base[TARGET].astype(float).reset_index(drop=True)
        y_val = val_base[TARGET].astype(float).reset_index(drop=True)

        tr_te = make_oof_target_encodings(tr_base, n_splits=min(inner_splits, max(2, len(tr_base) // 5000)))
        val_encoded = add_target_encodings_from_fit(tr_base, val_base.drop(columns=[TARGET], errors="ignore"))

        X_tr = subset_features(add_te_columns(tr_base, tr_te), cat_cols, num_cols)
        X_val = subset_features(val_encoded.drop(columns=[TARGET], errors="ignore"), cat_cols, num_cols)

        folds.append((tr_idx, val_idx, X_tr, y_tr, X_val, y_val))

    return FeatureSet(
        train_full=X_train_full,
        test_full=X_test_full,
        folds=folds,
        cat_cols=cat_cols,
        num_cols=num_cols,
    )


def make_ridge_pipeline(cat_cols: List[str], num_cols: List[str]) -> Pipeline:
    preprocess = ColumnTransformer(
        transformers=[
            ("num", Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]), num_cols),
            ("cat", one_hot_encoder(), cat_cols),
        ],
        sparse_threshold=0.4,
        remainder="drop",
    )

    model = Ridge(alpha=25.0, solver="lsqr", random_state=SEED)
    return Pipeline([("preprocess", preprocess), ("model", model)])


def run_ridge(features: FeatureSet, y: pd.Series, test_ids: pd.Series, clip: bool, out_dir: Path) -> Tuple[np.ndarray, np.ndarray, Dict]:
    log("Running Ridge model...")
    oof = np.zeros(len(y), dtype=float)
    fold_scores = []

    for fold_num, (tr_idx, val_idx, X_tr, y_tr, X_val, y_val) in enumerate(features.folds, start=1):
        model = make_ridge_pipeline(features.cat_cols, features.num_cols)
        model.fit(X_tr, y_tr)
        pred = clip_pred(model.predict(X_val), clip)
        oof[val_idx] = pred
        fold_scores.append({"model": "ridge", "fold": fold_num, "mse": mse(y_val, pred), "rmse": rmse(y_val, pred)})
        log(f"  Ridge fold {fold_num}: MSE={fold_scores[-1]['mse']:.4f}, RMSE={fold_scores[-1]['rmse']:.4f}")

    final_model = make_ridge_pipeline(features.cat_cols, features.num_cols)
    final_model.fit(features.train_full, y)
    test_pred = clip_pred(final_model.predict(features.test_full), clip)

    return oof, test_pred, {"fold_scores": fold_scores, "feature_importance": None}


def run_hgb(features: FeatureSet, y: pd.Series, test_ids: pd.Series, clip: bool, out_dir: Path) -> Tuple[np.ndarray, np.ndarray, Dict]:
    log("Running HistGradientBoostingRegressor model...")
    oof = np.zeros(len(y), dtype=float)
    fold_scores = []

    num_cols = features.num_cols
    X_full = features.train_full[num_cols].copy()
    X_test = features.test_full[num_cols].copy()

    for fold_num, (tr_idx, val_idx, X_tr_raw, y_tr, X_val_raw, y_val) in enumerate(features.folds, start=1):
        X_tr = X_tr_raw[num_cols].copy()
        X_val = X_val_raw[num_cols].copy()

        model = HistGradientBoostingRegressor(
            loss="squared_error",
            learning_rate=0.045,
            max_iter=900,
            max_leaf_nodes=31,
            min_samples_leaf=30,
            l2_regularization=0.05,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=50,
            random_state=SEED + fold_num,
        )
        model.fit(X_tr, y_tr)
        pred = clip_pred(model.predict(X_val), clip)
        oof[val_idx] = pred
        fold_scores.append({"model": "hgb", "fold": fold_num, "mse": mse(y_val, pred), "rmse": rmse(y_val, pred)})
        log(f"  HGB fold {fold_num}: MSE={fold_scores[-1]['mse']:.4f}, RMSE={fold_scores[-1]['rmse']:.4f}")

    final_model = HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=0.045,
        max_iter=900,
        max_leaf_nodes=31,
        min_samples_leaf=30,
        l2_regularization=0.05,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=50,
        random_state=SEED,
    )
    final_model.fit(X_full, y)
    test_pred = clip_pred(final_model.predict(X_test), clip)

    return oof, test_pred, {"fold_scores": fold_scores, "feature_importance": None}


def prep_category_dtypes(train_df: pd.DataFrame, test_df: pd.DataFrame, cat_cols: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train = train_df.copy()
    test = test_df.copy()
    for c in cat_cols:
        combined = pd.concat([train[c], test[c]], axis=0).astype("string").fillna("UNKNOWN").astype(str)
        cats = pd.Categorical(combined).categories
        train[c] = pd.Categorical(train[c].astype("string").fillna("UNKNOWN").astype(str), categories=cats)
        test[c] = pd.Categorical(test[c].astype("string").fillna("UNKNOWN").astype(str), categories=cats)
    return train, test


def run_lgbm(features: FeatureSet, y: pd.Series, test_ids: pd.Series, clip: bool, out_dir: Path) -> Tuple[np.ndarray, np.ndarray, Dict]:
    try:
        import lightgbm as lgb
    except Exception as e:
        log(f"Skipping LightGBM because it is not installed/importable: {e}")
        raise RuntimeError("LightGBM unavailable")

    log("Running LightGBM model...")
    oof = np.zeros(len(y), dtype=float)
    fold_scores = []
    best_iters = []
    importances = []

    # The high-cardinality SCHOOL and DISTRICT IDs are already represented by
    # smoothed target encodings and count features. Passing them as raw LightGBM
    # categoricals can be slow and can overfit, so LGBM gets the lower-cardinality
    # categorical fields plus all numeric features.
    lgb_cat_cols = [c for c in features.cat_cols if c not in ["SCHOOL", "DISTRICT"]]
    lgb_cols = lgb_cat_cols + features.num_cols
    X_full_lgb, X_test_lgb = prep_category_dtypes(features.train_full[lgb_cols], features.test_full[lgb_cols], lgb_cat_cols)

    for fold_num, (tr_idx, val_idx, X_tr_raw, y_tr, X_val_raw, y_val) in enumerate(features.folds, start=1):
        X_tr, X_val = prep_category_dtypes(X_tr_raw[lgb_cols], X_val_raw[lgb_cols], lgb_cat_cols)

        model = lgb.LGBMRegressor(
            objective="regression_l2",
            metric="l2",
            n_estimators=2200,
            learning_rate=0.035,
            num_leaves=31,
            max_depth=-1,
            min_child_samples=25,
            subsample=0.88,
            subsample_freq=1,
            colsample_bytree=0.88,
            reg_alpha=0.05,
            reg_lambda=2.0,
            random_state=SEED + fold_num,
            n_jobs=4,
            verbosity=-1,
        )
        callbacks = [
            lgb.early_stopping(stopping_rounds=100, verbose=False),
            lgb.log_evaluation(period=250),
        ]
        model.fit(
            X_tr,
            y_tr,
            eval_set=[(X_val, y_val)],
            eval_metric="l2",
            categorical_feature=lgb_cat_cols,
            callbacks=callbacks,
        )
        pred = clip_pred(model.predict(X_val, num_iteration=model.best_iteration_), clip)
        oof[val_idx] = pred
        best_iter = int(model.best_iteration_ or model.n_estimators)
        best_iters.append(best_iter)
        fold_scores.append({"model": "lgbm", "fold": fold_num, "mse": mse(y_val, pred), "rmse": rmse(y_val, pred), "best_iteration": best_iter})
        log(f"  LightGBM fold {fold_num}: MSE={fold_scores[-1]['mse']:.4f}, RMSE={fold_scores[-1]['rmse']:.4f}, best_iter={best_iter}")

        importances.append(pd.DataFrame({
            "feature": X_tr.columns,
            f"importance_fold_{fold_num}": model.feature_importances_,
        }))

    final_n_estimators = int(np.clip(np.mean(best_iters) * 1.10, 300, 2600)) if best_iters else 1200
    log(f"Training final LightGBM with n_estimators={final_n_estimators}...")
    final_model = lgb.LGBMRegressor(
        objective="regression_l2",
        metric="l2",
        n_estimators=final_n_estimators,
        learning_rate=0.025,
        num_leaves=63,
        max_depth=-1,
        min_child_samples=25,
        subsample=0.88,
        subsample_freq=1,
        colsample_bytree=0.88,
        reg_alpha=0.05,
        reg_lambda=2.0,
        random_state=SEED,
        n_jobs=4,
        verbosity=-1,
    )
    final_model.fit(X_full_lgb, y, categorical_feature=lgb_cat_cols)
    test_pred = clip_pred(final_model.predict(X_test_lgb), clip)

    # Feature importance summary.
    fi = pd.DataFrame({"feature": X_full_lgb.columns, "importance_final": final_model.feature_importances_})
    for imp in importances:
        fi = fi.merge(imp, on="feature", how="left")
    importance_cols = [c for c in fi.columns if c.startswith("importance")]
    fi["importance_mean"] = fi[importance_cols].mean(axis=1)
    fi = fi.sort_values("importance_mean", ascending=False)
    fi.to_csv(out_dir / "diagnostics" / "feature_importance_lgbm.csv", index=False)

    return oof, test_pred, {"fold_scores": fold_scores, "feature_importance": fi}


def run_catboost(features: FeatureSet, y: pd.Series, test_ids: pd.Series, clip: bool, out_dir: Path) -> Tuple[np.ndarray, np.ndarray, Dict]:
    try:
        from catboost import CatBoostRegressor, Pool
    except Exception as e:
        log(f"Skipping CatBoost because it is not installed/importable: {e}")
        raise RuntimeError("CatBoost unavailable")

    log("Running CatBoost model...")
    oof = np.zeros(len(y), dtype=float)
    fold_scores = []
    best_iters = []
    importances = []

    all_cols = features.cat_cols + features.num_cols
    cat_idx = [all_cols.index(c) for c in features.cat_cols]

    X_full = features.train_full[all_cols].copy()
    X_test = features.test_full[all_cols].copy()
    for c in features.cat_cols:
        X_full[c] = X_full[c].astype(str).fillna("UNKNOWN")
        X_test[c] = X_test[c].astype(str).fillna("UNKNOWN")

    for fold_num, (tr_idx, val_idx, X_tr_raw, y_tr, X_val_raw, y_val) in enumerate(features.folds, start=1):
        X_tr = X_tr_raw[all_cols].copy()
        X_val = X_val_raw[all_cols].copy()
        for c in features.cat_cols:
            X_tr[c] = X_tr[c].astype(str).fillna("UNKNOWN")
            X_val[c] = X_val[c].astype(str).fillna("UNKNOWN")

        train_pool = Pool(X_tr, y_tr, cat_features=cat_idx)
        val_pool = Pool(X_val, y_val, cat_features=cat_idx)

        model = CatBoostRegressor(
            loss_function="RMSE",
            eval_metric="RMSE",
            iterations=3500,
            learning_rate=0.025,
            depth=6,
            l2_leaf_reg=8.0,
            random_seed=SEED + fold_num,
            od_type="Iter",
            od_wait=150,
            verbose=250,
            allow_writing_files=False,
            thread_count=4,
        )
        model.fit(train_pool, eval_set=val_pool, use_best_model=True)
        pred = clip_pred(model.predict(val_pool), clip)
        oof[val_idx] = pred
        best_iter = int(model.get_best_iteration() or model.tree_count_)
        best_iters.append(best_iter)
        fold_scores.append({"model": "cat", "fold": fold_num, "mse": mse(y_val, pred), "rmse": rmse(y_val, pred), "best_iteration": best_iter})
        log(f"  CatBoost fold {fold_num}: MSE={fold_scores[-1]['mse']:.4f}, RMSE={fold_scores[-1]['rmse']:.4f}, best_iter={best_iter}")

        importances.append(pd.DataFrame({
            "feature": all_cols,
            f"importance_fold_{fold_num}": model.get_feature_importance(),
        }))

    final_iterations = int(np.clip(np.mean(best_iters) * 1.10, 300, 4500)) if best_iters else 1500
    log(f"Training final CatBoost with iterations={final_iterations}...")
    full_pool = Pool(X_full, y, cat_features=cat_idx)
    test_pool = Pool(X_test, cat_features=cat_idx)

    final_model = CatBoostRegressor(
        loss_function="RMSE",
        eval_metric="RMSE",
        iterations=final_iterations,
        learning_rate=0.025,
        depth=6,
        l2_leaf_reg=8.0,
        random_seed=SEED,
        verbose=250,
        allow_writing_files=False,
        thread_count=4,
    )
    final_model.fit(full_pool)
    test_pred = clip_pred(final_model.predict(test_pool), clip)

    fi = pd.DataFrame({"feature": all_cols, "importance_final": final_model.get_feature_importance(full_pool)})
    for imp in importances:
        fi = fi.merge(imp, on="feature", how="left")
    importance_cols = [c for c in fi.columns if c.startswith("importance")]
    fi["importance_mean"] = fi[importance_cols].mean(axis=1)
    fi = fi.sort_values("importance_mean", ascending=False)
    fi.to_csv(out_dir / "diagnostics" / "feature_importance_catboost.csv", index=False)

    return oof, test_pred, {"fold_scores": fold_scores, "feature_importance": fi}


def optimize_blend_weights(oof_preds: Dict[str, np.ndarray], y: np.ndarray, n_draws: int = 40000) -> Dict[str, float]:
    """
    Finds non-negative weights summing to one using random Dirichlet search.
    This is simple, transparent, and avoids another optimization dependency.
    """
    names = list(oof_preds.keys())
    if len(names) == 1:
        return {names[0]: 1.0}

    P = np.column_stack([oof_preds[n] for n in names])
    rng = np.random.default_rng(SEED)
    best_w = np.ones(len(names)) / len(names)
    best_score = mse(y, P @ best_w)

    # Include equal weights and single-model weights.
    candidates = [best_w]
    for i in range(len(names)):
        w = np.zeros(len(names))
        w[i] = 1.0
        candidates.append(w)

    # Random convex combinations.
    candidates.extend(rng.dirichlet(np.ones(len(names)), size=n_draws))

    for w in candidates:
        pred = P @ w
        score = mse(y, pred)
        if score < best_score:
            best_score = score
            best_w = w

    return {name: float(weight) for name, weight in zip(names, best_w)}


def save_submission(test_ids: pd.Series, pred: np.ndarray, path: Path) -> None:
    sub = pd.DataFrame({
        ID_COL: test_ids.values,
        TARGET: pred.astype(float),
    })
    if sub[ID_COL].duplicated().any():
        raise ValueError("Submission contains duplicated ASSESSMENT_ID values.")
    sub.to_csv(path, index=False)


def save_diagnostics(
    train_base: pd.DataFrame,
    y: pd.Series,
    oof_preds: Dict[str, np.ndarray],
    blend_pred: np.ndarray,
    scores: List[Dict],
    weights: Dict[str, float],
    out_dir: Path,
) -> None:
    diag_dir = out_dir / "diagnostics"
    diag_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(scores).to_csv(diag_dir / "cv_fold_scores.csv", index=False)

    summary_rows = []
    for name, pred in oof_preds.items():
        summary_rows.append({"model": name, "mse": mse(y, pred), "rmse": rmse(y, pred), "weight_in_blend": weights.get(name, 0.0)})
    summary_rows.append({"model": "blend", "mse": mse(y, blend_pred), "rmse": rmse(y, blend_pred), "weight_in_blend": 1.0})
    pd.DataFrame(summary_rows).sort_values("mse").to_csv(diag_dir / "cv_model_summary.csv", index=False)

    fold_pred = pd.DataFrame({
        ID_COL: train_base[ID_COL].values,
        "actual": y.values,
        "prediction_blend": blend_pred,
        "residual_blend": y.values - blend_pred,
        "squared_error_blend": (y.values - blend_pred) ** 2,
    })
    for name, pred in oof_preds.items():
        fold_pred[f"prediction_{name}"] = pred
    fold_pred.to_csv(diag_dir / "oof_predictions.csv", index=False)

    # Error drilldowns for pipeline-design and private-leaderboard risk checks.
    diag_source = train_base[[ID_COL, "SCHOOL", "DISTRICT", "COUNTY", "REGION", "DISTRICT_TYPE", "SUBGROUP_NAME", "ASSESSMENT_NAME", "SUBJECT", "ASSESSMENT_FAMILY", "GRADE_BAND", "N_STUDENTS"]].copy()
    diag_source["actual"] = y.values
    diag_source["pred"] = blend_pred
    diag_source["residual"] = diag_source["actual"] - diag_source["pred"]
    diag_source["squared_error"] = diag_source["residual"] ** 2
    diag_source["abs_error"] = diag_source["residual"].abs()

    for col in ["SUBGROUP_NAME", "ASSESSMENT_NAME", "SUBJECT", "ASSESSMENT_FAMILY", "GRADE_BAND", "COUNTY", "DISTRICT_TYPE", "REGION"]:
        if col in diag_source.columns:
            out = (
                diag_source.groupby(col, dropna=False)
                .agg(
                    n_rows=("squared_error", "size"),
                    mse=("squared_error", "mean"),
                    rmse=("squared_error", lambda s: float(np.sqrt(np.mean(s)))),
                    mae=("abs_error", "mean"),
                    mean_actual=("actual", "mean"),
                    mean_pred=("pred", "mean"),
                    mean_residual=("residual", "mean"),
                )
                .reset_index()
                .sort_values("mse", ascending=False)
            )
            out.to_csv(diag_dir / f"error_by_{col.lower()}.csv", index=False)

    # N_STUDENTS deciles and prediction deciles.
    try:
        diag_source["n_students_decile"] = pd.qcut(diag_source["N_STUDENTS"], q=10, duplicates="drop")
        ndec = (
            diag_source.groupby("n_students_decile", observed=False)
            .agg(
                n_rows=("squared_error", "size"),
                mse=("squared_error", "mean"),
                rmse=("squared_error", lambda s: float(np.sqrt(np.mean(s)))),
                mae=("abs_error", "mean"),
                mean_actual=("actual", "mean"),
                mean_pred=("pred", "mean"),
                mean_residual=("residual", "mean"),
            )
            .reset_index()
        )
        ndec.to_csv(diag_dir / "error_by_n_students_decile.csv", index=False)
    except Exception:
        pass

    try:
        diag_source["pred_decile"] = pd.qcut(diag_source["pred"], q=10, duplicates="drop")
        pdec = (
            diag_source.groupby("pred_decile", observed=False)
            .agg(
                n_rows=("squared_error", "size"),
                mse=("squared_error", "mean"),
                rmse=("squared_error", lambda s: float(np.sqrt(np.mean(s)))),
                mae=("abs_error", "mean"),
                mean_actual=("actual", "mean"),
                mean_pred=("pred", "mean"),
                mean_residual=("residual", "mean"),
            )
            .reset_index()
        )
        pdec.to_csv(diag_dir / "error_by_prediction_decile.csv", index=False)
    except Exception:
        pass

    with open(diag_dir / "blend_weights.json", "w", encoding="utf-8") as f:
        json.dump(weights, f, indent=2)

    env = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "python": sys.version,
        "platform": platform.platform(),
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "sklearn": __import__("sklearn").__version__,
        "seed": SEED,
        "models_in_blend": list(weights.keys()),
        "blend_weights": weights,
    }
    with open(diag_dir / "run_metadata.json", "w", encoding="utf-8") as f:
        json.dump(env, f, indent=2)


def main() -> None:
    args = parse_args()
    requested_models = [m.strip().lower() for m in args.models.split(",") if m.strip()]
    invalid = sorted(set(requested_models) - {"ridge", "lgbm", "cat", "hgb"})
    if invalid:
        raise ValueError(f"Invalid model name(s): {invalid}")

    project_root = find_project_root()
    data_dir = project_root / "data" / "raw"
    out_dir = project_root / "outputs"
    sub_dir = out_dir / "submissions"
    diag_dir = out_dir / "diagnostics"
    sub_dir.mkdir(parents=True, exist_ok=True)
    diag_dir.mkdir(parents=True, exist_ok=True)

    log(f"Project root: {project_root}")
    log(f"Data directory: {data_dir}")
    log("Loading data...")
    train, test, school, district = load_data(data_dir, args.sample_train_rows, args.sample_test_rows)

    log(f"Train rows: {len(train):,}; Test rows: {len(test):,}; Schools: {school['SCHOOL'].nunique():,}")
    if test[ID_COL].duplicated().any():
        raise ValueError("Test ASSESSMENT_ID contains duplicates.")
    if train[TARGET].isna().any():
        raise ValueError("Training target contains missing values.")

    log("Preparing joined base tables and engineered non-target features...")
    train_base, test_base = prepare_base_tables(train, test, school, district)

    y = train_base[TARGET].astype(float).reset_index(drop=True)
    test_ids = test_base[ID_COL].reset_index(drop=True)

    n_splits = min(args.n_splits, len(train_base))
    inner_splits = min(args.inner_splits, len(train_base))
    if n_splits < 2 or inner_splits < 2:
        raise ValueError("Need at least 2 folds for validation and target encoding.")

    features = build_feature_set(
        train_base=train_base,
        test_base=test_base,
        n_splits=n_splits,
        inner_splits=inner_splits,
    )

    all_scores: List[Dict] = []
    oof_preds: Dict[str, np.ndarray] = {}
    test_preds: Dict[str, np.ndarray] = {}

    runners = {
        "ridge": run_ridge,
        "hgb": run_hgb,
        "lgbm": run_lgbm,
        "cat": run_catboost,
    }

    do_clip = not args.no_clip

    for model_name in requested_models:
        try:
            oof, test_pred, info = runners[model_name](features, y, test_ids, do_clip, out_dir)
        except RuntimeError as e:
            log(f"Model {model_name} was skipped: {e}")
            continue

        oof_preds[model_name] = oof
        test_preds[model_name] = test_pred
        all_scores.extend(info.get("fold_scores", []))

        model_mse = mse(y, oof)
        model_rmse = math.sqrt(model_mse)
        log(f"{model_name} overall OOF: MSE={model_mse:.4f}, RMSE={model_rmse:.4f}")

        save_submission(test_ids, test_pred, sub_dir / f"submission_{model_name}.csv")

    if not oof_preds:
        raise RuntimeError("No models were successfully trained.")

    weights = optimize_blend_weights(oof_preds, y.to_numpy())
    log(f"Optimized blend weights: {weights}")

    blend_oof = np.zeros(len(y), dtype=float)
    blend_test = np.zeros(len(test_ids), dtype=float)
    for name, weight in weights.items():
        blend_oof += weight * oof_preds[name]
        blend_test += weight * test_preds[name]
    blend_oof = clip_pred(blend_oof, do_clip)
    blend_test = clip_pred(blend_test, do_clip)

    log(f"Blend overall OOF: MSE={mse(y, blend_oof):.4f}, RMSE={rmse(y, blend_oof):.4f}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_submission(test_ids, blend_test, sub_dir / "submission_blend.csv")
    save_submission(test_ids, blend_test, sub_dir / f"submission_blend_{timestamp}.csv")

    if not args.skip_diagnostics:
        log("Saving diagnostics...")
        save_diagnostics(
            train_base=train_base,
            y=y,
            oof_preds=oof_preds,
            blend_pred=blend_oof,
            scores=all_scores,
            weights=weights,
            out_dir=out_dir,
        )

    log("Done.")
    log(f"Primary Kaggle file: {sub_dir / 'submission_blend.csv'}")
    log(f"Individual submissions are in: {sub_dir}")
    log(f"Diagnostics are in: {diag_dir}")


if __name__ == "__main__":
    main()
