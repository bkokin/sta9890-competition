"""
STA 9890 Competition - single-file modeling pipeline v9

Purpose:
    Build a strong, reproducible model using only instructor-provided files.

What this script does:
    1. Loads the four course-provided CSV files from data/raw/.
    2. Joins school and district covariates to score records.
    3. Creates assessment, school-composition, staffing, funding, and count features.
    4. Adds unsupervised PCA and KMeans school/district structure features from covariates only.
    5. Builds leakage-safe target-encoding features inside cross-validation.
    6. Builds leakage-safe same-school/assessment peer features and deterministic
       count-space algebra predictions from subgroup identities: All = Male/Female
       weighted average and All = Econ/Not Econ weighted average.
    7. Trains the final focused optimized model stack for this data:
          - the two xlong LightGBM variants that improved v7
          - one additional seed for each xlong variant for controlled diversity
          - optional older/weaker models only if explicitly requested
    8. Blends models using out-of-fold predictions only.
    9. Post-processes the blend with leakage-safe algebra constraints.
    10. Saves Kaggle-ready submissions plus diagnostics for overfitting, fold stability,
       constraint reliability, and error drilldowns.

Important:
    - This script uses no outside data.
    - Validation target encodings are fit only on the training portion of each fold.
    - Final submission file columns are ASSESSMENT_ID and PERCENT_PROFICIENT.

Recommended full run:
    python sta9890_competition_model_v9.py

Efficient single-model run:
    python sta9890_competition_model_v9.py --models lgbm_deep_xlong

Focused diagnostic run:
    python sta9890_competition_model_v9.py --models lgbm_deep_xlong,lgbm_deep_xlong_seed2,lgbm_deep_bagged_xlong,lgbm_deep_bagged_xlong_seed2

Smoke test:
    python sta9890_competition_model_v9.py --models lgbm_deep_xlong --sample-train-rows 25000 --sample-test-rows 5000 --n-splits 2 --inner-splits 2
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

from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Ridge, RidgeCV, SGDRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore", category=FutureWarning)
pd.options.mode.copy_on_write = True
# Avoid harmless Windows/joblib warning during KMeans physical-core detection.
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 8))


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
    # v9 richer encodings: medium-cardinality interactions that generalize better than
    # exact SCHOOL + SUBGROUP + ASSESSMENT but are more specific than subject alone.
    ("te_school_family_subgroup", ["SCHOOL", "ASSESSMENT_FAMILY", "SUBGROUP_NAME"], 28.0),
    ("te_school_grade_band_subgroup", ["SCHOOL", "GRADE_BAND", "SUBGROUP_NAME"], 28.0),
    ("te_school", ["SCHOOL"], 50.0),

    ("te_district_assessment_subgroup", ["DISTRICT", "ASSESSMENT_NAME", "SUBGROUP_NAME"], 25.0),
    ("te_district_assessment", ["DISTRICT", "ASSESSMENT_NAME"], 30.0),
    ("te_district_subject_subgroup", ["DISTRICT", "SUBJECT", "SUBGROUP_NAME"], 30.0),
    ("te_district_family_grade_subgroup", ["DISTRICT", "ASSESSMENT_FAMILY", "GRADE_BAND", "SUBGROUP_NAME"], 38.0),
    ("te_district_grade_band_subgroup", ["DISTRICT", "GRADE_BAND", "SUBGROUP_NAME"], 38.0),
    ("te_district_subgroup", ["DISTRICT", "SUBGROUP_NAME"], 40.0),
    ("te_district", ["DISTRICT"], 80.0),

    ("te_county_assessment_subgroup", ["COUNTY", "ASSESSMENT_NAME", "SUBGROUP_NAME"], 35.0),
    ("te_county_assessment", ["COUNTY", "ASSESSMENT_NAME"], 50.0),
    ("te_county_subject_subgroup", ["COUNTY", "SUBJECT", "SUBGROUP_NAME"], 50.0),
    ("te_county_family_grade_subgroup", ["COUNTY", "ASSESSMENT_FAMILY", "GRADE_BAND", "SUBGROUP_NAME"], 60.0),
    ("te_county_grade_band_subgroup", ["COUNTY", "GRADE_BAND", "SUBGROUP_NAME"], 60.0),
    ("te_county_subgroup", ["COUNTY", "SUBGROUP_NAME"], 70.0),
    ("te_county", ["COUNTY"], 100.0),

    ("te_region_assessment_subgroup", ["REGION", "ASSESSMENT_NAME", "SUBGROUP_NAME"], 40.0),
    ("te_district_type_assessment_subgroup", ["DISTRICT_TYPE", "ASSESSMENT_NAME", "SUBGROUP_NAME"], 40.0),
    ("te_region_family_grade_subgroup", ["REGION", "ASSESSMENT_FAMILY", "GRADE_BAND", "SUBGROUP_NAME"], 65.0),
    ("te_district_type_family_grade_subgroup", ["DISTRICT_TYPE", "ASSESSMENT_FAMILY", "GRADE_BAND", "SUBGROUP_NAME"], 65.0),
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
    "ALGEBRA_METHOD",
    # Unsupervised covariate cluster labels. They are created from instructor-provided
    # school/district covariates only, with no target values.
    "SCHOOL_STRUCT_CLUSTER_8",
    "SCHOOL_STRUCT_CLUSTER_20",
    "DISTRICT_STRUCT_CLUSTER_5",
    "DISTRICT_STRUCT_CLUSTER_10",
]


@dataclass
class FeatureSet:
    train_full: pd.DataFrame
    test_full: pd.DataFrame
    folds: List[Tuple[np.ndarray, np.ndarray, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]]
    cat_cols: List[str]
    num_cols: List[str]
    # These are the algebra predictions actually available to the OUTER validation rows.
    # They are kept separately because train_full uses inner-OOF algebra features for model fitting.
    outer_alg_pred: np.ndarray
    outer_alg_method: np.ndarray
    test_alg_pred: np.ndarray
    test_alg_method: np.ndarray


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
        default="lgbm_deep_xlong,lgbm_deep_xlong_seed2,lgbm_deep_bagged_xlong,lgbm_deep_bagged_xlong_seed2",
        help=(
            "Comma-separated list from: ridge,enet,hgb,lgbm,lgbm_reg,"
            "lgbm_low_lr,lgbm_low_lr_deep,lgbm_deep_long,lgbm_deep_xlong,lgbm_deep_xlong_seed2,lgbm_deep_reg,lgbm_deep_bagged,lgbm_deep_bagged_xlong,lgbm_deep_bagged_xlong_seed2,"
            "lgbm_bagged,lgbm_small,lgbm_subsample,cat,sgd. "
            "Default is the focused v8 LightGBM-only ensemble: the two v7 xlong winners plus one additional seed for each. Elastic Net/XGBoost/CatBoost are omitted because they were slow, weaker, or low-value."
        ),
    )
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--inner-splits", type=int, default=5)
    parser.add_argument("--sample-train-rows", type=int, default=None)
    parser.add_argument("--sample-test-rows", type=int, default=None)
    parser.add_argument("--skip-diagnostics", action="store_true")
    parser.add_argument("--blend-draws", type=int, default=80000, help="Random convex blends to test for OOF blend optimization.")
    parser.add_argument("--no-stacking", action="store_true", help="Disable the OOF Ridge meta-stacker and use only convex blending.")
    parser.add_argument("--lgbm-threads", type=int, default=4, help="Threads for LightGBM/CatBoost where applicable.")
    parser.add_argument("--no-clip", action="store_true", help="Do not clip predictions to [0, 100].")
    parser.add_argument(
        "--postprocess-existing",
        action="store_true",
        help="Fast mode: use existing submission_lgbm.csv/submission_cat.csv and apply algebra constraints without retraining.",
    )
    parser.add_argument(
        "--existing-lgbm-weight",
        type=float,
        default=0.356321586758916,
        help="Weight for submission_lgbm.csv when --postprocess-existing uses both lgbm and cat submissions.",
    )
    parser.add_argument(
        "--algebra-trust",
        type=float,
        default=1.0,
        help="Fast postprocess mode weight on deterministic algebra predictions. Default 1.0.",
    )
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



def build_unsupervised_structure_map(
    df: pd.DataFrame,
    key_col: str,
    prefix: str,
    n_pca: int,
    cluster_sizes: Sequence[int],
) -> pd.DataFrame:
    """
    Create compact PCA and KMeans structure features from covariates only.

    This is an upstream dimensionality-reduction step: it summarizes correlated
    school/district covariates without using PERCENT_PROFICIENT. It is therefore
    safe with respect to target leakage and is aligned with the course notes on
    PCA, clustering, and ensemble feature construction.
    """
    if key_col not in df.columns:
        return pd.DataFrame(columns=[key_col])

    base = df.drop_duplicates(subset=[key_col]).copy().reset_index(drop=True)
    out = base[[key_col]].copy()

    numeric_cols = []
    for c in base.select_dtypes(include=[np.number]).columns:
        if c == key_col:
            continue
        vals = pd.to_numeric(base[c], errors="coerce")
        if vals.notna().sum() >= 5 and vals.nunique(dropna=True) > 1:
            numeric_cols.append(c)

    if len(numeric_cols) < 2 or len(base) < 10:
        return out

    X = base[numeric_cols].replace([np.inf, -np.inf], np.nan)
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(imputer.fit_transform(X))

    usable_pca = max(0, min(n_pca, X_scaled.shape[1], X_scaled.shape[0] - 1))
    if usable_pca >= 1:
        pca = PCA(n_components=usable_pca, random_state=SEED)
        comps = pca.fit_transform(X_scaled)
        for i in range(usable_pca):
            out[f"{prefix}_PCA_{i + 1}"] = comps[:, i]
        # Include compact explained-variance diagnostics as constant columns. They
        # are harmless for trees and make it easy to inspect how much variance the
        # PCA structure features capture later.
        out[f"{prefix}_PCA_EXPLAINED_TOTAL"] = float(np.sum(pca.explained_variance_ratio_))

    for k in cluster_sizes:
        if len(base) >= k and k >= 2:
            km = KMeans(n_clusters=k, random_state=SEED, n_init=10)
            labels = km.fit_predict(X_scaled)
            out[f"{prefix}_CLUSTER_{k}"] = pd.Series(labels, index=out.index).astype(str)

    return out


def add_unsupervised_structure_features(
    train_base: pd.DataFrame,
    test_base: pd.DataFrame,
    school: pd.DataFrame,
    district: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Add PCA/KMeans features for school and district covariate structure.

    These features are intentionally computed before any target encoding and do
    not use outcome values. They are efficient dimensionality-reduction features,
    not a separate complicated model layer.
    """
    school_map = build_unsupervised_structure_map(
        school,
        key_col="SCHOOL",
        prefix="SCHOOL_STRUCT",
        n_pca=10,
        cluster_sizes=(8, 20),
    )
    district_map = build_unsupervised_structure_map(
        district,
        key_col="DISTRICT",
        prefix="DISTRICT_STRUCT",
        n_pca=6,
        cluster_sizes=(5, 10),
    )

    log(
        "Unsupervised structure features: "
        f"school={max(0, school_map.shape[1] - 1)} cols; "
        f"district={max(0, district_map.shape[1] - 1)} cols."
    )

    train_out = train_base.merge(school_map, on="SCHOOL", how="left", validate="many_to_one")
    test_out = test_base.merge(school_map, on="SCHOOL", how="left", validate="many_to_one")

    if "DISTRICT" in train_out.columns and "DISTRICT" in district_map.columns:
        train_out = train_out.merge(district_map, on="DISTRICT", how="left", validate="many_to_one")
        test_out = test_out.merge(district_map, on="DISTRICT", how="left", validate="many_to_one")

    for c in [c for c in train_out.columns if c.endswith("_CLUSTER_8") or c.endswith("_CLUSTER_20") or c.endswith("_CLUSTER_5") or c.endswith("_CLUSTER_10")]:
        train_out[c] = train_out[c].astype("string").fillna("UNKNOWN").astype(str)
        if c in test_out.columns:
            test_out[c] = test_out[c].astype("string").fillna("UNKNOWN").astype(str)

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
    train_base, test_base = add_unsupervised_structure_features(train_base, test_base, school, district)
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


def make_stratification_labels(df: pd.DataFrame, n_splits: int) -> Optional[pd.Series]:
    """Build stable labels for stratified CV using assessment type and subgroup.

    The public/private split is row-level and appears to contain a broad mix of
    assessments and subgroup rows. Stratifying folds by assessment family, grade
    band, and subgroup makes local validation less dependent on a lucky fold mix.
    If a label is too sparse for StratifiedKFold, the function backs off to a
    coarser label and eventually returns None, which triggers regular KFold.
    """
    candidate_sets = [
        ["ASSESSMENT_FAMILY", "GRADE_BAND", "SUBGROUP_NAME"],
        ["SUBJECT", "GRADE_BAND", "SUBGROUP_NAME"],
        ["ASSESSMENT_FAMILY", "SUBGROUP_NAME"],
        ["SUBJECT", "SUBGROUP_NAME"],
        ["SUBJECT"],
    ]
    for cols in candidate_sets:
        if not all(c in df.columns for c in cols):
            continue
        labels = df[cols].astype("string").fillna("UNKNOWN").agg("|".join, axis=1)
        vc = labels.value_counts(dropna=False)
        if len(vc) > 1 and int(vc.min()) >= n_splits:
            return labels.reset_index(drop=True)
    return None


def make_cv_splits(df: pd.DataFrame, n_splits: int, seed: int = SEED) -> List[Tuple[np.ndarray, np.ndarray]]:
    labels = make_stratification_labels(df, n_splits)
    if labels is not None:
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        return list(splitter.split(np.zeros(len(df)), labels))
    splitter = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(splitter.split(df))


def make_oof_target_encodings(train_base_with_y: pd.DataFrame, n_splits: int) -> pd.DataFrame:
    """
    For every training row, creates target encodings from other folds only.
    v9 uses stratified inner folds when possible so each inner validation block
    has a similar assessment/subgroup mix.
    """
    splits = make_cv_splits(train_base_with_y, n_splits, seed=SEED + 13)
    te_cols = ["te_global"] + [name for name, _, _ in GROUP_SPECS]
    out = pd.DataFrame(index=train_base_with_y.index, columns=te_cols, dtype=float)

    for inner_i, (fit_idx, val_idx) in enumerate(splits, start=1):
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


PEER_SUBGROUPS = [
    "All Students",
    "Male",
    "Female",
    "Economically Disadvantaged",
    "Not Economically Disadvantaged",
]


def _safe_series(df: pd.DataFrame, col: str) -> pd.Series:
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce")
    return pd.Series(np.nan, index=df.index, dtype=float)


def add_peer_algebra_features_from_fit(fit_df: pd.DataFrame, dest_df: pd.DataFrame) -> pd.DataFrame:
    """
    Uses only target values available in fit_df to create peer features for dest_df.

    Core idea:
      Within the same SCHOOL + ASSESSMENT_NAME, subgroup rows obey weighted-count
      relationships. For example, All Students is approximately the weighted
      average of Male and Female, and of Economically Disadvantaged and Not
      Economically Disadvantaged.

    v5 improvement:
      The target is an integer percentage, so the most accurate algebra is done
      in inferred proficient-count space:
          proficient_count ~= round(percent * N_STUDENTS / 100)
      This substantially reduced OOF algebra error versus using raw percentages
      directly. The script keeps both raw and count-space algebra features, but
      uses the count-space prediction as ALGEBRA_PRED for downstream constraints.

    Leakage control:
      fit_df excludes the validation fold when creating validation features.
      For test prediction, fit_df is the full training set.
    """
    key = ["SCHOOL", "ASSESSMENT_NAME"]
    dest = dest_df.drop(columns=[TARGET], errors="ignore").copy().reset_index(drop=True)

    pivot = fit_df.pivot_table(
        index=key,
        columns="SUBGROUP_NAME",
        values=[TARGET, "N_STUDENTS"],
        aggfunc="first",
    )
    pivot.columns = [f"peer_{a}__{b}" for a, b in pivot.columns]
    pivot = pivot.reset_index()

    out = dest.merge(pivot, on=key, how="left")
    n = pd.to_numeric(out["N_STUDENTS"], errors="coerce").clip(lower=1).astype(float)
    sg = out["SUBGROUP_NAME"].astype(str)

    all_p = _safe_series(out, "peer_PERCENT_PROFICIENT__All Students")
    all_n = _safe_series(out, "peer_N_STUDENTS__All Students")
    male_p = _safe_series(out, "peer_PERCENT_PROFICIENT__Male")
    male_n = _safe_series(out, "peer_N_STUDENTS__Male")
    female_p = _safe_series(out, "peer_PERCENT_PROFICIENT__Female")
    female_n = _safe_series(out, "peer_N_STUDENTS__Female")
    econ_p = _safe_series(out, "peer_PERCENT_PROFICIENT__Economically Disadvantaged")
    econ_n = _safe_series(out, "peer_N_STUDENTS__Economically Disadvantaged")
    not_econ_p = _safe_series(out, "peer_PERCENT_PROFICIENT__Not Economically Disadvantaged")
    not_econ_n = _safe_series(out, "peer_N_STUDENTS__Not Economically Disadvantaged")

    raw_alg = np.full(len(out), np.nan, dtype=float)
    count_alg = np.full(len(out), np.nan, dtype=float)
    method = np.array(["NONE"] * len(out), dtype=object)

    def inferred_count(pct: pd.Series, count: pd.Series) -> pd.Series:
        return np.rint((pct * count) / 100.0)

    all_k = inferred_count(all_p, all_n)
    male_k = inferred_count(male_p, male_n)
    female_k = inferred_count(female_p, female_n)
    econ_k = inferred_count(econ_p, econ_n)
    not_econ_k = inferred_count(not_econ_p, not_econ_n)

    # Male/Female complement solve. Require count consistency to avoid rare invalid records.
    m = (
        (sg == "Male")
        & all_p.notna() & all_n.notna() & female_p.notna() & female_n.notna()
        & ((all_n - female_n - n).abs() <= 1)
    )
    raw_alg[m] = (all_p[m] * all_n[m] - female_p[m] * female_n[m]) / n[m]
    count_alg[m] = np.rint(100.0 * (all_k[m] - female_k[m]) / n[m])
    method[m] = "MF_SOLVE"

    m = (
        (sg == "Female")
        & all_p.notna() & all_n.notna() & male_p.notna() & male_n.notna()
        & ((all_n - male_n - n).abs() <= 1)
    )
    raw_alg[m] = (all_p[m] * all_n[m] - male_p[m] * male_n[m]) / n[m]
    count_alg[m] = np.rint(100.0 * (all_k[m] - male_k[m]) / n[m])
    method[m] = "MF_SOLVE"

    # Economic-disadvantage complement solve.
    m = (
        (sg == "Economically Disadvantaged")
        & all_p.notna() & all_n.notna() & not_econ_p.notna() & not_econ_n.notna()
        & ((all_n - not_econ_n - n).abs() <= 1)
    )
    raw_alg[m] = (all_p[m] * all_n[m] - not_econ_p[m] * not_econ_n[m]) / n[m]
    count_alg[m] = np.rint(100.0 * (all_k[m] - not_econ_k[m]) / n[m])
    method[m] = "ECON_SOLVE"

    m = (
        (sg == "Not Economically Disadvantaged")
        & all_p.notna() & all_n.notna() & econ_p.notna() & econ_n.notna()
        & ((all_n - econ_n - n).abs() <= 1)
    )
    raw_alg[m] = (all_p[m] * all_n[m] - econ_p[m] * econ_n[m]) / n[m]
    count_alg[m] = np.rint(100.0 * (all_k[m] - econ_k[m]) / n[m])
    method[m] = "ECON_SOLVE"

    # All Students can often be reconstructed from either pair. Average both if both are present.
    mf_all_raw = (male_p * male_n + female_p * female_n) / n
    econ_all_raw = (econ_p * econ_n + not_econ_p * not_econ_n) / n
    mf_all_count = np.rint(100.0 * (male_k + female_k) / n)
    econ_all_count = np.rint(100.0 * (econ_k + not_econ_k) / n)

    mf_ok = male_p.notna() & male_n.notna() & female_p.notna() & female_n.notna() & (((male_n + female_n - n).abs()) <= 1)
    econ_ok = econ_p.notna() & econ_n.notna() & not_econ_p.notna() & not_econ_n.notna() & (((econ_n + not_econ_n - n).abs()) <= 1)

    all_raw_val = np.where(
        mf_ok & econ_ok,
        (mf_all_raw + econ_all_raw) / 2.0,
        np.where(mf_ok, mf_all_raw, np.where(econ_ok, econ_all_raw, np.nan)),
    )
    all_count_val = np.where(
        mf_ok & econ_ok,
        np.rint((mf_all_count + econ_all_count) / 2.0),
        np.where(mf_ok, mf_all_count, np.where(econ_ok, econ_all_count, np.nan)),
    )
    m = (sg == "All Students") & ~np.isnan(all_count_val)
    raw_alg[m] = all_raw_val[m]
    count_alg[m] = all_count_val[m]
    method[m] = "ALL_WEIGHTED"

    raw_alg = np.clip(raw_alg, 0.0, 100.0)
    count_alg = np.clip(count_alg, 0.0, 100.0)

    # Peer features: raw available subgroup scores/counts plus compact summaries.
    peer_score_cols = [f"peer_PERCENT_PROFICIENT__{g}" for g in PEER_SUBGROUPS if f"peer_PERCENT_PROFICIENT__{g}" in out.columns]
    peer_count_cols = [f"peer_N_STUDENTS__{g}" for g in PEER_SUBGROUPS if f"peer_N_STUDENTS__{g}" in out.columns]

    feats = pd.DataFrame(index=out.index)
    feats["ALGEBRA_PRED"] = count_alg
    feats["ALGEBRA_PRED_RAW"] = raw_alg
    feats["ALGEBRA_PRED_COUNT"] = count_alg
    # A small raw/count blend was best for pure algebra rows, so keep it as a model feature.
    feats["ALGEBRA_PRED_RAW25_COUNT75"] = np.where(
        np.isfinite(raw_alg) & np.isfinite(count_alg),
        0.25 * raw_alg + 0.75 * count_alg,
        count_alg,
    )
    feats["ALGEBRA_AVAILABLE"] = (~np.isnan(count_alg)).astype(int)
    feats["ALGEBRA_METHOD"] = method

    if peer_score_cols:
        score_mat = out[peer_score_cols].apply(pd.to_numeric, errors="coerce")
        feats["peer_score_mean"] = score_mat.mean(axis=1)
        feats["peer_score_std"] = score_mat.std(axis=1)
        feats["peer_score_min"] = score_mat.min(axis=1)
        feats["peer_score_max"] = score_mat.max(axis=1)
        feats["peer_score_range"] = feats["peer_score_max"] - feats["peer_score_min"]
        feats["peer_score_count"] = score_mat.notna().sum(axis=1)
        for c in peer_score_cols:
            short = c.replace("peer_PERCENT_PROFICIENT__", "peer_score_")
            feats[short] = score_mat[c]
            feats[f"has_{short}"] = score_mat[c].notna().astype(int)

    if peer_count_cols:
        count_mat = out[peer_count_cols].apply(pd.to_numeric, errors="coerce")
        feats["peer_n_count"] = count_mat.notna().sum(axis=1)
        feats["peer_n_sum"] = count_mat.sum(axis=1)
        feats["peer_n_max"] = count_mat.max(axis=1)
        for c in peer_count_cols:
            short = c.replace("peer_N_STUDENTS__", "peer_n_")
            feats[short] = count_mat[c]

    # Relationship features that help fallback models even when exact algebra is not available.
    feats["diff_peer_all_minus_mean"] = all_p - feats.get("peer_score_mean", np.nan)
    feats["diff_peer_male_minus_female"] = male_p - female_p
    feats["diff_peer_econ_minus_not_econ"] = econ_p - not_econ_p
    feats["ratio_n_to_peer_all"] = n / all_n.replace(0, np.nan)
    feats["delta_n_all_minus_mf"] = all_n - male_n - female_n
    feats["delta_n_all_minus_econ"] = all_n - econ_n - not_econ_n

    return feats


def make_oof_peer_algebra_features(train_base_with_y: pd.DataFrame, n_splits: int) -> pd.DataFrame:
    splits = make_cv_splits(train_base_with_y, n_splits, seed=SEED + 29)
    out = None
    for fit_idx, val_idx in splits:
        fit_df = train_base_with_y.iloc[fit_idx].copy()
        dest_df = train_base_with_y.iloc[val_idx].drop(columns=[TARGET], errors="ignore").copy()
        feats = add_peer_algebra_features_from_fit(fit_df, dest_df)
        if out is None:
            out = pd.DataFrame(index=train_base_with_y.index, columns=feats.columns)
        out.iloc[val_idx, :] = feats.to_numpy()
    # Coerce numeric columns back to numeric, leave ALGEBRA_METHOD as string.
    for c in out.columns:
        if c != "ALGEBRA_METHOD":
            out[c] = pd.to_numeric(out[c], errors="coerce")
        else:
            out[c] = out[c].fillna("NONE").astype(str)
    return out


def add_peer_columns(base_features: pd.DataFrame, peer_df: pd.DataFrame) -> pd.DataFrame:
    out = base_features.copy()
    for c in peer_df.columns:
        out[c] = peer_df[c].to_numpy()
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
    full_train_peer = make_oof_peer_algebra_features(train_with_y, n_splits=inner_splits)
    full_fit_df = train_with_y.copy()
    test_encoded = add_target_encodings_from_fit(full_fit_df, test_base)
    test_peer = add_peer_algebra_features_from_fit(full_fit_df, test_base)

    X_train_full = add_peer_columns(add_te_columns(train_base, full_train_te), full_train_peer).reset_index(drop=True)
    X_test_full = add_peer_columns(test_encoded.drop(columns=[TARGET], errors="ignore"), test_peer).reset_index(drop=True)

    cat_cols, num_cols = choose_feature_columns(X_train_full)
    X_train_full = subset_features(X_train_full, cat_cols, num_cols)
    X_test_full = subset_features(X_test_full, cat_cols, num_cols)

    log(f"Selected {len(num_cols):,} numeric features and {len(cat_cols):,} categorical features.")

    outer_splits = make_cv_splits(train_base, n_splits, seed=SEED + 101)
    folds = []
    outer_alg_pred = np.full(len(train_base), np.nan, dtype=float)
    outer_alg_method = np.array(["NONE"] * len(train_base), dtype=object)
    for fold_num, (tr_idx, val_idx) in enumerate(outer_splits, start=1):
        log(f"Preparing fold {fold_num}/{n_splits} features...")
        tr_base = train_base.iloc[tr_idx].reset_index(drop=True)
        val_base = train_base.iloc[val_idx].reset_index(drop=True)
        y_tr = tr_base[TARGET].astype(float).reset_index(drop=True)
        y_val = val_base[TARGET].astype(float).reset_index(drop=True)

        tr_te = make_oof_target_encodings(tr_base, n_splits=min(inner_splits, max(2, len(tr_base) // 5000)))
        tr_peer = make_oof_peer_algebra_features(tr_base, n_splits=min(inner_splits, max(2, len(tr_base) // 5000)))
        val_encoded = add_target_encodings_from_fit(tr_base, val_base.drop(columns=[TARGET], errors="ignore"))
        val_peer = add_peer_algebra_features_from_fit(tr_base, val_base.drop(columns=[TARGET], errors="ignore"))

        outer_alg_pred[val_idx] = pd.to_numeric(val_peer["ALGEBRA_PRED"], errors="coerce").to_numpy(dtype=float)
        outer_alg_method[val_idx] = val_peer["ALGEBRA_METHOD"].astype(str).to_numpy()

        X_tr = subset_features(add_peer_columns(add_te_columns(tr_base, tr_te), tr_peer), cat_cols, num_cols)
        X_val = subset_features(add_peer_columns(val_encoded.drop(columns=[TARGET], errors="ignore"), val_peer), cat_cols, num_cols)

        folds.append((tr_idx, val_idx, X_tr, y_tr, X_val, y_val))

    return FeatureSet(
        train_full=X_train_full,
        test_full=X_test_full,
        folds=folds,
        cat_cols=cat_cols,
        num_cols=num_cols,
        outer_alg_pred=outer_alg_pred,
        outer_alg_method=outer_alg_method,
        test_alg_pred=pd.to_numeric(test_peer["ALGEBRA_PRED"], errors="coerce").to_numpy(dtype=float),
        test_alg_method=test_peer["ALGEBRA_METHOD"].astype(str).to_numpy(),
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


def _lgbm_feature_set(features: FeatureSet) -> Tuple[List[str], List[str]]:
    """
    LightGBM gets lower-cardinality categorical fields plus all numeric features.
    SCHOOL and DISTRICT are represented through leakage-safe target encodings,
    counts, peer features, and district/school covariates; passing them directly
    as raw categories can be slower and more likely to overfit.
    """
    lgb_cat_cols = [c for c in features.cat_cols if c not in ["SCHOOL", "DISTRICT"]]
    lgb_cols = lgb_cat_cols + features.num_cols
    return lgb_cat_cols, lgb_cols


def run_lgbm_variant(
    features: FeatureSet,
    y: pd.Series,
    test_ids: pd.Series,
    clip: bool,
    out_dir: Path,
    model_name: str,
    fold_params: Dict,
    final_params: Optional[Dict] = None,
    final_mult: float = 1.10,
    final_cap: int = 6000,
    seed_offset: int = 0,
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    try:
        import lightgbm as lgb
    except Exception as e:
        log(f"Skipping {model_name} because LightGBM is not installed/importable: {e}")
        raise RuntimeError("LightGBM unavailable")

    log(f"Running {model_name}...")
    oof = np.zeros(len(y), dtype=float)
    fold_scores = []
    best_iters = []
    importances = []

    lgb_cat_cols, lgb_cols = _lgbm_feature_set(features)
    X_full_lgb, X_test_lgb = prep_category_dtypes(features.train_full[lgb_cols], features.test_full[lgb_cols], lgb_cat_cols)

    for fold_num, (tr_idx, val_idx, X_tr_raw, y_tr, X_val_raw, y_val) in enumerate(features.folds, start=1):
        X_tr, X_val = prep_category_dtypes(X_tr_raw[lgb_cols], X_val_raw[lgb_cols], lgb_cat_cols)

        params = dict(
            objective="regression_l2",
            metric="l2",
            random_state=SEED + seed_offset + 101 * fold_num,
            n_jobs=4,
            verbosity=-1,
        )
        params.update(fold_params)
        model = lgb.LGBMRegressor(**params)
        callbacks = [
            lgb.early_stopping(stopping_rounds=150, verbose=False),
            lgb.log_evaluation(period=300),
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
        train_pred = clip_pred(model.predict(X_tr, num_iteration=model.best_iteration_), clip)
        oof[val_idx] = pred
        best_iter = int(model.best_iteration_ or model.n_estimators)
        best_iters.append(best_iter)
        train_mse = mse(y_tr, train_pred)
        val_mse = mse(y_val, pred)
        n_est = int(params.get("n_estimators", model.n_estimators))
        hit_cap = bool(best_iter >= 0.98 * n_est)
        fold_scores.append({
            "model": model_name,
            "fold": fold_num,
            "train_mse": train_mse,
            "mse": val_mse,
            "rmse": math.sqrt(val_mse),
            "overfit_ratio_val_to_train": float(val_mse / train_mse) if train_mse > 0 else np.nan,
            "best_iteration": best_iter,
            "n_estimators_requested": n_est,
            "hit_iteration_cap": hit_cap,
        })
        cap_msg = "; hit cap" if hit_cap else ""
        log(f"  {model_name} fold {fold_num}: MSE={fold_scores[-1]['mse']:.4f}, RMSE={fold_scores[-1]['rmse']:.4f}, best_iter={best_iter}{cap_msg}")

        importances.append(pd.DataFrame({
            "feature": X_tr.columns,
            f"importance_fold_{fold_num}": model.feature_importances_,
        }))

    final_n_estimators = int(np.clip(np.mean(best_iters) * final_mult, 300, final_cap)) if best_iters else int(fold_params.get("n_estimators", 1200))
    params_final = dict(
        objective="regression_l2",
        metric="l2",
        random_state=SEED + seed_offset,
        n_jobs=4,
        verbosity=-1,
    )
    params_final.update(fold_params)
    if final_params:
        params_final.update(final_params)
    params_final["n_estimators"] = final_n_estimators

    log(f"Training final {model_name} with n_estimators={final_n_estimators}...")
    final_model = lgb.LGBMRegressor(**params_final)
    final_model.fit(X_full_lgb, y, categorical_feature=lgb_cat_cols)
    test_pred = clip_pred(final_model.predict(X_test_lgb), clip)

    fi = pd.DataFrame({"feature": X_full_lgb.columns, "importance_final": final_model.feature_importances_})
    for imp in importances:
        fi = fi.merge(imp, on="feature", how="left")
    importance_cols = [c for c in fi.columns if c.startswith("importance")]
    fi["importance_mean"] = fi[importance_cols].mean(axis=1)
    fi = fi.sort_values("importance_mean", ascending=False)
    safe_name = re.sub(r"[^A-Za-z0-9_]+", "_", model_name)
    fi.to_csv(out_dir / "diagnostics" / f"feature_importance_{safe_name}.csv", index=False)

    return oof, test_pred, {"fold_scores": fold_scores, "feature_importance": fi}


def run_lgbm(features: FeatureSet, y: pd.Series, test_ids: pd.Series, clip: bool, out_dir: Path) -> Tuple[np.ndarray, np.ndarray, Dict]:
    return run_lgbm_variant(
        features, y, test_ids, clip, out_dir,
        model_name="lgbm",
        fold_params=dict(
            n_estimators=2400,
            learning_rate=0.035,
            num_leaves=31,
            max_depth=-1,
            min_child_samples=25,
            subsample=0.88,
            subsample_freq=1,
            colsample_bytree=0.88,
            reg_alpha=0.05,
            reg_lambda=2.0,
        ),
        final_params=None,
        final_mult=1.08,
        final_cap=3200,
    )


def run_lgbm_reg(features: FeatureSet, y: pd.Series, test_ids: pd.Series, clip: bool, out_dir: Path) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """More conservative LightGBM: stronger L2, more leaf restrictions, more subsampling."""
    return run_lgbm_variant(
        features, y, test_ids, clip, out_dir,
        model_name="lgbm_reg",
        fold_params=dict(
            n_estimators=3000,
            learning_rate=0.028,
            num_leaves=23,
            max_depth=8,
            min_child_samples=55,
            subsample=0.78,
            subsample_freq=1,
            colsample_bytree=0.72,
            reg_alpha=0.15,
            reg_lambda=8.0,
            min_split_gain=0.005,
        ),
        final_mult=1.08,
        final_cap=3600,
    )


def run_lgbm_low_lr(features: FeatureSet, y: pd.Series, test_ids: pd.Series, clip: bool, out_dir: Path) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """Lower learning-rate LightGBM: slower but often more accurate and smoother."""
    return run_lgbm_variant(
        features, y, test_ids, clip, out_dir,
        model_name="lgbm_low_lr",
        fold_params=dict(
            n_estimators=4800,
            learning_rate=0.018,
            num_leaves=31,
            max_depth=-1,
            min_child_samples=35,
            subsample=0.86,
            subsample_freq=1,
            colsample_bytree=0.82,
            reg_alpha=0.08,
            reg_lambda=4.0,
        ),
        final_mult=1.05,
        final_cap=5200,
    )


def run_lgbm_low_lr_deep(features: FeatureSet, y: pd.Series, test_ids: pd.Series, clip: bool, out_dir: Path) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """Slightly deeper low-learning-rate LightGBM for controlled diversity.

    This is intentionally close to the proven low_lr model, but allows more leaves
    and a bit more regularization. If it does not help, OOF blend optimization will
    give it little or zero weight.
    """
    return run_lgbm_variant(
        features, y, test_ids, clip, out_dir,
        model_name="lgbm_low_lr_deep",
        fold_params=dict(
            n_estimators=6200,
            learning_rate=0.015,
            num_leaves=47,
            max_depth=-1,
            min_child_samples=32,
            subsample=0.84,
            subsample_freq=1,
            colsample_bytree=0.78,
            reg_alpha=0.10,
            reg_lambda=5.5,
            min_split_gain=0.002,
        ),
        final_mult=1.03,
        final_cap=7000,
    )




def run_lgbm_deep_long(features: FeatureSet, y: pd.Series, test_ids: pd.Series, clip: bool, out_dir: Path) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """v6 primary model: longer version of the v5-dominant deep low-learning-rate LightGBM.

    In v5, lgbm_low_lr_deep received about 96% of the raw blend weight and hit the
    iteration cap in every fold, meaning validation error was still improving near
    the end of training. This variant raises the cap while keeping regularization
    and shrinkage controlled.
    """
    return run_lgbm_variant(
        features, y, test_ids, clip, out_dir,
        model_name="lgbm_deep_long",
        fold_params=dict(
            n_estimators=9000,
            learning_rate=0.012,
            num_leaves=47,
            max_depth=-1,
            min_child_samples=32,
            subsample=0.84,
            subsample_freq=1,
            colsample_bytree=0.78,
            reg_alpha=0.10,
            reg_lambda=5.5,
            min_split_gain=0.002,
        ),
        final_mult=1.015,
        final_cap=9800,
    )




def run_lgbm_deep_xlong(features: FeatureSet, y: pd.Series, test_ids: pd.Series, clip: bool, out_dir: Path) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """v7 primary candidate: extra-long low-learning-rate LightGBM.

    v6 improved public score but still hit the iteration cap in every fold, while
    validation loss continued to decrease. This model keeps the same basic shape
    as the v6 winner, lowers the learning rate slightly, and extends the tree cap.
    It is intentionally included with the proven v6 model rather than replacing it.
    The OOF blend will decide whether the additional trees help enough to use.
    """
    return run_lgbm_variant(
        features, y, test_ids, clip, out_dir,
        model_name="lgbm_deep_xlong",
        fold_params=dict(
            n_estimators=12500,
            learning_rate=0.0098,
            num_leaves=47,
            max_depth=-1,
            min_child_samples=32,
            subsample=0.84,
            subsample_freq=1,
            colsample_bytree=0.78,
            reg_alpha=0.11,
            reg_lambda=5.8,
            min_split_gain=0.002,
        ),
        final_mult=1.012,
        final_cap=13500,
    )

def run_lgbm_deep_xlong_seed2(features: FeatureSet, y: pd.Series, test_ids: pd.Series, clip: bool, out_dir: Path) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """v8 seed-diversity counterpart to lgbm_deep_xlong.

    Same hyperparameters as lgbm_deep_xlong, different seed. This adds controlled
    diversity without changing the model family or adding noisy weak learners.
    """
    return run_lgbm_variant(
        features, y, test_ids, clip, out_dir,
        model_name="lgbm_deep_xlong_seed2",
        fold_params=dict(
            n_estimators=12500,
            learning_rate=0.0098,
            num_leaves=47,
            max_depth=-1,
            min_child_samples=32,
            subsample=0.84,
            subsample_freq=1,
            colsample_bytree=0.78,
            reg_alpha=0.11,
            reg_lambda=5.8,
            min_split_gain=0.002,
        ),
        final_mult=1.012,
        final_cap=13500,
        seed_offset=7000,
    )


def run_lgbm_deep_reg(features: FeatureSet, y: pd.Series, test_ids: pd.Series, clip: bool, out_dir: Path) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """v6 stability model: similar capacity, stronger regularization.

    This model is not expected to beat the primary model alone. It is included to
    provide a slightly more conservative prediction path for blending/private-LB
    stability if OOF validation supports it.
    """
    return run_lgbm_variant(
        features, y, test_ids, clip, out_dir,
        model_name="lgbm_deep_reg",
        fold_params=dict(
            n_estimators=8500,
            learning_rate=0.0125,
            num_leaves=39,
            max_depth=-1,
            min_child_samples=44,
            subsample=0.80,
            subsample_freq=1,
            colsample_bytree=0.72,
            reg_alpha=0.18,
            reg_lambda=8.5,
            min_split_gain=0.004,
        ),
        final_mult=1.015,
        final_cap=9300,
    )


def run_lgbm_deep_bagged(features: FeatureSet, y: pd.Series, test_ids: pd.Series, clip: bool, out_dir: Path) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """v6 diversity model: stochastic deep LightGBM with heavier row/feature subsampling."""
    return run_lgbm_variant(
        features, y, test_ids, clip, out_dir,
        model_name="lgbm_deep_bagged",
        fold_params=dict(
            n_estimators=8500,
            learning_rate=0.013,
            num_leaves=47,
            max_depth=-1,
            min_child_samples=40,
            subsample=0.68,
            subsample_freq=1,
            colsample_bytree=0.62,
            reg_alpha=0.12,
            reg_lambda=7.0,
            min_split_gain=0.003,
        ),
        final_mult=1.015,
        final_cap=9300,
    )



def run_lgbm_deep_bagged_xlong(features: FeatureSet, y: pd.Series, test_ids: pd.Series, clip: bool, out_dir: Path) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """v7 diversity candidate: longer stochastic deep LightGBM.

    This is the longer counterpart to the v6 bagged model, which earned almost
    half of the v6 blend weight. It uses the same row/feature subsampling idea,
    with a slightly lower learning rate and a higher tree cap.
    """
    return run_lgbm_variant(
        features, y, test_ids, clip, out_dir,
        model_name="lgbm_deep_bagged_xlong",
        fold_params=dict(
            n_estimators=11500,
            learning_rate=0.0108,
            num_leaves=47,
            max_depth=-1,
            min_child_samples=40,
            subsample=0.68,
            subsample_freq=1,
            colsample_bytree=0.62,
            reg_alpha=0.13,
            reg_lambda=7.5,
            min_split_gain=0.003,
        ),
        final_mult=1.012,
        final_cap=12500,
    )

def run_lgbm_deep_bagged_xlong_seed2(features: FeatureSet, y: pd.Series, test_ids: pd.Series, clip: bool, out_dir: Path) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """v8 seed-diversity counterpart to lgbm_deep_bagged_xlong.

    Same stochastic xlong bagged model shape as the v7 contributor, different seed.
    If it does not add independent signal, OOF blend optimization will assign little
    or zero weight.
    """
    return run_lgbm_variant(
        features, y, test_ids, clip, out_dir,
        model_name="lgbm_deep_bagged_xlong_seed2",
        fold_params=dict(
            n_estimators=11500,
            learning_rate=0.0108,
            num_leaves=47,
            max_depth=-1,
            min_child_samples=40,
            subsample=0.68,
            subsample_freq=1,
            colsample_bytree=0.62,
            reg_alpha=0.13,
            reg_lambda=7.5,
            min_split_gain=0.003,
        ),
        final_mult=1.012,
        final_cap=12500,
        seed_offset=9000,
    )


def run_lgbm_bagged(features: FeatureSet, y: pd.Series, test_ids: pd.Series, clip: bool, out_dir: Path) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """More stochastic LightGBM for ensemble diversity and private-leaderboard stability."""
    return run_lgbm_variant(
        features, y, test_ids, clip, out_dir,
        model_name="lgbm_bagged",
        fold_params=dict(
            n_estimators=5200,
            learning_rate=0.017,
            num_leaves=31,
            max_depth=10,
            min_child_samples=45,
            subsample=0.70,
            subsample_freq=1,
            colsample_bytree=0.65,
            reg_alpha=0.12,
            reg_lambda=7.0,
            min_split_gain=0.004,
        ),
        final_mult=1.05,
        final_cap=6200,
    )


def run_lgbm_small(features: FeatureSet, y: pd.Series, test_ids: pd.Series, clip: bool, out_dir: Path) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """Small-leaf LightGBM: intentionally biased/regularized for ensemble diversity."""
    return run_lgbm_variant(
        features, y, test_ids, clip, out_dir,
        model_name="lgbm_small",
        fold_params=dict(
            n_estimators=3600,
            learning_rate=0.024,
            num_leaves=15,
            max_depth=6,
            min_child_samples=75,
            subsample=0.82,
            subsample_freq=1,
            colsample_bytree=0.78,
            reg_alpha=0.10,
            reg_lambda=10.0,
            min_split_gain=0.01,
        ),
        final_mult=1.08,
        final_cap=4200,
    )


def run_lgbm_subsample(features: FeatureSet, y: pd.Series, test_ids: pd.Series, clip: bool, out_dir: Path) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """Diversity model: more aggressive row/column subsampling with moderate tree size."""
    return run_lgbm_variant(
        features, y, test_ids, clip, out_dir,
        model_name="lgbm_subsample",
        fold_params=dict(
            n_estimators=3200,
            learning_rate=0.026,
            num_leaves=31,
            max_depth=9,
            min_child_samples=40,
            subsample=0.68,
            subsample_freq=1,
            colsample_bytree=0.62,
            reg_alpha=0.12,
            reg_lambda=6.0,
        ),
        final_mult=1.08,
        final_cap=3800,
    )



def run_sgd(features: FeatureSet, y: pd.Series, test_ids: pd.Series, clip: bool, out_dir: Path) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Fast linear diversity model using stochastic gradient descent.

    This is the efficient replacement for slow ElasticNetCV-style experimentation.
    It is not expected to beat LightGBM, but OOF blending will keep it only if it
    contributes useful calibration/diversity.
    """
    log("Running SGD linear diversity model...")
    oof = np.zeros(len(y), dtype=float)
    fold_scores = []

    cat_cols = [c for c in features.cat_cols if c not in ["SCHOOL", "DISTRICT", "ASSESSMENT_NAME"]]
    num_cols = features.num_cols
    cols = cat_cols + num_cols

    def make_model() -> Pipeline:
        pre = ColumnTransformer(
            transformers=[
                ("num", Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler(with_mean=False)),
                ]), num_cols),
                ("cat", one_hot_encoder(), cat_cols),
            ],
            remainder="drop",
            sparse_threshold=0.3,
        )
        return Pipeline([
            ("preprocess", pre),
            ("model", SGDRegressor(
                loss="squared_error",
                penalty="elasticnet",
                alpha=2e-5,
                l1_ratio=0.05,
                max_iter=2500,
                tol=1e-4,
                early_stopping=True,
                validation_fraction=0.1,
                n_iter_no_change=10,
                average=True,
                random_state=SEED,
            )),
        ])

    for fold_num, (tr_idx, val_idx, X_tr_raw, y_tr, X_val_raw, y_val) in enumerate(features.folds, start=1):
        X_tr = X_tr_raw[cols].copy()
        X_val = X_val_raw[cols].copy()
        model = make_model()
        model.fit(X_tr, y_tr)
        pred = clip_pred(model.predict(X_val), clip)
        oof[val_idx] = pred
        fold_scores.append({"model": "sgd", "fold": fold_num, "mse": mse(y_val, pred), "rmse": rmse(y_val, pred), "best_iteration": np.nan})
        log(f"  SGD fold {fold_num}: MSE={fold_scores[-1]['mse']:.4f}, RMSE={fold_scores[-1]['rmse']:.4f}")

    X_full = features.train_full[cols].copy()
    X_test = features.test_full[cols].copy()
    final_model = make_model()
    final_model.fit(X_full, y)
    test_pred = clip_pred(final_model.predict(X_test), clip)
    return oof, test_pred, {"fold_scores": fold_scores, "feature_importance": None}


def run_elasticnet(features: FeatureSet, y: pd.Series, test_ids: pd.Series, clip: bool, out_dir: Path) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Simple regularized linear diversity model.
    This is not expected to beat LightGBM by itself, but it can sometimes improve
    a blend by correcting broad calibration differences with lower variance.
    """
    log("Running Elastic Net model...")
    oof = np.zeros(len(y), dtype=float)
    fold_scores = []

    cat_cols = [c for c in features.cat_cols if c not in ["SCHOOL", "DISTRICT"]]
    num_cols = features.num_cols
    cols = cat_cols + num_cols

    def make_model(alpha: float = 0.0008, l1_ratio: float = 0.08) -> Pipeline:
        pre = ColumnTransformer(
            transformers=[
                ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler(with_mean=False))]), num_cols),
                ("cat", one_hot_encoder(), cat_cols),
            ],
            remainder="drop",
            sparse_threshold=0.3,
        )
        return Pipeline([
            ("preprocess", pre),
            ("model", ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=12000, random_state=SEED, selection="random")),
        ])

    for fold_num, (tr_idx, val_idx, X_tr_raw, y_tr, X_val_raw, y_val) in enumerate(features.folds, start=1):
        X_tr = X_tr_raw[cols].copy()
        X_val = X_val_raw[cols].copy()
        for c in cat_cols:
            X_tr[c] = X_tr[c].astype("string").fillna("UNKNOWN")
            X_val[c] = X_val[c].astype("string").fillna("UNKNOWN")
        model = make_model()
        model.fit(X_tr, y_tr)
        pred = clip_pred(model.predict(X_val), clip)
        oof[val_idx] = pred
        fold_scores.append({"model": "enet", "fold": fold_num, "mse": mse(y_val, pred), "rmse": rmse(y_val, pred), "best_iteration": np.nan})
        log(f"  Elastic Net fold {fold_num}: MSE={fold_scores[-1]['mse']:.4f}, RMSE={fold_scores[-1]['rmse']:.4f}")

    X_full = features.train_full[cols].copy()
    X_test = features.test_full[cols].copy()
    for c in cat_cols:
        X_full[c] = X_full[c].astype("string").fillna("UNKNOWN")
        X_test[c] = X_test[c].astype("string").fillna("UNKNOWN")
    final_model = make_model()
    final_model.fit(X_full, y)
    test_pred = clip_pred(final_model.predict(X_test), clip)
    return oof, test_pred, {"fold_scores": fold_scores, "feature_importance": None}

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


def _stack_feature_frame(preds: Dict[str, np.ndarray]) -> pd.DataFrame:
    names = list(preds.keys())
    arr = np.column_stack([preds[n] for n in names])
    df = pd.DataFrame(arr, columns=[f"pred__{n}" for n in names])
    if len(names) >= 2:
        df["pred_mean"] = arr.mean(axis=1)
        df["pred_std"] = arr.std(axis=1)
        df["pred_min"] = arr.min(axis=1)
        df["pred_max"] = arr.max(axis=1)
        df["pred_range"] = df["pred_max"] - df["pred_min"]
    return df


def fit_oof_stacker(
    oof_preds: Dict[str, np.ndarray],
    test_preds: Dict[str, np.ndarray],
    y: np.ndarray,
    stratify_df: pd.DataFrame,
    n_splits: int,
    clip: bool,
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """Second-level stacker trained only on OOF base-model predictions.

    This is a proper stacking diagnostic rather than training the meta-learner
    and evaluating it on the same rows. We create a second set of meta-folds over
    the OOF prediction matrix, fit RidgeCV on meta-training rows, and predict
    meta-validation rows. The final test meta-prediction is produced by fitting
    the same RidgeCV pipeline on all OOF prediction rows and applying it to the
    base-model test predictions.
    """
    names = list(oof_preds.keys())
    if len(names) < 2:
        only = names[0]
        return oof_preds[only].copy(), test_preds[only].copy(), {
            "used": False,
            "reason": "only_one_base_model",
            "alpha_by_fold": [],
            "fold_scores": [],
            "feature_names": [f"pred__{only}"],
        }

    Z_oof = _stack_feature_frame(oof_preds)
    Z_test = _stack_feature_frame(test_preds)
    # Match columns defensively in case a caller passes ordered dicts differently.
    Z_test = Z_test[Z_oof.columns]

    splits = make_cv_splits(stratify_df.reset_index(drop=True), min(n_splits, len(Z_oof)), seed=SEED + 501)
    alphas = np.array([0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0])
    meta_oof = np.zeros(len(y), dtype=float)
    fold_rows = []
    alpha_rows = []

    for fold_num, (tr_idx, val_idx) in enumerate(splits, start=1):
        model = Pipeline([
            ("scaler", StandardScaler()),
            ("ridge", RidgeCV(alphas=alphas)),
        ])
        model.fit(Z_oof.iloc[tr_idx], y[tr_idx])
        pred = model.predict(Z_oof.iloc[val_idx])
        pred = clip_pred(pred, clip)
        meta_oof[val_idx] = pred
        fold_mse = mse(y[val_idx], pred)
        ridge = model.named_steps["ridge"]
        fold_rows.append({
            "model": "stack_ridge_meta",
            "fold": fold_num,
            "mse": fold_mse,
            "rmse": math.sqrt(fold_mse),
            "alpha": float(ridge.alpha_),
        })
        alpha_rows.append(float(ridge.alpha_))

    final_model = Pipeline([
        ("scaler", StandardScaler()),
        ("ridge", RidgeCV(alphas=alphas)),
    ])
    final_model.fit(Z_oof, y)
    meta_test = clip_pred(final_model.predict(Z_test), clip)
    final_ridge = final_model.named_steps["ridge"]
    coef = final_ridge.coef_.ravel()
    info = {
        "used": True,
        "alpha_by_fold": alpha_rows,
        "final_alpha": float(final_ridge.alpha_),
        "fold_scores": fold_rows,
        "feature_names": list(Z_oof.columns),
        "coefficients": {name: float(val) for name, val in zip(Z_oof.columns, coef)},
        "intercept": float(final_ridge.intercept_),
    }
    return meta_oof, meta_test, info


def optimize_constraint_weights(base_oof: np.ndarray, alg_pred: np.ndarray, alg_method: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    """
    Learns how aggressively to trust deterministic algebra predictions.
    Usually the optimal weight is near 1.0, but learning the weight from OOF
    data protects against rounding and rare count inconsistencies.
    """
    weights = {}
    methods = sorted([m for m in pd.Series(alg_method).dropna().unique() if m != "NONE"])
    grid = np.concatenate([np.linspace(0.0, 0.95, 20), np.linspace(0.96, 1.0, 21)])
    for method in methods:
        mask = (alg_method == method) & np.isfinite(alg_pred)
        if mask.sum() < 50:
            continue
        best_w = 1.0
        best_score = np.mean((y[mask] - alg_pred[mask]) ** 2)
        for w in grid:
            p = w * alg_pred[mask] + (1.0 - w) * base_oof[mask]
            score = np.mean((y[mask] - p) ** 2)
            if score < best_score:
                best_score = score
                best_w = float(w)
        weights[method] = best_w
    return weights


def apply_constraint_weights(base_pred: np.ndarray, alg_pred: np.ndarray, alg_method: np.ndarray, weights: Dict[str, float], clip: bool) -> np.ndarray:
    out = np.asarray(base_pred, dtype=float).copy()
    for method, w in weights.items():
        mask = (alg_method == method) & np.isfinite(alg_pred)
        out[mask] = w * alg_pred[mask] + (1.0 - w) * out[mask]
    return clip_pred(out, clip)


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

    score_df = pd.DataFrame(scores)
    score_df.to_csv(diag_dir / "cv_fold_scores.csv", index=False)

    if not score_df.empty and "train_mse" in score_df.columns:
        overfit_rows = []
        for model, g in score_df.groupby("model", dropna=False):
            row = {
                "model": model,
                "folds": int(len(g)),
                "mean_train_mse": float(g["train_mse"].mean()),
                "mean_valid_mse": float(g["mse"].mean()),
                "std_valid_mse": float(g["mse"].std(ddof=0)),
                "mean_valid_rmse": float(np.sqrt(g["mse"].mean())),
                "mean_overfit_ratio_val_to_train": float(g["overfit_ratio_val_to_train"].mean()),
                "hit_iteration_cap_rate": float(g.get("hit_iteration_cap", pd.Series(False, index=g.index)).astype(bool).mean()),
            }
            if "best_iteration" in g.columns:
                row["mean_best_iteration"] = float(pd.to_numeric(g["best_iteration"], errors="coerce").mean())
            overfit_rows.append(row)
        pd.DataFrame(overfit_rows).sort_values("mean_valid_mse").to_csv(diag_dir / "overfitting_summary.csv", index=False)

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
        "script_version": "v5",
        "seed": SEED,
        "models_in_blend": list(weights.keys()),
        "blend_weights": weights,
    }
    with open(diag_dir / "run_metadata.json", "w", encoding="utf-8") as f:
        json.dump(env, f, indent=2)


def run_postprocess_existing(train: pd.DataFrame, test: pd.DataFrame, sub_dir: Path, diag_dir: Path, args: argparse.Namespace) -> None:
    """
    Fast mode for the situation where expensive model training already produced
    submission_lgbm.csv and/or submission_cat.csv. It blends those existing
    predictions and replaces algebra-solvable test rows with deterministic
    subgroup-identity predictions.
    """
    pred_sources = {}
    for name in ["lgbm", "cat", "ridge"]:
        path = sub_dir / f"submission_{name}.csv"
        if path.exists():
            df = pd.read_csv(path)
            if ID_COL not in df.columns or TARGET not in df.columns:
                raise ValueError(f"{path} must contain {ID_COL} and {TARGET}.")
            pred_sources[name] = df[[ID_COL, TARGET]].copy()

    if not pred_sources:
        raise FileNotFoundError(
            "No existing model submissions found. Expected at least one of "
            "outputs/submissions/submission_lgbm.csv or submission_cat.csv."
        )

    base = pd.DataFrame({ID_COL: test[ID_COL].values})
    for name, df in pred_sources.items():
        base = base.merge(df.rename(columns={TARGET: f"pred_{name}"}), on=ID_COL, how="left", validate="one_to_one")

    pred_cols = [c for c in base.columns if c.startswith("pred_")]
    missing_counts = {c: int(base[c].isna().sum()) for c in pred_cols}
    bad = {c: n for c, n in missing_counts.items() if n > 0}
    if bad:
        raise ValueError(
            "Existing submission file(s) do not cover the current test rows. "
            f"Missing predictions: {bad}. Run without sampling or regenerate the model submissions."
        )

    if "lgbm" in pred_sources and "cat" in pred_sources:
        wl = float(args.existing_lgbm_weight)
        wc = 1.0 - wl
        base_pred = wl * base["pred_lgbm"].to_numpy(dtype=float) + wc * base["pred_cat"].to_numpy(dtype=float)
        blend_used = {"lgbm": wl, "cat": wc}
    else:
        first = sorted(pred_sources.keys())[0]
        base_pred = base[f"pred_{first}"].to_numpy(dtype=float)
        blend_used = {first: 1.0}

    peer = add_peer_algebra_features_from_fit(train, test)
    alg_pred = pd.to_numeric(peer["ALGEBRA_PRED"], errors="coerce").to_numpy(dtype=float)
    alg_method = peer["ALGEBRA_METHOD"].astype(str).to_numpy()
    methods = sorted([m for m in pd.Series(alg_method).unique() if m != "NONE"])
    constraint_weights = {m: float(args.algebra_trust) for m in methods}
    final_pred = apply_constraint_weights(base_pred, alg_pred, alg_method, constraint_weights, clip=not args.no_clip)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = sub_dir / f"submission_constraint_from_existing_{timestamp}.csv"
    save_submission(test[ID_COL], final_pred, out_path)
    try:
        save_submission(test[ID_COL], final_pred, sub_dir / "submission_constraint_from_existing.csv")
    except PermissionError:
        log("Could not overwrite submission_constraint_from_existing.csv; timestamped version was saved successfully.")

    diag_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        ID_COL: test[ID_COL].values,
        "base_pred": base_pred,
        "algebra_pred": alg_pred,
        "algebra_method": alg_method,
        "final_pred": final_pred,
    }).to_csv(diag_dir / "test_constraint_from_existing_predictions.csv", index=False)
    with open(diag_dir / "constraint_from_existing_summary.json", "w", encoding="utf-8") as f:
        json.dump({
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "blend_used": blend_used,
            "algebra_trust": float(args.algebra_trust),
            "algebra_coverage_test": float(np.isfinite(alg_pred).mean()),
            "method_counts": pd.Series(alg_method).value_counts().to_dict(),
            "primary_submission": str(out_path),
        }, f, indent=2)

    log(f"Existing prediction blend used: {blend_used}")
    log(f"Algebra coverage on test: {np.isfinite(alg_pred).mean():.3%}")
    log(f"Primary Kaggle file: {out_path}")


def main() -> None:
    args = parse_args()
    requested_models = [m.strip().lower() for m in args.models.split(",") if m.strip()]
    invalid = sorted(set(requested_models) - {"ridge", "enet", "hgb", "lgbm", "lgbm_reg", "lgbm_low_lr", "lgbm_low_lr_deep", "lgbm_deep_long", "lgbm_deep_xlong", "lgbm_deep_xlong_seed2", "lgbm_deep_reg", "lgbm_deep_bagged", "lgbm_deep_bagged_xlong", "lgbm_deep_bagged_xlong_seed2", "lgbm_bagged", "lgbm_small", "lgbm_subsample", "cat", "sgd"})
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

    if args.postprocess_existing:
        log("Fast mode: post-processing existing submissions with algebra constraints...")
        run_postprocess_existing(train=train, test=test, sub_dir=sub_dir, diag_dir=diag_dir, args=args)
        return

    log("Preparing joined base tables, engineered features, and unsupervised structure features...")
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
        "enet": run_elasticnet,
        "sgd": run_sgd,
        "hgb": run_hgb,
        "lgbm": run_lgbm,
        "lgbm_reg": run_lgbm_reg,
        "lgbm_low_lr": run_lgbm_low_lr,
        "lgbm_low_lr_deep": run_lgbm_low_lr_deep,
        "lgbm_deep_long": run_lgbm_deep_long,
        "lgbm_deep_xlong": run_lgbm_deep_xlong,
        "lgbm_deep_xlong_seed2": run_lgbm_deep_xlong_seed2,
        "lgbm_deep_reg": run_lgbm_deep_reg,
        "lgbm_deep_bagged": run_lgbm_deep_bagged,
        "lgbm_deep_bagged_xlong": run_lgbm_deep_bagged_xlong,
        "lgbm_deep_bagged_xlong_seed2": run_lgbm_deep_bagged_xlong_seed2,
        "lgbm_bagged": run_lgbm_bagged,
        "lgbm_small": run_lgbm_small,
        "lgbm_subsample": run_lgbm_subsample,
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

    weights = optimize_blend_weights(oof_preds, y.to_numpy(), n_draws=args.blend_draws)
    log(f"Optimized blend weights: {weights}")

    blend_oof = np.zeros(len(y), dtype=float)
    blend_test = np.zeros(len(test_ids), dtype=float)
    for name, weight in weights.items():
        blend_oof += weight * oof_preds[name]
        blend_test += weight * test_preds[name]
    blend_oof = clip_pred(blend_oof, do_clip)
    blend_test = clip_pred(blend_test, do_clip)

    log(f"Blend overall OOF: MSE={mse(y, blend_oof):.4f}, RMSE={rmse(y, blend_oof):.4f}")

    stack_oof = None
    stack_test = None
    stack_info = {"used": False, "reason": "disabled"}
    if not args.no_stacking and len(oof_preds) >= 2:
        stack_oof, stack_test, stack_info = fit_oof_stacker(
            oof_preds=oof_preds,
            test_preds=test_preds,
            y=y.to_numpy(),
            stratify_df=train_base,
            n_splits=n_splits,
            clip=do_clip,
        )
        all_scores.extend(stack_info.get("fold_scores", []))
        log(f"Stacked meta-model OOF: MSE={mse(y, stack_oof):.4f}, RMSE={rmse(y, stack_oof):.4f}")

    raw_candidates: Dict[str, Tuple[np.ndarray, np.ndarray]] = {"blend": (blend_oof, blend_test)}
    if stack_oof is not None and stack_test is not None:
        raw_candidates["stack"] = (stack_oof, stack_test)

    raw_scores = {name: mse(y, pred_pair[0]) for name, pred_pair in raw_candidates.items()}
    chosen_raw_name = min(raw_scores, key=raw_scores.get)
    chosen_oof, chosen_test = raw_candidates[chosen_raw_name]
    log(f"Chosen raw base for constraints: {chosen_raw_name} (MSE={raw_scores[chosen_raw_name]:.4f})")

    constraint_weights = optimize_constraint_weights(
        base_oof=chosen_oof,
        alg_pred=features.outer_alg_pred,
        alg_method=features.outer_alg_method,
        y=y.to_numpy(),
    )
    constraint_oof = apply_constraint_weights(
        base_pred=chosen_oof,
        alg_pred=features.outer_alg_pred,
        alg_method=features.outer_alg_method,
        weights=constraint_weights,
        clip=do_clip,
    )
    constraint_test = apply_constraint_weights(
        base_pred=chosen_test,
        alg_pred=features.test_alg_pred,
        alg_method=features.test_alg_method,
        weights=constraint_weights,
        clip=do_clip,
    )

    alg_cov_train = float(np.isfinite(features.outer_alg_pred).mean())
    alg_cov_test = float(np.isfinite(features.test_alg_pred).mean())
    log(f"Algebra coverage: OOF train={alg_cov_train:.3%}; test={alg_cov_test:.3%}")
    log(f"Constraint weights by method: {constraint_weights}")
    log(f"Constraint {chosen_raw_name} OOF: MSE={mse(y, constraint_oof):.4f}, RMSE={rmse(y, constraint_oof):.4f}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Save timestamped files first so a locked generic CSV cannot destroy the run.
    save_submission(test_ids, blend_test, sub_dir / f"submission_raw_blend_{timestamp}.csv")
    if stack_test is not None:
        save_submission(test_ids, stack_test, sub_dir / f"submission_raw_stack_{timestamp}.csv")
    save_submission(test_ids, chosen_test, sub_dir / f"submission_raw_chosen_{chosen_raw_name}_{timestamp}.csv")
    save_submission(test_ids, constraint_test, sub_dir / f"submission_constraint_{chosen_raw_name}_{timestamp}.csv")
    save_submission(test_ids, constraint_test, sub_dir / f"submission_constraint_blend_{timestamp}.csv")
    # Also try writing friendly generic names. If Windows/OneDrive locks them, continue safely.
    generic_outputs = [("submission_blend.csv", blend_test), ("submission_constraint_blend.csv", constraint_test)]
    if stack_test is not None:
        generic_outputs.append(("submission_stack.csv", stack_test))
    for fname, pred in generic_outputs:
        try:
            save_submission(test_ids, pred, sub_dir / fname)
        except PermissionError:
            log(f"Could not overwrite {fname}; timestamped version was saved successfully.")

    if not args.skip_diagnostics:
        log("Saving diagnostics...")
        oof_for_diag = dict(oof_preds)
        oof_for_diag["raw_blend"] = blend_oof
        if stack_oof is not None:
            oof_for_diag["raw_stack"] = stack_oof
        oof_for_diag[f"raw_chosen_{chosen_raw_name}"] = chosen_oof
        oof_for_diag["constraint_blend"] = constraint_oof
        save_diagnostics(
            train_base=train_base,
            y=y,
            oof_preds=oof_for_diag,
            blend_pred=constraint_oof,
            scores=all_scores,
            weights=weights,
            out_dir=out_dir,
        )
        with open(diag_dir / "constraint_weights.json", "w", encoding="utf-8") as f:
            json.dump({
                "constraint_weights": constraint_weights,
                "algebra_coverage_train_oof": alg_cov_train,
                "algebra_coverage_test": alg_cov_test,
                "raw_blend_mse": mse(y, blend_oof),
                "raw_stack_mse": mse(y, stack_oof) if stack_oof is not None else None,
                "chosen_raw_base": chosen_raw_name,
                "chosen_raw_mse": mse(y, chosen_oof),
                "constraint_blend_mse": mse(y, constraint_oof),
            }, f, indent=2)
        with open(diag_dir / "stacking_diagnostics.json", "w", encoding="utf-8") as f:
            json.dump(stack_info, f, indent=2)
        constraint_train_df = pd.DataFrame({
            ID_COL: train_base[ID_COL].values,
            "actual": y.values,
            "raw_blend": blend_oof,
            "raw_stack": stack_oof if stack_oof is not None else np.nan,
            "raw_chosen": chosen_oof,
            "chosen_raw_base": chosen_raw_name,
            "algebra_pred": features.outer_alg_pred,
            "algebra_method": features.outer_alg_method,
            "constraint_blend": constraint_oof,
        })
        constraint_train_df["raw_sq_error"] = (constraint_train_df["actual"] - constraint_train_df["raw_blend"]) ** 2
        constraint_train_df["constraint_sq_error"] = (constraint_train_df["actual"] - constraint_train_df["constraint_blend"]) ** 2
        constraint_train_df.to_csv(diag_dir / "oof_constraint_predictions.csv", index=False)

        method_diag = (
            constraint_train_df.groupby("algebra_method", dropna=False)
            .agg(
                n_rows=("actual", "size"),
                raw_mse=("raw_sq_error", "mean"),
                constraint_mse=("constraint_sq_error", "mean"),
                mean_actual=("actual", "mean"),
                mean_raw_pred=("raw_blend", "mean"),
                mean_constraint_pred=("constraint_blend", "mean"),
            )
            .reset_index()
        )
        method_diag["mse_improvement"] = method_diag["raw_mse"] - method_diag["constraint_mse"]
        method_diag.to_csv(diag_dir / "constraint_method_diagnostics.csv", index=False)
        pd.DataFrame({
            ID_COL: test_ids.values,
            "raw_blend": blend_test,
            "raw_stack": stack_test if stack_test is not None else np.nan,
            "raw_chosen": chosen_test,
            "chosen_raw_base": chosen_raw_name,
            "algebra_pred": features.test_alg_pred,
            "algebra_method": features.test_alg_method,
            "constraint_blend": constraint_test,
        }).to_csv(diag_dir / "test_constraint_predictions.csv", index=False)

    log("Done.")
    log(f"Primary Kaggle file: {sub_dir / f'submission_constraint_blend_{timestamp}.csv'}")
    log(f"Chosen raw base: {chosen_raw_name}")
    log(f"Individual submissions are in: {sub_dir}")
    log(f"Diagnostics are in: {diag_dir}")


if __name__ == "__main__":
    main()
