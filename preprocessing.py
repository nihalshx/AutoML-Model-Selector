# preprocessing.py - Data preprocessing, feature engineering, feature selection
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import logging

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_classif, f_regression, mutual_info_classif, mutual_info_regression

logger = logging.getLogger(__name__)


# ─────────── Auto Detection ───────────

def auto_detect_task_type(df: pd.DataFrame, target_col: str) -> str:
    """Automatically infer classification vs regression from target column."""
    y = df[target_col]
    if y.dtype in ['object', 'category', 'bool']:
        return "classification"
    nunique = y.nunique()
    ratio = nunique / len(y)
    if nunique <= 20 or (nunique <= 50 and ratio < 0.05):
        return "classification"
    return "regression"


def auto_detect_target_column(df: pd.DataFrame) -> Optional[str]:
    """Auto-detect likely target column names."""
    common_targets = ['target', 'label', 'class', 'y', 'output', 'result',
                      'price', 'salary', 'survived', 'diagnosis', 'category',
                      'species', 'grade', 'rating', 'status', 'outcome']
    cols_lower = {c.lower().strip(): c for c in df.columns}
    for name in common_targets:
        if name in cols_lower:
            return cols_lower[name]
    # Check last column as common convention
    return df.columns[-1]


# ─────────── Validation ───────────

def validate_dataframe(df: pd.DataFrame, target_col: str) -> Optional[str]:
    """Validate uploaded dataframe. Returns error message if invalid."""
    if df.empty:
        return "Dataset is empty."
    if df.shape[0] < 10:
        return "Dataset must have at least 10 rows."
    if target_col not in df.columns:
        return f"Target column '{target_col}' not found. Available: {', '.join(df.columns)}"
    if df[target_col].isna().all():
        return "Target column contains only missing values."
    missing_pct = df[target_col].isna().sum() / len(df) * 100
    if missing_pct > 50:
        return f"Target column has {missing_pct:.1f}% missing values."
    return None


def validate_data_quality(df: pd.DataFrame, target_col: str) -> List[str]:
    """Extended data quality warnings."""
    warnings = []
    duplicates = df.duplicated().sum()
    if duplicates > 0:
        warnings.append(f"Found {duplicates} duplicate rows ({duplicates/len(df)*100:.1f}%)")
    
    for col in df.select_dtypes(include=['object']).columns:
        if col != target_col and df[col].nunique() > 50:
            warnings.append(f"Column '{col}' has {df[col].nunique()} unique values (high cardinality)")
    
    for col in df.columns:
        if col != target_col:
            miss = df[col].isna().sum() / len(df) * 100
            if miss > 80:
                warnings.append(f"Column '{col}' has {miss:.1f}% missing values")
    
    # Imbalance check
    if df[target_col].dtype in ['object', 'category'] or df[target_col].nunique() < 20:
        vc = df[target_col].value_counts()
        if len(vc) > 1:
            ratio = vc.min() / vc.max()
            if ratio < 0.1:
                warnings.append(f"Target is highly imbalanced (ratio: {ratio:.3f})")
    
    constant = [c for c in df.columns if c != target_col and df[c].nunique() == 1]
    if constant:
        warnings.append(f"Constant columns: {', '.join(constant)}")
    
    return warnings


# ─────────── Feature Engineering ───────────

def auto_feature_engineering(df: pd.DataFrame, target_col: str, max_interactions: int = 10) -> pd.DataFrame:
    """Create engineered features: date parsing, interactions, transformations."""
    logger.info("Starting feature engineering...")
    df_eng = df.copy()
    
    # Date features from datetime columns
    for col in df.select_dtypes(include=['datetime64']).columns:
        if col != target_col:
            df_eng[f'{col}_year'] = df[col].dt.year
            df_eng[f'{col}_month'] = df[col].dt.month
            df_eng[f'{col}_dayofweek'] = df[col].dt.dayofweek
            df_eng[f'{col}_quarter'] = df[col].dt.quarter
            df_eng[f'{col}_is_weekend'] = df[col].dt.dayofweek.isin([5, 6]).astype(int)
    
    # Try parsing string dates
    for col in df.select_dtypes(include=['object']).columns:
        if col != target_col and df[col].nunique() < len(df) * 0.5:
            try:
                parsed = pd.to_datetime(df[col], errors='coerce')
                if parsed.notna().sum() / len(df) > 0.5:
                    df_eng[f'{col}_year'] = parsed.dt.year
                    df_eng[f'{col}_month'] = parsed.dt.month
                    df_eng[f'{col}_dayofweek'] = parsed.dt.dayofweek
                    logger.info(f"Parsed dates in '{col}'")
            except Exception:
                pass
    
    # Interaction features
    num_cols = [c for c in df.select_dtypes(include=['int64', 'float64']).columns if c != target_col]
    if 2 <= len(num_cols) <= 10:
        created = 0
        for i, c1 in enumerate(num_cols[:5]):
            for c2 in num_cols[i+1:i+3]:
                if created < max_interactions:
                    df_eng[f'{c1}_x_{c2}'] = df[c1] * df[c2]
                    created += 1
        logger.info(f"Created {created} interaction features")
    
    # Polynomial features
    for col in num_cols[:5]:
        if df[col].std() > 0:
            df_eng[f'{col}_squared'] = df[col] ** 2
            df_eng[f'{col}_sqrt'] = np.sqrt(np.abs(df[col]))
    
    logger.info(f"Feature engineering: {len(df.columns)} -> {len(df_eng.columns)} columns")
    return df_eng


# ─────────── Feature Type Detection ───────────

def detect_feature_types(df: pd.DataFrame, target_col: str) -> Tuple[List[str], List[str]]:
    """Detect numeric and categorical features."""
    feature_cols = [c for c in df.columns if c != target_col]
    num_cols = df[feature_cols].select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_cols = df[feature_cols].select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    
    for col in feature_cols:
        if col not in num_cols and col not in cat_cols:
            if df[col].nunique() < 10 and df[col].nunique() / len(df) < 0.05:
                cat_cols.append(col)
            else:
                num_cols.append(col)
    
    logger.info(f"Detected {len(num_cols)} numeric, {len(cat_cols)} categorical features")
    return num_cols, cat_cols


# ─────────── Preprocessor Pipeline ───────────

def build_preprocessor(num_cols: List[str], cat_cols: List[str]) -> ColumnTransformer:
    """Build sklearn preprocessing pipeline."""
    transformers = []
    if num_cols:
        numeric_transformer = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ])
        transformers.append(("num", numeric_transformer, num_cols))
    if cat_cols:
        categorical_transformer = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False, max_categories=50))
        ])
        transformers.append(("cat", categorical_transformer, cat_cols))
    if not transformers:
        raise ValueError("No features to process")
    return ColumnTransformer(transformers=transformers, remainder="drop")


# ─────────── Feature Selection ───────────

def apply_feature_selection(X_train, y_train, X_test, task_type: str = "classification",
                            method: str = "mutual_info", k: int = 20) -> Tuple:
    """Apply feature selection and return transformed data + selected feature indices."""
    if task_type == "classification":
        score_func = mutual_info_classif if method == "mutual_info" else f_classif
    else:
        score_func = mutual_info_regression if method == "mutual_info" else f_regression
    
    k = min(k, X_train.shape[1])
    selector = SelectKBest(score_func=score_func, k=k)
    X_train_sel = selector.fit_transform(X_train, y_train)
    X_test_sel = selector.transform(X_test)
    
    selected_mask = selector.get_support()
    scores = selector.scores_
    
    logger.info(f"Feature selection: {X_train.shape[1]} -> {X_train_sel.shape[1]} features")
    return X_train_sel, X_test_sel, selector, selected_mask, scores

# [2026-02-11T16:00:00] Add preprocessing pipeline with missing value handling

# [2026-02-24T10:30:00] Fix preprocessing bug with categorical encoding

# [2026-03-03T13:45:00] Implement SMOTE for imbalanced dataset handling
