# ml_models.py - Model definitions, ensemble/stacking, class imbalance handling
# FIXES APPLIED:
#   [#8/dead_code_02] Removed early_stopping_rounds (eval_set was never passed)
#   [#11/dead_code_05] Removed unused ImbPipeline import

import numpy as np
import logging
from typing import Any, List, Tuple

from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor,
    StackingClassifier, StackingRegressor,
    VotingClassifier, VotingRegressor
)
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)

# ─────────── Optional Libraries ───────────
try:
    from xgboost import XGBClassifier, XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBClassifier = XGBRegressor = None
    XGBOOST_AVAILABLE = False

try:
    from lightgbm import LGBMClassifier, LGBMRegressor
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LGBMClassifier = LGBMRegressor = None
    LIGHTGBM_AVAILABLE = False

try:
    from catboost import CatBoostClassifier, CatBoostRegressor
    CATBOOST_AVAILABLE = True
except ImportError:
    CatBoostClassifier = CatBoostRegressor = None
    CATBOOST_AVAILABLE = False

try:
    from imblearn.over_sampling import SMOTE
    IMBLEARN_AVAILABLE = True
except ImportError:
    SMOTE = None
    IMBLEARN_AVAILABLE = False


def get_library_status():
    """Return availability of optional libraries."""
    return {
        "xgboost": XGBOOST_AVAILABLE,
        "lightgbm": LIGHTGBM_AVAILABLE,
        "catboost": CATBOOST_AVAILABLE,
        "imblearn": IMBLEARN_AVAILABLE,
    }


# ─────────── Base Models ───────────

def get_base_models(task_type: str = "classification", random_state: int = 42,
                    use_class_weight: bool = False) -> List[Tuple[str, Any]]:
    """Get list of base models."""
    models = []
    cw = "balanced" if use_class_weight else None
    
    if task_type == "classification":
        models.extend([
            ("LogisticRegression", LogisticRegression(
                max_iter=2000, random_state=random_state, class_weight=cw)),
            ("DecisionTree", DecisionTreeClassifier(
                max_depth=10, random_state=random_state, class_weight=cw)),
            ("RandomForest", RandomForestClassifier(
                n_estimators=100, max_depth=15, random_state=random_state,
                n_jobs=-1, class_weight=cw)),
            ("GradientBoosting", GradientBoostingClassifier(
                n_estimators=100, random_state=random_state)),
            ("KNN", KNeighborsClassifier(n_neighbors=5, n_jobs=-1)),
            ("NaiveBayes", GaussianNB()),
            ("SVM", SVC(probability=True, random_state=random_state,
                        max_iter=1000, class_weight=cw)),
        ])
        
        # [FIX #8] Removed early_stopping_rounds — it has no effect without eval_set
        if XGBOOST_AVAILABLE:
            models.append(("XGBoost", XGBClassifier(
                n_estimators=100, learning_rate=0.1, max_depth=6,
                random_state=random_state, eval_metric="logloss",
                n_jobs=-1)))
        if LIGHTGBM_AVAILABLE:
            models.append(("LightGBM", LGBMClassifier(
                n_estimators=100, learning_rate=0.1, max_depth=6,
                random_state=random_state, verbose=-1, n_jobs=-1,
                class_weight=cw)))
        if CATBOOST_AVAILABLE:
            models.append(("CatBoost", CatBoostClassifier(
                iterations=100, learning_rate=0.1, depth=6,
                random_state=random_state, verbose=False,
                auto_class_weights="Balanced" if use_class_weight else None)))
    
    elif task_type == "regression":
        models.extend([
            ("LinearRegression", LinearRegression(n_jobs=-1)),
            ("Ridge", Ridge(alpha=1.0, random_state=random_state)),
            ("Lasso", Lasso(alpha=1.0, random_state=random_state, max_iter=2000)),
            ("ElasticNet", ElasticNet(alpha=1.0, l1_ratio=0.5, random_state=random_state, max_iter=2000)),
            ("DecisionTree", DecisionTreeRegressor(max_depth=10, random_state=random_state)),
            ("RandomForest", RandomForestRegressor(
                n_estimators=100, max_depth=15, random_state=random_state, n_jobs=-1)),
            ("GradientBoosting", GradientBoostingRegressor(
                n_estimators=100, random_state=random_state)),
            ("KNN", KNeighborsRegressor(n_neighbors=5, n_jobs=-1)),
            ("SVR", SVR(max_iter=1000)),
        ])
        # [FIX #8] Removed early_stopping_rounds
        if XGBOOST_AVAILABLE:
            models.append(("XGBoost", XGBRegressor(
                n_estimators=100, learning_rate=0.1, max_depth=6,
                random_state=random_state, n_jobs=-1)))
        if LIGHTGBM_AVAILABLE:
            models.append(("LightGBM", LGBMRegressor(
                n_estimators=100, learning_rate=0.1, max_depth=6,
                random_state=random_state, verbose=-1, n_jobs=-1)))
        if CATBOOST_AVAILABLE:
            models.append(("CatBoost", CatBoostRegressor(
                iterations=100, learning_rate=0.1, depth=6,
                random_state=random_state, verbose=False)))
    
    return models


# ─────────── Ensemble / Stacking ───────────

def build_stacking_ensemble(top_models: List[Tuple[str, Any]], task_type: str = "classification",
                            random_state: int = 42) -> Tuple[str, Any]:
    """Build a stacking ensemble from the top N trained models."""
    if len(top_models) < 2:
        logger.warning("Need at least 2 models for stacking")
        return None
    
    estimators = [(name, model) for name, model in top_models[:3]]
    
    if task_type == "classification":
        meta_learner = LogisticRegression(max_iter=1000, random_state=random_state)
        ensemble = StackingClassifier(
            estimators=estimators, final_estimator=meta_learner,
            cv=3, n_jobs=-1, passthrough=False)
    else:
        meta_learner = Ridge(alpha=1.0, random_state=random_state)
        ensemble = StackingRegressor(
            estimators=estimators, final_estimator=meta_learner,
            cv=3, n_jobs=-1, passthrough=False)
    
    logger.info(f"Built stacking ensemble with {len(estimators)} base models")
    return ("StackingEnsemble", ensemble)


def build_voting_ensemble(top_models: List[Tuple[str, Any]], task_type: str = "classification") -> Tuple[str, Any]:
    """Build a voting ensemble from top models."""
    if len(top_models) < 2:
        return None
    estimators = [(name, model) for name, model in top_models[:3]]
    if task_type == "classification":
        return ("VotingEnsemble", VotingClassifier(estimators=estimators, voting='soft', n_jobs=-1))
    return ("VotingEnsemble", VotingRegressor(estimators=estimators, n_jobs=-1))


# ─────────── SMOTE Handling ───────────

def apply_smote(X_train, y_train, random_state=42):
    """Apply SMOTE oversampling for imbalanced datasets."""
    if not IMBLEARN_AVAILABLE:
        logger.warning("imblearn not available, skipping SMOTE")
        return X_train, y_train
    
    try:
        unique, counts = np.unique(y_train, return_counts=True)
        min_samples = counts.min()
        if min_samples < 2:
            logger.warning("Too few samples in minority class for SMOTE")
            return X_train, y_train
        
        k = min(5, min_samples - 1)
        smote = SMOTE(random_state=random_state, k_neighbors=k)
        X_res, y_res = smote.fit_resample(X_train, y_train)
        logger.info(f"SMOTE: {len(X_train)} -> {len(X_res)} samples")
        return X_res, y_res
    except Exception as e:
        logger.error(f"SMOTE failed: {e}")
        return X_train, y_train

# [2026-02-15T10:00:00] Add classification and regression model registry

# [2026-03-01T10:00:00] Add LightGBM and XGBoost to model registry

# [2026-02-15T10:00:00] Add classification and regression model registry

# [2026-03-01T10:00:00] Add LightGBM and XGBoost to model registry

# [2026-04-03T11:00:00] Implement ensemble stacking for top-N models
