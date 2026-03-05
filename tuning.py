# tuning.py - Expanded Optuna hyperparameter tuning
# FIX #4: Replaced FakeTrial hack with proper _rebuild_from_params()

import numpy as np
import logging
from typing import Any, Dict

from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

logger = logging.getLogger(__name__)

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    OPTUNA_AVAILABLE = True
except ImportError:
    optuna = None
    OPTUNA_AVAILABLE = False

from ml_models import (
    XGBOOST_AVAILABLE, LIGHTGBM_AVAILABLE, CATBOOST_AVAILABLE,
    XGBClassifier, XGBRegressor, LGBMClassifier, LGBMRegressor,
    CatBoostClassifier, CatBoostRegressor
)
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor
)
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

# Models that support Optuna tuning
TUNABLE_MODELS = [
    "RandomForest", "XGBoost", "LightGBM", "CatBoost",
    "GradientBoosting", "SVM", "SVR", "KNN"
]


def _suggest_params(trial, model_name: str, task_type: str, random_state: int = 42):
    """Suggest hyperparameters for a given model during Optuna search."""
    
    if model_name == "RandomForest":
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 300),
            "max_depth": trial.suggest_int("max_depth", 5, 30),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
        }
        if task_type == "classification":
            return RandomForestClassifier(**params, random_state=random_state, n_jobs=-1)
        return RandomForestRegressor(**params, random_state=random_state, n_jobs=-1)
    
    elif model_name == "GradientBoosting":
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 300),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        }
        if task_type == "classification":
            return GradientBoostingClassifier(**params, random_state=random_state)
        return GradientBoostingRegressor(**params, random_state=random_state)
    
    elif model_name in ("SVM", "SVR"):
        kernel = trial.suggest_categorical("kernel", ["rbf", "poly", "linear"])
        params = {"C": trial.suggest_float("C", 0.01, 100, log=True), "kernel": kernel, "max_iter": 2000}
        if kernel == "rbf":
            params["gamma"] = trial.suggest_float("gamma", 1e-4, 1.0, log=True)
        if task_type == "classification":
            return SVC(**params, probability=True, random_state=random_state)
        return SVR(**params)
    
    elif model_name == "KNN":
        params = {
            "n_neighbors": trial.suggest_int("n_neighbors", 3, 25),
            "weights": trial.suggest_categorical("weights", ["uniform", "distance"]),
            "p": trial.suggest_int("p", 1, 2),
        }
        if task_type == "classification":
            return KNeighborsClassifier(**params, n_jobs=-1)
        return KNeighborsRegressor(**params, n_jobs=-1)
    
    elif model_name == "XGBoost" and XGBOOST_AVAILABLE:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 300),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        }
        if task_type == "classification":
            return XGBClassifier(**params, random_state=random_state, eval_metric="logloss", n_jobs=-1)
        return XGBRegressor(**params, random_state=random_state, n_jobs=-1)
    
    elif model_name == "LightGBM" and LIGHTGBM_AVAILABLE:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 300),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 20, 150),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        }
        if task_type == "classification":
            return LGBMClassifier(**params, random_state=random_state, verbose=-1, n_jobs=-1)
        return LGBMRegressor(**params, random_state=random_state, verbose=-1, n_jobs=-1)
    
    elif model_name == "CatBoost" and CATBOOST_AVAILABLE:
        params = {
            "iterations": trial.suggest_int("iterations", 50, 300),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "depth": trial.suggest_int("depth", 4, 10),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-8, 10.0, log=True),
        }
        if task_type == "classification":
            return CatBoostClassifier(**params, random_state=random_state, verbose=False)
        return CatBoostRegressor(**params, random_state=random_state, verbose=False)
    
    return None


def _rebuild_from_params(best_params: Dict, model_name: str, task_type: str, random_state: int = 42):
    """
    [FIX #4] Properly rebuild a model from Optuna's best_params dict.
    Replaces the fragile FakeTrial hack that could silently produce misconfigured models.
    """
    rs = random_state
    
    if model_name == "RandomForest":
        p = {k: best_params[k] for k in ["n_estimators", "max_depth", "min_samples_split", "min_samples_leaf"]}
        return RandomForestClassifier(**p, random_state=rs, n_jobs=-1) if task_type == "classification" \
            else RandomForestRegressor(**p, random_state=rs, n_jobs=-1)
    
    elif model_name == "GradientBoosting":
        p = {k: best_params[k] for k in ["n_estimators", "learning_rate", "max_depth", "min_samples_split", "subsample"]}
        return GradientBoostingClassifier(**p, random_state=rs) if task_type == "classification" \
            else GradientBoostingRegressor(**p, random_state=rs)
    
    elif model_name in ("SVM", "SVR"):
        p = {"C": best_params["C"], "kernel": best_params["kernel"], "max_iter": 2000}
        if best_params["kernel"] == "rbf" and "gamma" in best_params:
            p["gamma"] = best_params["gamma"]
        return SVC(**p, probability=True, random_state=rs) if task_type == "classification" else SVR(**p)
    
    elif model_name == "KNN":
        p = {k: best_params[k] for k in ["n_neighbors", "weights", "p"]}
        return KNeighborsClassifier(**p, n_jobs=-1) if task_type == "classification" \
            else KNeighborsRegressor(**p, n_jobs=-1)
    
    elif model_name == "XGBoost" and XGBOOST_AVAILABLE:
        p = {k: best_params[k] for k in ["n_estimators", "learning_rate", "max_depth", "subsample",
                                           "colsample_bytree", "reg_alpha", "reg_lambda"]}
        return XGBClassifier(**p, random_state=rs, eval_metric="logloss", n_jobs=-1) if task_type == "classification" \
            else XGBRegressor(**p, random_state=rs, n_jobs=-1)
    
    elif model_name == "LightGBM" and LIGHTGBM_AVAILABLE:
        p = {k: best_params[k] for k in ["n_estimators", "learning_rate", "num_leaves",
                                           "min_child_samples", "subsample", "colsample_bytree"]}
        return LGBMClassifier(**p, random_state=rs, verbose=-1, n_jobs=-1) if task_type == "classification" \
            else LGBMRegressor(**p, random_state=rs, verbose=-1, n_jobs=-1)
    
    elif model_name == "CatBoost" and CATBOOST_AVAILABLE:
        p = {k: best_params[k] for k in ["iterations", "learning_rate", "depth", "l2_leaf_reg"]}
        return CatBoostClassifier(**p, random_state=rs, verbose=False) if task_type == "classification" \
            else CatBoostRegressor(**p, random_state=rs, verbose=False)
    
    return None


def tune_with_optuna(preprocessor: ColumnTransformer, X, y,
                     model_name: str, base_model: Any,
                     task_type: str = "classification",
                     n_trials: int = 20, cv: int = 3,
                     random_state: int = 42) -> Any:
    """Tune model hyperparameters using Optuna."""
    if not OPTUNA_AVAILABLE:
        logger.warning("Optuna not installed, skipping tuning")
        return base_model
    
    if model_name not in TUNABLE_MODELS:
        return base_model
    
    logger.info(f"Tuning {model_name} with Optuna ({n_trials} trials)")
    
    def objective(trial):
        try:
            clf = _suggest_params(trial, model_name, task_type, random_state)
            if clf is None:
                return -float("inf")
            pipe = Pipeline(steps=[("preprocessor", preprocessor), ("model", clf)])
            scoring = "f1_weighted" if task_type == "classification" else "neg_mean_squared_error"
            scores = cross_val_score(pipe, X, y, cv=cv, scoring=scoring, n_jobs=-1)
            return float(np.mean(scores))
        except Exception as e:
            logger.error(f"Optuna trial error: {e}")
            return -float("inf")
    
    study = optuna.create_study(direction="maximize", study_name=f"{model_name}_tuning")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    
    best_params = study.best_params
    logger.info(f"Best params for {model_name}: {best_params}")
    
    # [FIX #4] Rebuild model properly instead of using FakeTrial hack
    best_model = _rebuild_from_params(best_params, model_name, task_type, random_state)
    
    return best_model if best_model is not None else base_model

# [2026-02-17T13:15:00] Implement hyperparameter tuning with GridSearchCV

# [2026-03-05T11:30:00] Add Bayesian optimization for hyperparameter tuning

# [2026-02-17T13:15:00] Implement hyperparameter tuning with GridSearchCV

# [2026-03-05T11:30:00] Add Bayesian optimization for hyperparameter tuning
