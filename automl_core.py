# automl_core.py - Enhanced with feature engineering and improvements
import os
import uuid
import warnings
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import logging

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_classif, f_regression, mutual_info_classif, mutual_info_regression
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report, confusion_matrix, roc_curve, roc_auc_score,
    mean_absolute_error, mean_squared_error, r2_score, precision_score, recall_score
)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

# Optional libraries
try:
    from xgboost import XGBClassifier, XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBClassifier = XGBRegressor = None
    XGBOOST_AVAILABLE = False
    logger.warning("XGBoost not available")

try:
    from lightgbm import LGBMClassifier, LGBMRegressor
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LGBMClassifier = LGBMRegressor = None
    LIGHTGBM_AVAILABLE = False
    logger.warning("LightGBM not available")

try:
    from catboost import CatBoostClassifier, CatBoostRegressor
    CATBOOST_AVAILABLE = True
except ImportError:
    CatBoostClassifier = CatBoostRegressor = None
    CATBOOST_AVAILABLE = False
    logger.warning("CatBoost not available")

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    OPTUNA_AVAILABLE = True
except ImportError:
    optuna = None
    OPTUNA_AVAILABLE = False
    logger.warning("Optuna not available")

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    shap = None
    SHAP_AVAILABLE = False
    logger.warning("SHAP not available")

sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 100


def auto_feature_engineering(df: pd.DataFrame, target_col: str, max_features: int = 10) -> pd.DataFrame:
    """
    Automatically create engineered features.
    """
    logger.info("Starting automated feature engineering...")
    df_eng = df.copy()
    
    # Date features
    date_cols = df.select_dtypes(include=['datetime64']).columns
    for col in date_cols:
        if col != target_col:
            df_eng[f'{col}_year'] = df[col].dt.year
            df_eng[f'{col}_month'] = df[col].dt.month
            df_eng[f'{col}_dayofweek'] = df[col].dt.dayofweek
            df_eng[f'{col}_quarter'] = df[col].dt.quarter
            df_eng[f'{col}_is_weekend'] = df[col].dt.dayofweek.isin([5, 6]).astype(int)
            logger.info(f"Created date features from '{col}'")
    
    # Convert string dates to datetime
    for col in df.select_dtypes(include=['object']).columns:
        if col != target_col and df[col].nunique() < len(df) * 0.5:
            try:
                df_temp = pd.to_datetime(df[col], errors='coerce')
                if df_temp.notna().sum() / len(df) > 0.5:  # If >50% are valid dates
                    df_eng[f'{col}_year'] = df_temp.dt.year
                    df_eng[f'{col}_month'] = df_temp.dt.month
                    df_eng[f'{col}_dayofweek'] = df_temp.dt.dayofweek
                    logger.info(f"Detected and parsed dates in '{col}'")
            except Exception:
                pass
    
    # Interaction features (numeric only, limited)
    num_cols = df.select_dtypes(include=['int64', 'float64']).columns
    num_cols = [c for c in num_cols if c != target_col]
    
    if len(num_cols) >= 2 and len(num_cols) <= 10:
        # Create top interactions
        interactions_created = 0
        for i, col1 in enumerate(num_cols[:5]):  # Limit to first 5 columns
            for col2 in num_cols[i+1:i+3]:  # Only 2 interactions per column
                if interactions_created < max_features:
                    df_eng[f'{col1}_x_{col2}'] = df[col1] * df[col2]
                    interactions_created += 1
        
        logger.info(f"Created {interactions_created} interaction features")
    
    # Aggregation features (if applicable)
    for col in num_cols[:5]:  # Limit to first 5 numeric columns
        if df[col].std() > 0:  # Only if there's variance
            df_eng[f'{col}_squared'] = df[col] ** 2
            df_eng[f'{col}_sqrt'] = np.sqrt(np.abs(df[col]))
    
    logger.info(f"Feature engineering complete. Original: {len(df.columns)}, New: {len(df_eng.columns)}")
    
    return df_eng


def detect_feature_types(df: pd.DataFrame, target_col: str) -> Tuple[List[str], List[str]]:
    """Detect numeric and categorical features."""
    feature_cols = [c for c in df.columns if c != target_col]
    
    num_cols = df[feature_cols].select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_cols = df[feature_cols].select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    
    # Handle columns that might be categorical but stored as numeric
    for col in feature_cols:
        if col not in num_cols and col not in cat_cols:
            if df[col].nunique() < 10 and df[col].nunique() / len(df) < 0.05:
                cat_cols.append(col)
            else:
                num_cols.append(col)
    
    logger.info(f"Detected {len(num_cols)} numeric and {len(cat_cols)} categorical features")
    return num_cols, cat_cols


def build_preprocessor(num_cols: List[str], cat_cols: List[str]) -> ColumnTransformer:
    """Build preprocessing pipeline."""
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
    
    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop"
    )
    
    return preprocessor


def get_base_models(task_type: str = "classification", random_state: int = 42) -> List[Tuple[str, Any]]:
    """Get list of base models to train."""
    models = []
    
    if task_type == "classification":
        models.extend([
            ("LogisticRegression", LogisticRegression(max_iter=2000, random_state=random_state)),
            ("DecisionTree", DecisionTreeClassifier(max_depth=10, random_state=random_state)),
            ("RandomForest", RandomForestClassifier(n_estimators=100, max_depth=15, random_state=random_state, n_jobs=-1)),
            ("GradientBoosting", GradientBoostingClassifier(n_estimators=100, random_state=random_state)),
            ("KNN", KNeighborsClassifier(n_neighbors=5, n_jobs=-1)),
            ("NaiveBayes", GaussianNB())
        ])
        
        models.append(("SVM", SVC(probability=True, random_state=random_state, max_iter=1000)))
        
        if XGBOOST_AVAILABLE:
            models.append(("XGBoost", XGBClassifier(
                n_estimators=100, 
                learning_rate=0.1, 
                max_depth=6,
                random_state=random_state, 
                eval_metric="logloss",
                n_jobs=-1
            )))
        
        if LIGHTGBM_AVAILABLE:
            models.append(("LightGBM", LGBMClassifier(
                n_estimators=100, 
                learning_rate=0.1,
                max_depth=6,
                random_state=random_state,
                verbose=-1,
                n_jobs=-1
            )))
        
        if CATBOOST_AVAILABLE:
            models.append(("CatBoost", CatBoostClassifier(
                iterations=100,
                learning_rate=0.1,
                depth=6,
                random_state=random_state,
                verbose=False
            )))
    
    elif task_type == "regression":
        models.extend([
            ("LinearRegression", LinearRegression(n_jobs=-1)),
            ("Ridge", Ridge(alpha=1.0, random_state=random_state)),
            ("Lasso", Lasso(alpha=1.0, random_state=random_state, max_iter=2000)),
            ("ElasticNet", ElasticNet(alpha=1.0, l1_ratio=0.5, random_state=random_state, max_iter=2000)),
            ("DecisionTree", DecisionTreeRegressor(max_depth=10, random_state=random_state)),
            ("RandomForest", RandomForestRegressor(n_estimators=100, max_depth=15, random_state=random_state, n_jobs=-1)),
            ("GradientBoosting", GradientBoostingRegressor(n_estimators=100, random_state=random_state)),
            ("KNN", KNeighborsRegressor(n_neighbors=5, n_jobs=-1))
        ])
        
        models.append(("SVR", SVR(max_iter=1000)))
        
        if XGBOOST_AVAILABLE:
            models.append(("XGBoost", XGBRegressor(
                n_estimators=100, 
                learning_rate=0.1,
                max_depth=6,
                random_state=random_state,
                n_jobs=-1
            )))
        
        if LIGHTGBM_AVAILABLE:
            models.append(("LightGBM", LGBMRegressor(
                n_estimators=100, 
                learning_rate=0.1,
                max_depth=6,
                random_state=random_state,
                verbose=-1,
                n_jobs=-1
            )))
        
        if CATBOOST_AVAILABLE:
            models.append(("CatBoost", CatBoostRegressor(
                iterations=100,
                learning_rate=0.1,
                depth=6,
                random_state=random_state,
                verbose=False
            )))
    
    return models


def plot_confusion_matrix(cm: np.ndarray, class_names: List[str], out_path: str, title: str = "Confusion Matrix"):
    """Plot confusion matrix with better styling."""
    fig, ax = plt.subplots(figsize=(max(8, len(class_names) * 0.8), max(6, len(class_names) * 0.6)))
    
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    sns.heatmap(cm_norm, annot=cm, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Proportion'}, ax=ax)
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("True", fontsize=12)
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_model_scores(models_metrics: List[Dict[str, Any]], out_path: str, task_type: str = "classification"):
    """Plot model comparison scores."""
    names = [m["model_name"] for m in models_metrics]
    
    if task_type == "classification":
        accuracies = [m["accuracy"] for m in models_metrics]
        f1_scores = [m["f1"] for m in models_metrics]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(max(10, len(names) * 1.2), 5))
        
        ax1.bar(range(len(names)), accuracies, color='steelblue', alpha=0.8)
        ax1.set_xticks(range(len(names)))
        ax1.set_xticklabels(names, rotation=45, ha="right")
        ax1.set_ylabel("Accuracy", fontsize=11)
        ax1.set_title("Model Accuracy", fontweight='bold')
        ax1.set_ylim([0, 1])
        ax1.grid(axis='y', alpha=0.3)
        
        ax2.bar(range(len(names)), f1_scores, color='coral', alpha=0.8)
        ax2.set_xticks(range(len(names)))
        ax2.set_xticklabels(names, rotation=45, ha="right")
        ax2.set_ylabel("F1-Score", fontsize=11)
        ax2.set_title("Model F1-Score", fontweight='bold')
        ax2.set_ylim([0, 1])
        ax2.grid(axis='y', alpha=0.3)
    
    else:
        r2_scores = [m["r2"] for m in models_metrics]
        mae_scores = [m["mae"] for m in models_metrics]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(max(10, len(names) * 1.2), 5))
        
        ax1.bar(range(len(names)), r2_scores, color='steelblue', alpha=0.8)
        ax1.set_xticks(range(len(names)))
        ax1.set_xticklabels(names, rotation=45, ha="right")
        ax1.set_ylabel("R² Score", fontsize=11)
        ax1.set_title("Model R² Score", fontweight='bold')
        ax1.axhline(y=0, color='r', linestyle='--', alpha=0.5)
        ax1.grid(axis='y', alpha=0.3)
        
        ax2.bar(range(len(names)), mae_scores, color='coral', alpha=0.8)
        ax2.set_xticks(range(len(names)))
        ax2.set_xticklabels(names, rotation=45, ha="right")
        ax2.set_ylabel("MAE", fontsize=11)
        ax2.set_title("Model Mean Absolute Error", fontweight='bold')
        ax2.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_roc_curve_binary(y_true: np.ndarray, y_prob: np.ndarray, out_path: str, title: str = "ROC Curve"):
    """Plot ROC curve for binary classification."""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc_score = roc_auc_score(y_true, y_prob)
    
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, label=f"ROC (AUC = {auc_score:.3f})", linewidth=2, color='steelblue')
    ax.plot([0, 1], [0, 1], linestyle="--", color='gray', label='Random')
    ax.set_xlabel("False Positive Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate", fontsize=11)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='lower right')
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_regression_actual_vs_predicted(y_true: np.ndarray, y_pred: np.ndarray, out_path: str, title: str = "Actual vs Predicted"):
    """Plot actual vs predicted values for regression."""
    fig, ax = plt.subplots(figsize=(7, 7))
    
    ax.scatter(y_true, y_pred, alpha=0.5, s=30, color='steelblue', edgecolors='k', linewidth=0.5)
    
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Ideal')
    
    r2 = r2_score(y_true, y_pred)
    ax.text(0.05, 0.95, f'R² = {r2:.3f}', transform=ax.transAxes, 
            fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    ax.set_xlabel("Actual Values", fontsize=11)
    ax.set_ylabel("Predicted Values", fontsize=11)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_regression_residuals(y_true: np.ndarray, y_pred: np.ndarray, out_path: str, title: str = "Residuals Plot"):
    """Plot residuals for regression."""
    residuals = y_true - y_pred
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    ax1.scatter(y_pred, residuals, alpha=0.5, s=30, color='steelblue', edgecolors='k', linewidth=0.5)
    ax1.axhline(0, color='r', linestyle='--', linewidth=2)
    ax1.set_xlabel("Predicted Values", fontsize=11)
    ax1.set_ylabel("Residuals", fontsize=11)
    ax1.set_title("Residuals vs Predicted", fontsize=12, fontweight='bold')
    ax1.grid(alpha=0.3)
    
    ax2.hist(residuals, bins=30, color='steelblue', alpha=0.7, edgecolor='black')
    ax2.set_xlabel("Residuals", fontsize=11)
    ax2.set_ylabel("Frequency", fontsize=11)
    ax2.set_title("Residuals Distribution", fontsize=12, fontweight='bold')
    ax2.axvline(0, color='r', linestyle='--', linewidth=2)
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def tune_with_optuna(
    preprocessor: ColumnTransformer, 
    X: pd.DataFrame, 
    y: pd.Series, 
    model_name: str, 
    base_model: Any, 
    task_type: str = "classification", 
    n_trials: int = 20, 
    cv: int = 3, 
    random_state: int = 42
) -> Any:
    """Tune model hyperparameters using Optuna."""
    if not OPTUNA_AVAILABLE:
        logger.warning("Optuna not installed – skipping tuning")
        return base_model
    
    logger.info(f"Tuning {model_name} with Optuna ({n_trials} trials)")
    
    def objective(trial):
        params = {}
        clf = None
        
        try:
            if task_type == "classification":
                if model_name == "RandomForest":
                    params = {
                        "n_estimators": trial.suggest_int("n_estimators", 50, 300),
                        "max_depth": trial.suggest_int("max_depth", 5, 30),
                        "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
                        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10)
                    }
                    clf = RandomForestClassifier(**params, random_state=random_state, n_jobs=-1)
                
                elif model_name == "XGBoost" and XGBOOST_AVAILABLE:
                    params = {
                        "n_estimators": trial.suggest_int("n_estimators", 50, 300),
                        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                        "max_depth": trial.suggest_int("max_depth", 3, 12),
                        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0)
                    }
                    clf = XGBClassifier(**params, random_state=random_state, eval_metric="logloss", n_jobs=-1)
                
                elif model_name == "LightGBM" and LIGHTGBM_AVAILABLE:
                    params = {
                        "n_estimators": trial.suggest_int("n_estimators", 50, 300),
                        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                        "num_leaves": trial.suggest_int("num_leaves", 20, 150),
                        "min_child_samples": trial.suggest_int("min_child_samples", 5, 50)
                    }
                    clf = LGBMClassifier(**params, random_state=random_state, verbose=-1, n_jobs=-1)
            
            elif task_type == "regression":
                if model_name == "RandomForest":
                    params = {
                        "n_estimators": trial.suggest_int("n_estimators", 50, 300),
                        "max_depth": trial.suggest_int("max_depth", 5, 30),
                        "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
                        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10)
                    }
                    clf = RandomForestRegressor(**params, random_state=random_state, n_jobs=-1)
                
                elif model_name == "XGBoost" and XGBOOST_AVAILABLE:
                    params = {
                        "n_estimators": trial.suggest_int("n_estimators", 50, 300),
                        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                        "max_depth": trial.suggest_int("max_depth", 3, 12),
                        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0)
                    }
                    clf = XGBRegressor(**params, random_state=random_state, n_jobs=-1)
                
                elif model_name == "LightGBM" and LIGHTGBM_AVAILABLE:
                    params = {
                        "n_estimators": trial.suggest_int("n_estimators", 50, 300),
                        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                        "num_leaves": trial.suggest_int("num_leaves", 20, 150),
                        "min_child_samples": trial.suggest_int("min_child_samples", 5, 50)
                    }
                    clf = LGBMRegressor(**params, random_state=random_state, verbose=-1, n_jobs=-1)
            
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
    
    # Recreate model with best params
    if task_type == "classification":
        if model_name == "RandomForest":
            return RandomForestClassifier(**best_params, random_state=random_state, n_jobs=-1)
        elif model_name == "XGBoost" and XGBOOST_AVAILABLE:
            return XGBClassifier(**best_params, random_state=random_state, eval_metric="logloss", n_jobs=-1)
        elif model_name == "LightGBM" and LIGHTGBM_AVAILABLE:
            return LGBMClassifier(**best_params, random_state=random_state, verbose=-1, n_jobs=-1)
    else:
        if model_name == "RandomForest":
            return RandomForestRegressor(**best_params, random_state=random_state, n_jobs=-1)
        elif model_name == "XGBoost" and XGBOOST_AVAILABLE:
            return XGBRegressor(**best_params, random_state=random_state, n_jobs=-1)
        elif model_name == "LightGBM" and LIGHTGBM_AVAILABLE:
            return LGBMRegressor(**best_params, random_state=random_state, verbose=-1, n_jobs=-1)
    
    return base_model


def compute_and_save_shap(
    pipe: Pipeline, 
    X: pd.DataFrame, 
    output_dirs: Dict[str, str], 
    run_id: str, 
    max_shap_samples: int = 200
) -> Dict[str, Optional[str]]:
    """Compute and save SHAP explainability plots."""
    if not SHAP_AVAILABLE:
        logger.warning("SHAP not installed – skipping explainability")
        return {"shap_summary": None, "shap_bar": None, "shap_csv": None}
    
    logger.info("Computing SHAP values...")
    
    X_sample = X.sample(n=min(max_shap_samples, len(X)), random_state=42)
    
    plots_dir = os.path.join(output_dirs["static"], "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    shap_summary_rel = f"plots/shap_summary_{run_id}.png"
    shap_bar_rel = f"plots/shap_bar_{run_id}.png"
    shap_csv_rel = f"plots/shap_values_{run_id}.csv"
    
    try:
        model = pipe.named_steps.get("model")
        preprocessor = pipe.named_steps.get("preprocessor")
        
        if preprocessor is None or model is None:
            raise RuntimeError("Pipeline lacks required steps")
        
        X_trans = preprocessor.transform(X_sample)
        
        if hasattr(model, 'tree_') or hasattr(model, 'estimators_'):
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_trans)
        else:
            explainer = shap.KernelExplainer(model.predict, X_trans)
            shap_values = explainer.shap_values(X_trans)
        
        if isinstance(shap_values, list):
            shap_values = np.abs(shap_values).mean(axis=0)
        
        try:
            feature_names = preprocessor.get_feature_names_out()
        except Exception:
            feature_names = [f"feature_{i}" for i in range(X_trans.shape[1])]
        
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, X_trans, feature_names=feature_names, show=False)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dirs["static"], shap_summary_rel), bbox_inches="tight")
        plt.close()
        
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        feature_importance = pd.DataFrame({
            'feature': feature_names,
            'mean_abs_shap': mean_abs_shap
        }).sort_values('mean_abs_shap', ascending=False)
        
        top_features = feature_importance.head(20)
        
        fig, ax = plt.subplots(figsize=(10, max(6, len(top_features) * 0.3)))
        ax.barh(range(len(top_features)), top_features['mean_abs_shap'].values[::-1], color='steelblue')
        ax.set_yticks(range(len(top_features)))
        ax.set_yticklabels(top_features['feature'].values[::-1])
        ax.set_xlabel("Mean |SHAP value|", fontsize=11)
        ax.set_title("Feature Importance (Top 20)", fontsize=13, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        fig.savefig(os.path.join(output_dirs["static"], shap_bar_rel), bbox_inches="tight")
        plt.close(fig)
        
        feature_importance.to_csv(os.path.join(output_dirs["static"], shap_csv_rel), index=False)
        
        logger.info("SHAP computation completed")
        return {
            "shap_summary": shap_summary_rel,
            "shap_bar": shap_bar_rel,
            "shap_csv": shap_csv_rel
        }
    
    except Exception as e:
        logger.error(f"SHAP computation failed: {e}")
        return {"shap_summary": None, "shap_bar": None, "shap_csv": None}


def run_automl_classification(
    df: pd.DataFrame,
    target_col: str,
    output_dirs: Dict[str, str],
    test_size: float = 0.2,
    random_state: int = 42,
    use_optuna: bool = False,
    optuna_trials: int = 20,
    compute_shap: bool = False,
    shap_max_samples: int = 200,
    feature_engineering: bool = False
) -> Dict[str, Any]:
    """Main AutoML classification pipeline."""
    logger.info("Starting AutoML Classification")
    start_time = datetime.now()
    
    # Feature engineering
    if feature_engineering:
        df = auto_feature_engineering(df, target_col)
    
    num_cols, cat_cols = detect_feature_types(df, target_col)
    preprocessor = build_preprocessor(num_cols, cat_cols)
    
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    label_encoder = None
    if y.dtype == "object" or str(y.dtype).startswith("category"):
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    logger.info(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
    
    model_candidates = get_base_models(task_type="classification", random_state=random_state)
    models_metrics = []
    trained_models = []
    
    for name, base_model in model_candidates:
        logger.info(f"Training {name}...")
        
        tuned_model = base_model
        if use_optuna and name in ["RandomForest", "XGBoost", "LightGBM"]:
            try:
                tuned_model = tune_with_optuna(
                    preprocessor, X_train, y_train, name, base_model,
                    task_type="classification", n_trials=optuna_trials, 
                    cv=3, random_state=random_state
                )
            except Exception as e:
                logger.warning(f"Optuna tuning failed for {name}: {e}")
        
        pipe = Pipeline(steps=[("preprocessor", preprocessor), ("model", tuned_model)])
        
        try:
            pipe.fit(X_train, y_train)
            y_pred = pipe.predict(X_test)
            
            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average="weighted")
            precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
            recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
            
            y_prob = None
            if hasattr(pipe, "predict_proba"):
                try:
                    y_prob = pipe.predict_proba(X_test)
                except Exception:
                    pass
            
            metrics = {
                "model_name": name,
                "accuracy": acc,
                "f1": f1,
                "precision": precision,
                "recall": recall
            }
            models_metrics.append(metrics)
            trained_models.append((name, pipe, y_pred, y_prob))
            
            logger.info(f"{name} - Accuracy: {acc:.4f}, F1: {f1:.4f}")
        
        except Exception as e:
            logger.error(f"Training failed for {name}: {e}")
    
    if not models_metrics:
        raise RuntimeError("No models were successfully trained")
    
    models_metrics_sorted = sorted(models_metrics, key=lambda x: x["f1"], reverse=True)
    best_name = models_metrics_sorted[0]["model_name"]
    
    best_pipe = best_y_pred = best_y_prob = None
    for name, pipe, y_pred, y_prob in trained_models:
        if name == best_name:
            best_pipe, best_y_pred, best_y_prob = pipe, y_pred, y_prob
            break
    
    run_id = str(uuid.uuid4())
    
    preprocessor_step = best_pipe.named_steps["preprocessor"]
    X_all_transformed = preprocessor_step.transform(X)
    
    try:
        feature_names = preprocessor_step.get_feature_names_out()
    except Exception:
        feature_names = [f"f_{i}" for i in range(X_all_transformed.shape[1])]
    
    preprocessed_df = pd.DataFrame(X_all_transformed, columns=feature_names)
    preprocessed_path = os.path.join(output_dirs["preprocessed"], f"preprocessed_{run_id}.csv")
    preprocessed_df.to_csv(preprocessed_path, index=False)
    
    model_path = os.path.join(output_dirs["models"], f"best_model_{run_id}.pkl")
    joblib.dump({
        "model": best_pipe,
        "label_encoder": label_encoder,
        "target_col": target_col,
        "feature_names": list(X.columns),
        "task_type": "classification"
    }, model_path)
    
    class_names = list(label_encoder.classes_) if label_encoder else [str(c) for c in np.unique(y)]
    
    cm = confusion_matrix(y_test, best_y_pred)
    cm_rel = f"plots/confusion_{run_id}.png"
    plot_confusion_matrix(cm, class_names, os.path.join(output_dirs["static"], cm_rel))
    
    roc_rel = None
    if best_y_prob is not None and len(np.unique(y)) == 2:
        pos_prob = best_y_prob[:, 1]
        roc_rel = f"plots/roc_{run_id}.png"
        plot_roc_curve_binary(y_test, pos_prob, os.path.join(output_dirs["static"], roc_rel))
    
    scores_rel = f"plots/scores_{run_id}.png"
    plot_model_scores(models_metrics_sorted, os.path.join(output_dirs["static"], scores_rel), "classification")
    
    report_text = classification_report(y_test, best_y_pred, target_names=class_names, zero_division=0)
    
    html_report_path = os.path.join(output_dirs["reports"], f"report_{run_id}.html")
    with open(html_report_path, "w", encoding="utf-8") as f:
        f.write(f"""
        <html>
        <head>
            <title>AutoML Classification Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1 {{ color: #333; }}
                table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #4CAF50; color: white; }}
                tr:nth-child(even) {{ background-color: #f2f2f2; }}
                pre {{ background-color: #f4f4f4; padding: 10px; border-radius: 5px; }}
            </style>
        </head>
        <body>
            <h1>AutoML Classification Report</h1>
            <p><strong>Run ID:</strong> {run_id}</p>
            <p><strong>Best Model:</strong> {best_name}</p>
            <p><strong>Training Time:</strong> {(datetime.now() - start_time).total_seconds():.2f}s</p>
            
            <h2>Model Leaderboard</h2>
            <table>
                <tr>
                    <th>Model</th>
                    <th>Accuracy</th>
                    <th>F1-Score</th>
                    <th>Precision</th>
                    <th>Recall</th>
                </tr>
        """)
        
        for m in models_metrics_sorted:
            f.write(f"""
                <tr>
                    <td>{m['model_name']}</td>
                    <td>{m['accuracy']:.4f}</td>
                    <td>{m['f1']:.4f}</td>
                    <td>{m['precision']:.4f}</td>
                    <td>{m['recall']:.4f}</td>
                </tr>
            """)
        
        f.write(f"""
            </table>
            
            <h2>Classification Report</h2>
            <pre>{report_text}</pre>
        </body>
        </html>
        """)
    
    shap_outputs = {"shap_summary_path": None, "shap_bar_path": None, "shap_csv_path": None}
    if compute_shap:
        try:
            shap_results = compute_and_save_shap(best_pipe, X_train, output_dirs, run_id, shap_max_samples)
            shap_outputs = {
                "shap_summary_path": shap_results.get("shap_summary"),
                "shap_bar_path": shap_results.get("shap_bar"),
                "shap_csv_path": shap_results.get("shap_csv")
            }
        except Exception as e:
            logger.error(f"SHAP computation failed: {e}")
    
    logger.info(f"AutoML Classification completed in {(datetime.now() - start_time).total_seconds():.2f}s")
    
    return {
        "run_id": run_id,
        "models_metrics": models_metrics_sorted,
        "best_model_name": best_name,
        "confusion_matrix_path": cm_rel,
        "roc_curve_path": roc_rel,
        "scores_plot_path": scores_rel,
        "preprocessed_path": os.path.basename(preprocessed_path),
        "model_path": os.path.basename(model_path),
        "html_report_filename": os.path.basename(html_report_path),
        "classification_report_text": report_text,
        "class_names": class_names,
        **shap_outputs
    }


def run_automl_regression(
    df: pd.DataFrame,
    target_col: str,
    output_dirs: Dict[str, str],
    test_size: float = 0.2,
    random_state: int = 42,
    use_optuna: bool = False,
    optuna_trials: int = 20,
    compute_shap: bool = False,
    shap_max_samples: int = 200,
    feature_engineering: bool = False
) -> Dict[str, Any]:
    """Main AutoML regression pipeline."""
    logger.info("Starting AutoML Regression")
    start_time = datetime.now()
    
    if feature_engineering:
        df = auto_feature_engineering(df, target_col)
    
    num_cols, cat_cols = detect_feature_types(df, target_col)
    preprocessor = build_preprocessor(num_cols, cat_cols)
    
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    if not pd.api.types.is_numeric_dtype(y):
        raise ValueError("Target column must be numeric for regression")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    logger.info(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
    
    model_candidates = get_base_models(task_type="regression", random_state=random_state)
    models_metrics = []
    trained_models = []
    
    for name, base_model in model_candidates:
        logger.info(f"Training {name}...")
        
        tuned_model = base_model
        if use_optuna and name in ["RandomForest", "XGBoost", "LightGBM"]:
            try:
                tuned_model = tune_with_optuna(
                    preprocessor, X_train, y_train, name, base_model,
                    task_type="regression", n_trials=optuna_trials,
                    cv=3, random_state=random_state
                )
            except Exception as e:
                logger.warning(f"Optuna tuning failed for {name}: {e}")
        
        pipe = Pipeline(steps=[("preprocessor", preprocessor), ("model", tuned_model)])
        
        try:
            pipe.fit(X_train, y_train)
            y_pred = pipe.predict(X_test)
            
            mae = mean_absolute_error(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test, y_pred)
            
            metrics = {
                "model_name": name,
                "mae": mae,
                "mse": mse,
                "rmse": rmse,
                "r2": r2
            }
            models_metrics.append(metrics)
            trained_models.append((name, pipe, y_pred))
            
            logger.info(f"{name} - R²: {r2:.4f}, MAE: {mae:.4f}")
        
        except Exception as e:
            logger.error(f"Training failed for {name}: {e}")
    
    if not models_metrics:
        raise RuntimeError("No models were successfully trained")
    
    models_metrics_sorted = sorted(models_metrics, key=lambda x: x["r2"], reverse=True)
    best_name = models_metrics_sorted[0]["model_name"]
    
    best_pipe = best_y_pred = None
    for name, pipe, y_pred in trained_models:
        if name == best_name:
            best_pipe, best_y_pred = pipe, y_pred
            break
    
    run_id = str(uuid.uuid4())
    
    preprocessor_step = best_pipe.named_steps["preprocessor"]
    X_all_transformed = preprocessor_step.transform(X)
    
    try:
        feature_names = preprocessor_step.get_feature_names_out()
    except Exception:
        feature_names = [f"f_{i}" for i in range(X_all_transformed.shape[1])]
    
    preprocessed_df = pd.DataFrame(X_all_transformed, columns=feature_names)
    preprocessed_path = os.path.join(output_dirs["preprocessed"], f"preprocessed_{run_id}.csv")
    preprocessed_df.to_csv(preprocessed_path, index=False)
    
    model_path = os.path.join(output_dirs["models"], f"best_model_{run_id}.pkl")
    joblib.dump({
        "model": best_pipe,
        "target_col": target_col,
        "feature_names": list(X.columns),
        "task_type": "regression"
    }, model_path)
    
    avp_rel = f"plots/actual_vs_predicted_{run_id}.png"
    plot_regression_actual_vs_predicted(y_test, best_y_pred, os.path.join(output_dirs["static"], avp_rel))
    
    res_rel = f"plots/residuals_{run_id}.png"
    plot_regression_residuals(y_test, best_y_pred, os.path.join(output_dirs["static"], res_rel))
    
    scores_rel = f"plots/scores_{run_id}.png"
    plot_model_scores(models_metrics_sorted, os.path.join(output_dirs["static"], scores_rel), "regression")
    
    html_report_path = os.path.join(output_dirs["reports"], f"report_{run_id}.html")
    with open(html_report_path, "w", encoding="utf-8") as f:
        f.write(f"""
        <html>
        <head>
            <title>AutoML Regression Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1 {{ color: #333; }}
                table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #2196F3; color: white; }}
                tr:nth-child(even) {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <h1>AutoML Regression Report</h1>
            <p><strong>Run ID:</strong> {run_id}</p>
            <p><strong>Best Model:</strong> {best_name}</p>
            <p><strong>Training Time:</strong> {(datetime.now() - start_time).total_seconds():.2f}s</p>
            
            <h2>Model Leaderboard</h2>
            <table>
                <tr>
                    <th>Model</th>
                    <th>R² Score</th>
                    <th>MAE</th>
                    <th>RMSE</th>
                </tr>
        """)
        
        for m in models_metrics_sorted:
            f.write(f"""
                <tr>
                    <td>{m['model_name']}</td>
                    <td>{m['r2']:.4f}</td>
                    <td>{m['mae']:.4f}</td>
                    <td>{m['rmse']:.4f}</td>
                </tr>
            """)
        
        f.write("""
            </table>
        </body>
        </html>
        """)
    
    shap_outputs = {"shap_summary_path": None, "shap_bar_path": None, "shap_csv_path": None}
    if compute_shap:
        try:
            shap_results = compute_and_save_shap(best_pipe, X_train, output_dirs, run_id, shap_max_samples)
            shap_outputs = {
                "shap_summary_path": shap_results.get("shap_summary"),
                "shap_bar_path": shap_results.get("shap_bar"),
                "shap_csv_path": shap_results.get("shap_csv")
            }
        except Exception as e:
            logger.error(f"SHAP computation failed: {e}")
    
    logger.info(f"AutoML Regression completed in {(datetime.now() - start_time).total_seconds():.2f}s")
    
    return {
        "run_id": run_id,
        "models_metrics": models_metrics_sorted,
        "best_model_name": best_name,
        "actual_vs_predicted_path": avp_rel,
        "residuals_path": res_rel,
        "scores_plot_path": scores_rel,
        "preprocessed_path": os.path.basename(preprocessed_path),
        "model_path": os.path.basename(model_path),
        "html_report_filename": os.path.basename(html_report_path),
        "task_type": "regression",
        **shap_outputs
    }