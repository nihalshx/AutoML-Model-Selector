# automl_core.py
import os
import uuid
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report, confusion_matrix, roc_curve,
    mean_absolute_error, mean_squared_error, r2_score
)

import matplotlib.pyplot as plt
import joblib

# Optional libraries
try:
    from xgboost import XGBClassifier, XGBRegressor
except Exception:
    XGBClassifier = None
    XGBRegressor = None

try:
    from lightgbm import LGBMClassifier, LGBMRegressor
except Exception:
    LGBMClassifier = None
    LGBMRegressor = None

try:
    import optuna
except Exception:
    optuna = None

try:
    import shap
except Exception:
    shap = None


def detect_feature_types(df: pd.DataFrame, target_col: str):
    feature_cols = [c for c in df.columns if c != target_col]
    num_cols = df[feature_cols].select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_cols = [c for c in feature_cols if c not in num_cols]
    return num_cols, cat_cols


def build_preprocessor(num_cols: List[str], cat_cols: List[str]) -> ColumnTransformer:
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, num_cols),
            ("cat", categorical_transformer, cat_cols),
        ],
        remainder="drop"
    )

    return preprocessor


def get_base_models(task_type: str = "classification", random_state: int = 42):
    models = []
    if task_type == "classification":
        models.append(("LogisticRegression", LogisticRegression(max_iter=1000)))
        models.append(("DecisionTree", DecisionTreeClassifier(random_state=random_state)))
        models.append(("RandomForest", RandomForestClassifier(n_estimators=200, random_state=random_state)))
        models.append(("SVM", SVC(probability=True, random_state=random_state)))
        models.append(("KNN", KNeighborsClassifier()))

        if XGBClassifier is not None:
            models.append(("XGBoost", XGBClassifier(n_estimators=300, learning_rate=0.05, random_state=random_state, use_label_encoder=False, eval_metric="logloss")))

        if LGBMClassifier is not None:
            models.append(("LightGBM", LGBMClassifier(n_estimators=300, learning_rate=0.05, random_state=random_state)))

    elif task_type == "regression":
        models.append(("LinearRegression", LinearRegression()))
        models.append(("Ridge", Ridge(random_state=random_state)))
        models.append(("Lasso", Lasso(random_state=random_state)))
        models.append(("ElasticNet", ElasticNet(random_state=random_state)))
        models.append(("DecisionTree", DecisionTreeRegressor(random_state=random_state)))
        models.append(("RandomForest", RandomForestRegressor(n_estimators=200, random_state=random_state)))
        models.append(("SVR", SVR()))
        models.append(("KNN", KNeighborsRegressor()))

        if XGBRegressor is not None:
            models.append(("XGBoost", XGBRegressor(n_estimators=300, learning_rate=0.05, random_state=random_state)))

        if LGBMRegressor is not None:
            models.append(("LightGBM", LGBMRegressor(n_estimators=300, learning_rate=0.05, random_state=random_state)))

    return models


def plot_confusion_matrix(cm, class_names, out_path: str, title="Confusion Matrix"):
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_title(title)
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)

    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center")

    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_model_scores(models_metrics: List[Dict[str, Any]], out_path: str):
    names = [m["model_name"] for m in models_metrics]
    accuracies = [m["accuracy"] for m in models_metrics]
    f1_scores = [m["f1"] for m in models_metrics]

    x = np.arange(len(names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(max(6, len(names) * 1.2), 4))
    ax.bar(x - width / 2, accuracies, width, label="Accuracy")
    ax.bar(x + width / 2, f1_scores, width, label="F1-score")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_ylabel("Score")
    ax.set_title("Model Performance Comparison")
    ax.legend()
    fig.tight_layout()

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_roc_curve_binary(y_true, y_prob, out_path: str, title="ROC Curve"):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label="ROC")
    ax.plot([0, 1], [0, 1], linestyle="--")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_regression_actual_vs_predicted(y_true, y_pred, out_path: str, title="Actual vs Predicted"):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_true, y_pred, alpha=0.5)
    
    # Ideal line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
    
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.set_title(title)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_regression_residuals(y_true, y_pred, out_path: str, title="Residuals Plot"):
    residuals = y_true - y_pred
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(y_pred, residuals, alpha=0.5)
    ax.axhline(0, color='r', linestyle='--')
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Residuals")
    ax.set_title(title)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


# -------------------------
# Optuna tuning helpers
# -------------------------
def tune_with_optuna(preprocessor: ColumnTransformer, X: pd.DataFrame, y: pd.Series, model_name: str, base_model, task_type: str = "classification", n_trials: int = 20, cv: int = 3, random_state: int = 42):
    """
    Returns a model instance with best params found by Optuna (or original base_model if optuna not available).
    Uses cross_val_score(f1_weighted) for classification and neg_mean_squared_error for regression as objective.
    """
    if optuna is None:
        print("Optuna not installed — skipping tuning.")
        return base_model

    def objective(trial):
        # Define param spaces per model_name
        params = {}
        clf = None
        
        # --- Classification Models ---
        if task_type == "classification":
            if model_name == "RandomForest":
                params["n_estimators"] = trial.suggest_int("n_estimators", 50, 500)
                params["max_depth"] = trial.suggest_int("max_depth", 3, 30)
                params["min_samples_split"] = trial.suggest_int("min_samples_split", 2, 20)
                clf = RandomForestClassifier(**params, random_state=random_state)
            elif model_name == "XGBoost" and XGBClassifier is not None:
                params["n_estimators"] = trial.suggest_int("n_estimators", 50, 500)
                params["learning_rate"] = trial.suggest_float("learning_rate", 0.01, 0.3)
                params["max_depth"] = trial.suggest_int("max_depth", 3, 12)
                params["subsample"] = trial.suggest_float("subsample", 0.5, 1.0)
                params["colsample_bytree"] = trial.suggest_float("colsample_bytree", 0.4, 1.0)
                clf = XGBClassifier(**params, random_state=random_state, use_label_encoder=False, eval_metric="logloss")
            elif model_name == "LightGBM" and LGBMClassifier is not None:
                params["n_estimators"] = trial.suggest_int("n_estimators", 50, 500)
                params["learning_rate"] = trial.suggest_float("learning_rate", 0.01, 0.3)
                params["num_leaves"] = trial.suggest_int("num_leaves", 16, 256)
                clf = LGBMClassifier(**params, random_state=random_state)
            elif model_name == "SVM":
                params["C"] = trial.suggest_float("C", 0.01, 100.0, log=True)
                params["gamma"] = trial.suggest_categorical("gamma", ["scale", "auto"])
                params["kernel"] = trial.suggest_categorical("kernel", ["rbf", "poly", "sigmoid"])
                clf = SVC(probability=True, **params, random_state=random_state)
            elif model_name == "KNN":
                params["n_neighbors"] = trial.suggest_int("n_neighbors", 1, 30)
                params["weights"] = trial.suggest_categorical("weights", ["uniform", "distance"])
                clf = KNeighborsClassifier(**params)
            elif model_name == "LogisticRegression":
                params["C"] = trial.suggest_float("C", 1e-4, 1e2, log=True)
                params["penalty"] = trial.suggest_categorical("penalty", ["l2"])
                clf = LogisticRegression(max_iter=2000, **{k: v for k,v in params.items() if v is not None})
        
        # --- Regression Models ---
        elif task_type == "regression":
            if model_name == "RandomForest":
                params["n_estimators"] = trial.suggest_int("n_estimators", 50, 500)
                params["max_depth"] = trial.suggest_int("max_depth", 3, 30)
                params["min_samples_split"] = trial.suggest_int("min_samples_split", 2, 20)
                clf = RandomForestRegressor(**params, random_state=random_state)
            elif model_name == "XGBoost" and XGBRegressor is not None:
                params["n_estimators"] = trial.suggest_int("n_estimators", 50, 500)
                params["learning_rate"] = trial.suggest_float("learning_rate", 0.01, 0.3)
                params["max_depth"] = trial.suggest_int("max_depth", 3, 12)
                params["subsample"] = trial.suggest_float("subsample", 0.5, 1.0)
                params["colsample_bytree"] = trial.suggest_float("colsample_bytree", 0.4, 1.0)
                clf = XGBRegressor(**params, random_state=random_state)
            elif model_name == "LightGBM" and LGBMRegressor is not None:
                params["n_estimators"] = trial.suggest_int("n_estimators", 50, 500)
                params["learning_rate"] = trial.suggest_float("learning_rate", 0.01, 0.3)
                params["num_leaves"] = trial.suggest_int("num_leaves", 16, 256)
                clf = LGBMRegressor(**params, random_state=random_state)
            elif model_name == "SVR":
                params["C"] = trial.suggest_float("C", 0.01, 100.0, log=True)
                params["gamma"] = trial.suggest_categorical("gamma", ["scale", "auto"])
                params["kernel"] = trial.suggest_categorical("kernel", ["rbf", "poly", "sigmoid"])
                clf = SVR(**params)
            elif model_name == "KNN":
                params["n_neighbors"] = trial.suggest_int("n_neighbors", 1, 30)
                params["weights"] = trial.suggest_categorical("weights", ["uniform", "distance"])
                clf = KNeighborsRegressor(**params)
            elif model_name in ["Ridge", "Lasso", "ElasticNet"]:
                params["alpha"] = trial.suggest_float("alpha", 0.01, 10.0, log=True)
                if model_name == "Ridge":
                    clf = Ridge(random_state=random_state, **params)
                elif model_name == "Lasso":
                    clf = Lasso(random_state=random_state, **params)
                elif model_name == "ElasticNet":
                    params["l1_ratio"] = trial.suggest_float("l1_ratio", 0.1, 0.9)
                    clf = ElasticNet(random_state=random_state, **params)

        if clf is None:
            # fallback: no tuning for models we don't have a search space for
            clf = base_model

        # Pipeline for CV
        pipe = Pipeline(steps=[("preprocessor", preprocessor), ("model", clf)])
        try:
            scoring = "f1_weighted" if task_type == "classification" else "neg_mean_squared_error"
            scores = cross_val_score(pipe, X, y, cv=cv, scoring=scoring)
            return float(np.mean(scores))
        except Exception as e:
            # return a very low score on failure to guide optuna away
            print(f"Optuna trial error for {model_name}: {e}")
            return -float("inf")

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)
    best_params = study.best_params
    print(f"Optuna best params for {model_name}: {best_params}")

    # Create new model instance with best params where applicable
    tuned = base_model
    # Re-instantiate based on best_params (simplified logic: check model_name again)
    # Note: Ideally we'd refactor the instantiation logic to be reusable, but for now we duplicate slightly for clarity
    
    if task_type == "classification":
        if model_name == "RandomForest":
            tuned = RandomForestClassifier(**best_params, random_state=random_state)
        elif model_name == "XGBoost" and XGBClassifier is not None:
            tuned = XGBClassifier(**best_params, random_state=random_state, use_label_encoder=False, eval_metric="logloss")
        elif model_name == "LightGBM" and LGBMClassifier is not None:
            tuned = LGBMClassifier(**best_params, random_state=random_state)
        elif model_name == "SVM":
            tuned = SVC(probability=True, **best_params, random_state=random_state)
        elif model_name == "KNN":
            tuned = KNeighborsClassifier(**best_params)
        elif model_name == "LogisticRegression":
            tuned = LogisticRegression(max_iter=2000, **best_params)
            
    elif task_type == "regression":
        if model_name == "RandomForest":
            tuned = RandomForestRegressor(**best_params, random_state=random_state)
        elif model_name == "XGBoost" and XGBRegressor is not None:
            tuned = XGBRegressor(**best_params, random_state=random_state)
        elif model_name == "LightGBM" and LGBMRegressor is not None:
            tuned = LGBMRegressor(**best_params, random_state=random_state)
        elif model_name == "SVR":
            tuned = SVR(**best_params)
        elif model_name == "KNN":
            tuned = KNeighborsRegressor(**best_params)
        elif model_name == "Ridge":
            tuned = Ridge(random_state=random_state, **best_params)
        elif model_name == "Lasso":
            tuned = Lasso(random_state=random_state, **best_params)
        elif model_name == "ElasticNet":
            tuned = ElasticNet(random_state=random_state, **best_params)

    return tuned


# -------------------------
# SHAP explainability helpers
# -------------------------
def compute_and_save_shap(pipe: Pipeline, X: pd.DataFrame, output_dirs: Dict[str, str], run_id: str, max_shap_samples: int = 200):
    """
    Attempts to compute SHAP values for the pipeline `pipe` on a sample of X (raw dataframe).
    Saves:
      - shap_summary_plot_{run_id}.png
      - shap_bar_{run_id}.png (mean abs shap per feature)
      - shap_values_{run_id}.csv  (mean absolute shap per feature)
    Returns dict with relative paths or None if shap not available.
    """
    if shap is None:
        print("shap is not installed — skipping SHAP explainability.")
        return {
            "shap_summary": None,
            "shap_bar": None,
            "shap_csv": None
        }

    # sample X to keep runtime reasonable
    if X.shape[0] > max_shap_samples:
        X_sample = X.sample(n=max_shap_samples, random_state=42)
    else:
        X_sample = X.copy()

    # try to compute explainer using shap.Explainer on the pipeline (works often)
    try:
        explainer = shap.Explainer(pipe, X_sample)
        shap_values = explainer(X_sample)
        # shap_values is an Explanation object; shap_values.values shape varies
    except Exception as e:
        # fallback: try to explain inner model with preprocessed data for tree models
        print(f"shap.Explainer failed: {e}. Trying TreeExplainer on inner model using preprocessed data.")
        try:
            preprocessor = pipe.named_steps.get("preprocessor", None)
            model = pipe.named_steps.get("model", None)
            if preprocessor is None or model is None:
                raise RuntimeError("Pipeline lacks preprocessor/model steps.")

            X_trans = preprocessor.transform(X_sample)
            # use TreeExplainer for tree models
            explainer = shap.TreeExplainer(model)
            shap_values_raw = explainer.shap_values(X_trans)
            # create a simple wrapper object to unify handling below
            shap_values = (shap_values_raw, X_trans)
        except Exception as e2:
            print(f"Fallback SHAP failed: {e2}")
            return {"shap_summary": None, "shap_bar": None, "shap_csv": None}

    # Now produce plots
    plots_dir = os.path.join(output_dirs["static"], "plots")
    os.makedirs(plots_dir, exist_ok=True)

    shap_summary_rel = f"plots/shap_summary_{run_id}.png"
    shap_bar_rel = f"plots/shap_bar_{run_id}.png"
    shap_csv_rel = f"plots/shap_values_{run_id}.csv"
    shap_summary_abs = os.path.join(output_dirs["static"], shap_summary_rel)
    shap_bar_abs = os.path.join(output_dirs["static"], shap_bar_rel)
    shap_csv_abs = os.path.join(output_dirs["static"], shap_csv_rel)

    try:
        # If shap_values is Explanation object (newer shap versions)
        if hasattr(shap_values, "values"):
            # summary plot
            plt.figure(figsize=(8, 6))
            shap.plots.beeswarm(shap_values, show=False)
            plt.tight_layout()
            plt.savefig(shap_summary_abs, bbox_inches="tight")
            plt.close()

            # mean abs shap per feature (for tabular shap.Explanation we can use .abs.mean(0))
            if hasattr(shap_values, "values"):
                # shap_values.values: if multiclass, it's list/array per class
                vals = shap_values.values
                if isinstance(vals, list) or (isinstance(vals, np.ndarray) and vals.ndim == 3):
                    # multiclass: average abs across classes and samples
                    # convert to array: (n_classes, n_samples, n_features) or (n_samples, n_features) depending
                    arr = np.array(vals)
                    # try to get mean abs across samples & classes
                    mean_abs = np.mean(np.abs(arr), axis=(0, 1)) if arr.ndim == 3 else np.mean(np.abs(arr), axis=0)
                    feature_names = shap_values.feature_names if hasattr(shap_values, "feature_names") else [f"f{i}" for i in range(mean_abs.shape[0])]
                else:
                    # binary / single class: shape (n_samples, n_features)
                    mean_abs = np.mean(np.abs(vals), axis=0)
                    feature_names = shap_values.feature_names if hasattr(shap_values, "feature_names") else [f"f{i}" for i in range(mean_abs.shape[0])]

            else:
                mean_abs = None
                feature_names = None

        else:
            # older fallback: shap_values is tuple (raw_shap_values, X_trans)
            raw_vals, X_trans = shap_values
            arr = np.array(raw_vals)
            if arr.ndim == 3:
                mean_abs = np.mean(np.abs(arr), axis=(0, 1))
            elif arr.ndim == 2:
                mean_abs = np.mean(np.abs(arr), axis=0)
            else:
                mean_abs = np.mean(np.abs(arr))
            feature_names = [f"f_{i}" for i in range(mean_abs.shape[0])]

        # create bar plot of mean_abs
        if mean_abs is not None and feature_names is not None:
            order = np.argsort(mean_abs)[::-1]
            top_idx = order[:50]  # cap to top 50 features for readability
            top_feats = [feature_names[i] for i in top_idx]
            top_vals = mean_abs[top_idx]

            fig, ax = plt.subplots(figsize=(8, max(4, len(top_feats) * 0.2)))
            ax.barh(range(len(top_feats))[::-1], top_vals[::-1])
            ax.set_yticks(range(len(top_feats))[::-1])
            ax.set_yticklabels(top_feats[::-1])
            ax.set_xlabel("Mean |SHAP value|")
            ax.set_title("Feature importance (mean |SHAP|)")
            plt.tight_layout()
            fig.savefig(shap_bar_abs, bbox_inches="tight")
            plt.close(fig)

            # save CSV of mean_abs
            df_shap = pd.DataFrame({
                "feature": feature_names,
                "mean_abs_shap": mean_abs
            })
            df_shap = df_shap.sort_values("mean_abs_shap", ascending=False)
            df_shap.to_csv(shap_csv_abs, index=False)
        else:
            shap_summary_abs = shap_bar_abs = shap_csv_abs = None

    except Exception as e:
        print(f"Error while plotting SHAP: {e}")
        shap_summary_abs = shap_bar_abs = shap_csv_abs = None

    # Return relative paths (for static/<path>)
    rel_summary = shap_summary_rel if shap_summary_abs and os.path.exists(shap_summary_abs) else None
    rel_bar = shap_bar_rel if shap_bar_abs and os.path.exists(shap_bar_abs) else None
    rel_csv = shap_csv_rel if shap_csv_abs and os.path.exists(shap_csv_abs) else None

    return {"shap_summary": rel_summary, "shap_bar": rel_bar, "shap_csv": rel_csv}


# -------------------------
# Main runner (classification)
# -------------------------
def run_automl_classification(
    df: pd.DataFrame,
    target_col: str,
    output_dirs: Dict[str, str],
    test_size: float = 0.2,
    random_state: int = 42,
    use_optuna: bool = False,
    optuna_trials: int = 20,
    compute_shap: bool = False,
    shap_max_samples: int = 200
) -> Dict[str, Any]:
    """
    Main AutoML runner with optional Optuna and SHAP.
    """
    assert target_col in df.columns, "Target column not found in dataset."

    num_cols, cat_cols = detect_feature_types(df, target_col)
    preprocessor = build_preprocessor(num_cols, cat_cols)

    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Encode target if categorical
    label_encoder = None
    if y.dtype == "object" or str(y.dtype).startswith("category"):
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Prepare candidate models (base)
    model_candidates = get_base_models(random_state=random_state)

    models_metrics = []
    trained_models = []

    for name, base_model in model_candidates:
        print(f"Processing candidate: {name}")

        # Tune with Optuna if requested
        tuned_model = base_model
        if use_optuna:
            try:
                tuned_model = tune_with_optuna(preprocessor, X_train, y_train, name, base_model, n_trials=optuna_trials, cv=3, random_state=random_state)
            except Exception as e:
                print(f"Optuna tuning failed for {name}: {e} — using base model.")

        pipe = Pipeline(steps=[("preprocessor", preprocessor), ("model", tuned_model)])
        try:
            pipe.fit(X_train, y_train)
        except Exception as e:
            print(f"Training failed for {name}: {e}")
            continue

        y_pred = pipe.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="weighted")

        try:
            if hasattr(pipe, "predict_proba"):
                y_prob = pipe.predict_proba(X_test)
            else:
                y_prob = None
        except Exception:
            y_prob = None

        metrics = {"model_name": name, "accuracy": acc, "f1": f1}
        models_metrics.append(metrics)
        trained_models.append((name, pipe, y_pred, y_prob))

    # If no model trained
    if len(models_metrics) == 0:
        raise RuntimeError("No models were successfully trained.")

    # Rank models by F1-score
    models_metrics_sorted = sorted(models_metrics, key=lambda x: x["f1"], reverse=True)
    best_name = models_metrics_sorted[0]["model_name"]

    # retrieve best pipeline
    best_pipe = None
    best_y_pred = None
    best_y_prob = None
    for (name, pipe, y_pred, y_prob) in trained_models:
        if name == best_name:
            best_pipe = pipe
            best_y_pred = y_pred
            best_y_prob = y_prob
            break

    run_id = str(uuid.uuid4())

    # Save preprocessed dataset using best preprocessor
    preprocessor_step = best_pipe.named_steps["preprocessor"]
    try:
        X_all_transformed = preprocessor_step.transform(X)
    except Exception:
        # fit_transform as fallback
        X_all_transformed = preprocessor_step.fit_transform(X)

    try:
        if hasattr(preprocessor_step, "get_feature_names_out"):
            feature_names = preprocessor_step.get_feature_names_out()
        else:
            feature_names = [f"f_{i}" for i in range(X_all_transformed.shape[1])]
    except Exception:
        feature_names = [f"f_{i}" for i in range(X_all_transformed.shape[1])]

    preprocessed_df = pd.DataFrame(X_all_transformed, columns=feature_names)
    preprocessed_path = os.path.join(output_dirs["preprocessed"], f"preprocessed_{run_id}.csv")
    os.makedirs(output_dirs["preprocessed"], exist_ok=True)
    preprocessed_df.to_csv(preprocessed_path, index=False)

    # Save best model pipeline
    os.makedirs(output_dirs["models"], exist_ok=True)
    model_path = os.path.join(output_dirs["models"], f"best_model_{run_id}.pkl")
    joblib.dump({"model": best_pipe, "label_encoder": label_encoder, "target_col": target_col}, model_path)

    # Confusion matrix
    cm = confusion_matrix(y_test, best_y_pred)
    class_names = list(label_encoder.classes_) if label_encoder is not None else [str(c) for c in np.unique(y)]
    cm_rel = f"plots/confusion_{run_id}.png"
    cm_abs = os.path.join(output_dirs["static"], cm_rel)
    plot_confusion_matrix(cm, class_names, cm_abs)

    # ROC curve (binary)
    roc_rel = None
    if best_y_prob is not None and len(np.unique(y)) == 2:
        pos_prob = best_y_prob[:, 1] if best_y_prob.ndim == 2 and best_y_prob.shape[1] == 2 else best_y_prob
        roc_rel = f"plots/roc_{run_id}.png"
        roc_abs = os.path.join(output_dirs["static"], roc_rel)
        plot_roc_curve_binary(y_test, pos_prob, roc_abs)

    # Model score bar chart
    scores_rel = f"plots/scores_{run_id}.png"
    scores_abs = os.path.join(output_dirs["static"], scores_rel)
    plot_model_scores(models_metrics_sorted, scores_abs)

    # Classification report
    report_text = classification_report(y_test, best_y_pred, target_names=class_names)

    # HTML report
    os.makedirs(output_dirs["reports"], exist_ok=True)
    html_report_path = os.path.join(output_dirs["reports"], f"report_{run_id}.html")
    with open(html_report_path, "w", encoding="utf-8") as f:
        f.write("<html><body>")
        f.write(f"<h1>AutoML Classification Report - Run {run_id}</h1>")
        f.write(f"<h2>Best Model: {best_name}</h2>")
        f.write("<h3>Leaderboard</h3><ul>")
        for m in models_metrics_sorted:
            f.write(f"<li>{m['model_name']}: Accuracy={m['accuracy']:.4f}, F1={m['f1']:.4f}</li>")
        f.write("</ul>")
        f.write("<h3>Classification Report</h3>")
        f.write("<pre>")
        f.write(report_text)
        f.write("</pre>")
        f.write("</body></html>")

    # SHAP explainability (optional)
    shap_summary_rel = shap_bar_rel = shap_csv_rel = None
    if compute_shap:
        try:
            shap_outputs = compute_and_save_shap(best_pipe, X_train, output_dirs, run_id, max_shap_samples=shap_max_samples)
            shap_summary_rel = shap_outputs.get("shap_summary")
            shap_bar_rel = shap_outputs.get("shap_bar")
            shap_csv_rel = shap_outputs.get("shap_csv")
        except Exception as e:
            print(f"SHAP computation failed: {e}")

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
        "shap_summary_path": shap_summary_rel,
        "shap_bar_path": shap_bar_rel,
        "shap_csv_path": shap_csv_rel
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
    shap_max_samples: int = 200
) -> Dict[str, Any]:
    """
    Main AutoML runner for Regression.
    """
    assert target_col in df.columns, "Target column not found in dataset."

    num_cols, cat_cols = detect_feature_types(df, target_col)
    preprocessor = build_preprocessor(num_cols, cat_cols)

    X = df.drop(columns=[target_col])
    y = df[target_col]

    # For regression, we don't encode target, but check if it is numeric
    if not pd.api.types.is_numeric_dtype(y):
        raise ValueError("Target column must be numeric for regression.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # Prepare candidate models (regression)
    model_candidates = get_base_models(task_type="regression", random_state=random_state)

    models_metrics = []
    trained_models = []

    for name, base_model in model_candidates:
        print(f"Processing candidate: {name}")

        # Tune with Optuna if requested
        tuned_model = base_model
        if use_optuna:
            try:
                tuned_model = tune_with_optuna(
                    preprocessor, X_train, y_train, name, base_model, 
                    task_type="regression", n_trials=optuna_trials, cv=3, random_state=random_state
                )
            except Exception as e:
                print(f"Optuna tuning failed for {name}: {e} — using base model.")

        pipe = Pipeline(steps=[("preprocessor", preprocessor), ("model", tuned_model)])
        try:
            pipe.fit(X_train, y_train)
        except Exception as e:
            print(f"Training failed for {name}: {e}")
            continue

        y_pred = pipe.predict(X_test)

        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        metrics = {"model_name": name, "mae": mae, "mse": mse, "r2": r2}
        models_metrics.append(metrics)
        trained_models.append((name, pipe, y_pred))

    # If no model trained
    if len(models_metrics) == 0:
        raise RuntimeError("No models were successfully trained.")

    # Rank models by R2 score (higher is better)
    # Note: If R2 is negative, it means model is worse than mean predictor.
    models_metrics_sorted = sorted(models_metrics, key=lambda x: x["r2"], reverse=True)
    best_name = models_metrics_sorted[0]["model_name"]

    # retrieve best pipeline
    best_pipe = None
    best_y_pred = None
    for (name, pipe, y_pred) in trained_models:
        if name == best_name:
            best_pipe = pipe
            best_y_pred = y_pred
            break

    run_id = str(uuid.uuid4())

    # Save preprocessed dataset using best preprocessor
    preprocessor_step = best_pipe.named_steps["preprocessor"]
    try:
        X_all_transformed = preprocessor_step.transform(X)
    except Exception:
        X_all_transformed = preprocessor_step.fit_transform(X)

    try:
        if hasattr(preprocessor_step, "get_feature_names_out"):
            feature_names = preprocessor_step.get_feature_names_out()
        else:
            feature_names = [f"f_{i}" for i in range(X_all_transformed.shape[1])]
    except Exception:
        feature_names = [f"f_{i}" for i in range(X_all_transformed.shape[1])]

    preprocessed_df = pd.DataFrame(X_all_transformed, columns=feature_names)
    preprocessed_path = os.path.join(output_dirs["preprocessed"], f"preprocessed_{run_id}.csv")
    os.makedirs(output_dirs["preprocessed"], exist_ok=True)
    preprocessed_df.to_csv(preprocessed_path, index=False)

    # Save best model pipeline
    os.makedirs(output_dirs["models"], exist_ok=True)
    model_path = os.path.join(output_dirs["models"], f"best_model_{run_id}.pkl")
    joblib.dump({"model": best_pipe, "target_col": target_col, "task_type": "regression"}, model_path)

    # Plots
    # 1. Actual vs Predicted
    avp_rel = f"plots/actual_vs_predicted_{run_id}.png"
    avp_abs = os.path.join(output_dirs["static"], avp_rel)
    plot_regression_actual_vs_predicted(y_test, best_y_pred, avp_abs)

    # 2. Residuals
    res_rel = f"plots/residuals_{run_id}.png"
    res_abs = os.path.join(output_dirs["static"], res_rel)
    plot_regression_residuals(y_test, best_y_pred, res_abs)

    # Model score bar chart (R2)
    # Re-use plot_model_scores but adapt it? 
    # The existing plot_model_scores expects 'accuracy' and 'f1'.
    # Let's make a simple custom plot for regression scores here or adapt the existing one.
    # For simplicity, let's create a new plot function inline or just plot R2.
    
    names = [m["model_name"] for m in models_metrics_sorted]
    r2_scores = [m["r2"] for m in models_metrics_sorted]
    
    fig, ax = plt.subplots(figsize=(max(6, len(names) * 1.2), 4))
    x = np.arange(len(names))
    ax.bar(x, r2_scores, color='skyblue', label="R2 Score")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_ylabel("R2 Score")
    ax.set_title("Model Performance Comparison (Regression)")
    ax.legend()
    fig.tight_layout()
    scores_rel = f"plots/scores_{run_id}.png"
    scores_abs = os.path.join(output_dirs["static"], scores_rel)
    os.makedirs(os.path.dirname(scores_abs), exist_ok=True)
    fig.savefig(scores_abs, bbox_inches="tight")
    plt.close(fig)

    # HTML report
    os.makedirs(output_dirs["reports"], exist_ok=True)
    html_report_path = os.path.join(output_dirs["reports"], f"report_{run_id}.html")
    with open(html_report_path, "w", encoding="utf-8") as f:
        f.write("<html><body>")
        f.write(f"<h1>AutoML Regression Report - Run {run_id}</h1>")
        f.write(f"<h2>Best Model: {best_name}</h2>")
        f.write("<h3>Leaderboard</h3><ul>")
        for m in models_metrics_sorted:
            f.write(f"<li>{m['model_name']}: R2={m['r2']:.4f}, MAE={m['mae']:.4f}, MSE={m['mse']:.4f}</li>")
        f.write("</ul>")
        f.write("</body></html>")

    # SHAP explainability (optional)
    shap_summary_rel = shap_bar_rel = shap_csv_rel = None
    if compute_shap:
        try:
            shap_outputs = compute_and_save_shap(best_pipe, X_train, output_dirs, run_id, max_shap_samples=shap_max_samples)
            shap_summary_rel = shap_outputs.get("shap_summary")
            shap_bar_rel = shap_outputs.get("shap_bar")
            shap_csv_rel = shap_outputs.get("shap_csv")
        except Exception as e:
            print(f"SHAP computation failed: {e}")

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
        "shap_summary_path": shap_summary_rel,
        "shap_bar_path": shap_bar_rel,
        "shap_csv_path": shap_csv_rel
    }
