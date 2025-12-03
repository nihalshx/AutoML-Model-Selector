# AutoML Model Selector — Full Project

This document contains the **complete project structure** and full source code for the AutoML Model Selector Web App (Flask) with **Optuna hyperparameter tuning** and **SHAP explainability**.

> Save the files exactly as named (preserve paths). After that run `pip install -r requirements.txt` and then `python app.py` (or build Docker as shown below).

---

## Project tree

```
automl_model_selector/
├── README.md
├── .gitignore
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── app.py
├── automl_core.py
├── uploads/                # created at runtime
├── models/                 # created at runtime
├── reports/                # created at runtime
├── preprocessed/           # created at runtime
├── static/
│   ├── css/
│   │   └── style.css
│   └── plots/              # created at runtime
└── templates/
    ├── upload.html
    └── results.html
```

---

## 1) README.md

```markdown
# AutoML Model Selector

Flask-based AutoML application that accepts a CSV dataset and automatically trains, tunes (Optuna), and evaluates multiple classification models. It also produces SHAP explainability outputs for the selected best model.

## Features
- Data preprocessing (imputation, encoding, scaling)
- Trains multiple models: LogisticRegression, DecisionTree, RandomForest, SVM, KNN, XGBoost, LightGBM
- Optional Optuna hyperparameter tuning
- Optional SHAP explainability (summary + feature importance CSV)
- Visualizations: model score bar chart, confusion matrix, ROC (binary)
- Downloadable artifacts: best model (.pkl), preprocessed CSV, HTML report

## Quickstart

1. Create a virtualenv and activate it.

```bash
python -m venv venv
source venv/bin/activate   # on Windows: venv\Scripts\activate
pip install -r requirements.txt
```

2. Run the app

```bash
python app.py
```

3. Open `http://127.0.0.1:5000` in your browser and upload a CSV. Enter the target column name.

## Notes
- Optuna and SHAP can be compute heavy. Use small `optuna_trials` for quick tests.
- If you want background processing for longer jobs, consider adding Celery/RQ and a task queue.

```

```
```

---

## 2) .gitignore

```gitignore
venv/
__pycache__/
*.pyc
uploads/
models/
reports/
preprocessed/
static/plots/
.env
.DS_Store
```

---

## 3) Dockerfile (optional)

```dockerfile
# Use an official lightweight Python runtime as a parent image
FROM python:3.11-slim

WORKDIR /app

# system deps for xgboost/lightgbm if needed (may increase image size)
RUN apt-get update && apt-get install -y build-essential git libssl-dev libffi-dev && rm -rf /var/lib/apt/lists/*

# Copy project
COPY . /app

# Install Python deps
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Create folders
RUN mkdir -p uploads models reports preprocessed static/plots

EXPOSE 5000
CMD ["python", "app.py"]
```

---

## 4) docker-compose.yml (optional)

```yaml
version: '3.8'
services:
  web:
    build: .
    ports:
      - "5000:5000"
    volumes:
      - ./:/app
      - ./uploads:/app/uploads
      - ./models:/app/models
      - ./reports:/app/reports
      - ./preprocessed:/app/preprocessed
      - ./static/plots:/app/static/plots
```

---

## 5) requirements.txt

```
Flask>=2.0
pandas
numpy
scikit-learn
matplotlib
joblib
optuna
shap
xgboost
lightgbm
```

> If XGBoost/LightGBM or SHAP cause install issues, you can remove them for a lighter setup; the code will gracefully skip them if unavailable.

---

## 6) app.py

```python
# app.py
import os
from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    send_from_directory,
    flash
)
from werkzeug.utils import secure_filename
import pandas as pd

from automl_core import run_automl_classification

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
MODELS_FOLDER = os.path.join(BASE_DIR, "models")
REPORTS_FOLDER = os.path.join(BASE_DIR, "reports")
PREPROCESSED_FOLDER = os.path.join(BASE_DIR, "preprocessed")
STATIC_FOLDER = os.path.join(BASE_DIR, "static")

ALLOWED_EXTENSIONS = {"csv"}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(MODELS_FOLDER, exist_ok=True)
os.makedirs(REPORTS_FOLDER, exist_ok=True)
os.makedirs(PREPROCESSED_FOLDER, exist_ok=True)
os.makedirs(os.path.join(STATIC_FOLDER, "plots"), exist_ok=True)

app = Flask(__name__, static_folder="static", template_folder="templates")
app.secret_key = os.environ.get("FLASK_SECRET", "change_this_secret_key")


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/", methods=["GET", "POST"])
def upload_file():
    if request.method == "POST":
        if "dataset" not in request.files:
            flash("No file part")
            return redirect(request.url)

        file = request.files["dataset"]
        target_col = request.form.get("target_col")

        if file.filename == "":
            flash("No selected file")
            return redirect(request.url)

        if not target_col:
            flash("Please provide the target column name.")
            return redirect(request.url)

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(UPLOAD_FOLDER, filename)
            file.save(file_path)

            try:
                df = pd.read_csv(file_path)

                if target_col not in df.columns:
                    flash(f"Target column '{target_col}' not found in dataset.")
                    return redirect(request.url)

                # Read form options
                use_optuna = request.form.get("use_optuna") == "on"
                try:
                    optuna_trials = int(request.form.get("optuna_trials") or 20)
                except Exception:
                    optuna_trials = 20

                compute_shap = request.form.get("compute_shap") == "on"
                try:
                    shap_max_samples = int(request.form.get("shap_max_samples") or 200)
                except Exception:
                    shap_max_samples = 200

                output_dirs = {
                    "models": MODELS_FOLDER,
                    "reports": REPORTS_FOLDER,
                    "preprocessed": PREPROCESSED_FOLDER,
                    "static": STATIC_FOLDER,
                }

                results = run_automl_classification(
                    df, target_col,
                    output_dirs,
                    use_optuna=use_optuna,
                    optuna_trials=optuna_trials,
                    compute_shap=compute_shap,
                    shap_max_samples=shap_max_samples
                )

                # Build download URLs
                model_download_url = url_for(
                    "download_model",
                    filename=results["model_path"]
                )
                preprocessed_download_url = url_for(
                    "download_preprocessed",
                    filename=results["preprocessed_path"]
                )
                report_download_url = url_for(
                    "download_report",
                    filename=results["html_report_filename"]
                )

                confusion_img_url = url_for(
                    "static",
                    filename=results["confusion_matrix_path"]
                )
                roc_img_url = None
                if results["roc_curve_path"]:
                    roc_img_url = url_for(
                        "static",
                        filename=results["roc_curve_path"]
                    )
                scores_img_url = url_for(
                    "static",
                    filename=results["scores_plot_path"]
                )

                shap_summary_url = None
                shap_bar_url = None
                shap_csv_url = None
                if results.get("shap_summary_path"):
                    shap_summary_url = url_for("static", filename=results["shap_summary_path"]) if results["shap_summary_path"] else None
                if results.get("shap_bar_path"):
                    shap_bar_url = url_for("static", filename=results["shap_bar_path"]) if results["shap_bar_path"] else None
                if results.get("shap_csv_path"):
                    # CSV is inside static (plots) so we can link directly
                    shap_csv_url = url_for("static", filename=results["shap_csv_path"]) if results["shap_csv_path"] else None

                return render_template(
                    "results.html",
                    results=results,
                    model_download_url=model_download_url,
                    preprocessed_download_url=preprocessed_download_url,
                    report_download_url=report_download_url,
                    confusion_img_url=confusion_img_url,
                    roc_img_url=roc_img_url,
                    scores_img_url=scores_img_url,
                    shap_summary_url=shap_summary_url,
                    shap_bar_url=shap_bar_url,
                    shap_csv_url=shap_csv_url
                )

            except Exception as e:
                print(e)
                flash(f"Error processing dataset: {e}")
                return redirect(request.url)

        else:
            flash("Allowed file type is .csv")

    return render_template("upload.html")


@app.route("/download/model/<path:filename>")
def download_model(filename):
    return send_from_directory(MODELS_FOLDER, filename, as_attachment=True)


@app.route("/download/preprocessed/<path:filename>")
def download_preprocessed(filename):
    return send_from_directory(PREPROCESSED_FOLDER, filename, as_attachment=True)


@app.route("/download/report/<path:filename>")
def download_report(filename):
    return send_from_directory(REPORTS_FOLDER, filename, as_attachment=True)


# optional: route to download shap csv inside static
@app.route("/download/shap/<path:filename>")
def download_shap(filename):
    return send_from_directory(STATIC_FOLDER, filename, as_attachment=True)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0")
```

---

## 7) automl_core.py

> This is the full AutoML engine with Optuna tuning and SHAP explainability. It mirrors the version described earlier in the spec.

```python
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
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix, roc_curve
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

import matplotlib.pyplot as plt
import joblib

# Optional libraries
try:
    from xgboost import XGBClassifier
except Exception:
    XGBClassifier = None

try:
    from lightgbm import LGBMClassifier
except Exception:
    LGBMClassifier = None

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


def get_base_models(random_state: int = 42):
    models = []
    models.append(("LogisticRegression", LogisticRegression(max_iter=1000)))
    models.append(("DecisionTree", DecisionTreeClassifier(random_state=random_state)))
    models.append(("RandomForest", RandomForestClassifier(n_estimators=200, random_state=random_state)))
    models.append(("SVM", SVC(probability=True, random_state=random_state)))
    models.append(("KNN", KNeighborsClassifier()))

    if XGBClassifier is not None:
        models.append(("XGBoost", XGBClassifier(n_estimators=300, learning_rate=0.05, random_state=random_state, use_label_encoder=False, eval_metric="logloss")))

    if LGBMClassifier is not None:
        models.append(("LightGBM", LGBMClassifier(n_estimators=300, learning_rate=0.05, random_state=random_state)))

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


# -------------------------
# Optuna tuning helpers
# -------------------------
def tune_with_optuna(preprocessor: ColumnTransformer, X: pd.DataFrame, y: pd.Series, model_name: str, base_model, n_trials: int = 20, cv: int = 3, random_state: int = 42):
    """
    Returns a model instance with best params found by Optuna (or original base_model if optuna not available).
    Uses cross_val_score(f1_weighted) as objective to maximize.
    """
    if optuna is None:
        print("Optuna not installed — skipping tuning.")
        return base_model

    def objective(trial):
        # Define param spaces per model_name
        params = {}
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
        else:
            # fallback: no tuning for models we don't have a search space for
            clf = base_model

        # Pipeline for CV
        pipe = Pipeline(steps=[("preprocessor", preprocessor), ("model", clf)])
        try:
            scores = cross_val_score(pipe, X, y, cv=cv, scoring="f1_weighted")
            return float(np.mean(scores))
        except Exception as e:
            # return a very low score on failure to guide optuna away
            print(f"Optuna trial error for {model_name}: {e}")
            return 0.0

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)
    best_params = study.best_params
    print(f"Optuna best params for {model_name}: {best_params}")

    # Create new model instance with best params where applicable
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
    else:
        tuned = base_model

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
```

---

## 8) templates/upload.html

```html
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <title>AutoML Model Selector</title>
  <link rel="stylesheet" href="{{ url_for('static', filename='css/style.css') }}">
</head>
<body>
  <div class="container">
    <h1>AutoML Model Selector</h1>
    <p>Upload a CSV file and specify the target column. The app will train and rank multiple models automatically.</p>

    {% with messages = get_flashed_messages() %}
      {% if messages %}
        <ul class="flash-messages">
          {% for msg in messages %}
            <li>{{ msg }}</li>
          {% endfor %}
        </ul>
      {% endif %}
    {% endwith %}

    <form method="POST" enctype="multipart/form-data">
      <div class="form-group">
        <label for="dataset">Dataset (.csv)</label>
        <input type="file" name="dataset" id="dataset" required>
      </div>

      <div class="form-group">
        <label for="target_col">Target Column Name</label>
        <input type="text" name="target_col" id="target_col" placeholder="e.g. label" required>
      </div>

      <div class="form-group">
        <label><input type="checkbox" name="use_optuna"> Enable Optuna Hyperparameter Tuning</label>
      </div>

      <div class="form-group">
        <label for="optuna_trials">Optuna trials (recommended 10-50)</label>
        <input type="number" name="optuna_trials" id="optuna_trials" value="20">
      </div>

      <div class="form-group">
        <label><input type="checkbox" name="compute_shap"> Compute SHAP Explainability (may be slow)</label>
      </div>

      <div class="form-group">
        <label for="shap_max_samples">Max SHAP sample size (recommended ≤200)</label>
        <input type="number" name="shap_max_samples" id="shap_max_samples" value="200">
      </div>

      <button type="submit">Run AutoML</button>
    </form>
  </div>
</body>
</html>
```

---

## 9) templates/results.html

```html
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <title>AutoML Results</title>
  <link rel="stylesheet" href="{{ url_for('static', filename='css/style.css') }}">
</head>
<body>
  <div class="container">
    <h1>AutoML Results</h1>
    <p><strong>Run ID:</strong> {{ results.run_id }}</p>
    <p><strong>Best Model:</strong> {{ results.best_model_name }}</p>

    <h2>Model Leaderboard</h2>
    <table>
      <thead>
        <tr>
          <th>Model</th>
          <th>Accuracy</th>
          <th>F1-score</th>
        </tr>
      </thead>
      <tbody>
        {% for m in results.models_metrics %}
          <tr {% if m.model_name == results.best_model_name %}class="highlight"{% endif %}>
            <td>{{ m.model_name }}</td>
            <td>{{ "%.4f"|format(m.accuracy) }}</td>
            <td>{{ "%.4f"|format(m.f1) }}</td>
          </tr>
        {% endfor %}
      </tbody>
    </table>

    <h2>Plots</h2>

    <div class="plots">
      <div>
        <h3>Model Performance</h3>
        <img src="{{ scores_img_url }}" alt="Model Scores">
      </div>

      <div>
        <h3>Confusion Matrix</h3>
        <img src="{{ confusion_img_url }}" alt="Confusion Matrix">
      </div>

      {% if roc_img_url %}
      <div>
        <h3>ROC Curve (Binary)</h3>
        <img src="{{ roc_img_url }}" alt="ROC Curve">
      </div>
      {% endif %}
    </div>

    {% if shap_summary_url or shap_bar_url %}
      <h2>SHAP Explainability</h2>
      {% if shap_summary_url %}
        <div><h3>SHAP Summary</h3><img src="{{ shap_summary_url }}" alt="SHAP Summary"></div>
      {% endif %}
      {% if shap_bar_url %}
        <div><h3>Feature Importance (mean |SHAP|)</h3><img src="{{ shap_bar_url }}" alt="SHAP Feature Importance"></div>
      {% endif %}

      {% if shap_csv_url %}
        <p><a href="{{ shap_csv_url }}">Download SHAP feature importances (CSV)</a></p>
      {% endif %}
    {% endif %}

    <h2>Downloads</h2>
    <ul>
      <li><a href="{{ model_download_url }}">Download Best Model (.pkl)</a></li>
      <li><a href="{{ preprocessed_download_url }}">Download Preprocessed Dataset (.csv)</a></li>
      <li><a href="{{ report_download_url }}">Download HTML Performance Report</a></li>
    </ul>

    <h2>Classification Report (Text)</h2>
    <pre>{{ results.classification_report_text }}</pre>

    <p><a href="{{ url_for('upload_file') }}">Run another experiment</a></p>
  </div>
</body>
</html>
```

---

## 10) static/css/style.css

```css
body {
  font-family: Arial, sans-serif;
  background: #f7f7f7;
  margin: 0;
  padding: 0;
}

.container {
  max-width: 960px;
  margin: 30px auto;
  background: #fff;
  padding: 20px 30px;
  border-radius: 8px;
  box-shadow: 0 2px 6px rgba(0,0,0,0.1);
}

h1, h2, h3 {
  color: #333;
}

form .form-group {
  margin-bottom: 15px;
}

label {
  display: block;
  margin-bottom: 6px;
}

input[type="file"],
input[type="text"] {
  width: 100%;
  padding: 8px;
  box-sizing: border-box;
}

button {
  padding: 10px 18px;
  border: none;
  border-radius: 4px;
  background: #007bff;
  color: #fff;
  cursor: pointer;
}

button:hover {
  background: #0056b3;
}

table {
  width: 100%;
  border-collapse: collapse;
  margin: 15px 0;
}

th, td {
  padding: 8px;
  border-bottom: 1px solid #ddd;
  text-align: left;
}

tr.highlight {
  background-color: #e8f5e9;
}

.plots {
  display: flex;
  flex-wrap: wrap;
  gap: 20px;
}

.plots img {
  max-width: 300px;
  border: 1px solid #ddd;
  padding: 5px;
  background: #fff;
}

.flash-messages {
  list-style: none;
  padding: 0;
  color: red;
}
```

---

## 11) Final notes & tips

- If you plan to use large datasets and heavy tuning, run this on a machine with multiple CPUs or GPU (XGBoost/GPU builds) and avoid `app.run(debug=True)` in production.
- For long runs, integrate background job processing (Celery + Redis) and show job status updates in the UI. I can provide a Celery integration if you'd like.
- If you want the UI to upload multiple datasets, or support regression tasks, those are natural extensions.

---

That's the full project. If you'd like, I can:

- Convert this into a downloadable zip artifact (not possible directly here, but I can provide a script to generate it), or
- Add Celery + Redis for background jobs, or
- Add a Docker Compose + Nginx + Gunicorn production-ready setup.

Tell me which next step you want.

