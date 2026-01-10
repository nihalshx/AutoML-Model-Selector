# app.py - Enhanced with all features
import os
import json
from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    send_from_directory,
    flash,
    jsonify,
    Response
)
from werkzeug.utils import secure_filename
import pandas as pd
from typing import Optional
import logging
from datetime import datetime
import threading
import time

from automl_core import run_automl_classification, run_automl_regression

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Configuration
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
MODELS_FOLDER = os.path.join(BASE_DIR, "models")
REPORTS_FOLDER = os.path.join(BASE_DIR, "reports")
PREPROCESSED_FOLDER = os.path.join(BASE_DIR, "preprocessed")
STATIC_FOLDER = os.path.join(BASE_DIR, "static")
EXPERIMENTS_FOLDER = os.path.join(BASE_DIR, "experiments")

ALLOWED_EXTENSIONS = {"csv"}
MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB

# Create necessary directories
for folder in [UPLOAD_FOLDER, MODELS_FOLDER, REPORTS_FOLDER, 
               PREPROCESSED_FOLDER, EXPERIMENTS_FOLDER,
               os.path.join(STATIC_FOLDER, "plots")]:
    os.makedirs(folder, exist_ok=True)

app = Flask(__name__, static_folder="static", template_folder="templates")
# Security: Use environment variable for secret key, generate random key as fallback for development only
_secret_key = os.environ.get("FLASK_SECRET")
if not _secret_key:
    import secrets
    _secret_key = secrets.token_hex(32)
    logger.warning("FLASK_SECRET not set! Using randomly generated key. Set FLASK_SECRET environment variable for production.")
app.secret_key = _secret_key
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

# Global dict to track training progress
training_progress = {}


def allowed_file(filename: str) -> bool:
    """Check if file has allowed extension."""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def validate_dataframe(df: pd.DataFrame, target_col: str) -> Optional[str]:
    """
    Validate uploaded dataframe.
    Returns error message if invalid, None if valid.
    """
    if df.empty:
        return "Dataset is empty."
    
    if df.shape[0] < 10:
        return "Dataset must have at least 10 rows."
    
    if target_col not in df.columns:
        return f"Target column '{target_col}' not found. Available columns: {', '.join(df.columns)}"
    
    if df[target_col].isna().all():
        return "Target column contains only missing values."
    
    # Check for too many missing values in target
    missing_pct = df[target_col].isna().sum() / len(df) * 100
    if missing_pct > 50:
        return f"Target column has {missing_pct:.1f}% missing values. Please clean your data."
    
    return None


def validate_data_quality(df: pd.DataFrame, target_col: str) -> list:
    """
    Extended validation with data quality warnings.
    Returns list of warning messages.
    """
    warnings = []
    
    # Check for duplicate rows
    duplicates = df.duplicated().sum()
    if duplicates > 0:
        warnings.append(f"⚠️ Found {duplicates} duplicate rows ({duplicates/len(df)*100:.1f}%)")
    
    # Check for high cardinality categoricals
    for col in df.select_dtypes(include=['object']).columns:
        if col != target_col:
            unique_count = df[col].nunique()
            if unique_count > 50:
                warnings.append(f"⚠️ Column '{col}' has {unique_count} unique values (high cardinality)")
    
    # Check for columns with mostly missing values
    for col in df.columns:
        if col != target_col:
            missing_pct = df[col].isna().sum() / len(df) * 100
            if missing_pct > 80:
                warnings.append(f"⚠️ Column '{col}' has {missing_pct:.1f}% missing values")
    
    # Check for imbalanced target (classification)
    if df[target_col].dtype in ['object', 'category'] or df[target_col].nunique() < 20:
        value_counts = df[target_col].value_counts()
        if len(value_counts) > 1:
            imbalance_ratio = value_counts.min() / value_counts.max()
            if imbalance_ratio < 0.1:
                warnings.append(f"⚠️ Target is highly imbalanced (ratio: {imbalance_ratio:.3f})")
    
    # Check for constant columns
    constant_cols = [col for col in df.columns if col != target_col and df[col].nunique() == 1]
    if constant_cols:
        warnings.append(f"⚠️ Constant columns found: {', '.join(constant_cols)}")
    
    return warnings


def save_experiment_metadata(run_id: str, results: dict, config: dict, df_info: dict):
    """Save experiment details for reproducibility."""
    experiment_log = {
        "run_id": run_id,
        "timestamp": datetime.now().isoformat(),
        "dataset_info": df_info,
        "config": config,
        "results": {
            "best_model": results["best_model_name"],
            "models_tested": len(results["models_metrics"]),
            "metrics": results["models_metrics"]
        }
    }
    
    log_path = os.path.join(EXPERIMENTS_FOLDER, f"{run_id}.json")
    with open(log_path, "w") as f:
        json.dump(experiment_log, f, indent=2)
    
    logger.info(f"Experiment metadata saved: {log_path}")


@app.route("/", methods=["GET", "POST"])
def upload_file():
    """Main route for file upload and AutoML execution."""
    if request.method == "POST":
        try:
            # Validate file upload
            if "dataset" not in request.files:
                flash("No file uploaded. Please select a CSV file.")
                return redirect(request.url)

            file = request.files["dataset"]
            target_col = request.form.get("target_col", "").strip()
            task_type = request.form.get("task_type", "classification")

            if file.filename == "":
                flash("No file selected.")
                return redirect(request.url)

            if not target_col:
                flash("Please provide the target column name.")
                return redirect(request.url)

            if not allowed_file(file.filename):
                flash("Invalid file type. Only CSV files are allowed.")
                return redirect(request.url)

            # Save uploaded file
            filename = secure_filename(file.filename)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{timestamp}_{filename}"
            file_path = os.path.join(UPLOAD_FOLDER, filename)
            file.save(file_path)
            logger.info(f"File uploaded: {filename}")

            # Load and validate dataset
            try:
                df = pd.read_csv(file_path, encoding='utf-8')
            except UnicodeDecodeError:
                df = pd.read_csv(file_path, encoding='latin-1')
            except Exception as e:
                flash(f"Error reading CSV file: {str(e)}")
                return redirect(request.url)

            # Validate dataframe
            validation_error = validate_dataframe(df, target_col)
            if validation_error:
                flash(validation_error)
                return redirect(request.url)

            # Data quality warnings
            quality_warnings = validate_data_quality(df, target_col)
            for warning in quality_warnings:
                flash(warning)

            # Parse form options with safe integer conversion
            use_optuna = request.form.get("use_optuna") == "on"
            try:
                optuna_trials = int(request.form.get("optuna_trials", 20))
            except (ValueError, TypeError):
                optuna_trials = 20
            optuna_trials = max(5, min(optuna_trials, 100))

            compute_shap = request.form.get("compute_shap") == "on"
            try:
                shap_max_samples = int(request.form.get("shap_max_samples", 200))
            except (ValueError, TypeError):
                shap_max_samples = 200
            shap_max_samples = max(50, min(shap_max_samples, 1000))

            feature_engineering = request.form.get("feature_engineering") == "on"

            output_dirs = {
                "models": MODELS_FOLDER,
                "reports": REPORTS_FOLDER,
                "preprocessed": PREPROCESSED_FOLDER,
                "static": STATIC_FOLDER,
            }

            config = {
                "task_type": task_type,
                "use_optuna": use_optuna,
                "optuna_trials": optuna_trials,
                "compute_shap": compute_shap,
                "shap_max_samples": shap_max_samples,
                "feature_engineering": feature_engineering
            }

            df_info = {
                "filename": filename,
                "rows": len(df),
                "columns": len(df.columns),
                "target_column": target_col,
                "feature_names": [col for col in df.columns if col != target_col]
            }

            logger.info(f"Starting AutoML - Task: {task_type}, Optuna: {use_optuna}, SHAP: {compute_shap}")

            # Run AutoML
            if task_type == "regression":
                results = run_automl_regression(
                    df, target_col, output_dirs,
                    use_optuna=use_optuna,
                    optuna_trials=optuna_trials,
                    compute_shap=compute_shap,
                    shap_max_samples=shap_max_samples,
                    feature_engineering=feature_engineering
                )
            else:
                results = run_automl_classification(
                    df, target_col, output_dirs,
                    use_optuna=use_optuna,
                    optuna_trials=optuna_trials,
                    compute_shap=compute_shap,
                    shap_max_samples=shap_max_samples,
                    feature_engineering=feature_engineering
                )

            logger.info(f"AutoML completed - Best model: {results['best_model_name']}")

            # Save experiment metadata
            save_experiment_metadata(results['run_id'], results, config, df_info)

            # Build download URLs
            download_urls = {
                "model": url_for("download_model", filename=results["model_path"]),
                "preprocessed": url_for("download_preprocessed", filename=results["preprocessed_path"]),
                "report": url_for("download_report", filename=results["html_report_filename"])
            }

            # Build image URLs based on task type
            image_urls = {
                "scores": url_for("static", filename=results["scores_plot_path"])
            }

            if task_type == "classification":
                if results.get("confusion_matrix_path"):
                    image_urls["confusion"] = url_for("static", filename=results["confusion_matrix_path"])
                if results.get("roc_curve_path"):
                    image_urls["roc"] = url_for("static", filename=results["roc_curve_path"])
            else:
                if results.get("actual_vs_predicted_path"):
                    image_urls["actual_vs_predicted"] = url_for("static", filename=results["actual_vs_predicted_path"])
                if results.get("residuals_path"):
                    image_urls["residuals"] = url_for("static", filename=results["residuals_path"])

            # SHAP URLs
            shap_urls = {}
            if results.get("shap_summary_path"):
                shap_urls["summary"] = url_for("static", filename=results["shap_summary_path"])
            if results.get("shap_bar_path"):
                shap_urls["bar"] = url_for("static", filename=results["shap_bar_path"])
            if results.get("shap_csv_path"):
                shap_urls["csv"] = url_for("static", filename=results["shap_csv_path"])

            return render_template(
                "results.html",
                results=results,
                task_type=task_type,
                download_urls=download_urls,
                image_urls=image_urls,
                shap_urls=shap_urls,
                quality_warnings=quality_warnings
            )

        except ValueError as e:
            logger.error(f"Validation error: {str(e)}")
            flash(f"Validation error: {str(e)}")
            return redirect(request.url)
        except Exception as e:
            logger.error(f"Error processing dataset: {str(e)}", exc_info=True)
            flash(f"An error occurred while processing your dataset. Please check your data and try again.")
            return redirect(request.url)

    return render_template("upload.html")


@app.route("/api/preview", methods=["POST"])
def preview_dataset():
    """Preview dataset before training."""
    try:
        if "dataset" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400
        
        file = request.files["dataset"]
        if file.filename == "" or not allowed_file(file.filename):
            return jsonify({"error": "Invalid file"}), 400
        
        # Read dataset
        try:
            df = pd.read_csv(file, encoding='utf-8', nrows=100)
        except UnicodeDecodeError:
            file.seek(0)
            df = pd.read_csv(file, encoding='latin-1', nrows=100)
        
        # Get statistics
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        preview_data = {
            "shape": {"rows": len(df), "columns": len(df.columns)},
            "columns": df.columns.tolist(),
            "dtypes": df.dtypes.astype(str).to_dict(),
            "sample": df.head(10).to_dict('records'),
            "missing": df.isnull().sum().to_dict(),
            "numeric_columns": numeric_cols,
            "categorical_columns": categorical_cols,
            "summary": df.describe().to_dict() if numeric_cols else {}
        }
        
        return jsonify(preview_data)
    
    except Exception as e:
        logger.error(f"Preview error: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/columns", methods=["POST"])
def get_columns():
    """API endpoint to get column names from uploaded CSV."""
    try:
        if "dataset" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400
        
        file = request.files["dataset"]
        if file.filename == "" or not allowed_file(file.filename):
            return jsonify({"error": "Invalid file"}), 400
        
        # Read just the header
        df = pd.read_csv(file, nrows=0)
        columns = df.columns.tolist()
        
        return jsonify({"columns": columns})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/experiments")
def list_experiments():
    """List all past experiments."""
    try:
        experiments = []
        
        for filename in os.listdir(EXPERIMENTS_FOLDER):
            if filename.endswith('.json'):
                with open(os.path.join(EXPERIMENTS_FOLDER, filename), 'r') as f:
                    exp = json.load(f)
                    experiments.append({
                        "run_id": exp["run_id"],
                        "timestamp": exp["timestamp"],
                        "best_model": exp["results"]["best_model"],
                        "task_type": exp["config"]["task_type"],
                        "dataset": exp["dataset_info"]["filename"]
                    })
        
        # Sort by timestamp descending
        experiments.sort(key=lambda x: x["timestamp"], reverse=True)
        
        return render_template("experiments.html", experiments=experiments)
    
    except Exception as e:
        logger.error(f"Error listing experiments: {e}")
        flash("Error loading experiments")
        return redirect(url_for("upload_file"))


@app.route("/experiment/<run_id>")
def view_experiment(run_id):
    """View details of a specific experiment."""
    try:
        exp_path = os.path.join(EXPERIMENTS_FOLDER, f"{run_id}.json")
        
        if not os.path.exists(exp_path):
            flash("Experiment not found")
            return redirect(url_for("list_experiments"))
        
        with open(exp_path, 'r') as f:
            experiment = json.load(f)
        
        return render_template("experiment_detail.html", experiment=experiment)
    
    except Exception as e:
        logger.error(f"Error viewing experiment: {e}")
        flash("Error loading experiment details")
        return redirect(url_for("list_experiments"))


def is_safe_path(base_folder: str, filename: str) -> bool:
    """Check if filename is safe and doesn't contain path traversal."""
    # Get the absolute path of the requested file
    requested_path = os.path.abspath(os.path.join(base_folder, filename))
    # Ensure it's within the base folder
    return requested_path.startswith(os.path.abspath(base_folder) + os.sep)


@app.route("/download/model/<path:filename>")
def download_model(filename):
    """Download trained model file."""
    try:
        if not is_safe_path(MODELS_FOLDER, filename):
            logger.warning(f"Path traversal attempt detected: {filename}")
            flash("Invalid file path.")
            return redirect(url_for("upload_file"))
        file_path = os.path.join(MODELS_FOLDER, filename)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Model file not found: {filename}")
        return send_from_directory(MODELS_FOLDER, filename, as_attachment=True)
    except Exception as e:
        logger.error(f"Error downloading model: {str(e)}")
        flash("Model file not found or could not be downloaded.")
        return redirect(url_for("upload_file"))


@app.route("/download/preprocessed/<path:filename>")
def download_preprocessed(filename):
    """Download preprocessed dataset."""
    try:
        if not is_safe_path(PREPROCESSED_FOLDER, filename):
            logger.warning(f"Path traversal attempt detected: {filename}")
            flash("Invalid file path.")
            return redirect(url_for("upload_file"))
        file_path = os.path.join(PREPROCESSED_FOLDER, filename)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Preprocessed file not found: {filename}")
        return send_from_directory(PREPROCESSED_FOLDER, filename, as_attachment=True)
    except Exception as e:
        logger.error(f"Error downloading preprocessed data: {str(e)}")
        flash("Preprocessed file not found or could not be downloaded.")
        return redirect(url_for("upload_file"))


@app.route("/download/report/<path:filename>")
def download_report(filename):
    """Download HTML report."""
    try:
        if not is_safe_path(REPORTS_FOLDER, filename):
            logger.warning(f"Path traversal attempt detected: {filename}")
            flash("Invalid file path.")
            return redirect(url_for("upload_file"))
        file_path = os.path.join(REPORTS_FOLDER, filename)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Report file not found: {filename}")
        return send_from_directory(REPORTS_FOLDER, filename, as_attachment=True)
    except Exception as e:
        logger.error(f"Error downloading report: {str(e)}")
        flash("Report file not found or could not be downloaded.")
        return redirect(url_for("upload_file"))


@app.route("/health")
def health_check():
    """Health check endpoint."""
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0"
    }), 200


@app.errorhandler(404)
def not_found_error(error):
    """Handle 404 errors."""
    flash("Page not found. Redirecting to home page.")
    return redirect(url_for("upload_file"))


@app.errorhandler(413)
def request_entity_too_large(error):
    """Handle file too large error."""
    flash(f"File too large. Maximum size is {MAX_FILE_SIZE // (1024*1024)}MB.")
    return redirect(url_for("upload_file"))


@app.errorhandler(500)
def internal_error(error):
    """Handle internal server errors."""
    logger.error(f"Internal error: {str(error)}", exc_info=True)
    flash("An internal error occurred. Please try again.")
    return redirect(url_for("upload_file"))


@app.errorhandler(Exception)
def handle_exception(error):
    """Handle any unhandled exceptions."""
    logger.error(f"Unhandled exception: {str(error)}", exc_info=True)
    flash("An unexpected error occurred. Please try again.")
    return redirect(url_for("upload_file"))


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("DEBUG", "True").lower() == "true"
    app.run(debug=debug, host="0.0.0.0", port=port)