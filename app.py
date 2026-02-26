# app.py - AutoML Flask Application
# FIXES APPLIED:
#   [#3]  Data quality warnings now flashed to user
#   [#5]  uuid imported at module top (not inline)
#   [#7]  Removed duplicate progress_store; SSE reads from DB only
#   [#10] cleanup_old_files() now auto-runs on startup + admin route
#   [+]   Prediction API validates input columns
#   [+]   SSE endpoint has 10-minute timeout
#   [+]   Input sanitization on target_col

import os
import re
import json
import uuid
import time
import threading
import logging
import logging.handlers
from datetime import datetime, timedelta

import joblib
import pandas as pd
from flask import (
    Flask, render_template, request, redirect, url_for, flash,
    jsonify, send_from_directory, Response
)
from werkzeug.utils import secure_filename

from config import Config
from models_db import db, Experiment, init_db
from preprocessing import (
    validate_dataframe, validate_data_quality, auto_detect_target_column,
    auto_detect_task_type
)
from automl_engine import run_automl

# ── Extensions ──
from flask_caching import Cache
from flask_mail import Mail, Message

# ─── Logging (to file + stdout) ───
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler = logging.handlers.RotatingFileHandler('automl.log', maxBytes=10*1024*1024, backupCount=5)
file_handler.setFormatter(logging.Formatter('%(asctime)s [%(name)s] %(levelname)s: %(message)s'))
logging.getLogger().addHandler(file_handler)
logger = logging.getLogger(__name__)

Config.init_dirs()

app = Flask(__name__, static_folder="static", template_folder="templates")
app.config.from_object(Config)
init_db(app)
cache = Cache(app)
mail  = Mail(app)


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in Config.ALLOWED_EXTENSIONS


def _send_completion_email(exp, results):
    """Send training-complete notification email (non-fatal on failure)."""
    try:
        subject = f"[AutoML] Training complete — {results.get('best_model_name', 'N/A')}"
        body = (
            f"Your AutoML experiment has finished!\n\n"
            f"Run ID     : {exp.id}\n"
            f"Task       : {exp.task_type}\n"
            f"Best Model : {results.get('best_model_name', 'N/A')}\n"
            f"Train Time : {results.get('training_time', 0):.1f}s\n\n"
            f"View results: http://localhost:{Config.PORT}/results/{exp.id}\n"
        )
        msg = Message(subject=subject, recipients=[exp.notify_email], body=body)
        mail.send(msg)
        logger.info(f"Completion email sent to {exp.notify_email}")
    except Exception as e:
        logger.warning(f"Email notification failed: {e}")


def sanitize_column_name(name: str) -> str:
    """Remove potentially dangerous chars from column names."""
    return re.sub(r'[^\w\s\-.]', '', name).strip()[:128]


def progress_callback(run_id, message, percent, extra=None):
    """Write training progress to DB — includes structured data for live visualization."""
    try:
        with app.app_context():
            exp = db.session.get(Experiment, run_id)
            if exp:
                progress_data = {"message": message, "percent": percent}
                if extra:
                    progress_data.update(extra)
                exp.set_progress(progress_data)
                db.session.commit()
    except Exception:
        pass


def run_training_background(run_id, data_path, target_col, task_type, config):
    """
    Run training in a background thread.
    Reads data from disk instead of holding DataFrame in memory.
    """
    with app.app_context():
        exp = db.session.get(Experiment, run_id)
        if not exp:
            return
        
        exp.status = "running"
        exp.started_at = datetime.utcnow()
        db.session.commit()
        
        output_dirs = {
            "models": Config.MODELS_FOLDER,
            "reports": Config.REPORTS_FOLDER,
            "preprocessed": Config.PREPROCESSED_FOLDER,
            "static": Config.STATIC_FOLDER,
        }
        
        try:
            # Re-read from disk instead of holding in memory
            try:
                df = pd.read_csv(data_path, encoding='utf-8')
            except UnicodeDecodeError:
                df = pd.read_csv(data_path, encoding='latin-1')
            
            results = run_automl(
                df=df,
                target_col=target_col,
                task_type=task_type,
                output_dirs=output_dirs,
                run_id=run_id,
                use_optuna=config.get("use_optuna", False),
                optuna_trials=config.get("optuna_trials", 20),
                compute_shap=config.get("compute_shap", False),
                shap_max_samples=config.get("shap_max_samples", 200),
                feature_engineering=config.get("feature_engineering", False),
                use_ensemble=config.get("use_ensemble", False),
                use_smote_flag=config.get("use_smote", False),
                use_class_weight=config.get("use_class_weight", False),
                use_feature_selection=config.get("use_feature_selection", False),
                feature_selection_k=config.get("feature_selection_k", 20),
                use_cv=config.get("use_cv", True),
                cv_folds=config.get("cv_folds", 5),
                plot_learning=config.get("plot_learning", False),
                progress_callback=progress_callback,
            )
            
            exp.status = "completed"
            exp.completed_at = datetime.utcnow()
            exp.best_model_name = results.get("best_model_name")
            exp.model_path = results.get("model_path")
            exp.preprocessed_path = results.get("preprocessed_path")
            exp.report_path = results.get("pdf_report_filename")
            exp.training_time_seconds = results.get("training_time")
            exp.set_results(results)
            # Save drift stats so predict can compare
            if results.get("drift_stats"):
                exp.set_drift_stats(results["drift_stats"])
            db.session.commit()

            # ── Feature: Email Notification ──
            if exp.notify_email:
                _send_completion_email(exp, results)
            
        except Exception as e:
            logger.error(f"Training failed for {run_id}: {e}", exc_info=True)
            exp.status = "failed"
            exp.set_progress({"message": f"Error: {str(e)}", "percent": -1})
            db.session.commit()


# ─────────── Landing & About ───────────

@app.route("/landing")
def landing_page():
    return render_template("landing.html")

@app.route("/about")
def about_page():
    return render_template("about.html")

@app.route("/", methods=["GET", "POST"])
def upload_file():
    """Main upload page."""
    if request.method == "POST":
        try:
            if "dataset" not in request.files:
                flash("No file uploaded.")
                return redirect(request.url)
            
            file = request.files["dataset"]
            target_col = sanitize_column_name(request.form.get("target_col", ""))
            task_type = request.form.get("task_type", "auto")
            
            if file.filename == "":
                flash("No file selected.")
                return redirect(request.url)
            if not allowed_file(file.filename):
                flash("Only CSV files are allowed.")
                return redirect(request.url)
            if not target_col:
                flash("Please provide the target column name.")
                return redirect(request.url)
            
            # Save file
            filename = secure_filename(file.filename)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{ts}_{filename}"
            file_path = os.path.join(Config.UPLOAD_FOLDER, filename)
            # Ensure upload directory exists (guards against init_dirs() failing silently)
            os.makedirs(Config.UPLOAD_FOLDER, exist_ok=True)
            file.save(file_path)
            
            # Load dataset
            try:
                df = pd.read_csv(file_path, encoding='utf-8')
            except UnicodeDecodeError:
                df = pd.read_csv(file_path, encoding='latin-1')
            except Exception as e:
                flash(f"Error reading CSV: {str(e)}")
                return redirect(request.url)

            # ── Feature: Apply Column Mapping from UI ──
            col_mapping_raw = request.form.get("column_mapping", "").strip()
            if col_mapping_raw:
                try:
                    col_mapping = json.loads(col_mapping_raw)
                    if isinstance(col_mapping, dict) and col_mapping:
                        df.rename(columns=col_mapping, inplace=True)
                        # Also update target_col if it was renamed
                        if target_col in col_mapping:
                            target_col = col_mapping[target_col]
                        # !! Re-save to disk so background thread reads renamed columns
                        df.to_csv(file_path, index=False)
                        logger.info(f"Column mapping applied and saved: {col_mapping}")
                except Exception as _me:
                    logger.warning(f"Column mapping parse failed: {_me}")
            
            # Validate
            error = validate_dataframe(df, target_col)
            if error:
                flash(error)
                return redirect(request.url)
            
            # [FIX #3] Data quality warnings now shown to user
            warnings_list = validate_data_quality(df, target_col)
            for w in warnings_list:
                flash(w)
            
            # Auto-detect task type
            if task_type == "auto":
                task_type = auto_detect_task_type(df, target_col)
            
            # Parse config
            def safe_int(val, default, lo, hi):
                try:
                    return max(lo, min(hi, int(val)))
                except (ValueError, TypeError):
                    return default
            
            config = {
                "task_type": task_type,
                "use_optuna": request.form.get("use_optuna") == "on",
                "optuna_trials": safe_int(request.form.get("optuna_trials"), 20, 5, 100),
                "compute_shap": request.form.get("compute_shap") == "on",
                "shap_max_samples": safe_int(request.form.get("shap_max_samples"), 200, 50, 1000),
                "feature_engineering": request.form.get("feature_engineering") == "on",
                "use_ensemble": request.form.get("use_ensemble") == "on",
                "use_smote": request.form.get("use_smote") == "on",
                "use_class_weight": request.form.get("use_class_weight") == "on",
                "use_feature_selection": request.form.get("use_feature_selection") == "on",
                "feature_selection_k": safe_int(request.form.get("feature_selection_k"), 20, 5, 100),
                "use_cv": request.form.get("use_cv") == "on",
                "cv_folds": safe_int(request.form.get("cv_folds"), 5, 2, 10),
                "plot_learning": request.form.get("plot_learning") == "on",
            }
            
            # [FIX #5] uuid imported at top, not inline
            run_id = str(uuid.uuid4())
            exp = Experiment(
                id=run_id, status="pending", task_type=task_type,
                dataset_filename=filename, target_column=target_col,
                num_rows=len(df), num_columns=len(df.columns),
            )
            exp.set_feature_names([c for c in df.columns if c != target_col])
            exp.set_config(config)
            # ── Feature: Email Notification — capture email at upload time ──
            notify_email = request.form.get("notify_email", "").strip()[:256]
            exp.notify_email = notify_email
            db.session.add(exp)
            db.session.commit()
            
            # Start background training — pass file path, NOT dataframe
            t = threading.Thread(target=run_training_background,
                                args=(run_id, file_path, target_col, task_type, config),
                                daemon=True)
            t.start()
            
            return redirect(url_for("training_progress_page", run_id=run_id))
        
        except Exception as e:
            logger.error(f"Upload error: {e}", exc_info=True)
            flash(f"Error: {str(e)}")
            return redirect(request.url)
    
    return render_template("upload.html")


@app.route("/training/<run_id>")
def training_progress_page(run_id):
    """Show real-time training progress."""
    exp = db.session.get(Experiment, run_id)
    if not exp:
        flash("Experiment not found.")
        return redirect(url_for("upload_file"))
    if exp.status == "completed":
        return redirect(url_for("view_results", run_id=run_id))
    return render_template("training.html", run_id=run_id, experiment=exp.to_dict())


@app.route("/api/progress/<run_id>")
def sse_progress(run_id):
    """[FIX #7] SSE reads from DB only (no duplicate in-memory dict). Has 10-min timeout."""
    def generate():
        timeout = time.time() + 600  # 10-minute max
        while time.time() < timeout:
            with app.app_context():
                exp = db.session.get(Experiment, run_id)
                if not exp:
                    yield f"data: {json.dumps({'status': 'error', 'message': 'Not found'})}\n\n"
                    return
                
                if exp.status == "completed":
                    yield f"data: {json.dumps({'message': 'Complete!', 'percent': 100, 'status': 'completed', 'run_id': run_id})}\n\n"
                    return
                elif exp.status == "failed":
                    progress = exp.get_progress()
                    yield f"data: {json.dumps({'message': progress.get('message', 'Failed'), 'percent': -1, 'status': 'failed'})}\n\n"
                    return
                else:
                    progress = exp.get_progress()
                    yield f"data: {json.dumps(progress)}\n\n"
            
            time.sleep(1.5)
        
        yield f"data: {json.dumps({'status': 'timeout', 'message': 'Connection timed out'})}\n\n"
    
    return Response(generate(), mimetype='text/event-stream',
                    headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'})


@app.route("/results/<run_id>")
def view_results(run_id):
    """Display interactive results dashboard."""
    exp = db.session.get(Experiment, run_id)
    if not exp or exp.status != "completed":
        flash("Results not available yet.")
        return redirect(url_for("upload_file"))
    
    results = exp.get_results()
    config = exp.get_config()
    task_type = exp.task_type
    
    download_urls = {
        "model": url_for("download_model", filename=results.get("model_path", "")),
        "preprocessed": url_for("download_preprocessed", filename=results.get("preprocessed_path", "")),
        "report": url_for("download_report", filename=results.get("pdf_report_filename", "")),
    }
    if results.get("pickle_path"):
        download_urls["pickle"] = url_for("download_pickle", filename=results["pickle_path"])
    if results.get("onnx_path"):
        download_urls["onnx"] = url_for("download_onnx", filename=results["onnx_path"])
    
    image_urls = {"scores": url_for("static", filename=results.get("scores_plot_path", ""))}
    for key, result_key in [("radar", "radar_plot_path"), ("cv_scores", "cv_plot_path"),
                             ("learning", "learning_curves_path")]:
        if results.get(result_key):
            image_urls[key] = url_for("static", filename=results[result_key])
    
    if task_type == "classification":
        for key, rk in [("confusion", "confusion_matrix_path"), ("roc", "roc_curve_path")]:
            if results.get(rk):
                image_urls[key] = url_for("static", filename=results[rk])
    else:
        for key, rk in [("actual_vs_predicted", "actual_vs_predicted_path"), ("residuals", "residuals_path")]:
            if results.get(rk):
                image_urls[key] = url_for("static", filename=results[rk])
    
    shap_urls = {}
    for key, rk in [("summary", "shap_summary_path"), ("bar", "shap_bar_path")]:
        if results.get(rk):
            shap_urls[key] = url_for("static", filename=results[rk])
    
    return render_template("results.html", results=results, task_type=task_type,
                           download_urls=download_urls, image_urls=image_urls,
                           shap_urls=shap_urls, config=config, experiment=exp.to_dict())


# ─────────── API ───────────

@app.route("/api/preview", methods=["POST"])
def preview_dataset():
    """Preview dataset columns and stats."""
    try:
        if "dataset" not in request.files:
            return jsonify({"error": "No file"}), 400
        file = request.files["dataset"]
        if not file.filename or not allowed_file(file.filename):
            return jsonify({"error": "Invalid file"}), 400
        
        try:
            df = pd.read_csv(file, encoding='utf-8', nrows=200)
        except UnicodeDecodeError:
            file.seek(0)
            df = pd.read_csv(file, encoding='latin-1', nrows=200)
        
        suggested_target = auto_detect_target_column(df)
        suggested_task = auto_detect_task_type(df, suggested_target) if suggested_target else "classification"
        
        return jsonify({
            "shape": {"rows": len(df), "columns": len(df.columns)},
            "columns": df.columns.tolist(),
            "dtypes": df.dtypes.astype(str).to_dict(),
            "sample": df.head(5).fillna("").to_dict('records'),
            "missing": df.isnull().sum().to_dict(),
            "numeric_columns": df.select_dtypes(include=['int64', 'float64']).columns.tolist(),
            "categorical_columns": df.select_dtypes(include=['object', 'category']).columns.tolist(),
            "suggested_target": suggested_target,
            "suggested_task": suggested_task,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/predict/<run_id>", methods=["POST"])
def predict(run_id):
    """Make predictions — validates input columns match training schema."""
    try:
        exp = db.session.get(Experiment, run_id)
        if not exp or exp.status != "completed":
            return jsonify({"error": "Model not available"}), 404
        
        model_file = os.path.join(Config.MODELS_FOLDER, exp.model_path)
        if not os.path.exists(model_file):
            return jsonify({"error": "Model file not found"}), 404
        
        model_data = joblib.load(model_file)
        pipe = model_data["model"]
        expected_features = model_data.get("feature_names", [])
        
        if request.is_json:
            new_data = pd.DataFrame(request.json.get("data", []))
        elif "file" in request.files:
            new_data = pd.read_csv(request.files["file"])
        else:
            return jsonify({"error": "Provide JSON data or CSV file"}), 400
        
        # Validate input columns match expected features
        if expected_features:
            missing = set(expected_features) - set(new_data.columns)
            if missing:
                return jsonify({"error": f"Missing columns: {sorted(missing)}"}), 400
            new_data = new_data[expected_features]  # enforce order + drop extras
        
        # Handle pretransformed models (SMOTE/feature selection)
        if model_data.get("pretransformed") and model_data.get("preprocessor"):
            new_data_t = model_data["preprocessor"].transform(new_data)
            predictions = pipe.predict(new_data_t)
        else:
            predictions = pipe.predict(new_data)
        
        result = {"predictions": predictions.tolist()}
        if hasattr(pipe, "predict_proba"):
            try:
                if model_data.get("pretransformed") and model_data.get("preprocessor"):
                    probs = pipe.predict_proba(new_data_t)
                else:
                    probs = pipe.predict_proba(new_data)
                result["probabilities"] = probs.tolist()
            except Exception:
                pass
        
        le = model_data.get("label_encoder")
        if le:
            result["labels"] = le.inverse_transform(predictions.astype(int)).tolist()

        # ── Feature: Data Drift Detection ──
        drift_stats = exp.get_drift_stats()
        if drift_stats:
            drift_warnings = _check_drift(new_data, drift_stats)
            result["drift_warnings"] = drift_warnings
            if drift_warnings:
                result["drift_detected"] = True
                logger.warning(f"Drift detected for {run_id}: {drift_warnings}")
        
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ─────────── Experiments ───────────

@app.route("/experiments")
def list_experiments():
    experiments = Experiment.query.order_by(Experiment.created_at.desc()).all()
    return render_template("experiments.html", experiments=[e.to_dict() for e in experiments])

@app.route("/experiment/<run_id>")
def view_experiment(run_id):
    exp = db.session.get(Experiment, run_id)
    if not exp:
        flash("Experiment not found.")
        return redirect(url_for("list_experiments"))
    return render_template("experiment_detail.html", experiment=exp.to_dict())

@app.route("/experiment/<run_id>/delete", methods=["POST"])
def delete_experiment(run_id):
    exp = db.session.get(Experiment, run_id)
    if exp:
        for folder, fname in [(Config.MODELS_FOLDER, exp.model_path),
                                (Config.PREPROCESSED_FOLDER, exp.preprocessed_path),
                                (Config.REPORTS_FOLDER, exp.report_path)]:
            if fname:
                p = os.path.join(folder, fname)
                if os.path.exists(p):
                    os.remove(p)
        db.session.delete(exp)
        db.session.commit()
        flash("Experiment deleted.")
    return redirect(url_for("list_experiments"))


# ─────────── 3D Visualizer ───────────

@app.route("/visualizer")
def neural_visualizer():
    return render_template("visualizer.html")

@app.route("/api/experiments/list")
def api_list_experiments():
    """JSON list of experiments for the live visualizer."""
    exps = Experiment.query.order_by(Experiment.created_at.desc()).limit(20).all()
    return jsonify([{
        "id": e.id, "status": e.status, "task_type": e.task_type,
        "best_model": e.best_model_name, "dataset": e.dataset_filename,
        "rows": e.num_rows, "cols": e.num_columns,
        "created": e.created_at.isoformat() if e.created_at else None,
        "time": e.training_time_seconds,
    } for e in exps])


@app.route("/api/experiment/<run_id>/results")
def api_experiment_results(run_id):
    """Full results JSON for the visualizer to render completed experiments."""
    try:
        exp = db.session.get(Experiment, run_id)
        if not exp:
            return jsonify({"error": "Not found"}), 404

        results = exp.get_results()
        metrics = results.get("models_metrics", [])

        # Build the same structure that the live SSE sends during training
        model_names = []
        completed_models = []
        for m in metrics:
            name = m.get("model_name") or m.get("name")
            if not name:
                continue
            model_names.append(name)
            completed_models.append({
                "name": name,
                "primary_metric": round(float(m.get("f1", m.get("r2", 0)) or 0), 4),
                "metrics": {k: round(v, 4) if isinstance(v, float) else v
                            for k, v in m.items() if k not in ("cv_scores",)}
            })

        return jsonify({
            "status": exp.status,
            "task_type": exp.task_type,
            "best_model": exp.best_model_name,
            "training_time": exp.training_time_seconds,
            "model_names": model_names,
            "completed_models": completed_models,
            "total_models": len(model_names),
            "dataset_info": {
                "filename": exp.dataset_filename,
                "rows": exp.num_rows,
                "cols": exp.num_columns,
            },
        })
    except Exception as e:
        logger.error(f"api_experiment_results error for {run_id}: {e}", exc_info=True)
        return jsonify({"error": f"Server error: {str(e)}"}), 500


# ─────────── Downloads ───────────

def _safe_download(folder, filename):
    if not filename:
        flash("File not available.")
        return redirect(url_for("upload_file"))
    abs_folder = os.path.abspath(folder)
    abs_path = os.path.abspath(os.path.join(folder, filename))
    if not abs_path.startswith(abs_folder + os.sep):
        flash("Invalid path.")
        return redirect(url_for("upload_file"))
    if not os.path.exists(abs_path):
        flash("File not found.")
        return redirect(url_for("upload_file"))
    return send_from_directory(folder, filename, as_attachment=True)

@app.route("/download/model/<path:filename>")
def download_model(filename):
    return _safe_download(Config.MODELS_FOLDER, filename)

@app.route("/download/preprocessed/<path:filename>")
def download_preprocessed(filename):
    return _safe_download(Config.PREPROCESSED_FOLDER, filename)

@app.route("/download/report/<path:filename>")
def download_report(filename):
    return _safe_download(Config.REPORTS_FOLDER, filename)


# ── Feature: Pickle / ONNX Downloads ──

@app.route("/download/pickle/<path:filename>")
def download_pickle(filename):
    return _safe_download(Config.MODELS_FOLDER, filename)

@app.route("/download/onnx/<path:filename>")
def download_onnx(filename):
    return _safe_download(Config.MODELS_FOLDER, filename)


# ── Feature: Tagging & Notes ──

@app.route("/experiment/<run_id>/tags", methods=["POST"])
def update_tags(run_id):
    exp = db.session.get(Experiment, run_id)
    if not exp:
        return jsonify({"error": "Not found"}), 404
    data = request.get_json(force=True, silent=True) or {}
    raw_tags = data.get("tags", "")
    tags = [t.strip() for t in raw_tags.split(",") if t.strip()]
    exp.set_tags(tags)
    exp.notes = (data.get("notes", exp.notes or ""))[:2000]
    db.session.commit()
    return jsonify({"tags": exp.get_tags(), "notes": exp.notes})


# ── Feature: Re-run with New Config ──

@app.route("/experiment/<run_id>/rerun")
def rerun_experiment(run_id):
    exp = db.session.get(Experiment, run_id)
    if not exp:
        flash("Experiment not found.")
        return redirect(url_for("list_experiments"))
    return render_template("upload.html", prefill=exp.to_dict())


# ── Feature: Correlation Heatmap API ──

@app.route("/api/correlation", methods=["POST"])
def correlation_heatmap():
    try:
        file = request.files.get("dataset")
        if not file:
            return jsonify({"error": "No file"}), 400
        try:
            df = pd.read_csv(file, nrows=500, encoding="utf-8")
        except UnicodeDecodeError:
            file.seek(0)
            df = pd.read_csv(file, nrows=500, encoding="latin-1")
        numeric = df.select_dtypes(include=["int64", "float64"])
        if numeric.shape[1] < 2:
            return jsonify({"error": "Need at least 2 numeric columns"}), 400
        corr = numeric.corr().round(3)
        return jsonify({"columns": corr.columns.tolist(), "matrix": corr.values.tolist()})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── Feature: Data Drift Detection helper ──

def _check_drift(new_data: pd.DataFrame, training_stats: dict, z_threshold: float = 3.0):
    warnings_out = []
    for col, stats in training_stats.items():
        if col not in new_data.columns:
            continue
        vals = new_data[col].dropna()
        if len(vals) == 0:
            continue
        mean_diff = abs(vals.mean() - stats["mean"])
        if stats["std"] > 0 and mean_diff / stats["std"] > z_threshold:
            warnings_out.append(
                f"'{col}': mean shifted by {mean_diff:.2f} "
                f"(train mean={stats['mean']:.2f}, σ={stats['std']:.2f})"
            )
        out_of_range = int(((vals < stats["min"]) | (vals > stats["max"])).sum())
        if out_of_range > 0:
            warnings_out.append(
                f"'{col}': {out_of_range} value(s) outside training range "
                f"[{stats['min']:.2f}, {stats['max']:.2f}]"
            )
    return warnings_out


# ── Feature: Dataset Comparison ──

@app.route("/compare", methods=["GET"])
def compare_datasets():
    exps = Experiment.query.filter_by(status="completed").order_by(Experiment.created_at.desc()).all()
    return render_template("compare.html", experiments=[e.to_dict() for e in exps])

@app.route("/api/compare", methods=["POST"])
def api_compare():
    """Train a second model on a new dataset using the same config as a base experiment."""
    try:
        base_id = request.form.get("base_exp_id", "")
        base_exp = db.session.get(Experiment, base_id) if base_id else None

        if "dataset" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400
        file = request.files["dataset"]
        if not file.filename or not allowed_file(file.filename):
            return jsonify({"error": "Invalid file"}), 400

        try:
            df = pd.read_csv(file, encoding="utf-8")
        except UnicodeDecodeError:
            file.seek(0)
            df = pd.read_csv(file, encoding="latin-1")

        # Derive config from base experiment or form
        if base_exp:
            config = base_exp.get_config()
            target_col = base_exp.target_column
            task_type  = base_exp.task_type
        else:
            target_col = sanitize_column_name(request.form.get("target_col", ""))
            task_type  = request.form.get("task_type", "auto")
            config = {"task_type": task_type, "use_cv": True, "cv_folds": 5,
                      "use_optuna": False, "optuna_trials": 20, "compute_shap": False,
                      "shap_max_samples": 200, "feature_engineering": False,
                      "use_ensemble": False, "use_smote": False,
                      "use_class_weight": False, "use_feature_selection": False,
                      "feature_selection_k": 20, "plot_learning": False}

        if not target_col or target_col not in df.columns:
            return jsonify({"error": f"Target column '{target_col}' not found"}), 400

        err = validate_dataframe(df, target_col)
        if err:
            return jsonify({"error": err}), 400
        if task_type == "auto":
            task_type = auto_detect_task_type(df, target_col)

        filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{secure_filename(file.filename)}"
        file_path = os.path.join(Config.UPLOAD_FOLDER, filename)
        os.makedirs(Config.UPLOAD_FOLDER, exist_ok=True)
        df.to_csv(file_path, index=False)

        run_id = str(uuid.uuid4())
        exp = Experiment(
            id=run_id, status="pending", task_type=task_type,
            dataset_filename=filename, target_column=target_col,
            num_rows=len(df), num_columns=len(df.columns),
        )
        exp.set_feature_names([c for c in df.columns if c != target_col])
        exp.set_config(config)
        # Tag as a comparison run
        exp.set_tags(["comparison", f"base:{base_id[:8]}"] if base_exp else ["comparison"])
        exp.notes = f"Comparison run against base experiment {base_id}" if base_exp else "Comparison run"
        db.session.add(exp)
        db.session.commit()

        t = threading.Thread(target=run_training_background,
                             args=(run_id, file_path, target_col, task_type, config),
                             daemon=True)
        t.start()
        return jsonify({"run_id": run_id, "redirect": url_for("training_progress_page", run_id=run_id)})
    except Exception as e:
        logger.error(f"Compare error: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


# ─────────── [FIX #10] Cleanup — now actually runs ───────────

def cleanup_old_files():
    """Remove experiments older than retention period."""
    with app.app_context():
        cutoff = datetime.utcnow() - timedelta(days=Config.FILE_RETENTION_DAYS)
        old = Experiment.query.filter(Experiment.created_at < cutoff).all()
        count = 0
        for exp in old:
            for folder, fname in [(Config.MODELS_FOLDER, exp.model_path),
                                   (Config.PREPROCESSED_FOLDER, exp.preprocessed_path),
                                   (Config.REPORTS_FOLDER, exp.report_path)]:
                if fname:
                    p = os.path.join(folder, fname)
                    if os.path.exists(p):
                        try:
                            os.remove(p)
                        except OSError:
                            pass
            db.session.delete(exp)
            count += 1
        db.session.commit()
        if count:
            logger.info(f"Cleaned up {count} old experiments")

@app.route("/admin/cleanup", methods=["POST"])
def trigger_cleanup():
    """Manual cleanup trigger."""
    cleanup_old_files()
    flash("Cleanup completed.")
    return redirect(url_for("list_experiments"))


# ─────────── Health ───────────

@app.route("/health")
def health():
    """Health check with real diagnostics."""
    checks = {}
    try:
        db.session.execute(db.text("SELECT 1"))
        checks["database"] = "ok"
    except Exception:
        checks["database"] = "error"
    
    import shutil
    disk = shutil.disk_usage("/")
    checks["disk_free_gb"] = round(disk.free / (1024**3), 1)
    checks["active_jobs"] = Experiment.query.filter_by(status="running").count()
    
    status = "healthy" if checks["database"] == "ok" else "unhealthy"
    return jsonify({"status": status, "timestamp": datetime.now().isoformat(), **checks}), 200 if status == "healthy" else 503


@app.errorhandler(404)
def not_found(e):
    flash("Page not found.")
    return redirect(url_for("upload_file"))

@app.errorhandler(413)
def too_large(e):
    flash(f"File too large. Max {Config.MAX_CONTENT_LENGTH // (1024*1024)}MB.")
    return redirect(url_for("upload_file"))

@app.errorhandler(500)
def internal_error(e):
    flash("Internal error. Please try again.")
    return redirect(url_for("upload_file"))


if __name__ == "__main__":
    # Run cleanup on startup
    cleanup_old_files()
    app.run(debug=Config.DEBUG, host="0.0.0.0", port=Config.PORT)

# [2026-02-05T11:00:00] Initialize Flask app with basic routing

# [2026-02-26T11:00:00] Implement dataset upload and validation endpoint

# [2026-03-12T14:00:00] Implement REST API endpoints for model training

# [2026-02-05T11:00:00] Initialize Flask app with basic routing

# [2026-02-26T11:00:00] Implement dataset upload and validation endpoint
