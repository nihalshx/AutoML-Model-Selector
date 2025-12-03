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

from automl_core import run_automl_classification, run_automl_regression

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
        task_type = request.form.get("task_type", "classification")

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

                if task_type == "regression":
                    results = run_automl_regression(
                        df, target_col,
                        output_dirs,
                        use_optuna=use_optuna,
                        optuna_trials=optuna_trials,
                        compute_shap=compute_shap,
                        shap_max_samples=shap_max_samples
                    )
                else:
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

                # Handle images based on task type
                confusion_img_url = None
                roc_img_url = None
                actual_vs_predicted_img_url = None
                residuals_img_url = None

                if task_type == "classification":
                    if results.get("confusion_matrix_path"):
                        confusion_img_url = url_for("static", filename=results["confusion_matrix_path"])
                    if results.get("roc_curve_path"):
                        roc_img_url = url_for("static", filename=results["roc_curve_path"])
                else:
                    if results.get("actual_vs_predicted_path"):
                        actual_vs_predicted_img_url = url_for("static", filename=results["actual_vs_predicted_path"])
                    if results.get("residuals_path"):
                        residuals_img_url = url_for("static", filename=results["residuals_path"])

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
                    task_type=task_type,
                    model_download_url=model_download_url,
                    preprocessed_download_url=preprocessed_download_url,
                    report_download_url=report_download_url,
                    confusion_img_url=confusion_img_url,
                    roc_img_url=roc_img_url,
                    actual_vs_predicted_img_url=actual_vs_predicted_img_url,
                    residuals_img_url=residuals_img_url,
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
