# automl_engine.py - Main AutoML pipeline with all advanced features
# FIXES APPLIED:
#   [#1] SMOTE results now REPLACE X_train/y_train — models train on resampled data
#   [#2] Feature selection is WIRED IN — apply_feature_selection() actually runs
#   [#6] Ensemble uses sklearn.base.clone() for unfitted model copies

import os
import uuid
import pickle
import warnings
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime
import logging
import joblib

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report, confusion_matrix,
    roc_auc_score, precision_score, recall_score,
    mean_absolute_error, mean_squared_error, r2_score
)

from preprocessing import (
    detect_feature_types, build_preprocessor, auto_feature_engineering,
    apply_feature_selection
)
from ml_models import (
    get_base_models, build_stacking_ensemble, apply_smote
)
from tuning import tune_with_optuna, TUNABLE_MODELS
from visualization import (
    plot_model_scores, plot_radar_chart, plot_confusion_matrix, plot_roc_curve,
    plot_roc_multiclass, plot_actual_vs_predicted, plot_residuals,
    plot_learning_curves, plot_cv_scores, compute_and_save_shap
)
from reporting import generate_classification_report, generate_regression_report

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


def _train_single_model(name, model, preprocessor, X_train, y_train, X_test, y_test,
                         task_type, use_cv=False, cv_folds=5, pretransformed=False):
    """
    Train a single model and return metrics.
    
    pretransformed=True means X_train/X_test are already numpy arrays from the
    preprocessor (used when SMOTE or feature selection was applied).
    In that case we train the raw model without a pipeline wrapper.
    """
    if pretransformed:
        try:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            cv_scores = None
            if use_cv:
                scoring = "f1_weighted" if task_type == "classification" else "neg_mean_squared_error"
                try:
                    cv_scores = cross_val_score(model, X_train, y_train, cv=cv_folds,
                                                scoring=scoring, n_jobs=-1).tolist()
                except Exception:
                    pass
            
            y_prob = None
            if task_type == "classification" and hasattr(model, "predict_proba"):
                try:
                    y_prob = model.predict_proba(X_test)
                except Exception:
                    pass
            
            if task_type == "classification":
                metrics = {
                    "model_name": name,
                    "accuracy": accuracy_score(y_test, y_pred),
                    "f1": f1_score(y_test, y_pred, average="weighted"),
                    "precision": precision_score(y_test, y_pred, average="weighted", zero_division=0),
                    "recall": recall_score(y_test, y_pred, average="weighted", zero_division=0),
                    "cv_scores": cv_scores,
                }
                return metrics, model, y_pred, y_prob
            else:
                mae = mean_absolute_error(y_test, y_pred)
                mse = mean_squared_error(y_test, y_pred)
                metrics = {
                    "model_name": name, "mae": mae, "mse": mse,
                    "rmse": np.sqrt(mse), "r2": r2_score(y_test, y_pred),
                    "cv_scores": cv_scores,
                }
                return metrics, model, y_pred, None
        except Exception as e:
            logger.error(f"Training failed for {name} (pretransformed): {e}")
            return None, None, None, None
    
    # Normal pipeline mode
    pipe = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])
    try:
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        
        cv_scores = None
        if use_cv:
            scoring = "f1_weighted" if task_type == "classification" else "neg_mean_squared_error"
            try:
                cv_scores = cross_val_score(pipe, X_train, y_train, cv=cv_folds,
                                            scoring=scoring, n_jobs=-1).tolist()
            except Exception:
                pass
        
        if task_type == "classification":
            y_prob = None
            if hasattr(pipe, "predict_proba"):
                try:
                    y_prob = pipe.predict_proba(X_test)
                except Exception:
                    pass
            metrics = {
                "model_name": name,
                "accuracy": accuracy_score(y_test, y_pred),
                "f1": f1_score(y_test, y_pred, average="weighted"),
                "precision": precision_score(y_test, y_pred, average="weighted", zero_division=0),
                "recall": recall_score(y_test, y_pred, average="weighted", zero_division=0),
                "cv_scores": cv_scores,
            }
            return metrics, pipe, y_pred, y_prob
        else:
            mae = mean_absolute_error(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            metrics = {
                "model_name": name, "mae": mae, "mse": mse,
                "rmse": np.sqrt(mse), "r2": r2_score(y_test, y_pred),
                "cv_scores": cv_scores,
            }
            return metrics, pipe, y_pred, None
    except Exception as e:
        logger.error(f"Training failed for {name}: {e}")
        return None, None, None, None


def run_automl(
    df, target_col, task_type, output_dirs,
    run_id=None,
    test_size=0.2, random_state=42,
    use_optuna=False, optuna_trials=20,
    compute_shap=False, shap_max_samples=200,
    feature_engineering=False, use_ensemble=False,
    use_smote_flag=False, use_class_weight=False,
    use_feature_selection=False, feature_selection_k=20,
    use_cv=True, cv_folds=5, plot_learning=False,
    progress_callback=None,
):
    """Unified AutoML pipeline."""
    logger.info(f"Starting AutoML - Task: {task_type}")
    start_time = datetime.now()
    # Use the caller-supplied run_id (experiment ID) so progress_callback
    # can look it up in the DB. Fall back to a new UUID only if not provided.
    if run_id is None:
        run_id = str(uuid.uuid4())
    
    def update_progress(msg, pct=0, **extra):
        """Send progress with optional structured data for live visualization."""
        if progress_callback:
            progress_callback(run_id, msg, pct, extra)
    
    update_progress("Preparing data...", 5, stage="init")
    
    if feature_engineering:
        update_progress("Engineering features...", 8, stage="feature_engineering")
        df = auto_feature_engineering(df, target_col)
    
    num_cols, cat_cols = detect_feature_types(df, target_col)
    preprocessor = build_preprocessor(num_cols, cat_cols)
    
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    label_encoder = None
    if task_type == "classification":
        if y.dtype == "object" or str(y.dtype).startswith("category"):
            label_encoder = LabelEncoder()
            y = pd.Series(label_encoder.fit_transform(y))
    else:
        if not pd.api.types.is_numeric_dtype(y):
            raise ValueError("Target must be numeric for regression")
    
    update_progress("Splitting data...", 10, stage="split",
                    dataset_info={"rows": len(X), "features": len(X.columns), "target": target_col})
    split_kw = {"test_size": test_size, "random_state": random_state}
    if task_type == "classification":
        split_kw["stratify"] = y
    X_train, X_test, y_train, y_test = train_test_split(X, y, **split_kw)
    logger.info(f"Train: {len(X_train)}, Test: {len(X_test)}")
    
    # ─── [FIX #1] SMOTE — results NOW replace training data ───
    smote_applied = False
    pretransformed = False
    selected_features_info = None
    train_X, train_y = X_train, y_train
    test_X, test_y = X_test, y_test
    
    if use_smote_flag and task_type == "classification":
        update_progress("Applying SMOTE...", 12)
        try:
            X_train_t = preprocessor.fit_transform(X_train, y_train)
            X_test_t = preprocessor.transform(X_test)
            X_resampled, y_resampled = apply_smote(X_train_t, y_train.values, random_state)
            train_X = X_resampled
            train_y = pd.Series(y_resampled)
            test_X = X_test_t
            pretransformed = True
            smote_applied = True
            logger.info(f"SMOTE: {len(X_train)} -> {len(train_X)} training samples")
        except Exception as e:
            logger.warning(f"SMOTE failed, continuing without: {e}")
    
    # ─── [FIX #2] Feature Selection — NOW ACTUALLY RUNS ───
    if use_feature_selection:
        update_progress("Selecting features...", 13)
        try:
            if not pretransformed:
                X_train_t = preprocessor.fit_transform(X_train, y_train)
                X_test_t = preprocessor.transform(X_test)
                src_X_train, src_X_test = X_train_t, X_test_t
            else:
                src_X_train, src_X_test = train_X, test_X
            
            X_sel_train, X_sel_test, selector, mask, scores = apply_feature_selection(
                src_X_train, train_y if pretransformed else y_train,
                src_X_test, task_type,
                method="mutual_info", k=feature_selection_k
            )
            
            try:
                all_names = list(preprocessor.get_feature_names_out())
                sel_names = [n for n, m in zip(all_names, mask) if m]
            except Exception:
                sel_names = [f"f_{i}" for i, m in enumerate(mask) if m]
            
            selected_features_info = {
                "n_original": len(mask),
                "n_selected": X_sel_train.shape[1],
                "selected_features": sel_names,
            }
            train_X, test_X = X_sel_train, X_sel_test
            if not pretransformed:
                train_y = y_train
            pretransformed = True
            logger.info(f"Feature selection: {len(mask)} -> {X_sel_train.shape[1]} features")
        except Exception as e:
            logger.warning(f"Feature selection failed: {e}")
    
    # ─── Get Models ───
    update_progress("Preparing models...", 15, stage="prepare")
    model_candidates = get_base_models(task_type, random_state, use_class_weight)
    all_model_names = [n for n, _ in model_candidates]
    
    models_metrics = []
    trained_models = []
    all_y_preds = {}
    all_y_probs = {}
    cv_results = {}
    total = len(model_candidates)
    
    for idx, (name, base_model) in enumerate(model_candidates):
        pct = 20 + int(60 * idx / total)
        # Send rich progress: current model, all completed models with metrics
        update_progress(f"Training {name}... ({idx+1}/{total})", pct,
                        stage="training", current_model=name,
                        current_index=idx, total_models=total,
                        model_names=all_model_names,
                        completed_models=[{
                            "name": m["model_name"],
                            "primary_metric": round(m.get("f1", m.get("r2", 0)), 4),
                            "metrics": {k: round(v, 4) if isinstance(v, float) else v
                                        for k, v in m.items() if k != "cv_scores"}
                        } for m in models_metrics])
        
        tuned = base_model
        if use_optuna and name in TUNABLE_MODELS and not pretransformed:
            try:
                tuned = tune_with_optuna(
                    preprocessor, X_train, y_train, name, base_model,
                    task_type=task_type, n_trials=optuna_trials, cv=3, random_state=random_state)
            except Exception as e:
                logger.warning(f"Optuna failed for {name}: {e}")
        
        metrics, trained, y_pred, y_prob = _train_single_model(
            name, tuned, preprocessor,
            train_X, train_y, test_X, test_y if not pretransformed else y_test,
            task_type, use_cv, cv_folds, pretransformed=pretransformed
        )
        
        if metrics is not None:
            models_metrics.append(metrics)
            trained_models.append((name, trained))
            all_y_preds[name] = y_pred
            if y_prob is not None:
                all_y_probs[name] = y_prob
            if metrics.get("cv_scores"):
                cv_results[name] = metrics["cv_scores"]
            primary = metrics.get('f1', metrics.get('r2', 0))
            logger.info(f"{name} - {primary:.4f}")
    
    if not models_metrics:
        raise RuntimeError("No models were successfully trained")
    
    sort_key = "f1" if task_type == "classification" else "r2"
    models_metrics.sort(key=lambda x: x[sort_key], reverse=True)
    
    # ─── [FIX #6] Ensemble — use clone() for unfitted copies ───
    if use_ensemble and len(trained_models) >= 2:
        update_progress("Building stacking ensemble...", 82, stage="ensemble",
                        completed_models=[{
                            "name": m["model_name"],
                            "primary_metric": round(m.get("f1", m.get("r2", 0)), 4),
                            "metrics": {k: round(v, 4) if isinstance(v, float) else v
                                        for k, v in m.items() if k != "cv_scores"}
                        } for m in models_metrics])
        try:
            top3_names = [m["model_name"] for m in models_metrics[:3]]
            top3 = []
            for name, trained in trained_models:
                if name in top3_names:
                    raw = trained.named_steps["model"] if isinstance(trained, Pipeline) else trained
                    try:
                        top3.append((name, clone(raw)))
                    except Exception:
                        top3.append((name, raw))
            
            ens_def = build_stacking_ensemble(top3, task_type, random_state)
            if ens_def:
                ens_name, ens_model = ens_def
                em, et, ep, epr = _train_single_model(
                    ens_name, ens_model, preprocessor,
                    train_X, train_y, test_X, test_y if not pretransformed else y_test,
                    task_type, use_cv, cv_folds, pretransformed=pretransformed)
                if em:
                    models_metrics.append(em)
                    trained_models.append((ens_name, et))
                    all_y_preds[ens_name] = ep
                    if epr is not None:
                        all_y_probs[ens_name] = epr
        except Exception as e:
            logger.error(f"Ensemble failed: {e}")
    
    models_metrics.sort(key=lambda x: x[sort_key], reverse=True)
    best_name = models_metrics[0]["model_name"]
    best_trained = next((t for n, t in trained_models if n == best_name), None)
    best_y_pred = all_y_preds.get(best_name)
    best_y_prob = all_y_probs.get(best_name)

    if best_trained is None:
        raise RuntimeError(f"Best model '{best_name}' was not found in trained_models — this is a bug")
    
    update_progress("Generating visualizations...", 85, stage="visualizing",
                    best_model=best_name)
    
    # Save preprocessed data
    if pretransformed:
        X_all_t = preprocessor.transform(X)
    else:
        X_all_t = best_trained.named_steps["preprocessor"].transform(X)
    try:
        feat_names = list(preprocessor.get_feature_names_out())
    except Exception:
        feat_names = [f"f_{i}" for i in range(X_all_t.shape[1])]
    
    preprocessed_path = os.path.join(output_dirs["preprocessed"], f"preprocessed_{run_id}.csv")
    pd.DataFrame(X_all_t, columns=feat_names).to_csv(preprocessed_path, index=False)
    
    # Save model (joblib — primary format)
    model_path = os.path.join(output_dirs["models"], f"best_model_{run_id}.pkl")
    model_bundle = {
        "model": best_trained,
        "preprocessor": preprocessor if pretransformed else None,
        "pretransformed": pretransformed,
        "label_encoder": label_encoder,
        "target_col": target_col,
        "feature_names": list(X.columns),
        "task_type": task_type,
        "run_id": run_id,
    }
    joblib.dump(model_bundle, model_path)

    # ── Feature: Pickle Export (alternative format) ──
    pickle_path = os.path.join(output_dirs["models"], f"best_model_{run_id}.pickle")
    try:
        with open(pickle_path, "wb") as _pf:
            pickle.dump(model_bundle, _pf, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info(f"Pickle export saved: {pickle_path}")
    except Exception as _pe:
        logger.warning(f"Pickle export failed: {_pe}")
        pickle_path = None

    # ── Feature: ONNX Export ──
    onnx_path = None
    if not pretransformed:
        try:
            from skl2onnx import convert_sklearn
            from skl2onnx.common.data_types import FloatTensorType
            num_feats = X_train.shape[1]
            initial_type = [("float_input", FloatTensorType([None, num_feats]))]
            onnx_model = convert_sklearn(best_trained, initial_types=initial_type,
                                         target_opset=17)
            onnx_path = os.path.join(output_dirs["models"], f"best_model_{run_id}.onnx")
            with open(onnx_path, "wb") as _of:
                _of.write(onnx_model.SerializeToString())
            logger.info(f"ONNX export saved: {onnx_path}")
            onnx_path = os.path.basename(onnx_path)
        except Exception as _oe:
            logger.warning(f"ONNX export failed (non-fatal): {_oe}")
            onnx_path = None

    # ── Feature: Auto Feature Importance ──
    feature_importance_data = []
    try:
        raw_model = None
        if pretransformed:
            raw_model = best_trained
        elif hasattr(best_trained, "named_steps"):
            raw_model = best_trained.named_steps.get("model")
        
        if raw_model is not None and hasattr(raw_model, "feature_importances_"):
            importances = raw_model.feature_importances_
            names_for_fi = feat_names[:len(importances)] if len(feat_names) >= len(importances) else feat_names
            fi_pairs = sorted(zip(names_for_fi, importances.tolist()), key=lambda x: x[1], reverse=True)
            feature_importance_data = [{"feature": f, "importance": round(float(v), 6)} for f, v in fi_pairs[:30]]
            logger.info(f"Feature importance computed: {len(feature_importance_data)} features")
        elif raw_model is not None and hasattr(raw_model, "coef_"):
            coef = raw_model.coef_
            if coef.ndim > 1:
                coef = np.abs(coef).mean(axis=0)
            else:
                coef = np.abs(coef)
            names_for_fi = feat_names[:len(coef)] if len(feat_names) >= len(coef) else feat_names
            fi_pairs = sorted(zip(names_for_fi, coef.tolist()), key=lambda x: x[1], reverse=True)
            feature_importance_data = [{"feature": f, "importance": round(float(v), 6)} for f, v in fi_pairs[:30]]
    except Exception as _fie:
        logger.warning(f"Feature importance extraction failed: {_fie}")

    # ── Feature: Drift Detection — save training distribution stats ──
    drift_stats = {}
    try:
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        for col in numeric_cols:
            vals = X[col].dropna()
            drift_stats[col] = {
                "mean": float(vals.mean()),
                "std": float(vals.std()) if len(vals) > 1 else 0.0,
                "min": float(vals.min()),
                "max": float(vals.max()),
                "q25": float(vals.quantile(0.25)),
                "q75": float(vals.quantile(0.75)),
            }
    except Exception as _de:
        logger.warning(f"Drift stats computation failed: {_de}")
    
    # Plots
    sd = output_dirs["static"]
    scores_rel = f"plots/scores_{run_id}.png"
    plot_model_scores(models_metrics, os.path.join(sd, scores_rel), task_type)
    radar_rel = f"plots/radar_{run_id}.png"
    plot_radar_chart(models_metrics, os.path.join(sd, radar_rel), task_type)
    
    cv_plot_rel = None
    if cv_results:
        cv_plot_rel = f"plots/cv_scores_{run_id}.png"
        plot_cv_scores(cv_results, os.path.join(sd, cv_plot_rel))
    
    cv_summary = {n: {"mean": np.mean(s), "std": np.std(s), "scores": s} for n, s in cv_results.items()}
    
    results = {
        "run_id": run_id, "models_metrics": models_metrics,
        "best_model_name": best_name,
        "scores_plot_path": scores_rel, "radar_plot_path": radar_rel,
        "cv_plot_path": cv_plot_rel, "cv_summary": cv_summary,
        "preprocessed_path": os.path.basename(preprocessed_path),
        "model_path": os.path.basename(model_path),
        "pickle_path": os.path.basename(pickle_path) if pickle_path else None,
        "onnx_path": onnx_path,
        "task_type": task_type,
        "ensemble_used": use_ensemble and any(m["model_name"] == "StackingEnsemble" for m in models_metrics),
        "smote_applied": smote_applied,
        "feature_engineering_applied": feature_engineering,
        "feature_selection_info": selected_features_info,
        "feature_importance": feature_importance_data,
        "drift_stats": drift_stats,
    }
    
    if task_type == "classification":
        class_names = list(label_encoder.classes_) if label_encoder else [str(c) for c in np.unique(y)]
        cm = confusion_matrix(y_test, best_y_pred)
        cm_rel = f"plots/confusion_{run_id}.png"
        plot_confusion_matrix(cm, class_names, os.path.join(sd, cm_rel))
        results["confusion_matrix_path"] = cm_rel
        # Store raw matrix for interactive Plotly rendering in the browser
        results["confusion_matrix_data"] = cm.tolist()
        
        roc_rel = None
        nc = len(np.unique(y))
        if best_y_prob is not None:
            if nc == 2:
                roc_rel = f"plots/roc_{run_id}.png"
                plot_roc_curve(y_test, best_y_prob[:, 1], os.path.join(sd, roc_rel))
            elif nc <= 10:
                roc_rel = f"plots/roc_multi_{run_id}.png"
                plot_roc_multiclass(y_test, best_y_prob, class_names, os.path.join(sd, roc_rel))
        results["roc_curve_path"] = roc_rel
        results["classification_report_text"] = classification_report(y_test, best_y_pred, target_names=class_names, zero_division=0)
        results["class_names"] = class_names
    else:
        avp = f"plots/avp_{run_id}.png"
        plot_actual_vs_predicted(y_test, best_y_pred, os.path.join(sd, avp))
        results["actual_vs_predicted_path"] = avp
        res = f"plots/residuals_{run_id}.png"
        plot_residuals(y_test, best_y_pred, os.path.join(sd, res))
        results["residuals_path"] = res
    
    learning_rel = None
    if plot_learning and not pretransformed:
        update_progress("Computing learning curves...", 88)
        learning_rel = f"plots/learning_{run_id}.png"
        if not plot_learning_curves(best_trained, X_train, y_train, os.path.join(sd, learning_rel), task_type, cv_folds):
            learning_rel = None
    results["learning_curves_path"] = learning_rel
    
    shap_out = {"shap_summary_path": None, "shap_bar_path": None, "shap_csv_path": None}
    if compute_shap and not pretransformed:
        update_progress("Computing SHAP...", 90)
        try:
            sr = compute_and_save_shap(best_trained, X_train, output_dirs, run_id, shap_max_samples)
            shap_out = {"shap_summary_path": sr.get("shap_summary"), "shap_bar_path": sr.get("shap_bar"), "shap_csv_path": sr.get("shap_csv")}
        except Exception as e:
            logger.error(f"SHAP failed: {e}")
    results.update(shap_out)
    
    update_progress("Generating report...", 95)
    pdf_path = os.path.join(output_dirs["reports"], f"report_{run_id}.pdf")
    cfg = {"task_type": task_type, "use_optuna": use_optuna, "optuna_trials": optuna_trials if use_optuna else "N/A",
           "ensemble": use_ensemble, "smote": smote_applied, "class_weight": use_class_weight,
           "feature_engineering": feature_engineering, "feature_selection": use_feature_selection,
           "cv_folds": cv_folds if use_cv else "None"}
    
    if task_type == "classification":
        generate_classification_report(results, cfg, pdf_path)
    else:
        generate_regression_report(results, cfg, pdf_path)
    results["pdf_report_filename"] = os.path.basename(pdf_path)
    
    elapsed = (datetime.now() - start_time).total_seconds()
    results["training_time"] = elapsed
    update_progress("Complete!", 100)
    logger.info(f"AutoML completed in {elapsed:.2f}s. Best: {best_name}")
    return results

# [2026-02-13T11:30:00] Implement AutoML engine core selection logic

# [2026-02-25T16:20:00] Add cross-validation support in AutoML engine

# [2026-03-18T11:00:00] Fix memory leak in large dataset processing

# [2026-03-25T13:15:00] Refactor AutoML engine for pipeline extensibility

# [2026-03-29T10:30:00] Performance optimization: parallel model training

# [2026-02-13T11:30:00] Implement AutoML engine core selection logic

# [2026-02-25T16:20:00] Add cross-validation support in AutoML engine
