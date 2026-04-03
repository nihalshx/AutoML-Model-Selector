# visualization.py - All plotting functions including radar charts, learning curves
import os
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
import logging

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, roc_auc_score, r2_score
from sklearn.model_selection import learning_curve

logger = logging.getLogger(__name__)

sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 120
plt.rcParams['savefig.dpi'] = 120
plt.rcParams['font.family'] = 'sans-serif'

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    shap = None
    SHAP_AVAILABLE = False


def _save_fig(fig, out_path):
    """Save figure and close."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", facecolor='white')
    plt.close(fig)


# ─────────── Model Scores Bar Chart ───────────

def plot_model_scores(metrics: List[Dict], out_path: str, task_type: str = "classification"):
    """Plot model comparison bar charts."""
    names = [m["model_name"] for m in metrics]
    
    if task_type == "classification":
        fig, axes = plt.subplots(1, 2, figsize=(max(10, len(names)*1.2), 5))
        
        axes[0].bar(range(len(names)), [m["accuracy"] for m in metrics], color='#2196F3', alpha=0.85)
        axes[0].set_xticks(range(len(names)))
        axes[0].set_xticklabels(names, rotation=45, ha="right", fontsize=9)
        axes[0].set_ylabel("Accuracy")
        axes[0].set_title("Model Accuracy", fontweight='bold')
        axes[0].set_ylim([0, 1])
        axes[0].grid(axis='y', alpha=0.3)
        
        axes[1].bar(range(len(names)), [m["f1"] for m in metrics], color='#FF5722', alpha=0.85)
        axes[1].set_xticks(range(len(names)))
        axes[1].set_xticklabels(names, rotation=45, ha="right", fontsize=9)
        axes[1].set_ylabel("F1-Score")
        axes[1].set_title("Model F1-Score", fontweight='bold')
        axes[1].set_ylim([0, 1])
        axes[1].grid(axis='y', alpha=0.3)
    else:
        fig, axes = plt.subplots(1, 2, figsize=(max(10, len(names)*1.2), 5))
        
        axes[0].bar(range(len(names)), [m["r2"] for m in metrics], color='#2196F3', alpha=0.85)
        axes[0].set_xticks(range(len(names)))
        axes[0].set_xticklabels(names, rotation=45, ha="right", fontsize=9)
        axes[0].set_ylabel("R² Score")
        axes[0].set_title("Model R² Score", fontweight='bold')
        axes[0].axhline(y=0, color='r', linestyle='--', alpha=0.5)
        axes[0].grid(axis='y', alpha=0.3)
        
        axes[1].bar(range(len(names)), [m["mae"] for m in metrics], color='#FF5722', alpha=0.85)
        axes[1].set_xticks(range(len(names)))
        axes[1].set_xticklabels(names, rotation=45, ha="right", fontsize=9)
        axes[1].set_ylabel("MAE")
        axes[1].set_title("Model MAE", fontweight='bold')
        axes[1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    _save_fig(fig, out_path)


# ─────────── Radar / Spider Chart ───────────

def plot_radar_chart(metrics: List[Dict], out_path: str, task_type: str = "classification"):
    """Plot radar/spider chart comparing top models across all metrics."""
    top = metrics[:min(5, len(metrics))]
    
    if task_type == "classification":
        metric_keys = ["accuracy", "f1", "precision", "recall"]
        metric_labels = ["Accuracy", "F1-Score", "Precision", "Recall"]
    else:
        metric_keys = ["r2", "mae", "rmse"]
        metric_labels = ["R²", "MAE (inv)", "RMSE (inv)"]
    
    n_metrics = len(metric_keys)
    angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
    angles += angles[:1]
    
    colors = ['#2196F3', '#4CAF50', '#FF5722', '#9C27B0', '#FF9800']
    
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    ax.set_facecolor('#fafafa')
    
    for i, model in enumerate(top):
        values = []
        for key in metric_keys:
            v = model.get(key, 0)
            # Invert MAE/RMSE so higher = better on radar
            if key in ["mae", "rmse"] and task_type == "regression":
                max_v = max(m.get(key, 1) for m in top) or 1
                v = 1 - (v / max_v)
            values.append(v)
        values += values[:1]
        
        ax.plot(angles, values, 'o-', linewidth=2, label=model["model_name"],
                color=colors[i % len(colors)], markersize=6)
        ax.fill(angles, values, alpha=0.1, color=colors[i % len(colors)])
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metric_labels, fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=9)
    ax.set_title("Model Comparison Radar", fontsize=14, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    _save_fig(fig, out_path)


# ─────────── Confusion Matrix ───────────

def plot_confusion_matrix(cm, class_names, out_path, title="Confusion Matrix"):
    """Plot confusion matrix heatmap."""
    fig, ax = plt.subplots(figsize=(max(8, len(class_names)*0.8), max(6, len(class_names)*0.6)))
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    sns.heatmap(cm_norm, annot=cm, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Proportion'}, ax=ax)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("True", fontsize=12)
    plt.tight_layout()
    _save_fig(fig, out_path)


# ─────────── ROC Curve ───────────

def plot_roc_curve(y_true, y_prob, out_path, title="ROC Curve"):
    """Plot ROC curve for binary classification."""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)
    
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, label=f"ROC (AUC = {auc:.3f})", linewidth=2, color='#2196F3')
    ax.plot([0, 1], [0, 1], linestyle="--", color='gray')
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='lower right')
    ax.grid(alpha=0.3)
    plt.tight_layout()
    _save_fig(fig, out_path)


# ─────────── Multiclass ROC ───────────

def plot_roc_multiclass(y_true, y_prob, class_names, out_path):
    """Plot One-vs-Rest ROC curves for multiclass."""
    from sklearn.preprocessing import label_binarize
    n_classes = len(class_names)
    y_bin = label_binarize(y_true, classes=list(range(n_classes)))
    
    fig, ax = plt.subplots(figsize=(8, 7))
    colors = plt.cm.Set1(np.linspace(0, 1, n_classes))
    
    for i in range(n_classes):
        if y_bin[:, i].sum() > 0:
            fpr, tpr, _ = roc_curve(y_bin[:, i], y_prob[:, i])
            auc = roc_auc_score(y_bin[:, i], y_prob[:, i])
            ax.plot(fpr, tpr, color=colors[i], linewidth=2,
                    label=f"{class_names[i]} (AUC={auc:.3f})")
    
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Multiclass ROC (One-vs-Rest)", fontweight='bold')
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    _save_fig(fig, out_path)


# ─────────── Regression Plots ───────────

def plot_actual_vs_predicted(y_true, y_pred, out_path, title="Actual vs Predicted"):
    """Scatter plot of actual vs predicted."""
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(y_true, y_pred, alpha=0.5, s=30, color='#2196F3', edgecolors='k', linewidth=0.5)
    mn, mx = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
    ax.plot([mn, mx], [mn, mx], 'r--', lw=2, label='Ideal')
    r2 = r2_score(y_true, y_pred)
    ax.text(0.05, 0.95, f'R² = {r2:.3f}', transform=ax.transAxes,
            fontsize=12, va='top', bbox=dict(boxstyle='round', fc='wheat', alpha=0.5))
    ax.set_xlabel("Actual"); ax.set_ylabel("Predicted")
    ax.set_title(title, fontweight='bold'); ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    _save_fig(fig, out_path)


def plot_residuals(y_true, y_pred, out_path):
    """Plot residuals distribution."""
    residuals = y_true - y_pred
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    ax1.scatter(y_pred, residuals, alpha=0.5, s=30, color='#2196F3', edgecolors='k', linewidth=0.5)
    ax1.axhline(0, color='r', linestyle='--', lw=2)
    ax1.set_xlabel("Predicted"); ax1.set_ylabel("Residuals")
    ax1.set_title("Residuals vs Predicted", fontweight='bold'); ax1.grid(alpha=0.3)
    
    ax2.hist(residuals, bins=30, color='#2196F3', alpha=0.7, edgecolor='black')
    ax2.axvline(0, color='r', linestyle='--', lw=2)
    ax2.set_xlabel("Residuals"); ax2.set_ylabel("Frequency")
    ax2.set_title("Residuals Distribution", fontweight='bold'); ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    _save_fig(fig, out_path)


# ─────────── Learning Curves ───────────

def plot_learning_curves(estimator, X, y, out_path, task_type="classification", cv=5):
    """Plot learning curves for the best model."""
    scoring = "f1_weighted" if task_type == "classification" else "neg_mean_squared_error"
    
    try:
        train_sizes, train_scores, val_scores = learning_curve(
            estimator, X, y, cv=cv, scoring=scoring,
            train_sizes=np.linspace(0.1, 1.0, 10),
            n_jobs=-1, random_state=42
        )
        
        train_mean = train_scores.mean(axis=1)
        train_std = train_scores.std(axis=1)
        val_mean = val_scores.mean(axis=1)
        val_std = val_scores.std(axis=1)
        
        fig, ax = plt.subplots(figsize=(9, 6))
        ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.15, color='#2196F3')
        ax.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.15, color='#4CAF50')
        ax.plot(train_sizes, train_mean, 'o-', color='#2196F3', label='Training Score', lw=2)
        ax.plot(train_sizes, val_mean, 'o-', color='#4CAF50', label='Validation Score', lw=2)
        
        ax.set_xlabel("Training Set Size", fontsize=12)
        ax.set_ylabel("Score", fontsize=12)
        ax.set_title("Learning Curves", fontsize=14, fontweight='bold')
        ax.legend(loc='lower right', fontsize=10)
        ax.grid(alpha=0.3)
        plt.tight_layout()
        _save_fig(fig, out_path)
        return True
    except Exception as e:
        logger.error(f"Learning curve plot failed: {e}")
        return False


# ─────────── Cross-Validation Scores ───────────

def plot_cv_scores(cv_results: Dict[str, List[float]], out_path: str):
    """Plot cross-validation scores for each model."""
    fig, ax = plt.subplots(figsize=(max(10, len(cv_results)*1.5), 6))
    
    positions = list(range(len(cv_results)))
    names = list(cv_results.keys())
    data = [cv_results[n] for n in names]
    
    bp = ax.boxplot(data, positions=positions, widths=0.5, patch_artist=True)
    colors = plt.cm.Set2(np.linspace(0, 1, len(names)))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax.set_xticks(positions)
    ax.set_xticklabels(names, rotation=45, ha='right', fontsize=10)
    ax.set_ylabel("CV Score", fontsize=12)
    ax.set_title("Cross-Validation Scores Distribution", fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    _save_fig(fig, out_path)


# ─────────── SHAP ───────────

def compute_and_save_shap(pipe, X_train, output_dirs, run_id, max_samples=200):
    """Compute SHAP values and save plots."""
    if not SHAP_AVAILABLE:
        logger.warning("SHAP not available")
        return {"shap_summary": None, "shap_bar": None, "shap_csv": None}
    
    logger.info("Computing SHAP values...")
    X_sample = X_train.sample(n=min(max_samples, len(X_train)), random_state=42)
    
    plots_dir = os.path.join(output_dirs["static"], "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    shap_summary_rel = f"plots/shap_summary_{run_id}.png"
    shap_bar_rel = f"plots/shap_bar_{run_id}.png"
    shap_csv_rel = f"plots/shap_values_{run_id}.csv"
    
    try:
        model = pipe.named_steps.get("model")
        preprocessor = pipe.named_steps.get("preprocessor")
        if preprocessor is None or model is None:
            raise RuntimeError("Pipeline missing steps")
        
        X_trans = preprocessor.transform(X_sample)
        
        if hasattr(model, 'tree_') or hasattr(model, 'estimators_'):
            explainer = shap.TreeExplainer(model)
        else:
            explainer = shap.KernelExplainer(model.predict, X_trans[:50])
        shap_values = explainer.shap_values(X_trans)
        
        if isinstance(shap_values, list):
            shap_values = np.abs(np.array(shap_values)).mean(axis=0)
        
        try:
            feature_names = preprocessor.get_feature_names_out()
        except:
            feature_names = [f"f_{i}" for i in range(X_trans.shape[1])]
        
        # Summary plot
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, X_trans, feature_names=feature_names, show=False)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dirs["static"], shap_summary_rel), bbox_inches="tight")
        plt.close()
        
        # Bar plot
        mean_abs = np.abs(shap_values).mean(axis=0)
        importance = pd.DataFrame({'feature': feature_names, 'mean_abs_shap': mean_abs})
        importance = importance.sort_values('mean_abs_shap', ascending=False)
        top = importance.head(20)
        
        fig, ax = plt.subplots(figsize=(10, max(6, len(top)*0.3)))
        ax.barh(range(len(top)), top['mean_abs_shap'].values[::-1], color='#2196F3')
        ax.set_yticks(range(len(top)))
        ax.set_yticklabels(top['feature'].values[::-1])
        ax.set_xlabel("Mean |SHAP value|")
        ax.set_title("Feature Importance (Top 20)", fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        _save_fig(fig, os.path.join(output_dirs["static"], shap_bar_rel))
        
        importance.to_csv(os.path.join(output_dirs["static"], shap_csv_rel), index=False)
        
        return {"shap_summary": shap_summary_rel, "shap_bar": shap_bar_rel, "shap_csv": shap_csv_rel}
    except Exception as e:
        logger.error(f"SHAP failed: {e}")
        return {"shap_summary": None, "shap_bar": None, "shap_csv": None}

# [2026-02-19T15:45:00] Add visualization module for model comparison charts

# [2026-02-28T09:15:00] Improve visualization with interactive Plotly charts

# [2026-03-20T16:30:00] Add model explainability with SHAP values

# [2026-02-19T15:45:00] Add visualization module for model comparison charts

# [2026-02-28T09:15:00] Improve visualization with interactive Plotly charts

# [2026-03-11T11:45:00] Fix NaN handling in feature importance visualization

# [2026-04-03T13:00:00] Add heatmap visualization for correlation matrix
