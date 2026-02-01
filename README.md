# 🧠 AutoML Platform

A production-grade AutoML web application with 3D neural network visualization, ensemble learning, and interactive dashboards.

## Features

### ML Pipeline
- **10+ ML Models**: LogisticRegression, RandomForest, GradientBoosting, XGBoost, LightGBM, CatBoost, SVM, KNN, and more
- **Stacking Ensemble**: Combines top 3 models with a meta-learner for 1-3% accuracy boost
- **Stratified K-Fold CV**: Reports mean ± std across folds instead of single split
- **Optuna Tuning**: Expanded to all model types (RF, XGB, LGB, CatBoost, GB, SVM, KNN)
- **SMOTE + Class Weights**: Handles imbalanced datasets
- **Auto Feature Engineering**: Date parsing, interactions, polynomial features
- **Feature Selection**: SelectKBest with mutual information or F-test
- **Early Stopping**: For boosting models
- **Learning Curves**: Bias/variance analysis
- **SHAP Explainability**: Feature importance and summary plots
- **Multi-class ROC**: One-vs-Rest ROC curves for multi-class problems

### Architecture
- **Modular Design**: 7 focused modules (preprocessing, ml_models, tuning, visualization, reporting, automl_engine, config)
- **SQLite Database**: Replaces in-memory dict; persists experiments across restarts
- **Background Training**: Threading-based workers; Flask doesn't block during training
- **SSE Progress**: Real-time Server-Sent Events for live training updates
- **Path Traversal Protection**: Safe file downloads
- **File Cleanup**: Auto-delete experiments older than N days

### UI/UX
- **Drag-and-Drop Upload**: With instant CSV preview
- **Auto-Detection**: Target column and task type auto-detected
- **Dark/Light Mode**: CSS variables-based theme toggle
- **Responsive Design**: Works on mobile
- **Radar Chart**: Multi-metric model comparison
- **3D Neural Network Visualizer**: Interactive Three.js visualization with:
  - Forward/backward propagation animation
  - Neuron activation glow
  - Weight connection coloring
  - Real-time loss/accuracy charts
  - Weight statistics per layer
  - Activation heatmap
- **Prediction API**: POST endpoint for inference with trained models

## Project Structure

```
advanced_automl/
├── app.py                 # Flask application (routes, SSE, API)
├── config.py              # Configuration management
├── models_db.py           # SQLAlchemy database models
├── preprocessing.py       # Data validation, feature engineering, selection
├── ml_models.py           # Model definitions, ensemble, SMOTE
├── tuning.py              # Expanded Optuna hyperparameter tuning
├── visualization.py       # All plotting (radar, learning curves, SHAP, etc.)
├── reporting.py           # PDF report generation
├── automl_engine.py       # Main AutoML pipeline
├── requirements.txt       # Dependencies
├── templates/
│   ├── base.html          # Base template (dark mode, nav, responsive)
│   ├── upload.html         # Drag-drop upload with preview
│   ├── training.html       # Real-time SSE progress page
│   ├── results.html        # Interactive results dashboard
│   ├── experiments.html    # Experiment history with delete
│   ├── experiment_detail.html
│   └── visualizer.html     # 3D Neural Network Visualizer
└── static/
    └── plots/              # Generated plot images
```

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the application
python app.py

# Open in browser
http://localhost:5000
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET/POST | `/` | Upload dataset and configure training |
| GET | `/training/<id>` | Real-time training progress |
| GET | `/results/<id>` | Interactive results dashboard |
| GET | `/experiments` | List all experiments |
| GET | `/visualizer` | 3D Neural Network Visualizer |
| POST | `/api/preview` | Preview uploaded CSV |
| POST | `/api/predict/<id>` | Make predictions |
| GET | `/api/progress/<id>` | SSE progress stream |
| GET | `/health` | Health check |

## Configuration

Set via environment variables:
- `FLASK_SECRET` - Secret key for sessions
- `DATABASE_URL` - Database URI (default: SQLite)
- `PORT` - Server port (default: 5000)
- `DEBUG` - Debug mode (default: False)
- `MAX_FILE_SIZE` - Upload limit in bytes (default: 100MB)
- `FILE_RETENTION_DAYS` - Auto-cleanup period (default: 30)

# [2026-02-01T09:15:00] Initial project setup: scaffold AutoML-Model-Selector structure
