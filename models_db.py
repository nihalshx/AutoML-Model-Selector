# models_db.py - SQLAlchemy database models
import json
from datetime import datetime
from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()


class Experiment(db.Model):
    """Stores experiment metadata and results."""
    __tablename__ = "experiments"
    
    id = db.Column(db.String(36), primary_key=True)
    status = db.Column(db.String(20), default="pending")  # pending, running, completed, failed
    task_type = db.Column(db.String(20), nullable=False)
    dataset_filename = db.Column(db.String(256))
    target_column = db.Column(db.String(128))
    
    # Dataset info
    num_rows = db.Column(db.Integer)
    num_columns = db.Column(db.Integer)
    feature_names_json = db.Column(db.Text)
    
    # Config
    config_json = db.Column(db.Text)
    
    # Results
    best_model_name = db.Column(db.String(64))
    results_json = db.Column(db.Text)
    
    # File paths
    model_path = db.Column(db.String(512))
    preprocessed_path = db.Column(db.String(512))
    report_path = db.Column(db.String(512))
    
    # Timestamps
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    started_at = db.Column(db.DateTime)
    completed_at = db.Column(db.DateTime)
    training_time_seconds = db.Column(db.Float)
    
    # Training progress tracking
    progress_json = db.Column(db.Text, default='{}')

    # Feature: Tagging & Notes
    tags_json = db.Column(db.Text, default='[]')
    notes = db.Column(db.Text, default='')

    # Feature: Email Notification
    notify_email = db.Column(db.String(256), default='')

    # Feature: Drift Detection stats (mean/std/min/max per feature at training time)
    drift_stats_json = db.Column(db.Text, default='{}')
    
    def set_config(self, config_dict):
        self.config_json = json.dumps(config_dict)
    
    def get_config(self):
        return json.loads(self.config_json) if self.config_json else {}
    
    def set_results(self, results_dict):
        self.results_json = json.dumps(results_dict, default=str)
    
    def get_results(self):
        return json.loads(self.results_json) if self.results_json else {}
    
    def set_progress(self, progress_dict):
        self.progress_json = json.dumps(progress_dict, default=str)
    
    def get_progress(self):
        return json.loads(self.progress_json) if self.progress_json else {}
    
    def set_feature_names(self, names):
        self.feature_names_json = json.dumps(names)
    
    def get_feature_names(self):
        return json.loads(self.feature_names_json) if self.feature_names_json else []

    # ── Tags & Notes ──
    def get_tags(self):
        try:
            return json.loads(self.tags_json) if self.tags_json else []
        except Exception:
            return []

    def set_tags(self, tags_list):
        self.tags_json = json.dumps([str(t).strip()[:40] for t in tags_list if str(t).strip()])

    # ── Drift Stats ──
    def get_drift_stats(self):
        try:
            return json.loads(self.drift_stats_json) if self.drift_stats_json else {}
        except Exception:
            return {}

    def set_drift_stats(self, stats_dict):
        self.drift_stats_json = json.dumps(stats_dict, default=str)
    
    def to_dict(self):
        return {
            "id": self.id,
            "status": self.status,
            "task_type": self.task_type,
            "dataset_filename": self.dataset_filename,
            "target_column": self.target_column,
            "num_rows": self.num_rows,
            "num_columns": self.num_columns,
            "best_model_name": self.best_model_name,
            "config": self.get_config(),
            "results": self.get_results(),
            "tags": self.get_tags(),
            "notes": self.notes or "",
            "notify_email": self.notify_email or "",
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "training_time_seconds": self.training_time_seconds,
        }


def init_db(app):
    """Initialize database with app context."""
    db.init_app(app)
    with app.app_context():
        db.create_all()

# [2026-02-09T09:45:00] Implement database models for experiment tracking

# [2026-02-27T14:30:00] Add experiment history tracking in database

# [2026-03-23T10:00:00] Implement model persistence and versioning
