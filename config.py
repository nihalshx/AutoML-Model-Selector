# config.py - Centralized configuration management
import os
import secrets

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


class Config:
    """Base configuration."""
    # Persist SECRET_KEY to disk so Flask sessions survive server restarts.
    # A new random key is generated only on first run.
    _key_file = os.path.join(BASE_DIR, ".secret_key")
    if os.environ.get("FLASK_SECRET"):
        SECRET_KEY = os.environ["FLASK_SECRET"]
    elif os.path.exists(_key_file):
        with open(_key_file, "r") as _f:
            SECRET_KEY = _f.read().strip()
    else:
        SECRET_KEY = secrets.token_hex(32)
        with open(_key_file, "w") as _f:
            _f.write(SECRET_KEY)
    
    # Directories
    UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
    MODELS_FOLDER = os.path.join(BASE_DIR, "models")
    REPORTS_FOLDER = os.path.join(BASE_DIR, "reports")
    PREPROCESSED_FOLDER = os.path.join(BASE_DIR, "preprocessed")
    STATIC_FOLDER = os.path.join(BASE_DIR, "static")
    EXPERIMENTS_FOLDER = os.path.join(BASE_DIR, "experiments")
    
    # Database
    SQLALCHEMY_DATABASE_URI = os.environ.get("DATABASE_URL") or \
        "sqlite:///" + os.path.join(BASE_DIR, "automl.db")
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    
    # File upload
    ALLOWED_EXTENSIONS = {"csv"}
    MAX_CONTENT_LENGTH = int(os.environ.get("MAX_FILE_SIZE", 100 * 1024 * 1024))
    
    # AutoML defaults
    DEFAULT_TEST_SIZE = 0.2
    DEFAULT_CV_FOLDS = 5
    DEFAULT_OPTUNA_TRIALS = 20
    DEFAULT_SHAP_SAMPLES = 200
    MAX_OPTUNA_TRIALS = 100
    MAX_SHAP_SAMPLES = 1000
    
    # Cleanup
    FILE_RETENTION_DAYS = int(os.environ.get("FILE_RETENTION_DAYS", 30))
    
    # Server
    PORT = int(os.environ.get("PORT", 5000))
    DEBUG = os.environ.get("DEBUG", "False").lower() == "true"

    # ── Flask-Caching ──
    CACHE_TYPE = "SimpleCache"
    CACHE_DEFAULT_TIMEOUT = 300

    # ── Flask-Mail ──
    MAIL_SERVER   = os.environ.get("MAIL_SERVER", "smtp.gmail.com")
    MAIL_PORT     = int(os.environ.get("MAIL_PORT", 587))
    MAIL_USE_TLS  = os.environ.get("MAIL_USE_TLS", "true").lower() == "true"
    MAIL_USERNAME = os.environ.get("MAIL_USERNAME", "")
    MAIL_PASSWORD = os.environ.get("MAIL_PASSWORD", "")
    MAIL_DEFAULT_SENDER = os.environ.get("MAIL_SENDER", MAIL_USERNAME)
    
    @classmethod
    def init_dirs(cls):
        """Create all necessary directories. Each folder is handled independently
        so a failure on one doesn't prevent the others from being created."""
        import logging
        for folder in [cls.UPLOAD_FOLDER, cls.MODELS_FOLDER, cls.REPORTS_FOLDER,
                       cls.PREPROCESSED_FOLDER, cls.EXPERIMENTS_FOLDER,
                       os.path.join(cls.STATIC_FOLDER, "plots")]:
            try:
                os.makedirs(folder, exist_ok=True)
            except OSError as e:
                logging.getLogger(__name__).error(f"Could not create directory {folder}: {e}")


class DevelopmentConfig(Config):
    DEBUG = True


class ProductionConfig(Config):
    DEBUG = False


config_map = {
    "development": DevelopmentConfig,
    "production": ProductionConfig,
    "default": DevelopmentConfig
}

# [2026-02-07T14:20:00] Add config module with environment-based settings

# [2026-02-07T14:20:00] Add config module with environment-based settings
