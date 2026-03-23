import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
ARTIFACT_DIR = BASE_DIR/"artifacts"
LOG_DIR = BASE_DIR/"logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

# =========================================================
# MODEL CONFIGURATION
# =========================================================

MODEL_PATHS = {
    "arima" : ARTIFACT_DIR/"arima.pkl",
    "sarima" : ARTIFACT_DIR/"sarima.pkl",
    "prophet" : ARTIFACT_DIR/"prophet.json",
    "xgboost" : ARTIFACT_DIR/"xgboost.json"
}

DEFAULT_FORECAST_STEPS = 30
CONFIDENCE_LEVEL = 0.95
ALPHA = 0.05

CACHE_TTL = 3600 

# =========================================================
# API & SERVER CONFIGURATION (FastAPI)
# =========================================================

PROJECT_NAME = os.getenv("Arya_TS_Project", "Gold Price Forecasting API V2")
VERSION = "2.0.0"
API_PREFIX = "/api/v2"

HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", 8000))

DEBUG_MODE = os.getenv("DEBUG_MODE", "True").lower() == "true"


# =========================================================
# LOGGING CONFIGURATION
# =========================================================

LOG_CONFIG = {
    "version": 2,
    "disable_existing_loggers": False,
    "formatters": {
        "standard": {
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S",
        },
    },
    "handlers": {
        "console" : {
            "class" : "logging.StreamHandler",
            "formatter" : "standard",
            "level" : "INFO"
        },
        "file" : {
            "class": "logging.handlers.RotatingFileHandler",
            "filename": LOG_DIR/"app.log",
            "formatter": "standard",
            "level" : "ERROR",
            "maxBytes" : 10485760,
            "backupCount": 5
        },
    },
    "root" : {
        "handlers" : ["console", "file"],
        "level" : "INFO"
    },
}

# =========================================================
# DATABASE CONFIGURATION (SUPABASE)
# =========================================================
DATABASE_URL = os.getenv("DATABASE_URL", "")

if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

# =========================================================
# CACHE CONFIGURATION (UPSTASH REDIS)
# =========================================================
REDIS_URL = os.getenv("UPSTASH_REDIS_URL", "")

if REDIS_URL.startswith("redis://"):
    REDIS_URL = REDIS_URL.replace("redis://", "rediss://", 1)

