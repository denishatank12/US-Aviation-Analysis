from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]

CODE_DIR = BASE_DIR / "Code"
DATA_DIR = BASE_DIR / "Data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = BASE_DIR / "Models"
FIGURES_DIR = BASE_DIR / "Figures"
REPORTS_DIR = BASE_DIR / "Reports"

for folder in [RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, FIGURES_DIR, REPORTS_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

RAW_INPUT = RAW_DATA_DIR / "kv_dashboard_sample.csv"

CLEANED_DATA = PROCESSED_DATA_DIR / "cleaned_flights.csv"
MODEL_READY_3CLASS = PROCESSED_DATA_DIR / "flights_model_ready_3class.csv"
DASHBOARD_PREDICTIONS_3CLASS = PROCESSED_DATA_DIR / "dashboard_predictions_3class.csv"

CLEANING_REPORT = REPORTS_DIR / "cleaning_report.json"
MODEL_TRAINING_REPORT = REPORTS_DIR / "model_training_report.json"
PER_CLASS_METRICS_CSV = REPORTS_DIR / "per_class_metrics.csv"
FEATURE_IMPORTANCE_CSV = REPORTS_DIR / "feature_importance.csv"
DASHBOARD_DATA_REPORT = REPORTS_DIR / "dashboard_data_report.json"