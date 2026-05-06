import json
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split

from project_config import (
    MODEL_READY_3CLASS,
    MODELS_DIR,
    REPORTS_DIR,
    FIGURES_DIR,
)

TARGET_COL = "delay_class_3"
RANDOM_STATE = 42

PERM_CSV = REPORTS_DIR / "feature_importance.csv"
EXPLAIN_REPORT = REPORTS_DIR / "explainability_report.json"

PERM_FIG = FIGURES_DIR / "permutation_importance_top15.png"
PERM_FIG_CLEAN = FIGURES_DIR / "permutation_importance_top15_clean.png"
LOCAL_EXPLANATION_CSV = REPORTS_DIR / "local_prediction_examples.csv"


def clean_feature_name(name: str) -> str:
    replacements = {
        "CRS_DEP_TIME_MIN": "Scheduled Departure Time",
        "CRS_ARR_TIME_MIN": "Scheduled Arrival Time",
        "CRS_ELAPSED_TIME": "Scheduled Elapsed Time",
        "SCHED_BLOCK_MINS": "Scheduled Block Time",
        "DISTANCE": "Distance",
        "FL_DATE": "Flight Date",
        "AIRLINE": "Airline",
        "AIRLINE_CODE": "Airline Code",
        "ORIGIN": "Origin Airport",
        "DEST": "Destination Airport",
        "ORIGIN_STATE": "Origin State",
        "DEST_STATE": "Destination State",
        "dep_hour": "Departure Hour",
        "arr_hour": "Arrival Hour",
        "day_of_week": "Day of Week",
        "is_weekend": "Weekend Flag",
        "month": "Month",
        "day": "Day",
        "year": "Year",
        "route": "Route",
        "season": "Season",
        "dep_time_bucket": "Departure Time Bucket",
    }
    return replacements.get(name, name.replace("_", " ").title())


def main():
    df = pd.read_csv(MODEL_READY_3CLASS, low_memory=False)

    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    pipeline = joblib.load(MODELS_DIR / "random_forest_3class.joblib")

    perm = permutation_importance(
        pipeline,
        X_test,
        y_test,
        n_repeats=3,
        random_state=RANDOM_STATE,
        scoring="f1_weighted",
        n_jobs=-1,
    )

    perm_df = pd.DataFrame({
        "feature": X_test.columns,
        "clean_feature": [clean_feature_name(c) for c in X_test.columns],
        "importance_mean": perm.importances_mean,
        "importance_std": perm.importances_std,
    }).sort_values("importance_mean", ascending=False)

    perm_df.to_csv(PERM_CSV, index=False)

    top15 = perm_df.head(15).sort_values("importance_mean", ascending=True)

    plt.figure(figsize=(10, 7))
    plt.barh(top15["feature"], top15["importance_mean"], xerr=top15["importance_std"])
    plt.title("Permutation Feature Importance - Top 15")
    plt.xlabel("Decrease in Weighted F1 After Shuffling")
    plt.tight_layout()
    plt.savefig(PERM_FIG, dpi=200, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(10, 7))
    plt.barh(top15["clean_feature"], top15["importance_mean"], xerr=top15["importance_std"])
    plt.title("Permutation Feature Importance - Top 15")
    plt.xlabel("Decrease in Weighted F1 After Shuffling")
    plt.tight_layout()
    plt.savefig(PERM_FIG_CLEAN, dpi=200, bbox_inches="tight")
    plt.close()

    probs = pipeline.predict_proba(X_test)
    preds = pipeline.predict(X_test)

    local_df = X_test.copy().reset_index(drop=True)
    local_df["actual_class"] = y_test.reset_index(drop=True)
    local_df["predicted_class"] = preds
    local_df["prob_class_0"] = probs[:, 0]
    local_df["prob_class_1"] = probs[:, 1]
    local_df["prob_class_2"] = probs[:, 2]
    local_df["prediction_confidence"] = local_df[["prob_class_0", "prob_class_1", "prob_class_2"]].max(axis=1)

    examples = pd.concat([
        local_df[local_df["actual_class"] == local_df["predicted_class"]]
        .sort_values("prediction_confidence", ascending=False)
        .head(5),
        local_df[local_df["actual_class"] != local_df["predicted_class"]]
        .sort_values("prediction_confidence", ascending=False)
        .head(5),
    ], ignore_index=True)

    examples.to_csv(LOCAL_EXPLANATION_CSV, index=False)

    report = {
        "model_used": "random_forest_3class",
        "input_data": str(MODEL_READY_3CLASS),
        "method": "Permutation Feature Importance",
        "metric_used": "weighted_f1",
        "permutation_importance_csv": str(PERM_CSV),
        "permutation_importance_figure": str(PERM_FIG),
        "permutation_importance_clean_figure": str(PERM_FIG_CLEAN),
        "local_prediction_examples_csv": str(LOCAL_EXPLANATION_CSV),
        "top_10_features": perm_df.head(10).to_dict(orient="records"),
    }

    EXPLAIN_REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("Saved permutation importance CSV to:", PERM_CSV)
    print("Saved permutation importance figure to:", PERM_FIG)
    print("Saved clean permutation importance figure to:", PERM_FIG_CLEAN)
    print("Saved local prediction examples to:", LOCAL_EXPLANATION_CSV)
    print("Saved explainability report to:", EXPLAIN_REPORT)


if __name__ == "__main__":
    main()