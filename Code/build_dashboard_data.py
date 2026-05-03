import json
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split

from project_config import (
    MODEL_READY_3CLASS,
    DASHBOARD_PREDICTIONS_3CLASS,
    MODELS_DIR,
    DASHBOARD_DATA_REPORT,
)

TARGET_COL = "delay_class_3"
RANDOM_STATE = 42

CLASS_LABELS = {
    0: "<=15 min",
    1: "16-60 min",
    2: ">60 min"
}

def main():
    df = pd.read_csv(MODEL_READY_3CLASS, low_memory=False)
    model = joblib.load(MODELS_DIR / "random_forest_3class.joblib")

    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL].astype(int)

    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)

    out = X_test.copy()
    out["actual_class"] = y_test.values
    out["predicted_class"] = preds
    out["actual_label"] = out["actual_class"].map(CLASS_LABELS)
    out["predicted_label"] = out["predicted_class"].map(CLASS_LABELS)

    out["prob_class_0"] = probs[:, 0]
    out["prob_class_1"] = probs[:, 1]
    out["prob_class_2"] = probs[:, 2]

    out["is_correct"] = (out["actual_class"] == out["predicted_class"]).astype(int)
    out["prediction_confidence"] = out[["prob_class_0", "prob_class_1", "prob_class_2"]].max(axis=1)

    out.to_csv(DASHBOARD_PREDICTIONS_3CLASS, index=False)

    report = {
        "input_data": str(MODEL_READY_3CLASS),
        "output_csv": str(DASHBOARD_PREDICTIONS_3CLASS),
        "row_count": int(len(out)),
        "column_count": int(len(out.columns)),
        "columns": list(out.columns),
    }

    DASHBOARD_DATA_REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("Saved dashboard prediction dataset to:", DASHBOARD_PREDICTIONS_3CLASS)

if __name__ == "__main__":
    main()