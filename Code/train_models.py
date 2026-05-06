import json
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

from project_config import (
    MODEL_READY_3CLASS,
    MODELS_DIR,
    FIGURES_DIR,
    MODEL_TRAINING_REPORT,
    PER_CLASS_METRICS_CSV,
    FEATURE_IMPORTANCE_CSV,
)

TARGET_COL = "delay_class_3"
RANDOM_STATE = 42

def evaluate_model(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_test, y_pred, average="macro", zero_division=0
    )
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_test, y_pred, average="weighted", zero_division=0
    )

    class_report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)

    return {
        "accuracy": accuracy,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "f1_macro": f1_macro,
        "precision_weighted": precision_weighted,
        "recall_weighted": recall_weighted,
        "f1_weighted": f1_weighted,
        "classification_report": class_report,
        "confusion_matrix": cm,
        "model": model,
    }

def save_confusion_matrix(cm, title, output_path):
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1, 2])
    disp.plot(ax=ax)
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()

def save_model_comparison(results):
    comparison_df = pd.DataFrame([
        {
            "Model": r["name"],
            "Accuracy": r["accuracy"],
            "Macro F1": r["f1_macro"],
            "Weighted F1": r["f1_weighted"],
        }
        for r in results
    ])

    ax = comparison_df.set_index("Model").plot(kind="bar", figsize=(8, 5))
    ax.set_ylabel("Score")
    ax.set_title("Model Comparison - 3-Class Flight Delay Classification")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "model_comparison_3class.png", dpi=200, bbox_inches="tight")
    plt.close()

    ax = comparison_df.set_index("Model").plot(kind="bar", figsize=(8, 5))
    ax.set_ylabel("Score")
    ax.set_title("Baseline Model Performance Summary")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "model_metrics_bar_clean.png", dpi=200, bbox_inches="tight")
    plt.close()

def extract_per_class_metrics(metrics_summary):
    rows = []
    for model_name, model_info in metrics_summary.items():
        report = model_info["classification_report"]
        for class_label in ["0", "1", "2"]:
            rows.append({
                "model": model_name,
                "class": class_label,
                "precision": report[class_label]["precision"],
                "recall": report[class_label]["recall"],
                "f1-score": report[class_label]["f1-score"],
                "support": report[class_label]["support"],
            })
    return pd.DataFrame(rows)

def save_per_class_plots(df):
    for metric, filename in [
        ("precision", "per_class_precision_3class.png"),
        ("recall", "per_class_recall_3class.png"),
        ("f1-score", "per_class_f1_3class.png"),
    ]:
        pivot_df = df.pivot(index="class", columns="model", values=metric)
        ax = pivot_df.plot(kind="bar", figsize=(8, 5))
        ax.set_title(f"Per-Class {metric.title()} Comparison")
        ax.set_ylabel(metric.title())
        ax.set_xlabel("Delay Class")
        plt.xticks(rotation=0)
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / filename, dpi=200, bbox_inches="tight")
        plt.close()

    support_df = df[["class", "support"]].drop_duplicates().sort_values("class")
    ax = support_df.plot(x="class", y="support", kind="bar", figsize=(7, 5), legend=False)
    ax.set_title("Test Set Class Support")
    ax.set_ylabel("Count")
    ax.set_xlabel("Delay Class")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "class_support_3class.png", dpi=200, bbox_inches="tight")
    plt.close()

def clean_feature_name(name: str) -> str:
    name = name.replace("num__", "")
    name = name.replace("cat__", "")
    replacements = {
        "CRS_DEP_TIME_MIN": "Scheduled Departure Time",
        "CRS_ARR_TIME_MIN": "Scheduled Arrival Time",
        "CRS_ELAPSED_TIME": "Scheduled Elapsed Time",
        "SCHED_BLOCK_MINS": "Scheduled Block Time",
        "DISTANCE": "Distance",
        "dep_hour": "Departure Hour",
        "arr_hour": "Arrival Hour",
        "day_of_week": "Day of Week",
        "is_weekend": "Weekend Flag",
        "month": "Month",
        "day": "Day",
        "year": "Year",
        "dep_time_bucket_Early Morning": "Departure Time: Early Morning",
        "dep_time_bucket_Evening": "Departure Time: Evening",
        "AIRLINE_Southwest Airlines Co.": "Airline: Southwest",
        "AIRLINE_CODE_WN": "Airline Code: WN",
        "season_Summer": "Season: Summer",
        "season_Fall": "Season: Fall",
        "season_Winter": "Season: Winter",
        "season_Spring": "Season: Spring",
    }
    for old, new in replacements.items():
        name = name.replace(old, new)
    return name.replace("_", " ")

def save_feature_importance(rf_model, X):
    preprocessor = rf_model.named_steps["preprocessor"]
    classifier = rf_model.named_steps["classifier"]

    feature_names = preprocessor.get_feature_names_out()
    importances = classifier.feature_importances_

    fi_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances
    }).sort_values("importance", ascending=False)

    fi_df["clean_feature"] = fi_df["feature"].apply(clean_feature_name)
    fi_df.to_csv(FEATURE_IMPORTANCE_CSV, index=False)

    top20 = fi_df.head(20).sort_values("importance", ascending=True)
    plt.figure(figsize=(10, 7))
    plt.barh(top20["feature"], top20["importance"])
    plt.title("Random Forest Feature Importance - Top 20")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "random_forest_feature_importance_top20.png", dpi=200, bbox_inches="tight")
    plt.close()

    top15 = fi_df.head(15).sort_values("importance", ascending=True)
    plt.figure(figsize=(10, 7))
    plt.barh(top15["clean_feature"], top15["importance"])
    plt.title("Random Forest Feature Importance - Top 15")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "rf_feature_importance_top15_clean.png", dpi=200, bbox_inches="tight")
    plt.close()

def main():
    df = pd.read_csv(MODEL_READY_3CLASS, low_memory=False)

    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL].astype(int)

    categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
    numeric_cols = X.select_dtypes(include=["int64", "float64", "Int64"]).columns.tolist()

    # if "FL_DATE" in X.columns:
    #     if "FL_DATE" not in categorical_cols:
    #         categorical_cols.append("FL_DATE")
    #     if "FL_DATE" in numeric_cols:
    #         numeric_cols.remove("FL_DATE")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
        ]
    )

    logistic_model = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=RANDOM_STATE
        )),
    ])

    random_forest_model = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            random_state=RANDOM_STATE,
            n_jobs=-1
        )),
    ])

    models = [
        ("logistic_regression", logistic_model),
        ("random_forest", random_forest_model),
    ]

    all_results = []

    for name, model in models:
        print(f"Training {name}...")
        result = evaluate_model(model, X_train, X_test, y_train, y_test)
        result["name"] = name

        model_path = MODELS_DIR / f"{name}_3class.joblib"
        joblib.dump(result["model"], model_path)

        cm_path = FIGURES_DIR / f"{name}_confusion_matrix_3class.png"
        save_confusion_matrix(
            result["confusion_matrix"],
            f"{name.replace('_', ' ').title()} - Confusion Matrix",
            cm_path
        )

        result["model_path"] = str(model_path)
        result["confusion_matrix_figure"] = str(cm_path)
        all_results.append(result)

    save_model_comparison(all_results)

    metrics_summary = {
        r["name"]: {
            "accuracy": r["accuracy"],
            "precision_macro": r["precision_macro"],
            "recall_macro": r["recall_macro"],
            "f1_macro": r["f1_macro"],
            "precision_weighted": r["precision_weighted"],
            "recall_weighted": r["recall_weighted"],
            "f1_weighted": r["f1_weighted"],
            "classification_report": r["classification_report"],
            "confusion_matrix": r["confusion_matrix"].tolist(),
            "model_path": r["model_path"],
            "confusion_matrix_figure": r["confusion_matrix_figure"],
        }
        for r in all_results
    }

    per_class_df = extract_per_class_metrics(metrics_summary)
    per_class_df.to_csv(PER_CLASS_METRICS_CSV, index=False)
    save_per_class_plots(per_class_df)

    rf_model = next(r["model"] for r in all_results if r["name"] == "random_forest")
    save_feature_importance(rf_model, X)

    report = {
        "input_file": str(MODEL_READY_3CLASS),
        "dataset_shape": [int(df.shape[0]), int(df.shape[1])],
        "target_column": TARGET_COL,
        "categorical_columns": categorical_cols,
        "numeric_columns": numeric_cols,
        "metrics_summary": metrics_summary,
    }

    MODEL_TRAINING_REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("Saved training report to:", MODEL_TRAINING_REPORT)

if __name__ == "__main__":
    main()