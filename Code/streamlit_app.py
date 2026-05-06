from pathlib import Path
import json

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


BASE_DIR = Path(__file__).resolve().parents[1]
REPORTS_DIR = BASE_DIR / "Reports"
FIGURES_DIR = BASE_DIR / "Figures"

MODEL_TRAINING_REPORT = REPORTS_DIR / "model_training_report.json"
PER_CLASS_METRICS_CSV = REPORTS_DIR / "per_class_metrics.csv"
FEATURE_IMPORTANCE_CSV = REPORTS_DIR / "feature_importance.csv"
EXPLAINABILITY_REPORT = REPORTS_DIR / "explainability_report.json"
LOCAL_EXAMPLES_CSV = REPORTS_DIR / "local_prediction_examples.csv"


CLASS_LABEL_MAP = {
    "0": "<=15 min",
    "1": "16-60 min",
    "2": ">60 min",
    0: "<=15 min",
    1: "16-60 min",
    2: ">60 min",
}


st.set_page_config(
    page_title="Flight Delay Severity Model Explorer",
    page_icon="✈️",
    layout="wide",
)


@st.cache_data
def load_training_report():
    with open(MODEL_TRAINING_REPORT, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data
def load_per_class_metrics():
    df = pd.read_csv(PER_CLASS_METRICS_CSV)
    df["class"] = df["class"].map(CLASS_LABEL_MAP)
    return df


@st.cache_data
def load_feature_importance():
    df = pd.read_csv(FEATURE_IMPORTANCE_CSV)
    if "clean_feature" not in df.columns:
        df["clean_feature"] = df["feature"]
    return df


@st.cache_data
def load_local_examples():
    if LOCAL_EXAMPLES_CSV.exists():
        return pd.read_csv(LOCAL_EXAMPLES_CSV)
    return pd.DataFrame()


def build_model_summary_df(report_json: dict) -> pd.DataFrame:
    rows = []
    for model_name, info in report_json["metrics_summary"].items():
        rows.append(
            {
                "Model": model_name.replace("_", " ").title(),
                "Accuracy": info["accuracy"],
                "Macro F1": info["f1_macro"],
                "Weighted F1": info["f1_weighted"],
                "Precision Macro": info["precision_macro"],
                "Recall Macro": info["recall_macro"],
            }
        )
    return pd.DataFrame(rows)


def get_confusion_matrix_df(report_json: dict, model_key: str) -> pd.DataFrame:
    cm = report_json["metrics_summary"][model_key]["confusion_matrix"]
    labels = ["<=15 min", "16-60 min", ">60 min"]
    return pd.DataFrame(cm, index=labels, columns=labels)


def make_model_comparison_chart(summary_df: pd.DataFrame):
    plot_df = summary_df.melt(
        id_vars="Model",
        value_vars=["Accuracy", "Macro F1", "Weighted F1"],
        var_name="Metric",
        value_name="Score",
    )
    fig = px.bar(
        plot_df,
        x="Model",
        y="Score",
        color="Metric",
        barmode="group",
        title="Model Comparison",
        text_auto=".3f",
    )
    fig.update_layout(height=450)
    fig.update_yaxes(range=[0, 1])
    return fig


def make_confusion_heatmap(cm_df: pd.DataFrame, model_title: str):
    fig = px.imshow(
        cm_df,
        text_auto=True,
        aspect="auto",
        labels=dict(x="Predicted Class", y="Actual Class", color="Count"),
        title=f"{model_title} Confusion Matrix",
    )
    fig.update_layout(height=450)
    return fig


def make_per_class_metric_chart(per_class_df: pd.DataFrame, metric: str):
    fig = px.bar(
        per_class_df,
        x="class",
        y=metric,
        color="model",
        barmode="group",
        title=f"Per-Class {metric.title()}",
        labels={"class": "Delay Class", "model": "Model"},
        text_auto=".3f",
    )
    fig.update_layout(height=450)
    fig.update_yaxes(range=[0, 1])
    return fig


def make_feature_importance_chart(fi_df: pd.DataFrame, top_n: int):
    top_df = fi_df.head(top_n).iloc[::-1]
    fig = px.bar(
        top_df,
        x="importance_mean" if "importance_mean" in top_df.columns else "importance",
        y="clean_feature",
        orientation="h",
        title=f"Top {top_n} Explainability Features",
        labels={
            "importance_mean": "Importance",
            "importance": "Importance",
            "clean_feature": "Feature",
        },
    )
    fig.update_layout(height=550)
    return fig


def style_local_examples(df: pd.DataFrame) -> pd.DataFrame:
    cols = []
    for c in [
        "AIRLINE",
        "ORIGIN",
        "DEST",
        "dep_hour",
        "month",
        "actual_class",
        "predicted_class",
        "prob_class_0",
        "prob_class_1",
        "prob_class_2",
        "prediction_confidence",
    ]:
        if c in df.columns:
            cols.append(c)

    out = df[cols].copy()
    if "actual_class" in out.columns:
        out["actual_class"] = out["actual_class"].map(CLASS_LABEL_MAP)
    if "predicted_class" in out.columns:
        out["predicted_class"] = out["predicted_class"].map(CLASS_LABEL_MAP)
    return out


def main():
    report_json = load_training_report()
    per_class_df = load_per_class_metrics()
    fi_df = load_feature_importance()
    local_examples_df = load_local_examples()

    summary_df = build_model_summary_df(report_json)

    st.title("Flight Delay Severity Model Explorer")
    st.caption("Streamlit dashboard focused on model performance and explainability.")

    st.markdown(
        """
        This dashboard answers two questions:
        1. **How well does the delay severity classifier perform?**
        2. **Which features drive its predictions?**
        """
    )

    st.subheader("1. Model Summary")
    c1, c2, c3 = st.columns(3)
    best_row = summary_df.sort_values("Macro F1", ascending=False).iloc[0]
    c1.metric("Best Model", best_row["Model"])
    c2.metric("Best Macro F1", f"{best_row['Macro F1']:.3f}")
    c3.metric("Best Weighted F1", f"{best_row['Weighted F1']:.3f}")

    st.plotly_chart(make_model_comparison_chart(summary_df), use_container_width=True)

    st.subheader("2. Confusion Matrix Analysis")
    model_option = st.selectbox(
        "Choose model for confusion matrix",
        options=list(report_json["metrics_summary"].keys()),
        format_func=lambda x: x.replace("_", " ").title(),
        index=1 if "random_forest" in report_json["metrics_summary"] else 0,
    )
    cm_df = get_confusion_matrix_df(report_json, model_option)
    st.plotly_chart(
        make_confusion_heatmap(cm_df, model_option.replace("_", " ").title()),
        use_container_width=True,
    )

    st.info(
        "Interpretation: the diagonal shows correct classifications. Off-diagonal cells show which delay severity classes are being confused."
    )

    st.subheader("3. Per-Class Performance")
    metric_choice = st.radio(
        "Choose per-class metric",
        options=["precision", "recall", "f1-score"],
        horizontal=True,
    )
    st.plotly_chart(
        make_per_class_metric_chart(per_class_df, metric_choice),
        use_container_width=True,
    )

    with st.expander("View per-class metrics table"):
        show_df = per_class_df.copy()
        show_df["model"] = show_df["model"].str.replace("_", " ").str.title()
        st.dataframe(show_df, use_container_width=True)

    st.subheader("4. Explainability: Feature Importance")
    top_n = st.slider("Number of top features to display", min_value=5, max_value=20, value=15)
    st.plotly_chart(make_feature_importance_chart(fi_df, top_n), use_container_width=True)

    st.markdown(
        """
        **How to read this chart**
        - Features higher on the chart matter more to the model.
        - Larger importance means model performance drops more when that feature is disrupted.
        - In this project, schedule timing and temporal structure are among the strongest drivers.
        """
    )

    st.subheader("5. Prediction Example Explorer")
    if local_examples_df.empty:
        st.warning("No local prediction examples file found. Run the explainability script first.")
    else:
        examples = style_local_examples(local_examples_df)
        example_view = st.radio(
            "Example set",
            ["All saved examples", "Correct high-confidence examples", "Incorrect high-confidence examples"],
            horizontal=True,
        )

        if example_view == "Correct high-confidence examples":
            filtered = local_examples_df[
                local_examples_df["actual_class"] == local_examples_df["predicted_class"]
            ].copy()
            filtered = filtered.sort_values("prediction_confidence", ascending=False)
            examples = style_local_examples(filtered)
        elif example_view == "Incorrect high-confidence examples":
            filtered = local_examples_df[
                local_examples_df["actual_class"] != local_examples_df["predicted_class"]
            ].copy()
            filtered = filtered.sort_values("prediction_confidence", ascending=False)
            examples = style_local_examples(filtered)

        st.dataframe(examples, use_container_width=True)

        if "prediction_confidence" in local_examples_df.columns:
            conf_fig = px.histogram(
                local_examples_df,
                x="prediction_confidence",
                nbins=20,
                title="Prediction Confidence Distribution for Saved Examples",
            )
            conf_fig.update_layout(height=400)
            st.plotly_chart(conf_fig, use_container_width=True)

    st.subheader("Dashboard Takeaways")
    st.markdown(
        """
        - **Random Forest is the stronger baseline** compared with Logistic Regression.
        - **Minority delay classes remain harder to predict**, especially the most severe class.
        - **Schedule timing features dominate** the model's decision process.
        - This suggests delay severity is driven by structured operational timing patterns rather than random noise.
        """
    )


if __name__ == "__main__":
    main()