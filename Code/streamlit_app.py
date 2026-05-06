from pathlib import Path
import json

import pandas as pd
import plotly.express as px
import streamlit as st


BASE_DIR = Path(__file__).resolve().parents[1]
REPORTS_DIR = BASE_DIR / "Reports"

MODEL_TRAINING_REPORT = REPORTS_DIR / "model_training_report.json"
PER_CLASS_METRICS_CSV = REPORTS_DIR / "per_class_metrics.csv"
FEATURE_IMPORTANCE_CSV = REPORTS_DIR / "feature_importance.csv"
LOCAL_EXAMPLES_CSV = REPORTS_DIR / "local_prediction_examples.csv"

CLASS_LABEL_MAP = {
    "0": "<=15 min",
    "1": "16-60 min",
    "2": ">60 min",
    0: "<=15 min",
    1: "16-60 min",
    2: ">60 min",
}

MODEL_NAME_MAP = {
    "logistic_regression": "Logistic Regression",
    "random_forest": "Random Forest",
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
    df["model_clean"] = df["model"].map(MODEL_NAME_MAP).fillna(df["model"])
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
                "model_key": model_name,
                "Model": MODEL_NAME_MAP.get(model_name, model_name),
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
        text_auto=".3f",
        title="Baseline Model Comparison",
    )
    fig.update_layout(
        height=430,
        margin=dict(l=20, r=20, t=60, b=20),
        legend_title_text="Metric",
    )
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
    fig.update_layout(
        height=450,
        margin=dict(l=20, r=20, t=60, b=20),
    )
    return fig


def make_per_class_metric_chart(per_class_df: pd.DataFrame, metric: str):
    fig = px.bar(
        per_class_df,
        x="class",
        y=metric,
        color="model_clean",
        barmode="group",
        text_auto=".3f",
        title=f"Per-Class {metric.title()} Comparison",
        labels={
            "class": "Delay Class",
            "model_clean": "Model",
        },
    )
    fig.update_layout(
        height=430,
        margin=dict(l=20, r=20, t=60, b=20),
        legend_title_text="Model",
    )
    fig.update_yaxes(range=[0, 1])
    return fig


def make_feature_importance_chart(fi_df: pd.DataFrame, top_n: int):
    top_df = fi_df.head(top_n).iloc[::-1].copy()
    value_col = "importance_mean" if "importance_mean" in top_df.columns else "importance"

    fig = px.bar(
        top_df,
        x=value_col,
        y="clean_feature",
        orientation="h",
        title=f"Top {top_n} Explainability Features",
        labels={
            value_col: "Importance",
            "clean_feature": "Feature",
        },
    )
    fig.update_layout(
        height=520,
        margin=dict(l=20, r=20, t=60, b=20),
    )
    return fig


def make_confidence_histogram(df: pd.DataFrame):
    fig = px.histogram(
        df,
        x="prediction_confidence",
        nbins=20,
        title="Prediction Confidence Distribution",
    )
    fig.update_layout(
        height=360,
        margin=dict(l=20, r=20, t=60, b=20),
    )
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

    rename_map = {
        "AIRLINE": "Airline",
        "ORIGIN": "Origin",
        "DEST": "Destination",
        "dep_hour": "Departure Hour",
        "month": "Month",
        "actual_class": "Actual Class",
        "predicted_class": "Predicted Class",
        "prob_class_0": "P(Class <=15)",
        "prob_class_1": "P(Class 16-60)",
        "prob_class_2": "P(Class >60)",
        "prediction_confidence": "Confidence",
    }
    out = out.rename(columns=rename_map)
    return out


def inject_css():
    st.markdown(
        """
        <style>
        .block-container {
            padding-top: 1.5rem;
            padding-bottom: 2rem;
            max-width: 1350px;
        }
        .hero-box {
            padding: 1.2rem 1.3rem;
            border-radius: 16px;
            background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
            color: white;
            margin-bottom: 1rem;
        }
        .section-box {
            padding: 1rem 1rem 0.5rem 1rem;
            border-radius: 14px;
            background-color: rgba(120,120,120,0.07);
            margin-bottom: 1rem;
        }
        .small-note {
            font-size: 0.95rem;
            color: #cbd5e1;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def main():
    inject_css()

    report_json = load_training_report()
    per_class_df = load_per_class_metrics()
    fi_df = load_feature_importance()
    local_examples_df = load_local_examples()

    summary_df = build_model_summary_df(report_json)
    best_row = summary_df.sort_values("Macro F1", ascending=False).iloc[0]
    best_model_key = best_row["model_key"]
    best_model_name = best_row["Model"]

    st.sidebar.title("Navigation")
    section = st.sidebar.radio(
        "Go to section",
        [
            "Overview",
            "Model Performance",
            "Class-Level Analysis",
            "Explainability",
            "Prediction Examples",
            "Takeaways",
        ],
    )

    st.markdown(
        f"""
        <div class="hero-box">
            <h1 style="margin-bottom:0.4rem;">Flight Delay Severity Model Explorer</h1>
            <div class="small-note">
                A story-driven dashboard focused on multiclass flight delay prediction,
                model performance, and explainability.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if section == "Overview":
        st.markdown("## Project Story")
        st.write(
            """
            This dashboard is designed to answer two central questions:

            **1. How well can we predict flight delay severity?**  
            **2. Which factors most strongly drive the model’s predictions?**
            """
        )

        a, b, c, d = st.columns(4)
        a.metric("Best Model", best_model_name)
        b.metric("Best Accuracy", f"{best_row['Accuracy']:.3f}")
        c.metric("Best Macro F1", f"{best_row['Macro F1']:.3f}")
        d.metric("Best Weighted F1", f"{best_row['Weighted F1']:.3f}")

        st.markdown("### Why this matters")
        st.write(
            """
            Predicting delay severity is more useful than predicting a simple yes/no delay outcome,
            because it helps separate minor disruptions from more severe operational risk.
            """
        )

        st.markdown("### Delay Classes Used")
        col1, col2, col3 = st.columns(3)
        col1.info("**Class 0**\n\n<= 15 minutes")
        col2.warning("**Class 1**\n\n16 to 60 minutes")
        col3.error("**Class 2**\n\n> 60 minutes")

        st.markdown("### Dashboard Flow")
        st.write(
            """
            This app walks through the project in sequence:
            - compare models
            - inspect class-level strengths and weaknesses
            - explain why the model relies on certain features
            - review example predictions
            """
        )

    elif section == "Model Performance":
        st.markdown("## Model Performance")
        st.write(
            "We first compare the two baseline models and then inspect the stronger model through its confusion matrix."
        )

        left, right = st.columns([1.5, 1])
        with left:
            st.plotly_chart(make_model_comparison_chart(summary_df), width="stretch")
        with right:
            st.markdown("### Interpretation")
            st.write(
                """
                The strongest model is selected based on **Macro F1**, since this project
                cares about all delay severity classes, not just the majority class.
                """
            )
            st.dataframe(
                summary_df[["Model", "Accuracy", "Macro F1", "Weighted F1"]]
                .sort_values("Macro F1", ascending=False)
                .reset_index(drop=True),
                width="stretch",
            )

        st.markdown("### Confusion Matrix")
        model_option = st.selectbox(
            "Choose model",
            options=list(report_json["metrics_summary"].keys()),
            format_func=lambda x: MODEL_NAME_MAP.get(x, x),
            index=1 if "random_forest" in report_json["metrics_summary"] else 0,
        )

        cm_df = get_confusion_matrix_df(report_json, model_option)

        left, right = st.columns([1.45, 1])
        with left:
            st.plotly_chart(
                make_confusion_heatmap(cm_df, MODEL_NAME_MAP.get(model_option, model_option)),
                width="stretch",
            )
        with right:
            st.markdown("### What to look for")
            st.write(
                """
                - The **diagonal** shows correct predictions  
                - Off-diagonal cells show where the model confuses classes  
                - Severe-delay cases are usually the hardest to classify correctly
                """
            )

    elif section == "Class-Level Analysis":
        st.markdown("## Class-Level Analysis")
        st.write(
            """
            Overall accuracy can hide weak performance on minority classes.
            This section compares precision, recall, and F1 for each delay group.
            """
        )

        metric_choice = st.radio(
            "Choose metric",
            options=["precision", "recall", "f1-score"],
            horizontal=True,
        )

        left, right = st.columns([1.5, 1])
        with left:
            st.plotly_chart(
                make_per_class_metric_chart(per_class_df, metric_choice),
                width="stretch",
            )
        with right:
            st.markdown("### Why this matters")
            st.write(
                """
                - **Precision**: how reliable the predicted class is  
                - **Recall**: how many true cases the model captures  
                - **F1-score**: balance between precision and recall  
                """
            )
            st.write(
                """
                For this problem, minority classes matter because large delays are operationally more disruptive,
                even if they are less frequent.
                """
            )

        with st.expander("View detailed per-class metrics table"):
            table_df = per_class_df.copy()
            table_df = table_df.rename(
                columns={
                    "model_clean": "Model",
                    "class": "Delay Class",
                    "precision": "Precision",
                    "recall": "Recall",
                    "f1-score": "F1 Score",
                    "support": "Support",
                }
            )
            st.dataframe(
                table_df[["Model", "Delay Class", "Precision", "Recall", "F1 Score", "Support"]],
                width="stretch",
            )

    elif section == "Explainability":
        st.markdown("## Explainability")
        st.write(
            """
            This section explains **which features most influence model performance**.
            We use permutation-based feature importance to identify which variables matter most.
            """
        )

        top_n = st.slider("Top features to display", min_value=5, max_value=20, value=15)

        left, right = st.columns([1.5, 1])
        with left:
            st.plotly_chart(make_feature_importance_chart(fi_df, top_n), width="stretch")
        with right:
            st.markdown("### Reading the chart")
            st.write(
                """
                A feature is important if model performance drops when that feature is disrupted.
                Higher importance means the model depends more on that variable.
                """
            )
            st.markdown("### Main insight")
            st.write(
                """
                The strongest drivers are mostly related to:
                - scheduled timing
                - hour-of-day structure
                - trip duration and distance
                - calendar effects
                """
            )

        st.markdown("### Operational Interpretation")
        st.write(
            """
            These results suggest delay severity is not random. It is strongly tied to
            structured schedule patterns, temporal congestion, and route characteristics.
            """
        )

    elif section == "Prediction Examples":
        st.markdown("## Prediction Example Explorer")
        st.write(
            """
            This section shows example predictions so you can compare
            actual outcomes, predicted outcomes, and confidence levels.
            """
        )

        if local_examples_df.empty:
            st.warning("No local prediction examples file found. Run the explainability script first.")
        else:
            example_view = st.radio(
                "Choose example view",
                ["All saved examples", "Correct high-confidence examples", "Incorrect high-confidence examples"],
                horizontal=True,
            )

            filtered = local_examples_df.copy()

            if example_view == "Correct high-confidence examples":
                filtered = filtered[
                    filtered["actual_class"] == filtered["predicted_class"]
                ].sort_values("prediction_confidence", ascending=False)
            elif example_view == "Incorrect high-confidence examples":
                filtered = filtered[
                    filtered["actual_class"] != filtered["predicted_class"]
                ].sort_values("prediction_confidence", ascending=False)

            left, right = st.columns([1.45, 1])
            with left:
                st.dataframe(style_local_examples(filtered), width="stretch")
            with right:
                if "prediction_confidence" in filtered.columns and not filtered.empty:
                    st.plotly_chart(make_confidence_histogram(filtered), width="stretch")
                st.markdown("### Why examples matter")
                st.write(
                    """
                    Example-level views make the model easier to interpret in practical terms.
                    They show where the classifier is confident, where it struggles, and how
                    mistakes are distributed.
                    """
                )

    elif section == "Takeaways":
        st.markdown("## Final Takeaways")

        t1, t2 = st.columns(2)
        with t1:
            st.success(
                """
                **What worked**
                - Random Forest outperformed Logistic Regression
                - The model captures strong schedule and time-based patterns
                - Feature importance provides interpretable operational insight
                """
            )
        with t2:
            st.warning(
                """
                **What remains challenging**
                - Minority severe-delay classes are harder to predict
                - Overall performance is stronger than minority-class recall
                - Rare disruption events remain difficult for baseline models
                """
            )

        st.markdown("### Story in one sentence")
        st.write(
            """
            Flight delay severity can be predicted to a meaningful extent, and the model shows that
            timing structure, route characteristics, and calendar patterns are major drivers of disruption risk.
            """
        )

        st.markdown("### Recommendation")
        st.write(
            """
            This dashboard supports decision-making by showing not just whether the model performs well,
            but also **where it struggles** and **which factors matter most**.
            """
        )


if __name__ == "__main__":
    main()