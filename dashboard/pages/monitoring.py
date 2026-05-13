"""Model monitoring page — drift metrics, prediction stats, model health."""

import json
from pathlib import Path

import httpx
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import streamlit as st

st.header("Model Monitoring")
st.markdown("Track model health, prediction distribution, and data drift.")

API_URL = "http://localhost:8000"

# Health check
st.subheader("Service Health")
try:
    health = httpx.get(f"{API_URL}/health", timeout=5.0).json()
    col1, col2, col3 = st.columns(3)
    with col1:
        status_color = "green" if health["status"] == "healthy" else "red"
        st.markdown(f"**Status:** :{status_color}[{health['status']}]")
    with col2:
        st.markdown(f"**Model Loaded:** {'Yes' if health['model_loaded'] else 'No'}")
    with col3:
        st.markdown(f"**Model Type:** {health['model_type']}")
except httpx.ConnectError:
    st.error("API is not reachable. Start the server with `make serve`")
    st.stop()
except Exception as e:
    st.error(f"Health check failed: {e}")
    st.stop()

st.markdown("---")

# Live metrics from API
st.subheader("Prediction Metrics")
try:
    metrics = httpx.get(f"{API_URL}/metrics", timeout=5.0).json()

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Predictions", metrics["total_predictions"])
    with col2:
        st.metric("Avg Confidence", f"{metrics['avg_confidence']:.1%}")
    with col3:
        drift_status = "DETECTED" if metrics["drift_detected"] else "None"
        drift_color = "red" if metrics["drift_detected"] else "green"
        st.markdown(f"**Drift:** :{drift_color}[{drift_status}]")

    # Prediction distribution
    if metrics["predictions_per_class"]:
        st.subheader("Prediction Distribution")
        pred_df = pd.DataFrame(
            list(metrics["predictions_per_class"].items()),
            columns=["Category", "Count"],
        ).sort_values("Count", ascending=False)

        fig = px.bar(
            pred_df,
            x="Category",
            y="Count",
            color="Count",
            color_continuous_scale="Viridis",
        )
        fig.update_layout(height=400, xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)

    # Drift details
    if metrics.get("drift_details"):
        st.subheader("Drift Details")
        details = metrics["drift_details"]
        st.metric("Max Class Shift", f"{details['max_shift']:.1%}")

        if details.get("class_shifts"):
            shifts_data = []
            for cls, shift_info in details["class_shifts"].items():
                shifts_data.append(
                    {
                        "Category": cls,
                        "Reference": shift_info["reference"],
                        "Current": shift_info["current"],
                        "Shift": shift_info["shift"],
                    }
                )

            shifts_df = pd.DataFrame(shifts_data).sort_values(
                "Shift", ascending=False
            )
            st.dataframe(shifts_df, use_container_width=True)

except Exception as e:
    st.warning(f"Could not load metrics: {e}")

st.markdown("---")

# Reference distribution (from training)
st.subheader("Reference Distribution (Training Data)")
ref_path = Path("data/reference/reference_distribution.json")
if ref_path.exists():
    with open(ref_path) as f:
        ref_data = json.load(f)

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Training Samples", ref_data["n_samples"])
        st.metric("Avg Text Length", f"{ref_data['text_length_mean']:.0f} chars")
    with col2:
        st.metric("Avg Word Count", f"{ref_data['word_count_mean']:.0f}")

    if ref_data.get("label_distribution"):
        label_df = pd.DataFrame(
            list(ref_data["label_distribution"].items()),
            columns=["Category", "Count"],
        ).sort_values("Count", ascending=False)

        fig = px.pie(label_df, names="Category", values="Count", title="Training Label Distribution")
        st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Reference distribution not found. Train the model first: `make train`")

# Saved metrics
st.subheader("Model Performance (Test Set)")
metrics_dir = Path("artifacts/metrics")
if metrics_dir.exists():
    for metrics_file in sorted(metrics_dir.glob("*.json")):
        with open(metrics_file) as f:
            saved_metrics = json.load(f)

        model_name = metrics_file.stem.replace("_metrics", "").replace("_", " ").title()
        with st.expander(f"{model_name}", expanded=False):
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Accuracy", f"{saved_metrics.get('accuracy', 0):.4f}")
            with col2:
                st.metric("Macro F1", f"{saved_metrics.get('macro_f1', 0):.4f}")
            with col3:
                st.metric("Weighted F1", f"{saved_metrics.get('weighted_f1', 0):.4f}")

            # Confusion matrix
            cm = saved_metrics.get("confusion_matrix")
            if cm:
                label_names = list(saved_metrics.get("per_class", {}).keys())
                fig = px.imshow(
                    cm,
                    labels=dict(x="Predicted", y="Actual", color="Count"),
                    x=label_names,
                    y=label_names,
                    color_continuous_scale="Blues",
                    aspect="auto",
                )
                fig.update_layout(height=500, title="Confusion Matrix")
                st.plotly_chart(fig, use_container_width=True)
else:
    st.info("No model metrics found. Train the model first: `make train`")
