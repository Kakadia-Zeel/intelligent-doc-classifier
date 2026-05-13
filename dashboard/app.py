"""Streamlit dashboard for Intelligent Document Classifier."""

import streamlit as st

st.set_page_config(
    page_title="Intelligent Document Classifier",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("Intelligent Document Classifier")
st.markdown(
    """
    Production-grade ML system for classifying consumer financial documents
    into product categories using transformer-based NLP.

    ### Navigation
    Use the sidebar to navigate between pages:
    - **Classify**: Classify a single document with confidence scores
    - **Batch**: Upload a CSV file for batch classification
    - **Explainability**: See which words drive classification decisions (LIME)
    - **Monitoring**: View model health, drift metrics, and prediction stats
    """
)

st.sidebar.success("Select a page above.")

st.markdown("---")
st.markdown(
    "Built by [Zeel Kakadia](https://github.com/kakadia-zeel) | "
    "Powered by DistilBERT + FastAPI"
)
