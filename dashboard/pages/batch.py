"""Batch classification page — upload CSV, download results."""

import io

import httpx
import pandas as pd
import streamlit as st

st.header("Batch Classification")
st.markdown("Upload a CSV file with a text column to classify multiple documents at once.")

API_URL = "http://localhost:8000"

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("Preview")
    st.dataframe(df.head(10), use_container_width=True)

    text_columns = df.select_dtypes(include=["object"]).columns.tolist()
    if not text_columns:
        st.error("No text columns found in the uploaded CSV.")
    else:
        text_col = st.selectbox("Select the text column:", text_columns)
        max_rows = st.slider("Max rows to classify:", 1, min(len(df), 100), min(len(df), 50))

        if st.button("Classify Batch", type="primary"):
            texts = df[text_col].dropna().head(max_rows).tolist()

            with st.spinner(f"Classifying {len(texts)} documents..."):
                try:
                    response = httpx.post(
                        f"{API_URL}/classify/batch",
                        json={"texts": texts},
                        timeout=120.0,
                    )
                    response.raise_for_status()
                    result = response.json()

                    # Build results DataFrame
                    results_data = []
                    for i, pred in enumerate(result["predictions"]):
                        results_data.append(
                            {
                                "text": texts[i][:200] + "..." if len(texts[i]) > 200 else texts[i],
                                "predicted_class": pred["predicted_class"],
                                "confidence": pred["confidence"],
                            }
                        )

                    results_df = pd.DataFrame(results_data)

                    st.success(f"Classified {result['count']} documents!")
                    st.dataframe(results_df, use_container_width=True)

                    # Summary stats
                    st.subheader("Classification Summary")
                    col1, col2 = st.columns(2)

                    with col1:
                        class_counts = results_df["predicted_class"].value_counts()
                        st.bar_chart(class_counts)

                    with col2:
                        st.metric(
                            "Average Confidence",
                            f"{results_df['confidence'].mean():.1%}",
                        )
                        st.metric(
                            "Low Confidence (<50%)",
                            f"{(results_df['confidence'] < 0.5).sum()} docs",
                        )

                    # Download results
                    csv_buffer = io.StringIO()
                    results_df.to_csv(csv_buffer, index=False)
                    st.download_button(
                        "Download Results CSV",
                        csv_buffer.getvalue(),
                        "classification_results.csv",
                        "text/csv",
                    )

                except httpx.ConnectError:
                    st.error(
                        "Cannot connect to the API. Make sure the server is running: `make serve`"
                    )
                except Exception as e:
                    st.error(f"Error: {str(e)}")
