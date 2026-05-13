"""Explainability page — LIME visualization for classification decisions."""

import httpx
import plotly.graph_objects as go
import streamlit as st

st.header("Model Explainability")
st.markdown(
    """
    Understand **why** the model makes its classification decisions using
    [LIME](https://github.com/marcotcr/lime) (Local Interpretable Model-agnostic Explanations).

    LIME perturbs the input text and observes how predictions change,
    identifying which words are most important for the classification.
    """
)

API_URL = "http://localhost:8000"

text_input = st.text_area(
    "Enter text to explain:",
    height=150,
    placeholder="Paste a complaint text here to see which words drive the classification...",
    value="I have been trying to get a refund on my credit card for an unauthorized charge of $450. The bank keeps telling me to wait but it has been over 60 days and nothing has been resolved.",
)

if st.button("Generate Explanation", type="primary"):
    if not text_input or len(text_input) < 10:
        st.warning("Please enter at least 10 characters.")
    else:
        with st.spinner("Generating LIME explanation (this may take 10-30 seconds)..."):
            try:
                response = httpx.post(
                    f"{API_URL}/classify",
                    json={"text": text_input, "explain": True},
                    timeout=120.0,
                )
                response.raise_for_status()
                result = response.json()

                st.success(
                    f"**Predicted:** {result['predicted_class']} "
                    f"(confidence: {result['confidence']:.1%})"
                )

                explanation = result.get("explanation")
                if explanation:
                    # Top features bar chart
                    st.subheader("Top Contributing Words")
                    features = explanation.get("top_features", [])

                    if features:
                        words = [f["word"] for f in features]
                        weights = [f["weight"] for f in features]
                        colors = [
                            "#2ecc71" if w > 0 else "#e74c3c" for w in weights
                        ]

                        fig = go.Figure(
                            go.Bar(
                                x=weights,
                                y=words,
                                orientation="h",
                                marker_color=colors,
                            )
                        )
                        fig.update_layout(
                            title=f"Word importance for '{result['predicted_class']}'",
                            xaxis_title="LIME Weight",
                            yaxis_title="",
                            height=max(300, len(features) * 35),
                            yaxis={"autorange": "reversed"},
                        )
                        st.plotly_chart(fig, use_container_width=True)

                        st.markdown("---")
                        st.markdown("**Legend:**")
                        st.markdown(
                            "- :green[Green bars]: Words that **support** the predicted class"
                        )
                        st.markdown(
                            "- :red[Red bars]: Words that push **against** the predicted class"
                        )

                    # Multi-class explanation
                    all_class_feats = explanation.get("all_class_features", {})
                    if all_class_feats:
                        st.subheader("Per-Class Word Importance")
                        selected_class = st.selectbox(
                            "Select class to inspect:",
                            sorted(all_class_feats.keys()),
                        )

                        class_feats = all_class_feats.get(selected_class, [])
                        if class_feats:
                            c_words = [f["word"] for f in class_feats]
                            c_weights = [f["weight"] for f in class_feats]
                            c_colors = [
                                "#2ecc71" if w > 0 else "#e74c3c"
                                for w in c_weights
                            ]

                            fig2 = go.Figure(
                                go.Bar(
                                    x=c_weights,
                                    y=c_words,
                                    orientation="h",
                                    marker_color=c_colors,
                                )
                            )
                            fig2.update_layout(
                                title=f"Word importance for '{selected_class}'",
                                xaxis_title="LIME Weight",
                                height=max(300, len(class_feats) * 35),
                                yaxis={"autorange": "reversed"},
                            )
                            st.plotly_chart(fig2, use_container_width=True)
                else:
                    st.warning("No explanation returned. Check API configuration.")

            except httpx.ConnectError:
                st.error(
                    "Cannot connect to the API. Make sure the server is running: `make serve`"
                )
            except Exception as e:
                st.error(f"Error: {str(e)}")
