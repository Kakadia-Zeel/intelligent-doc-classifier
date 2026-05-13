"""Single document classification page."""

import streamlit as st
import httpx
import plotly.express as px
import pandas as pd

st.header("Classify Document")
st.markdown("Enter a consumer complaint text to classify it into a product category.")

API_URL = "http://localhost:8000"

# Example texts
EXAMPLES = {
    "Credit Card": "I noticed unauthorized charges on my credit card statement. Someone made purchases totaling $2,500 that I did not authorize. I called the bank but they said I need to file a dispute.",
    "Mortgage": "My mortgage servicer keeps sending me incorrect statements showing a different balance than what I owe. I have made all my payments on time but they are reporting late payments to the credit bureaus.",
    "Debt Collection": "A debt collector is calling me multiple times a day about a medical bill that I already paid. I sent them proof of payment but they continue to harass me and threaten legal action.",
    "Student Loan": "I applied for an income-driven repayment plan for my student loans three months ago and still have not received a response. Meanwhile, my payments remain at the original higher amount.",
    "Banking": "My bank charged me overdraft fees even though I opted out of overdraft protection. When I called to dispute, they refused to refund the fees.",
}

# Example selector
example = st.selectbox(
    "Try an example:", ["Custom input"] + list(EXAMPLES.keys())
)

if example != "Custom input":
    default_text = EXAMPLES[example]
else:
    default_text = ""

text_input = st.text_area(
    "Complaint Text",
    value=default_text,
    height=200,
    placeholder="Enter the document text to classify...",
)

col1, col2 = st.columns(2)
with col1:
    include_explanation = st.checkbox("Include LIME explanation (slower)", value=False)
with col2:
    classify_button = st.button("Classify", type="primary", use_container_width=True)

if classify_button and text_input:
    with st.spinner("Classifying..."):
        try:
            response = httpx.post(
                f"{API_URL}/classify",
                json={"text": text_input, "explain": include_explanation},
                timeout=60.0,
            )
            response.raise_for_status()
            result = response.json()

            # Display results
            st.success(f"**Predicted Category:** {result['predicted_class']}")
            st.metric("Confidence", f"{result['confidence']:.1%}")

            # Probability chart
            st.subheader("Class Probabilities")
            probs_df = pd.DataFrame(
                list(result["probabilities"].items()),
                columns=["Category", "Probability"],
            ).sort_values("Probability", ascending=True)

            fig = px.bar(
                probs_df,
                x="Probability",
                y="Category",
                orientation="h",
                color="Probability",
                color_continuous_scale="Blues",
            )
            fig.update_layout(
                height=400,
                showlegend=False,
                xaxis_title="Probability",
                yaxis_title="",
            )
            st.plotly_chart(fig, use_container_width=True)

            # LIME explanation
            if result.get("explanation"):
                st.subheader("Explanation (LIME)")
                explanation = result["explanation"]

                st.markdown("**Top contributing words:**")
                features = explanation.get("top_features", [])
                for feat in features:
                    word = feat["word"]
                    weight = feat["weight"]
                    direction = "supports" if weight > 0 else "opposes"
                    color = "green" if weight > 0 else "red"
                    st.markdown(
                        f"- :{color}[**{word}**] ({direction}, weight: {weight:.4f})"
                    )

        except httpx.ConnectError:
            st.error(
                "Cannot connect to the API. Make sure the server is running: `make serve`"
            )
        except Exception as e:
            st.error(f"Error: {str(e)}")

elif classify_button:
    st.warning("Please enter some text to classify.")
