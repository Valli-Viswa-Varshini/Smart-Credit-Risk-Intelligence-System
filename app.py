import streamlit as st
import pandas as pd
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
import joblib
import numpy as np

# Load model
@st.cache_resource
def load_model():
    return joblib.load("xgb_model.pkl")

model = load_model()

st.title("SCRIS - Smart Credit Risk Intelligence System")
st.write("""
Upload applicant data CSV (preprocessed & one-hot encoded).  
Your file **does not need to match all columns exactly** — we’ll auto-fix it.
""")

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

# Fix uploaded data to match model features
def align_features_to_model(input_df, model_features):
    # Add missing columns as 0
    for col in model_features:
        if col not in input_df.columns:
            input_df[col] = 0
    # Drop extra columns
    input_df = input_df[model_features]
    return input_df

if uploaded_file:
    try:
        data = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"❌ Error reading CSV file: {e}")
        st.stop()

    st.write("📄 Uploaded Data Preview:")
    st.dataframe(data.head())

    model_features = model.get_booster().feature_names
    if model_features is None:
        st.error("❌ Could not read model feature names.")
        st.stop()

    # Fix columns
    # Convert string columns to numeric using one-hot encoding
    data_encoded = pd.get_dummies(data)

    # Align with model features
    input_fixed = align_features_to_model(data_encoded.copy(), model_features)


    if st.button("🔍 Predict Defaults"):
        try:
            preds = model.predict(input_fixed)
            pred_probs = model.predict_proba(input_fixed)[:, 1]
            results = data.copy()
            results['Predicted Default'] = preds
            results['Default Probability'] = pred_probs.round(3)

            st.success("✅ Prediction completed!")
            st.dataframe(results[['Predicted Default', 'Default Probability']])

            # SHAP
            st.write("### 🧠 SHAP Summary Plot")
            explainer = shap.TreeExplainer(model)
            sample_data = input_fixed.sample(min(100, len(input_fixed)), random_state=42)
            shap_values = explainer.shap_values(sample_data)

            fig, ax = plt.subplots(figsize=(10, 6))
            shap.summary_plot(shap_values, sample_data, show=False)
            st.pyplot(fig)

            # Individual prediction
            st.write("### 🔍 Explain Individual Prediction")
            idx = st.number_input("Choose row index:", min_value=0, max_value=len(sample_data)-1, value=0)

            st.write("SHAP Waterfall Plot for Row:", idx)

            # Use waterfall plot instead of force_plot
            fig_explain = shap.plots._waterfall.waterfall_legacy(
                explainer.expected_value,
                shap_values[idx],
                feature_names=sample_data.columns
            )

            st.pyplot(fig_explain)


        except Exception as e:
            st.error(f"🚨 Error during prediction or SHAP explanation: {e}")

else:
    st.info("📂 Please upload a properly preprocessed CSV to start.")
