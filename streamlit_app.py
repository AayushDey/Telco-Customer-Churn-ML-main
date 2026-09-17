"""
⚡ ChurnSight — Telco Customer Churn ML
Streamlit Cloud Application
=======================================
Interactive dashboard powered by XGBoost and SHAP explainability.
Deployable on Streamlit Community Cloud (share.streamlit.io).
"""

import os
import sys
import io
import pandas as pd
import numpy as np
import streamlit as st

# Ensure root directory is in python path
ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.serving.inference import predict

# ── Streamlit Page Configuration ──────────────────────────────────────────────
st.set_page_config(
    page_title="ChurnSight — Telco Customer Churn ML",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS for Premium Design ────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .main-header {
        background: linear-gradient(135deg, #1e1b4b 0%, #312e81 50%, #4338ca 100%);
        padding: 2rem;
        border-radius: 16px;
        color: white;
        margin-bottom: 2rem;
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.3);
    }
    
    .main-header h1 {
        font-size: 2.2rem;
        font-weight: 800;
        margin-bottom: 0.5rem;
        color: #ffffff;
    }
    
    .main-header p {
        font-size: 1.05rem;
        color: #cbd5e1;
        margin-bottom: 0;
    }
    
    .prediction-card {
        background: rgba(30, 41, 59, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 14px;
        padding: 1.5rem;
        backdrop-filter: blur(10px);
        margin-top: 1rem;
        margin-bottom: 1.5rem;
    }
    
    .badge-churn {
        background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
        color: white;
        padding: 0.4rem 1rem;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 0.95rem;
        display: inline-block;
        box-shadow: 0 4px 12px rgba(239, 68, 68, 0.4);
    }
    
    .badge-retained {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: white;
        padding: 0.4rem 1rem;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 0.95rem;
        display: inline-block;
        box-shadow: 0 4px 12px rgba(16, 185, 129, 0.4);
    }
    
    .stMetric {
        background: rgba(15, 23, 42, 0.6);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/isometric/100/combo-chart.png", width=64)
    st.title("⚡ ChurnSight")
    st.caption("AI-Powered Telco Customer Churn Prediction")
    
    st.markdown("---")
    st.markdown("### 🏆 Model Performance")
    col_sb1, col_sb2 = st.columns(2)
    with col_sb1:
        st.metric("ROC-AUC", "0.839")
    with col_sb2:
        st.metric("Recall", "82.4%")
        
    st.markdown("""
    **Core Stack:**
    - 🌲 **Model:** XGBoost Classifier
    - 🔍 **Explainability:** SHAP TreeExplainer
    - 🎯 **Decision Threshold:** 0.35
    - ⚡ **Framework:** Streamlit + Python 3.11
    """)
    
    st.markdown("---")
    st.markdown("### ⚡ Quick Presets")
    preset = st.selectbox(
        "Load sample profile:",
        ["Custom Input", "🔴 High Risk (Likely to Churn)", "🟢 Low Risk (Loyal Customer)", "🟡 Moderate / Borderline"],
        help="Quickly populate the form with realistic customer archetypes"
    )

# ── Main Header ──────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>⚡ ChurnSight — Telco Customer Churn Intelligence</h1>
    <p>Real-time machine learning prediction with SHAP explainability and targeted retention playbooks.</p>
</div>
""", unsafe_allow_html=True)

# Preset Values Configuration
preset_values = {
    "gender": "Female", "Partner": "No", "Dependents": "No",
    "tenure": 1, "PhoneService": "Yes", "MultipleLines": "No",
    "InternetService": "Fiber optic", "OnlineSecurity": "No", "OnlineBackup": "No",
    "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "Yes",
    "StreamingMovies": "Yes", "Contract": "Month-to-month", "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check", "MonthlyCharges": 85.50, "TotalCharges": 85.50
}

if preset == "🔴 High Risk (Likely to Churn)":
    preset_values = {
        "gender": "Female", "Partner": "No", "Dependents": "No",
        "tenure": 2, "PhoneService": "Yes", "MultipleLines": "No",
        "InternetService": "Fiber optic", "OnlineSecurity": "No", "OnlineBackup": "No",
        "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "Yes",
        "StreamingMovies": "Yes", "Contract": "Month-to-month", "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check", "MonthlyCharges": 90.45, "TotalCharges": 180.90
    }
elif preset == "🟢 Low Risk (Loyal Customer)":
    preset_values = {
        "gender": "Male", "Partner": "Yes", "Dependents": "Yes",
        "tenure": 60, "PhoneService": "Yes", "MultipleLines": "Yes",
        "InternetService": "DSL", "OnlineSecurity": "Yes", "OnlineBackup": "Yes",
        "DeviceProtection": "Yes", "TechSupport": "Yes", "StreamingTV": "No",
        "StreamingMovies": "No", "Contract": "Two year", "PaperlessBilling": "No",
        "PaymentMethod": "Credit card (automatic)", "MonthlyCharges": 65.00, "TotalCharges": 3900.00
    }
elif preset == "🟡 Moderate / Borderline":
    preset_values = {
        "gender": "Male", "Partner": "No", "Dependents": "No",
        "tenure": 12, "PhoneService": "Yes", "MultipleLines": "No",
        "InternetService": "DSL", "OnlineSecurity": "No", "OnlineBackup": "Yes",
        "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "Yes",
        "StreamingMovies": "No", "Contract": "One year", "PaperlessBilling": "Yes",
        "PaymentMethod": "Mailed check", "MonthlyCharges": 55.20, "TotalCharges": 662.40
    }

# ── Tabs Navigation ──────────────────────────────────────────────────────────
tab_single, tab_batch, tab_about = st.tabs([
    "🔮 Single Customer Prediction",
    "📂 Batch CSV Prediction",
    "📊 Model Details & Features"
])

# ═══════════════════════════════════════════════════════════════════════════════
# Tab 1: Single Customer Prediction
# ═══════════════════════════════════════════════════════════════════════════════
with tab_single:
    st.subheader("Customer Profile Inputs")
    st.caption("Adjust the attributes below to run instant inference and compute SHAP impact.")

    with st.form(key="prediction_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("##### 👤 Demographics")
            gender = st.selectbox("Gender", ["Female", "Male"], index=0 if preset_values["gender"] == "Female" else 1)
            partner = st.selectbox("Has Partner?", ["No", "Yes"], index=0 if preset_values["Partner"] == "No" else 1)
            dependents = st.selectbox("Has Dependents?", ["No", "Yes"], index=0 if preset_values["Dependents"] == "No" else 1)
            
            st.markdown("##### 💳 Billing & Contract")
            contract = st.selectbox(
                "Contract Type",
                ["Month-to-month", "One year", "Two year"],
                index=["Month-to-month", "One year", "Two year"].index(preset_values["Contract"])
            )
            paperless = st.selectbox(
                "Paperless Billing",
                ["No", "Yes"],
                index=0 if preset_values["PaperlessBilling"] == "No" else 1
            )
            payment = st.selectbox(
                "Payment Method",
                [
                    "Electronic check",
                    "Mailed check",
                    "Bank transfer (automatic)",
                    "Credit card (automatic)"
                ],
                index=[
                    "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
                ].index(preset_values["PaymentMethod"])
            )

        with col2:
            st.markdown("##### 📞 Phone & Internet Services")
            phone = st.selectbox("Phone Service", ["No", "Yes"], index=0 if preset_values["PhoneService"] == "No" else 1)
            multilines = st.selectbox(
                "Multiple Lines",
                ["No", "Yes", "No phone service"],
                index=["No", "Yes", "No phone service"].index(preset_values["MultipleLines"])
            )
            internet = st.selectbox(
                "Internet Service",
                ["Fiber optic", "DSL", "No"],
                index=["Fiber optic", "DSL", "No"].index(preset_values["InternetService"])
            )
            security = st.selectbox(
                "Online Security",
                ["No", "Yes", "No internet service"],
                index=["No", "Yes", "No internet service"].index(preset_values["OnlineSecurity"])
            )
            backup = st.selectbox(
                "Online Backup",
                ["No", "Yes", "No internet service"],
                index=["No", "Yes", "No internet service"].index(preset_values["OnlineBackup"])
            )
            device_prot = st.selectbox(
                "Device Protection",
                ["No", "Yes", "No internet service"],
                index=["No", "Yes", "No internet service"].index(preset_values["DeviceProtection"])
            )

        with col3:
            st.markdown("##### 📺 Add-ons & Charges")
            tech_support = st.selectbox(
                "Tech Support",
                ["No", "Yes", "No internet service"],
                index=["No", "Yes", "No internet service"].index(preset_values["TechSupport"])
            )
            stream_tv = st.selectbox(
                "Streaming TV",
                ["No", "Yes", "No internet service"],
                index=["No", "Yes", "No internet service"].index(preset_values["StreamingTV"])
            )
            stream_movies = st.selectbox(
                "Streaming Movies",
                ["No", "Yes", "No internet service"],
                index=["No", "Yes", "No internet service"].index(preset_values["StreamingMovies"])
            )
            
            st.markdown("##### ⏱️ Account Metrics")
            tenure = st.slider("Tenure (Months with company)", min_value=0, max_value=72, value=int(preset_values["tenure"]))
            monthly = st.number_input("Monthly Charges ($)", min_value=18.0, max_value=120.0, value=float(preset_values["MonthlyCharges"]), step=1.0)
            total = st.number_input("Total Charges ($)", min_value=18.0, max_value=9000.0, value=float(preset_values["TotalCharges"]), step=10.0)

        submit_btn = st.form_submit_button("⚡ Predict Churn Risk", use_container_width=True, type="primary")

    # When form is submitted (or on initial load with preset)
    customer_payload = {
        "gender": gender,
        "Partner": partner,
        "Dependents": dependents,
        "PhoneService": phone,
        "MultipleLines": multilines,
        "InternetService": internet,
        "OnlineSecurity": security,
        "OnlineBackup": backup,
        "DeviceProtection": device_prot,
        "TechSupport": tech_support,
        "StreamingTV": stream_tv,
        "StreamingMovies": stream_movies,
        "Contract": contract,
        "PaperlessBilling": paperless,
        "PaymentMethod": payment,
        "tenure": int(tenure),
        "MonthlyCharges": float(monthly),
        "TotalCharges": float(total),
    }

    try:
        result = predict(customer_payload)
        is_churn = result["prediction"] == "Likely to churn"
        probability = result["probability"]
        confidence = result["confidence"]
        shap_values = result.get("shap_values", [])

        st.markdown("### 🎯 Prediction Results")
        
        # Result summary metrics
        res_col1, res_col2, res_col3, res_col4 = st.columns(4)
        
        with res_col1:
            st.metric(
                label="Churn Probability",
                value=f"{probability * 100:.1f}%",
                delta="High Risk" if is_churn else "Low Risk",
                delta_color="inverse" if is_churn else "normal"
            )
        with res_col2:
            st.metric(
                label="Prediction",
                value=result["prediction"]
            )
        with res_col3:
            st.metric(
                label="Confidence Level",
                value=confidence
            )
        with res_col4:
            st.metric(
                label="Threshold",
                value="0.35 (Optimized)",
                help="Tuned for high churn recall (82.4%)"
            )

        # Visual progress bar
        st.progress(min(max(probability, 0.0), 1.0))

        # SHAP Explainability & Recommendations
        st.markdown("---")
        shap_col, rec_col = st.columns([3, 2])

        with shap_col:
            st.subheader("🔍 SHAP Feature Explainability")
            st.caption("Which customer attributes drove this prediction higher or lower?")
            
            if shap_values:
                shap_df = pd.DataFrame(shap_values)
                # Sort for clean display
                shap_df["abs_impact"] = shap_df["impact"].abs()
                shap_df = shap_df.sort_values(by="abs_impact", ascending=True)

                # Color red if increases risk, green if reduces risk
                colors = ["#ef4444" if x > 0 else "#10b981" for x in shap_df["impact"]]

                chart_data = pd.DataFrame({
                    "Feature": shap_df["feature"],
                    "Impact": shap_df["impact"]
                }).set_index("Feature")

                st.bar_chart(chart_data, horizontal=True)
                st.info("💡 **Red/Positive values** push towards Churn. **Green/Negative values** push towards Retention.")
            else:
                st.warning("SHAP values unavailable for this model artifact.")

        with rec_col:
            st.subheader("💡 Retention Playbook")
            st.caption("Actionable interventions based on risk factors:")
            
            if is_churn:
                if contract == "Month-to-month":
                    st.warning("🏷️ **Contract Lock-in Offer:** Month-to-month customers represent the #1 churn risk. Offer 20% discount on a 1-year agreement.")
                if security == "No" or tech_support == "No":
                    st.info("🛡️ **Tech & Security Bundle:** Add 3 months of complimentary Online Security & Tech Support to build retention stickiness.")
                if payment == "Electronic check":
                    st.info("💳 **Auto-Pay Incentive:** Offer a $10 bill credit to switch from Electronic Check to Auto-Pay credit card/bank transfer.")
                if tenure < 6:
                    st.error("📞 **VIP Onboarding Call:** Customer is in critical early lifecycle (< 6 months). Schedule a proactive check-in call.")
            else:
                st.success("✅ **Customer is stable:** Churn probability is low. Target with loyalty rewards or optional service cross-sells.")

    except Exception as e:
        st.error(f"Prediction Error: {e}")

# ═══════════════════════════════════════════════════════════════════════════════
# Tab 2: Batch CSV Prediction
# ═══════════════════════════════════════════════════════════════════════════════
with tab_batch:
    st.subheader("📂 Batch Customer CSV Inference")
    st.markdown("Upload a CSV file containing customer rows to generate bulk predictions, probabilities, and confidence scores.")

    # Sample template download
    sample_data = pd.DataFrame([
        {
            "gender": "Female", "Partner": "Yes", "Dependents": "No", "PhoneService": "Yes",
            "MultipleLines": "No", "InternetService": "Fiber optic", "OnlineSecurity": "No",
            "OnlineBackup": "No", "DeviceProtection": "No", "TechSupport": "No",
            "StreamingTV": "No", "StreamingMovies": "No", "Contract": "Month-to-month",
            "PaperlessBilling": "Yes", "PaymentMethod": "Electronic check", "tenure": 1,
            "MonthlyCharges": 70.35, "TotalCharges": 70.35
        },
        {
            "gender": "Male", "Partner": "No", "Dependents": "No", "PhoneService": "Yes",
            "MultipleLines": "Yes", "InternetService": "DSL", "OnlineSecurity": "Yes",
            "OnlineBackup": "Yes", "DeviceProtection": "Yes", "TechSupport": "Yes",
            "StreamingTV": "No", "StreamingMovies": "No", "Contract": "Two year",
            "PaperlessBilling": "No", "PaymentMethod": "Credit card (automatic)", "tenure": 65,
            "MonthlyCharges": 53.85, "TotalCharges": 3500.25
        }
    ])
    
    csv_buffer = io.StringIO()
    sample_data.to_csv(csv_buffer, index=False)
    st.download_button(
        label="📥 Download Sample Batch CSV Template",
        data=csv_buffer.getvalue(),
        file_name="telco_batch_sample.csv",
        mime="text/csv",
    )

    uploaded_file = st.file_uploader("Upload customer CSV file", type=["csv"])

    if uploaded_file is not None:
        try:
            batch_df = pd.read_csv(uploaded_file)
            st.success(f"Uploaded CSV with {len(batch_df)} rows and {len(batch_df.columns)} columns.")

            if st.button("🚀 Run Batch Prediction", type="primary"):
                progress_bar = st.progress(0)
                predictions = []
                probabilities = []
                confidences = []

                for idx, row in batch_df.iterrows():
                    row_dict = row.to_dict()
                    try:
                        res = predict(row_dict)
                        predictions.append(res["prediction"])
                        probabilities.append(res["probability"])
                        confidences.append(res["confidence"])
                    except Exception:
                        predictions.append("Error")
                        probabilities.append(0.5)
                        confidences.append("Error")
                    progress_bar.progress((idx + 1) / len(batch_df))

                batch_df["Prediction"] = predictions
                batch_df["Churn_Probability"] = probabilities
                batch_df["Confidence"] = confidences

                # Summary Metrics
                churn_count = sum(1 for p in predictions if p == "Likely to churn")
                avg_prob = np.mean(probabilities)
                
                c1, c2, c3 = st.columns(3)
                c1.metric("Total Customers Evaluated", len(batch_df))
                c2.metric("High Risk / Churners", f"{churn_count} ({(churn_count / len(batch_df))*100:.1f}%)")
                c3.metric("Average Churn Probability", f"{avg_prob * 100:.1f}%")

                # Show preview table
                st.dataframe(batch_df, use_container_width=True)

                # Export button
                out_csv = io.StringIO()
                batch_df.to_csv(out_csv, index=False)
                st.download_button(
                    label="📥 Download Predictions CSV",
                    data=out_csv.getvalue(),
                    file_name="churn_predictions_results.csv",
                    mime="text/csv"
                )

        except Exception as err:
            st.error(f"Error processing uploaded CSV: {err}")

# ═══════════════════════════════════════════════════════════════════════════════
# Tab 3: Model Details & Methodology
# ═══════════════════════════════════════════════════════════════════════════════
with tab_about:
    st.subheader("Model Architecture & Technical Documentation")
    st.markdown("""
    ### 🔬 Machine Learning Pipeline
    
    1. **Data Preprocessing & Validation**:
       - Handled whitespace coercion and missing values in `TotalCharges`.
       - Binary encoding applied to 5 binary features with deterministic dictionary mappings.
       - One-hot encoding with `drop_first=True` applied to multi-class categoricals.
       - Validated with **Great Expectations** assertions.
    
    2. **Classifier & Hyperparameters**:
       - Algorithm: **XGBoost (Extreme Gradient Boosting)**
       - Imbalance Handling: `scale_pos_weight` tuned to balance churn class representation.
       - Serving format: MLflow model format with scikit-learn compatibility.
       - Performance: **ROC-AUC = 0.839**, **Recall = 82.4%** at decision threshold **0.35**.
    
    3. **SHAP Explainability**:
       - TreeExplainer calculates local Shapley contributions for each feature in real-time.
       - Converts raw one-hot encoded variables into user-friendly diagnostic bars.
    """)
