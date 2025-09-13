import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import lime.lime_tabular
import matplotlib.pyplot as plt

# -------------------- FEATURE DEFINITIONS --------------------
FEATURES = [
    "income", "age", "loan_amnt", "last_pymnt_amnt",
    "total_pymnt", "recoveries", "funded_amnt", "total_rec_prncp",
    "total_pymnt_inv", "funded_amnt_inv"
]

# Human-readable mapping
NAME_MAP = {
    "income": "Monthly Income (USD)",
    "age": "Age (Years)",
    "loan_amnt": "Requested Loan Amount (USD)",
    "last_pymnt_amnt": "Last Payment Amount (USD)",
    "total_pymnt": "Total Payments Made (USD)",
    "recoveries": "Recovered Amount (USD)",
    "funded_amnt": "Funded Loan Amount (USD)",
    "total_rec_prncp": "Total Principal Repaid (USD)",
    "total_pymnt_inv": "Total Payments to Investors (USD)",
    "funded_amnt_inv": "Funded Amount by Investors (USD)"
}

# -------------------- CREDIT SCORE CALCULATION --------------------
def compute_credit_score(prob_default):
    """Convert probability of default into a credit score on a 300–850 scale."""
    score = int(850 - prob_default * 550)
    return max(300, min(850, score))

# -------------------- AI ASSISTANT FUNCTION --------------------
def ai_assistant(pred, prob, shap_values, lime_exp, applicant_aligned, credit_score):
    explanation_text = ""

    # 1. Prediction summary
    if pred[0] == 1:
        explanation_text += f"⚠️ The model predicts this applicant is at HIGH RISK of default with probability {prob[0]:.2f}.\n\n"
    else:
        explanation_text += f"✅ The model predicts this applicant is at LOW RISK of default with probability {prob[0]:.2f}.\n\n"

    explanation_text += f"💳 **Calculated Credit Score:** {credit_score}\n\n"

    # 2. SHAP Insights
    explanation_text += "📊 **SHAP Insights:**\n"
    important_features = shap_values.values[0].argsort()[-3:][::-1]  # top 3
    for i in important_features:
        feature = NAME_MAP.get(applicant_aligned.columns[i], applicant_aligned.columns[i])
        value = applicant_aligned.iloc[0, i]
        shap_val = shap_values.values[0][i]
        explanation_text += f"- {feature} = {value} contributed {'positively' if shap_val > 0 else 'negatively'} to the risk score.\n"

    # 3. LIME Insights
    explanation_text += "\n📌 **LIME Explanation:**\n"
    for feature, weight in lime_exp.as_list(label=1):
        explanation_text += f"- {feature} with weight {weight:.2f}\n"

    # 4. Policy & Loan Advice
    explanation_text += "\n💡 **Policy Recommendations:**\n"
    if pred[0] == 0:  # Non-default
        explanation_text += "- The applicant qualifies for a personal loan.\n"
        if prob[0] < 0.3:
            explanation_text += "- Recommended: Higher loan amount with longer repayment period (24–36 months).\n"
        elif prob[0] < 0.6:
            explanation_text += "- Recommended: Medium loan amount with repayment period of 12–24 months.\n"
        else:
            explanation_text += "- Recommended: Lower loan amount with strict monitoring and shorter repayment period (6–12 months).\n"
    else:
        explanation_text += "- The applicant should be carefully monitored or rejected due to high risk of default.\n"
        explanation_text += "- Recommend financial literacy training or credit repair program before loan approval.\n"

    return explanation_text

# -------------------- LOAD MODEL & DATA --------------------
mlp_model = joblib.load("mlp_model.pkl")
scaler = joblib.load("scaler.pkl")
df = pd.read_csv("credit_risk_dataset.csv")
feature_names = joblib.load("features.pkl")  # original order of features

st.title("📊 Credit Risk Assessment Tool")

# Sidebar options
st.sidebar.header("Applicant Data Input")

# Choose input mode
input_mode = st.sidebar.radio("Choose input mode:", ["Manual Entry", "Upload CSV"])

applicant_data = None
if input_mode == "Manual Entry":
    # Create sidebar inputs for all features EXCEPT credit_score
    income = st.sidebar.number_input(NAME_MAP["income"], min_value=0.0, step=100.0)
    age = st.sidebar.number_input(NAME_MAP["age"], min_value=18, max_value=100, step=1)
    loan_amnt = st.sidebar.number_input(NAME_MAP["loan_amnt"], min_value=0.0, step=100.0)
    last_pymnt_amnt = st.sidebar.number_input(NAME_MAP["last_pymnt_amnt"], min_value=0.0, step=50.0)
    total_pymnt = st.sidebar.number_input(NAME_MAP["total_pymnt"], min_value=0.0, step=100.0)
    recoveries = st.sidebar.number_input(NAME_MAP["recoveries"], min_value=0.0, step=10.0)
    funded_amnt = st.sidebar.number_input(NAME_MAP["funded_amnt"], min_value=0.0, step=100.0)
    total_rec_prncp = st.sidebar.number_input(NAME_MAP["total_rec_prncp"], min_value=0.0, step=100.0)
    total_pymnt_inv = st.sidebar.number_input(NAME_MAP["total_pymnt_inv"], min_value=0.0, step=100.0)
    funded_amnt_inv = st.sidebar.number_input(NAME_MAP["funded_amnt_inv"], min_value=0.0, step=100.0)

    applicant_data = pd.DataFrame({
        "income": [income],
        "age": [age],
        "loan_amnt": [loan_amnt],
        "last_pymnt_amnt": [last_pymnt_amnt],
        "total_pymnt": [total_pymnt],
        "recoveries": [recoveries],
        "funded_amnt": [funded_amnt],
        "total_rec_prncp": [total_rec_prncp],
        "total_pymnt_inv": [total_pymnt_inv],
        "funded_amnt_inv": [funded_amnt_inv]
    })

elif input_mode == "Upload CSV":
    uploaded_file = st.sidebar.file_uploader("Upload Applicant CSV", type=["csv"])
    if uploaded_file:
        applicant_data = pd.read_csv(uploaded_file)

# -------------------- PREDICT BUTTON --------------------
if st.sidebar.button("Predict"):
    if applicant_data is not None:
        # Align applicant data with training features
        applicant_aligned = pd.DataFrame(columns=feature_names)
        applicant_aligned.loc[0] = 0
        for col in applicant_data.columns:
            if col in applicant_aligned.columns:
                applicant_aligned.loc[0, col] = applicant_data[col].values[0]

        # Scale aligned data
        scaled = scaler.transform(applicant_aligned)

        # Prediction
        prob = mlp_model.predict_proba(scaled)[:, 1]
        pred = mlp_model.predict(scaled)

        # Results
        results = []
        for i in range(len(prob)):
            credit_score = compute_credit_score(prob[i])
            if prob[i] > 0.7:
                risk = "High Risk"
            elif prob[i] > 0.4:
                risk = "Medium Risk"
            else:
                risk = "Low Risk"
            # Replace technical names with human-readable in output
            applicant_readable = {NAME_MAP.get(k, k): applicant_aligned.iloc[i][k] for k in FEATURES}
            results.append({
                "Prediction": "Default" if pred[i] == 1 else "Non-Default",
                "Probability of Default": prob[i],
                "Risk Category": risk,
                "Calculated Credit Score": credit_score,
                **applicant_readable
            })

        results_df = pd.DataFrame(results)
        st.subheader("Prediction Results")
        st.write(results_df)

        # ✅ SHAP Explanation
        st.subheader("SHAP Explanation")
        explainer = shap.Explainer(mlp_model.predict, scaler.transform(df.drop("default_ind", axis=1)))
        shap_values = explainer(scaled)
        fig, ax = plt.subplots()
        shap.summary_plot(
            shap_values, applicant_aligned,
            feature_names=[NAME_MAP.get(f, f) for f in feature_names],
            plot_type="bar", show=False
        )
        st.pyplot(fig)

        # ✅ LIME Explanation
        st.subheader("LIME Explanation")
        lime_explainer = lime.lime_tabular.LimeTabularExplainer(
            training_data=scaler.transform(df.drop("default_ind", axis=1).values),
            feature_names=[NAME_MAP.get(f, f) for f in df.drop("default_ind", axis=1).columns.tolist()],
            class_names=["Non-Default", "Default"],
            mode="classification"
        )
        explanation = lime_explainer.explain_instance(
            data_row=scaled[0],
            predict_fn=mlp_model.predict_proba,
            num_features=4
        )
        fig = explanation.as_pyplot_figure(label=1)
        st.pyplot(fig)

        # ✅ AI Assistant Advice
        st.subheader("🤖 AI Assistant Advice")
        assistant_text = ai_assistant(pred, prob, shap_values, explanation, applicant_aligned, results_df["Calculated Credit Score"].iloc[0])
        st.write(assistant_text)

    else:
        st.warning("Please provide applicant data (manual entry or CSV).")

# -------------------- RETRAIN OPTION --------------------
if st.sidebar.button("Retrain Model"):
    from sklearn.neural_network import MLPClassifier

    X = df.drop("default_ind", axis=1)
    y = df["default_ind"]

    scaler.fit(X)
    X_scaled = scaler.transform(X)

    mlp_model = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42)
    mlp_model.fit(X_scaled, y)

    joblib.dump(mlp_model, "mlp_model.pkl")
    joblib.dump(scaler, "scaler.pkl")

    st.success("Model retrained successfully with updated dataset!")



























