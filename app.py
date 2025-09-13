import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import lime.lime_tabular
import matplotlib.pyplot as plt

# -------------------- HELPER FUNCTIONS --------------------
def safe_float(val, default=0.0):
    """Safely convert a value to float."""
    try:
        return float(val)
    except (TypeError, ValueError):
        return default

def compute_credit_score(prob_default):
    score = int(850 - prob_default * 550)
    return max(300, min(850, score))

def ai_assistant(pred, prob, shap_values, lime_exp, applicant_aligned, credit_score):
    explanation_text = ""
    if pred[0] == 1:
        explanation_text += f"⚠️ High risk of default with probability {prob[0]:.2f}\n\n"
    else:
        explanation_text += f"✅ Low risk of default with probability {prob[0]:.2f}\n\n"

    explanation_text += f"💳 Calculated Credit Score: {credit_score}\n\n"

    # SHAP
    explanation_text += "📊 SHAP Insights:\n"
    top_features = shap_values.values[0].argsort()[-3:][::-1]
    for i in top_features:
        feature = NAME_MAP.get(applicant_aligned.columns[i], applicant_aligned.columns[i])
        value = applicant_aligned.iloc[0, i]
        shap_val = shap_values.values[0][i]
        explanation_text += f"- {feature} = {value} contributed {'positively' if shap_val>0 else 'negatively'}\n"

    # LIME
    explanation_text += "\n📌 LIME Explanation:\n"
    for feature, weight in lime_exp.as_list(label=1):
        explanation_text += f"- {feature} with weight {weight:.2f}\n"

    return explanation_text

# -------------------- FEATURES --------------------
FEATURES = [
    "income", "age", "loan_amnt", "credit_score", "last_pymnt_amnt",
    "total_pymnt", "recoveries", "funded_amnt", "total_rec_prncp",
    "total_pymnt_inv", "funded_amnt_inv"
]

NAME_MAP = {
    "income": "Monthly Income (USD)",
    "age": "Age (Years)",
    "loan_amnt": "Requested Loan Amount (USD)",
    "credit_score": "Credit Score",
    "last_pymnt_amnt": "Last Payment Amount (USD)",
    "total_pymnt": "Total Payments Made (USD)",
    "recoveries": "Recovered Amount (USD)",
    "funded_amnt": "Funded Loan Amount (USD)",
    "total_rec_prncp": "Total Principal Repaid (USD)",
    "total_pymnt_inv": "Total Payments to Investors (USD)",
    "funded_amnt_inv": "Funded Amount by Investors (USD)"
}

# -------------------- LOAD MODEL --------------------
mlp_model = joblib.load("mlp_model.pkl")
scaler = joblib.load("scaler.pkl")
df = pd.read_csv("credit_risk_dataset.csv")
feature_names = joblib.load("features.pkl")

st.title("🤖 AI Loan Assistant & Credit Risk Tool")

# -------------------- CHATBOT --------------------
if "chat_stage" not in st.session_state:
    st.session_state.chat_stage = 0
    st.session_state.chat_data = {}

questions = [
    ("Full Name", "text", "Please enter the applicant's full name"),
    ("National ID / Passport Number", "text", "Enter ID or passport number"),
    ("Age (Years)", "number", "Enter applicant's age"),
    ("Residential Address", "text", "Enter current residential address"),
    ("Contact Phone", "text", "Enter phone number"),
    ("Contact Email", "text", "Enter email address"),
    ("Monthly Income (USD)", "number", "Enter monthly income"),
    ("Employer Name", "text", "Enter employer name"),
    ("Employer Contact", "text", "Enter employer contact details"),
    ("Employment Type", "select", ["Permanent", "Contract", "Self-Employed"]),
    ("Employment Duration (years)", "number", "How long has the applicant worked at current employer?"),
    ("Existing Debts / Other Obligations (USD)", "number", "Enter total outstanding debts"),
    ("Previous Loan Repayments (USD)", "number", "Enter total repayments from past loans"),
    ("Requested Loan Amount (USD)", "number", "Enter requested loan amount")
]

# -------------------- CHATBOT SIDEBAR --------------------
st.sidebar.header("💬 Chatbot Responses")
if st.session_state.chat_data:
    for k, v in st.session_state.chat_data.items():
        st.sidebar.write(f"**{k}:** {v}")
else:
    st.sidebar.write("No responses yet.")

# -------------------- CHATBOT FLOW --------------------
if st.session_state.chat_stage < len(questions):
    q_label, q_type, q_prompt = questions[st.session_state.chat_stage]
    st.subheader(f"AI Assistant Question {st.session_state.chat_stage+1}/{len(questions)}")
    st.write(f"**AI Assistant:** {q_prompt}")

    if q_type=="text":
        answer = st.text_input("Your answer:", key=f"q{st.session_state.chat_stage}")
    elif q_type=="number":
        answer = st.number_input("Your answer:", min_value=0.0, step=1.0, key=f"q{st.session_state.chat_stage}")
    elif q_type=="select":
        answer = st.selectbox("Select option:", q_prompt, key=f"q{st.session_state.chat_stage}")

    if st.button("Next"):
        if answer is not None and answer!="":
            st.session_state.chat_data[q_label] = answer
            st.session_state.chat_stage += 1
        else:
            st.warning("Please provide an answer.")

# -------------------- APPLICANT DATA FORM --------------------
else:
    st.success("✅ Chatbot completed!")
    st.write("Summary of applicant info:")
    st.table(pd.DataFrame([st.session_state.chat_data]))

    st.header("📋 Applicant Data Input Form")

    # Prefill chatbot answers safely
    def prefill(key, default=0.0):
        return safe_float(st.session_state.chat_data.get(key), default)

    income = st.number_input(NAME_MAP["income"], value=prefill("Monthly Income (USD)", 0.0), step=100.0)
    age = st.number_input(NAME_MAP["age"], value=prefill("Age (Years)", 18), step=1.0)
    loan_amnt = st.number_input(NAME_MAP["loan_amnt"], value=prefill("Requested Loan Amount (USD)", 0.0), step=100.0)
    credit_score = st.number_input(NAME_MAP["credit_score"], value=prefill("Credit Score", 0.0), step=1.0)
    last_pymnt_amnt = st.number_input(NAME_MAP["last_pymnt_amnt"], value=prefill("Last Payment Amount (USD)", 0.0), step=50.0)
    total_pymnt = st.number_input(NAME_MAP["total_pymnt"], value=prefill("Total Payments Made (USD)", 0.0), step=100.0)
    recoveries = st.number_input(NAME_MAP["recoveries"], value=prefill("Recovered Amount (USD)", 0.0), step=10.0)
    funded_amnt = st.number_input(NAME_MAP["funded_amnt"], value=prefill("Funded Loan Amount (USD)", 0.0), step=100.0)
    total_rec_prncp = st.number_input(NAME_MAP["total_rec_prncp"], value=prefill("Total Principal Repaid (USD)", 0.0), step=100.0)
    total_pymnt_inv = st.number_input(NAME_MAP["total_pymnt_inv"], value=prefill("Total Payments to Investors (USD)", 0.0), step=100.0)
    funded_amnt_inv = st.number_input(NAME_MAP["funded_amnt_inv"], value=prefill("Funded Amount by Investors (USD)", 0.0), step=100.0)

    applicant_data = pd.DataFrame({
        "income":[income], "age":[age], "loan_amnt":[loan_amnt], "credit_score":[credit_score],
        "last_pymnt_amnt":[last_pymnt_amnt], "total_pymnt":[total_pymnt], "recoveries":[recoveries],
        "funded_amnt":[funded_amnt], "total_rec_prncp":[total_rec_prncp], "total_pymnt_inv":[total_pymnt_inv],
        "funded_amnt_inv":[funded_amnt_inv]
    })

    if st.button("Predict"):
        applicant_aligned = pd.DataFrame(columns=feature_names)
        applicant_aligned.loc[0] = 0
        for col in applicant_data.columns:
            if col in applicant_aligned.columns:
                applicant_aligned.loc[0,col] = applicant_data[col].values[0]

        scaled = scaler.transform(applicant_aligned)
        prob = mlp_model.predict_proba(scaled)[:,1]
        pred = mlp_model.predict(scaled)
        credit_score_val = compute_credit_score(prob[0])

        risk = "High Risk" if prob[0]>0.7 else "Medium Risk" if prob[0]>0.4 else "Low Risk"
        results_df = pd.DataFrame({
            "Prediction":["Default" if pred[0]==1 else "Non-Default"],
            "Probability of Default":[prob[0]],
            "Risk Category":[risk],
            "Credit Score":[credit_score_val],
            **{NAME_MAP[k]: applicant_data.iloc[0][k] for k in applicant_data.columns}
        }, index=[0])

        st.subheader("Prediction Results")
        st.write(results_df)

        # SHAP
        st.subheader("SHAP Explanation")
        explainer = shap.Explainer(mlp_model.predict, scaler.transform(df.drop("default_ind", axis=1)))
        shap_values = explainer(scaled)
        fig, ax = plt.subplots()
        shap.summary_plot(shap_values, applicant_aligned, feature_names=[NAME_MAP.get(f,f) for f in feature_names], plot_type="bar", show=False)
        st.pyplot(fig)

        # LIME
        st.subheader("LIME Explanation")
        lime_explainer = lime.lime_tabular.LimeTabularExplainer(
            training_data=scaler.transform(df.drop("default_ind", axis=1).values),
            feature_names=[NAME_MAP.get(f,f) for f in df.drop("default_ind", axis=1).columns.tolist()],
            class_names=["Non-Default","Default"],
            mode="classification"
        )
        explanation = lime_explainer.explain_instance(scaled[0], mlp_model.predict_proba, num_features=4)
        st.pyplot(explanation.as_pyplot_figure(label=1))

        st.subheader("🤖 AI Assistant Advice")
        st.write(ai_assistant(pred, prob, shap_values, explanation, applicant_aligned, credit_score_val))

# -------------------- RETRAIN --------------------
if st.sidebar.button("Retrain Model"):
    from sklearn.neural_network import MLPClassifier
    X = df.drop("default_ind",axis=1)
    y = df["default_ind"]
    scaler.fit(X)
    X_scaled = scaler.transform(X)
    mlp_model = MLPClassifier(hidden_layer_sizes=(100,50), max_iter=1000, random_state=42)
    mlp_model.fit(X_scaled,y)
    joblib.dump(mlp_model,"mlp_model.pkl")
    joblib.dump(scaler,"scaler.pkl")
    st.success("Model retrained successfully!")











































