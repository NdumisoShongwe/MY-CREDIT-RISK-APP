import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import lime.lime_tabular
import plotly.graph_objects as go
import plotly.express as px

# -------------------- HELPER FUNCTIONS --------------------
def safe_float(val, default=0.0):
    try:
        return float(val)
    except (TypeError, ValueError):
        return default

def compute_credit_score(prob_default):
    """Convert probability of default into a 300–850 credit score"""
    return int(max(300, min(850, 850 - prob_default*550)))

def ai_assistant(pred, prob, shap_values, lime_exp, applicant_aligned, credit_score):
    text = ""
    # Risk prediction
    if pred[0]==1:
        text += f"⚠️ This applicant is HIGH RISK (Probability of default: {prob[0]:.2f}).\n\n"
        text += "- Policy Advice: Reject or monitor closely. Consider financial literacy training.\n"
    else:
        text += f"✅ LOW RISK (Probability of default: {prob[0]:.2f}).\n\n"
        text += "- Policy Advice: Eligible for loan.\n"
        if prob[0]<0.3:
            text += "- Recommend higher loan amount with longer repayment term.\n"
        elif prob[0]<0.6:
            text += "- Medium loan amount with standard repayment term.\n"
        else:
            text += "- Lower loan amount, shorter repayment term, monitor payments.\n"

    text += f"💳 Calculated Credit Score: {credit_score}\n\n"

    # SHAP insights summary
    text += "📊 Top SHAP Features (impact on risk):\n"
    top_features = shap_values.values[0].argsort()[-3:][::-1]
    for i in top_features:
        feature = NAME_MAP.get(applicant_aligned.columns[i], applicant_aligned.columns[i])
        val = applicant_aligned.iloc[0,i]
        shap_val = shap_values.values[0][i]
        text += f"- {feature}={val} contributed {'positively' if shap_val>0 else 'negatively'} to risk\n"

    text += "\n📌 LIME explanation:\n"
    for f,w in lime_exp.as_list(label=1):
        text += f"- {f} with weight {w:.2f}\n"

    return text

# -------------------- FEATURES --------------------
FEATURES = ["income","age","loan_amnt","last_pymnt_amnt","total_pymnt",
            "recoveries","funded_amnt","total_rec_prncp","total_pymnt_inv","funded_amnt_inv"]

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

# -------------------- LOAD MODEL --------------------
mlp_model = joblib.load("mlp_model.pkl")
scaler = joblib.load("scaler.pkl")
df = pd.read_csv("credit_risk_dataset.csv")
feature_names = joblib.load("features.pkl")  # original order of features

st.set_page_config(page_title="AI Loan Assistant", layout="wide")
st.title("🤖 AI Loan Assistant & Credit Risk Tool")

# -------------------- SESSION STATE --------------------
if "chat_stage" not in st.session_state:
    st.session_state.chat_stage = 0
    st.session_state.chat_data = {}

# -------------------- CHATBOT QUESTIONS --------------------
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

# Chatbot sidebar responses
st.sidebar.header("💬 Chatbot Responses")
for k,v in st.session_state.chat_data.items():
    st.sidebar.write(f"**{k}:** {v}")

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
        if answer is not None and answer != "":
            st.session_state.chat_data[q_label] = answer
            st.session_state.chat_stage += 1
        else:
            st.warning("Please provide an answer before continuing.")

# -------------------- APPLICANT DATA INPUT --------------------
else:
    st.success("✅ Chatbot completed!")
    st.write("Summary of applicant info:")
    st.table(pd.DataFrame([st.session_state.chat_data]))

    st.header("📋 Applicant Data Input Form")

    def prefill(key, default=0.0):
        return safe_float(st.session_state.chat_data.get(key), default)

    # Applicant numeric inputs
    income = st.number_input(NAME_MAP["income"], value=prefill("Monthly Income (USD)",0.0), step=100.0,
                             help="Enter the applicant's monthly income in USD")
    age = st.number_input(NAME_MAP["age"], value=prefill("Age (Years)",18), step=1.0,
                          help="Enter applicant age in years")
    loan_amnt = st.number_input(NAME_MAP["loan_amnt"], value=prefill("Requested Loan Amount (USD)",0.0), step=100.0)
    last_pymnt_amnt = st.number_input(NAME_MAP["last_pymnt_amnt"], value=0.0, step=50.0)
    total_pymnt = st.number_input(NAME_MAP["total_pymnt"], value=0.0, step=100.0)
    recoveries = st.number_input(NAME_MAP["recoveries"], value=0.0, step=10.0)
    funded_amnt = st.number_input(NAME_MAP["funded_amnt"], value=0.0, step=100.0)
    total_rec_prncp = st.number_input(NAME_MAP["total_rec_prncp"], value=0.0, step=100.0)
    total_pymnt_inv = st.number_input(NAME_MAP["total_pymnt_inv"], value=0.0, step=100.0)
    funded_amnt_inv = st.number_input(NAME_MAP["funded_amnt_inv"], value=0.0, step=100.0)

    applicant_data = pd.DataFrame({
        "income":[income], "age":[age], "loan_amnt":[loan_amnt],
        "last_pymnt_amnt":[last_pymnt_amnt], "total_pymnt":[total_pymnt],
        "recoveries":[recoveries], "funded_amnt":[funded_amnt],
        "total_rec_prncp":[total_rec_prncp], "total_pymnt_inv":[total_pymnt_inv],
        "funded_amnt_inv":[funded_amnt_inv]
    })

    # -------------------- PREDICTION --------------------
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

        st.subheader("Prediction Result")
        st.write({
            "Prediction":"Default" if pred[0]==1 else "Non-Default",
            "Probability of Default":prob[0],
            "Calculated Credit Score":credit_score_val
        })

        # -------------------- SHAP INTERACTIVE --------------------
        st.subheader("SHAP Interactive Dashboard")
        explainer = shap.Explainer(mlp_model.predict, scaler.transform(df.drop("default_ind",axis=1)))
        shap_values = explainer(scaled)
        fig = shap.plots.bar(shap_values, max_display=10, show=False)
        st.pyplot(fig)

        # -------------------- LIME INTERACTIVE --------------------
        st.subheader("LIME Explanation")
        lime_explainer = lime.lime_tabular.LimeTabularExplainer(
            training_data=scaler.transform(df.drop("default_ind",axis=1).values),
            feature_names=[NAME_MAP.get(f,f) for f in df.drop("default_ind",axis=1).columns.tolist()],
            class_names=["Non-Default","Default"],
            mode="classification"
        )
        explanation = lime_explainer.explain_instance(scaled[0], mlp_model.predict_proba, num_features=5)
        fig = explanation.as_pyplot_figure(label=1)
        st.pyplot(fig)

        # -------------------- AI ASSISTANT --------------------
        st.subheader("🤖 AI Assistant Advice")
        st.write(ai_assistant(pred, prob, shap_values, explanation, applicant_aligned, credit_score_val))

















































