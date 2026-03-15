import streamlit as st
import joblib
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
from explain import get_top3_reasons, FEATURE_NAMES

# Load models


@st.cache_resource
def load_models():
    model = joblib.load('models/xgb_fair_base.pkl')
    explainer = shap.TreeExplainer(model)
    return model, explainer


model, explainer = load_models()

# App title
st.title("🏦 FairCredit")
st.subheader("Fair & Explainable Credit Risk Assessment")
st.markdown("---")

# Input form
st.header("Enter Applicant Details")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=18, max_value=100, value=35)
    monthly_income = st.number_input(
        "Monthly Income ($)", min_value=0, max_value=100000, value=5000)
    debt_ratio = st.number_input(
        "Debt Ratio", min_value=0.0, max_value=10.0, value=0.3)
    revolving_utilization = st.number_input(
        "Credit Utilization Rate (0-1)", min_value=0.0, max_value=5.0, value=0.3)
    total_past_due = st.number_input(
        "Total Past Due Payments", min_value=0, max_value=50, value=0)

with col2:
    open_credit_lines = st.number_input(
        "Number of Open Credit Lines", min_value=0, max_value=50, value=5)
    times_90_days_late = st.number_input(
        "Times 90+ Days Late", min_value=0, max_value=20, value=0)
    real_estate_loans = st.number_input(
        "Number of Real Estate Loans", min_value=0, max_value=20, value=1)
    times_60_89_late = st.number_input(
        "Times 60-89 Days Late", min_value=0, max_value=20, value=0)
    times_30_59_late = st.number_input(
        "Times 30-59 Days Late", min_value=0, max_value=20, value=0)
    dependents = st.number_input(
        "Number of Dependents", min_value=0, max_value=20, value=0)

st.markdown("---")

if st.button("🔍 Assess Credit Risk", use_container_width=True):
    # Build input dataframe
    input_dict = {
        'RevolvingUtilizationOfUnsecuredLines': revolving_utilization,
        'age': age,
        'NumberOfTime30-59DaysPastDueNotWorse': times_30_59_late,
        'DebtRatio': debt_ratio,
        'MonthlyIncome': monthly_income,
        'NumberOfOpenCreditLinesAndLoans': open_credit_lines,
        'NumberOfTimes90DaysLate': times_90_days_late,
        'NumberRealEstateLoansOrLines': real_estate_loans,
        'NumberOfTime60-89DaysPastDueNotWorse': times_60_89_late,
        'NumberOfDependents': dependents,
        'TotalPastDue': total_past_due
    }

    input_df = pd.DataFrame([input_dict])

    # Predict
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0]

    # Result
    st.markdown("---")
    st.header("Assessment Result")

    if prediction == 0:
        st.success("✅ Low Risk — Credit Approved")
        confidence = probability[0] * 100
        st.metric("Approval Confidence", f"{confidence:.1f}%")
    else:
        st.error("❌ High Risk — Credit Not Approved")
        confidence = probability[1] * 100
        st.metric("Risk Score", f"{confidence:.1f}%")

    # SHAP explanation
    shap_vals = explainer.shap_values(input_df)
    reasons, label = get_top3_reasons(
        shap_vals[0], input_df.columns.tolist(), prediction)

    st.markdown("---")
    st.header("Why this decision?")

    for i, r in enumerate(reasons):
        with st.container():
            st.markdown(
                f"**{i+1}. {r['readable_name']}** — impact score: `{r['shap_value']}`")
            if prediction == 1:
                st.info(f"💡 {r['suggestion']}")

    # Waterfall plot
    st.markdown("---")
    st.header("Decision Breakdown")
    shap_explanation = explainer(input_df)
    fig, ax = plt.subplots()
    shap.plots.waterfall(shap_explanation[0], show=False)
    st.pyplot(plt.gcf())
    plt.clf()
