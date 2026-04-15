import streamlit as st
import pandas as pd
import joblib
import numpy as np

@st.cache_resource
def load_model():
    return joblib.load('loan_default_model.pkl')

model = load_model()

st.title("🏦 Loan Default Risk Predictor")
st.markdown("**Built for Momentum Financial Services Group Data Scientist interview**")

sex_map = {
    1: "Male",
    2: "Female"
}

education_map = {
    1: "Graduate school",
    2: "University",
    3: "High school",
    4: "Others",
    5: "Unknown",
    6: "Unknown"
}

marriage_map = {
    1: "Married",
    2: "Single",
    3: "Others"
}

st.sidebar.header("Customer Profile")
limit_bal = st.sidebar.slider("Credit Limit (NTD)", 10000, 1000000, 200000)

sex = st.sidebar.radio(
    "Sex",
    options=list(sex_map.keys()),
    format_func=lambda x: sex_map[x]
)

education = st.sidebar.selectbox(
    "Education",
    options=list(education_map.keys()),
    format_func=lambda x: education_map[x]
)

marriage = st.sidebar.selectbox(
    "Marriage",
    options=list(marriage_map.keys()),
    format_func=lambda x: marriage_map[x]
)

age = st.sidebar.slider("Age", 18, 80, 30)

st.sidebar.header("Repayment History (Sep-Apr)")
pay_0 = st.sidebar.slider("Sep Repayment Status", -1, 9, 0)
pay_2 = st.sidebar.slider("Aug Repayment Status", -1, 9, 0)
pay_3 = st.sidebar.slider("Jul Repayment Status", -1, 9, 0)
pay_4 = st.sidebar.slider("Jun Repayment Status", -1, 9, 0)
pay_5 = st.sidebar.slider("May Repayment Status", -1, 9, 0)
pay_6 = st.sidebar.slider("Apr Repayment Status", -1, 9, 0)

st.sidebar.header("Bill & Payment Amounts (Sep-Apr, NTD)")
bill_amt1 = st.sidebar.number_input("Sep Bill", 0, 1000000, 50000)
bill_amt2 = st.sidebar.number_input("Aug Bill", 0, 1000000, 50000)
bill_amt3 = st.sidebar.number_input("Jul Bill", 0, 1000000, 50000)
bill_amt4 = st.sidebar.number_input("Jun Bill", 0, 1000000, 50000)
bill_amt5 = st.sidebar.number_input("May Bill", 0, 1000000, 50000)
bill_amt6 = st.sidebar.number_input("Apr Bill", 0, 1000000, 50000)

pay_amt1 = st.sidebar.number_input("Sep Payment", 0, 500000, 20000)
pay_amt2 = st.sidebar.number_input("Aug Payment", 0, 500000, 20000)
pay_amt3 = st.sidebar.number_input("Jul Payment", 0, 500000, 20000)
pay_amt4 = st.sidebar.number_input("Jun Payment", 0, 500000, 20000)
pay_amt5 = st.sidebar.number_input("May Payment", 0, 500000, 20000)
pay_amt6 = st.sidebar.number_input("Apr Payment", 0, 500000, 20000)

input_data = pd.DataFrame({
    'LIMIT_BAL': [limit_bal],
    'SEX': [sex],
    'EDUCATION': [education],
    'MARRIAGE': [marriage],
    'AGE': [age],
    'PAY_0': [pay_0], 'PAY_2': [pay_2], 'PAY_3': [pay_3], 'PAY_4': [pay_4],
    'PAY_5': [pay_5], 'PAY_6': [pay_6],
    'BILL_AMT1': [bill_amt1], 'BILL_AMT2': [bill_amt2], 'BILL_AMT3': [bill_amt3],
    'BILL_AMT4': [bill_amt4], 'BILL_AMT5': [bill_amt5], 'BILL_AMT6': [bill_amt6],
    'PAY_AMT1': [pay_amt1], 'PAY_AMT2': [pay_amt2], 'PAY_AMT3': [pay_amt3],
    'PAY_AMT4': [pay_amt4], 'PAY_AMT5': [pay_amt5], 'PAY_AMT6': [pay_amt6],
    'AVG_BILL': [(bill_amt1+bill_amt2+bill_amt3+bill_amt4+bill_amt5+bill_amt6)/6],
    'TOTAL_BILL': [bill_amt1+bill_amt2+bill_amt3+bill_amt4+bill_amt5+bill_amt6],
    'AVG_PAY': [(pay_amt1+pay_amt2+pay_amt3+pay_amt4+pay_amt5+pay_amt6)/6],
    'TOTAL_PAY': [pay_amt1+pay_amt2+pay_amt3+pay_amt4+pay_amt5+pay_amt6],
    'PAY_TO_BILL_RATIO': [(pay_amt1+pay_amt2+pay_amt3+pay_amt4+pay_amt5+pay_amt6)/((bill_amt1+bill_amt2+bill_amt3+bill_amt4+bill_amt5+bill_amt6)+1)],
    'AVG_REPAY_STATUS': [(pay_0+pay_2+pay_3+pay_4+pay_5+pay_6)/6]
})

if st.button("🚀 Predict Default Risk", type="primary"):
    prediction = model.predict(input_data)[0]
    prob = model.predict_proba(input_data)[0][1]

    st.subheader("📊 Prediction Results")
    col1, col2 = st.columns(2)

    with col1:
        st.metric("Default Probability", f"{prob:.1%}")

    with col2:
        if prediction == 1:
            st.error("⚠️ HIGH RISK - Likely to default")
        else:
            st.success("✅ LOW RISK - Unlikely to default")

    st.markdown("---")
    st.caption("**Model**: Random Forest | **AUC**: 0.78+ | **Dataset**: UCI Credit Card (30K customers)")

st.markdown("---")
st.markdown("""
### 🎯 How to use
1. Adjust customer profile in sidebar
2. Set repayment history & bill amounts  
3. Click Predict → see default risk instantly

### 📈 Model Insights
- **Repayment status** is strongest predictor
- Higher credit limits → lower default risk
- Payment-to-bill ratio matters most

**Built for MFSG Data Scientist interview** 💼
""")