import streamlit as st
import numpy as np
from sklearn.linear_model import LogisticRegression

X = np.array([
    [5,1,5,10],
    [4,2,4,8],
    [3,3,3,5],
    [2,4,2,2],
    [1,5,1,1]
])

y = np.array([0,0,1,2,2])

model = LogisticRegression(max_iter=1000)
model.fit(X,y)

st.title("HR Analytics Turnover Risk Dashboard")
st.subheader("Machine Learning Model for Employee Attrition Risk Prediction")

st.title("HR Analytics Turnover Risk Dashboard")
st.subheader("AI-powered Employee Attrition Prediction Tool")

st.markdown("""
### About this tool
This dashboard uses machine learning to estimate employee turnover risk based on key job attitude variables commonly studied in Industrial-Organizational Psychology.
""")

job_sat = st.slider("Job Satisfaction",1,5,3)
workload = st.slider("Workload",1,5,3)
pay_sat = st.slider("Pay Satisfaction",1,5,3)
tenure = st.slider("Tenure",0.0,15.0,3.0)

if st.button("Predict"):
    probs = model.predict_proba([[job_sat, workload, pay_sat, tenure]])[0]
    pred = np.argmax(probs)

    st.write("### Risk Breakdown")

    st.write(f"Low Risk Probability: {probs[0]:.2f}")
    st.write(f"Medium Risk Probability: {probs[1]:.2f}")
    st.write(f"High Risk Probability: {probs[2]:.2f}")

    if pred == 0:
        st.success("Low Turnover Risk")
    elif pred == 1:
        st.warning("Medium Turnover Risk")
    else:
        st.error("High Turnover Risk")

