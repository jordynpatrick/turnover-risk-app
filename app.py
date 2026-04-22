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

st.title("Turnover Risk Predictor")

job_sat = st.slider("Job Satisfaction",1,5,3)
workload = st.slider("Workload",1,5,3)
pay_sat = st.slider("Pay Satisfaction",1,5,3)
tenure = st.slider("Tenure",0.0,15.0,3.0)

if st.button("Predict"):
    pred = model.predict([[job_sat,workload,pay_sat,tenure]])[0]

    if pred == 0:
        st.success("Low Risk")
    elif pred == 1:
        st.warning("Medium Risk")
    else:
        st.error("High Risk")

