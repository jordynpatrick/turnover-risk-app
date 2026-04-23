import streamlit as st
import numpy as np
from sklearn.linear_model import LogisticRegression
import plotly.graph_objects as go

# -------------------------
# MODEL (keep as-is)
# -------------------------
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

# -------------------------
# TABS
# -------------------------
tab1, tab2, tab3 = st.tabs(["📊 Predictor", "🧠 Model Info", "📁 About Data"])

# -------------------------
# TAB 1: PREDICTOR
# -------------------------
with tab1:
    st.title("HR Analytics Turnover Risk Dashboard")
    st.subheader("AI-powered Employee Attrition Prediction Tool")

    job_sat = st.slider("Job Satisfaction", 1, 5, 3)
    workload = st.slider("Workload", 1, 5, 3)
    pay_sat = st.slider("Pay Satisfaction", 1, 5, 3)
    tenure = st.slider("Tenure (Years)", 0.0, 15.0, 3.0)

    if st.button("Predict"):
        probs = model.predict_proba([[job_sat, workload, pay_sat, tenure]])[0]
        pred = np.argmax(probs)

        st.write("### Risk Breakdown")

        st.write(f"🟢 Low Risk Probability: {probs[0]:.2f}")
        st.write(f"🟠 Medium Risk Probability: {probs[1]:.2f}")
        st.write(f"🔴 High Risk Probability: {probs[2]:.2f}")

        st.markdown("### Risk Visualization")

        fig = go.Figure(data=[
            go.Bar(name="Low Risk", x=["Risk"], y=[probs[0]]),
            go.Bar(name="Medium Risk", x=["Risk"], y=[probs[1]]),
            go.Bar(name="High Risk", x=["Risk"], y=[probs[2]])
        ])

        fig.update_layout(
            barmode='group',
            yaxis=dict(title="Probability"),
            template="simple_white"
        )

        fig.update_traces(marker_color=["#2E86AB", "#F4A261", "#E76F51"])

        st.plotly_chart(fig, use_container_width=True)

        st.write("---")
        st.write("### Prediction")

        if pred == 0:
            st.success("Low Turnover Risk")
            st.write("🟢 Recommendation: Employee is likely to stay. Standard engagement maintenance recommended.")

        elif pred == 1:
            st.warning("Medium Turnover Risk")
            st.write("🟠 Recommendation: Monitor engagement and workload balance to prevent escalation.")

        else:
            st.error("High Turnover Risk")
            st.write("🔴 Recommendation: Consider retention intervention (workload, compensation, or managerial support review).")

# -------------------------
# TAB 2: MODEL INFO
# -------------------------
with tab2:
    st.header("Model Information")

    st.write("""
    This project uses Logistic Regression to predict employee turnover risk.

    It models relationships between:
    - Job satisfaction
    - Workload
    - Pay satisfaction
    - Tenure

    The output is a probability-based classification into Low, Medium, or High risk.
    """)

# -------------------------
# TAB 3: DATA INFO
# -------------------------
with tab3:
    st.header("Dataset Information")

    st.write("""
    This dataset is simulated for educational purposes in Industrial-Organizational Psychology.

    It represents common employee survey constructs used in turnover prediction research.
    """)
