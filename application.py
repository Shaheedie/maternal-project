# 📦 Step 1: Import required libraries
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import warnings

warnings.filterwarnings("ignore")

# 🧼 Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv('Maternal Health Risk Data Set.csv')

    le = LabelEncoder()
    df['RiskLevel'] = le.fit_transform(df['RiskLevel'])
    return df, le

df, le = load_data()

# 🎯 Prepare data
X = df.drop("RiskLevel", axis=1)
y = df["RiskLevel"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 🧠 Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 🎨 Streamlit UI
st.title("🤰 Maternal Health Risk Predictor")
st.write("This app predicts maternal health risk and gives health recommendations based on input parameters.")

# 📥 Input widgets
age = st.slider("Age", 15, 50, 30)
systolic_bp = st.slider("Systolic Blood Pressure", 80, 180, 120)
diastolic_bp = st.slider("Diastolic Blood Pressure", 50, 120, 80)
bs = st.number_input("Blood Sugar Level (mmol/L)", min_value=0.0, max_value=20.0, value=6.0)
body_temp = st.slider("Body Temperature (°C)", 35.0, 42.0, 37.0)
heart_rate = st.slider("Heart Rate (bpm)", 60, 140, 80)

# 🧮 Prepare input vector
input_data = np.array([age, systolic_bp, diastolic_bp, bs, body_temp, heart_rate]).reshape(1, -1)

# 🧪 Prediction
if st.button("Predict Risk Level"):
    prediction = model.predict(input_data)[0]
    risk_label = le.inverse_transform([prediction])[0]

    # 📌 Explain potential reason
    reasons = []
    if systolic_bp > 140 or diastolic_bp > 90:
        reasons.append("High blood pressure")
    if bs > 7.8:
        reasons.append("Elevated blood sugar level")
    if body_temp > 38:
        reasons.append("Fever (high body temperature)")
    if heart_rate > 100:
        reasons.append("Increased heart rate")
    if age > 40:
        reasons.append("Higher maternal age")

    if not reasons:
        reasons.append("Overall health indicators are moderately balanced.")

    reason_text = " | ".join(reasons)

    # 💡 Health Recommendation
    if risk_label == "Low":
        recommendation = (
            "✅ **Low Risk:** Maintain a healthy lifestyle. "
            "Continue regular prenatal checkups, balanced nutrition, and light physical activity."
        )
    elif risk_label == "Mid":
        recommendation = (
            "⚠️ **Moderate Risk:** Monitor blood pressure, sugar levels, and heart rate closely. "
            "Consult your doctor for lifestyle and dietary adjustments."
        )
    else:
        recommendation = (
            "🚨 **High Risk:** Immediate consultation with a healthcare provider is advised. "
            "Ensure close medical supervision and avoid stress or physical strain."
        )

    st.success(f"Predicted Risk Level: **{risk_label}**")
    st.info(f"🧾 **Possible Reasons:** {reason_text}")
    st.info(recommendation)
