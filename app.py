
import streamlit as st
import pandas as pd
import joblib

# Load trained model
model = joblib.load("model.pkl")

# Page configuration
st.set_page_config(
    page_title="Fake Job Detection",
    page_icon="🔍",
    layout="centered"
)

# Application title
st.title("🔍 Fake Job Detection System")

st.write(
    "This application helps identify whether a job "
    "posting is Fake or Real using Machine Learning."
)

st.divider()

# User inputs
st.subheader("Enter Job Details")

telecommuting = st.selectbox(
    "Is this a remote job?",
    ["No", "Yes"]
)

has_company_logo = st.selectbox(
    "Does the company have a logo?",
    ["No", "Yes"]
)

has_questions = st.selectbox(
    "Does the job posting contain screening questions?",
    ["No", "Yes"]
)

# Convert inputs into numerical values
telecommuting = 1 if telecommuting == "Yes" else 0
has_company_logo = 1 if has_company_logo == "Yes" else 0
has_questions = 1 if has_questions == "Yes" else 0

# Prediction button
if st.button("Detect Job"):

    new_job = pd.DataFrame({
        "telecommuting": [telecommuting],
        "has_company_logo": [has_company_logo],
        "has_questions": [has_questions]
    })

    predicted_class = model.predict(new_job)[0]

    predicted_probability = model.predict_proba(new_job)[0]

    st.subheader("Prediction Result")

    if predicted_class == 0:
        st.error("⚠️ This job posting is FAKE.")
    else:
        st.success("✅ This job posting is REAL.")

    st.write("Prediction Probability:")

    for class_label, probability in zip(
        model.classes_,
        predicted_probability
    ):
        label = "Fake" if class_label == 0 else "Real"
        st.write(f"{label}: {probability * 100:.2f}%")
