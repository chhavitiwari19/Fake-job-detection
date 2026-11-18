
import streamlit as st
import pandas as pd
import joblib
from scipy.sparse import hstack, csr_matrix

# Load trained files
model = joblib.load("fake_job_model.pkl")
vectorizer = joblib.load("fake_job_vectorizer.pkl")

# Page configuration
st.set_page_config(
    page_title="Fake Job Detection AI",
    page_icon="🔍",
    layout="wide"
)

# Title
st.title("🔍 Fake Job Detection System")
st.write("AI-powered job posting verification using Machine Learning.")

st.divider()

st.subheader("Enter Job Details")

# Input fields
title = st.text_input("Job Title")
company_profile = st.text_area("Company Profile")
description = st.text_area("Job Description")
requirements = st.text_area("Job Requirements")
benefits = st.text_area("Job Benefits")

col1, col2, col3 = st.columns(3)

with col1:
    telecommuting = st.selectbox("Telecommuting", [0, 1])

with col2:
    has_company_logo = st.selectbox("Company Logo", [0, 1])

with col3:
    has_questions = st.selectbox("Has Screening Questions", [0, 1])

# Prediction
if st.button("Detect Job", use_container_width=True):

    if not title or not description:
        st.warning("Please enter Job Title and Description.")

    else:
        new_job = pd.DataFrame({
            "title": [title],
            "company_profile": [company_profile],
            "description": [description],
            "requirements": [requirements],
            "benefits": [benefits]
        })

        new_job["combined_text"] = new_job[
            ["title", "company_profile", "description",
             "requirements", "benefits"]
        ].agg(" ".join, axis=1)

        text_data = vectorizer.transform(
            new_job["combined_text"]
        )

        numeric_data = csr_matrix([[
            telecommuting,
            has_company_logo,
            has_questions
        ]])

        final_data = hstack([text_data, numeric_data])

        prediction = model.predict(final_data)[0]
        probability = model.predict_proba(final_data)[0]

        st.divider()
        st.subheader("Prediction Result")

        if prediction == 1:
            st.error("⚠️ This job posting is FAKE.")
            st.metric("Fake Probability",
                      f"{probability[1] * 100:.2f}%")
        else:
            st.success("✅ This job posting is REAL.")
            st.metric("Real Probability",
                      f"{probability[0] * 100:.2f}%")

        st.caption(
            "Predictions are based on learned patterns, "
            "not a guarantee of job authenticity."
        )
