
import streamlit as st
import pandas as pd
import joblib

# ---------------- PAGE CONFIGURATION ----------------

st.set_page_config(
    page_title="Fake Job Detection",
    page_icon="🔍",
    layout="wide"
)

# ---------------- CUSTOM CSS ----------------

st.markdown("""
<style>
.stApp {
    background-color: #F4F7FB;
}

.main-title {
    font-size: 38px;
    font-weight: 800;
    color: #17365D;
}

.subtitle {
    font-size: 16px;
    color: #64748B;
}

.metric-card {
    background-color: white;
    padding: 22px;
    border-radius: 12px;
    border: 1px solid #E2E8F0;
    text-align: center;
}

.section-title {
    color: #17365D;
    font-size: 23px;
    font-weight: 700;
}

div.stButton > button {
    background-color: #17365D;
    color: white;
    border-radius: 8px;
    height: 48px;
    font-weight: 600;
    width: 100%;
}

div.stButton > button:hover {
    background-color: #24558B;
    color: white;
}
</style>
""", unsafe_allow_html=True)

# ---------------- LOAD MODEL ----------------

@st.cache_resource
def load_model():
    return joblib.load("Fake job detection.pkl")

model = load_model()

# ---------------- SESSION STATE ----------------

if "history" not in st.session_state:
    st.session_state.history = []

# ---------------- SIDEBAR ----------------

st.sidebar.title("🔍 Fake Job Detector")

page = st.sidebar.radio(
    "Navigation",
    ["Home", "Prediction History", "About Model"]
)

st.sidebar.divider()

st.sidebar.info(
    "Machine Learning Based Job Fraud Detection System"
)

# ---------------- HOME PAGE ----------------

if page == "Home":

    st.markdown(
        '<div class="main-title">Fake Job Detection System</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">Identify suspicious job postings using Machine Learning.</div>',
        unsafe_allow_html=True
    )

    st.write("")

    # Dashboard cards
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            '<div class="metric-card"><h3>🤖</h3><b>ML Powered</b><p>Logistic Regression</p></div>',
            unsafe_allow_html=True
        )

    with col2:
        st.markdown(
            '<div class="metric-card"><h3>🔎</h3><b>Fraud Detection</b><p>Fake or Real Prediction</p></div>',
            unsafe_allow_html=True
        )

    with col3:
        st.markdown(
            '<div class="metric-card"><h3>📊</h3><b>Probability</b><p>Prediction Confidence</p></div>',
            unsafe_allow_html=True
        )

    st.write("")
    st.markdown(
        '<div class="section-title">Enter Job Information</div>',
        unsafe_allow_html=True
    )

    st.write("Provide the following details to analyze a job posting.")

    with st.form("job_form"):

        col1, col2 = st.columns(2)

        with col1:
            telecommuting = st.selectbox(
                "🏠 Is this a remote job?",
                ["No", "Yes"]
            )

        with col2:
            has_company_logo = st.selectbox(
                "🏢 Does the company have a logo?",
                ["No", "Yes"]
            )

        has_questions = st.selectbox(
            "❓ Does the job posting contain screening questions?",
            ["No", "Yes"]
        )

        submitted = st.form_submit_button(
            "🔍 Analyze Job Posting"
        )

    if submitted:

        new_job = pd.DataFrame({
            "telecommuting": [
                1 if telecommuting == "Yes" else 0
            ],
            "has_company_logo": [
                1 if has_company_logo == "Yes" else 0
            ],
            "has_questions": [
                1 if has_questions == "Yes" else 0
            ]
        })

        predicted_class = model.predict(new_job)[0]

        probabilities = model.predict_proba(new_job)[0]

        fake_probability = probabilities[
            list(model.classes_).index(0)
        ]

        real_probability = probabilities[
            list(model.classes_).index(1)
        ]

        result = "Fake" if predicted_class == 0 else "Real"

        st.session_state.history.append({
            "Prediction": result,
            "Fake Probability": round(fake_probability * 100, 2),
            "Real Probability": round(real_probability * 100, 2)
        })

        st.divider()

        st.markdown(
            '<div class="section-title">Prediction Results</div>',
            unsafe_allow_html=True
        )

        if predicted_class == 0:
            st.error("⚠️ This job posting is predicted to be FAKE.")
            st.warning(
                "Be careful! Verify the recruiter and company before sharing personal information or making payments."
            )
        else:
            st.success("✅ This job posting is predicted to be REAL.")
            st.info(
                "The model predicts this posting as real. Always verify the employer independently."
            )

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "Fake Probability",
                f"{fake_probability * 100:.2f}%"
            )

        with col2:
            st.metric(
                "Real Probability",
                f"{real_probability * 100:.2f}%"
            )

        chart_data = pd.DataFrame({
            "Category": ["Fake", "Real"],
            "Probability": [
                fake_probability * 100,
                real_probability * 100
            ]
        })

        st.subheader("Prediction Probability Chart")

        st.bar_chart(
            chart_data.set_index("Category")
        )

        st.caption(
            "Probabilities are model estimates, not guarantees of job authenticity."
        )

        st.subheader("Job Safety Recommendations")

        st.write("- Verify the official company website.")
        st.write("- Check the recruiter's identity.")
        st.write("- Never pay money for a job offer.")
        st.write("- Avoid sharing sensitive personal information.")

# ---------------- HISTORY PAGE ----------------

elif page == "Prediction History":

    st.title("📊 Prediction History")

    if st.session_state.history:

        history_df = pd.DataFrame(
            st.session_state.history
        )

        st.dataframe(
            history_df,
            use_container_width=True
        )

        if st.button("Clear History"):
            st.session_state.history = []
            st.rerun()

    else:
        st.info("No predictions available yet. Analyze a job first.")

# ---------------- ABOUT MODEL PAGE ----------------

elif page == "About Model":

    st.title("🤖 About Our Machine Learning Model")

    st.write(
        "This application uses a trained Logistic Regression "
        "classification model to predict whether a job posting "
        "is fake or real."
    )

    st.subheader("Algorithm")

    st.write("Logistic Regression")

    st.subheader("Input Features")

    st.write("1. Telecommuting")
    st.write("2. Has Company Logo")
    st.write("3. Has Questions")

    st.subheader("Target Variable")

    st.write("0 = Fake Job")
    st.write("1 = Real Job")

    st.subheader("Disclaimer")

    st.write(
        "This model provides predictions based on its training data "
        "and selected features. It cannot guarantee that a job is "
        "genuine or fraudulent. Always verify job opportunities "
        "through official sources."
    )

# ---------------- FOOTER ----------------

st.sidebar.divider()

st.sidebar.caption(
    "Fake Job Detection System | Machine Learning Project"
)
