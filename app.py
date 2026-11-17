
import streamlit as st
import pandas as pd
import joblib

# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Fake Job Detection | AI",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# CUSTOM AI GRADIENT THEME
# ==========================================

st.markdown("""
<style>

.stApp {
    background: linear-gradient(
        135deg,
        #100B2B 0%,
        #191044 45%,
        #101C42 100%
    );
    color: #FFFFFF;
}

header[data-testid="stHeader"] {
    background: transparent;
}

section[data-testid="stSidebar"] {
    background: linear-gradient(
        180deg,
        #1D1450,
        #111936
    );
    border-right: 1px solid #493C85;
}

section[data-testid="stSidebar"] * {
    color: #FFFFFF;
}

.main-title {
    font-size: 43px;
    font-weight: 800;
    background: linear-gradient(
        90deg,
        #C084FC,
        #818CF8,
        #38BDF8
    );
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.subtitle {
    color: #C4B5FD;
    font-size: 17px;
}

.section-title {
    color: #C4B5FD;
    font-size: 25px;
    font-weight: 700;
}

.metric-card {
    background: linear-gradient(
        135deg,
        #29205B,
        #1B2855
    );
    border: 1px solid #6652B8;
    border-radius: 16px;
    padding: 23px;
    text-align: center;
    min-height: 145px;
}

.metric-card h2 {
    color: #C4B5FD;
    font-size: 30px;
}

.metric-card p {
    color: #CBD5E1;
}

div[data-testid="stForm"] {
    background: linear-gradient(
        135deg,
        #20194D,
        #18264F
    );
    border: 1px solid #6554AE;
    border-radius: 18px;
    padding: 25px;
}

.stButton > button,
.stFormSubmitButton > button {
    background: linear-gradient(
        90deg,
        #9333EA,
        #4F46E5,
        #2563EB
    );
    color: white;
    border: none;
    border-radius: 10px;
    height: 48px;
    font-weight: bold;
    width: 100%;
}

.stButton > button:hover,
.stFormSubmitButton > button:hover {
    border: 1px solid #C084FC;
    color: white;
}

div[data-baseweb="select"] > div {
    background-color: #29205B;
    border-color: #7764C9;
    color: white;
}

div[data-baseweb="select"] * {
    color: white;
}

label, p, h1, h2, h3 {
    color: #FFFFFF;
}

div[data-testid="stMetric"] {
    background: #241C55;
    border: 1px solid #6554AE;
    padding: 17px;
    border-radius: 13px;
}

div[data-testid="stMetricLabel"] {
    color: #C4B5FD;
}

div[data-testid="stMetricValue"] {
    color: #FFFFFF;
}

hr {
    border-color: #574B91;
}

.footer {
    text-align: center;
    color: #A5B4FC;
    padding: 25px;
    font-size: 13px;
}

</style>
""", unsafe_allow_html=True)

# ==========================================
# LOAD TRAINED MODEL
# ==========================================

@st.cache_resource
def load_model():
    return joblib.load("Fake job detection.pkl")

model = load_model()

# ==========================================
# SESSION HISTORY
# ==========================================

if "history" not in st.session_state:
    st.session_state.history = []

# ==========================================
# SIDEBAR
# ==========================================

st.sidebar.title("🔍 AI Job Detector")

st.sidebar.caption("SMART FRAUD DETECTION SYSTEM")

st.sidebar.divider()

page = st.sidebar.radio(
    "NAVIGATION",
    [
        "🏠 Home",
        "📊 Prediction History",
        "🤖 About Model"
    ]
)

st.sidebar.divider()

st.sidebar.info(
    "Powered by Machine Learning"
)

st.sidebar.caption(
    "Fake Job Detection | AI Project"
)

# ==========================================
# HOME PAGE
# ==========================================

if page == "🏠 Home":

    st.markdown(
        '<div class="main-title">AI Fake Job Detection</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">Intelligent job fraud analysis using Machine Learning</div>',
        unsafe_allow_html=True
    )

    st.write("")

    # DASHBOARD CARDS

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class="metric-card">
            <h2>🤖</h2>
            <h3>AI Powered</h3>
            <p>Logistic Regression</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="metric-card">
            <h2>🛡️</h2>
            <h3>Fraud Detection</h3>
            <p>Fake or Real Analysis</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="metric-card">
            <h2>📈</h2>
            <h3>Smart Prediction</h3>
            <p>Probability Analysis</p>
        </div>
        """, unsafe_allow_html=True)

    st.write("")
    st.divider()

    st.markdown(
        '<div class="section-title">✨ Analyze Job Posting</div>',
        unsafe_allow_html=True
    )

    st.write(
        "Enter the job details below to check the possibility of job fraud."
    )

    # INPUT FORM

    with st.form("job_form"):

        col1, col2 = st.columns(2)

        with col1:
            telecommuting = st.selectbox(
                "🏠 Remote Job",
                ["No", "Yes"]
            )

        with col2:
            has_company_logo = st.selectbox(
                "🏢 Company Logo Available",
                ["No", "Yes"]
            )

        has_questions = st.selectbox(
            "❓ Screening Questions Available",
            ["No", "Yes"]
        )

        submitted = st.form_submit_button(
            "🔍 ANALYZE JOB NOW"
        )

    # ======================================
    # PREDICTION
    # ======================================

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

        classes = list(model.classes_)

        fake_probability = probabilities[
            classes.index(0)
        ]

        real_probability = probabilities[
            classes.index(1)
        ]

        result = (
            "Fake" if predicted_class == 0
            else "Real"
        )

        # SAVE HISTORY

        st.session_state.history.append({
            "Prediction": result,
            "Fake Probability": round(
                fake_probability * 100, 2
            ),
            "Real Probability": round(
                real_probability * 100, 2
            )
        })

        st.divider()

        st.markdown(
            '<div class="section-title">📊 Analysis Results</div>',
            unsafe_allow_html=True
        )

        if predicted_class == 0:

            st.error(
                "⚠️ AI Prediction: This job posting is FAKE."
            )

            st.warning(
                "Be careful! Verify the company and recruiter "
                "before sharing personal details or making payments."
            )

        else:

            st.success(
                "✅ AI Prediction: This job posting is REAL."
            )

            st.info(
                "The model predicts this posting as real. "
                "Always verify the employer independently."
            )

        # PROBABILITY METRICS

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "🔴 Fake Probability",
                f"{fake_probability * 100:.2f}%"
            )

        with col2:
            st.metric(
                "🟢 Real Probability",
                f"{real_probability * 100:.2f}%"
            )

        # CHART

        st.subheader("Prediction Probability")

        chart_data = pd.DataFrame(
            {
                "Probability (%)": [
                    fake_probability * 100,
                    real_probability * 100
                ]
            },
            index=["Fake Job", "Real Job"]
        )

        st.bar_chart(chart_data)

        st.caption(
            "Probabilities are model estimates, not guarantees "
            "of whether a job is genuine or fraudulent."
        )

        # SAFETY TIPS

        st.divider()

        st.markdown(
            '<div class="section-title">🛡️ Job Safety Tips</div>',
            unsafe_allow_html=True
        )

        st.write("✔ Verify the official company website.")
        st.write("✔ Check the recruiter's identity.")
        st.write("✔ Never pay money to get a job.")
        st.write("✔ Avoid sharing sensitive personal information.")

# ==========================================
# PREDICTION HISTORY
# ==========================================

elif page == "📊 Prediction History":

    st.title("📊 Prediction History")

    if st.session_state.history:

        history_df = pd.DataFrame(
            st.session_state.history
        )

        st.dataframe(
            history_df,
            use_container_width=True,
            hide_index=True
        )

        if st.button("🗑️ Clear History"):

            st.session_state.history = []

            st.rerun()

    else:

        st.info(
            "No prediction history available. "
            "Analyze a job posting first."
        )

# ==========================================
# ABOUT MODEL
# ==========================================

elif page == "🤖 About Model":

    st.markdown(
        '<div class="main-title">About Our AI Model</div>',
        unsafe_allow_html=True
    )

    st.write(
        "This application uses a trained Logistic Regression "
        "classification model to identify potentially fraudulent "
        "job postings."
    )

    st.subheader("🧠 Algorithm")

    st.info("Logistic Regression")

    st.subheader("📌 Input Features")

    st.write("1. Telecommuting")
    st.write("2. Has Company Logo")
    st.write("3. Has Questions")

    st.subheader("🎯 Target Variable")

    st.write("0 = Fake Job")
    st.write("1 = Real Job")

    st.subheader("⚙️ Model Information")

    st.write(
        "The model analyzes the three selected features "
        "and generates a classification prediction with "
        "estimated probabilities."
    )

    st.subheader("⚠️ Disclaimer")

    st.write(
        "This model is an educational machine learning project. "
        "Its predictions are not a guarantee of job authenticity. "
        "Always independently verify job opportunities."
    )

# ==========================================
# FOOTER
# ==========================================

st.markdown(
    """
    <div class="footer">
        ✨ AI Fake Job Detection System
        <br>
        Built with Python, Streamlit & Machine Learning
    </div>
    """,
    unsafe_allow_html=True
)
