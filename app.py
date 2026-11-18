
import streamlit as st
import pandas as pd
import numpy as np
import joblib

from scipy.sparse import hstack, csr_matrix


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Fake Job Detection AI",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# --------------------------------------------------
# LOAD MODEL AND VECTORIZER
# --------------------------------------------------

@st.cache_resource
def load_files():
    model = joblib.load("fake_job_model.pkl")
    vectorizer = joblib.load("fake_job_vectorizer.pkl")
    return model, vectorizer


try:
    model, vectorizer = load_files()
except Exception as e:
    st.error("Model files could not be loaded.")
    st.write("Please check that both .pkl files are in your GitHub repository.")
    st.stop()


# --------------------------------------------------
# PREMIUM GRADIENT PURPLE THEME
# --------------------------------------------------

st.markdown("""
<style>

.stApp {
    background:
        radial-gradient(circle at top left, #4c1d95 0%, transparent 35%),
        radial-gradient(circle at bottom right, #701a75 0%, transparent 35%),
        linear-gradient(135deg, #0f1024 0%, #17132f 50%, #21113b 100%);
    color: #ffffff;
}

header[data-testid="stHeader"] {
    background: transparent;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1200px;
}

h1, h2, h3, p, label {
    color: #ffffff !important;
}

.hero {
    background: linear-gradient(
        120deg,
        rgba(124, 58, 237, 0.35),
        rgba(192, 38, 211, 0.20)
    );
    border: 1px solid rgba(216, 180, 254, 0.30);
    border-radius: 22px;
    padding: 30px;
    margin-bottom: 25px;
    text-align: center;
    box-shadow: 0 10px 35px rgba(0, 0, 0, 0.25);
}

.hero h1 {
    font-size: 38px;
    font-weight: 800;
    margin-bottom: 10px;
    background: linear-gradient(90deg, #ffffff, #d8b4fe, #f0abfc);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.hero p {
    color: #e9d5ff !important;
    font-size: 16px;
}

.section-card {
    background: rgba(255, 255, 255, 0.07);
    border: 1px solid rgba(216, 180, 254, 0.20);
    border-radius: 18px;
    padding: 22px;
    margin-bottom: 20px;
}

div[data-testid="stTextInput"] input,
div[data-testid="stTextArea"] textarea,
div[data-testid="stSelectbox"] div[data-baseweb="select"] {
    background-color: rgba(255, 255, 255, 0.10);
    color: white;
    border-radius: 10px;
}

div[data-testid="stTextInput"] input,
div[data-testid="stTextArea"] textarea {
    border: 1px solid rgba(216, 180, 254, 0.35);
}

div[data-testid="stSelectbox"] div[data-baseweb="select"] * {
    color: white;
}

div.stButton > button {
    background: linear-gradient(90deg, #7c3aed, #c026d3);
    color: white;
    border: none;
    border-radius: 12px;
    padding: 12px 28px;
    font-weight: 700;
    width: 100%;
    transition: 0.3s;
}

div.stButton > button:hover {
    background: linear-gradient(90deg, #9333ea, #db2777);
    color: white;
    border: 1px solid #e9d5ff;
    transform: translateY(-2px);
}

.result-card {
    background: rgba(255, 255, 255, 0.09);
    border: 1px solid rgba(216, 180, 254, 0.30);
    border-radius: 18px;
    padding: 22px;
    text-align: center;
    margin-top: 15px;
}

.real-result {
    color: #86efac;
    font-size: 30px;
    font-weight: 800;
}

.fake-result {
    color: #fda4af;
    font-size: 30px;
    font-weight: 800;
}

.small-note {
    color: #d8b4fe;
    font-size: 13px;
    text-align: center;
}

hr {
    border-color: rgba(216, 180, 254, 0.25);
}

</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.markdown("""
<div class="hero">
    <h1>🔍 Fake Job Detection AI</h1>
    <p>
        Smart Machine Learning System to Identify
        Fake and Genuine Job Postings
    </p>
    <p>Powered by Artificial Intelligence & Logistic Regression</p>
</div>
""", unsafe_allow_html=True)


# --------------------------------------------------
# SAMPLE JOB DATA
# --------------------------------------------------

samples = {

    "Software Developer": {
        "title": "Software Developer",
        "company_profile": "We are a technology company developing software solutions for businesses.",
        "description": "We are looking for a software developer to build and maintain applications. The candidate will work with our development team.",
        "requirements": "Bachelor degree in computer science. Knowledge of Python, Java, programming and problem solving.",
        "benefits": "Competitive salary, professional training and career growth opportunities.",
        "telecommuting": "No",
        "has_company_logo": "Yes",
        "has_questions": "Yes"
    },

    "Python Developer": {
        "title": "Python Developer",
        "company_profile": "Our company provides software development and technology services.",
        "description": "We need a Python developer to develop applications, write clean code and work with databases.",
        "requirements": "Knowledge of Python, Django, Flask, SQL and problem solving skills.",
        "benefits": "Flexible working hours, training and professional development.",
        "telecommuting": "Yes",
        "has_company_logo": "Yes",
        "has_questions": "Yes"
    },

    "Data Analyst": {
        "title": "Data Analyst",
        "company_profile": "We are a data analytics company helping businesses make better decisions.",
        "description": "The candidate will analyze datasets, prepare reports and create visualizations.",
        "requirements": "Knowledge of Excel, Python, SQL, statistics and data visualization.",
        "benefits": "Competitive salary, learning opportunities and career development.",
        "telecommuting": "No",
        "has_company_logo": "Yes",
        "has_questions": "Yes"
    },

    "Web Developer": {
        "title": "Web Developer",
        "company_profile": "We create modern websites and digital solutions for clients.",
        "description": "We are hiring a web developer to design, develop and maintain websites.",
        "requirements": "Knowledge of HTML, CSS, JavaScript, PHP and responsive web design.",
        "benefits": "Paid training, supportive team and growth opportunities.",
        "telecommuting": "No",
        "has_company_logo": "Yes",
        "has_questions": "Yes"
    },

    "IT Support": {
        "title": "IT Support Executive",
        "company_profile": "We provide IT support and technical services to business customers.",
        "description": "The employee will help users solve technical problems and maintain computer systems.",
        "requirements": "Basic computer knowledge, troubleshooting and communication skills.",
        "benefits": "Fixed salary, training and career growth.",
        "telecommuting": "No",
        "has_company_logo": "Yes",
        "has_questions": "Yes"
    },

    "Marketing Assistant": {
        "title": "Marketing Assistant",
        "company_profile": "We are a growing marketing company working with different business clients.",
        "description": "Assist the marketing team with campaigns, research and promotional activities.",
        "requirements": "Communication skills, basic computer knowledge and interest in marketing.",
        "benefits": "Professional training and performance-based growth.",
        "telecommuting": "No",
        "has_company_logo": "Yes",
        "has_questions": "Yes"
    },

    "HR Executive": {
        "title": "HR Executive",
        "company_profile": "Our organization provides professional business services.",
        "description": "We are looking for an HR executive to support recruitment and employee management.",
        "requirements": "Knowledge of recruitment, communication and MS Office.",
        "benefits": "Fixed salary, professional environment and growth opportunities.",
        "telecommuting": "No",
        "has_company_logo": "Yes",
        "has_questions": "Yes"
    },

    "Sales Executive": {
        "title": "Sales Executive",
        "company_profile": "We are a company offering products and services to business customers.",
        "description": "The candidate will communicate with customers and support sales activities.",
        "requirements": "Good communication, customer handling and negotiation skills.",
        "benefits": "Salary with incentives and career growth.",
        "telecommuting": "No",
        "has_company_logo": "Yes",
        "has_questions": "Yes"
    },

    "Accountant": {
        "title": "Accountant",
        "company_profile": "We provide accounting and financial services to clients.",
        "description": "Responsible for maintaining financial records, preparing reports and managing accounts.",
        "requirements": "Knowledge of accounting, Excel, bookkeeping and financial records.",
        "benefits": "Competitive salary and professional development.",
        "telecommuting": "No",
        "has_company_logo": "Yes",
        "has_questions": "Yes"
    },

    "Business Analyst": {
        "title": "Business Analyst",
        "company_profile": "We help organizations improve their business processes using data.",
        "description": "Analyze business requirements, prepare reports and work with project teams.",
        "requirements": "Analytical thinking, Excel, SQL and communication skills.",
        "benefits": "Learning opportunities and career advancement.",
        "telecommuting": "Yes",
        "has_company_logo": "Yes",
        "has_questions": "Yes"
    },

    "Content Writer": {
        "title": "Content Writer",
        "company_profile": "We are a digital media company creating online content.",
        "description": "Create articles, blogs and website content for different audiences.",
        "requirements": "Good writing skills, grammar, research and basic SEO knowledge.",
        "benefits": "Flexible schedule and professional growth.",
        "telecommuting": "Yes",
        "has_company_logo": "Yes",
        "has_questions": "Yes"
    },

    "Graphic Designer": {
        "title": "Graphic Designer",
        "company_profile": "Our creative agency provides branding and design services.",
        "description": "Create graphics, social media designs and marketing materials.",
        "requirements": "Knowledge of Photoshop, Illustrator and creative design.",
        "benefits": "Creative work environment and career opportunities.",
        "telecommuting": "Yes",
        "has_company_logo": "Yes",
        "has_questions": "Yes"
    },

    "Customer Support": {
        "title": "Customer Support Executive",
        "company_profile": "We provide customer service solutions to businesses.",
        "description": "Assist customers with questions, service requests and product information.",
        "requirements": "Communication skills, patience and basic computer knowledge.",
        "benefits": "Training, fixed salary and career growth.",
        "telecommuting": "No",
        "has_company_logo": "Yes",
        "has_questions": "Yes"
    },

    "Digital Marketing": {
        "title": "Digital Marketing Executive",
        "company_profile": "We help brands grow through online marketing strategies.",
        "description": "Manage social media, online campaigns and digital marketing activities.",
        "requirements": "Knowledge of SEO, social media, content marketing and analytics.",
        "benefits": "Learning opportunities and performance incentives.",
        "telecommuting": "Yes",
        "has_company_logo": "Yes",
        "has_questions": "Yes"
    },

    "Data Entry Work From Home": {
        "title": "Data Entry Work From Home",
        "company_profile": "We are seeking support for data processing and record management.",
        "description": "Enter and organize information in company databases while following data quality guidelines.",
        "requirements": "Basic computer knowledge, typing skills and attention to detail.",
        "benefits": "Flexible working hours and remote work opportunity.",
        "telecommuting": "Yes",
        "has_company_logo": "No",
        "has_questions": "No"
    }
}


# --------------------------------------------------
# LOAD SAMPLE CALLBACK
# --------------------------------------------------

def load_sample():
    selected = st.session_state.sample_choice

    if selected in samples:
        sample = samples[selected]

        for key, value in sample.items():
            st.session_state[key] = value


# --------------------------------------------------
# INITIALIZE INPUT VALUES
# --------------------------------------------------

default_values = {
    "title": "",
    "company_profile": "",
    "description": "",
    "requirements": "",
    "benefits": "",
    "telecommuting": "No",
    "has_company_logo": "No",
    "has_questions": "No"
}

for key, value in default_values.items():
    if key not in st.session_state:
        st.session_state[key] = value


# --------------------------------------------------
# SAMPLE JOB SELECTION
# --------------------------------------------------

st.markdown("## ✨ Try a Suggested Job")

st.selectbox(
    "Choose a sample job profile",
    ["Choose a sample..."] + list(samples.keys()),
    key="sample_choice",
    on_change=load_sample
)

st.caption("Select a sample to automatically fill the job details, or enter your own details below.")


# --------------------------------------------------
# JOB DETAILS INPUT
# --------------------------------------------------

st.markdown("""
<div class="section-card">
    <h3>📝 Enter Job Details</h3>
</div>
""", unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    st.text_input(
        "Job Title",
        key="title",
        placeholder="Enter job title"
    )

    st.text_area(
        "Company Profile",
        key="company_profile",
        height=130,
        placeholder="Enter company information"
    )

    st.text_area(
        "Job Description",
        key="description",
        height=180,
        placeholder="Enter job description"
    )

with col2:
    st.text_area(
        "Requirements",
        key="requirements",
        height=130,
        placeholder="Enter job requirements"
    )

    st.text_area(
        "Benefits",
        key="benefits",
        height=130,
        placeholder="Enter job benefits"
    )

    st.markdown("#### Additional Information")

    st.selectbox(
        "Does the company allow telecommuting?",
        ["No", "Yes"],
        key="telecommuting"
    )

    st.selectbox(
        "Does the company have a logo?",
        ["No", "Yes"],
        key="has_company_logo"
    )

    st.selectbox(
        "Does the job posting have screening questions?",
        ["No", "Yes"],
        key="has_questions"
    )


# --------------------------------------------------
# PREDICTION
# --------------------------------------------------

st.markdown("---")
st.markdown("## 🚀 Detect Job Authenticity")

if st.button("🔍 Predict Job Authenticity", use_container_width=True):

    if not st.session_state.title.strip() or not st.session_state.description.strip():
        st.warning("Please enter at least the Job Title and Job Description.")

    else:

        # Combine the five text features in the same order
        # used during model training.

        combined_text = " ".join([
            st.session_state.title,
            st.session_state.company_profile,
            st.session_state.description,
            st.session_state.requirements,
            st.session_state.benefits
        ])

        # Convert text into TF-IDF features
        text_data = vectorizer.transform([combined_text])

        # Convert Yes/No into numeric values
        numeric_data = np.array([[
            1 if st.session_state.telecommuting == "Yes" else 0,
            1 if st.session_state.has_company_logo == "Yes" else 0,
            1 if st.session_state.has_questions == "Yes" else 0
        ]])

        # Combine text and numeric features
        final_data = hstack([
            text_data,
            csr_matrix(numeric_data)
        ])

        # Predict class and probabilities
        predicted_class = model.predict(final_data)[0]
        probabilities = model.predict_proba(final_data)[0]

        classes = list(model.classes_)

        fake_probability = probabilities[classes.index(1)] * 100
        real_probability = probabilities[classes.index(0)] * 100

        # Save result for display
        st.session_state.prediction_done = True
        st.session_state.predicted_class = predicted_class
        st.session_state.real_probability = real_probability
        st.session_state.fake_probability = fake_probability


# --------------------------------------------------
# DISPLAY RESULT
# --------------------------------------------------

if st.session_state.get("prediction_done", False):

    st.markdown("---")
    st.markdown("## 📊 Prediction Result")

    predicted_class = st.session_state.predicted_class
    real_probability = st.session_state.real_probability
    fake_probability = st.session_state.fake_probability

    if predicted_class == 1:

        st.markdown("""
        <div class="result-card">
            <div class="fake-result">⚠️ FAKE JOB POSTING</div>
            <p>This job posting has been classified as potentially fraudulent.</p>
        </div>
        """, unsafe_allow_html=True)

    else:

        st.markdown("""
        <div class="result-card">
            <div class="real-result">✅ REAL JOB POSTING</div>
            <p>This job posting has been classified as likely genuine.</p>
        </div>
        """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Real Probability",
            f"{real_probability:.2f}%"
        )
        st.progress(int(real_probability))

    with col2:
        st.metric(
            "Fake Probability",
            f"{fake_probability:.2f}%"
        )
        st.progress(int(fake_probability))

    st.info(
        "This prediction is based on patterns learned from the training dataset. "
        "It is not a guarantee that a job is genuine or fraudulent. "
        "Always verify the company and job details independently."
    )


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.markdown("---")

st.markdown("""
<p class="small-note">
    Fake Job Detection AI | Machine Learning Project
    <br>
    Built with Python, Streamlit, TF-IDF and Logistic Regression
</p>
""", unsafe_allow_html=True)
