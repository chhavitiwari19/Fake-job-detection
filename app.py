import streamlit as st
import numpy as np
import joblib
import re

from scipy.sparse import hstack, csr_matrix


# --------------------------------------------------
# 1. PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Fake Job Detection AI",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# --------------------------------------------------
# 2. LOAD MODEL AND VECTORIZER
# --------------------------------------------------

@st.cache_resource
def load_files():
    model = joblib.load("fake_job_model.pkl")
    vectorizer = joblib.load("fake_job_vectorizer.pkl")
    return model, vectorizer


try:
    model, vectorizer = load_files()
except Exception as e:
    st.error("Unable to load model files.")
    st.error(str(e))
    st.stop()


# --------------------------------------------------
# 3. GLASSMORPHISM DESIGN SYSTEM
# --------------------------------------------------

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&family=Space+Grotesk:wght@500;600;700&display=swap');

:root {
    --bg: #0b1020;
    --panel: rgba(255, 255, 255, 0.075);
    --panel-strong: rgba(255, 255, 255, 0.11);
    --border: rgba(255, 255, 255, 0.16);
    --text: #f4f5ff;
    --muted: #aeb7d0;
    --violet: #a78bfa;
    --cyan: #67e8f9;
    --mint: #6ee7b7;
    --coral: #fda4af;
}

.stApp {
    background:
        radial-gradient(ellipse at 8% 8%, rgba(91, 72, 180, 0.36), transparent 34%),
        radial-gradient(ellipse at 92% 12%, rgba(28, 153, 177, 0.24), transparent 32%),
        radial-gradient(ellipse at 55% 100%, rgba(112, 74, 166, 0.22), transparent 38%),
        linear-gradient(135deg, #0b1020 0%, #111a31 52%, #101629 100%);
    color: var(--text);
    font-family: 'DM Sans', sans-serif;
}

header[data-testid="stHeader"] {
    background: rgba(11, 16, 32, 0.35);
}

.block-container {
    max-width: 1320px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

h1, h2, h3, h4, p, label, li {
    color: var(--text);
    font-family: 'DM Sans', sans-serif;
}

h1, h2, h3 {
    font-family: 'Space Grotesk', sans-serif;
}

[data-testid="stCaptionContainer"] p,
.stCaption {
    color: var(--muted) !important;
}

/* Hero panel */
.hero {
    position: relative;
    overflow: hidden;
    background: linear-gradient(125deg, rgba(255,255,255,0.14), rgba(255,255,255,0.045));
    border: 1px solid rgba(255,255,255,0.22);
    border-radius: 26px;
    padding: 34px 28px;
    margin: 8px 0 26px 0;
    text-align: left;
    box-shadow: 0 18px 55px rgba(0,0,0,0.24);
    backdrop-filter: blur(18px);
    -webkit-backdrop-filter: blur(18px);
}

.hero:after {
    content: "";
    position: absolute;
    width: 230px;
    height: 230px;
    right: -70px;
    top: -95px;
    border-radius: 50%;
    background: radial-gradient(circle, rgba(103,232,249,0.26), transparent 70%);
    pointer-events: none;
}

.hero h1 {
    color: #ffffff !important;
    font-size: clamp(30px, 4vw, 46px);
    line-height: 1.08;
    font-weight: 800;
    margin: 8px 0 12px 0;
    letter-spacing: -1.5px;
}

.hero p {
    color: #c5cee5 !important;
    font-size: 15px;
    margin: 4px 0;
}

.eyebrow {
    color: #9feaf2;
    font-size: 11px;
    font-weight: 800;
    letter-spacing: 2px;
    text-transform: uppercase;
}

/* Glass panels */
.glass-panel {
    background: linear-gradient(145deg, rgba(255,255,255,0.115), rgba(255,255,255,0.045));
    border: 1px solid var(--border);
    border-radius: 20px;
    padding: 22px;
    box-shadow: 0 12px 34px rgba(0,0,0,0.16);
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    margin-bottom: 16px;
}

.section-heading {
    color: #ffffff;
    font-family: 'Space Grotesk', sans-serif;
    font-size: 22px;
    font-weight: 700;
    margin: 8px 0 4px 0;
}

.section-subtitle {
    color: #aeb7d0;
    font-size: 13px;
    margin-bottom: 18px;
}

/* Input controls */
.stTextInput input,
.stTextArea textarea {
    background: rgba(255,255,255,0.09) !important;
    color: #ffffff !important;
    border: 1px solid rgba(255,255,255,0.19) !important;
    border-radius: 12px !important;
}

.stTextInput input::placeholder,
.stTextArea textarea::placeholder {
    color: #9ca8c4 !important;
}

.stTextInput input:focus,
.stTextArea textarea:focus {
    border-color: #a78bfa !important;
    box-shadow: 0 0 0 1px rgba(167,139,250,0.45) !important;
}

[data-baseweb="select"] > div {
    background: rgba(255,255,255,0.09) !important;
    border: 1px solid rgba(255,255,255,0.19) !important;
    border-radius: 12px !important;
    color: #ffffff !important;
}

[data-baseweb="select"] * {
    color: #ffffff !important;
}

[data-baseweb="popover"],
[data-baseweb="menu"] {
    background: #202943 !important;
}

[data-baseweb="menu"] li:hover {
    background: #34405f !important;
}

/* Buttons */
div.stButton > button {
    background: linear-gradient(105deg, #8b5cf6, #4f9de8);
    color: #ffffff !important;
    border: 1px solid rgba(255,255,255,0.2);
    border-radius: 13px;
    min-height: 48px;
    font-weight: 800;
    box-shadow: 0 8px 24px rgba(105,91,220,0.28);
    transition: all 0.2s ease;
}

div.stButton > button:hover {
    transform: translateY(-2px);
    border-color: rgba(255,255,255,0.45);
    box-shadow: 0 12px 30px rgba(105,91,220,0.4);
}

/* Result panel */
.result-card {
    background: linear-gradient(135deg, rgba(255,255,255,0.14), rgba(255,255,255,0.055));
    border: 1px solid rgba(255,255,255,0.22);
    border-radius: 20px;
    padding: 22px;
    min-height: 150px;
    box-shadow: 0 12px 34px rgba(0,0,0,0.18);
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
}

.fake-result {
    color: #fda4af;
    font-size: 30px;
    font-weight: 800;
    font-family: 'Space Grotesk', sans-serif;
}

.real-result {
    color: #6ee7b7;
    font-size: 30px;
    font-weight: 800;
    font-family: 'Space Grotesk', sans-serif;
}

/* Probability and metric cards */
[data-testid="stMetric"] {
    background: rgba(255,255,255,0.075);
    border: 1px solid rgba(255,255,255,0.15);
    border-radius: 16px;
    padding: 14px 16px;
    backdrop-filter: blur(12px);
}

[data-testid="stMetricLabel"] {
    color: #c4cce0 !important;
}

[data-testid="stMetricValue"] {
    color: #ffffff !important;
    font-family: 'Space Grotesk', sans-serif;
}

.stProgress > div > div > div {
    background: linear-gradient(90deg, #8b5cf6, #67e8f9);
    border-radius: 20px;
}

/* Explanation cards */
.factor-card {
    background: rgba(255,255,255,0.075);
    border: 1px solid rgba(255,255,255,0.15);
    border-left: 4px solid #a78bfa;
    border-radius: 14px;
    padding: 15px;
    margin-bottom: 10px;
    backdrop-filter: blur(12px);
}

/* Expander */
[data-testid="stExpander"] {
    background: rgba(255,255,255,0.055);
    border: 1px solid rgba(255,255,255,0.16);
    border-radius: 16px;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    background: rgba(255,255,255,0.045);
    padding: 7px;
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 14px;
}

.stTabs [data-baseweb="tab"] {
    background: transparent;
    color: #c4cce0;
    border-radius: 10px;
    padding: 9px 16px;
}

.stTabs [aria-selected="true"] {
    background: rgba(167,139,250,0.22) !important;
    color: #ffffff !important;
}

.stTabs [data-baseweb="tab-highlight"] {
    background: #a78bfa;
}

/* Messages */
.stAlert {
    border-radius: 14px;
    border: 1px solid rgba(255,255,255,0.14);
}

/* Divider and footer */
hr {
    border-color: rgba(255,255,255,0.13);
}

.small-note {
    color: #9da9c5 !important;
    text-align: center;
    font-size: 12px;
    line-height: 1.8;
}

/* Checkbox and radio labels */
[data-testid="stCheckbox"] label,
[data-testid="stRadio"] label {
    color: #e9edfa !important;
}

/* Mobile adjustments */
@media (max-width: 700px) {
    .block-container {
        padding-left: 1rem;
        padding-right: 1rem;
        padding-top: 1rem;
    }

    .hero {
        padding: 24px 20px;
        border-radius: 20px;
    }

    .hero h1 {
        font-size: 30px;
    }

    .glass-panel {
        padding: 16px;
    }
}
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# 4. HEADER
# --------------------------------------------------

st.markdown("""
<div class="hero">
    <div class="eyebrow">AI-powered recruitment safety</div>
    <h1>Fake Job Detection AI</h1>
    <p>Know more before you apply. Analyse job postings with machine learning.</p>
    <p>Powered by TF-IDF and Logistic Regression</p>
</div>
""", unsafe_allow_html=True)


# --------------------------------------------------
# 5. SAMPLE JOB DATA
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
# 6. SAMPLE JOB CALLBACK
# --------------------------------------------------

def load_sample():
    selected = st.session_state.sample_choice

    if selected in samples:
        sample = samples[selected]
        for key, value in sample.items():
            st.session_state[key] = value


# --------------------------------------------------
# 7. INITIAL INPUT VALUES
# --------------------------------------------------

default_values = {
    "title": "",
    "company_profile": "",
    "description": "",
    "requirements": "",
    "benefits": "",
    "telecommuting": "No",
    "has_company_logo": "No",
    "has_questions": "No",
    "company_cin": "",
    "company_email": ""
}

for key, value in default_values.items():
    if key not in st.session_state:
        st.session_state[key] = value


# --------------------------------------------------
# 8. SAMPLE JOB SELECTION
# --------------------------------------------------

st.markdown("""
<div class="glass-panel">
    <div class="eyebrow">Quick start</div>
    <div class="section-heading">Try a suggested job</div>
    <div class="section-subtitle">
        Choose a sample profile to automatically fill the job details,
        or enter your own information below.
    </div>
</div>
""", unsafe_allow_html=True)

st.selectbox(
    "Choose a sample job profile",
    ["Choose a sample..."] + list(samples.keys()),
    key="sample_choice",
    on_change=load_sample
)


# --------------------------------------------------
# 9. JOB DETAILS INPUT
# --------------------------------------------------

st.markdown("""
<div class="glass-panel">
    <div class="eyebrow">Job information</div>
    <div class="section-heading">Enter job details</div>
    <div class="section-subtitle">
        Add the information from the job posting for AI analysis.
    </div>
</div>
""", unsafe_allow_html=True)

input_col1, input_col2 = st.columns(2, gap="large")

with input_col1:
    st.text_input(
        "Job Title",
        key="title",
        placeholder="e.g. Software Developer"
    )

    st.text_area(
        "Company Profile",
        key="company_profile",
        height=125,
        placeholder="Describe the company and its work..."
    )

    st.text_area(
        "Job Description",
        key="description",
        height=180,
        placeholder="Paste the complete job description..."
    )

with input_col2:
    st.text_area(
        "Requirements",
        key="requirements",
        height=125,
        placeholder="Skills, education and experience required..."
    )

    st.text_area(
        "Benefits",
        key="benefits",
        height=125,
        placeholder="Salary, benefits and other facilities..."
    )

    st.markdown("#### Additional job information")

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
# 10. OPTIONAL COMPANY DETAILS
# --------------------------------------------------

with st.expander("🏢 Add optional company details", expanded=False):
    company_col1, company_col2 = st.columns(2)

    with company_col1:
        st.text_input(
            "Company Registration Number (CIN)",
            key="company_cin",
            placeholder="Enter 21-character CIN"
        )

    with company_col2:
        st.text_input(
            "Company Official Email",
            key="company_email",
            placeholder="hr@companyname.com"
        )

    st.caption(
        "These details are checked for format only. "
        "They are not included in the machine learning prediction."
    )


# --------------------------------------------------
# 11. PREDICTION BUTTON AND MODEL
# --------------------------------------------------

st.markdown("---")
st.markdown("""
<div style="text-align:center; padding:8px 0 16px 0;">
    <div class="eyebrow">Ready for analysis?</div>
    <h2 style="color:#ffffff;">Check job authenticity</h2>
    <p style="color:#aeb7d0;">
        The model will analyse the job details you entered.
    </p>
</div>
""", unsafe_allow_html=True)

if st.button("🔍  Analyse Job Posting", use_container_width=True):

    if (
        not st.session_state.title.strip()
        or not st.session_state.description.strip()
    ):
        st.warning(
            "Please enter at least the Job Title and Job Description."
        )

    else:
        try:
            # Combine the five text features in training order
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
            ]).tocsr()

            # Check model feature count
            if final_data.shape[1] != model.coef_.shape[1]:
                st.error(
                    "The model feature count does not match the app inputs. "
                    "Please check your training code and saved model."
                )
                st.stop()

            # Predict class and probabilities
            predicted_class = model.predict(final_data)[0]
            probabilities = model.predict_proba(final_data)[0]
            classes = list(model.classes_)

            # Target mapping: 0 = Fake, 1 = Real
            if 0 not in classes or 1 not in classes:
                st.error(
                    "The saved model must use labels 0 = Fake and 1 = Real."
                )
                st.stop()

            fake_probability = probabilities[classes.index(0)] * 100
            real_probability = probabilities[classes.index(1)] * 100

            # ------------------------------------------
            # AI ANALYSIS DASHBOARD
            # ------------------------------------------

            st.markdown("---")
            st.markdown("""
            <div class="hero" style="padding:24px;">
                <div class="eyebrow">Analysis complete</div>
                <h1 style="font-size:32px;">Your job analysis</h1>
                <p>Review the prediction, company details and model insights below.</p>
            </div>
            """, unsafe_allow_html=True)

            result_col, real_col, fake_col = st.columns([2, 1, 1], gap="medium")

            with result_col:
                if predicted_class == 0:
                    result_text = "⚠️ FAKE JOB"
                    result_color = "#fda4af"
                    result_description = "The model classified this posting as potentially fraudulent."
                else:
                    result_text = "✓ REAL JOB"
                    result_color = "#6ee7b7"
                    result_description = "The model classified this posting as likely genuine."

                st.markdown(
                    f"""
                    <div class="result-card">
                        <div class="eyebrow">Job authenticity result</div>
                        <div class="{'fake-result' if predicted_class == 0 else 'real-result'}"
                             style="color:{result_color}; margin:10px 0;">
                            {result_text}
                        </div>
                        <p style="color:#c5cee5;">{result_description}</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            with real_col:
                st.metric("Real Probability", f"{real_probability:.2f}%")
                st.progress(int(round(real_probability)))

            with fake_col:
                st.metric("Fake Probability", f"{fake_probability:.2f}%")
                st.progress(int(round(fake_probability)))

            # Company verification section
            st.markdown("### 🏢 Company details check")
            st.caption(
                "These checks validate the entered formats only. "
                "They do not confirm company registration or email ownership."
            )

            cin_value = st.session_state.get(
                "company_cin", ""
            ).strip().upper()

            email_value = st.session_state.get(
                "company_email", ""
            ).strip()

            company_col1, company_col2 = st.columns(2, gap="large")

            with company_col1:
                with st.container(border=True):
                    st.markdown("#### Company CIN")

                    if not cin_value:
                        st.info("CIN not provided.")
                    elif re.fullmatch(
                        r"[LU]\d{5}[A-Z]{2}\d{4}[A-Z]{3}\d{6}",
                        cin_value
                    ):
                        st.success("CIN format looks valid (21 characters).")
                        st.caption(
                            "Format check only. MCA registration has not been verified."
                        )
                    else:
                        st.error(
                            "CIN format is invalid. An Indian CIN normally contains 21 characters."
                        )

            with company_col2:
                with st.container(border=True):
                    st.markdown("#### Company email")

                    if not email_value:
                        st.info("Company email not provided.")
                    elif re.fullmatch(
                        r"[^\s@]+@[^\s@]+\.[^\s@]+",
                        email_value
                    ):
                        st.success("Email address format looks valid.")

                        email_domain = email_value.rsplit("@", 1)[-1].lower()

                        common_free_domains = {
                            "gmail.com", "yahoo.com", "yahoo.in",
                            "outlook.com", "hotmail.com", "live.com",
                            "icloud.com", "rediffmail.com"
                        }

                        if email_domain in common_free_domains:
                            st.warning(
                                "This uses a common free email provider. "
                                "It is not automatically fake, but independently verify the employer."
                            )
                        else:
                            st.caption(
                                "Company-style domain entered; domain ownership and mailbox existence are not verified."
                            )
                    else:
                        st.error(
                            "Email format is invalid. Enter an address like hr@company.com."
                        )

            # Model explanation in expandable area
            with st.expander("🧠 Explore AI prediction explanation", expanded=False):
                st.write(
                    "These words and job details influenced the Logistic Regression model's prediction."
                )

                feature_names = vectorizer.get_feature_names_out()
                coefficients = model.coef_[0]

                text_coefficients = coefficients[:len(feature_names)]
                numeric_coefficients = coefficients[len(feature_names):]
                text_values = text_data.toarray()[0]

                word_contributions = text_values * text_coefficients
                factors = []

                for i in range(len(feature_names)):
                    contribution = word_contributions[i]

                    if abs(contribution) > 0.000001:
                        factors.append({
                            "factor": feature_names[i],
                            "contribution": contribution
                        })

                numeric_names = [
                    "Telecommuting",
                    "Company Logo",
                    "Screening Questions"
                ]

                numeric_values = numeric_data[0]

                if len(numeric_coefficients) == 3:
                    for i in range(3):
                        contribution = numeric_values[i] * numeric_coefficients[i]

                        if abs(contribution) > 0.000001:
                            value_label = "Yes" if numeric_values[i] == 1 else "No"

                            factors.append({
                                "factor": numeric_names[i] + " = " + value_label,
                                "contribution": contribution
                            })

                factors.sort(
                    key=lambda x: abs(x["contribution"]),
                    reverse=True
                )

                if factors:
                    left_col, right_col = st.columns(2)

                    for index, factor in enumerate(factors[:8]):
                        if factor["contribution"] > 0:
                            direction = "Pushes toward REAL"
                            color = "#6ee7b7"
                            symbol = "✓"
                        else:
                            direction = "Pushes toward FAKE"
                            color = "#fda4af"
                            symbol = "⚠"

                        target_col = left_col if index % 2 == 0 else right_col

                        with target_col:
                            st.markdown(
                                f"""
                                <div class="factor-card">
                                    <b style="color:#f4f5ff;">
                                        {symbol} {factor["factor"]}
                                    </b>
                                    <p style="color:{color}; margin:6px 0;">
                                        {direction}
                                    </p>
                                    <small style="color:#aeb7d0;">
                                        Contribution: {factor["contribution"]:.5f}
                                    </small>
                                </div>
                                """,
                                unsafe_allow_html=True
                            )
                else:
                    st.info(
                        "The model did not find individual feature contributions large enough to display."
                    )

                st.caption(
                    "Positive contribution pushes toward Real (1). "
                    "Negative contribution pushes toward Fake (0). "
                    "These are model associations, not proof of authenticity."
                )

            st.info(
                "Important: This is a machine learning prediction, not a verified investigation. "
                "Always check the company website, official contact details and job offer "
                "before sharing personal information or paying any fees."
            )

        except Exception as e:
            st.error("Something went wrong during prediction.")
            st.exception(e)


# --------------------------------------------------
# 12. FOOTER
# --------------------------------------------------

st.markdown("---")
st.markdown("""
<p class="small-note">
    FAKE JOB DETECTION AI<br>
    Built with Python • Streamlit • TF-IDF • Logistic Regression<br>
    Machine Learning Project
</p>
""", unsafe_allow_html=True)
