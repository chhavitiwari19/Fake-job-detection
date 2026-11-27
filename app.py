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
    page_icon="🌙",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --------------------------------------------------
# 2. LOAD ML FILES
# --------------------------------------------------

@st.cache_resource
def load_files():
    model = joblib.load("fake_job_model.pkl")
    vectorizer = joblib.load("fake_job_vectorizer.pkl")
    return model, vectorizer

try:
    model, vectorizer = load_files()
except Exception as e:
    st.error("Unable to load model files. Check your repository.")
    st.exception(e)
    st.stop()

# --------------------------------------------------
# 3. CELESTIAL TWILIGHT THEME
# --------------------------------------------------

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&family=Playfair+Display:wght@600;700;800&display=swap');

:root {
    --night: #25223F;
    --purple: #51466F;
    --lavender: #9385B5;
    --peach: #E9A18F;
    --sunset: #D98279;
    --cream: #FFF7F0;
    --ink: #302B40;
    --muted: #777087;
}

.stApp {
    background:
        radial-gradient(ellipse at 8% 5%, rgba(147,133,181,.30), transparent 32%),
        radial-gradient(ellipse at 92% 12%, rgba(233,161,143,.35), transparent 32%),
        radial-gradient(ellipse at 50% 100%, rgba(217,130,121,.18), transparent 40%),
        linear-gradient(135deg, #F7F0F5, #FCE9E0 55%, #F1EAF7);
    color: var(--ink);
    font-family: 'DM Sans', sans-serif;
}

.block-container {
    max-width: 1200px;
    padding-top: 1.6rem;
    padding-bottom: 3rem;
}

h1, h2, h3, h4, p, label, li {
    color: var(--ink);
    font-family: 'DM Sans', sans-serif;
}

h1, h2 {
    font-family: 'Playfair Display', serif;
}

[data-testid="stCaptionContainer"] p {
    color: var(--muted) !important;
}

/* HERO */
.hero {
    position: relative;
    overflow: hidden;
    padding: 43px;
    margin: 10px 0 25px;
    border-radius: 27px;
    background:
        radial-gradient(circle at 86% 18%, rgba(255,226,207,.25), transparent 23%),
        radial-gradient(circle at 90% 15%, rgba(255,255,255,.13), transparent 36%),
        linear-gradient(120deg, #25223F 0%, #51466F 53%, #A16E86 100%);
    border: 1px solid rgba(255,225,213,.35);
    box-shadow: 0 18px 45px rgba(37,34,63,.20);
}

.hero:after {
    content: "☾";
    position: absolute;
    right: 7%;
    top: 0;
    color: rgba(255,244,225,.17);
    font-size: 145px;
    line-height: 1.3;
    pointer-events: none;
}

.hero h1 {
    position: relative;
    z-index: 1;
    color: #FFF8F3 !important;
    font-family: 'Playfair Display', serif;
    font-size: clamp(32px, 4vw, 49px);
    margin: 13px 0;
    letter-spacing: -.7px;
}

.hero p {
    position: relative;
    z-index: 1;
    color: #F0E6F1 !important;
    font-size: 15px;
    line-height: 1.7;
}

.eyebrow {
    position: relative;
    z-index: 1;
    color: #F6C9B7;
    font-size: 11px;
    font-weight: 800;
    letter-spacing: 2px;
    text-transform: uppercase;
}

.hero-badge {
    position: relative;
    z-index: 1;
    display: inline-block;
    padding: 7px 13px;
    margin: 12px 5px 0 0;
    border-radius: 30px;
    background: rgba(255,255,255,.10);
    border: 1px solid rgba(255,230,220,.35);
    color: #FFF1E9;
    font-size: 11px;
    font-weight: 700;
}

/* PANELS */
.panel {
    background: rgba(255,250,248,.92);
    border: 1px solid rgba(147,133,181,.25);
    border-radius: 21px;
    padding: 24px;
    margin: 14px 0;
    box-shadow: 0 9px 27px rgba(49,40,75,.07);
}

.section-heading {
    font-family: 'Playfair Display', serif;
    font-size: 24px;
    font-weight: 700;
    color: #51466F !important;
}

.section-subtitle {
    font-size: 13px;
    color: #777087 !important;
    margin-top: 5px;
    line-height: 1.7;
}

.gold-line {
    width: 52px;
    height: 3px;
    background: linear-gradient(90deg, #9385B5, #E9A18F);
    border-radius: 10px;
    margin: 12px 0;
}

/* INPUTS */
.stTextInput input,
.stTextArea textarea {
    background: #FFFCFA !important;
    color: #302B40 !important;
    border: 1px solid #DCD2E3 !important;
    border-radius: 12px !important;
}

.stTextInput input:focus,
.stTextArea textarea:focus {
    border-color: #9385B5 !important;
    box-shadow: 0 0 0 1px #9385B5 !important;
}

.stTextInput input::placeholder,
.stTextArea textarea::placeholder {
    color: #A49BAD !important;
    opacity: 1 !important;
}

[data-baseweb="select"] > div {
    background: #FFFCFA !important;
    border: 1px solid #DCD2E3 !important;
    border-radius: 12px !important;
}

[data-baseweb="select"] * {
    color: #302B40 !important;
}

[data-baseweb="popover"],
[data-baseweb="menu"] {
    background: #FFFCFA !important;
}

[data-baseweb="menu"] * {
    color: #302B40 !important;
}

/* BUTTONS */
div.stButton > button {
    background: linear-gradient(110deg, #51466F, #80658F 65%, #D98279);
    color: #FFFFFF !important;
    border: 1px solid rgba(255,255,255,.35);
    border-radius: 13px;
    min-height: 48px;
    font-weight: 800;
    transition: all .2s ease;
    box-shadow: 0 7px 19px rgba(81,70,111,.20);
}

div.stButton > button p {
    color: #FFFFFF !important;
}

div.stButton > button:hover {
    background: linear-gradient(110deg, #383251, #69567F, #C87575);
    color: #FFFFFF !important;
    transform: translateY(-2px);
    box-shadow: 0 11px 24px rgba(81,70,111,.27);
}

/* RESULT CARDS */
.result-card {
    background: rgba(255,252,250,.95);
    border: 1px solid #E2D7E6;
    border-radius: 20px;
    padding: 25px;
    min-height: 190px;
    box-shadow: 0 10px 25px rgba(49,40,75,.07);
}

.result-card.real {
    background: linear-gradient(135deg, #E8F2EB, #FFFCFA);
    border-color: #B9D4C2;
}

.result-card.fake {
    background: linear-gradient(135deg, #F9E5E6, #FFFCFA);
    border-color: #E7B9C1;
}

.result-emoji {
    font-size: 42px;
    margin: 12px 0;
}

.real-result {
    color: #397451;
    font: 800 clamp(25px,3vw,34px) 'Playfair Display', serif;
}

.fake-result {
    color: #A33E58;
    font: 800 clamp(25px,3vw,34px) 'Playfair Display', serif;
}

.result-description {
    color: #625B70 !important;
    font-size: 14px;
    line-height: 1.7;
}

/* METRICS */
[data-testid="stMetric"] {
    background: rgba(255,252,250,.95);
    border: 1px solid #E2D7E6;
    border-radius: 17px;
    padding: 17px;
    box-shadow: 0 7px 20px rgba(49,40,75,.05);
}

[data-testid="stMetricLabel"],
[data-testid="stMetricLabel"] * {
    color: #777087 !important;
}

[data-testid="stMetricValue"],
[data-testid="stMetricValue"] * {
    color: #51466F !important;
    font-family: 'Playfair Display', serif;
}

/* PROGRESS */
.stProgress > div > div > div {
    background: linear-gradient(90deg, #9385B5, #E9A18F);
    border-radius: 20px;
}

/* EXPLANATION */
.factor-card {
    background: #FFFCFA;
    border: 1px solid #E2D7E6;
    border-left: 4px solid #D98279;
    border-radius: 13px;
    padding: 16px;
    margin-bottom: 12px;
}

.factor-card b {
    color: #51466F;
}

.factor-card p {
    margin: 7px 0;
}

.factor-card small {
    color: #777087;
}

[data-testid="stExpander"] {
    background: rgba(255,252,250,.92);
    border: 1px solid #DED4E5;
    border-radius: 15px;
}

[data-testid="stExpander"] summary,
[data-testid="stExpander"] summary span {
    color: #51466F !important;
}


/* LIGHT, READABLE ALERTS */
[data-testid="stAlert"] {
    background: #FFF8F3 !important;
    color: #302B40 !important;
    border: 1px solid #E7D5D8 !important;
    border-radius: 14px !important;
}

[data-testid="stAlert"] *,
[data-testid="stAlert"] p,
[data-testid="stAlert"] span,
[data-testid="stAlert"] div {
    color: #302B40 !important;
}

[data-testid="stAlert"] svg {
    fill: currentColor;
}


/* FOOTER */
.footer {
    text-align: center;
    color: #777087;
    font-size: 12px;
    line-height: 2;
    padding: 22px 5px;
}

.footer strong {
    color: #51466F;
    letter-spacing: 2px;
}

/* MOBILE */
@media(max-width:700px) {
    .block-container {
        padding: 1rem;
    }
    .hero {
        padding: 23px;
    }
    .hero:after {
        font-size: 95px;
        right: 4%;
    }
    .panel {
        padding: 16px;
    }
    .result-card {
        padding: 19px;
    }
}
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# 4. SAMPLE JOBS
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
        "benefits": "Fixed salary, professional environment and growth.",
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
# 5. SESSION STATE
# --------------------------------------------------

defaults = {
    "title": "",
    "company_profile": "",
    "description": "",
    "requirements": "",
    "benefits": "",
    "telecommuting": "No",
    "has_company_logo": "No",
    "has_questions": "No",
    "company_cin": "",
    "company_email": "",
    "page": "input",
    "prediction": None,
    "sample_choice": "Choose a sample..."
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


def load_sample():
    chosen = st.session_state.sample_choice
    if chosen in samples:
        for key, value in samples[chosen].items():
            st.session_state[key] = value


# --------------------------------------------------
# 6. PREDICTION FUNCTION
# --------------------------------------------------

def run_prediction():
    combined_text = " ".join([
        st.session_state.title,
        st.session_state.company_profile,
        st.session_state.description,
        st.session_state.requirements,
        st.session_state.benefits
    ])

    text_data = vectorizer.transform([combined_text])

    numeric_data = np.array([[
        1 if st.session_state.telecommuting == "Yes" else 0,
        1 if st.session_state.has_company_logo == "Yes" else 0,
        1 if st.session_state.has_questions == "Yes" else 0
    ]])

    final_data = hstack([
        text_data,
        csr_matrix(numeric_data)
    ]).tocsr()

    if final_data.shape[1] != model.coef_.shape[1]:
        raise ValueError("Model feature count does not match app inputs.")

    predicted_class = int(model.predict(final_data)[0])
    probabilities = model.predict_proba(final_data)[0]
    classes = list(model.classes_)

    if 0 not in classes or 1 not in classes:
        raise ValueError("The saved model must use labels 0 = Fake and 1 = Real.")

    st.session_state.prediction = {
        "predicted_class": predicted_class,
        "fake_probability": float(probabilities[classes.index(0)] * 100),
        "real_probability": float(probabilities[classes.index(1)] * 100),
        "text_data": text_data,
        "numeric_data": numeric_data
    }
    st.session_state.page = "result"


# --------------------------------------------------
# 7. FOOTER
# --------------------------------------------------

def footer():
    st.markdown("""
    <div class="footer">
        <strong>FAKE JOB DETECTION AI</strong><br>
        Powered by Python • Streamlit • TF-IDF • Logistic Regression<br>
        Intelligent Recruitment Safety System
    </div>
    """, unsafe_allow_html=True)


# --------------------------------------------------
# 8. HEADER FUNCTION
# --------------------------------------------------

def header(step, title, subtitle):
    st.markdown(f"""
    <div class="hero">
        <div class="eyebrow">
            ☾ RECRUITMENT SECURITY • {step}
        </div>
        <h1>{title}</h1>
        <p>{subtitle}</p>
        <span class="hero-badge">✦ AI POWERED</span>
        <span class="hero-badge">✦ JOB SAFETY</span>
        <span class="hero-badge">✦ SMART ANALYSIS</span>
    </div>
    """, unsafe_allow_html=True)


# --------------------------------------------------
# 9. PAGE 1 - INPUT SCREEN
# --------------------------------------------------

if st.session_state.page == "input":
    header(
        "01 / 02",
        "Fake Job Detection AI",
        "An intelligent way to examine job postings and identify potential recruitment risks."
    )

    st.markdown("""
    <div class="panel">
        <div class="eyebrow">01 • QUICK START</div>
        <div class="section-heading">Explore a job profile</div>
        <div class="gold-line"></div>
        <div class="section-subtitle">
            Select a sample job to automatically fill the form,
            or enter your own job information.
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.selectbox(
        "Choose a sample job profile",
        ["Choose a sample..."] + list(samples.keys()),
        key="sample_choice",
        on_change=load_sample
    )

    st.markdown("""
    <div class="panel">
        <div class="eyebrow">02 • JOB INFORMATION</div>
        <div class="section-heading">Job posting details</div>
        <div class="gold-line"></div>
        <div class="section-subtitle">
            Provide the information available in the job advertisement.
            Fields marked with * are required.
        </div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.text_input(
            "💼 Job Title *",
            key="title",
            placeholder="e.g. Software Developer"
        )
        st.text_area(
            "🏢 Company Profile",
            key="company_profile",
            height=135,
            placeholder="Describe the company and its activities..."
        )
        st.text_area(
            "📄 Job Description *",
            key="description",
            height=190,
            placeholder="Paste the complete job description..."
        )

    with col2:
        st.text_area(
            "🎯 Requirements",
            key="requirements",
            height=135,
            placeholder="Skills, education and experience..."
        )
        st.text_area(
            "🎁 Benefits",
            key="benefits",
            height=135,
            placeholder="Salary, facilities and other benefits..."
        )
        st.markdown("""
        <div class="section-heading">Additional information</div>
        """, unsafe_allow_html=True)
        st.selectbox("🏠 Telecommuting", ["No", "Yes"], key="telecommuting")
        st.selectbox("🖼️ Company logo available?", ["No", "Yes"], key="has_company_logo")
        st.selectbox("❓ Screening questions available?", ["No", "Yes"], key="has_questions")

    with st.expander("🏛️ Optional company verification details"):
        c1, c2 = st.columns(2)
        with c1:
            st.text_input(
                "Company Registration Number (CIN)",
                key="company_cin",
                placeholder="Enter 21-character CIN"
            )
        with c2:
            st.text_input(
                "Company Official Email",
                key="company_email",
                placeholder="hr@companyname.com"
            )
        st.caption(
            "These fields perform format checks only. "
            "They are not included in the machine learning prediction."
        )

    st.markdown("""
    <div class="panel" style="text-align:center">
        <div class="eyebrow">03 • SECURITY ANALYSIS</div>
        <div class="section-heading">Ready to examine this opportunity?</div>
        <div class="gold-line" style="margin:12px auto"></div>
        <div class="section-subtitle">
            Your AI-generated prediction will appear on a separate report screen.
        </div>
    </div>
    """, unsafe_allow_html=True)

    if st.button("🌙  Analyse Job Posting", use_container_width=True):
        if (
            not st.session_state.title.strip()
            or not st.session_state.description.strip()
        ):
            st.warning("Please enter both Job Title and Job Description.")
        else:
            try:
                with st.spinner("Examining job posting..."):
                    run_prediction()
                st.rerun()
            except Exception as e:
                st.error("Prediction could not be completed.")
                st.exception(e)

    footer()


# --------------------------------------------------
# 10. PAGE 2 - RESULT SCREEN
# --------------------------------------------------

else:
    result = st.session_state.prediction

    if result is None:
        st.session_state.page = "input"
        st.rerun()

    predicted_class = result["predicted_class"]
    fake_probability = result["fake_probability"]
    real_probability = result["real_probability"]
    text_data = result["text_data"]
    numeric_data = result["numeric_data"]

    header(
        "02 / 02",
        "Your Job Analysis Report",
        "Your AI assessment is ready. Review the results and safety information below."
    )

    if st.button("← Back to Edit Job Details"):
        st.session_state.page = "input"
        st.rerun()

    if predicted_class == 0:
        result_text = "FAKE JOB"
        result_emoji = "🚨"
        result_description = (
            "The model classified this posting as potentially fraudulent. "
            "Verify the employer carefully before proceeding."
        )
        result_class = "fake-result"
        card_class = "fake"
    else:
        result_text = "REAL JOB"
        result_emoji = "🛡️"
        result_description = (
            "The model classified this posting as likely genuine. "
            "Independent verification is still important."
        )
        result_class = "real-result"
        card_class = "real"

    result_col, real_col, fake_col = st.columns([2, 1, 1], gap="medium")

    with result_col:
        st.markdown(f"""
        <div class="result-card {card_class}">
            <div class="eyebrow">JOB AUTHENTICITY ASSESSMENT</div>
            <div class="result-emoji">{result_emoji}</div>
            <div class="{result_class}">{result_text}</div>
            <p class="result-description">{result_description}</p>
        </div>
        """, unsafe_allow_html=True)

    with real_col:
        st.metric("🛡️ Real Probability", f"{real_probability:.2f}%")
        st.progress(int(round(real_probability)))

    with fake_col:
        st.metric("🚨 Fake Probability", f"{fake_probability:.2f}%")
        st.progress(int(round(fake_probability)))

    st.markdown("""
    <div class="panel">
        <div class="eyebrow">ASSESSMENT SUMMARY</div>
        <div class="section-heading">Prediction overview</div>
        <div class="gold-line"></div>
    </div>
    """, unsafe_allow_html=True)

    if predicted_class == 0:
        st.error(
            "High concern: The model predicts this posting as FAKE. "
            "Do not pay fees or share sensitive documents without verification."
        )
    else:
        st.success(
            "The model predicts this posting as REAL. "
            "This is not a guarantee of authenticity; verify the employer independently."
        )

    st.markdown("""
    <div class="panel">
        <div class="eyebrow">COMPANY INFORMATION</div>
        <div class="section-heading">Company details review</div>
        <div class="gold-line"></div>
        <div class="section-subtitle">
            Check the format of the supplied company information.
        </div>
    </div>
    """, unsafe_allow_html=True)

    cin_value = st.session_state.company_cin.strip().upper()
    email_value = st.session_state.company_email.strip()
    company_col1, company_col2 = st.columns(2, gap="large")

    with company_col1:
        with st.container(border=True):
            st.markdown("#### 🏛️ Company CIN")
            if not cin_value:
                st.info("CIN was not provided.")
            elif re.fullmatch(r"[LU]\d{5}[A-Z]{2}\d{4}[A-Z]{3}\d{6}", cin_value):
                st.success("CIN format looks valid.")
                st.caption("Format check only. MCA registration has not been verified.")
            else:
                st.error("Invalid CIN format. An Indian CIN normally contains 21 characters.")

    with company_col2:
        with st.container(border=True):
            st.markdown("#### 📧 Company Email")
            if not email_value:
                st.info("Company email was not provided.")
            elif re.fullmatch(r"[^\s@]+@[^\s@]+\.[^\s@]+", email_value):
                st.success("Email address format looks valid.")
                email_domain = email_value.rsplit("@", 1)[-1].lower()
                free_domains = {
                    "gmail.com", "yahoo.com", "yahoo.in", "outlook.com",
                    "hotmail.com", "live.com", "icloud.com", "rediffmail.com"
                }
                if email_domain in free_domains:
                    st.warning(
                        "This uses a common free email provider. "
                        "It is not automatically fake, but independently verify the employer."
                    )
                else:
                    st.caption(
                        "Company-style domain entered. "
                        "Domain ownership and mailbox existence are not verified."
                    )
            else:
                st.error("Invalid email format. Example: hr@company.com")

    with st.expander("🧠 Explore AI Prediction Explanation", expanded=False):
        st.write(
            "The following words and job features contributed to the model's prediction."
        )

        feature_names = vectorizer.get_feature_names_out()
        coefficients = model.coef_[0]
        text_coefficients = coefficients[:len(feature_names)]
        numeric_coefficients = coefficients[len(feature_names):]
        text_values = text_data.toarray()[0]
        word_contributions = text_values * text_coefficients
        factors = []

        for i, contribution in enumerate(word_contributions):
            if abs(contribution) > 0.000001:
                factors.append({
                    "factor": feature_names[i],
                    "contribution": float(contribution)
                })

        numeric_names = ["Telecommuting", "Company Logo", "Screening Questions"]
        for i in range(min(3, len(numeric_coefficients))):
            contribution = numeric_data[0][i] * numeric_coefficients[i]
            if abs(contribution) > 0.000001:
                value_label = "Yes" if numeric_data[0][i] == 1 else "No"
                factors.append({
                    "factor": numeric_names[i] + " = " + value_label,
                    "contribution": float(contribution)
                })

        factors.sort(key=lambda x: abs(x["contribution"]), reverse=True)

        if factors:
            left_col, right_col = st.columns(2)
            for index, factor in enumerate(factors[:8]):
                positive = factor["contribution"] > 0
                if positive:
                    direction = "Pushes toward REAL"
                    color = "#397451"
                    symbol = "🟢"
                else:
                    direction = "Pushes toward FAKE"
                    color = "#A33E58"
                    symbol = "🔴"

                target = left_col if index % 2 == 0 else right_col
                with target:
                    st.markdown(f"""
                    <div class="factor-card">
                        <b>{symbol} {factor['factor']}</b>
                        <p style="color:{color}">{direction}</p>
                        <small>Contribution: {factor['contribution']:.5f}</small>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.info(
                "The model did not find individual feature contributions large enough to display."
            )

        st.caption(
            "Positive contribution pushes toward Real (1). "
            "Negative contribution pushes toward Fake (0). "
            "These are model associations, not proof of authenticity."
        )

    st.warning(
        "IMPORTANT SAFETY NOTICE: This is a machine learning prediction, "
        "not a verified investigation. Always check the company's official "
        "website, contact details and job offer before sharing personal "
        "information or paying any fees."
    )

    footer()
