
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
    layout="wide"
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
# 3. MIDNIGHT TEAL + SOFT MINT THEME
# --------------------------------------------------

st.markdown("""
<style>

.stApp {
    background: #142B35;
    color: #E7F0ED;
}

header[data-testid="stHeader"] {
    background: transparent;
}

.block-container {
    max-width: 1200px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

h1, h2, h3, h4, p, label {
    color: #E7F0ED !important;
}

/* Header */
.hero {
    background: #1D4148;
    border: 1px solid #37616A;
    border-radius: 18px;
    padding: 30px;
    text-align: center;
    margin-bottom: 25px;
}

.hero h1 {
    color: #78DCC5 !important;
    font-size: 38px;
    font-weight: 800;
}

.hero p {
    color: #C2DAD4 !important;
}

/* Section cards */
.section-card {
    background: #1D4148;
    border: 1px solid #37616A;
    border-radius: 15px;
    padding: 20px;
    margin-bottom: 15px;
}

/* Input fields */
.stTextInput input,
.stTextArea textarea {
    background: #F4F8F6 !important;
    color: #142B35 !important;
    border: 1px solid #8BAFA6 !important;
    border-radius: 9px !important;
}

.stTextInput input:focus,
.stTextArea textarea:focus {
    border-color: #63C7B2 !important;
    box-shadow: 0 0 0 1px #63C7B2 !important;
}

/* Dropdowns */
[data-baseweb="select"] > div {
    background: #F4F8F6 !important;
    border-color: #8BAFA6 !important;
    color: #142B35 !important;
}

/* Buttons */
div.stButton > button {
    background: #63C7B2;
    color: #142B35 !important;
    border: none;
    border-radius: 10px;
    padding: 12px;
    font-weight: 750;
}

div.stButton > button:hover {
    background: #85E0CB;
    color: #142B35 !important;
    border: none;
}

/* Prediction result */
.result-card {
    background: #1D4148;
    border: 1px solid #37616A;
    border-radius: 16px;
    padding: 25px;
    text-align: center;
    margin: 15px 0;
}

.fake-result {
    color: #FF929B;
    font-size: 30px;
    font-weight: 800;
}

.real-result {
    color: #78DCC5;
    font-size: 30px;
    font-weight: 800;
}

/* Feature cards */
.factor-card {
    background: #1D4148;
    border: 1px solid #37616A;
    border-left: 4px solid #63C7B2;
    border-radius: 12px;
    padding: 15px;
    margin-bottom: 10px;
}

/* Footer */
.small-note {
    color: #A9C5BD !important;
    text-align: center;
    font-size: 13px;
}

/* Metrics */
[data-testid="stMetricValue"] {
    color: #78DCC5 !important;
}

/* Progress bar */
.stProgress > div > div > div {
    background-color: #63C7B2;
}

/* Divider */
hr {
    border-color: #37616A;
}

</style>
""", unsafe_allow_html=True)




# --------------------------------------------------
# 4. HEADER
# --------------------------------------------------

st.markdown("""
<div class="hero">
    <h1>🔍 Fake Job Detection AI</h1>
    <p>Smart Machine Learning System to Identify Fake and Genuine Jobs</p>
    <p>Powered by Artificial Intelligence and Logistic Regression</p>
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
    "has_questions": "No"
}

for key, value in default_values.items():

    if key not in st.session_state:
        st.session_state[key] = value


# --------------------------------------------------
# 8. SAMPLE JOB SELECTION
# --------------------------------------------------

st.markdown("## ✨ Try a Suggested Job")

st.selectbox(
    "Choose a sample job profile",
    ["Choose a sample..."] + list(samples.keys()),
    key="sample_choice",
    on_change=load_sample
)

st.caption(
    "Select a sample job to automatically fill the details, "
    "or enter your own job details."
)


# --------------------------------------------------
# 9. JOB DETAILS
# --------------------------------------------------

st.markdown("## 📝 Enter Job Details")

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
        height=130
    )

    st.text_input(
        "Company Registration Number (CIN)",
        key="company_cin",
        placeholder="Enter 21-character CIN (optional)"
    )

    st.text_input(
        "Company Official Email",
        key="company_email",
        placeholder="hr@companyname.com (optional)"
    )

    st.text_area(
        "Job Description",
        key="description",
        height=180
    )


with col2:

    st.text_area(
        "Requirements",
        key="requirements",
        height=130
    )

    st.text_area(
        "Benefits",
        key="benefits",
        height=130
    )

    st.markdown("### Additional Information")

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
# 10. PREDICTION BUTTON
# --------------------------------------------------

st.markdown("---")
st.markdown("## 🚀 Detect Job Authenticity")

if st.button("🔍 Predict Job Authenticity", use_container_width=True):

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

            # Predict class

            predicted_class = model.predict(final_data)[0]

            # Predict probabilities

            probabilities = model.predict_proba(final_data)[0]

            classes = list(model.classes_)

            # New target mapping: 0 = Fake, 1 = Real
            fake_probability = (
                probabilities[classes.index(0)] * 100
            )

            real_probability = (
                probabilities[classes.index(1)] * 100
            )

            # ------------------------------------------
            # AI ANALYSIS DASHBOARD
            # ------------------------------------------

            st.markdown("---")

            st.markdown("""
            <div style="text-align:center; padding:10px;">
                <h2 style="color:#78DCC5;">AI Analysis Dashboard</h2>
                <p style="color:#A9C5BD;">
                    Job authenticity analysis and verification insights
                </p>
            </div>
            """, unsafe_allow_html=True)

            # ------------------------------------------
            # MAIN PREDICTION AREA
            # ------------------------------------------

            result_col, real_col, fake_col = st.columns([2, 1, 1])

            with result_col:

                if predicted_class == 0:
                    result_text = "⚠️ FAKE JOB"
                    result_color = "#FF929B"
                    result_description = "Potentially fraudulent posting"
                else:
                    result_text = "✅ REAL JOB"
                    result_color = "#78DCC5"
                    result_description = "Likely genuine job posting"

                st.markdown(
                    f"""
                    <div style="
                        background:#1D4148;
                        border:1px solid #37616A;
                        border-left:5px solid {result_color};
                        border-radius:14px;
                        padding:22px;
                        min-height:125px;
                    ">
                        <small style="color:#A9C5BD;">
                            JOB AUTHENTICITY RESULT
                        </small>
                        <h2 style="color:{result_color}; margin:10px 0;">
                            {result_text}
                        </h2>
                        <p style="color:#C2DAD4;">
                            {result_description}
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            with real_col:
                st.metric(
                    "Real Probability",
                    f"{real_probability:.2f}%"
                )
                st.progress(int(round(real_probability)))

            with fake_col:
                st.metric(
                    "Fake Probability",
                    f"{fake_probability:.2f}%"
                )
                st.progress(int(round(fake_probability)))

            # ------------------------------------------
            # COMPANY VERIFICATION PANEL
            # ------------------------------------------

            st.markdown("### 🏢 Company Verification")

            cin_value = st.session_state.get(
                "company_cin", ""
            ).strip().upper()

            email_value = st.session_state.get(
                "company_email", ""
            ).strip()

            company_col1, company_col2 = st.columns(2)

            with company_col1:

                st.markdown("#### Registration Number (CIN)")

                if not cin_value:
                    st.info("CIN not provided.")

                elif re.fullmatch(
                    r"[LU]\d{5}[A-Z]{2}\d{4}[A-Z]{3}\d{6}",
                    cin_value
                ):
                    st.success("CIN format looks valid.")
                    st.caption(
                        "Format checked only. MCA registration not verified."
                    )

                else:
                    st.error("Invalid CIN format.")

            with company_col2:

                st.markdown("#### Official Email")

                if not email_value:
                    st.info("Company email not provided.")

                elif re.fullmatch(
                    r"[^\s@]+@[^\s@]+\.[^\s@]+",
                    email_value
                ):
                    st.success("Email format looks valid.")

                    email_domain = email_value.rsplit(
                        "@", 1
                    )[-1].lower()

                    common_free_domains = {
                        "gmail.com", "yahoo.com", "yahoo.in",
                        "outlook.com", "hotmail.com",
                        "live.com", "icloud.com",
                        "rediffmail.com"
                    }

                    if email_domain in common_free_domains:
                        st.warning(
                            "Free email provider. Verify the employer independently."
                        )
                    else:
                        st.caption(
                            "Domain ownership and mailbox not verified."
                        )

                else:
                    st.error("Invalid email format.")

            # ------------------------------------------
            # AI FEATURE ANALYSIS
            # ------------------------------------------

            st.markdown("---")

            with st.expander(
                "🧠 Explore AI Prediction Explanation",
                expanded=False
            ):

                st.write(
                    "These are the words and job details that influenced "
                    "the Logistic Regression prediction."
                )

                if (
                    len(model.classes_) != 2
                    or 1 not in classes
                    or 0 not in classes
                ):

                    st.warning(
                        "Explanation requires a binary model "
                        "with labels 0 = Fake and 1 = Real."
                    )

                else:

                    feature_names = vectorizer.get_feature_names_out()

                    coefficients = model.coef_[0]

                    text_coefficients = coefficients[
                        :len(feature_names)
                    ]

                    numeric_coefficients = coefficients[
                        len(feature_names):
                    ]

                    text_values = text_data.toarray()[0]

                    word_contributions = (
                        text_values * text_coefficients
                    )

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

                            contribution = (
                                numeric_values[i] *
                                numeric_coefficients[i]
                            )

                            if abs(contribution) > 0.000001:

                                value_label = (
                                    "Yes" if numeric_values[i] == 1
                                    else "No"
                                )

                                factors.append({
                                    "factor": (
                                        numeric_names[i] +
                                        " = " + value_label
                                    ),
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
                                direction = "Towards REAL"
                                color = "#78DCC5"
                                symbol = "✅"
                            else:
                                direction = "Towards FAKE"
                                color = "#FF929B"
                                symbol = "⚠️"

                            target_col = (
                                left_col if index % 2 == 0
                                else right_col
                            )

                            with target_col:

                                st.markdown(
                                    f"""
                                    <div style="
                                        background:#1D4148;
                                        border:1px solid #37616A;
                                        border-radius:10px;
                                        padding:13px;
                                        margin-bottom:10px;
                                    ">
                                        <b style="color:#E7F0ED;">
                                            {symbol} {factor["factor"]}
                                        </b>
                                        <p style="
                                            color:{color};
                                            margin:5px 0;
                                        ">
                                            {direction}
                                        </p>
                                        <small style="color:#A9C5BD;">
                                            Contribution:
                                            {factor["contribution"]:.5f}
                                        </small>
                                    </div>
                                    """,
                                    unsafe_allow_html=True
                                )

                    else:

                        st.info(
                            "No significant individual feature contributions found."
                        )

                    st.caption(
                        "Positive contribution pushes toward Real (1). "
                        "Negative contribution pushes toward Fake (0). "
                        "These are model associations, not proof of authenticity."
                    )

            # ------------------------------------------
            # SAFETY MESSAGE
            # ------------------------------------------

            st.info(
                "Important: This is a machine learning prediction, "
                "not a verified investigation. Always check the "
                "company website, official contact details and "
                "job offer before sharing personal information "
                "or paying any fees."
            )


        except Exception as e:

            st.error("Something went wrong during prediction.")
            st.exception(e)


# --------------------------------------------------
# 11. FOOTER
# --------------------------------------------------

st.markdown("---")

st.markdown("""
<p class="small-note">
    Fake Job Detection AI | Machine Learning Project
    <br>
    Built using Python, Streamlit, TF-IDF and Logistic Regression
</p>
""", unsafe_allow_html=True)
