import streamlit as st
import numpy as np
import joblib
import re
from scipy.sparse import hstack, csr_matrix

st.set_page_config(
    page_title="Fake Job Detection AI",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="collapsed"
)

@st.cache_resource
def load_files():
    model = joblib.load("fake_job_model.pkl")
    vectorizer = joblib.load("fake_job_vectorizer.pkl")
    return model, vectorizer

try:
    model, vectorizer = load_files()
except Exception as e:
    st.error("Unable to load model files. Check that both .pkl files are in the repository.")
    st.exception(e)
    st.stop()

# -------------------- AURORA DESIGN --------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&family=Manrope:wght@500;600;700;800&display=swap');

:root {
  --ink:#34445a; --muted:#718096; --mint:#dff5ed;
  --lavender:#eee9ff; --blue:#e4f2ff; --peach:#fff0e8;
}
.stApp {
 background:
   radial-gradient(ellipse at 8% 5%,rgba(197,235,222,.85),transparent 35%),
   radial-gradient(ellipse at 92% 7%,rgba(220,211,255,.78),transparent 34%),
   radial-gradient(ellipse at 75% 95%,rgba(255,224,207,.72),transparent 38%),
   linear-gradient(135deg,#f7fbf9 0%,#f5f7ff 52%,#fff9f5 100%);
 color:var(--ink); font-family:'DM Sans',sans-serif;
}
.block-container {max-width:1180px;padding-top:1.7rem;padding-bottom:3rem}
h1,h2,h3,h4,p,label,li {color:var(--ink);font-family:'DM Sans',sans-serif}
h1,h2,h3 {font-family:'Manrope',sans-serif}
[data-testid="stCaptionContainer"] p,.stCaption {color:var(--muted)!important}

.hero,.glass-panel,.result-card,.info-card {
 background:rgba(255,255,255,.76);
 border:1px solid rgba(255,255,255,.95); border-radius:24px;
 box-shadow:0 12px 35px rgba(100,125,145,.10);
 backdrop-filter:blur(16px); -webkit-backdrop-filter:blur(16px);
}
.hero {
 padding:32px;margin:8px 0 22px;position:relative;overflow:hidden;
 background:
   radial-gradient(ellipse at 90% 10%,rgba(204,228,255,.75),transparent 38%),
   radial-gradient(ellipse at 8% 100%,rgba(213,244,229,.8),transparent 42%),
   rgba(255,255,255,.78);
}
.hero h1 {font-size:clamp(31px,4vw,46px);margin:9px 0;color:#33465c;letter-spacing:-1.4px}
.hero p {color:#687b8e;margin-bottom:5px}
.eyebrow {color:#579a83;font-size:11px;font-weight:800;letter-spacing:2px;text-transform:uppercase}
.glass-panel {padding:22px;margin:12px 0}
.section-heading {font-family:'Manrope',sans-serif;font-size:22px;font-weight:800;color:#3d5368}
.section-subtitle {color:var(--muted);font-size:13px;margin:4px 0 12px}
.stTextInput input,.stTextArea textarea {
 background:rgba(255,255,255,.92)!important;color:#34445a!important;
 border:1px solid #dce7e7!important;border-radius:13px!important;
}
.stTextInput input:focus,.stTextArea textarea:focus {
 border-color:#8fcbb5!important;box-shadow:0 0 0 1px #8fcbb5!important;
}
.stTextInput input::placeholder,.stTextArea textarea::placeholder {color:#9aa9b5!important}
[data-baseweb="select"]>div {
 background:rgba(255,255,255,.94)!important;border:1px solid #dce7e7!important;border-radius:13px!important;
}
[data-baseweb="select"] * {color:#34445a!important}
[data-baseweb="popover"],[data-baseweb="menu"] {background:#fff!important}
div.stButton>button {
 background:linear-gradient(105deg,#bce9d8 0%,#c9dcf8 52%,#e4dafa 100%);
 color:#344b5c!important;border:1px solid #dcebe8;
 border-radius:14px;min-height:49px;font-weight:800;
 box-shadow:0 8px 22px rgba(119,158,153,.16);transition:all .2s ease;
}
div.stButton>button:hover {transform:translateY(-2px);box-shadow:0 12px 26px rgba(119,158,153,.22);border-color:#b6d9ce}
.result-card {padding:25px;min-height:180px}
.result-card.real {border-color:#c3e8d8;background:linear-gradient(135deg,#e5f7ef,#fff 88%)}
.result-card.fake {border-color:#f2d0c6;background:linear-gradient(135deg,#fff0eb,#fff 88%)}
.result-emoji {font-size:43px;line-height:1.2;margin:8px 0}
.real-result {color:#398568;font:800 clamp(25px,3vw,34px) 'Manrope',sans-serif}
.fake-result {color:#c56d63;font:800 clamp(25px,3vw,34px) 'Manrope',sans-serif}
.result-description {color:#66798a!important;font-size:14px;margin-top:8px}
[data-testid="stMetric"] {
 background:rgba(255,255,255,.82);border:1px solid #e5ecec;
 border-radius:17px;padding:15px;
}
[data-testid="stMetricLabel"] {color:#718394!important}
[data-testid="stMetricValue"] {color:#3d5368!important;font-family:'Manrope',sans-serif}
.stProgress>div>div>div {background:linear-gradient(90deg,#9bd8bd,#a9c9f0);border-radius:20px}
.factor-card {
 background:rgba(255,255,255,.82);border:1px solid #e5ecec;
 border-left:4px solid #a9cdbd;border-radius:15px;padding:15px;margin-bottom:10px;
}
[data-testid="stExpander"] {background:rgba(255,255,255,.76);border:1px solid #e4ebeb;border-radius:16px}
.stAlert {border-radius:15px}
hr {border-color:#e2eaea}
.small-note {color:#82919d!important;text-align:center;font-size:12px;line-height:1.9}
.pill {
 display:inline-block;padding:6px 11px;border-radius:99px;
 background:#edf6f2;border:1px solid #d9ebe3;
 color:#5d7e72;font-size:11px;font-weight:700;letter-spacing:.4px;
}
@media(max-width:700px) {
 .block-container{padding:1rem}.hero{padding:22px}.glass-panel{padding:16px}
 .result-card{padding:19px}
}
</style>
""", unsafe_allow_html=True)

# -------------------- SAMPLE JOBS --------------------
samples = {
    "Software Developer": {
        "title":"Software Developer",
        "company_profile":"We are a technology company developing software solutions for businesses.",
        "description":"We are looking for a software developer to build and maintain applications. The candidate will work with our development team.",
        "requirements":"Bachelor degree in computer science. Knowledge of Python, Java, programming and problem solving.",
        "benefits":"Competitive salary, professional training and career growth opportunities.",
        "telecommuting":"No","has_company_logo":"Yes","has_questions":"Yes"},
    "Python Developer": {
        "title":"Python Developer",
        "company_profile":"Our company provides software development and technology services.",
        "description":"We need a Python developer to develop applications, write clean code and work with databases.",
        "requirements":"Knowledge of Python, Django, Flask, SQL and problem solving skills.",
        "benefits":"Flexible working hours, training and professional development.",
        "telecommuting":"Yes","has_company_logo":"Yes","has_questions":"Yes"},
    "Data Analyst": {
        "title":"Data Analyst",
        "company_profile":"We are a data analytics company helping businesses make better decisions.",
        "description":"The candidate will analyze datasets, prepare reports and create visualizations.",
        "requirements":"Knowledge of Excel, Python, SQL, statistics and data visualization.",
        "benefits":"Competitive salary, learning opportunities and career development.",
        "telecommuting":"No","has_company_logo":"Yes","has_questions":"Yes"},
    "Web Developer": {
        "title":"Web Developer",
        "company_profile":"We create modern websites and digital solutions for clients.",
        "description":"We are hiring a web developer to design, develop and maintain websites.",
        "requirements":"Knowledge of HTML, CSS, JavaScript, PHP and responsive web design.",
        "benefits":"Paid training, supportive team and growth opportunities.",
        "telecommuting":"No","has_company_logo":"Yes","has_questions":"Yes"},
    "IT Support": {
        "title":"IT Support Executive",
        "company_profile":"We provide IT support and technical services to business customers.",
        "description":"The employee will help users solve technical problems and maintain computer systems.",
        "requirements":"Basic computer knowledge, troubleshooting and communication skills.",
        "benefits":"Fixed salary, training and career growth.",
        "telecommuting":"No","has_company_logo":"Yes","has_questions":"Yes"},
    "Marketing Assistant": {
        "title":"Marketing Assistant",
        "company_profile":"We are a growing marketing company working with different business clients.",
        "description":"Assist the marketing team with campaigns, research and promotional activities.",
        "requirements":"Communication skills, basic computer knowledge and interest in marketing.",
        "benefits":"Professional training and performance-based growth.",
        "telecommuting":"No","has_company_logo":"Yes","has_questions":"Yes"},
    "HR Executive": {
        "title":"HR Executive",
        "company_profile":"Our organization provides professional business services.",
        "description":"We are looking for an HR executive to support recruitment and employee management.",
        "requirements":"Knowledge of recruitment, communication and MS Office.",
        "benefits":"Fixed salary, professional environment and growth opportunities.",
        "telecommuting":"No","has_company_logo":"Yes","has_questions":"Yes"},
    "Sales Executive": {
        "title":"Sales Executive",
        "company_profile":"We are a company offering products and services to business customers.",
        "description":"The candidate will communicate with customers and support sales activities.",
        "requirements":"Good communication, customer handling and negotiation skills.",
        "benefits":"Salary with incentives and career growth.",
        "telecommuting":"No","has_company_logo":"Yes","has_questions":"Yes"},
    "Accountant": {
        "title":"Accountant",
        "company_profile":"We provide accounting and financial services to clients.",
        "description":"Responsible for maintaining financial records, preparing reports and managing accounts.",
        "requirements":"Knowledge of accounting, Excel, bookkeeping and financial records.",
        "benefits":"Competitive salary and professional development.",
        "telecommuting":"No","has_company_logo":"Yes","has_questions":"Yes"},
    "Business Analyst": {
        "title":"Business Analyst",
        "company_profile":"We help organizations improve their business processes using data.",
        "description":"Analyze business requirements, prepare reports and work with project teams.",
        "requirements":"Analytical thinking, Excel, SQL and communication skills.",
        "benefits":"Learning opportunities and career advancement.",
        "telecommuting":"Yes","has_company_logo":"Yes","has_questions":"Yes"},
    "Content Writer": {
        "title":"Content Writer",
        "company_profile":"We are a digital media company creating online content.",
        "description":"Create articles, blogs and website content for different audiences.",
        "requirements":"Good writing skills, grammar, research and basic SEO knowledge.",
        "benefits":"Flexible schedule and professional growth.",
        "telecommuting":"Yes","has_company_logo":"Yes","has_questions":"Yes"},
    "Graphic Designer": {
        "title":"Graphic Designer",
        "company_profile":"Our creative agency provides branding and design services.",
        "description":"Create graphics, social media designs and marketing materials.",
        "requirements":"Knowledge of Photoshop, Illustrator and creative design.",
        "benefits":"Creative work environment and career opportunities.",
        "telecommuting":"Yes","has_company_logo":"Yes","has_questions":"Yes"},
    "Customer Support": {
        "title":"Customer Support Executive",
        "company_profile":"We provide customer service solutions to businesses.",
        "description":"Assist customers with questions, service requests and product information.",
        "requirements":"Communication skills, patience and basic computer knowledge.",
        "benefits":"Training, fixed salary and career growth.",
        "telecommuting":"No","has_company_logo":"Yes","has_questions":"Yes"},
    "Digital Marketing": {
        "title":"Digital Marketing Executive",
        "company_profile":"We help brands grow through online marketing strategies.",
        "description":"Manage social media, online campaigns and digital marketing activities.",
        "requirements":"Knowledge of SEO, social media, content marketing and analytics.",
        "benefits":"Learning opportunities and performance incentives.",
        "telecommuting":"Yes","has_company_logo":"Yes","has_questions":"Yes"},
    "Data Entry Work From Home": {
        "title":"Data Entry Work From Home",
        "company_profile":"We are seeking support for data processing and record management.",
        "description":"Enter and organize information in company databases while following data quality guidelines.",
        "requirements":"Basic computer knowledge, typing skills and attention to detail.",
        "benefits":"Flexible working hours and remote work opportunity.",
        "telecommuting":"Yes","has_company_logo":"No","has_questions":"No"}
}

# -------------------- STATE --------------------
defaults = {
    "title":"", "company_profile":"", "description":"", "requirements":"",
    "benefits":"", "telecommuting":"No", "has_company_logo":"No",
    "has_questions":"No", "company_cin":"", "company_email":"",
    "page":"input", "prediction":None, "sample_choice":"Choose a sample..."
}
for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value

def load_sample():
    chosen = st.session_state.sample_choice
    if chosen in samples:
        for key, value in samples[chosen].items():
            st.session_state[key] = value

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
    final_data = hstack([text_data, csr_matrix(numeric_data)]).tocsr()

    if final_data.shape[1] != model.coef_.shape[1]:
        raise ValueError("Model feature count does not match the app inputs. Check the training code and saved model.")

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

def footer():
    st.markdown("---")
    st.markdown("""
    <p class="small-note">🌈 FAKE JOB DETECTION AI<br>
    Built with Python • Streamlit • TF-IDF • Logistic Regression<br>
    ✨ Machine Learning Project</p>
    """, unsafe_allow_html=True)

# -------------------- PAGE 1: INPUT --------------------
if st.session_state.page == "input":
    st.markdown("""
    <div class="hero">
      <div class="eyebrow">🌌 AURORA AI • RECRUITMENT SAFETY • STEP 1 OF 2</div>
      <h1>🌈 Fake Job Detection AI</h1>
      <p>Check a job posting with an AI-powered text analysis model.</p>
      <span class="pill">✨ TF-IDF</span> <span class="pill">🧠 Logistic Regression</span> <span class="pill">🛡️ Job Safety</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="glass-panel">
      <div class="eyebrow">⚡ QUICK START</div>
      <div class="section-heading">Try a suggested job</div>
      <div class="section-subtitle">Choose a sample profile to fill the form automatically, or enter your own job details.</div>
    </div>
    """, unsafe_allow_html=True)
    st.selectbox("Choose a sample job profile", ["Choose a sample..."] + list(samples.keys()),
                 key="sample_choice", on_change=load_sample)

    st.markdown("""
    <div class="glass-panel">
      <div class="eyebrow">📝 JOB INFORMATION</div>
      <div class="section-heading">Enter job details</div>
      <div class="section-subtitle">Complete the required fields marked with an asterisk.</div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2, gap="large")
    with col1:
        st.text_input("💼 Job Title *", key="title", placeholder="e.g. Software Developer")
        st.text_area("🏢 Company Profile", key="company_profile", height=125, placeholder="Describe the company and its work...")
        st.text_area("📄 Job Description *", key="description", height=180, placeholder="Paste the complete job description...")
    with col2:
        st.text_area("🎯 Requirements", key="requirements", height=125, placeholder="Skills, education and experience required...")
        st.text_area("🎁 Benefits", key="benefits", height=125, placeholder="Salary, benefits and other facilities...")
        st.markdown("#### ⚙️ Additional job information")
        st.selectbox("🏠 Does the company allow telecommuting?", ["No", "Yes"], key="telecommuting")
        st.selectbox("🖼️ Does the company have a logo?", ["No", "Yes"], key="has_company_logo")
        st.selectbox("❓ Does the job posting have screening questions?", ["No", "Yes"], key="has_questions")

    with st.expander("🏢 Add optional company details", expanded=False):
        c1, c2 = st.columns(2)
        with c1:
            st.text_input("Company Registration Number (CIN)", key="company_cin", placeholder="Enter 21-character CIN")
        with c2:
            st.text_input("Company Official Email", key="company_email", placeholder="hr@companyname.com")
        st.caption("These details are checked for format only. They are not included in the machine learning prediction.")

    st.markdown("""
    <div class="glass-panel" style="text-align:center">
      <div class="eyebrow">🔮 READY FOR ANALYSIS?</div>
      <div class="section-heading">Let's check this job</div>
      <div class="section-subtitle">Your prediction will open on a separate result screen.</div>
    </div>
    """, unsafe_allow_html=True)

    if st.button("🚀  Analyse Job Posting", use_container_width=True):
        if not st.session_state.title.strip() or not st.session_state.description.strip():
            st.warning("⚠️ Please enter at least the Job Title and Job Description.")
        else:
            try:
                with st.spinner("🔮 Analysing job posting..."):
                    run_prediction()
                st.rerun()
            except Exception as e:
                st.error("Something went wrong during prediction.")
                st.exception(e)

    footer()

# -------------------- PAGE 2: RESULTS --------------------
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

    st.markdown("""
    <div class="hero">
      <div class="eyebrow">✨ ANALYSIS COMPLETE • STEP 2 OF 2</div>
      <h1>🔮 Your Job Analysis</h1>
      <p>Your AI report is ready. Review the prediction and safety checks below.</p>
    </div>
    """, unsafe_allow_html=True)

    if st.button("⬅️  Back to Edit Details", use_container_width=False):
        st.session_state.page = "input"
        st.rerun()

    result_col, real_col, fake_col = st.columns([2, 1, 1], gap="medium")
    if predicted_class == 0:
        result_text = "🚨 FAKE JOB"
        result_emoji = "🛑"
        result_description = "The model classified this posting as potentially fraudulent. Verify the employer carefully."
        result_class = "fake-result"
        card_class = "fake"
    else:
        result_text = "🛡️ REAL JOB"
        result_emoji = "🌿"
        result_description = "The model classified this posting as likely genuine. Still verify the employer independently."
        result_class = "real-result"
        card_class = "real"

    with result_col:
        st.markdown(f"""
        <div class="result-card {card_class}">
          <div class="eyebrow">🎯 JOB AUTHENTICITY RESULT</div>
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

    st.markdown("### 🧾 Company details check")
    st.caption("These checks validate the entered formats only. They do not confirm company registration or email ownership.")
    cin_value = st.session_state.company_cin.strip().upper()
    email_value = st.session_state.company_email.strip()
    company_col1, company_col2 = st.columns(2, gap="large")

    with company_col1:
        with st.container(border=True):
            st.markdown("#### 🏢 Company CIN")
            if not cin_value:
                st.info("ℹ️ CIN not provided.")
            elif re.fullmatch(r"[LU]\\d{5}[A-Z]{2}\\d{4}[A-Z]{3}\\d{6}", cin_value):
                st.success("✅ CIN format looks valid (21 characters).")
                st.caption("Format check only. MCA registration has not been verified.")
            else:
                st.error("❌ CIN format is invalid. An Indian CIN normally contains 21 characters.")

    with company_col2:
        with st.container(border=True):
            st.markdown("#### 📧 Company email")
            if not email_value:
                st.info("ℹ️ Company email not provided.")
            elif re.fullmatch(r"[^\\s@]+@[^\\s@]+\\.[^\\s@]+", email_value):
                st.success("✅ Email address format looks valid.")
                email_domain = email_value.rsplit("@", 1)[-1].lower()
                free_domains = {"gmail.com", "yahoo.com", "yahoo.in", "outlook.com",
                                "hotmail.com", "live.com", "icloud.com", "rediffmail.com"}
                if email_domain in free_domains:
                    st.warning("⚠️ This uses a common free email provider. It is not automatically fake, but independently verify the employer.")
                else:
                    st.caption("Company-style domain entered; domain ownership and mailbox existence are not verified.")
            else:
                st.error("❌ Email format is invalid. Enter an address like hr@company.com.")

    with st.expander("🧠 Explore AI prediction explanation", expanded=False):
        st.write("🔍 These words and job details influenced the Logistic Regression model's prediction.")
        feature_names = vectorizer.get_feature_names_out()
        coefficients = model.coef_[0]
        text_coefficients = coefficients[:len(feature_names)]
        numeric_coefficients = coefficients[len(feature_names):]
        text_values = text_data.toarray()[0]
        word_contributions = text_values * text_coefficients
        factors = []

        for i, contribution in enumerate(word_contributions):
            if abs(contribution) > 0.000001:
                factors.append({"factor": feature_names[i], "contribution": float(contribution)})

        numeric_names = ["Telecommuting", "Company Logo", "Screening Questions"]
        for i in range(min(3, len(numeric_coefficients))):
            contribution = numeric_data[0][i] * numeric_coefficients[i]
            if abs(contribution) > 0.000001:
                value_label = "Yes" if numeric_data[0][i] == 1 else "No"
                factors.append({"factor": numeric_names[i] + " = " + value_label,
                                "contribution": float(contribution)})

        factors.sort(key=lambda x: abs(x["contribution"]), reverse=True)
        if factors:
            left_col, right_col = st.columns(2)
            for index, factor in enumerate(factors[:8]):
                positive = factor["contribution"] > 0
                direction = "Pushes toward REAL" if positive else "Pushes toward FAKE"
                color = "#9af4c9" if positive else "#ffafc0"
                symbol = "🟢" if positive else "🔴"
                target = left_col if index % 2 == 0 else right_col
                with target:
                    st.markdown(f"""
                    <div class="factor-card">
                      <b style="color:#f7f7ff">{symbol} {factor['factor']}</b>
                      <p style="color:{color};margin:6px 0">{direction}</p>
                      <small style="color:#c3c9df">Contribution: {factor['contribution']:.5f}</small>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.info("ℹ️ The model did not find individual feature contributions large enough to display.")
        st.caption("Positive contribution pushes toward Real (1). Negative contribution pushes toward Fake (0). These are model associations, not proof of authenticity.")

    st.warning("⚠️ This is a machine learning prediction, not a verified investigation. Always check the company's official website, contact details and job offer before sharing personal information or paying any fees.")
    footer()
