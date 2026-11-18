
import streamlit as st
import pandas as pd
import joblib
from scipy.sparse import hstack, csr_matrix
import numpy as np

# --------------------------------
# 1. Load Model and Vectorizer
# --------------------------------

model = joblib.load("fake_job_model.pkl")
vectorizer = joblib.load("fake_job_vectorizer.pkl")

# --------------------------------
# 2. Page Configuration
# --------------------------------

st.set_page_config(
    page_title="Fake Job Detection AI",
    page_icon="🔍",
    layout="wide"
)

# --------------------------------
# 3. AI Gradient Theme
# --------------------------------

st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #10172f, #202b50, #39245c);
    color: white;
}
h1, h2, h3, p, label {
    color: white !important;
}
.stButton button {
    background: linear-gradient(90deg, #00c6ff, #7b2ff7);
    color: white;
    border-radius: 12px;
    border: none;
    height: 48px;
    font-weight: bold;
}
.stButton button:hover {
    border: 1px solid white;
}
div[data-testid="stMetric"] {
    background: rgba(255,255,255,0.08);
    padding: 15px;
    border-radius: 12px;
}
</style>
""", unsafe_allow_html=True)

# --------------------------------
# 4. Sample Job Details
# --------------------------------

samples = {
    "Marketing Assistant": {
        "title": "Marketing Assistant",
        "company_profile": "We are an established marketing company working with different businesses.",
        "description": "We are hiring a marketing assistant to support our marketing team, prepare reports and manage social media campaigns.",
        "requirements": "Good communication skills, basic computer knowledge and interest in marketing.",
        "benefits": "Professional training, career development and standard company benefits.",
        "telecommuting": 0,
        "has_company_logo": 1,
        "has_questions": 1
    },

    "Software Developer": {
        "title": "Software Developer",
        "company_profile": "Our software company develops web applications and business technology solutions.",
        "description": "We are looking for a software developer to develop and maintain applications using Python and web technologies.",
        "requirements": "Knowledge of Python, HTML, CSS, JavaScript and problem-solving skills.",
        "benefits": "Learning opportunities, team collaboration and professional growth.",
        "telecommuting": 1,
        "has_company_logo": 1,
        "has_questions": 1
    },

    "Data Entry Work From Home": {
        "title": "Work From Home Data Entry",
        "company_profile": "",
        "description": "Earn a very high income from home with no experience. Get started immediately with guaranteed earnings.",
        "requirements": "No experience or special skills required. Immediate joining.",
        "benefits": "Very high earnings and flexible working hours.",
        "telecommuting": 1,
        "has_company_logo": 0,
        "has_questions": 0
    }
    
samples = {
    "Software Developer": {
        "title": "Software Developer",
        "company_profile": "Technology company developing software applications.",
        "description": "Develop and maintain applications using Python and web technologies.",
        "requirements": "Python, HTML, CSS, JavaScript and problem solving.",
        "benefits": "Training, career growth and standard company benefits.",
        "telecommuting": 1,
        "has_company_logo": 1,
        "has_questions": 1
    },

    "Python Developer": {
        "title": "Python Developer",
        "company_profile": "Software development company.",
        "description": "Hiring Python developer to build applications and work with databases.",
        "requirements": "Python, Django, SQL and basic programming knowledge.",
        "benefits": "Professional training and flexible working hours.",
        "telecommuting": 0,
        "has_company_logo": 1,
        "has_questions": 1
    },

    "Data Analyst": {
        "title": "Data Analyst",
        "company_profile": "Business analytics and consulting company.",
        "description": "Analyze datasets and prepare reports using Python and Excel.",
        "requirements": "Python, SQL, Excel and data analysis skills.",
        "benefits": "Learning opportunities and career development.",
        "telecommuting": 1,
        "has_company_logo": 1,
        "has_questions": 1
    },

    "Web Developer": {
        "title": "Web Developer",
        "company_profile": "Web development agency creating business websites.",
        "description": "Develop responsive websites and maintain web applications.",
        "requirements": "HTML, CSS, JavaScript and PHP.",
        "benefits": "Training and professional development.",
        "telecommuting": 0,
        "has_company_logo": 1,
        "has_questions": 1
    },

    "IT Support": {
        "title": "IT Support Executive",
        "company_profile": "IT services and technical support company.",
        "description": "Provide technical assistance and troubleshoot computer problems.",
        "requirements": "Computer knowledge and communication skills.",
        "benefits": "Paid training and employee benefits.",
        "telecommuting": 0,
        "has_company_logo": 1,
        "has_questions": 1
    },

    "Marketing Assistant": {
        "title": "Marketing Assistant",
        "company_profile": "Established marketing company.",
        "description": "Support marketing campaigns and manage social media.",
        "requirements": "Communication and marketing skills.",
        "benefits": "Training and career development.",
        "telecommuting": 0,
        "has_company_logo": 1,
        "has_questions": 1
    },

    "HR Executive": {
        "title": "HR Executive",
        "company_profile": "Human resources consulting company.",
        "description": "Manage recruitment, employee records and interviews.",
        "requirements": "HR knowledge and communication skills.",
        "benefits": "Professional development and standard benefits.",
        "telecommuting": 0,
        "has_company_logo": 1,
        "has_questions": 1
    },

    "Sales Executive": {
        "title": "Sales Executive",
        "company_profile": "Business sales and distribution company.",
        "description": "Manage customer relationships and achieve sales targets.",
        "requirements": "Communication and customer management skills.",
        "benefits": "Salary and performance incentives.",
        "telecommuting": 0,
        "has_company_logo": 1,
        "has_questions": 1
    },

    "Accountant": {
        "title": "Accountant",
        "company_profile": "Financial and accounting services company.",
        "description": "Maintain financial records and prepare accounting reports.",
        "requirements": "Accounting knowledge, Excel and financial reporting.",
        "benefits": "Standard employee benefits and career growth.",
        "telecommuting": 0,
        "has_company_logo": 1,
        "has_questions": 1
    },

    "Business Analyst": {
        "title": "Business Analyst",
        "company_profile": "Business consulting organization.",
        "description": "Analyze business requirements and prepare reports.",
        "requirements": "Analytical thinking, Excel and communication skills.",
        "benefits": "Training and professional development.",
        "telecommuting": 1,
        "has_company_logo": 1,
        "has_questions": 1
    },

    "Content Writer": {
        "title": "Content Writer",
        "company_profile": "Digital media and content company.",
        "description": "Create articles, blogs and website content.",
        "requirements": "Writing, grammar and research skills.",
        "benefits": "Flexible working hours and learning opportunities.",
        "telecommuting": 1,
        "has_company_logo": 1,
        "has_questions": 1
    },

    "Graphic Designer": {
        "title": "Graphic Designer",
        "company_profile": "Creative design and advertising agency.",
        "description": "Create graphics and visual designs for digital campaigns.",
        "requirements": "Photoshop, Illustrator and creative skills.",
        "benefits": "Creative work environment and career growth.",
        "telecommuting": 1,
        "has_company_logo": 1,
        "has_questions": 1
    },

    "Customer Support": {
        "title": "Customer Support Executive",
        "company_profile": "Customer service organization.",
        "description": "Help customers resolve queries through calls and emails.",
        "requirements": "Communication and problem-solving skills.",
        "benefits": "Training and standard company benefits.",
        "telecommuting": 0,
        "has_company_logo": 1,
        "has_questions": 1
    },

    "Digital Marketing": {
        "title": "Digital Marketing Executive",
        "company_profile": "Digital marketing agency.",
        "description": "Manage online campaigns and social media accounts.",
        "requirements": "SEO, social media and digital marketing knowledge.",
        "benefits": "Training and professional growth.",
        "telecommuting": 1,
        "has_company_logo": 1,
        "has_questions": 1
    },

    "Data Entry Work From Home": {
        "title": "Work From Home Data Entry",
        "company_profile": "",
        "description": "Earn extremely high income with no experience. Guaranteed earnings and immediate joining.",
        "requirements": "No experience or special skills required.",
        "benefits": "Very high earnings and flexible working hours.",
        "telecommuting": 1,
        "has_company_logo": 0,
        "has_questions": 0
    }
}
}

# --------------------------------
# 5. Load Selected Sample
# --------------------------------

def load_sample():
    selected = st.session_state.sample_choice

    if selected in samples:
        sample = samples[selected]

        for key in sample:
            st.session_state[key] = sample[key]


# --------------------------------
# 6. Application Heading
# --------------------------------

st.title("🔍 Fake Job Detection System")

st.write(
    "AI-powered application to identify potentially "
    "fraudulent job postings using Machine Learning."
)

st.divider()

st.subheader("📝 Enter Job Details")

st.info(
    "Choose a sample job below to automatically fill "
    "the form. You can edit every field before prediction."
)

# --------------------------------
# 7. Sample Selection
# --------------------------------

st.selectbox(
    "Choose Suggested Job",
    ["Choose a sample..."] + list(samples.keys()),
    key="sample_choice",
    on_change=load_sample
)

# --------------------------------
# 8. Job Input Fields
# --------------------------------

col1, col2 = st.columns(2)

with col1:
    st.text_input(
        "Job Title",
        key="title",
        placeholder="Example: Python Developer"
    )

    st.text_area(
        "Company Profile",
        key="company_profile",
        placeholder="Enter company information..."
    )

    st.text_area(
        "Job Description",
        key="description",
        placeholder="Enter complete job description..."
    )

with col2:
    st.text_area(
        "Job Requirements",
        key="requirements",
        placeholder="Enter required skills and qualifications..."
    )

    st.text_area(
        "Job Benefits",
        key="benefits",
        placeholder="Enter salary and other benefits..."
    )

# --------------------------------
# 9. Additional Job Information
# --------------------------------

st.subheader("🏢 Additional Job Information")

col1, col2, col3 = st.columns(3)

with col1:
    telecommuting = st.selectbox(
        "Remote Work",
        ["No", "Yes"],
        key="telecommuting"
    )

with col2:
    company_logo = st.selectbox(
        "Company Logo",
        ["No", "Yes"],
        key="has_company_logo"
    )

with col3:
    questions = st.selectbox(
        "Screening Questions",
        ["No", "Yes"],
        key="has_questions"
    )

st.caption(
    "These indicators provide context but do not prove "
    "whether a job is genuine or fraudulent."
)

# --------------------------------
# 10. Prediction Button
# --------------------------------

st.divider()

if st.button("🔍 Detect Job", use_container_width=True):

    if not st.session_state.title.strip():
        st.warning("Please enter a Job Title.")

    elif not st.session_state.description.strip():
        st.warning("Please enter a Job Description.")

    else:

        # Prepare text input
        new_job = pd.DataFrame({
            "title": [st.session_state.title],
            "company_profile": [st.session_state.company_profile],
            "description": [st.session_state.description],
            "requirements": [st.session_state.requirements],
            "benefits": [st.session_state.benefits]
        })

        text_columns = [
            "title",
            "company_profile",
            "description",
            "requirements",
            "benefits"
        ]

        new_job["combined_text"] = new_job[
            text_columns
        ].agg(" ".join, axis=1)

        # TF-IDF conversion
        text_data = vectorizer.transform(
            new_job["combined_text"]
        )

        # Numeric features in training order
        numeric_data = csr_matrix([[
            1 if telecommuting == "Yes" else 0,
            1 if company_logo == "Yes" else 0,
            1 if questions == "Yes" else 0
        ]])

        # Combine features
        final_data = hstack([
            text_data,
            numeric_data
        ])

        # Prediction
        prediction = model.predict(final_data)[0]

        probability = model.predict_proba(
            final_data
        )[0]

        # --------------------------------
        # 11. Prediction Result
        # --------------------------------

        st.divider()
        st.subheader("📊 Prediction Result")

        col1, col2 = st.columns(2)

        if prediction == 1:

            with col1:
                st.error("⚠️ FAKE JOB")

            with col2:
                st.metric(
                    "Fake Probability",
                    f"{probability[1] * 100:.2f}%"
                )

        else:

            with col1:
                st.success("✅ REAL JOB")

            with col2:
                st.metric(
                    "Real Probability",
                    f"{probability[0] * 100:.2f}%"
                )

        # --------------------------------
        # 12. Model Explanation
        # --------------------------------

        st.subheader("💡 Why did the model predict this?")

        word_names = vectorizer.get_feature_names_out()

        word_weights = model.coef_[0][
            :len(word_names)
        ]

        word_values = text_data.toarray()[0]

        contributions = word_values * word_weights

        fake_words = np.argsort(contributions)[::-1][:5]
        real_words = np.argsort(contributions)[:5]

        if prediction == 1:

            st.write(
                "These words had the strongest positive "
                "influence toward the Fake category:"
            )

            reasons = [
                word_names[i]
                for i in fake_words
                if contributions[i] > 0
            ]

        else:

            st.write(
                "These words had the strongest negative "
                "influence toward the Fake category, "
                "pushing the result toward Real:"
            )

            reasons = [
                word_names[i]
                for i in real_words
                if contributions[i] < 0
            ]

        if reasons:
            for word in reasons:
                st.write("•", word)
        else:
            st.write(
                "No strong individual text-word signals found."
            )

        st.warning(
            "This explanation shows learned text patterns, "
            "not verified facts about the employer. "
            "Always verify the company independently."
        )

# --------------------------------
# 13. Footer
# --------------------------------

st.divider()

st.caption(
    "Fake Job Detection AI | Machine Learning Project"
)


