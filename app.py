
import streamlit as st
import numpy as np
import joblib

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
# 3. PURPLE GRADIENT THEME
# --------------------------------------------------

st.markdown("""
<style>

.stApp {
    background:
        radial-gradient(circle at top left, #4c1d95 0%, transparent 35%),
        radial-gradient(circle at bottom right, #701a75 0%, transparent 35%),
        linear-gradient(135deg, #0f1024, #17132f, #21113b);
    color: white;
}

header[data-testid="stHeader"] {
    background: transparent;
}

.block-container {
    max-width: 1200px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

h1, h2, h3, p, label {
    color: white !important;
}

.hero {
    background: linear-gradient(
        120deg,
        rgba(124, 58, 237, 0.4),
        rgba(192, 38, 211, 0.2)
    );
    border: 1px solid rgba(216, 180, 254, 0.3);
    border-radius: 22px;
    padding: 30px;
    text-align: center;
    margin-bottom: 25px;
}

.hero h1 {
    font-size: 38px;
    font-weight: 800;
    background: linear-gradient(90deg, white, #d8b4fe, #f0abfc);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.hero p {
    color: #e9d5ff !important;
}

.section-card {
    background: rgba(255, 255, 255, 0.07);
    border: 1px solid rgba(216, 180, 254, 0.2);
    border-radius: 18px;
