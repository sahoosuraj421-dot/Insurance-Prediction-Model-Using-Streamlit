import streamlit as st
import joblib
import pandas as pd
import numpy as np

scaler = joblib.load("standard_scaler.pkl")
le_gender = joblib.load("label_encoder_gender.pkl")
le_smoker = joblib.load("label_encoder_smoker.pkl")
le_diabetic = joblib.load("label_encoder_diabetic.pkl")
model = joblib.load("best_model.pkl")

# Page Configuration
st.set_page_config(
    page_title="Insurance Premium Prediction", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Beautiful Design
st.markdown("""
<style>
    /* Main background and colors */
    :root {
        --primary-color: #FF6B6B;
        --secondary-color: #4ECDC4;
        --accent-color: #FFE66D;
        --dark-bg: #1a1a2e;
        --light-bg: #f8f9fa;
        --text-dark: #2d3436;
    }
    
    /* Header styling */
    .main-header {
        background: linear-gradient(135deg, #FF6B6B 0%, #4ECDC4 100%);
        padding: 40px 20px;
        border-radius: 15px;
        text-align: center;
        margin-bottom: 30px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        color: white;
    }
    
    .main-header h1 {
        font-size: 2.5em;
        margin: 0;
        font-weight: 700;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
    }
    
    .main-header p {
        font-size: 1.1em;
        margin: 10px 0 0 0;
        opacity: 0.95;
    }
    
    /* Input form styling */
    .input-section {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 30px;
        border-radius: 15px;
        margin-bottom: 30px;
        box-shadow: 0 5px 20px rgba(0,0,0,0.1);
    }
    
    /* Input labels */
    .input-label {
        font-weight: 600;
        color: #2d3436;
        margin-bottom: 8px;
        font-size: 1.05em;
    }
    
    /* Submit button styling */
    .stButton > button {
        background: linear-gradient(135deg, #FF6B6B 0%, #FF8E8E 100%);
        color: white;
        border: none;
        padding: 15px 40px;
        border-radius: 25px;
        font-weight: 600;
        font-size: 1.1em;
        cursor: pointer;
        transition: all 0.3s ease;
        box-shadow: 0 5px 15px rgba(255,107,107,0.4);
        width: 100%;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(255,107,107,0.6);
    }
    
    /* Result styling */
    .success-box {
        background: linear-gradient(135deg, #00d2d3 0%, #00a8cc 100%);
        padding: 30px;
        border-radius: 15px;
        color: white;
        text-align: center;
        font-size: 1.5em;
        font-weight: 700;
        box-shadow: 0 10px 30px rgba(0,210,211,0.4);
        margin-top: 20px;
    }
    
    /* Info cards */
    .info-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 12px;
        color: white;
        margin: 15px 0;
        box-shadow: 0 5px 15px rgba(102,126,234,0.3);
    }
    
    /* Column styling */
    .input-column {
        background: white;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
    }
    
    /* Metric styling */
    .metric-badge {
        display: inline-block;
        background: #FFE66D;
        color: #2d3436;
        padding: 8px 16px;
        border-radius: 20px;
        font-weight: 600;
        margin: 5px 5px 5px 0;
        font-size: 0.9em;
    }
    
    /* Radio button styling */
    .radio-container {
        display: flex;
        gap: 15px;
        margin: 15px 0;
        flex-wrap: wrap;
    }
    
    .radio-option {
        flex: 1;
        min-width: 120px;
        padding: 15px 20px;
        border: 2px solid #e0e0e0;
        border-radius: 12px;
        cursor: pointer;
        transition: all 0.3s ease;
        text-align: center;
        background: white;
        font-weight: 600;
        font-size: 1em;
    }
    
    .radio-option:hover {
        border-color: #FF6B6B;
        background: #fff5f5;
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(255,107,107,0.2);
    }
    
    .radio-option-active {
        background: linear-gradient(135deg, #FF6B6B 0%, #FF8E8E 100%);
        color: white;
        border-color: #FF6B6B;
        box-shadow: 0 5px 15px rgba(255,107,107,0.4);
    }
</style>
""", unsafe_allow_html=True)

# Header Section
st.markdown("""
<div class="main-header">
    <h1>💳 Health Insurance Premium Prediction</h1>
    <p>Get an accurate estimate of your insurance premium in seconds</p>
</div>
""", unsafe_allow_html=True)

# Information Section
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("""
    <div class="info-card">
        <h3>🔒 Secure</h3>
        <p>Your data is processed safely and never stored.</p>
    </div>
    """, unsafe_allow_html=True)
    
with col2:
    st.markdown("""
    <div class="info-card">
        <h3>⚡ Instant</h3>
        <p>Get results in milliseconds.</p>
    </div>
    """, unsafe_allow_html=True)
    
with col3:
    st.markdown("""
    <div class="info-card">
        <h3>📊 Accurate</h3>
        <p>Based on advanced ML models.</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)


# Input Form Section
st.markdown("""
<div class="input-section">
    <h2 style="color: #2d3436; margin-top: 0;">📋 Your Health Information</h2>
    <p style="color: #636e72;">Fill in your details below for an accurate premium estimate</p>
</div>
""", unsafe_allow_html=True)

with st.form("input_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<p class="input-label">👤 Age</p>', unsafe_allow_html=True)
        age = st.number_input("Age", min_value=0, max_value=100, value=30, label_visibility="collapsed")
        
        st.markdown('<p class="input-label">📏 BMI (Body Mass Index)</p>', unsafe_allow_html=True)
        bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0, label_visibility="collapsed")
        
        st.markdown('<p class="input-label">👨‍👩‍👧‍👦 Number of Children</p>', unsafe_allow_html=True)
        children = st.number_input("Number of Children", min_value=0, max_value=10, value=0, label_visibility="collapsed")
    
    with col2:
        st.markdown('<p class="input-label">❤️ Blood Pressure (mmHg)</p>', unsafe_allow_html=True)
        bloodpressure = st.number_input("Blood Pressure", min_value=60, max_value=200, value=120, label_visibility="collapsed")
        
        st.markdown('<p class="input-label">⚧ Gender</p>', unsafe_allow_html=True)
        gender = st.radio("Gender", options=le_gender.classes_.tolist(), horizontal=True, label_visibility="collapsed")
        
        st.markdown('<p class="input-label">🚭 Smoker Status</p>', unsafe_allow_html=True)
        smoker = st.radio("Smoker", options=le_smoker.classes_.tolist(), horizontal=True, label_visibility="collapsed")
        
        st.markdown('<p class="input-label">🏥 Diabetic Status</p>', unsafe_allow_html=True)
        diabetic = st.selectbox("Diabetic", options=le_diabetic.classes_.tolist(), label_visibility="collapsed")

    # Centered submit button
    col_btn_left, col_btn_center, col_btn_right = st.columns([1, 2, 1])
    with col_btn_center:
        submitted = st.form_submit_button("🔮 Calculate My Insurance Premium")

if submitted:
    input_data = pd.DataFrame({
        "age": [age],
        "bmi": [bmi],
        "children": [children],
        # match notebook feature name
        "bloodpressure": [bloodpressure],
        "gender": [gender],
        "diabetic": [diabetic],
        "smoker": [smoker],
    })

    input_data["gender"] = le_gender.transform(input_data["gender"])
    input_data["smoker"] = le_smoker.transform(input_data["smoker"])
    input_data["diabetic"] = le_diabetic.transform(input_data["diabetic"])

    num_cols = ["age", "bmi", "bloodpressure", "children"]
    try:
        input_data[num_cols] = scaler.transform(input_data[num_cols])

        # Ensure column order matches training `x` used in the notebook
        training_cols = ["age", "gender", "bmi", "bloodpressure", "diabetic", "children", "smoker"]
        input_data = input_data[training_cols]

        # Predict and show result; wrap in try/except to display errors
        prediction = float(model.predict(input_data)[0])
        
        # Display result with custom styling
        st.markdown(f"""
        <div class="success-box">
            ✨ Your Estimated Annual Insurance Premium ✨
            <br><br>
            <span style="font-size: 2em; font-weight: 900;">${prediction:,.2f}</span>
        </div>
        """, unsafe_allow_html=True)
        
        # Display breakdown
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Age", f"{age} years", delta=None)
        with col2:
            st.metric("BMI", f"{bmi:.1f}", delta=None)
        with col3:
            st.metric("Blood Pressure", f"{bloodpressure} mmHg", delta=None)
        with col4:
            st.metric("Children", f"{children}", delta=None)
        
        # Additional info
        st.markdown("""
        <div class="info-card" style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); margin-top: 20px;">
            <h3>💡 Helpful Tips</h3>
            <ul>
                <li>Maintaining a healthy BMI can significantly reduce your premiums</li>
                <li>Non-smokers typically enjoy much lower insurance rates</li>
                <li>Regular health checkups help manage risk factors</li>
                <li>Good blood pressure control is essential for lower costs</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
    except Exception as e:
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #ff6b6b 0%, #ee5a6f 100%); padding: 20px; border-radius: 12px; color: white;">
            <h3>⚠️ Prediction Error</h3>
            <p>{str(e)}</p>
        </div>
        """, unsafe_allow_html=True)
        st.info("If you trained a PolynomialRegression model, save it together with its PolynomialFeatures transformer (or wrap both in a Pipeline) so the app can transform inputs before predicting.")
