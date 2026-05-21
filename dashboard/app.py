import streamlit as st
import pickle
import pandas as pd
import numpy as np

# Model load karo directly
with open('src/model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('src/feature_names.pkl', 'rb') as f:
    feature_names = pickle.load(f)

st.set_page_config(
    page_title="Credit Risk Predictor",
    page_icon="🏦",
    layout="wide"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;600&display=swap');
    * { font-family: 'DM Sans', sans-serif; }
    h1 { 
        font-family: 'Space Mono', monospace !important;
        color: #00ff9d !important;
        font-size: 2.5rem !important;
    }
    .stButton>button {
        background: linear-gradient(135deg, #00ff9d, #00b4d8);
        color: #0a0f1e;
        font-weight: 700;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 4px;
        font-size: 1rem;
        width: 100%;
        font-family: 'Space Mono', monospace;
    }
    .metric-box {
        background: #111827;
        border: 1px solid #1f2937;
        border-radius: 8px;
        padding: 1.5rem;
        text-align: center;
    }
    .approve {
        background: #052e16;
        border: 2px solid #00ff9d;
        border-radius: 8px;
        padding: 2rem;
        text-align: center;
    }
    .reject {
        background: #2d0a0a;
        border: 2px solid #ff4444;
        border-radius: 8px;
        padding: 2rem;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("# 🏦 Credit Risk Predictor")
st.markdown("##### AI-powered loan default prediction for BFSI sector")
st.markdown("---")

col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("### Customer Profile")
    age = st.slider("Age (Years)", 18, 70, 35)
    years_employed = st.slider("Years Employed", 0.0, 40.0, 5.0)
    credit_term = st.slider("Credit Term (Months)", 8, 45, 24)

with col2:
    st.markdown("### Financial Profile")
    ext_source_2 = st.slider("Credit Score 2 (CIBIL-like)", 0.0, 1.0, 0.5)
    ext_source_3 = st.slider("Credit Score 3 (Bureau)", 0.0, 1.0, 0.5)
    debt_ratio = st.slider("Debt to Income Ratio", 0.0, 13.0, 3.0)

st.markdown("---")

if st.button("🔍 ANALYZE CREDIT RISK"):
    # Saare features ke saath dataframe banao
    input_df = pd.DataFrame([np.zeros(len(feature_names))], columns=feature_names)
    
    # Values fill karo
    input_df['EXT_SOURCE_2'] = ext_source_2
    input_df['EXT_SOURCE_3'] = ext_source_3
    input_df['AGE_YEARS'] = age
    input_df['YEARS_EMPLOYED'] = years_employed
    input_df['DEBT_INCOME_RATIO'] = debt_ratio
    input_df['CREDIT_TERM'] = credit_term
    
    prob = model.predict_proba(input_df)[:,1][0]
    decision = "REJECT" if prob >= 0.1 else "APPROVE"
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div class='metric-box'>
            <h3 style='color:#00b4d8'>Default Probability</h3>
            <h1 style='color:white'>{prob*100:.1f}%</h1>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class='metric-box'>
            <h3 style='color:#00b4d8'>Risk Threshold</h3>
            <h1 style='color:white'>10%</h1>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        if decision == "APPROVE":
            st.markdown(f"""
            <div class='approve'>
                <h2 style='color:#00ff9d'>✅ APPROVED</h2>
                <p style='color:#86efac'>Low default risk detected</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class='reject'>
                <h2 style='color:#ff4444'>❌ REJECTED</h2>
                <p style='color:#fca5a5'>High default risk detected</p>
            </div>
            """, unsafe_allow_html=True)