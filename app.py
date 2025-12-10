
import streamlit as st
import joblib
import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go

# Page Configuration
st.set_page_config(
    page_title="Quantum Nifty Bot",
    page_icon="⚛️",
    layout="wide"
)

# Custom CSS for a "Nifty" look
st.markdown("""
<style>
    .stApp {
        background-color: #0e1117;
        color: #fafafa;
    }
    .stButton>button {
        background-color: #00d4ff;
        color: black;
        border-radius: 20px;
        border: none;
        padding: 10px 24px;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #00a0c0;
    }
</style>
""", unsafe_allow_html=True)

# Header
col1, col2 = st.columns([1, 4])
with col1:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/c/c3/Python_logo_notext.svg/1200px-Python_logo_notext.svg.png", width=60) # Placeholder or relevant icon
with col2:
    st.title("Quantum Nifty Bot ⚛️")
    st.caption("AI-Powered Hybrid Quantum-Classical Market Predictor")

st.divider()

# Load Model and Scaler
# --- PATCH: Fix for Legacy Qiskit Model Loading ---
import sys
import types
try:
    # ensuring qiskit is imported
    import qiskit.circuit.library.data_preparation
    
    # The old model looks for '_zz_feature_map', but newer qiskit has 'zz_feature_map' (no underscore)
    # We alias the new module to the old name in sys.modules so pickle finds it.
    from qiskit.circuit.library.data_preparation import zz_feature_map
    sys.modules['qiskit.circuit.library.data_preparation._zz_feature_map'] = zz_feature_map
except ImportError:
    print("Warning: Could not patch _zz_feature_map. Model loading might fail.")
except Exception as e:
    print(f"Warning: Error during patching: {e}")
# --------------------------------------------------

@st.cache_resource
def load_model_scaler():
    try:
        # Load the saved model and scaler
        model = joblib.load('quantum_ensemble_model.pkl')
        scaler = joblib.load('scaler.pkl')
        return model, scaler
    except Exception as e:
        st.error(f"Failed to load model/scaler: {e}")
        return None, None

model, scaler = load_models()

if not model or not scaler:
    st.stop()

# Sidebar for controls
st.sidebar.header("Configuration")
mode = st.sidebar.radio("Mode", ["Manual Input", "Live Market Data (NIFTY 50)"])

# Layout
if mode == "Manual Input":
    st.subheader("Manual Feature Input")
    st.info("Enter the 3 features expected by the model.")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        f1 = st.number_input("Feature 1", value=0.0, format="%.4f")
    with col2:
        f2 = st.number_input("Feature 2", value=0.0, format="%.4f")
    with col3:
        f3 = st.number_input("Feature 3", value=0.0, format="%.4f")
        
    if st.button("Generate Prediction"):
        try:
            inputs = np.array([[f1, f2, f3]])
            scaled_inputs = scaler.transform(inputs)
            prediction = model.predict(scaled_inputs)
            probability = model.predict_proba(scaled_inputs) if hasattr(model, "predict_proba") else None
            
            st.success(f"Prediction: **{prediction[0]}**")
            if probability is not None:
                st.write(f"Confidence: {np.max(probability)*100:.2f}%")
                
        except Exception as e:
            st.error(f"Prediction failed: {e}")

else:
    st.subheader("Live Market Data (Beta)")
    st.warning("Ensure the live data features match exactly what the model was trained on. Currently fetching generic NIFTY data.")
    
    ticker = "^NSEI"
    data = yf.download(ticker, period="1mo", interval="1d")
    
    if not data.empty:
        st.write("Recent Data:", data.tail())
        
        # Plot
        fig = go.Figure(data=[go.Candlestick(x=data.index,
                        open=data['Open'],
                        high=data['High'],
                        low=data['Low'],
                        close=data['Close'])])
        st.plotly_chart(fig, use_container_width=True)
        
        # Placeholder for feature extraction
        # We need to know EXACTLY what the 3 features are to automate this.
        # usually it's something like (Close-Open), (High-Low), etc.
        st.info("To predict on live data, we need to map the market data to the 3 required features. Update the code logic here once you verify feature engineering.")
        
    else:
        st.error("Failed to fetch data.")

st.markdown("---")
st.markdown("Built with Streamlit & Qiskit")
