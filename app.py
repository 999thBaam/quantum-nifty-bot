
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


model, scaler = load_model_scaler()


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
    
    # Fix for yfinance returning MultiIndex columns
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    
    if not data.empty:
        st.write("Recent Data:", data.tail())
        
        # Plot
        fig = go.Figure(data=[go.Candlestick(x=data.index,
                        open=data['Open'],
                        high=data['High'],
                        low=data['Low'],
                        close=data['Close'])])
        st.plotly_chart(fig, use_container_width=True)
        

        
        # Feature Engineering (Must match train_model.py EXACTLY)
        # 1. Daily Return
        data['Return'] = data['Close'].pct_change()
        # 2. Volatility (5-day rolling std dev of returns)
        data['Volatility'] = data['Return'].rolling(window=5).std()
        # 3. Momentum (Close - Close 5 days ago)
        data['Momentum'] = data['Close'] - data['Close'].shift(5)
        
        # Get the latest row
        latest_data = data.iloc[-1]
        
        # Check for NaNs (e.g. if not enough history)
        if pd.isna(latest_data['Return']) or pd.isna(latest_data['Volatility']) or pd.isna(latest_data['Momentum']):
            st.warning("Not enough data to calculate features. Fetching more history...")
        else:
            st.subheader("Live Prediction for Next Trading Day")
            
            # Prepare input
            inputs = np.array([[latest_data['Return'], latest_data['Volatility'], latest_data['Momentum']]])
            
            # Scale
            try:
                scaled_inputs = scaler.transform(inputs)
                
                # Dynamic Fix for QSVC missing predict_proba
                if hasattr(model, 'voting') and model.voting == 'soft':
                    # QSVC in 0.6.0 sometimes lacks predict_proba, breaking soft voting
                    model.voting = 'hard'
                
                # Predict
                prediction = model.predict(scaled_inputs)
                
                # Try getting probability if possible (unlikely with hard voting, but safe)
                probability = None
                try:
                    if hasattr(model, "predict_proba"):
                         probability = model.predict_proba(scaled_inputs)
                except:
                    pass
                
                # Display Config
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Model Input: Return", f"{latest_data['Return']:.2%}")
                    st.metric("Model Input: Volatility", f"{latest_data['Volatility']:.4f}")
                    st.metric("Model Input: Momentum", f"{latest_data['Momentum']:.2f}")

                with col2:
                    if prediction[0] == 1:
                        st.success("## 📈 BULLISH Signal")
                        st.write("Model predicts price INCREASE.")
                    else:
                        st.error("## 📉 BEARISH Signal")
                        st.write("Model predicts price DECREASE.")
                    
                    if probability is not None:
                        prob_val = np.max(probability)*100
                        st.progress(int(prob_val))
                        st.caption(f"Confidence: {prob_val:.2f}%")
                        
            except Exception as e:
                st.error(f"Prediction error: {e}")
        
    else:
        st.error("Failed to fetch data.")

st.markdown("---")
st.markdown("Built with Streamlit & Qiskit")
