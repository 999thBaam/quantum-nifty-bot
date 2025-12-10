
import joblib
import sys

try:
    model = joblib.load('quantum_ensemble_model.pkl')
    scaler = joblib.load('scaler.pkl')
    
    print(f"Model Type: {type(model)}")
    print(f"Scaler Type: {type(scaler)}")
    
    if hasattr(model, 'feature_names_in_'):
        print(f"Model Features: {model.feature_names_in_}")
    elif hasattr(model, 'n_features_in_'):
        print(f"Model Feature Count: {model.n_features_in_}")
    else:
        print("Model features unknown")
        
    if hasattr(scaler, 'feature_names_in_'):
        print(f"Scaler Features: {scaler.feature_names_in_}")
    elif hasattr(scaler, 'n_features_in_'):
        print(f"Scaler Feature Count: {scaler.n_features_in_}")
        
except Exception as e:
    print(f"Error: {e}")
