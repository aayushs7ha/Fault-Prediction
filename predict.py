import pandas as pd
import joblib
from feature_engineering import create_features

# Load the trained model and scaler
model = joblib.load('models/fault_model.pkl')
scaler = joblib.load('models/scaler.pkl')

# Example input for prediction
input_data = {
    'driving_pattern': ['urban'],
    'historical_service_count': [5],
    'fault_signal_strength': [0.8],
    'fmi': [3],
    'spn': [2000]
}

# Create DataFrame
input_df = pd.DataFrame(input_data)

# Apply feature engineering
input_df = create_features(input_df)

# Preprocess input data
input_df = pd.get_dummies(input_df)
input_df = input_df.reindex(columns=scaler.feature_names_in_, fill_value=0)
input_df = scaler.transform(input_df)

# Predict
prediction = model.predict(input_df)
print('Prediction:', prediction)