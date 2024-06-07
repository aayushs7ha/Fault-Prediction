import pandas as pd
import numpy as np

# Generate synthetic data
np.random.seed(42)

data = pd.DataFrame({
    'driving_pattern': np.random.choice(['urban', 'highway', 'offroad'], size=1000),
    'historical_service_count': np.random.randint(0, 20, size=1000),
    'fault_signal_strength': np.random.uniform(0, 1, size=1000),
    'fmi': np.random.randint(0, 10, size=1000),
    'spn': np.random.randint(1000, 5000, size=1000),
    'fault': np.random.choice([0, 1], size=1000, p=[0.7, 0.3])
})

# Save to CSV
#data.to_csv('/Users/maverick/Fault Prediction/data/raw/fault_codes_data.csv', index=False)
