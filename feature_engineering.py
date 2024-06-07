import pandas as pd

def create_features(data):
    # Example feature engineering steps
    data['service_intensity'] = data['historical_service_count'] / (data['spn'] + 1)
    data['fault_signal_strength_squared'] = data['fault_signal_strength'] ** 2
    
    # Add more feature engineering steps as needed
    
    return data

def main():
    # Load raw data
    data = pd.read_csv('data/raw/fault_codes_data.csv')
    
    # Apply feature engineering
    data = create_features(data)
    
    # Save processed data
    data.to_csv('data/processed/processed_data.csv', index=False)

if __name__ == '__main__':
    main()
