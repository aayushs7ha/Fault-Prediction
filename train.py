import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import joblib
from feature_engineering import create_features

# Load data
data = pd.read_csv('data/raw/fault_codes_data.csv')

# Data preprocessing
data.ffill(inplace=True)
data = pd.get_dummies(data, drop_first=True)

# Feature engineering
data = create_features(data)

# Feature selection
X = data.drop('fault', axis=1)
y = data['fault']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Model training
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Save the trained model and scaler
joblib.dump(model, 'models/fault_model.pkl')
joblib.dump(scaler, 'models/scaler.pkl')
