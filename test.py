import pandas as pd
from sklearn.metrics import classification_report, roc_auc_score
import joblib
from sklearn.model_selection import train_test_split
from feature_engineering import create_features

# Load data
data = pd.read_csv('data/raw/fault_codes_data.csv')

# Data preprocessing
data.fillna(method='ffill', inplace=True)
data = pd.get_dummies(data, drop_first=True)

# Feature engineering
data = create_features(data)

# Feature selection
X = data.drop('fault', axis=1)
y = data['fault']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Load the trained model and scaler
model = joblib.load('models/fault_model.pkl')
scaler = joblib.load('models/scaler.pkl')

# Feature scaling
X_test = scaler.transform(X_test)

# Model evaluation
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
print('ROC AUC Score:', roc_auc_score(y_test, y_pred))