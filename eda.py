import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load data
data = pd.read_csv('data/raw/fault_codes_data.csv')

# Data preprocessing
data.ffill(inplace=True)

# One-hot encode categorical variables
data = pd.get_dummies(data, drop_first=True)

# Exploratory Data Analysis
# Visualize the distribution of the target variable
sns.countplot(x='fault', data=data)
plt.title('Fault Distribution')
plt.show()

# Correlation heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
plt.title('Feature Correlation')
plt.show()
