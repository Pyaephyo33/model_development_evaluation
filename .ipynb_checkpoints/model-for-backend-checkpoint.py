import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
import os

# Load actual CSV file
df = pd.read_csv("data/cleaned_dataset.csv")  # Make sure filename matches

# Fix column names if needed
df.columns = df.columns.str.strip()  # remove accidental whitespaces

# Create expected features
df['historicalPrice'] = df['Base Price']
df['historicalRent'] = 0  # You don’t have rent, fill with dummy or estimate
df['Effective Date'] = pd.to_datetime(df['Effective Date'], errors='coerce')
df['year'] = df['Effective Date'].dt.year
df['month'] = df['Effective Date'].dt.month

# Drop rows where any feature or target is missing
features = ['historicalPrice', 'historicalRent', 'year', 'month']
target = 'Base Price'
df = df.dropna(subset=features + [target])

# Split data
X = df[features]
y = df[target]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
model.fit(X_train_scaled, y_train)

# Save model as JSON
model.save_model("ml_model.json")

# Save scaler
joblib.dump(scaler, "ml_model_scaler.pkl")

# Save feature list
with open("ml_model_features.txt", "w") as f:
    for col in X.columns:
        f.write(f"{col}\n")

print("✅ Model, scaler, and features saved successfully.")
