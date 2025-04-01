import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
import joblib

# Load dataset
df = pd.read_csv("data/cleaned_dataset.csv")
df.columns = df.columns.str.strip()

# Rename important columns
df = df.rename(columns={
    "Building Size - GIA (M2)": "propertySize",
    "Site Area (Hectares)": "siteArea",
    "Council Tax": "councilTax",
    "Property Type": "propertyType",
    "Base Price": "basePrice",
    "Effective Date": "effectiveDate",
    "Occupied by Council / Direct Service Property": "councilOwned",
    "Price Per Sq Meter": "pricePerSqMeter",
    "Property History Count": "historyCount"
})

# Drop rows missing base price
df = df.dropna(subset=["basePrice"])

# Fill missing numerics
numeric_cols = ["propertySize", "siteArea", "councilTax", "pricePerSqMeter", "historyCount"]
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

# Extract year and month
df["effectiveDate"] = pd.to_datetime(df["effectiveDate"], errors="coerce")
df["year"] = df["effectiveDate"].dt.year.fillna(df["Effective Year"]).astype("Int64")
df["month"] = df["effectiveDate"].dt.month.fillna(6).astype("Int64")

# Simulate history
df["historicalPrice"] = df["basePrice"] * np.random.uniform(0.9, 1.1, size=len(df))
df["historicalRent"] = df["basePrice"] * 0.005

# One-hot encode categoricals
df = pd.get_dummies(df, columns=["propertyType", "councilOwned"], drop_first=True)

# Feature list
features = [
    "historicalPrice", "historicalRent", "year", "month",
    "propertySize", "siteArea", "councilTax", "pricePerSqMeter", "historyCount"
] + [col for col in df.columns if col.startswith("propertyType_") or col.startswith("councilOwned_")]

X = df[features]
y = df["basePrice"]

# Standardize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train model
model = XGBRegressor(n_estimators=10, learning_rate=0.1, max_depth=3, random_state=42)
model.fit(X_scaled, y)

# Save assets
model.save_model("ml_model.json")
joblib.dump(scaler, "ml_model_scaler.pkl")
with open("ml_model_features.txt", "w") as f:
    for col in X.columns:
        f.write(f"{col}\n")

print("Model, scaler, and features saved.")
