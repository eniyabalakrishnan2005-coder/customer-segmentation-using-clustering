
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

# Load dataset
df = pd.read_csv("customers.csv")

print("First 5 records:")
print(df.head())

print("\nMissing values:")
print(df.isnull().sum())

# Remove duplicates and handle missing values
df = df.drop_duplicates()
df = df.fillna(df.median(numeric_only=True))

# Select features
features = [
    "Age",
    "Annual_Income",
    "Spending_Score",
    "Purchase_Frequency",
    "Total_Spending"
]

X = df[features]

# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Save processed data
joblib.dump(X_scaled, "processed_data.pkl")
joblib.dump(scaler, "scaler.pkl")
df.to_csv("cleaned_customers.csv", index=False)

print("\nData processing completed!")
print("Total customers:", len(df))
