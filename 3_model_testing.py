
import pandas as pd
import joblib

# Load trained model
model = joblib.load("kmeans_model.pkl")
scaler = joblib.load("scaler.pkl")

# New customer details
new_customer = pd.DataFrame({
    "Age": [27],
    "Annual_Income": [40],
    "Spending_Score": [75],
    "Purchase_Frequency": [10],
    "Total_Spending": [13000]
})

# Scale and predict
new_scaled = scaler.transform(new_customer)
prediction = model.predict(new_scaled)

print("New Customer Details:")
print(new_customer)

print("\nPredicted Customer Cluster:", prediction[0])
