
import pandas as pd
import joblib
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score

# Load processed data
X_scaled = joblib.load("processed_data.pkl")

# K-Means
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
kmeans.fit(X_scaled)

score1 = silhouette_score(X_scaled, kmeans.labels_)
joblib.dump(kmeans, "kmeans_model.pkl")

print("K-Means training completed!")
print("Silhouette Score:", round(score1, 3))

# Hierarchical Clustering
hierarchical = AgglomerativeClustering(n_clusters=3)
labels = hierarchical.fit_predict(X_scaled)

score2 = silhouette_score(X_scaled, labels)

print("\nHierarchical Clustering completed!")
print("Silhouette Score:", round(score2, 3))

# Save customer segments
df = pd.read_csv("cleaned_customers.csv")
df["KMeans_Cluster"] = kmeans.labels_
df["Hierarchical_Cluster"] = labels
df.to_csv("customer_segments.csv", index=False)

print("\nCustomer segments saved!")
print(df[["Age", "Annual_Income", "Spending_Score",
          "KMeans_Cluster", "Hierarchical_Cluster"]])
