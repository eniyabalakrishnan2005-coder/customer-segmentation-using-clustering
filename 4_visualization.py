
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("customer_segments.csv")

# K-Means graph
plt.figure(figsize=(8, 5))
sns.scatterplot(
    data=df,
    x="Annual_Income",
    y="Spending_Score",
    hue="KMeans_Cluster",
    palette="Set1",
    s=100
)
plt.title("Customer Segmentation - K-Means")
plt.tight_layout()
plt.show()

# Hierarchical graph
plt.figure(figsize=(8, 5))
sns.scatterplot(
    data=df,
    x="Annual_Income",
    y="Spending_Score",
    hue="Hierarchical_Cluster",
    palette="Set2",
    s=100
)
plt.title("Customer Segmentation - Hierarchical")
plt.tight_layout()
plt.show()
