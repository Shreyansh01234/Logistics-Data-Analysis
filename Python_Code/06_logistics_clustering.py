import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Load dataset
file_path = r"C:\Users\shrey\Downloads\archive (4)\dynamic_supply_chain_logistics_dataset.csv"

df = pd.read_csv(file_path)

# Select operational features
features = [
    "shipping_costs",
    "lead_time_days",
    "historical_demand",
    "traffic_congestion_level",
    "route_risk_level",
    "delay_probability"
]

X = df[features]

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# K-Means clustering
kmeans = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10
)

df["logistics_cluster"] = kmeans.fit_predict(X_scaled)

print("===== CLUSTER DISTRIBUTION =====")
print(df["logistics_cluster"].value_counts().sort_index())

print("\n===== CLUSTER PROFILE =====")

cluster_profile = df.groupby("logistics_cluster")[features].mean()

print(cluster_profile)