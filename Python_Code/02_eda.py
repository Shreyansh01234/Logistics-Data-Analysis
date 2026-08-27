import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
file_path = r"C:\Users\shrey\Downloads\archive (4)\dynamic_supply_chain_logistics_dataset.csv"

df = pd.read_csv(file_path)

# Convert timestamp to datetime
df["timestamp"] = pd.to_datetime(df["timestamp"])

print("===== DATASET INFORMATION =====")
print(df.info())

print("\n===== RISK CLASSIFICATION COUNTS =====")
print(df["risk_classification"].value_counts())

print("\n===== RISK CLASSIFICATION PERCENTAGE =====")
print(df["risk_classification"].value_counts(normalize=True) * 100)

print("\n===== AVERAGE DELIVERY TIME DEVIATION =====")
print(df["delivery_time_deviation"].mean())

print("\n===== AVERAGE ETA VARIATION =====")
print(df["eta_variation_hours"].mean())

print("\n===== AVERAGE SHIPPING COST =====")
print(df["shipping_costs"].mean())

print("\n===== AVERAGE LEAD TIME =====")
print(df["lead_time_days"].mean())

# -------------------------------------------------
# Correlation with delivery time deviation
# -------------------------------------------------

numeric_df = df.select_dtypes(include="number")

correlation = numeric_df.corr()["delivery_time_deviation"].sort_values(
    ascending=False
)

print("\n===== CORRELATION WITH DELIVERY TIME DEVIATION =====")
print(correlation)

# -------------------------------------------------
# Average delay by traffic congestion
# -------------------------------------------------

traffic_delay = df.groupby(
    "traffic_congestion_level"
)["delivery_time_deviation"].mean()

print("\n===== AVERAGE DELIVERY DEVIATION BY TRAFFIC =====")
print(traffic_delay)

# -------------------------------------------------
# Average delay by route risk
# -------------------------------------------------

route_delay = df.groupby(
    "route_risk_level"
)["delivery_time_deviation"].mean()

print("\n===== AVERAGE DELIVERY DEVIATION BY ROUTE RISK =====")
print(route_delay)

# -------------------------------------------------
# Average delay by weather severity
# -------------------------------------------------

weather_delay = df.groupby(
    "weather_condition_severity"
)["delivery_time_deviation"].mean()

print("\n===== AVERAGE DELIVERY DEVIATION BY WEATHER =====")
print(weather_delay)

# -------------------------------------------------
# Plot 1: Risk Classification
# -------------------------------------------------

df["risk_classification"].value_counts().plot(kind="bar")

plt.title("Distribution of Logistics Risk Classification")
plt.xlabel("Risk Classification")
plt.ylabel("Number of Records")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

# -------------------------------------------------
# Plot 2: Traffic vs Delivery Delay
# -------------------------------------------------

traffic_delay.plot(kind="line", marker="o")

plt.title("Traffic Congestion vs Delivery Time Deviation")
plt.xlabel("Traffic Congestion Level")
plt.ylabel("Average Delivery Time Deviation")
plt.tight_layout()
plt.show()

# -------------------------------------------------
# Plot 3: Route Risk vs Delivery Delay
# -------------------------------------------------

route_delay.plot(kind="line", marker="o")

plt.title("Route Risk Level vs Delivery Time Deviation")
plt.xlabel("Route Risk Level")
plt.ylabel("Average Delivery Time Deviation")
plt.tight_layout()
plt.show()