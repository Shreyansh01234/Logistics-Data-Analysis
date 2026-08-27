import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
file_path = r"C:\Users\shrey\Downloads\archive (4)\dynamic_supply_chain_logistics_dataset.csv"

df = pd.read_csv(file_path)

# Select numerical columns
numeric_df = df.select_dtypes(include="number")

# Correlation with delivery time deviation
correlation = (
    numeric_df.corr()["delivery_time_deviation"]
    .drop("delivery_time_deviation")
    .sort_values()
)

print("===== FACTORS ASSOCIATED WITH DELIVERY TIME DEVIATION =====")
print(correlation)

# Plot
correlation.plot(kind="barh", figsize=(10, 8))

plt.title("Correlation of Logistics Variables with Delivery Time Deviation")
plt.xlabel("Correlation")
plt.ylabel("Logistics Variables")
plt.tight_layout()

plt.savefig("../Report/correlation_with_delivery_delay.png", dpi=300)

plt.show()