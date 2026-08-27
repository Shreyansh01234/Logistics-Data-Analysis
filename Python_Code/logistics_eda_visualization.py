import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set visual style
sns.set_theme(style="whitegrid")

# 1. Generate Synthetic Logistics Data
np.random.seed(42)
n = 1000
df = pd.DataFrame({
    'distance_km': np.random.uniform(2.0, 50.0, n),
    'traffic_density_score': np.random.uniform(1.0, 10.0, n),
    'payload_weight_kg': np.random.uniform(1.0, 30.0, n),
    'weather_condition': np.random.choice(['Clear', 'Rain', 'Fog', 'Storm'], size=n, p=[0.55, 0.25, 0.12, 0.08])
})

# Calculate Dependent Target Metrics
weather_impact = {'Clear': 0, 'Rain': 8, 'Fog': 14, 'Storm': 25}
df['weather_delay'] = df['weather_condition'].map(weather_impact)
df['delivery_time_min'] = (df['distance_km'] * 2.1) + (df['traffic_density_score'] * 4.2) + (df['payload_weight_kg'] * 0.25) + df['weather_delay'] + np.random.normal(5, 3, n)
df['delivery_cost_usd'] = (df['distance_km'] * 1.4) + (df['payload_weight_kg'] * 0.9) + (df['traffic_density_score'] * 1.1) + np.random.normal(2, 1, n)

# 2. Plot 1: Correlation Matrix
plt.figure(figsize=(7, 5))
sns.heatmap(df[['distance_km', 'traffic_density_score', 'payload_weight_kg', 'delivery_time_min', 'delivery_cost_usd']].corr(), annot=True, cmap='Blues', fmt='.2f')
plt.title('Logistics Feature Correlation Matrix', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('correlation_matrix.png', dpi=300)

# 3. Plot 2: Weather Impact Boxplot
plt.figure(figsize=(7, 4.5))
sns.boxplot(x='weather_condition', y='delivery_time_min', data=df, palette='Set2')
plt.title('Delivery Duration Distribution by Weather Condition', fontsize=12, fontweight='bold')
plt.xlabel('Weather Condition')
plt.ylabel('Delivery Duration (Minutes)')
plt.tight_layout()
plt.savefig('weather_delivery_time.png', dpi=300)

# 4. Plot 3: Scatter Plot Cost vs Distance
plt.figure(figsize=(7, 4.5))
scatter = plt.scatter(df['distance_km'], df['delivery_cost_usd'], c=df['traffic_density_score'], cmap='viridis', alpha=0.7)
plt.colorbar(scatter, label='Traffic Density Score (1-10)')
plt.title('Shipment Distance vs Operational Cost', fontsize=12, fontweight='bold')
plt.xlabel('Distance (km)')
plt.ylabel('Delivery Cost (USD)')
plt.tight_layout()
plt.savefig('distance_vs_cost.png', dpi=300)

print("[SUCCESS] All Logistics EDA Visualizations Generated and Saved Successfully.")