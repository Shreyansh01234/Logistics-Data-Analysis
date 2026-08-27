import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# ==============================================================================
# STEP 1: LOGISTICS DATA COLLECTION & SYNTHETIC SIMULATION
# ==============================================================================
def simulate_raw_logistics_data(n_samples=1000, seed=42):
    np.random.seed(seed)
    
    distances = np.random.uniform(2.0, 45.0, n_samples)
    traffic = np.random.uniform(1.0, 10.0, n_samples)
    payload = np.random.uniform(0.5, 25.0, n_samples)
    
    # Calculate delivery time with linear relations + noise
    delivery_time = (distances * 2.2) + (traffic * 4.5) + (payload * 0.3) + np.random.normal(5, 2, n_samples)
    delivery_cost = (distances * 1.5) + (payload * 0.8) + np.random.normal(2, 0.5, n_samples)
    
    df = pd.DataFrame({
        'order_id': [f"ORD_{i:04d}" for i in range(1, n_samples + 1)],
        'distance_km': distances,
        'traffic_density_score': traffic,
        'payload_weight_kg': payload,
        'delivery_time_min': delivery_time,
        'delivery_cost_usd': delivery_cost,
        'weather_condition': np.random.choice(['Clear', 'Rain', 'Fog', 'Storm'], size=n_samples, p=[0.6, 0.2, 0.1, 0.1])
    })
    
    # Introduce Artificial Data Imperfections (Missing Values & Outliers)
    df.loc[np.random.choice(df.index, 30), 'distance_km'] = np.nan
    df.loc[np.random.choice(df.index, 20), 'weather_condition'] = np.nan
    df.loc[np.random.choice(df.index, 10), 'delivery_time_min'] = 450.0  # Outliers
    
    return df

# Ingest data
raw_df = simulate_raw_logistics_data()
print(f"[INFO] Raw Dataset Loaded: {raw_df.shape[0]} rows, {raw_df.shape[1]} columns")

# ==============================================================================
# STEP 2: DATA CLEANING (MISSING VALUE IMPUTATION)
# ==============================================================================
cleaned_df = raw_df.copy()

# Impute Numerical Missing Values using Median
num_imputer = SimpleImputer(strategy='median')
cleaned_df['distance_km'] = num_imputer.fit_transform(cleaned_df[['distance_km']])

# Impute Categorical Missing Values using Mode
cat_imputer = SimpleImputer(strategy='most_frequent')
cleaned_df['weather_condition'] = cat_imputer.fit_transform(cleaned_df[['weather_condition']]).ravel()

print(f"[INFO] Missing Values Post-Imputation:\n{cleaned_df.isnull().sum()}")

# ==============================================================================
# STEP 3: OUTLIER REMOVAL (IQR METHOD)
# ==============================================================================
Q1 = cleaned_df['delivery_time_min'].quantile(0.25)
Q3 = cleaned_df['delivery_time_min'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

initial_count = len(cleaned_df)
filtered_df = cleaned_df[(cleaned_df['delivery_time_min'] >= lower_bound) & 
                         (cleaned_df['delivery_time_min'] <= upper_bound)].copy()
removed_outliers = initial_count - len(filtered_df)
print(f"[INFO] Outliers Removed: {removed_outliers} rows filtered out.")

# ==============================================================================
# STEP 4: FEATURE NORMALIZATION & SCALING
# ==============================================================================
num_features = ['distance_km', 'traffic_density_score', 'payload_weight_kg', 'delivery_cost_usd']
scaler = StandardScaler()

scaled_array = scaler.fit_transform(filtered_df[num_features])
scaled_feature_cols = [f"{col}_scaled" for col in num_features]
scaled_df = pd.DataFrame(scaled_array, columns=scaled_feature_cols, index=filtered_df.index)

# Combine Cleaned & Scaled Data
final_df = pd.concat([filtered_df, scaled_df], axis=1)

print("\n=== PREPROCESSING PIPELINE EXECUTED SUCCESSFULLY ===")
print(final_df[['order_id', 'distance_km', 'distance_km_scaled', 'delivery_time_min']].head())