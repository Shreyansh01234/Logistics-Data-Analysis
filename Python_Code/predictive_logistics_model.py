import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# 1. Dataset Generation & Simulation
np.random.seed(42)
n = 1000
df = pd.DataFrame({
    'distance_km': np.random.uniform(2.0, 50.0, n),
    'traffic_density_score': np.random.uniform(1.0, 10.0, n),
    'payload_weight_kg': np.random.uniform(1.0, 30.0, n),
    'weather_condition': np.random.choice(['Clear', 'Rain', 'Fog', 'Storm'], size=n, p=[0.55, 0.25, 0.12, 0.08])
})

weather_impact = {'Clear': 0, 'Rain': 8, 'Fog': 14, 'Storm': 25}
df['weather_delay'] = df['weather_condition'].map(weather_impact)
df['delivery_time_min'] = (df['distance_km'] * 2.1) + (df['traffic_density_score'] * 4.2) + (df['payload_weight_kg'] * 0.25) + df['weather_delay'] + np.random.normal(5, 3, n)

# One-Hot Encoding for Categorical Variables
df_encoded = pd.get_dummies(df.drop(columns=['weather_delay']), columns=['weather_condition'], drop_first=True)

X = df_encoded.drop(columns=['delivery_time_min'])
y = df_encoded['delivery_time_min']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Baseline Model Training: Ridge Regression
ridge = Ridge()
ridge.fit(X_train, y_train)
y_pred_ridge = ridge.predict(X_test)

# 3. Model Tuning: Random Forest Regressor
param_grid = {'n_estimators': [50, 100], 'max_depth': [10, 20, None]}
rf_grid = GridSearchCV(RandomForestRegressor(random_state=42), param_grid, cv=5, scoring='neg_mean_squared_error')
rf_grid.fit(X_train, y_train)

best_rf = rf_grid.best_estimator_
y_pred_rf = best_rf.predict(X_test)

# 4. Evaluation Metrics Output
print("=== PREDICTIVE MODEL PERFORMANCE METRICS ===")
print(f"Random Forest RMSE : {np.sqrt(mean_squared_error(y_test, y_pred_rf)):.2f} mins")
print(f"Random Forest MAE  : {mean_absolute_error(y_test, y_pred_rf):.2f} mins")
print(f"Random Forest R2   : {r2_score(y_test, y_pred_rf):.4f}")

# 5. Operational Optimization: Dynamic Dispatch Buffer Allocation
def calculate_optimal_schedule(pred_time, traffic_score, is_storm):
    if traffic_score > 7.0 or is_storm:
        return pred_time * 1.25  # 25% safety buffer for high risk routes
    return pred_time * 1.05      # 5% baseline operational buffer

X_test_copy = X_test.copy()
X_test_copy['pred_duration'] = y_pred_rf
X_test_copy['optimal_dispatch_window'] = X_test_copy.apply(
    lambda r: calculate_optimal_schedule(r['pred_duration'], r['traffic_density_score'], r.get('weather_condition_Storm', 0)), axis=1
)

print("\n=== OPTIMIZED DISPATCH TIMETABLE SAMPLE ===")
print(X_test_copy[['distance_km', 'traffic_density_score', 'pred_duration', 'optimal_dispatch_window']].head())