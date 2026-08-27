# 🚚 Strategic Planning and Data Exploration in Logistics

## Data-Driven Logistics Delivery Delay and Risk Analysis

![Python](https://img.shields.io/badge/Python-3.13-blue?logo=python)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-150458?logo=pandas)
![Scikit Learn](https://img.shields.io/badge/Scikit--learn-Machine%20Learning-F7931E?logo=scikit-learn)
![Matplotlib](https://img.shields.io/badge/Matplotlib-Visualization-11557C)
![Kaggle](https://img.shields.io/badge/Dataset-Kaggle-20BEFF?logo=kaggle)
![Status](https://img.shields.io/badge/Project-Completed-success)

> **Week 1 Internship Task — Strategic Planning and Data Exploration in Logistics**

---

## 📌 Project Overview

Modern logistics operations generate large amounts of data from vehicles, warehouses, suppliers, transportation networks, environmental conditions, and delivery systems.

This project presents a **data-driven logistics analytics framework** designed to explore the factors associated with delivery delays, assess operational risk, identify logistics segments, and demonstrate how machine learning can support better supply chain decision-making.

The project uses a publicly available logistics dataset containing:

- **32,065 records**
- **26 variables**
- **0 missing values**
- **0 duplicate records**
- **3 risk categories**

The analysis combines **Exploratory Data Analysis (EDA), correlation analysis, regression, classification, and clustering** to establish an end-to-end analytical roadmap for logistics operations.

---

## 🎯 Problem Statement

Logistics companies need to balance:

- Delivery speed
- Transportation cost
- Inventory availability
- Route safety
- Supplier reliability
- Operational risk

Delivery delays can be influenced by traffic congestion, weather conditions, loading/unloading time, route risk, supplier lead time, customs clearance, disruptions, and other operational factors.

### Core Business Question

> **How can historical logistics data be analyzed using Python and data science techniques to identify factors associated with delivery delays, assess operational risk, and support better logistics planning and resource allocation?**

---

## 🎯 Project Objectives

The project aims to:

1. Explore and understand logistics operational data.
2. Identify factors associated with delivery-time deviation.
3. Analyze important logistics KPIs.
4. Perform data quality validation and exploratory analysis.
5. Investigate delivery-time deviation prediction using regression.
6. Analyze logistics risk classification using machine learning.
7. Segment logistics operations using K-Means clustering.
8. Propose an optimization framework for route and resource allocation.
9. Translate analytical results into business recommendations.
10. Establish a roadmap for developing a future logistics decision-support system.

---

# 📊 Dataset

The project uses a publicly available:

**Dynamic Supply Chain Logistics Dataset**

### Dataset Statistics

| Property | Value |
|---|---:|
| Records | 32,065 |
| Features | 26 |
| Missing Values | 0 |
| Duplicate Records | 0 |
| Risk Categories | 3 |

### Major Feature Groups

#### 🚛 Transportation
- Vehicle GPS Latitude
- Vehicle GPS Longitude
- Fuel Consumption Rate
- Traffic Congestion Level

#### 🏭 Warehouse & Operations
- Warehouse Inventory Level
- Loading/Unloading Time
- Handling Equipment Availability
- Order Fulfillment Status

#### 🌦️ Environmental
- Weather Condition Severity
- Port Congestion Level
- IoT Temperature

#### 💰 Cost & Supplier
- Shipping Costs
- Supplier Reliability Score
- Lead Time
- Historical Demand

#### ⚠️ Risk & Delivery
- Route Risk Level
- Customs Clearance Time
- Driver Behavior Score
- Fatigue Monitoring Score
- Disruption Likelihood Score
- Delay Probability
- Risk Classification
- Delivery Time Deviation

---

# 📈 Key Performance Indicators (KPIs)

The following KPIs were selected for logistics performance monitoring:

| KPI | Variable | Purpose |
|---|---|---|
| Delivery Time Deviation | `delivery_time_deviation` | Measures deviation from expected delivery schedule |
| ETA Variation | `eta_variation_hours` | Measures variation in estimated arrival time |
| Delay Probability | `delay_probability` | Indicates likelihood of delivery delay |
| Shipping Cost | `shipping_costs` | Monitors transportation cost |
| Order Fulfillment | `order_fulfillment_status` | Evaluates order completion performance |

---

# 🔬 Analytical Methodology

The project follows an end-to-end data science workflow:

```text
                 Logistics Dataset
                        │
                        ▼
                Data Collection
                        │
                        ▼
                Data Validation
                        │
                        ▼
                 Data Cleaning
                        │
                        ▼
                       EDA
                        │
              ┌─────────┼─────────┐
              ▼         ▼         ▼
            Delay      Risk      Cost
           Analysis   Analysis  Analysis
              │         │         │
              └─────────┼─────────┘
                        ▼
                Feature Engineering
                        │
             ┌──────────┴──────────┐
             ▼                     ▼
        Regression            Classification
             │                     │
             ▼                     ▼
      Delay Prediction       Risk Analysis
             │                     │
             └──────────┬──────────┘
                        ▼
                    Clustering
                        │
                        ▼
               Logistics Segmentation
                        │
                        ▼
                   Optimization
                        │
                        ▼
             Business Recommendations
🤖 Machine Learning Approaches
1. Regression

A Random Forest Regressor was used as an initial baseline model.

Target
delivery_time_deviation
Evaluation Metrics
Mean Absolute Error (MAE)
Root Mean Squared Error (RMSE)
R² Score
Initial Results
Metric	Result
MAE	3.725 hours
RMSE	4.182 hours
R²	-0.014
Interpretation

The initial regression model demonstrated limited predictive capability.

The negative R² indicates that the current feature set does not adequately explain the variation in delivery-time deviation.

This result establishes a useful baseline and highlights the need for:

Better feature engineering
Additional operational variables
Temporal analysis
Alternative algorithms
More representative real-world data
⚠️ 2. Risk Classification

A Random Forest Classifier was used to classify logistics operations into:

High Risk
Moderate Risk
Low Risk
Initial Accuracy
100%

The classification model achieved 1.00 precision, recall, and F1-score across the observed classes.

Important Validation Note

The unusually high classification performance requires further investigation.

Variables such as:

delay_probability
disruption_likelihood_score
route_risk_level

may contain information closely related to the construction of the risk classification label.

Therefore, feature leakage analysis is required before real-world deployment.

This project treats the 100% accuracy as an initial analytical result rather than evidence of perfect production performance.

🧩 3. K-Means Clustering

K-Means clustering was proposed to identify groups of logistics operations with similar characteristics.

Clustering Features
shipping_costs
lead_time_days
historical_demand
traffic_congestion_level
route_risk_level
delay_probability

Standardization is applied before clustering to prevent variables with larger numerical ranges from dominating the analysis.

Potential Business Use

Clustering can help identify:

High-cost operations
High-risk operations
High-demand operations
Delay-prone operations
Relatively efficient logistics segments

This can enable differentiated operational strategies rather than applying the same strategy to every shipment.

📊 Exploratory Data Analysis

The project performs exploratory analysis of:

Risk classification distribution
Delivery-time deviation
ETA variation
Shipping costs
Lead time
Traffic congestion
Route risk
Weather severity
Correlation between operational variables

Example analysis outputs include:

Correlation Analysis

The project investigates the relationship between numerical logistics variables and:

delivery_time_deviation

The resulting visualization is stored in:

Report/correlation_with_delivery_delay.png
🐍 Python Implementation

The project uses Python and major data science libraries including:

Python
Pandas
NumPy
Matplotlib
Scikit-learn
Example
import pandas as pd

df = pd.read_csv("dynamic_supply_chain_logistics_dataset.csv")

print(df.shape)
print(df.head())
print(df.isnull().sum())
Regression Example
from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

predictions = model.predict(X_test)
Clustering Example
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

kmeans = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10
)

clusters = kmeans.fit_predict(X_scaled)
💼 Business Impact

The proposed analytics framework can support logistics managers in:

🚚 Delivery Planning

Identify operational factors associated with delivery deviations.

⚠️ Risk Management

Detect potentially high-risk logistics operations.

💰 Cost Control

Analyze shipping and operational costs.

📦 Resource Allocation

Support better allocation of vehicles, warehouse resources, and delivery capacity.

📍 Route Planning

Use route-risk and traffic information to support route decisions.

📊 Decision Support

Transform historical logistics data into actionable operational insights.

🔮 Future Scope

The project can be extended into a complete logistics intelligence system.

Future improvements include:
Real-time GPS integration
Real-time traffic data
Weather API integration
Advanced time-series forecasting
XGBoost / LightGBM models
Deep learning
Vehicle Routing Problem (VRP)
Mixed Integer Linear Programming (MILP)
Real-time risk scoring
Automated route recommendations
Power BI / Tableau dashboard
Cloud deployment
Real-time logistics monitoring
📁 Project Structure
Week_1_Logistics_Data_Analysis/
│
├── Dataset/
│   └── dynamic_supply_chain_logistics_dataset.csv
│
├── Python_Code/
│   ├── 01_dataset_exploration.py
│   ├── 02_eda.py
│   ├── 03_correlation_analysis.py
│   ├── 04_delay_prediction.py
│   ├── 05_risk_classification.py
│   └── 06_logistics_clustering.py
│
└── Report/
    ├── correlation_with_delivery_delay.png
    ├── EDA screenshots
    └── Week_1_Strategic_Planning_Logistics_Data_Analysis.docx
🛠️ How to Run
1. Clone the repository
git clone < https://shreyansh01234.github.io/Week-1-Logistics-Data-Analysis/>
2. Navigate to the project
cd Week-1-Logistics-Data-Analysis
3. Install dependencies
pip install pandas numpy matplotlib scikit-learn
4. Run dataset exploration
python Python_Code/01_dataset_exploration.py
5. Run EDA
python Python_Code/02_eda.py
6. Run correlation analysis
python Python_Code/03_correlation_analysis.py
7. Run delay prediction
python Python_Code/04_delay_prediction.py
8. Run risk classification
python Python_Code/05_risk_classification.py
9. Run clustering
python Python_Code/06_logistics_clustering.py
📚 References
Kaggle — Dynamic Supply Chain Logistics Dataset.
Amazon Last Mile Routing Research Challenge.
Research literature on data-driven inventory routing and supply chain optimization.
Scikit-learn Documentation — Machine Learning in Python.
McKinney, W. — Python for Data Analysis, O'Reilly Media.
Python Software Foundation — Python Documentation.
👨‍💻 Author

Shreyansh Pandey

B.Tech — Computer Science & Information Technology

SAGE University, Indore

⭐ Project Status

Completed — Week 1 Strategic Planning & Data Exploration

This project demonstrates the use of Python-based data analytics and machine learning concepts to investigate logistics delivery performance, operational risk, and supply chain decision-making.
