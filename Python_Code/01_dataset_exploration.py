import pandas as pd

file_path = r"C:\Users\shrey\Downloads\archive (4)\dynamic_supply_chain_logistics_dataset.csv"

df = pd.read_csv(file_path)

print("===== DATASET SHAPE =====")
print(df.shape)

print("\n===== COLUMN NAMES =====")
print(df.columns.tolist())

print("\n===== FIRST 5 ROWS =====")
print(df.head())

print("\n===== DATA TYPES =====")
print(df.dtypes)

print("\n===== MISSING VALUES =====")
print(df.isnull().sum())

print("\n===== DUPLICATE ROWS =====")
print(df.duplicated().sum())

print("\n===== BASIC STATISTICS =====")
print(df.describe(include="all"))