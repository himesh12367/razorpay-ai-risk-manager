import pandas as pd

# Path to our dataset
file_path = "data/raw/PS_20174392719_1491204439457_log.csv"

# Load the dataset
df = pd.read_csv(file_path)

# Basic information
print("\n===== DATASET SHAPE =====")
print(df.shape)

print("\n===== COLUMN NAMES =====")
print(df.columns.tolist())

print("\n===== FIRST 5 ROWS =====")
print(df.head())

print("\n===== DATA TYPES =====")
print(df.dtypes)

print("\n===== MISSING VALUES =====")
print(df.isnull().sum())

print("\n===== FRAUD DISTRIBUTION =====")
print(df["isFraud"].value_counts())

print("\n===== FRAUD PERCENTAGE =====")
print(df["isFraud"].value_counts(normalize=True) * 100)