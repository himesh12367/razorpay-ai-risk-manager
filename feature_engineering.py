import pandas as pd
import numpy as np
import glob
import os

print("Loading dataset...")

# Automatically find the CSV file
files = glob.glob("data/raw/*.csv")

if not files:
    raise FileNotFoundError("No CSV file found inside data/raw")

file_path = files[0]

print("Dataset found:", os.path.basename(file_path))

# Load dataset
df = pd.read_csv(file_path)

print("Original shape:", df.shape)


# ============================================================
# 1. CREATE RISK FEATURES
# ============================================================

# Avoid division by zero
df["amount_to_origin_balance"] = (
    df["amount"] / (df["oldbalanceOrg"] + 1)
)

df["amount_to_destination_balance"] = (
    df["amount"] / (df["oldbalanceDest"] + 1)
)

# Whether the account had zero balance before transaction
df["origin_balance_zero"] = (
    df["oldbalanceOrg"] == 0
).astype(int)

df["destination_balance_zero"] = (
    df["oldbalanceDest"] == 0
).astype(int)

# Log transformation helps handle very large transaction amounts
df["log_amount"] = np.log1p(df["amount"])


# ============================================================
# 2. SELECT FEATURES
# ============================================================

features = [
    "step",
    "type",
    "amount",
    "oldbalanceOrg",
    "oldbalanceDest",
    "amount_to_origin_balance",
    "amount_to_destination_balance",
    "origin_balance_zero",
    "destination_balance_zero",
    "log_amount"
]

target = "isFraud"

model_data = df[features + [target]].copy()


# ============================================================
# 3. CHECK THE RESULT
# ============================================================

print("\n===== MODEL DATA SHAPE =====")
print(model_data.shape)

print("\n===== MODEL FEATURES =====")
print(features)

print("\n===== TARGET DISTRIBUTION =====")
print(model_data[target].value_counts())

print("\n===== FIRST 5 ROWS =====")
print(model_data.head())

print("\n===== MISSING VALUES =====")
print(model_data.isnull().sum())

print("\nFeature engineering completed successfully!")