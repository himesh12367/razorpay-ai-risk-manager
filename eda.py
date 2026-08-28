import pandas as pd

# Dataset path
file_path = "data/raw/PS_20174392719_1491204439457_log.csv"

# Load dataset
df = pd.read_csv(file_path)

print("\n========== TRANSACTION TYPES ==========")
print(df["type"].value_counts())

print("\n========== FRAUD BY TRANSACTION TYPE ==========")
print(pd.crosstab(df["type"], df["isFraud"]))

print("\n========== FRAUD RATE BY TRANSACTION TYPE ==========")
fraud_rate = df.groupby("type")["isFraud"].mean() * 100
print(fraud_rate.sort_values(ascending=False))

print("\n========== AVERAGE TRANSACTION AMOUNT ==========")
print(df.groupby("isFraud")["amount"].mean())

print("\n========== MEDIAN TRANSACTION AMOUNT ==========")
print(df.groupby("isFraud")["amount"].median())

print("\n========== FRAUD TRANSACTION SUMMARY ==========")
fraud_transactions = df[df["isFraud"] == 1]

print(fraud_transactions[[
    "type",
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest"
]].describe())