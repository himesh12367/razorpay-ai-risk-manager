# ============================================================
# RAZORPAY AI RISK MANAGER
# PHASE 4 - MODEL TRAINING
# ============================================================

import os
import glob
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score
)


# ============================================================
# PROJECT SETTINGS
# ============================================================

print("\n==============================================")
print("       RAZORPAY AI RISK MANAGER")
print("          PHASE 4 - MODEL TRAINING")
print("==============================================\n")


# ============================================================
# 1. FIND DATASET
# ============================================================

print("Loading dataset...")

DATA_FOLDER = os.path.join("data", "raw")

csv_files = glob.glob(os.path.join(DATA_FOLDER, "*.csv"))

if not csv_files:
    raise FileNotFoundError(
        "No CSV dataset found inside data/raw folder."
    )

# Use the first CSV file found
DATASET_PATH = csv_files[0]

print(f"Dataset: {os.path.basename(DATASET_PATH)}")


# ============================================================
# 2. LOAD DATASET
# ============================================================

df = pd.read_csv(DATASET_PATH)

print("Dataset loaded successfully!")
print(f"Shape: {df.shape}")


# ============================================================
# 3. CREATE ENGINEERED FEATURES
# ============================================================

print("\nCreating engineered features...")

# ------------------------------------------------------------
# Amount compared with origin account balance
# ------------------------------------------------------------

df["amount_to_origin_balance"] = (
    df["amount"] /
    (df["oldbalanceOrg"] + 1)
)


# ------------------------------------------------------------
# Amount compared with destination account balance
# ------------------------------------------------------------

df["amount_to_destination_balance"] = (
    df["amount"] /
    (df["oldbalanceDest"] + 1)
)


# ------------------------------------------------------------
# Origin balance is zero
# ------------------------------------------------------------

df["origin_balance_zero"] = (
    df["oldbalanceOrg"] == 0
).astype(int)


# ------------------------------------------------------------
# Destination balance is zero
# ------------------------------------------------------------

df["destination_balance_zero"] = (
    df["oldbalanceDest"] == 0
).astype(int)


# ------------------------------------------------------------
# Log transformed transaction amount
# ------------------------------------------------------------

df["log_amount"] = np.log1p(df["amount"])


print("Engineered features created successfully!")


# ============================================================
# 4. REQUIRED COLUMNS
# ============================================================

FEATURE_COLUMNS = [
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

TARGET_COLUMN = "isFraud"


# ============================================================
# 5. CHECK REQUIRED COLUMNS
# ============================================================

required_columns = FEATURE_COLUMNS + [TARGET_COLUMN]

missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing_columns:

    raise ValueError(
        f"Missing columns in dataset: {missing_columns}"
    )

print("\nAll required columns found!")


# ============================================================
# 6. PREPARE FEATURES
# ============================================================

print("\n==============================================")
print("              FEATURE DATA")
print("==============================================")

X = df[FEATURE_COLUMNS].copy()
y = df[TARGET_COLUMN].copy()


# ============================================================
# 7. ENCODE TRANSACTION TYPE
# ============================================================

# Convert transaction type into numbers
#
# CASH_IN  -> 0
# CASH_OUT -> 1
# DEBIT    -> 2
# PAYMENT  -> 3
# TRANSFER -> 4

type_mapping = {
    "CASH_IN": 0,
    "CASH_OUT": 1,
    "DEBIT": 2,
    "PAYMENT": 3,
    "TRANSFER": 4
}

X["type"] = X["type"].map(type_mapping)


# Check for unknown transaction types
if X["type"].isna().any():

    unknown_types = df.loc[
        X["type"].isna(),
        "type"
    ].unique()

    raise ValueError(
        f"Unknown transaction types found: {unknown_types}"
    )


# ============================================================
# 8. CHECK DATA TYPES
# ============================================================

print("\nFeature columns:")

for column in X.columns:
    print(f"- {column}")


# ============================================================
# 9. DATA VALIDATION
# ============================================================

print("\n==============================================")
print("              DATA VALIDATION")
print("==============================================")

print("\nMissing values:")

missing_values = X.isnull().sum()

print(missing_values)

total_missing = missing_values.sum()

print(f"\nTotal missing values: {total_missing}")


if total_missing > 0:
    raise ValueError(
        "Missing values detected. Please clean the dataset."
    )

print("No missing values found! ✓")


# ------------------------------------------------------------
# Infinite value check
# ------------------------------------------------------------

print("\nChecking infinite values...")

infinite_values = np.isinf(
    X.select_dtypes(include=np.number)
).sum().sum()

print(f"Total infinite values: {infinite_values}")


if infinite_values > 0:

    print("Replacing infinite values with 0...")

    X.replace(
        [np.inf, -np.inf],
        0,
        inplace=True
    )

print("No infinite values found! ✓")


# ============================================================
# 10. TARGET DISTRIBUTION
# ============================================================

print("\n==============================================")
print("             TARGET DISTRIBUTION")
print("==============================================")

print(y.value_counts())

print("\nTarget percentage:")

target_percentage = (
    y.value_counts(normalize=True) * 100
)

print(target_percentage)


# ============================================================
# 11. TRAIN / TEST SPLIT
# ============================================================

print("\n==============================================")
print("             TRAIN / TEST SPLIT")
print("==============================================")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print(f"Training data: {X_train.shape}")
print(f"Testing data : {X_test.shape}")


print("\nTraining target distribution:")

print(y_train.value_counts())


print("\nTesting target distribution:")

print(y_test.value_counts())


# ============================================================
# 12. HANDLE CLASS IMBALANCE
# ============================================================

print("\n==============================================")
print("         HANDLING CLASS IMBALANCE")
print("==============================================")

fraud_count = (y_train == 1).sum()
normal_count = (y_train == 0).sum()

print(f"Normal transactions: {normal_count}")
print(f"Fraud transactions : {fraud_count}")


# ------------------------------------------------------------
# Keep all fraud transactions.
#
# Randomly select a limited number of normal transactions.
#
# Ratio:
# 1 fraud : 10 normal
# ------------------------------------------------------------

normal_indices = np.where(
    y_train.values == 0
)[0]

fraud_indices = np.where(
    y_train.values == 1
)[0]


# Maximum normal transactions = 10 times fraud
desired_normal_count = min(
    len(normal_indices),
    len(fraud_indices) * 10
)


np.random.seed(42)

selected_normal_indices = np.random.choice(
    normal_indices,
    size=desired_normal_count,
    replace=False
)


selected_indices = np.concatenate([
    fraud_indices,
    selected_normal_indices
])


np.random.shuffle(selected_indices)


X_train_balanced = X_train.iloc[
    selected_indices
].copy()

y_train_balanced = y_train.iloc[
    selected_indices
].copy()


print("\nBalanced training data:")

print(
    f"Normal transactions: {(y_train_balanced == 0).sum()}"
)

print(
    f"Fraud transactions : {(y_train_balanced == 1).sum()}"
)

print(
    f"Total training rows : {len(y_train_balanced)}"
)


# ============================================================
# 13. TRAIN RANDOM FOREST MODEL
# ============================================================

print("\n==============================================")
print("          TRAINING RANDOM FOREST")
print("==============================================")

print("Training model...")
print("This may take some time...\n")


model = RandomForestClassifier(
    n_estimators=100,
    max_depth=15,
    min_samples_split=5,
    min_samples_leaf=2,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)


model.fit(
    X_train_balanced,
    y_train_balanced
)


print("\nModel training completed successfully! ✓")


# ============================================================
# 14. MAKE PREDICTIONS
# ============================================================

print("\n==============================================")
print("              MODEL PREDICTION")
print("==============================================")

y_pred = model.predict(X_test)

y_probability = model.predict_proba(
    X_test
)[:, 1]


print("Predictions generated successfully! ✓")


# ============================================================
# 15. CONFUSION MATRIX
# ============================================================

print("\n==============================================")
print("             CONFUSION MATRIX")
print("==============================================")

cm = confusion_matrix(
    y_test,
    y_pred
)

print(cm)


# ============================================================
# 16. CLASSIFICATION REPORT
# ============================================================

print("\n==============================================")
print("          CLASSIFICATION REPORT")
print("==============================================")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=[
            "Normal",
            "Fraud"
        ],
        digits=4
    )
)


# ============================================================
# 17. EVALUATION METRICS
# ============================================================

print("\n==============================================")
print("            MODEL PERFORMANCE")
print("==============================================")


precision = precision_score(
    y_test,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    zero_division=0
)

roc_auc = roc_auc_score(
    y_test,
    y_probability
)


print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")
print(f"ROC-AUC   : {roc_auc:.4f}")


# ============================================================
# 18. FEATURE IMPORTANCE
# ============================================================

print("\n==============================================")
print("            FEATURE IMPORTANCE")
print("==============================================")

feature_importance = pd.DataFrame({
    "feature": FEATURE_COLUMNS,
    "importance": model.feature_importances_
})

feature_importance = feature_importance.sort_values(
    by="importance",
    ascending=False
)

print(feature_importance.to_string(index=False))


# ============================================================
# 19. SAVE MODEL
# ============================================================

print("\n==============================================")
print("              SAVING MODEL")
print("==============================================")


MODEL_FOLDER = "models"

os.makedirs(
    MODEL_FOLDER,
    exist_ok=True
)


MODEL_PATH = os.path.join(
    MODEL_FOLDER,
    "fraud_detection_model.pkl"
)


joblib.dump(
    model,
    MODEL_PATH
)


print(
    f"Model saved successfully:\n{MODEL_PATH}"
)


# ============================================================
# 20. SAVE FEATURE INFORMATION
# ============================================================

FEATURE_INFO = {
    "feature_columns": FEATURE_COLUMNS,
    "type_mapping": type_mapping,
    "target_column": TARGET_COLUMN
}


FEATURE_INFO_PATH = os.path.join(
    MODEL_FOLDER,
    "feature_info.pkl"
)


joblib.dump(
    FEATURE_INFO,
    FEATURE_INFO_PATH
)


print(
    f"Feature information saved:\n{FEATURE_INFO_PATH}"
)


# ============================================================
# 21. SAVE FEATURE IMPORTANCE
# ============================================================

IMPORTANCE_PATH = os.path.join(
    MODEL_FOLDER,
    "feature_importance.csv"
)


feature_importance.to_csv(
    IMPORTANCE_PATH,
    index=False
)


print(
    f"Feature importance saved:\n{IMPORTANCE_PATH}"
)


# ============================================================
# 22. FINAL SUMMARY
# ============================================================

print("\n==============================================")
print("              TRAINING SUMMARY")
print("==============================================")

print(
    f"Total transactions : {len(df)}"
)

print(
    f"Training rows      : {len(X_train_balanced)}"
)

print(
    f"Testing rows       : {len(X_test)}"
)

print(
    f"Fraud transactions : {(y == 1).sum()}"
)

print(
    f"Fraud percentage   : {((y == 1).mean() * 100):.4f}%"
)

print(
    f"Precision          : {precision:.4f}"
)

print(
    f"Recall             : {recall:.4f}"
)

print(
    f"F1 Score           : {f1:.4f}"
)

print(
    f"ROC-AUC            : {roc_auc:.4f}"
)


print("\n==============================================")
print("       PHASE 4 - STEP 3 COMPLETED!")
print("==============================================")

print("\nModel files created:")
print(f"1. {MODEL_PATH}")
print(f"2. {FEATURE_INFO_PATH}")
print(f"3. {IMPORTANCE_PATH}")

print("\nNext step: Build the fraud prediction system/API.")