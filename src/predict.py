import os
import sys
import joblib
import numpy as np
import pandas as pd


# ============================================================
# RAZORPAY AI RISK MANAGER
# FRAUD TRANSACTION PREDICTION SYSTEM
# ============================================================


# ------------------------------------------------------------
# PATH CONFIGURATION
# ------------------------------------------------------------

# Get project root directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "fraud_detection_model.pkl"
)

FEATURE_INFO_PATH = os.path.join(
    BASE_DIR,
    "models",
    "feature_info.pkl"
)


# ------------------------------------------------------------
# EXPECTED FEATURES
# ------------------------------------------------------------

DEFAULT_FEATURES = [
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


# ------------------------------------------------------------
# TYPE ENCODING
# ------------------------------------------------------------

# PaySim transaction types
# These values are used only if feature_info.pkl
# does not contain a type mapping.

DEFAULT_TYPE_MAPPING = {
    "PAYMENT": 0,
    "TRANSFER": 1,
    "CASH_OUT": 2,
    "DEBIT": 3,
    "CASH_IN": 4
}


# ------------------------------------------------------------
# DISPLAY HEADER
# ------------------------------------------------------------

def print_header():
    print("=" * 60)
    print("                 RAZORPAY AI RISK MANAGER")
    print("                  FRAUD PREDICTION SYSTEM")
    print("=" * 60)


# ------------------------------------------------------------
# LOAD MODEL
# ------------------------------------------------------------

def load_model():
    print("\nLoading trained model...")

    if not os.path.exists(MODEL_PATH):
        print("\nERROR: Model file not found!")
        print("Expected location:")
        print(MODEL_PATH)
        sys.exit(1)

    try:
        model = joblib.load(MODEL_PATH)

        print("Model loaded successfully! ✓")

        return model

    except Exception as e:
        print("\nModel loading failed!")
        print("Error:", e)
        sys.exit(1)


# ------------------------------------------------------------
# LOAD FEATURE INFORMATION
# ------------------------------------------------------------

def load_feature_info():
    print("\nLoading feature information...")

    if not os.path.exists(FEATURE_INFO_PATH):
        print("feature_info.pkl not found.")
        print("Using default feature configuration.")

        return {
            "features": DEFAULT_FEATURES,
            "feature_names": DEFAULT_FEATURES,
            "type_mapping": DEFAULT_TYPE_MAPPING
        }

    try:
        feature_info = joblib.load(FEATURE_INFO_PATH)

        print("Feature information loaded successfully! ✓")

        return feature_info

    except Exception as e:
        print("Could not load feature information.")
        print("Using default configuration.")
        print("Error:", e)

        return {
            "features": DEFAULT_FEATURES,
            "feature_names": DEFAULT_FEATURES,
            "type_mapping": DEFAULT_TYPE_MAPPING
        }


# ------------------------------------------------------------
# GET FEATURE LIST
# ------------------------------------------------------------

def get_feature_names(feature_info, model):
    """
    Find the feature names from feature_info.pkl.
    Different versions of the training script may store
    the feature list under different keys.
    """

    # Try common keys first
    possible_keys = [
        "feature_names",
        "features",
        "columns",
        "feature_columns"
    ]

    for key in possible_keys:

        if isinstance(feature_info, dict):

            if key in feature_info:

                value = feature_info[key]

                if isinstance(value, (list, tuple, np.ndarray)):

                    features = list(value)

                    if len(features) == 10:
                        return features

    # Try model feature_names_in_
    if hasattr(model, "feature_names_in_"):

        features = list(model.feature_names_in_)

        if len(features) == 10:
            return features

    # Final fallback
    return DEFAULT_FEATURES


# ------------------------------------------------------------
# GET TYPE MAPPING
# ------------------------------------------------------------

def get_type_mapping(feature_info):

    if isinstance(feature_info, dict):

        possible_keys = [
            "type_mapping",
            "type_map",
            "transaction_type_mapping",
            "type_encoder"
        ]

        for key in possible_keys:

            if key in feature_info:

                mapping = feature_info[key]

                if isinstance(mapping, dict):

                    return mapping

    return DEFAULT_TYPE_MAPPING


# ------------------------------------------------------------
# SAFE FLOAT INPUT
# ------------------------------------------------------------

def get_float(prompt, minimum=None):

    while True:

        try:

            value = float(input(prompt))

            if minimum is not None and value < minimum:
                print(
                    f"Please enter a value greater than or equal to {minimum}."
                )
                continue

            return value

        except ValueError:

            print("Invalid input. Please enter a number.")


# ------------------------------------------------------------
# SAFE INTEGER INPUT
# ------------------------------------------------------------

def get_integer(prompt, minimum=None):

    while True:

        try:

            value = int(input(prompt))

            if minimum is not None and value < minimum:
                print(
                    f"Please enter a value greater than or equal to {minimum}."
                )
                continue

            return value

        except ValueError:

            print("Invalid input. Please enter a whole number.")


# ------------------------------------------------------------
# TRANSACTION TYPE INPUT
# ------------------------------------------------------------

def get_transaction_type(type_mapping):

    print("\nTransaction types:")

    # Display available transaction types
    for transaction_type in type_mapping.keys():
        print("-", transaction_type)

    while True:

        transaction_type = input(
            "\nEnter transaction type: "
        ).strip().upper()

        if transaction_type in type_mapping:

            return transaction_type, type_mapping[transaction_type]

        print("\nInvalid transaction type.")

        print("Please choose one of:")

        for transaction_type in type_mapping.keys():
            print("-", transaction_type)


# ------------------------------------------------------------
# CREATE ENGINEERED FEATURES
# ------------------------------------------------------------

def create_features(
    step,
    transaction_type,
    amount,
    oldbalance_org,
    oldbalance_dest,
    type_mapping
):

    # Encode transaction type
    if transaction_type in type_mapping:

        type_encoded = type_mapping[transaction_type]

    else:

        # Try converting if already numeric
        try:
            type_encoded = int(transaction_type)

        except Exception:
            type_encoded = 0

    # --------------------------------------------------------
    # Engineered features
    # --------------------------------------------------------

    # Avoid division by zero
    amount_to_origin_balance = (
        amount / (oldbalance_org + 1)
    )

    amount_to_destination_balance = (
        amount / (oldbalance_dest + 1)
    )

    # Check whether origin balance is zero
    if oldbalance_org == 0:
        origin_balance_zero = 1
    else:
        origin_balance_zero = 0

    # Check whether destination balance is zero
    if oldbalance_dest == 0:
        destination_balance_zero = 1
    else:
        destination_balance_zero = 0

    # Log transformed amount
    log_amount = np.log1p(amount)

    # --------------------------------------------------------
    # Create dataframe
    # --------------------------------------------------------

    data = {
        "step": step,
        "type": type_encoded,
        "amount": amount,
        "oldbalanceOrg": oldbalance_org,
        "oldbalanceDest": oldbalance_dest,
        "amount_to_origin_balance": amount_to_origin_balance,
        "amount_to_destination_balance": amount_to_destination_balance,
        "origin_balance_zero": origin_balance_zero,
        "destination_balance_zero": destination_balance_zero,
        "log_amount": log_amount
    }

    return pd.DataFrame([data])


# ------------------------------------------------------------
# ALIGN FEATURES WITH MODEL
# ------------------------------------------------------------

def prepare_input(data, feature_names):

    # Make sure all expected columns exist
    for feature in feature_names:

        if feature not in data.columns:

            data[feature] = 0

    # Keep only required features
    data = data[feature_names]

    return data


# ------------------------------------------------------------
# GET FRAUD PROBABILITY
# ------------------------------------------------------------

def get_fraud_probability(model, data):

    try:

        probabilities = model.predict_proba(data)

        # Fraud is normally class 1
        if probabilities.shape[1] >= 2:

            fraud_probability = probabilities[0][1]

        else:

            fraud_probability = probabilities[0][0]

        return float(fraud_probability)

    except Exception:

        # If predict_proba is not available
        prediction = model.predict(data)[0]

        if prediction == 1:
            return 1.0

        return 0.0


# ------------------------------------------------------------
# DETERMINE RISK LEVEL
# ------------------------------------------------------------

def get_risk_level(fraud_probability):

    percentage = fraud_probability * 100

    if percentage >= 80:

        return "CRITICAL"

    elif percentage >= 50:

        return "HIGH"

    elif percentage >= 20:

        return "MEDIUM"

    else:

        return "LOW"


# ------------------------------------------------------------
# DISPLAY RESULT
# ------------------------------------------------------------

def display_result(
    prediction,
    fraud_probability,
    transaction_type,
    amount
):

    percentage = fraud_probability * 100

    risk_level = get_risk_level(fraud_probability)

    print("\n")
    print("=" * 60)
    print("                 FRAUD PREDICTION RESULT")
    print("=" * 60)

    print(f"\nTransaction Type : {transaction_type}")
    print(f"Transaction Amount : ₹{amount:,.2f}")

    print(
        f"\nFraud Probability : {percentage:.2f}%"
    )

    print(
        f"Risk Level        : {risk_level}"
    )

    print(
        f"Model Prediction  : "
        f"{'FRAUDULENT' if prediction == 1 else 'NORMAL'}"
    )

    print("\n" + "-" * 60)

    # Explanation
    if prediction == 1:

        print("⚠ WARNING: Potential fraudulent transaction detected!")

        if risk_level == "CRITICAL":
            print("Recommendation: BLOCK transaction immediately.")

        elif risk_level == "HIGH":
            print("Recommendation: Hold transaction for manual review.")

        else:
            print("Recommendation: Perform additional verification.")

    else:

        print("✓ Transaction appears to be legitimate.")
        print("Recommendation: Transaction can proceed.")

    print("-" * 60)


# ------------------------------------------------------------
# PREDICTION FUNCTION
# ------------------------------------------------------------

def predict_transaction(
    model,
    feature_info
):

    feature_names = get_feature_names(
        feature_info,
        model
    )

    type_mapping = get_type_mapping(
        feature_info
    )

    print("\n")
    print("=" * 60)
    print("              FRAUD TRANSACTION PREDICTION")
    print("=" * 60)

    print("\nEnter transaction details:\n")

    # --------------------------------------------------------
    # STEP
    # --------------------------------------------------------

    step = get_integer(
        "Step: ",
        minimum=0
    )

    # --------------------------------------------------------
    # TRANSACTION TYPE
    # --------------------------------------------------------

    transaction_type, _ = get_transaction_type(
        type_mapping
    )

    # --------------------------------------------------------
    # AMOUNT
    # --------------------------------------------------------

    amount = get_float(
        "Amount: ",
        minimum=0
    )

    # --------------------------------------------------------
    # ORIGIN BALANCE
    # --------------------------------------------------------

    oldbalance_org = get_float(
        "Old Balance (Origin): ",
        minimum=0
    )

    # --------------------------------------------------------
    # DESTINATION BALANCE
    # --------------------------------------------------------

    oldbalance_dest = get_float(
        "Old Balance (Destination): ",
        minimum=0
    )

    # --------------------------------------------------------
    # CREATE FEATURES
    # --------------------------------------------------------

    data = create_features(
        step,
        transaction_type,
        amount,
        oldbalance_org,
        oldbalance_dest,
        type_mapping
    )

    # --------------------------------------------------------
    # ALIGN FEATURES
    # --------------------------------------------------------

    data = prepare_input(
        data,
        feature_names
    )

    print("\nProcessing transaction...")

    # --------------------------------------------------------
    # PREDICTION
    # --------------------------------------------------------

    try:

        prediction = int(
            model.predict(data)[0]
        )

        fraud_probability = get_fraud_probability(
            model,
            data
        )

        display_result(
            prediction,
            fraud_probability,
            transaction_type,
            amount
        )

    except Exception as e:

        print("\nPrediction failed!")
        print("Error:", e)

        print("\nFeatures sent to model:")
        print(data)

        return False

    return True


# ------------------------------------------------------------
# MAIN PROGRAM
# ------------------------------------------------------------

def main():

    print_header()

    # Load model
    model = load_model()

    # Load feature information
    feature_info = load_feature_info()

    # Display feature information
    feature_names = get_feature_names(
        feature_info,
        model
    )

    print("\nModel features:")

    for i, feature in enumerate(feature_names, start=1):

        print(f"{i}. {feature}")

    # --------------------------------------------------------
    # CONTINUOUS PREDICTION LOOP
    # --------------------------------------------------------

    while True:

        try:

            predict_transaction(
                model,
                feature_info
            )

        except KeyboardInterrupt:

            print("\n\nProgram stopped by user.")
            break

        except Exception as e:

            print("\nUnexpected error:")
            print(e)

        print("\n")

        choice = input(
            "Do you want to test another transaction? (y/n): "
        ).strip().lower()

        if choice not in ["y", "yes"]:

            print("\n")
            print("=" * 60)
            print("Thank you for using RazorPay AI Risk Manager!")
            print("=" * 60)

            break


# ------------------------------------------------------------
# PROGRAM START
# ------------------------------------------------------------

if __name__ == "__main__":
    main()