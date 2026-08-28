from flask import Flask, request, render_template_string
import joblib
import numpy as np
import os

app = Flask(__name__)

# ============================================================
# LOAD TRAINED MODEL
# ============================================================

MODEL_PATH = "models/fraud_detection_model.pkl"

try:
    model = joblib.load(MODEL_PATH)
    print("Model loaded successfully!")
except Exception as e:
    model = None
    print("Model loading failed:", e)


# ============================================================
# TRANSACTION TYPE ENCODING
# ============================================================

TYPE_MAPPING = {
    "CASH_IN": 0,
    "CASH_OUT": 1,
    "DEBIT": 2,
    "PAYMENT": 3,
    "TRANSFER": 4
}


# ============================================================
# CREATE FEATURES
# ============================================================

def create_features(
    step,
    transaction_type,
    amount,
    oldbalance_org,
    oldbalance_dest
):

    # Balance changes
    amount_to_origin_balance = (
        amount / oldbalance_org
        if oldbalance_org > 0
        else 0
    )

    amount_to_destination_balance = (
        amount / oldbalance_dest
        if oldbalance_dest > 0
        else 0
    )

    # Zero balance indicators
    origin_balance_zero = 1 if oldbalance_org == 0 else 0

    destination_balance_zero = 1 if oldbalance_dest == 0 else 0

    # Log transformed amount
    log_amount = np.log1p(amount)

    # Encode transaction type
    type_encoded = TYPE_MAPPING.get(transaction_type, 0)

    # IMPORTANT:
    # These must match the order used while training
    features = [
        step,
        type_encoded,
        amount,
        oldbalance_org,
        oldbalance_dest,
        amount_to_origin_balance,
        amount_to_destination_balance,
        origin_balance_zero,
        destination_balance_zero,
        log_amount
    ]

    return np.array(features).reshape(1, -1)


# ============================================================
# HTML PAGE
# ============================================================

HTML = """
<!DOCTYPE html>

<html>

<head>

    <title>Razorpay AI Risk Manager</title>

    <style>

        body {
            font-family: Arial, sans-serif;
            background: #f4f6f8;
            margin: 0;
            padding: 0;
        }

        .container {
            width: 500px;
            margin: 50px auto;
            background: white;
            padding: 30px;
            border-radius: 12px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.15);
        }

        h1 {
            text-align: center;
            margin-bottom: 5px;
        }

        .subtitle {
            text-align: center;
            color: #666;
            margin-bottom: 25px;
        }

        label {
            display: block;
            margin-top: 15px;
            margin-bottom: 5px;
            font-weight: bold;
        }

        input, select {
            width: 100%;
            padding: 10px;
            border: 1px solid #ccc;
            border-radius: 6px;
            box-sizing: border-box;
        }

        button {
            width: 100%;
            margin-top: 25px;
            padding: 12px;
            border: none;
            border-radius: 6px;
            background: #2563eb;
            color: white;
            font-size: 16px;
            cursor: pointer;
        }

        button:hover {
            background: #1d4ed8;
        }

        .result {
            margin-top: 25px;
            padding: 20px;
            border-radius: 8px;
            background: #f1f5f9;
        }

        .fraud {
            color: #dc2626;
            font-weight: bold;
        }

        .normal {
            color: #16a34a;
            font-weight: bold;
        }

        .error {
            color: #dc2626;
            font-weight: bold;
        }

    </style>

</head>


<body>

<div class="container">

    <h1>Razorpay AI Risk Manager</h1>

    <div class="subtitle">
        Fraud Transaction Detection System
    </div>


    <form method="POST">

        <label>Step</label>

        <input
            type="number"
            name="step"
            value="1"
            min="1"
            required
        >


        <label>Transaction Type</label>

        <select name="transaction_type" required>

            <option value="CASH_IN">CASH_IN</option>

            <option value="CASH_OUT">CASH_OUT</option>

            <option value="DEBIT">DEBIT</option>

            <option value="PAYMENT">PAYMENT</option>

            <option value="TRANSFER">TRANSFER</option>

        </select>


        <label>Amount</label>

        <input
            type="number"
            name="amount"
            step="0.01"
            value="10000"
            min="0"
            required
        >


        <label>Old Balance (Origin)</label>

        <input
            type="number"
            name="oldbalance_org"
            step="0.01"
            value="50000"
            min="0"
            required
        >


        <label>Old Balance (Destination)</label>

        <input
            type="number"
            name="oldbalance_dest"
            step="0.01"
            value="20000"
            min="0"
            required
        >


        <button type="submit">
            Check Transaction
        </button>

    </form>


    {% if result %}

    <div class="result">

        <h2>Prediction Result</h2>

        <p>
            <strong>Transaction Type:</strong>
            {{ result.transaction_type }}
        </p>

        <p>
            <strong>Amount:</strong>
            ₹{{ result.amount }}
        </p>

        <p>
            <strong>Fraud Probability:</strong>
            {{ result.probability }}%
        </p>

        <p>
            <strong>Risk Level:</strong>
            {{ result.risk }}
        </p>

        <p>
            <strong>Model Prediction:</strong>

            {% if result.prediction == 1 %}

                <span class="fraud">
                    FRAUD
                </span>

            {% else %}

                <span class="normal">
                    NORMAL
                </span>

            {% endif %}

        </p>


        {% if result.prediction == 1 %}

            <p class="fraud">
                ⚠ Transaction appears suspicious.
            </p>

        {% else %}

            <p class="normal">
                ✓ Transaction appears legitimate.
            </p>

        {% endif %}

    </div>

    {% endif %}


    {% if error %}

    <div class="result">

        <p class="error">
            Error: {{ error }}
        </p>

    </div>

    {% endif %}

</div>

</body>

</html>
"""


# ============================================================
# HOME ROUTE
# ============================================================

@app.route("/", methods=["GET", "POST"])
def home():

    result = None
    error = None

    if request.method == "POST":

        try:

            # Get values from form
            step = int(request.form["step"])

            transaction_type = request.form["transaction_type"]

            amount = float(request.form["amount"])

            oldbalance_org = float(
                request.form["oldbalance_org"]
            )

            oldbalance_dest = float(
                request.form["oldbalance_dest"]
            )


            # Create model features
            features = create_features(
                step,
                transaction_type,
                amount,
                oldbalance_org,
                oldbalance_dest
            )


            # Check model
            if model is None:
                raise Exception(
                    "Trained model could not be loaded."
                )


            # Make prediction
            prediction = int(
                model.predict(features)[0]
            )


            # Fraud probability
            if hasattr(model, "predict_proba"):

                probabilities = model.predict_proba(features)[0]

                probability = float(
                    probabilities[1] * 100
                )

            else:

                probability = (
                    100.0 if prediction == 1 else 0.0
                )


            # Determine risk
            if probability >= 70:

                risk = "HIGH"

            elif probability >= 30:

                risk = "MEDIUM"

            else:

                risk = "LOW"


            # Store result
            result = {

                "transaction_type":
                    transaction_type,

                "amount":
                    f"{amount:,.2f}",

                "probability":
                    f"{probability:.2f}",

                "risk":
                    risk,

                "prediction":
                    prediction
            }


        except Exception as e:

            error = str(e)

    return render_template_string(
        HTML,
        result=result,
        error=error
    )


# ============================================================
# START FLASK SERVER
# ============================================================

if __name__ == "__main__":

    print("=" * 60)

    print("RAZORPAY AI RISK MANAGER")

    print("FRAUD DETECTION WEB APPLICATION")

    print("=" * 60)

    print()

    print("Starting Flask server...")

    print("Open this URL in your browser:")

    print("http://127.0.0.1:5000")

    print()

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True
    )