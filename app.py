# app.py
import joblib
import pandas as pd
from flask import Flask, request, jsonify

app = Flask(__name__)

# Path to your saved model file
MODEL_PATH = "hgb_model_with_threshold.joblib"

# Columns expected by the model (exclude ID and target)
FEATURE_COLS = [
    "Warehouse_block",
    "Mode_of_Shipment",
    "Customer_care_calls",
    "Customer_rating",
    "Cost_of_the_Product",
    "Prior_purchases",
    "Product_importance",
    "Gender",
    "Discount_offered",
    "Weight_in_gms"
]

# Load saved pipeline + threshold
print("Loading model...")
saved = joblib.load(MODEL_PATH)
pipeline = saved["pipeline"]
THRESHOLD = float(saved.get("threshold", 0.5))  # fallback to 0.5 if missing
print(f"Loaded pipeline. Using threshold = {THRESHOLD}")

# Helper: validate and prepare input JSON -> DataFrame
def parse_input_json(json_data):
    """
    Accepts either:
      - a single dict with feature keys, or
      - a list of dicts
    Returns pandas.DataFrame with columns FEATURE_COLS in order.
    """
    if isinstance(json_data, dict):
        data = [json_data]
    elif isinstance(json_data, list):
        data = json_data
    else:
        raise ValueError("JSON must be an object or an array of objects.")

    df = pd.DataFrame(data)

    # Check missing columns
    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Keep only required columns and in correct order
    df = df[FEATURE_COLS].copy()

    # Convert numeric columns to numeric types (coerce errors -> NaN)
    # adjust names if they differ in your dataset
    numeric_cols = ["Customer_care_calls", "Customer_rating", "Cost_of_the_Product",
                    "Prior_purchases", "Discount_offered", "Weight_in_gms"]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Optional: handle NaN - here we return error if any NaN exists
    if df.isnull().any(axis=None):
        
        raise ValueError("One or more input values are missing or invalid (NaN). Please provide valid values for all features.")

    return df

@app.route("/predict", methods=["POST"])
def predict():
    """
    POST JSON example (single):
    {
      "Warehouse_block": "D",
      "Mode_of_Shipment": "Flight",
      "Customer_care_calls": 4,
      "Customer_rating": 2,
      "Cost_of_the_Product": 177,
      "Prior_purchases": 3,
      "Product_importance": "low",
      "Gender": "F",
      "Discount_offered": 44,
      "Weight_in_gms": 1233
    }

    Or a list of such objects for batch predictions.
    """
    try:
        json_data = request.get_json(force=True)
        df = parse_input_json(json_data)
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400

    try:
        probs = pipeline.predict_proba(df)[:, 1]  # probability of class 1 (late)
        preds = (probs >= THRESHOLD).astype(int)

        results = []
        for i in range(len(df)):
            results.append({
                "input_index": i,
                "probability_late": float(probs[i]),
                "predicted_label": int(preds[i])  # 1 => NOT reached on time (late), 0 => reached on time
            })

        return jsonify({
            "success": True,
            "threshold": THRESHOLD,
            "n_predictions": len(results),
            "results": results
        })
    except Exception as e:
        return jsonify({"success": False, "error": f"Prediction failed: {str(e)}"}), 500

@app.route("/", methods=["GET"])
def home():
    return "<h3>HGB Late Delivery Prediction API</h3><p>POST JSON to /predict</p>"

if __name__ == "__main__":
    app.run(debug=True)
