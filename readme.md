# Data-Driven Analytics for Late Product Delivery

## Project title
Data-Driven Analytics for Late Product Delivery

## Objective
Predict whether a shipment will be delivered late (or not) using historical shipping data. The notebook focus on building a reproducible pipeline that transforms raw features, trains a classifier, and exposes a minimal Flask API for serving predictions.

## Basic techniques used (high-level)
- Exploratory Data Analysis (EDA) and data cleaning
- Feature engineering (categorical encoding, numeric transformations)
- Train / validation split and cross-validation
- Preprocessing + modeling pipeline (scikit-learn `Pipeline` / `ColumnTransformer`)
- Thresholding on predicted probabilities to control trade-offs between precision/recall
- Saving & loading model + pipeline using `joblib`
- Lightweight REST API using Flask for online predictions



## Files in this repository
- `shipping.ipynb` – Jupyter notebook with data exploration, preprocessing, model training, evaluation, and model saving.
- `app.py` – Minimal Flask application that loads the saved pipeline and returns predictions from a `/predict` endpoint.
- `requirements.txt` – List of Python dependencies to install the runtime environment.
- `hgb_model_with_threshold.joblib` – saved model + pipeline file used by `app.py`. 

## Installation
1. Create a virtual environment (recommended):

```bash
python -m venv venv
source venv/bin/activate    # Linux / macOS
venv\Scripts\activate     # Windows
```

2. Install dependencies from `requirements.txt`:

```bash
pip install -r requirements.txt
```

## How to run the notebook
Open `shipping.ipynb` with Jupyter or Jupyter Lab to reproduce the data processing, model training, and the steps that produce the saved `hgb_model_with_threshold.joblib` file.

```bash
jupyter lab shipping.ipynb
# or
jupyter notebook shipping.ipynb
```

## How to run the Flask API
1. Ensure `hgb_model_with_threshold.joblib` is placed in the same directory as `app.py` (or update `MODEL_PATH` inside `app.py`).
2. Start the API:

```bash
python app.py
```

3. Endpoints:
- `GET /` – simple home page confirming the API is running.
- `POST /predict` – accepts JSON (single object or list of objects) and returns predicted probability and binary prediction per input.

### Example request (single input)
```json
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
```

### Example response 
```json
{
  "success": true,
  "threshold": 0.5,
  "n_predictions": 1,
  "results": [
    {
      "input_index": 0,
      "probability_late": 0.72,
      "predicted_label": 1
    }
  ]
}
```

> In `app.py` the label `1` indicates *late* (not reached on time) and `0` indicates *on time*.

## Required feature columns
The Flask API expects the following columns in each prediction JSON: `Warehouse_block`, `Mode_of_Shipment`, `Customer_care_calls`, `Customer_rating`, `Cost_of_the_Product`, `Prior_purchases`, `Product_importance`, `Gender`, `Discount_offered`, `Weight_in_gms`. The same order is enforced when the input is parsed.



