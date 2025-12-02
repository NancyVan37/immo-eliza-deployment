# predict.py
import joblib
import pandas as pd

# Load trained model once
MODEL_PATH = "model.pkl"
model = joblib.load(MODEL_PATH)

def predict_price(features: dict):
    """
    features = {"area": 120, "rooms": 3, "postcode": 1000, ...}
    """
    try:
        input_df = pd.DataFrame([features])
        prediction = model.predict(input_df)[0]
        return {"prediction": round(prediction, 2)}

    except Exception as e:
        return {"error": str(e)}
