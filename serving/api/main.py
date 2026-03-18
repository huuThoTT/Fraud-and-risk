import pickle
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from .schemas import TransactionInput, PredictionResponse
import os

app = FastAPI(title="Fraud Detection Serving API")

# Global variables for model and features
model = None
features = None

MODEL_PATH = os.getenv("MODEL_PATH", "../../ml/fraud_model.pkl")

@app.on_event("startup")
def load_model():
    global model, features
    try:
        with open(MODEL_PATH, 'rb') as f:
            data = pickle.load(f)
            model = data['model']
            features = data['features']
        print(f"Model loaded successfully from {MODEL_PATH}")
    except FileNotFoundError:
        print(f"Error: Model file not found at {MODEL_PATH}")
    except Exception as e:
        print(f"Error loading model: {str(e)}")

@app.get("/health")
def health_check():
    if model is not None:
        return {"status": "healthy", "model_loaded": True}
    return {"status": "unhealthy", "model_loaded": False}

@app.post("/predict", response_model=PredictionResponse)
def predict(transaction: TransactionInput):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # Convert Pydantic model to dict
        data = transaction.dict()
        
        # Create DataFrame (1 row)
        df = pd.DataFrame([data])
        
        # ── Feature Engineering (Matching train_model.py) ───────────────
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['hour_of_day'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        
        # ── Categorical Encoding ────────────────────────────────────────
        # Apply get_dummies
        categorical_cols = ['device_type', 'location', 'payment_method']
        df_encoded = pd.get_dummies(df, columns=categorical_cols)
        
        # ── Align features with training ────────────────────────────────
        # Create a DataFrame with all zeros for the training features
        X = pd.DataFrame(0, index=[0], columns=features)
        
        # Overwrite with existing encoded columns that match
        for col in df_encoded.columns:
            if col in features:
                X[col] = df_encoded[col]
        
        # ── Predict ─────────────────────────────────────────────────────
        fraud_prob = float(model.predict_proba(X)[0, 1])
        is_fraud = bool(model.predict(X)[0])
        
        return PredictionResponse(
            transaction_id=transaction.transaction_id,
            fraud_probability=round(fraud_prob, 4),
            is_fraud=is_fraud
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")
