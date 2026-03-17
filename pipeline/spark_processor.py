import os
import json
import pickle
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.functions import from_json, col, udf
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, IntegerType, FloatType

# Disable PySpark warnings
import logging
logging.getLogger("py4j").setLevel(logging.ERROR)

# --- 1. Load ML Model ---
MODEL_PATH = '../ml/fraud_model.pkl'
print(f"Loading ML model from {MODEL_PATH}...")
with open(MODEL_PATH, 'rb') as f:
    model_data = pickle.load(f)
    model = model_data['model']
    features = model_data['features']

def predict_fraud_probability(data_json):
    """
    UDF (User Defined Function) to parse JSON, transform data to match model features,
    and run XGBoost inference.
    """
    try:
        # data_json is a JSON string of the transaction
        data = json.loads(data_json)
        
        # We need to construct a DataFrame with a single row to feed to the model
        df = pd.DataFrame([data])
        
        # 1. Feature Engineering (must match train_model.py)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['hour_of_day'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        
        # 2. Categorical Encoding (One-Hot)
        # In a real system, we must ensure all one-hot columns expected by the model exist.
        categorical_cols = ['device_type', 'location', 'payment_method']
        df_encoded = pd.get_dummies(df, columns=categorical_cols)
        
        # Ensure all required features are present, fill missing with 0
        for f in features:
            if f not in df_encoded.columns:
                df_encoded[f] = 0
                
        # Reorder columns to match the trained model exactly
        X = df_encoded[features]
        
        # 3. Predict Probability
        prob = float(model.predict_proba(X)[0][1]) # Get probability of class 1 (Fraud)
        return prob
    except Exception as e:
        print(f"Error during prediction: {e}")
        return -1.0

# --- 2. Initialize Spark Session ---
# Note: In production we would use true Spark Structured Streaming with Kafka package.
# For local POC simplicity without relying on JVM Kafka packages, we use standard kafka-python
from kafka import KafkaConsumer, KafkaProducer

consumer = KafkaConsumer(
    'transactions_raw',
    bootstrap_servers=['localhost:9092'],
    auto_offset_reset='latest',
    enable_auto_commit=True,
    value_deserializer=lambda x: json.loads(x.decode('utf-8'))
)

producer = KafkaProducer(
    bootstrap_servers=['localhost:9092'],
    value_serializer=lambda x: json.dumps(x).encode('utf-8')
)

print("Started listening to Kafka topic 'transactions_raw'...")

def process_stream():
    for message in consumer:
        transaction = message.value
        
        # Construct JSON string for our UDF-like function
        txn_json = json.dumps(transaction)
        
        # Run ML Inference
        risk_score = predict_fraud_probability(txn_json)
        
        # Determine if it's an alert
        is_fraud_alert = 1 if risk_score > 0.8 else 0
        
        # Prepare Output Message
        output_msg = {
            'transaction_id': transaction.get('transaction_id'),
            'user_id': transaction.get('user_id'),
            'amount': transaction.get('amount'),
            'device_type': transaction.get('device_type'),
            'location': transaction.get('location'),
            'risk_score': round(risk_score, 4),
            'is_fraud_alert': is_fraud_alert,
            'ground_truth': transaction.get('ground_truth', 0),
            'timestamp': transaction.get('timestamp')
        }
        
        # Print Alert locally
        if is_fraud_alert:
            print(f"🚨 FRAUD ALERT! Score: {risk_score:.2f} | TXN: {output_msg['transaction_id']}")
        else:
            print(f"✅ Approved: {output_msg['transaction_id']} | Risk: {risk_score:.2f}")

        # Push to downstream topic for Node.js Dashboard to consume
        producer.send('transactions_scored', value=output_msg)

if __name__ == "__main__":
    process_stream()
