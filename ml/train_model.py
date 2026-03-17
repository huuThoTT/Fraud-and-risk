import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import pickle
import os

def train_fraud_model(data_path='../data/synthetic_transactions.csv', model_path='fraud_model.pkl'):
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    
    # Feature Engineering (Basic for POC)
    # Convert timestamp to hour of day for simple pattern recognition
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['hour_of_day'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    
    # Categorical encoding (One-Hot for simplicity in POC)
    categorical_cols = ['device_type', 'location', 'payment_method']
    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    
    # Select features based on generated columns
    # Exclude non-predictive or target columns
    exclude_cols = ['transaction_id', 'user_id', 'timestamp', 'is_fraud']
    features = [col for col in df_encoded.columns if col not in exclude_cols]
    
    X = df_encoded[features]
    y = df_encoded['is_fraud']
    
    print(f"Features used: {features}")
    print(f"Dataset shape: {X.shape}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Train XGBoost Model
    print("\nTraining XGBoost Classifier...")
    # Calculate scale_pos_weight for imbalanced classes
    scale_pos_weight = (len(y) - y.sum()) / y.sum()
    
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        eval_metric='logloss'
    )
    
    model.fit(X_train, y_train)
    
    # Evaluate
    print("\nEvaluating Model on Test Set:")
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    # Save the model
    os.makedirs(os.path.dirname(model_path) if os.path.dirname(model_path) else '.', exist_ok=True)
    with open(model_path, 'wb') as f:
        pickle.dump({'model': model, 'features': features}, f)
        
    print(f"\nModel saved to {model_path}")

if __name__ == "__main__":
    train_fraud_model()
