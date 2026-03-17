import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
import os

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

def generate_synthetic_transactions(num_samples=100000, fraud_ratio=0.03):
    print(f"Generating {num_samples} synthetic transactions...")
    
    # Define realistic features
    users = [f"user_{i}" for i in range(1, int(num_samples * 0.4))] # 40% unique users
    devices = ["mobile_app", "mobile_web", "desktop_web", "tablet"]
    locations = ["US", "VN", "UK", "IN", "BR", "JP", "FR", "DE"]
    payment_methods = ["credit_card", "debit_card", "paypal", "crypto", "ewallet"]
    
    # Generate base normal transactions
    data = {
        'transaction_id': [f"TXN_{100000 + i}" for i in range(num_samples)],
        'user_id': np.random.choice(users, num_samples),
        'timestamp': [datetime(2026, 1, 1) + timedelta(minutes=random.randint(0, 100000)) for _ in range(num_samples)],
        'amount': np.random.lognormal(mean=np.log(20), sigma=1.2, size=num_samples).round(2),
        'device_type': np.random.choice(devices, num_samples, p=[0.6, 0.2, 0.15, 0.05]),
        'location': np.random.choice(locations, num_samples, p=[0.4, 0.2, 0.1, 0.1, 0.05, 0.05, 0.05, 0.05]),
        'payment_method': np.random.choice(payment_methods, num_samples, p=[0.5, 0.2, 0.15, 0.05, 0.1]),
        'is_guest_checkout': np.random.choice([0, 1], num_samples, p=[0.8, 0.2]),
        'time_since_last_login_hours': np.random.exponential(scale=24, size=num_samples).round(1),
        'is_fraud': np.zeros(num_samples) # Default to 0
    }
    
    df = pd.DataFrame(data)
    
    # Inject anomalies/fraud cases
    num_frauds = int(num_samples * fraud_ratio)
    fraud_indices = np.random.choice(df.index, num_frauds, replace=False)
    
    print(f"Injecting {num_frauds} fraudulent patterns...")
    
    for idx in fraud_indices:
        df.at[idx, 'is_fraud'] = 1
        
        # Fraud patterns:
        fraud_type = random.choice(['high_amount', 'crypto_guest', 'rapid_login', 'unusual_location'])
        
        if fraud_type == 'high_amount':
            # Abnormally high amount
            df.at[idx, 'amount'] = round(random.uniform(1000, 5000), 2)
            df.at[idx, 'time_since_last_login_hours'] = 0.1 # logged in and immediately bought huge
        elif fraud_type == 'crypto_guest':
            # Crypto guest checkout
            df.at[idx, 'payment_method'] = 'crypto'
            df.at[idx, 'is_guest_checkout'] = 1
            df.at[idx, 'amount'] = round(random.uniform(200, 1000), 2)
        elif fraud_type == 'rapid_login':
            # Very short time since login with new device
            df.at[idx, 'time_since_last_login_hours'] = round(random.uniform(0.01, 0.05), 3)
            df.at[idx, 'device_type'] = 'mobile_web'
        elif fraud_type == 'unusual_location':
            # Specific high-risk location + guest + high amount
            df.at[idx, 'location'] = 'BR'
            df.at[idx, 'is_guest_checkout'] = 1
            df.at[idx, 'amount'] = round(random.uniform(500, 2000), 2)
    
    # Ensure amount is not negative
    df['amount'] = df['amount'].clip(lower=0.5)
    
    # Sort by timestamp to simulate realistic stream
    df = df.sort_values(by='timestamp').reset_index(drop=True)
    
    # Save to CSV
    os.makedirs('../data', exist_ok=True)
    output_path = '../data/synthetic_transactions.csv'
    df.to_csv(output_path, index=False)
    print(f"Saved synthetic dataset to {output_path}")
    print("\nDataset Summary:")
    print(df['is_fraud'].value_counts(normalize=True))
    print("\nSample Data:")
    print(df.head())

if __name__ == "__main__":
    generate_synthetic_transactions(150000, 0.04) # 150k rows, 4% fraud rate
