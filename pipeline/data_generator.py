import json
import time
import pandas as pd
from kafka import KafkaProducer
import random
from datetime import datetime

# Initialize Kafka Producer
producer = KafkaProducer(
    bootstrap_servers=['localhost:9092'],
    value_serializer=lambda x: json.dumps(x).encode('utf-8')
)

def stream_transactions(file_path='../data/synthetic_transactions.csv', speed=0.1):
    print(f"Loading data from {file_path}...")
    df = pd.read_csv(file_path)
    
    print("Starting transaction stream...")
    
    # We simulate a real-time stream by sending rows one by one
    for index, row in df.iterrows():
        # Convert row to dictionary
        transaction = row.to_dict()
        
        # Override timestamp with current time to make it truly realistic for dashboard
        transaction['timestamp'] = datetime.now().isoformat()
        
        # We don't send the 'is_fraud' label in a real scenario to the inference engine,
        # but we send it here just so the backend knows the ground truth for evaluation
        is_fraud_ground_truth = transaction.pop('is_fraud')
        transaction['ground_truth'] = is_fraud_ground_truth
        
        # Send to Kafka Topic
        producer.send('transactions_raw', value=transaction)
        
        if index % 100 == 0:
            print(f"Sent {index} transactions...")
            producer.flush()
            
        # Add random delay to simulate real network traffic (between speed/2 and speed*1.5)
        delay = random.uniform(speed / 2, speed * 1.5)
        time.sleep(delay)

if __name__ == "__main__":
    try:
        # Send 10 transactions per second (avg delay 0.1s)
        stream_transactions(speed=0.1)
    except KeyboardInterrupt:
        print("Streaming stopped.")
    finally:
        producer.close()
