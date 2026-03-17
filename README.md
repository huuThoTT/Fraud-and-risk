# Real-time E-commerce Fraud Detection System

This project implements an end-to-end Big Data pipeline for detecting fraudulent e-commerce transactions in real-time using Apache Kafka, PySpark, Machine Learning (XGBoost), and a Node.js + Socket.io Glassmorphism Web Dashboard.

## 🚀 Features
- **Machine Learning Model**: XGBoost trained on 150,000 synthetic e-commerce transactions with complex fraud patterns.
- **Data Generator**: A Python script simulating a live backend that streams continuous transactions (with random delays) into Apache Kafka.
- **Real-time Pipeline**: A Python processor consumes the Kafka topic, runs feature engineering on-the-fly, calculates risk scores with the ML model, and emits alerts back to Kafka.
- **Glassmorphism Dashboard**: A premium, futuristic Web UI built with Express & Socket.io that consumes the scored transactions and displays real-time statistics, threat maps, and visual alerts.

## 🛠 Tech Stack
- **Machine Learning**: Python, Pandas, Scikit-learn, XGBoost.
- **Streaming Pipeline**: Apache Kafka, `kafka-python`.
- **Backend Analytics**: Node.js, Express, Socket.io, `kafkajs`.
- **Frontend**: HTML5, Vanilla CSS, JS (Dark Mode, Glassmorphism).

## ⚡ How to Run

### 1. Start Apache Kafka
Make sure you have Kafka and Zookeeper installed (e.g. via Homebrew):
```bash
brew services start kafka
```

### 2. Start the Pipeline
Open two separate terminals to run the stream generator and processor:
```bash
# Terminal 1: Run Stream Processor (Inference)
python pipeline/spark_processor.py

# Terminal 2: Run Data Generator (Producer)
python pipeline/data_generator.py
```

### 3. Start the Web Dashboard
Open a third terminal, navigate to the `dashboard/` directory, install dependencies, and start the frontend:
```bash
cd dashboard
npm install
node server.js
```

**View Live Dashboard:** Go to http://localhost:4000
