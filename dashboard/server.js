const express = require('express');
const { createServer } = require('http');
const { Server } = require('socket.io');
const { Kafka } = require('kafkajs');
const path = require('path');

const app = express();
const httpServer = createServer(app);
const io = new Server(httpServer, {
    cors: { origin: "*" }
});

// Serve static frontend files
app.use(express.static(path.join(__dirname, 'public')));

// Kafka Configuration
const kafka = new Kafka({
    clientId: 'fraud-dashboard-app',
    brokers: ['localhost:9092']
});

const consumer = kafka.consumer({ groupId: 'dashboard-group' });

// WebSocket Connection
io.on('connection', (socket) => {
    console.log(`🔌 Client connected: ${socket.id}`);
    
    // Send a welcome message
    socket.emit('status', { message: 'Connected to Fraud Detection Stream' });
    
    socket.on('disconnect', () => {
        console.log(`❌ Client disconnected: ${socket.id}`);
    });
});

// Run Kafka Consumer and Broadcast to WebSockets
const runKafkaListener = async () => {
    try {
        await consumer.connect();
        console.log("✅ Kafka Consumer connected");
        
        await consumer.subscribe({ topic: 'transactions_scored', fromBeginning: false });
        console.log("📡 Subscribed to 'transactions_scored' topic.");

        await consumer.run({
            eachMessage: async ({ topic, partition, message }) => {
                const value = message.value.toString();
                try {
                    const transactionData = JSON.parse(value);
                    
                    // Broadcast the transaction to all connected WebSocket clients
                    io.emit('new_transaction', transactionData);
                    
                    if (transactionData.is_fraud_alert) {
                        console.log(`🚨 FRAUD ALERT EMITTED: ${transactionData.transaction_id} (Score: ${transactionData.risk_score})`);
                    }
                } catch (err) {
                    console.error("Error parsing message from Kafka:", err);
                }
            },
        });
    } catch (err) {
        console.error("❌ Error starting Kafka listener:", err);
    }
};

const PORT = 4000;
httpServer.listen(PORT, async () => {
    console.log(`🚀 Fraud Dashboard Backend running on http://localhost:${PORT}`);
    // Start Kafka consumer when server starts
    runKafkaListener();
});
