const socket = io();

// DOM Elements
const connectionStatus = document.getElementById('connection-status');
const tbody = document.getElementById('transactions-body');
const tpsCounter = document.getElementById('tps-counter');
const avgRiskCounter = document.getElementById('avg-risk');
const threatCounter = document.getElementById('threat-counter');
const gaugeRing = document.getElementById('latest-risk-ring');
const gaugeValue = document.getElementById('latest-risk-value');
const threatDetails = document.getElementById('threat-details');
const dynamicPulse = document.getElementById('dynamic-pulse');

// State
let transactionsProcessed = 0;
let fraudCount = 0;
let riskSum = 0;
let recentTransactions = []; // Keep track to limit DOM rows
const MAX_ROWS = 50;
let lastUpdateSec = Math.floor(Date.now() / 1000);
let currentSecondCount = 0;

// Socket Connection Status
socket.on('connect', () => {
    connectionStatus.textContent = "Live";
    connectionStatus.className = "badge badge-success";
});

socket.on('disconnect', () => {
    connectionStatus.textContent = "Disconnected";
    connectionStatus.className = "badge badge-danger";
});

// Calculate TPS
setInterval(() => {
    tpsCounter.textContent = currentSecondCount;
    currentSecondCount = 0;
}, 1000);

// Format currency
const formatter = new Intl.NumberFormat('en-US', {
    style: 'currency',
    currency: 'USD',
});

// Format risk value
const formatRisk = (score) => {
    return (score * 100).toFixed(1) + '%';
};

// Handle incoming transactions
socket.on('new_transaction', (txn) => {
    transactionsProcessed++;
    currentSecondCount++;
    riskSum += txn.risk_score;
    
    if (txn.is_fraud_alert) {
        fraudCount++;
        threatCounter.textContent = fraudCount;
        updateThreatPanel(txn);
    }
    
    // Update Avg Risk
    if (transactionsProcessed % 5 === 0) {
        avgRiskCounter.textContent = formatRisk(riskSum / transactionsProcessed);
    }

    insertTransactionRow(txn);
});

function insertTransactionRow(txn) {
    const tr = document.createElement('tr');
    tr.className = txn.is_fraud_alert ? 'row-fraud' : 'row-new';
    
    // Remove animation class after it plays so if it re-renders it doesn't stay stuck
    if (!txn.is_fraud_alert) {
        setTimeout(() => tr.classList.remove('row-new'), 1000);
    }
    
    // Determine status badge
    let statusBadge = '';
    if (txn.is_fraud_alert) {
        statusBadge = '<span class="badge badge-danger">BLOCKED</span>';
    } else if (txn.risk_score > 0.4) {
        statusBadge = '<span class="badge badge-warning">REVIEW</span>';
    } else {
        statusBadge = '<span class="badge badge-success">APPROVED</span>';
    }

    // Determine device icon
    const deviceIcon = txn.device_type.includes('mobile') ? '<i class="fa-solid fa-mobile-screen"></i>' 
                     : txn.device_type.includes('desktop') ? '<i class="fa-solid fa-desktop"></i>' : '<i class="fa-solid fa-tablet"></i>';

    tr.innerHTML = `
        <td>${txn.transaction_id}</td>
        <td>${formatter.format(txn.amount)}</td>
        <td>${txn.location}</td>
        <td>${deviceIcon}</td>
        <td class="${txn.is_fraud_alert ? 'text-danger' : ''}">${formatRisk(txn.risk_score)}</td>
        <td>${statusBadge}</td>
    `;
    
    // Add to top of list
    tbody.insertBefore(tr, tbody.firstChild);
    
    // Cleanup old rows to maintain performance
    if (tbody.children.length > MAX_ROWS) {
        tbody.removeChild(tbody.lastChild);
    }
}

function updateThreatPanel(txn) {
    document.getElementById('risk-description').textContent = "CRITICAL THREAT DETECTED";
    document.getElementById('risk-description').style.color = 'var(--danger)';
    
    gaugeValue.textContent = formatRisk(txn.risk_score);
    gaugeValue.style.color = 'var(--danger)';
    gaugeRing.style.borderTopColor = 'var(--danger)';
    gaugeRing.style.boxShadow = 'inset 0 0 30px rgba(255, 71, 87, 0.5)';
    
    threatDetails.innerHTML = `
        <p><strong>Target:</strong> <span>${txn.transaction_id}</span></p>
        <p><strong>Value:</strong> <span>${formatter.format(txn.amount)}</span></p>
        <p><strong>Origin:</strong> <span>${txn.location}</span></p>
        <p><strong>Vector:</strong> <span>${txn.device_type}</span></p>
    `;

    // Map pulse
    dynamicPulse.style.display = 'block';
    // Randomize position slightly for visual effect
    dynamicPulse.style.top = Math.floor(Math.random() * 60 + 20) + '%';
    dynamicPulse.style.left = Math.floor(Math.random() * 60 + 20) + '%';
    
    // Reset gauge after 5 seconds if no new threat
    setTimeout(() => {
        if(gaugeValue.textContent === formatRisk(txn.risk_score)) {
            document.getElementById('risk-description').textContent = "Monitoring streams...";
            document.getElementById('risk-description').style.color = 'var(--text-muted)';
            gaugeValue.style.color = 'var(--text-main)';
            gaugeRing.style.borderTopColor = 'var(--accent)';
            gaugeRing.style.boxShadow = 'inset 0 0 20px rgba(0,0,0,0.5)';
            dynamicPulse.style.display = 'none';
        }
    }, 5000);
}
