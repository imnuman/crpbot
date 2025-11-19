// V7 Trading Signals Dashboard JavaScript

// Initialize dashboard
document.addEventListener('DOMContentLoaded', function() {
    console.log('V7 Dashboard initialized');
    fetchAllData();

    // Auto-refresh every 5 seconds
    setInterval(fetchAllData, 5000);
});

// Fetch all dashboard data
async function fetchAllData() {
    try {
        await Promise.all([
            fetchSystemStatus(),
            fetchV7Statistics(),
            fetchV7RecentSignals()
        ]);
        updateLastUpdateTime();
    } catch (error) {
        console.error('Error fetching data:', error);
        updateSystemStatus(false);
    }
}

// Fetch system status
async function fetchSystemStatus() {
    try {
        const response = await fetch('/api/status');
        const data = await response.json();
        updateSystemStatus(data.status === 'live');
    } catch (error) {
        console.error('Error fetching status:', error);
        updateSystemStatus(false);
    }
}

// Fetch V7 statistics
async function fetchV7Statistics() {
    try {
        const response = await fetch('/api/v7/statistics');
        const data = await response.json();

        // Update statistics cards
        document.getElementById('v7TotalSignals').textContent = data.total_signals || 0;
        document.getElementById('v7BuyCount').textContent = data.by_direction?.long || 0;
        document.getElementById('v7SellCount').textContent = data.by_direction?.short || 0;
        document.getElementById('v7HoldCount').textContent = data.by_direction?.hold || 0;
        document.getElementById('v7AvgConfidence').textContent =
            ((data.avg_confidence || 0) * 100).toFixed(1) + '%';
        document.getElementById('v7Cost').textContent =
            '$' + (data.total_cost || 0).toFixed(4);

        // Update breakdowns
        updateBreakdown('v7ByDirection', data.by_direction);
        updateBreakdown('v7BySymbol', data.by_symbol);
        updateBreakdown('v7ByTier', data.by_tier);

    } catch (error) {
        console.error('Error fetching V7 statistics:', error);
    }
}

// Fetch V7 recent signals
async function fetchV7RecentSignals() {
    try {
        const response = await fetch('/api/v7/signals/recent/24');
        const signals = await response.json();

        const tableBody = document.getElementById('v7SignalsTable');

        if (!signals || signals.length === 0) {
            tableBody.innerHTML = '<tr><td colspan="9" class="no-data">No V7 signals in the last 24 hours</td></tr>';
            return;
        }

        tableBody.innerHTML = signals.map(signal => {
            const direction = signal.direction === 'long' ? 'BUY' :
                            signal.direction === 'short' ? 'SELL' : 'HOLD';
            const directionClass = signal.direction === 'long' ? 'signal-buy' :
                                  signal.direction === 'short' ? 'signal-sell' : 'signal-hold';
            const confidence = (signal.confidence * 100).toFixed(1) + '%';

            // Format prices
            const entry = formatPrice(signal.entry_price);
            const sl = formatPrice(signal.sl_price);
            const tp = formatPrice(signal.tp_price);

            // Calculate R:R ratio
            let rr = 'N/A';
            if (signal.entry_price && signal.sl_price && signal.tp_price) {
                const risk = Math.abs(signal.entry_price - signal.sl_price);
                const reward = Math.abs(signal.tp_price - signal.entry_price);
                if (risk > 0) {
                    rr = '1:' + (reward / risk).toFixed(2);
                }
            }

            // Parse reasoning
            let reasoning = 'N/A';
            try {
                const reasoningObj = JSON.parse(signal.reasoning || '{}');
                reasoning = reasoningObj.reasoning || reasoningObj.summary || 'N/A';
            } catch (e) {
                reasoning = signal.reasoning || 'N/A';
            }

            // Truncate reasoning
            if (reasoning.length > 100) {
                reasoning = reasoning.substring(0, 100) + '...';
            }

            return `
                <tr>
                    <td>${formatTime(signal.timestamp)}</td>
                    <td class="symbol-cell">${signal.symbol}</td>
                    <td><span class="signal-badge ${directionClass}">${direction}</span></td>
                    <td>${confidence}</td>
                    <td>${entry}</td>
                    <td>${sl}</td>
                    <td>${tp}</td>
                    <td>${rr}</td>
                    <td class="reasoning-cell">${reasoning}</td>
                </tr>
            `;
        }).join('');

    } catch (error) {
        console.error('Error fetching V7 signals:', error);
    }
}

// Helper: Format price
function formatPrice(price) {
    if (!price || price === null) return 'N/A';
    return '$' + price.toFixed(2).replace(/\B(?=(\d{3})+(?!\d))/g, ',');
}

// Helper: Format time
function formatTime(timestamp) {
    if (!timestamp) return 'N/A';
    const date = new Date(timestamp);
    return date.toLocaleTimeString('en-US', {
        hour: '2-digit',
        minute: '2-digit',
        hour12: false
    });
}

// Helper: Update breakdown display
function updateBreakdown(elementId, data) {
    const element = document.getElementById(elementId);
    if (!element || !data) return;

    const items = Object.entries(data)
        .sort((a, b) => b[1] - a[1])
        .map(([key, value]) => `<div class="breakdown-item">${key}: <strong>${value}</strong></div>`)
        .join('');

    element.innerHTML = items || '<div class="breakdown-item">No data</div>';
}

// Update system status indicator
function updateSystemStatus(isOnline) {
    const statusDot = document.getElementById('systemStatus');
    const statusText = document.getElementById('statusText');

    if (isOnline) {
        statusDot.className = 'status-dot status-online';
        statusText.textContent = 'Online';
    } else {
        statusDot.className = 'status-dot status-offline';
        statusText.textContent = 'Offline';
    }
}

// Update last update time
function updateLastUpdateTime() {
    const now = new Date();
    const timeString = now.toLocaleTimeString('en-US', {
        hour: '2-digit',
        minute: '2-digit',
        second: '2-digit',
        hour12: false
    });
    document.getElementById('lastUpdate').textContent = timeString;
}
