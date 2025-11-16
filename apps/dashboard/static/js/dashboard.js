// V6 Enhanced Model Dashboard JavaScript

// Chart instances
let charts = {};

// Initialize dashboard
document.addEventListener('DOMContentLoaded', function() {
    console.log('Dashboard initialized');
    initCharts();
    fetchAllData();

    // Auto-refresh every 5 seconds
    setInterval(fetchAllData, 5000);
});

// Initialize prediction charts
function initCharts() {
    const symbols = ['BTC', 'ETH', 'SOL'];

    symbols.forEach(symbol => {
        const ctx = document.getElementById(`chart${symbol}`).getContext('2d');
        charts[symbol] = new Chart(ctx, {
            type: 'doughnut',
            data: {
                labels: ['Down', 'Neutral', 'Up'],
                datasets: [{
                    data: [33, 34, 33],
                    backgroundColor: [
                        'rgba(239, 68, 68, 0.8)',   // Red
                        'rgba(107, 114, 128, 0.8)', // Gray
                        'rgba(16, 185, 129, 0.8)'   // Green
                    ],
                    borderWidth: 2,
                    borderColor: '#1a1f2e'
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        display: true,
                        position: 'bottom',
                        labels: {
                            color: '#e6e6e6',
                            font: { size: 11 }
                        }
                    },
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                return context.label + ': ' + (context.parsed * 100).toFixed(1) + '%';
                            }
                        }
                    }
                }
            }
        });
    });
}

// Fetch all dashboard data
async function fetchAllData() {
    try {
        await Promise.all([
            fetchSystemStatus(),
            fetchLivePredictions(),
            fetchSignalStats(),
            fetchRecentSignals()
        ]);
        updateLastUpdateTime();
    } catch (error) {
        console.error('Error fetching data:', error);
        updateSystemStatus(false);
    }
}

// Fetch system status
async function fetchSystemStatus() {
    const response = await fetch('/api/status');
    const data = await response.json();

    // Update status indicator
    updateSystemStatus(data.status === 'live');

    // Update overview
    document.getElementById('systemMode').textContent = data.mode.toUpperCase();
    document.getElementById('confidenceThreshold').textContent =
        (data.confidence_threshold * 100).toFixed(0) + '%';
    document.getElementById('avgAccuracy').textContent =
        (data.models.accuracy.average * 100).toFixed(2) + '%';
    document.getElementById('architecture').textContent = data.models.architecture;

    // Update data sources
    updateDataSource('Coinbase', data.data_sources.coinbase);
    updateDataSource('Kraken', data.data_sources.kraken);
    updateDataSource('Coingecko', data.data_sources.coingecko);
}

// Update system status indicator
function updateSystemStatus(isActive) {
    const statusDot = document.getElementById('systemStatus');
    const statusText = document.getElementById('statusText');

    if (isActive) {
        statusDot.classList.add('active');
        statusText.textContent = 'LIVE';
    } else {
        statusDot.classList.remove('active');
        statusText.textContent = 'OFFLINE';
    }
}

// Update data source card
function updateDataSource(name, source) {
    const card = document.getElementById(`source${name}`);
    if (!card) return;

    const statusDot = card.querySelector('.source-status');
    const typeEl = card.querySelector('.source-type');
    const intervalEl = card.querySelector('.source-interval');
    const descEl = card.querySelector('.source-desc');

    if (source.status === 'active') {
        statusDot.classList.add('active');
    } else {
        statusDot.classList.remove('active');
    }

    typeEl.textContent = `Type: ${source.type}`;
    intervalEl.textContent = `Interval: ${source.interval}`;
    descEl.textContent = source.description;
}

// Fetch live predictions
async function fetchLivePredictions() {
    const response = await fetch('/api/predictions/live');
    const predictions = await response.json();

    Object.keys(predictions).forEach(symbol => {
        const pred = predictions[symbol];
        const symbolCode = symbol.split('-')[0];

        // Update chart
        if (charts[symbolCode]) {
            charts[symbolCode].data.datasets[0].data = [
                pred.down_prob * 100,
                pred.neutral_prob * 100,
                pred.up_prob * 100
            ];
            charts[symbolCode].update('none');
        }

        // Update prediction info
        updatePredictionInfo(symbolCode, pred);
    });
}

// Update prediction information
function updatePredictionInfo(symbol, pred) {
    const directionEl = document.getElementById(`direction${symbol}`);
    const confidenceEl = document.getElementById(`confidence${symbol}`);
    const tierEl = document.getElementById(`tier${symbol}`);
    const timeEl = document.getElementById(`time${symbol}`);

    // Direction
    directionEl.textContent = pred.direction.toUpperCase();
    directionEl.className = `prediction-direction ${pred.direction}`;

    // Confidence
    confidenceEl.textContent = (pred.confidence * 100).toFixed(1) + '%';

    // Tier
    tierEl.textContent = pred.tier.toUpperCase();
    tierEl.className = `prediction-tier tier-${pred.tier}`;

    // Timestamp
    const time = new Date(pred.timestamp);
    timeEl.textContent = time.toLocaleTimeString();
}

// Fetch signal statistics
async function fetchSignalStats() {
    const response = await fetch('/api/signals/stats/24');
    const stats = await response.json();

    // Update main stats
    document.getElementById('totalSignals').textContent = stats.total || 0;
    document.getElementById('avgConfidence').textContent =
        stats.total > 0 ? (stats.avg_confidence * 100).toFixed(1) + '%' : '--';
    document.getElementById('hourlyRate').textContent =
        stats.total > 0 ? stats.hourly_rate.toFixed(2) + '/hr' : '--';

    // Update breakdowns
    if (stats.total > 0) {
        updateBreakdown('statsBySymbol', stats.by_symbol, stats.total);
        updateBreakdown('statsByDirection', stats.by_direction, stats.total);
        updateBreakdown('statsByTier', stats.by_tier, stats.total);
    } else {
        document.getElementById('statsBySymbol').innerHTML = '<p class="no-data">No signals yet</p>';
        document.getElementById('statsByDirection').innerHTML = '<p class="no-data">No signals yet</p>';
        document.getElementById('statsByTier').innerHTML = '<p class="no-data">No signals yet</p>';
    }
}

// Update breakdown section
function updateBreakdown(elementId, data, total) {
    const container = document.getElementById(elementId);
    let html = '';

    Object.entries(data).forEach(([key, count]) => {
        const pct = ((count / total) * 100).toFixed(1);
        html += `
            <div class="breakdown-item">
                <span>${key}</span>
                <span>${count} (${pct}%)</span>
            </div>
        `;
    });

    container.innerHTML = html;
}

// Fetch recent signals
async function fetchRecentSignals() {
    const response = await fetch('/api/signals/recent/24');
    const signals = await response.json();

    const tbody = document.getElementById('recentSignalsTable');

    if (signals.length === 0) {
        tbody.innerHTML = '<tr><td colspan="5" class="no-data">No signals in the last 24 hours</td></tr>';
        return;
    }

    let html = '';
    signals.slice(0, 10).forEach(signal => {
        const time = new Date(signal.timestamp);
        const directionClass = signal.direction === 'long' ? 'signal-long' : 'signal-short';
        const tierClass = `tier-${signal.tier}`;

        html += `
            <tr>
                <td>${time.toLocaleString()}</td>
                <td>${signal.symbol}</td>
                <td class="${directionClass}">${signal.direction.toUpperCase()}</td>
                <td>${(signal.confidence * 100).toFixed(1)}%</td>
                <td class="${tierClass}">${signal.tier.toUpperCase()}</td>
            </tr>
        `;
    });

    tbody.innerHTML = html;
}

// Update last update timestamp
function updateLastUpdateTime() {
    const now = new Date();
    document.getElementById('lastUpdate').textContent = now.toLocaleTimeString();
}

// Helper: Format timestamp
function formatTimestamp(isoString) {
    const date = new Date(isoString);
    return date.toLocaleString('en-US', {
        month: 'short',
        day: 'numeric',
        hour: '2-digit',
        minute: '2-digit',
        second: '2-digit'
    });
}
