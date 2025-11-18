// V6 Enhanced Model Dashboard JavaScript

// Chart instances
let charts = {};

// Initialize dashboard
document.addEventListener('DOMContentLoaded', function() {
    console.log('Dashboard initialized');
    initCharts();
    fetchAllData();

    // Auto-refresh every 5 seconds
    setInterval(fetchAllData, 1000);
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
            fetchLiveMarket(),
            fetchLivePredictions(),
            fetchSignalStats(),
            fetchRecentSignals(),
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

// Fetch live market prices
async function fetchLiveMarket() {
    try {
        const response = await fetch('/api/market/live');
        const data = await response.json();

        Object.keys(data).forEach(symbol => {
            const market = data[symbol];
            const symbolCode = symbol.split('-')[0];

            // Update price
            const priceEl = document.getElementById(`price${symbolCode}`);
            if (priceEl && market.price) {
                priceEl.textContent = `$${market.price.toLocaleString('en-US', {minimumFractionDigits: 2, maximumFractionDigits: 2})}`;
            }

            // Update change
            const changeEl = document.getElementById(`change${symbolCode}`);
            if (changeEl && market.change_pct !== undefined) {
                const changePct = market.change_pct;
                changeEl.textContent = `${changePct >= 0 ? '+' : ''}${changePct.toFixed(2)}%`;
                changeEl.className = 'market-change ' + (changePct >= 0 ? 'positive' : 'negative');
            }

            // Update high/low/volume
            if (market.high) {
                const highEl = document.getElementById(`high${symbolCode}`);
                if (highEl) highEl.textContent = `$${market.high.toFixed(2)}`;
            }
            if (market.low) {
                const lowEl = document.getElementById(`low${symbolCode}`);
                if (lowEl) lowEl.textContent = `$${market.low.toFixed(2)}`;
            }
            if (market.volume) {
                const volEl = document.getElementById(`vol${symbolCode}`);
                if (volEl) volEl.textContent = market.volume.toFixed(0);
            }
        });
    } catch (error) {
        console.error('Error fetching live market data:', error);
    }
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

        // Update confidence display under market price
        const confEl = document.getElementById(`conf${symbolCode}`);
        if (confEl) {
            confEl.textContent = `Confidence: ${(pred.confidence * 100).toFixed(1)}%`;
            // Color-code based on tier
            if (pred.tier === 'high') {
                confEl.style.color = '#10b981'; // Green
            } else if (pred.tier === 'medium') {
                confEl.style.color = '#f59e0b'; // Orange
            } else {
                confEl.style.color = '#6b7280'; // Gray
            }
        }
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

// ============================================================================
// V7 ULTIMATE FUNCTIONS
// ============================================================================

// Fetch V7 statistics
async function fetchV7Statistics() {
    try {
        const response = await fetch('/api/v7/statistics');
        const data = await response.json();

        // Update stats
        document.getElementById('v7TotalSignals').textContent = data.total_signals || 0;
        document.getElementById('v7AvgConfidence').textContent =
            data.avg_confidence > 0 ? (data.avg_confidence * 100).toFixed(1) + '%' : '--';

        // Latest signal
        if (data.latest_signal) {
            const latest = data.latest_signal;
            const dirClass = latest.direction === 'long' ? 'signal-long' : (latest.direction === 'short' ? 'signal-short' : 'signal-hold');
            const dir = latest.direction.toUpperCase();
            const conf = (latest.confidence * 100).toFixed(0);
            document.getElementById('v7LatestSignal').innerHTML =
                `<span class="${dirClass}">${latest.symbol} ${dir}</span> @ ${conf}%`;
        } else {
            document.getElementById('v7LatestSignal').textContent = 'No signals yet';
        }

        // By direction
        let directionHtml = '';
        for (const [dir, count] of Object.entries(data.by_direction)) {
            directionHtml += `<div><strong>${dir}:</strong> ${count}</div>`;
        }
        document.getElementById('v7ByDirection').innerHTML = directionHtml || 'No data';

        // By tier
        let tierHtml = '';
        for (const [tier, count] of Object.entries(data.by_tier)) {
            tierHtml += `<div><strong>${tier}:</strong> ${count}</div>`;
        }
        document.getElementById('v7ByTier').innerHTML = tierHtml || 'No data';

        // By symbol
        let symbolHtml = '';
        for (const [symbol, count] of Object.entries(data.by_symbol)) {
            symbolHtml += `<div><strong>${symbol}:</strong> ${count}</div>`;
        }
        document.getElementById('v7BySymbol').innerHTML = symbolHtml || 'No data';
    } catch (error) {
        console.error('Error fetching V7 statistics:', error);
    }
}

// Fetch V7 recent signals
async function fetchV7RecentSignals() {
    try {
        const response = await fetch('/api/v7/signals/recent/24');
        const signals = await response.json();

        const tbody = document.getElementById('v7SignalsTable');

        if (signals.length === 0) {
            tbody.innerHTML = '<tr><td colspan="6" class="no-data">No V7 signals in the last 24 hours</td></tr>';
            return;
        }

        let html = '';
        signals.slice(0, 10).forEach(signal => {
            const time = new Date(signal.timestamp);
            const dirClass = signal.direction === 'long' ? 'signal-long' : (signal.direction === 'short' ? 'signal-short' : 'signal-hold');
            const tierClass = `tier-${signal.tier}`;

            // Truncate reasoning for display
            const reasoning = signal.reasoning ?
                (signal.reasoning.length > 100 ? signal.reasoning.substring(0, 100) + '...' : signal.reasoning)
                : 'No reasoning provided';

            const dir = signal.direction.toUpperCase();
            const conf = (signal.confidence * 100).toFixed(1);

            html += `
                <tr>
                    <td>${time.toLocaleString()}</td>
                    <td>${signal.symbol}</td>
                    <td class="${dirClass}">${dir}</td>
                    <td>${conf}%</td>
                    <td class="${tierClass}">${signal.tier.toUpperCase()}</td>
                    <td class="v7-reasoning" title="${signal.reasoning || ''}">${reasoning}</td>
                </tr>
            `;
        });

        tbody.innerHTML = html;
    } catch (error) {
        console.error('Error fetching V7 signals:', error);
    }
}
