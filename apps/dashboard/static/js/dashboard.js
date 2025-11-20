// V7 Trading Signals Dashboard JavaScript

// Chart instances
let charts = {
    timeline: null,
    distribution: null,
    confidence: null
};

// Initialize dashboard
document.addEventListener('DOMContentLoaded', function() {
    console.log('V7 Dashboard initialized');
    initCharts();
    fetchAllData();

    // Auto-refresh every 2 seconds
    setInterval(fetchAllData, 2000);
});

// Fetch all dashboard data
async function fetchAllData() {
    try {
        await Promise.all([
            fetchSystemStatus(),
            fetchV7Statistics(),
            fetchV7RecentSignals(),
            fetchV7ChartData(),
            fetchV7Costs(),
            fetchV7Performance(),
            fetchLivePrices()
        ]);
        updateLastUpdateTime();
    } catch (error) {
        console.error('Error fetching data:', error);
        updateSystemStatus(false);
    }
}

// Fetch live market prices
async function fetchLivePrices() {
    try {
        const symbols = ['BTC-USD', 'ETH-USD', 'SOL-USD'];
        const prices = {};

        for (const symbol of symbols) {
            const response = await fetch(`https://api.coinbase.com/v2/prices/${symbol}/spot`);
            const data = await response.json();
            prices[symbol] = parseFloat(data.data.amount);
        }

        // Update live price display
        document.getElementById('liveBTCPrice').textContent = prices['BTC-USD'].toLocaleString('en-US', {minimumFractionDigits: 2, maximumFractionDigits: 2});
        document.getElementById('liveETHPrice').textContent = prices['ETH-USD'].toLocaleString('en-US', {minimumFractionDigits: 2, maximumFractionDigits: 2});
        document.getElementById('liveSOLPrice').textContent = prices['SOL-USD'].toLocaleString('en-US', {minimumFractionDigits: 2, maximumFractionDigits: 2});

        const now = new Date();
        document.getElementById('priceUpdateTime').textContent = 'Updated ' + now.toLocaleTimeString('en-US', {hour: '2-digit', minute: '2-digit', second: '2-digit', hour12: false});

    } catch (error) {
        console.error('Error fetching live prices:', error);
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
        const allSignals = await response.json();

        const tableBody = document.getElementById('v7SignalsTable');

        if (!allSignals || allSignals.length === 0) {
            tableBody.innerHTML = '<tr><td colspan="9" class="no-data">No V7 signals in the last 24 hours</td></tr>';
            return;
        }

        // Update total signals counter
        document.getElementById('totalV7Signals').textContent = allSignals.length;

        // Update DeepSeek Analysis Box with most recent signal
        if (allSignals.length > 0) {
            updateDeepSeekAnalysis(allSignals[0]);
        }

        // Limit to most recent 20 signals for readability
        const signals = allSignals.slice(0, 20);

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

    // Check if date is valid
    if (isNaN(date.getTime())) {
        console.error('Invalid timestamp:', timestamp);
        return 'Invalid';
    }

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

// Update DeepSeek Analysis Box
function updateDeepSeekAnalysis(signal) {
    const analysisBox = document.getElementById('deepseekAnalysis');
    const thinkingElement = document.getElementById('deepseekThinking');
    const symbolElement = document.getElementById('analysisSymbol');
    const confidenceElement = document.getElementById('analysisConfidence');
    const directionElement = document.getElementById('analysisDirection');
    const timestampElement = document.getElementById('analysisTimestamp');

    // Only show if signal has reasoning (DeepSeek analysis)
    const reasoning = signal.reasoning || signal.notes;
    if (reasoning && reasoning.length > 10) {
        analysisBox.style.display = 'block';

        // Extract reasoning text from JSON if needed
        let reasoningText = reasoning;
        try {
            const parsed = JSON.parse(reasoning);
            reasoningText = parsed.reasoning || reasoning;
        } catch (e) {
            // Not JSON, use as-is
        }

        thinkingElement.textContent = reasoningText;
        symbolElement.textContent = signal.symbol;
        confidenceElement.textContent = (signal.confidence * 100).toFixed(1) + '%';

        const direction = signal.direction === 'long' ? '🟢 BUY' :
                         signal.direction === 'short' ? '🔴 SELL' : '🟡 HOLD';
        directionElement.textContent = direction;

        const timestamp = new Date(signal.timestamp);
        timestampElement.textContent = timestamp.toLocaleTimeString('en-US', {
            hour: '2-digit',
            minute: '2-digit',
            hour12: false
        });
    } else {
        analysisBox.style.display = 'none';
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

// ==================== CHART FUNCTIONS ====================

// Initialize all charts
function initCharts() {
    const chartOptions = {
        responsive: true,
        maintainAspectRatio: true,
        aspectRatio: 2,
        plugins: {
            legend: {
                labels: {
                    color: '#e6e6e6',
                    font: { size: 12 }
                }
            }
        },
        scales: {
            x: {
                ticks: { color: '#999' },
                grid: { color: 'rgba(255,255,255,0.1)' }
            },
            y: {
                ticks: { color: '#999' },
                grid: { color: 'rgba(255,255,255,0.1)' }
            }
        }
    };

    // Signal Timeline Chart (Bar chart)
    const timelineCtx = document.getElementById('signalTimelineChart');
    if (timelineCtx) {
        charts.timeline = new Chart(timelineCtx, {
            type: 'bar',
            data: {
                labels: [],
                datasets: [
                    {
                        label: 'BUY',
                        data: [],
                        backgroundColor: 'rgba(16, 185, 129, 0.7)',
                        borderColor: 'rgba(16, 185, 129, 1)',
                        borderWidth: 1
                    },
                    {
                        label: 'SELL',
                        data: [],
                        backgroundColor: 'rgba(239, 68, 68, 0.7)',
                        borderColor: 'rgba(239, 68, 68, 1)',
                        borderWidth: 1
                    },
                    {
                        label: 'HOLD',
                        data: [],
                        backgroundColor: 'rgba(245, 158, 11, 0.7)',
                        borderColor: 'rgba(245, 158, 11, 1)',
                        borderWidth: 1
                    }
                ]
            },
            options: {
                ...chartOptions,
                scales: {
                    ...chartOptions.scales,
                    x: { ...chartOptions.scales.x, stacked: true },
                    y: { ...chartOptions.scales.y, stacked: true, beginAtZero: true }
                }
            }
        });
    }

    // Signal Distribution Chart (Doughnut chart)
    const distributionCtx = document.getElementById('signalDistributionChart');
    if (distributionCtx) {
        charts.distribution = new Chart(distributionCtx, {
            type: 'doughnut',
            data: {
                labels: ['BUY', 'SELL', 'HOLD'],
                datasets: [{
                    data: [0, 0, 0],
                    backgroundColor: [
                        'rgba(16, 185, 129, 0.8)',
                        'rgba(239, 68, 68, 0.8)',
                        'rgba(245, 158, 11, 0.8)'
                    ],
                    borderColor: [
                        'rgba(16, 185, 129, 1)',
                        'rgba(239, 68, 68, 1)',
                        'rgba(245, 158, 11, 1)'
                    ],
                    borderWidth: 2
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: true,
                aspectRatio: 1.5,
                plugins: {
                    legend: {
                        position: 'bottom',
                        labels: {
                            color: '#e6e6e6',
                            font: { size: 12 }
                        }
                    }
                }
            }
        });
    }

    // Confidence Trend Chart (Line chart)
    const confidenceCtx = document.getElementById('confidenceTrendChart');
    if (confidenceCtx) {
        charts.confidence = new Chart(confidenceCtx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: 'Average Confidence',
                    data: [],
                    borderColor: 'rgba(99, 102, 241, 1)',
                    backgroundColor: 'rgba(99, 102, 241, 0.2)',
                    borderWidth: 2,
                    fill: true,
                    tension: 0.4
                }]
            },
            options: {
                ...chartOptions,
                scales: {
                    ...chartOptions.scales,
                    y: { 
                        ...chartOptions.scales.y,
                        beginAtZero: false,
                        min: 0,
                        max: 1,
                        ticks: {
                            color: '#999',
                            callback: function(value) {
                                return (value * 100).toFixed(0) + '%';
                            }
                        }
                    }
                }
            }
        });
    }
}

// Fetch chart data
async function fetchV7ChartData() {
    try {
        // Fetch timeseries data for last 24 hours
        const response = await fetch('/api/v7/signals/timeseries/24');
        const data = await response.json();

        if (!data.timeseries || data.timeseries.length === 0) {
            console.log('No chart data available');
            return;
        }

        // Update timeline chart
        if (charts.timeline) {
            const labels = data.timeseries.map(d => {
                const date = new Date(d.timestamp);
                return date.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit', hour12: false });
            });
            const buyData = data.timeseries.map(d => d.long_count || 0);
            const sellData = data.timeseries.map(d => d.short_count || 0);
            const holdData = data.timeseries.map(d => d.hold_count || 0);

            charts.timeline.data.labels = labels;
            charts.timeline.data.datasets[0].data = buyData;
            charts.timeline.data.datasets[1].data = sellData;
            charts.timeline.data.datasets[2].data = holdData;
            charts.timeline.update('none'); // Update without animation
        }

        // Update distribution chart (aggregate)
        if (charts.distribution) {
            const totalBuy = data.timeseries.reduce((sum, d) => sum + (d.long_count || 0), 0);
            const totalSell = data.timeseries.reduce((sum, d) => sum + (d.short_count || 0), 0);
            const totalHold = data.timeseries.reduce((sum, d) => sum + (d.hold_count || 0), 0);

            charts.distribution.data.datasets[0].data = [totalBuy, totalSell, totalHold];
            charts.distribution.update('none');
        }

        // Update confidence trend chart
        if (charts.confidence) {
            const labels = data.timeseries.map(d => {
                const date = new Date(d.timestamp);
                return date.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit', hour12: false });
            });
            const confidenceData = data.timeseries.map(d => d.avg_confidence || 0);

            charts.confidence.data.labels = labels;
            charts.confidence.data.datasets[0].data = confidenceData;
            charts.confidence.update('none');
        }

    } catch (error) {
        console.error('Error fetching chart data:', error);
    }
}

// ==================== PERFORMANCE MONITORING ====================

// Fetch V7 cost tracking data
async function fetchV7Costs() {
    try {
        const response = await fetch('/api/v7/costs');
        const data = await response.json();

        // Today's cost
        document.getElementById('costToday').textContent = '$' + (data.today?.cost || 0).toFixed(4);
        document.getElementById('costTodayPercent').textContent =
            (data.today?.percent_used || 0).toFixed(2) + '% of daily budget';
        document.getElementById('costTodayRemaining').textContent = '$' + (data.today?.remaining || 0).toFixed(2);

        // Monthly cost
        document.getElementById('costMonth').textContent = '$' + (data.month?.cost || 0).toFixed(4);
        document.getElementById('costMonthPercent').textContent =
            (data.month?.percent_used || 0).toFixed(2) + '% of monthly budget';
        document.getElementById('costMonthRemaining').textContent = '$' + (data.month?.remaining || 0).toFixed(2);

        // Average and total
        document.getElementById('avgCostPerSignal').textContent = '$' + (data.avg_cost_per_signal || 0).toFixed(6);
        document.getElementById('totalSignalsForCost').textContent = data.total_signals || 0;
        document.getElementById('totalCost').textContent = '$' + (data.total_cost || 0).toFixed(4);

        // Color code based on budget usage
        const todayCard = document.getElementById('costToday').parentElement;
        const monthCard = document.getElementById('costMonth').parentElement;

        if (data.today?.percent_used >= 95) {
            todayCard.style.borderLeft = '4px solid #ef4444';
        } else if (data.today?.percent_used >= 80) {
            todayCard.style.borderLeft = '4px solid #f59e0b';
        } else {
            todayCard.style.borderLeft = '4px solid #10b981';
        }

        if (data.month?.percent_used >= 95) {
            monthCard.style.borderLeft = '4px solid #ef4444';
        } else if (data.month?.percent_used >= 80) {
            monthCard.style.borderLeft = '4px solid #f59e0b';
        } else {
            monthCard.style.borderLeft = '4px solid #10b981';
        }

    } catch (error) {
        console.error('Error fetching V7 costs:', error);
    }
}

// Fetch V7 performance data
async function fetchV7Performance() {
    try {
        const response = await fetch('/api/v7/performance');
        const data = await response.json();

        // Update performance stats
        document.getElementById('perfTotalTrades').textContent = data.total_trades || 0;
        document.getElementById('perfWins').textContent = data.wins || 0;
        document.getElementById('perfLosses').textContent = data.losses || 0;

        if (data.total_trades > 0) {
            document.getElementById('perfWinRate').textContent = data.win_rate.toFixed(1) + '%';
            document.getElementById('perfTotalPnl').textContent = '$' + data.total_pnl.toFixed(2);
            document.getElementById('perfAvgPnl').textContent = '$' + data.avg_pnl_per_trade.toFixed(2);

            // Color code P&L
            const pnlElement = document.getElementById('perfTotalPnl');
            if (data.total_pnl > 0) {
                pnlElement.style.color = '#10b981';
            } else if (data.total_pnl < 0) {
                pnlElement.style.color = '#ef4444';
            }

            const avgPnlElement = document.getElementById('perfAvgPnl');
            if (data.avg_pnl_per_trade > 0) {
                avgPnlElement.style.color = '#10b981';
            } else if (data.avg_pnl_per_trade < 0) {
                avgPnlElement.style.color = '#ef4444';
            }

            // Color code win rate
            const winRateElement = document.getElementById('perfWinRate');
            if (data.win_rate >= 70) {
                winRateElement.style.color = '#10b981';
            } else if (data.win_rate >= 60) {
                winRateElement.style.color = '#f59e0b';
            } else {
                winRateElement.style.color = '#ef4444';
            }
        } else {
            // No trades yet
            document.getElementById('perfWinRate').textContent = 'No trades tracked yet';
            document.getElementById('perfTotalPnl').textContent = '--';
            document.getElementById('perfAvgPnl').textContent = '--';
        }

    } catch (error) {
        console.error('Error fetching V7 performance:', error);
    }
}
