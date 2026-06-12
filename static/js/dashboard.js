/**
 * NLP Sentiment Analysis Dashboard — JavaScript
 * ================================================
 * Handles API calls, chart rendering, animations,
 * sentiment prediction, and batch processing.
 */

// ── Chart.js Global Defaults ──────────────────────────────────────────────────
Chart.defaults.color = '#475569';
Chart.defaults.font.family = "'Inter', sans-serif";
Chart.defaults.font.size = 12;
Chart.defaults.plugins.legend.labels.padding = 16;
Chart.defaults.plugins.legend.labels.usePointStyle = true;
Chart.defaults.plugins.legend.labels.pointStyle = 'circle';

// ── Color Palette ─────────────────────────────────────────────────────────────
const COLORS = {
    positive: '#10b981',
    neutral: '#f59e0b',
    negative: '#f43f5e',
    positiveBg: 'rgba(16, 185, 129, 0.1)',
    neutralBg: 'rgba(245, 158, 11, 0.1)',
    negativeBg: 'rgba(244, 63, 94, 0.1)',
    indigo: '#4f46e5',
    purple: '#7c3aed',
    blue: '#2563eb',
    cyan: '#0891b2',
    emerald: '#10b981',
    rose: '#f43f5e',
    amber: '#f59e0b',
    barGradient: ['#2563eb', '#3b82f6', '#60a5fa', '#93c5fd', '#bfdbfe'],
};

// ── Utility Functions ─────────────────────────────────────────────────────────

/**
 * Animate counting up a number in a DOM element.
 */
function animateCounter(elementId, targetValue, duration = 1500, isFloat = false) {
    const element = document.getElementById(elementId);
    if (!element) return;

    const startTime = performance.now();
    const startValue = 0;

    function update(currentTime) {
        const elapsed = currentTime - startTime;
        const progress = Math.min(elapsed / duration, 1);

        // Ease-out cubic
        const eased = 1 - Math.pow(1 - progress, 3);
        const currentValue = startValue + (targetValue - startValue) * eased;

        if (isFloat) {
            element.textContent = currentValue.toFixed(2);
        } else {
            element.textContent = Math.round(currentValue).toLocaleString();
        }

        if (progress < 1) {
            requestAnimationFrame(update);
        }
    }

    requestAnimationFrame(update);
}

/**
 * Fetch JSON from API endpoint.
 */
async function fetchAPI(endpoint) {
    try {
        const response = await fetch(endpoint);
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        return await response.json();
    } catch (error) {
        console.error(`API Error [${endpoint}]:`, error);
        return null;
    }
}

// ── Dashboard Initialization ──────────────────────────────────────────────────

/**
 * Initialize the dashboard — load all data and render charts.
 */
async function initDashboard() {
    // Load all data in parallel
    const [stats, sentimentDist, ratingDist, modelPerf, keywords] = await Promise.all([
        fetchAPI('/api/statistics'),
        fetchAPI('/api/sentiment-distribution'),
        fetchAPI('/api/rating-distribution'),
        fetchAPI('/api/model-performance'),
        fetchAPI('/api/top-keywords'),
    ]);

    // Render KPI cards
    if (stats) renderKPIs(stats);

    // Render charts
    if (sentimentDist) renderSentimentPieChart(sentimentDist);
    if (ratingDist) renderRatingBarChart(ratingDist);

    // Render model performance
    if (modelPerf) renderModelPerformance(modelPerf);

    // Render keywords
    if (keywords) renderTopKeywords(keywords);
}

// ── KPI Cards ─────────────────────────────────────────────────────────────────

function renderKPIs(stats) {
    animateCounter('kpi-total', stats.total_reviews);
    animateCounter('kpi-positive', stats.total_positive);
    animateCounter('kpi-neutral', stats.total_neutral);
    animateCounter('kpi-negative', stats.total_negative);
    animateCounter('kpi-rating', stats.average_rating, 1500, true);
}

// ── Sentiment Pie Chart ───────────────────────────────────────────────────────

function renderSentimentPieChart(data) {
    const ctx = document.getElementById('sentimentPieChart');
    if (!ctx) return;

    // Map labels to colors
    const colorMap = {
        Positive: COLORS.positive,
        Neutral: COLORS.neutral,
        Negative: COLORS.negative,
    };
    const bgMap = {
        Positive: COLORS.positiveBg,
        Neutral: COLORS.neutralBg,
        Negative: COLORS.negativeBg,
    };

    const backgroundColors = data.labels.map(l => colorMap[l] || '#818cf8');
    const hoverColors = data.labels.map(l => bgMap[l] || 'rgba(99,102,241,0.15)');

    new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: data.labels,
            datasets: [{
                data: data.values,
                backgroundColor: backgroundColors,
                borderColor: 'rgba(10, 10, 26, 0.8)',
                borderWidth: 3,
                hoverBackgroundColor: backgroundColors,
                hoverBorderColor: backgroundColors,
                hoverOffset: 8,
            }],
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            cutout: '65%',
            plugins: {
                legend: {
                    position: 'bottom',
                    labels: {
                        padding: 20,
                        font: { size: 13, weight: '600' },
                    },
                },
                tooltip: {
                    backgroundColor: '#ffffff',
                    borderColor: '#cbd5e1',
                    borderWidth: 1,
                    titleColor: '#0f172a',
                    bodyColor: '#475569',
                    titleFont: { weight: '700' },
                    bodyFont: { size: 13 },
                    padding: 12,
                    cornerRadius: 8,
                    callbacks: {
                        label: (ctx) => {
                            const total = ctx.dataset.data.reduce((a, b) => a + b, 0);
                            const pct = ((ctx.parsed / total) * 100).toFixed(1);
                            return ` ${ctx.label}: ${ctx.parsed.toLocaleString()} (${pct}%)`;
                        },
                    },
                },
            },
            animation: {
                animateRotate: true,
                animateScale: true,
                duration: 1200,
            },
        },
    });
}

// ── Rating Bar Chart ──────────────────────────────────────────────────────────

function renderRatingBarChart(data) {
    const ctx = document.getElementById('ratingBarChart');
    if (!ctx) return;

    new Chart(ctx, {
        type: 'bar',
        data: {
            labels: data.labels,
            datasets: [{
                label: 'Number of Reviews',
                data: data.values,
                backgroundColor: COLORS.barGradient,
                borderColor: COLORS.barGradient.map(c => c),
                borderWidth: 1,
                borderRadius: 6,
                borderSkipped: false,
                maxBarThickness: 50,
            }],
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            indexAxis: 'x',
            plugins: {
                legend: { display: false },
                tooltip: {
                    backgroundColor: '#ffffff',
                    borderColor: '#cbd5e1',
                    borderWidth: 1,
                    titleColor: '#0f172a',
                    bodyColor: '#475569',
                    padding: 12,
                    cornerRadius: 8,
                    callbacks: {
                        label: (ctx) => ` Reviews: ${ctx.parsed.y.toLocaleString()}`,
                    },
                },
            },
            scales: {
                x: {
                    grid: { display: false },
                    ticks: { font: { weight: '600' } },
                },
                y: {
                    beginAtZero: true,
                    grid: { color: 'rgba(255, 255, 255, 0.04)' },
                    ticks: {
                        callback: (v) => v >= 1000 ? (v / 1000).toFixed(0) + 'k' : v,
                    },
                },
            },
            animation: {
                duration: 1000,
                easing: 'easeOutQuart',
            },
        },
    });
}

// ── Model Performance ─────────────────────────────────────────────────────────

function renderModelPerformance(data) {
    // Update metric cards
    const metrics = ['accuracy', 'precision', 'recall', 'f1_score'];
    const displayNames = { accuracy: 'accuracy', precision: 'precision', recall: 'recall', f1_score: 'f1' };

    metrics.forEach(metric => {
        const value = data[metric] || 0;
        const pct = (value * 100).toFixed(1);
        const displayName = displayNames[metric];

        const valueEl = document.getElementById(`metric-${displayName}`);
        if (valueEl) valueEl.textContent = pct + '%';

        const barEl = document.getElementById(`bar-${displayName}`);
        if (barEl) {
            setTimeout(() => { barEl.style.width = pct + '%'; }, 300);
        }
    });

    // Update best model name
    const bestModelEl = document.getElementById('best-model-name');
    if (bestModelEl) bestModelEl.textContent = data.best_model;

    // Render model comparison table
    if (data.all_models) renderModelTable(data.all_models);

    // Set confusion matrix image
    const cmImg = document.getElementById('confusion-matrix-img');
    if (cmImg && data.confusion_matrix_url) {
        cmImg.src = data.confusion_matrix_url;
    }
}

function renderModelTable(models) {
    const tbody = document.getElementById('model-table-body');
    if (!tbody) return;

    tbody.innerHTML = '';

    models.forEach(model => {
        const row = document.createElement('tr');
        if (model.is_best) row.classList.add('best-model');

        row.innerHTML = `
            <td>
                ${model.name}
                ${model.is_best ? '<span class="best-badge">Best</span>' : ''}
            </td>
            <td>${(model.accuracy * 100).toFixed(1)}%</td>
            <td>${(model.precision * 100).toFixed(1)}%</td>
            <td>${(model.recall * 100).toFixed(1)}%</td>
            <td>${(model.f1_score * 100).toFixed(1)}%</td>
        `;
        tbody.appendChild(row);
    });
}

// ── Top Keywords ──────────────────────────────────────────────────────────────

function renderTopKeywords(data) {
    if (data.positive) renderKeywordChart('positiveKeywordsChart', data.positive, COLORS.positive);
    if (data.negative) renderKeywordChart('negativeKeywordsChart', data.negative, COLORS.negative);
}

function renderKeywordChart(canvasId, keywords, color) {
    const ctx = document.getElementById(canvasId);
    if (!ctx) return;

    const labels = keywords.map(k => k.word).reverse();
    const values = keywords.map(k => k.count).reverse();

    new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [{
                label: 'Frequency',
                data: values,
                backgroundColor: color + '33',
                borderColor: color,
                borderWidth: 1,
                borderRadius: 4,
                borderSkipped: false,
            }],
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            indexAxis: 'y',
            plugins: {
                legend: { display: false },
                tooltip: {
                    backgroundColor: '#ffffff',
                    borderColor: '#cbd5e1',
                    borderWidth: 1,
                    titleColor: '#0f172a',
                    bodyColor: '#475569',
                    padding: 10,
                    cornerRadius: 8,
                },
            },
            scales: {
                x: {
                    beginAtZero: true,
                    grid: { color: 'rgba(255,255,255,0.03)' },
                    ticks: { font: { size: 11 } },
                },
                y: {
                    grid: { display: false },
                    ticks: { font: { size: 11, weight: '500' } },
                },
            },
            animation: {
                duration: 800,
                easing: 'easeOutQuart',
            },
        },
    });
}

// ── Sentiment Checker ─────────────────────────────────────────────────────────

/**
 * Switch between Single and Batch tabs.
 */
function switchTab(tabName) {
    // Update tab buttons
    document.querySelectorAll('.checker-tab').forEach(tab => {
        tab.classList.remove('active');
    });
    document.querySelector(`[data-tab="${tabName}"]`).classList.add('active');

    // Update tab content
    document.querySelectorAll('.tab-content').forEach(content => {
        content.classList.remove('active');
    });
    document.getElementById(`tab-${tabName}`).classList.add('active');
}

/**
 * Analyze single text sentiment.
 */
async function analyzeSentiment() {
    const textInput = document.getElementById('single-text-input');
    const resultContainer = document.getElementById('single-result');
    const spinner = document.getElementById('single-spinner');
    const analyzeBtn = document.getElementById('analyze-btn');

    const text = textInput.value.trim();
    if (!text) {
        alert('Please enter text to analyze.');
        return;
    }

    // Show loading
    analyzeBtn.disabled = true;
    analyzeBtn.innerHTML = '<span class="spinner-dot"></span><span class="spinner-dot"></span><span class="spinner-dot"></span>';
    spinner.classList.add('show');
    resultContainer.classList.remove('show');

    try {
        const response = await fetch('/api/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text: text }),
        });

        const data = await response.json();

        if (data.error) {
            alert('Error: ' + data.error);
            return;
        }

        renderSingleResult(data);
    } catch (error) {
        console.error('Prediction error:', error);
        alert('An error occurred. Please try again.');
    } finally {
        analyzeBtn.disabled = false;
        analyzeBtn.innerHTML = '🔍 Analyze Sentiment';
        spinner.classList.remove('show');
    }
}

/**
 * Render single prediction result.
 */
function renderSingleResult(data) {
    const container = document.getElementById('single-result');

    // Sentiment badge
    const sentimentClass = data.sentiment.toLowerCase();
    const emojis = { positive: '😊', neutral: '😐', negative: '😞' };
    const emoji = emojis[sentimentClass] || '🤔';

    document.getElementById('result-sentiment').className = `sentiment-badge ${sentimentClass}`;
    document.getElementById('result-sentiment').innerHTML = `${emoji} ${data.sentiment}`;

    // Confidence
    document.getElementById('result-confidence').textContent = data.confidence + '%';
    document.getElementById('result-confidence').className = `confidence-value text-${sentimentClass === 'positive' ? 'success' : sentimentClass === 'negative' ? 'danger' : 'warning'}`;

    // Cleaned text
    document.getElementById('result-cleaned-text').textContent = data.cleaned_text || '-';

    // Probability bars
    const probs = data.probabilities || {};
    const probOrder = ['Positive', 'Neutral', 'Negative'];
    const probClasses = { Positive: 'positive', Neutral: 'neutral', Negative: 'negative' };

    probOrder.forEach(label => {
        const value = probs[label] || 0;
        const barEl = document.getElementById(`prob-bar-${label.toLowerCase()}`);
        const valEl = document.getElementById(`prob-val-${label.toLowerCase()}`);

        if (barEl) barEl.style.width = value + '%';
        if (valEl) valEl.textContent = value + '%';
    });

    // Show result
    container.classList.add('show');
}

/**
 * Analyze batch reviews.
 */
async function analyzeBatch() {
    const textArea = document.getElementById('batch-text-input');
    const resultContainer = document.getElementById('batch-result');
    const spinner = document.getElementById('batch-spinner');
    const batchBtn = document.getElementById('batch-btn');

    const text = textArea.value.trim();
    if (!text) {
        alert('Please enter reviews to analyze (one per line).');
        return;
    }

    const reviews = text.split('\n').map(r => r.trim()).filter(r => r.length > 0);
    if (reviews.length === 0) {
        alert('No valid reviews found.');
        return;
    }

    // Show loading
    batchBtn.disabled = true;
    batchBtn.innerHTML = '<span class="spinner-dot"></span><span class="spinner-dot"></span><span class="spinner-dot"></span>';
    spinner.classList.add('show');
    resultContainer.classList.remove('show');

    try {
        const response = await fetch('/api/batch-predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ reviews: reviews }),
        });

        const data = await response.json();

        if (data.error) {
            alert('Error: ' + data.error);
            return;
        }

        renderBatchResults(data.results);
    } catch (error) {
        console.error('Batch prediction error:', error);
        alert('An error occurred. Please try again.');
    } finally {
        batchBtn.disabled = false;
        batchBtn.innerHTML = '📊 Analyze All Reviews';
        spinner.classList.remove('show');
    }
}

/**
 * Render batch prediction results in a table.
 */
function renderBatchResults(results) {
    const container = document.getElementById('batch-result');
    const tbody = document.getElementById('batch-table-body');
    const countEl = document.getElementById('batch-count');

    tbody.innerHTML = '';
    countEl.textContent = results.length;

    // Count sentiments
    let pos = 0, neu = 0, neg = 0;

    results.forEach((r, i) => {
        const sentClass = r.sentiment.toLowerCase();
        const emojis = { positive: '😊', neutral: '😐', negative: '😞' };

        if (sentClass === 'positive') pos++;
        else if (sentClass === 'neutral') neu++;
        else neg++;

        const row = document.createElement('tr');
        row.innerHTML = `
            <td>${i + 1}</td>
            <td title="${r.original_text}">${r.original_text.substring(0, 60)}${r.original_text.length > 60 ? '...' : ''}</td>
            <td><span class="badge-sm ${sentClass}">${emojis[sentClass] || ''} ${r.sentiment}</span></td>
            <td>${r.confidence}%</td>
        `;
        tbody.appendChild(row);
    });

    // Update summary
    document.getElementById('batch-positive').textContent = pos;
    document.getElementById('batch-neutral').textContent = neu;
    document.getElementById('batch-negative').textContent = neg;

    // Store results for CSV download
    window._batchResults = results;

    container.classList.add('show');
}

/**
 * Download batch results as CSV.
 */
async function downloadBatchCSV() {
    const textArea = document.getElementById('batch-text-input');
    const text = textArea.value.trim();
    const reviews = text.split('\n').map(r => r.trim()).filter(r => r.length > 0);

    if (reviews.length === 0) {
        alert('No reviews to download.');
        return;
    }

    try {
        const response = await fetch('/api/batch-predict-csv', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ reviews: reviews }),
        });

        const blob = await response.blob();
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = 'batch_predictions.csv';
        document.body.appendChild(a);
        a.click();
        window.URL.revokeObjectURL(url);
        a.remove();
    } catch (error) {
        console.error('CSV download error:', error);
        alert('Failed to download CSV.');
    }
}

// ── Auto-initialize on page load ──────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
    // If dashboard page, initialize charts
    if (document.getElementById('sentimentPieChart')) {
        initDashboard();
    }
});
