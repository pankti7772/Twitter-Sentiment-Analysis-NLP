// Dashboard JavaScript

let sentimentBarChart = null;
let keywordStackedChart = null;
let toxicityPieChart = null;

// Sentiment emoji mapping
const sentimentEmojis = {
    'Positive': '😊',
    'Negative': '😞',
    'Neutral': '😐',
    'Irrelevant': '🤷'
};

// Get base URL (works for both development and production)
function getBaseUrl() {
    return window.location.origin;
}

// Analyze sentiment
async function analyzeSentiment() {
    const textInput = document.getElementById('textInput');
    const text = textInput.value.trim();
    
    if (!text) {
        alert('Please enter some text to analyze.');
        return;
    }

    // Show loading, hide results
    document.getElementById('loading').classList.remove('hidden');
    document.getElementById('results').classList.add('hidden');

    try {
        const response = await fetch(`${getBaseUrl()}/predict`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ text: text })
        });

        if (!response.ok) {
            throw new Error('Analysis failed');
        }

        const data = await response.json();
        
        // Hide loading, show results
        document.getElementById('loading').classList.add('hidden');
        document.getElementById('results').classList.remove('hidden');
        
        // Display results
        displayResults(data, text);
        
    } catch (error) {
        console.error('Error:', error);
        document.getElementById('loading').classList.add('hidden');
        alert('Error analyzing sentiment. Please try again.');
    }
}

// Display results
function displayResults(data, originalText) {
    // Display sentiment badge
    const sentiment = data.label;
    const sentimentBadge = document.getElementById('sentimentBadge');
    const emoji = sentimentEmojis[sentiment] || '📊';
    
    let badgeClass = 'sentiment-neutral';
    if (sentiment === 'Positive') badgeClass = 'sentiment-positive';
    else if (sentiment === 'Negative') badgeClass = 'sentiment-negative';
    else if (sentiment === 'Irrelevant') badgeClass = 'sentiment-irrelevant';
    
    sentimentBadge.innerHTML = `<span class="sentiment-badge ${badgeClass}">${emoji} ${sentiment}</span>`;
    
    // Display toxicity badge (simplified - using negative sentiment as toxic indicator)
    const toxicityBadge = document.getElementById('toxicityBadge');
    const isToxic = sentiment === 'Negative' && data.scores[sentiment] > 0.7;
    const toxicityClass = isToxic ? 'toxicity-toxic' : 'toxicity-safe';
    const toxicityLabel = isToxic ? '⚠️ Toxic' : '✅ Safe';
    toxicityBadge.innerHTML = `<span class="toxicity-badge ${toxicityClass}">${toxicityLabel}</span>`;
    
    // Extract keywords from text (simple word extraction)
    const keywords = extractKeywords(originalText);
    
    // Display sentiment table
    displaySentimentTable(data.scores, keywords);
    
    // Create charts
    createSentimentBarChart(data.scores);
    createKeywordStackedChart(data.scores, keywords);
    createToxicityPieChart(data.scores);
}

// Extract keywords (simple implementation)
function extractKeywords(text) {
    const words = text.toLowerCase()
        .replace(/[^\w\s]/g, '')
        .split(/\s+/)
        .filter(word => word.length > 3)
        .slice(0, 5);
    return [...new Set(words)]; // Remove duplicates
}

// Display sentiment table
function displaySentimentTable(scores, keywords) {
    const tbody = document.getElementById('sentimentTableBody');
    tbody.innerHTML = '';
    
    // If no keywords, show overall sentiment
    if (keywords.length === 0) {
        keywords = ['Overall'];
    }
    
    keywords.forEach(keyword => {
        const row = document.createElement('tr');
        
        // Get dominant sentiment for this keyword (simplified - using overall scores)
        const dominantSentiment = Object.keys(scores).reduce((a, b) => 
            scores[a] > scores[b] ? a : b
        );
        const emoji = sentimentEmojis[dominantSentiment] || '📊';
        
        // Create breakdown bar
        const breakdownBar = createBreakdownBar(scores);
        
        row.innerHTML = `
            <td><strong>${keyword}</strong></td>
            <td>${emoji} ${dominantSentiment}</td>
            <td>${breakdownBar}</td>
        `;
        tbody.appendChild(row);
    });
}

// Create breakdown bar
function createBreakdownBar(scores) {
    const sortedScores = Object.entries(scores)
        .sort((a, b) => b[1] - a[1]);
    
    let barHTML = '<div class="sentiment-bar">';
    
    sortedScores.forEach(([label, score]) => {
        const percentage = (score * 100).toFixed(0);
        let barClass = 'bar-gray';
        
        if (label === 'Positive') barClass = 'bar-green';
        else if (label === 'Negative') barClass = 'bar-red';
        else if (label === 'Neutral') barClass = 'bar-orange';
        
        if (percentage > 5) { // Only show if significant
            barHTML += `<div class="bar-segment ${barClass}" style="width: ${percentage}%">${percentage}%</div>`;
        }
    });
    
    barHTML += '</div>';
    return barHTML;
}

// Create sentiment probability bar chart
function createSentimentBarChart(scores) {
    const ctx = document.getElementById('sentimentBarChart').getContext('2d');
    
    if (sentimentBarChart) {
        sentimentBarChart.destroy();
    }
    
    const labels = Object.keys(scores);
    const values = Object.values(scores).map(v => (v * 100).toFixed(1));
    const colors = labels.map(label => {
        if (label === 'Positive') return '#10B981';
        if (label === 'Negative') return '#EF4444';
        if (label === 'Neutral') return '#F97316';
        return '#6B7280';
    });
    
    sentimentBarChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [{
                label: 'Probability (%)',
                data: values,
                backgroundColor: colors,
                borderRadius: 8
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: false
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    max: 100,
                    ticks: {
                        callback: function(value) {
                            return value + '%';
                        }
                    }
                }
            }
        }
    });
}

// Create keyword stacked bar chart
function createKeywordStackedChart(scores, keywords) {
    const ctx = document.getElementById('keywordStackedChart').getContext('2d');
    
    if (keywordStackedChart) {
        keywordStackedChart.destroy();
    }
    
    // Use overall scores for each keyword (simplified)
    const labels = keywords.length > 0 ? keywords : ['Overall'];
    const sentimentLabels = Object.keys(scores);
    
    const datasets = sentimentLabels.map((sentiment, idx) => {
        let color = '#6B7280';
        if (sentiment === 'Positive') color = '#10B981';
        else if (sentiment === 'Negative') color = '#EF4444';
        else if (sentiment === 'Neutral') color = '#F97316';
        
        return {
            label: sentiment,
            data: labels.map(() => (scores[sentiment] * 100).toFixed(1)),
            backgroundColor: color
        };
    });
    
    keywordStackedChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: datasets
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: {
                    stacked: true
                },
                y: {
                    stacked: true,
                    beginAtZero: true,
                    max: 100,
                    ticks: {
                        callback: function(value) {
                            return value + '%';
                        }
                    }
                }
            }
        }
    });
}

// Create toxicity pie chart
function createToxicityPieChart(scores) {
    const ctx = document.getElementById('toxicityPieChart').getContext('2d');
    
    if (toxicityPieChart) {
        toxicityPieChart.destroy();
    }
    
    // Calculate toxic vs safe based on negative sentiment
    const toxicScore = scores['Negative'] || 0;
    const safeScore = 1 - toxicScore;
    
    toxicityPieChart = new Chart(ctx, {
        type: 'pie',
        data: {
            labels: ['Safe', 'Toxic'],
            datasets: [{
                data: [(safeScore * 100).toFixed(1), (toxicScore * 100).toFixed(1)],
                backgroundColor: ['#10B981', '#EF4444'],
                borderWidth: 0
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'bottom'
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            return context.label + ': ' + context.parsed + '%';
                        }
                    }
                }
            }
        }
    });
}

// Allow Enter key to trigger analysis
document.addEventListener('DOMContentLoaded', function() {
    const textInput = document.getElementById('textInput');
    textInput.addEventListener('keydown', function(e) {
        if (e.key === 'Enter' && e.ctrlKey) {
            analyzeSentiment();
        }
    });
});

