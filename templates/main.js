const historicalPlot = JSON.parse(historicalPlotData);
Plotly.newPlot('historicalPlot', historicalPlot.data, historicalPlot.layout);

const featureImportancePlot = JSON.parse(featureImportancePlotData);
Plotly.newPlot('featureImportancePlot', featureImportancePlot.data, featureImportancePlot.layout);

document.getElementById('predictionForm').addEventListener('submit', async function(e) {
    e.preventDefault();
    
    document.getElementById('customPrediction').style.display = 'none';
    document.getElementById('customPredictionValue').textContent = '$0.00';
    document.getElementById('customPredictionAccuracy').textContent = '0%';
    document.getElementById('marketCondition').textContent = '-';
    document.getElementById('priceChange').textContent = '-';
    
    const submitButton = this.querySelector('button[type="submit"]');
    const originalButtonText = submitButton.innerHTML;
    submitButton.innerHTML = '<i class="fas fa-spinner fa-spin me-2"></i>Calculating...';
    submitButton.disabled = true;
    
    try {
        const formData = {
            silver_price: document.getElementById('silverPrice').value,
            usd_index: document.getElementById('usdIndex').value,
            inflation_rate: document.getElementById('inflationRate').value,
            interest_rate: document.getElementById('interestRate').value,
            sp500_index: document.getElementById('sp500Index').value,
            oil_price: document.getElementById('oilPrice').value,
            geopolitical_risk: document.getElementById('geopoliticalRisk').value,
            selectedDate: document.getElementById('selectedDate').value
        };
        
        const response = await fetch('/api/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(formData)
        });
        
        if (!response.ok) {
            throw new Error('Prediction failed');
        }
        
        const result = await response.json();
        
        updatePredictionDisplay(result);
        
    } catch (error) {
        console.error('Error:', error);
        alert('Failed to get prediction. Please try again.');
    } finally {
        
        submitButton.innerHTML = originalButtonText;
        submitButton.disabled = false;
    }
});

function fetchMarketDataForDate(date) {
    let url = '/api/market-data';
    if (date) {
        url += `?date=${date}`;
    }
    fetch(url)
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                throw new Error(data.error);
            }
            if (data.gold_price) {
                document.querySelector('.price-card .display-4').textContent = 
                    '$' + data.gold_price.toFixed(2);
            }
            if (data.date) {
                document.querySelector('.date-display').textContent = 
                    'As of ' + data.date;
            }
        })
        .catch(error => {
            console.error('Error fetching market data:', error);
        });
}

setInterval(function() {
    const selectedDate = document.getElementById('selectedDate').value;
    fetchMarketDataForDate(selectedDate);
}, 300000);

const dateInput = document.getElementById('selectedDate');
dateInput.addEventListener('change', function() {
    fetchMarketDataForDate(this.value);
});

window.addEventListener('DOMContentLoaded', function() {
    const today = new Date().toISOString().split('T')[0];
    dateInput.value = today;
    fetchMarketDataForDate(today);
});

function updatePredictionDisplay(data) {
    const predictionDiv = document.getElementById('customPrediction');
    const predictionValue = document.getElementById('customPredictionValue');
    const predictionAccuracy = document.getElementById('customPredictionAccuracy');
    const marketCondition = document.getElementById('marketCondition');
    const priceChange = document.getElementById('priceChange');

    predictionValue.textContent = `$${data.predicted_price.toFixed(2)}`;
    predictionAccuracy.textContent = `${data.model_accuracy.toFixed(2)}%`;

    const isBullish = data.market_context.market_condition === 'Bullish';
    marketCondition.textContent = data.market_context.market_condition;
    marketCondition.className = isBullish ? 'market-condition-bullish' : 'market-condition-bearish';

    const priceDiff = data.market_context.price_difference;
    const percentChange = data.market_context.percentage_change;
    const isPositive = priceDiff > 0;
    priceChange.textContent = `${isPositive ? '+' : ''}$${priceDiff.toFixed(2)} (${isPositive ? '+' : ''}${percentChange.toFixed(2)}%)`;
    priceChange.className = isPositive ? 'price-change-positive' : 'price-change-negative';

    predictionDiv.style.display = 'block';
} 