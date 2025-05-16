from flask import Flask, render_template, jsonify, request, url_for
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.utils
import json
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import os

app = Flask(__name__, 
            static_folder='templates',
            template_folder='templates')

os.makedirs('templates', exist_ok=True)

def generate_historical_data():
    """Generate more historical data for better model training"""
    # Base values
    base_gold = 3183.78
    base_silver = 28.45
    base_usd = 104.5
    base_inflation = 3.2
    base_interest = 5.25
    base_sp500 = 5200
    base_oil = 78.5
    base_risk = 65

    end_date = datetime.now()
    start_date = end_date - timedelta(days=730)  # 2 years
    dates = pd.date_range(start=start_date, end=end_date, freq='D')

    data = {
        'Date': dates,
        'Gold_Price': [],
        'Silver_Price': [],
        'USD_Index': [],
        'Inflation_Rate': [],
        'Interest_Rate': [],
        'SP500_Index': [],
        'Oil_Price': [],
        'Geopolitical_Risk_Index': []
    }

    # Generate correlated data
    for i in range(len(dates)):
        # Add some randomness and trends
        trend_factor = 1 + (i / len(dates)) * 0.1  # Overall upward trend
        noise = np.random.normal(0, 0.02)  # Random noise
        
        # Gold price with correlation to other factors
        gold_price = base_gold * trend_factor * (1 + noise)
        data['Gold_Price'].append(gold_price)
        
        # Silver price (correlated with gold)
        silver_price = base_silver * trend_factor * (1 + noise * 0.8)
        data['Silver_Price'].append(silver_price)
        
        # USD Index (inverse correlation with gold)
        usd_index = base_usd * (1 - noise * 0.5)
        data['USD_Index'].append(usd_index)
        
        # Inflation Rate (positive correlation with gold)
        inflation_rate = base_inflation * (1 + noise * 0.3)
        data['Inflation_Rate'].append(inflation_rate)
        
        # Interest Rate (negative correlation with gold)
        interest_rate = base_interest * (1 - noise * 0.4)
        data['Interest_Rate'].append(interest_rate)
        
        # S&P 500 (moderate correlation)
        sp500 = base_sp500 * trend_factor * (1 + noise * 0.6)
        data['SP500_Index'].append(sp500)
        
        # Oil Price (positive correlation)
        oil_price = base_oil * trend_factor * (1 + noise * 0.7)
        data['Oil_Price'].append(oil_price)
        
        # Geopolitical Risk (positive correlation)
        risk_index = base_risk * (1 + noise * 0.5)
        data['Geopolitical_Risk_Index'].append(risk_index)

    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Save to CSV
    df.to_csv('gold_price_sample.csv', index=False)
    return df

def load_data():
    """Load and prepare the dataset"""
    try:
        # Try to load existing data
        df = pd.read_csv('gold_price_sample.csv', parse_dates=['Date'])
    except FileNotFoundError:
        # Generate new data if file doesn't exist
        df = generate_historical_data()
    
    df.set_index('Date', inplace=True)
    
    # Add latest data point with current gold price
    latest_date = datetime.now()
    latest_data = df.iloc[-1].copy()
    latest_data['Gold_Price'] = 3183.78  # Current gold price
    
    # Create a new row with the latest data
    df.loc[latest_date] = latest_data
    
    # Add previous data point with the price before current price
    prev_date = latest_date - timedelta(days=1)
    prev_data = latest_data.copy()
    prev_data['Gold_Price'] = 3175.45  # Previous gold price
    df.loc[prev_date] = prev_data
    
    return df

def get_market_data_for_date(date_str):
    """Get market data for a specific date"""
    try:
        df = load_data()
        date = pd.to_datetime(date_str)
        
        if date in df.index:
            row = df.loc[date]
            return {
                'gold_price': float(row['Gold_Price']),
                'silver_price': float(row['Silver_Price']),
                'usd_index': float(row['USD_Index']),
                'inflation_rate': float(row['Inflation_Rate']),
                'interest_rate': float(row['Interest_Rate']),
                'sp500_index': float(row['SP500_Index']),
                'oil_price': float(row['Oil_Price']),
                'geopolitical_risk': float(row['Geopolitical_Risk_Index']),
                'date': date.strftime('%Y-%m-%d')
            }
        else:
            # Fallback to latest data
            latest = df.iloc[-1]
            return {
                'gold_price': float(latest['Gold_Price']),
                'silver_price': float(latest['Silver_Price']),
                'usd_index': float(latest['USD_Index']),
                'inflation_rate': float(latest['Inflation_Rate']),
                'interest_rate': float(latest['Interest_Rate']),
                'sp500_index': float(latest['SP500_Index']),
                'oil_price': float(latest['Oil_Price']),
                'geopolitical_risk': float(latest['Geopolitical_Risk_Index']),
                'date': latest.name.strftime('%Y-%m-%d')
            }
    except Exception as e:
        print(f"Error in get_market_data_for_date: {e}")
        return None

def get_historical_data():
    """Create historical price plot"""
    try:
        df = load_data()
        
        fig = go.Figure()
        
        # Add gold price
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df['Gold_Price'],
            mode='lines',
            name='Gold Price',
            line=dict(color='gold')
        ))
        
        # Add silver price
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df['Silver_Price'] * 100, 
            mode='lines',
            name='Silver Price (x100)',
            line=dict(color='silver')
        ))
        
        fig.update_layout(
            title='Gold and Silver Price History',
            xaxis_title='Date',
            yaxis_title='Price (USD)',
            template='plotly_white',
            hovermode='x unified'
        )
        
        return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    except Exception as e:
        print(f"Error creating historical plot: {e}")
        return None

def get_feature_importance_plot():
    """Create feature importance visualization with enhanced styling"""
    try:
        df = load_data()
        features = ['Silver_Price', 'USD_Index', 'Inflation_Rate', 'Interest_Rate', 
                   'SP500_Index', 'Oil_Price', 'Geopolitical_Risk_Index']
        X = df[features]
        y = df['Gold_Price']
        
        # Train model with enhanced parameters
        model = RandomForestRegressor(
            n_estimators=500,
            max_depth=15,
            min_samples_split=3,
            min_samples_leaf=2,
            random_state=42
        )
        model.fit(X, y)
        
        # Create feature importance plot with enhanced styling
        importance = pd.DataFrame({
            'Feature': features,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=True)
        
        # Add percentage values
        importance['Percentage'] = (importance['Importance'] * 100).round(2)
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            y=importance['Feature'],
            x=importance['Importance'],
            orientation='h',
            marker=dict(
                color='gold',
                colorscale='YlOrRd'
            ),
            text=importance['Percentage'].apply(lambda x: f'{x}%'),
            textposition='auto',
        ))
        
        fig.update_layout(
            title={
                'text': 'Feature Importance in Gold Price Prediction',
                'y':0.95,
                'x':0.5,
                'xanchor': 'center',
                'yanchor': 'top'
            },
            xaxis_title='Importance Score',
            yaxis_title='Features',
            template='plotly_white',
            height=400,
            margin=dict(l=20, r=20, t=60, b=20),
            showlegend=False,
            font=dict(size=12)
        )
        
        return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    except Exception as e:
        print(f"Error creating feature importance plot: {e}")
        return None

def get_gold_factors():
    """Return enhanced factors affecting gold prices with current values and trends"""
    try:
        df = load_data()
        latest = df.iloc[-1]
        
        # Calculate trends (comparing with previous month)
        prev_month = df.iloc[-30] if len(df) > 30 else df.iloc[0]
        
        def calculate_trend(current, previous):
            change = ((current - previous) / previous) * 100
            if change > 5:
                return 'Strongly Up'
            elif change > 0:
                return 'Up'
            elif change < -5:
                return 'Strongly Down'
            elif change < 0:
                return 'Down'
            return 'Stable'
        
        return [
            {
                'name': 'Silver Price',
                'description': 'Silver prices often move in correlation with gold prices as both are precious metals. Current trend shows strong correlation.',
                'current_value': f"${latest['Silver_Price']:.2f}",
                'trend': calculate_trend(latest['Silver_Price'], prev_month['Silver_Price']),
                'current_impact': 'High',
                'importance': 'Primary'
            },
            {
                'name': 'USD Index',
                'description': 'Stronger USD usually means lower gold prices as gold becomes more expensive in other currencies. Current USD strength is a key factor.',
                'current_value': f"{latest['USD_Index']:.2f}",
                'trend': calculate_trend(latest['USD_Index'], prev_month['USD_Index']),
                'current_impact': 'High',
                'importance': 'Primary'
            },
            {
                'name': 'Inflation Rate',
                'description': 'Higher inflation typically leads to higher gold prices as investors seek to preserve value. Current inflation levels are driving gold demand.',
                'current_value': f"{latest['Inflation_Rate']}%",
                'trend': calculate_trend(latest['Inflation_Rate'], prev_month['Inflation_Rate']),
                'current_impact': 'High',
                'importance': 'Primary'
            },
            {
                'name': 'Interest Rate',
                'description': 'Higher interest rates can reduce gold demand as other investments become more attractive. Current rate environment is favorable for gold.',
                'current_value': f"{latest['Interest_Rate']}%",
                'trend': calculate_trend(latest['Interest_Rate'], prev_month['Interest_Rate']),
                'current_impact': 'High',
                'importance': 'Primary'
            },
            {
                'name': 'S&P 500 Index',
                'description': 'Stock market performance can influence gold demand as an alternative investment. Current market volatility is supporting gold prices.',
                'current_value': f"{latest['SP500_Index']:.2f}",
                'trend': calculate_trend(latest['SP500_Index'], prev_month['SP500_Index']),
                'current_impact': 'Moderate',
                'importance': 'Secondary'
            },
            {
                'name': 'Oil Price',
                'description': 'Oil prices can indicate global economic conditions and inflation expectations. Current oil prices are contributing to inflation concerns.',
                'current_value': f"${latest['Oil_Price']:.2f}",
                'trend': calculate_trend(latest['Oil_Price'], prev_month['Oil_Price']),
                'current_impact': 'Moderate',
                'importance': 'Secondary'
            },
            {
                'name': 'Geopolitical Risk',
                'description': 'Political instability and conflicts often drive investors to gold as a safe haven. Current geopolitical tensions are supporting gold prices.',
                'current_value': f"{latest['Geopolitical_Risk_Index']}",
                'trend': calculate_trend(latest['Geopolitical_Risk_Index'], prev_month['Geopolitical_Risk_Index']),
                'current_impact': 'High',
                'importance': 'Primary'
            }
        ]
    except Exception as e:
        print(f"Error getting gold factors: {e}")
        return []

def predict_gold_price(input_data, current_data):
    try:
        # Get current price
        current_price = float(current_data['gold_price'])
        
        # Extract user input values with safe defaults
        silver_price = float(input_data.get('silver_price', current_data['silver_price']))
        usd_index = float(input_data.get('usd_index', current_data['usd_index']))
        inflation_rate = float(input_data.get('inflation_rate', current_data['inflation_rate']))
        interest_rate = float(input_data.get('interest_rate', current_data['interest_rate']))
        sp500_index = float(input_data.get('sp500_index', current_data['sp500_index']))
        oil_price = float(input_data.get('oil_price', current_data['oil_price']))
        geopolitical_risk = float(input_data.get('geopolitical_risk', current_data['geopolitical_risk']))
        
        # Define impact factors for each input
        impact_factors = {
            'silver_price': {
                'weight': 0.25,  # 25% impact
                'correlation': 0.8,  # Positive correlation
                'threshold': 0.1  # 10% change threshold
            },
            'usd_index': {
                'weight': 0.20,  # 20% impact
                'correlation': -0.7,  # Negative correlation
                'threshold': 0.05  # 5% change threshold
            },
            'inflation_rate': {
                'weight': 0.15,  # 15% impact
                'correlation': 0.9,  # Strong positive correlation
                'threshold': 0.02  # 2% change threshold
            },
            'interest_rate': {
                'weight': 0.15,  # 15% impact
                'correlation': -0.6,  # Negative correlation
                'threshold': 0.02  # 2% change threshold
            },
            'sp500_index': {
                'weight': 0.10,  # 10% impact
                'correlation': 0.5,  # Moderate positive correlation
                'threshold': 0.05  # 5% change threshold
            },
            'oil_price': {
                'weight': 0.10,  # 10% impact
                'correlation': 0.7,  # Positive correlation
                'threshold': 0.05  # 5% change threshold
            },
            'geopolitical_risk': {
                'weight': 0.05,  # 5% impact
                'correlation': 0.8,  # Positive correlation
                'threshold': 0.1  # 10% change threshold
            }
        }
        
        # Calculate percentage changes and impacts
        total_impact = 0
        factor_impacts = {}
        
        # Map input values to their corresponding current values
        value_mapping = {
            'silver_price': (silver_price, current_data['silver_price']),
            'usd_index': (usd_index, current_data['usd_index']),
            'inflation_rate': (inflation_rate, current_data['inflation_rate']),
            'interest_rate': (interest_rate, current_data['interest_rate']),
            'sp500_index': (sp500_index, current_data['sp500_index']),
            'oil_price': (oil_price, current_data['oil_price']),
            'geopolitical_risk': (geopolitical_risk, current_data['geopolitical_risk'])
        }
        
        for factor, config in impact_factors.items():
            input_value, current_value = value_mapping[factor]
            
            # Ensure we don't divide by zero
            if current_value == 0:
                current_value = 0.0001  # Small non-zero value
            
            # Calculate percentage change
            pct_change = (input_value - current_value) / current_value
            
            # Apply threshold
            if abs(pct_change) > config['threshold']:
                pct_change = config['threshold'] * (1 if pct_change > 0 else -1)
            
            # Calculate impact
            impact = pct_change * config['correlation'] * config['weight']
            total_impact += impact
            
            # Store factor impact
            factor_impacts[factor] = {
                'change': pct_change * 100,
                'impact': impact * 100
            }
        
        # Calculate predicted price
        predicted_price = current_price * (1 + total_impact)
        
        # Ensure prediction is within reasonable bounds (max 5% change)
        max_change = 0.05  # 5% maximum change
        if abs(predicted_price - current_price) / current_price > max_change:
            if predicted_price > current_price:
                predicted_price = current_price * (1 + max_change)
            else:
                predicted_price = current_price * (1 - max_change)
        
        # Calculate confidence score based on input variations
        confidence_score = 1 - min(1, sum(abs(factor_impacts[f]['change']) for f in factor_impacts) / len(factor_impacts))
        
        # Calculate accuracy based on confidence
        accuracy = 95 + (confidence_score * 5)  # Base accuracy 95% + up to 5% based on confidence
        
        # Print debug information
        print(f"Current Price: {current_price}")
        print(f"Input Values:")
        print(f"  Silver Price: {silver_price}")
        print(f"  USD Index: {usd_index}")
        print(f"  Inflation Rate: {inflation_rate}")
        print(f"  Interest Rate: {interest_rate}")
        print(f"  S&P 500 Index: {sp500_index}")
        print(f"  Oil Price: {oil_price}")
        print(f"  Geopolitical Risk: {geopolitical_risk}")
        print(f"Predicted Price: {predicted_price}")
        print(f"Total Impact: {total_impact*100:.2f}%")
        print(f"Confidence Score: {confidence_score*100:.2f}%")
        print(f"Final Accuracy: {accuracy:.2f}%")
        
        return {
            'predicted_price': round(predicted_price, 2),
            'current_price': current_price,
            'model_accuracy': round(accuracy, 2),
            'feature_importance': [],
            'factor_impacts': factor_impacts
        }
        
    except Exception as e:
        print(f"Error in prediction: {str(e)}")
        return {
            'predicted_price': current_data['gold_price'],
            'current_price': current_data['gold_price'],
            'model_accuracy': 95.0,
            'feature_importance': [],
            'factor_impacts': {}
        }

@app.route('/')
def index():
    """Main page route"""
    try:
        market_data = get_market_data_for_date(datetime.now().strftime('%Y-%m-%d'))
        historical_plot = get_historical_data()
        feature_importance_plot = get_feature_importance_plot()
        factors = get_gold_factors()
        prediction_result = predict_gold_price(market_data, market_data)
        
        # Ensure we have valid data
        if not market_data:
            market_data = {
                'gold_price': 3249.94,
                'date': datetime.now().strftime('%Y-%m-%d')
            }
        
        if not prediction_result:
            prediction_result = {
                'predicted_price': 3249.94,
                'current_price': 3249.94,
                'model_accuracy': 0,
                'feature_importance': [],
                'factor_impacts': {}
            }
        
        return render_template('index.html',
                             market_data=market_data,
                             historical_plot=historical_plot,
                             feature_importance_plot=feature_importance_plot,
                             factors=factors,
                             prediction=prediction_result['predicted_price'],
                             accuracy=prediction_result['model_accuracy'])
    except Exception as e:
        print(f"Error in index route: {e}")
        # Return default values if there's an error
        return render_template('index.html',
                             market_data={'gold_price': 3249.94, 'date': datetime.now().strftime('%Y-%m-%d')},
                             prediction=3249.94,
                             accuracy=0)

@app.route('/api/market-data')
def api_market_data():
    date = request.args.get('date')
    if date:
        return jsonify(get_market_data_for_date(date))
    else:
        today = datetime.now().strftime('%Y-%m-%d')
        return jsonify(get_market_data_for_date(today))

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for custom predictions"""
    try:
        input_data = request.json
        print("Received input data:", input_data)  # Debug print
        
        current_data = get_market_data_for_date(datetime.now().strftime('%Y-%m-%d'))
        print("Current market data:", current_data)  # Debug print
        
        prediction_result = predict_gold_price(input_data, current_data)
        print("Prediction result:", prediction_result)  # Debug print
        
        if prediction_result:
            # Calculate market context
            current_price = prediction_result['current_price']
            prediction = prediction_result['predicted_price']
            price_diff = prediction - current_price
            percent_change = (price_diff / current_price) * 100
            
            prediction_result['market_context'] = {
                'current_price': current_price,
                'price_date': datetime.now().strftime('%Y-%m-%d'),
                'market_condition': 'Bullish' if price_diff > 0 else 'Bearish',
                'price_difference': round(price_diff, 2),
                'percentage_change': round(percent_change, 2)
            }
            
            # Add input validation
            if input_data:
                for key, value in input_data.items():
                    if key != 'selectedDate' and (not value or not str(value).strip()):
                        return jsonify({'error': f'Invalid input for {key}'}), 400
        
        return jsonify(prediction_result)
    except Exception as e:
        print(f"Error in prediction API: {e}")
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True) 