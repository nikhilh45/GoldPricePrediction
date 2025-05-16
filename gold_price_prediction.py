import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

def fetch_gold_data():
    """Fetch historical gold price data from a local CSV file"""
    df = pd.read_csv('gold_price_sample.csv', parse_dates=['Date'])
    df.set_index('Date', inplace=True)
    print(f"Loaded data shape: {df.shape}")
    if df.empty:
        print("No data loaded. Please check the CSV file.")
        exit(1)
    return df

def prepare_features(df):
    """Prepare features for the model"""
    # Create technical indicators with smaller windows for small dataset
    df['SMA_3'] = df['Close'].rolling(window=3).mean()
    df['SMA_5'] = df['Close'].rolling(window=5).mean()
    df['RSI'] = calculate_rsi(df['Close'], period=5)
    
    # Create target variable (next day's closing price)
    df['Target'] = df['Close'].shift(-1)
    
    # Drop rows with NaN values
    df = df.dropna()
    
    # Select features
    features = ['Open', 'High', 'Low', 'Close', 'Volume', 'SMA_3', 'SMA_5', 'RSI']
    X = df[features]
    y = df['Target']
    
    return X, y

def calculate_rsi(prices, period=14):
    """Calculate Relative Strength Index"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def train_model(X, y):
    """Train Random Forest model"""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train Random Forest model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    # Calculate and print model performance
    train_score = model.score(X_train_scaled, y_train)
    test_score = model.score(X_test_scaled, y_test)
    print(f"Training R² score: {train_score:.4f}")
    print(f"Testing R² score: {test_score:.4f}")
    
    return model, scaler

def plot_predictions(y_true, y_pred, title):
    """Plot actual vs predicted values"""
    plt.figure(figsize=(12, 6))
    plt.plot(y_true.values, label='Actual', color='blue')
    plt.plot(y_pred, label='Predicted', color='red')
    plt.title(title)
    plt.xlabel('Time')
    plt.ylabel('Gold Price')
    plt.legend()
    plt.show()

def main():
    # Fetch and prepare data
    print("Fetching gold price data...")
    df = fetch_gold_data()
    
    print("Preparing features...")
    X, y = prepare_features(df)
    
    # Train model
    print("Training Random Forest model...")
    model, scaler = train_model(X, y)
    
    # Make predictions on test set
    X_test = X[-30:]  # Last 30 days
    X_test_scaled = scaler.transform(X_test)
    predictions = model.predict(X_test_scaled)
    
    # Plot results
    plot_predictions(y[-30:], predictions, "Gold Price Prediction (Last 30 Days)")
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    print("\nFeature Importance:")
    print(feature_importance)
    
    # Make prediction for tomorrow
    latest_data = X.iloc[-1:].copy()
    latest_data_scaled = scaler.transform(latest_data)
    tomorrow_prediction = model.predict(latest_data_scaled)[0]
    
    print(f"\nPredicted gold price for tomorrow: ${tomorrow_prediction:.2f}")

if __name__ == "__main__":
    main() 