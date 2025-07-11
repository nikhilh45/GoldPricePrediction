# Gold Price Prediction using Random Forest

This application predicts gold prices using a Random Forest machine learning model. It uses historical gold price data and technical indicators to make predictions.

## Features

- Fetches 5 years of historical gold price data
- Uses technical indicators (SMA, RSI) as features
- Implements Random Forest regression for prediction
- Visualizes actual vs predicted prices
- Shows feature importance
- Provides next-day price prediction

## Requirements

- Python 3.8 or higher
- Required packages listed in `requirements.txt`

## Installation

1. Clone this repository
2. Install the required packages:
```bash
pip install -r requirements.txt
```

## Usage

Run the main script:
```bash
python gold_price_prediction.py
```

The script will:
1. Fetch historical gold price data
2. Prepare features and train the model
3. Display model performance metrics
4. Show a plot of actual vs predicted prices
5. Display feature importance
6. Provide a prediction for tomorrow's gold price
