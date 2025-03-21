import pandas as pd
import numpy as np
import yfinance as yf
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from datetime import datetime, timedelta
import re
import math
import warnings
warnings.filterwarnings('ignore')

def validate_indian_stock_symbol(symbol):
    """Validate and format Indian stock symbol"""
    symbol = symbol.strip().upper()
    
    if not re.match(r'^[A-Z0-9]+\.(NS|BO)$', symbol):
        if not symbol.endswith(('.NS', '.BO')):
            symbol += '.NS'
    
    return symbol

def get_stock_data(ticker):
    """Fetch stock data from Yahoo Finance"""
    print(f"Fetching data from Yahoo Finance for {ticker}...")
    try:
        stock = yf.Ticker(ticker)
        yahoo_data = stock.history(period="2y")[['Close']].rename(columns={'Close': 'Yahoo_Close'})
        if yahoo_data.empty:
            print("Warning: No data received from Yahoo Finance")
            return pd.DataFrame()
            
        if len(yahoo_data) < 60:
            print("Warning: Insufficient historical data from Yahoo Finance")
            return pd.DataFrame()
        
        return yahoo_data
        
    except Exception as e:
        print(f"Warning: Error fetching Yahoo Finance data: {str(e)}")
        return pd.DataFrame()

def prepare_data(df):
    # Data Cleaning
    df = df.copy()
    
    required_columns = ['Yahoo_Close']
    if not all(col in df.columns for col in required_columns):
        print("Error: Missing required columns in the data")
        return None, None, None
    
    # Print initial data info
    print("\nInitial data info:")
    print(f"Shape: {df.shape}")
    print(f"Missing values:\n{df.isnull().sum()}")
    print(f"Data types:\n{df.dtypes}")
    
    df.fillna(method='ffill', inplace=True)
    df.fillna(method='bfill', inplace=True)
    
    df.dropna(inplace=True)
    
    df = df[df > 0]
    df = df.replace([np.inf, -np.inf], np.nan)
    
    for column in df.columns:
        df[column].fillna(df[column].mean(), inplace=True)
        df[column] = df[column].replace([np.inf, -np.inf], df[column].mean())
    
    if len(df) < 60:
        print("Error: Insufficient valid data points after cleaning")
        return None, None, None

    print("\nCleaned data info:")
    print(f"Shape: {df.shape}")
    print(f"Missing values:\n{df.isnull().sum()}")
    print(f"Data types:\n{df.dtypes}")
    print("\nData statistics:")
    print(df.describe())

    # Normalize Data
    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_data = scaler.fit_transform(df)

    # Validate scaled data
    if np.any(np.isnan(scaled_data)) or np.any(np.isinf(scaled_data)):
        print("\nError: Invalid values in scaled data")
        print("Checking for problematic values:")
        for i, col in enumerate(df.columns):
            nan_count = np.isnan(scaled_data[:, i]).sum()
            inf_count = np.isinf(scaled_data[:, i]).sum()
            if nan_count > 0 or inf_count > 0:
                print(f"Column {col}: {nan_count} NaN values, {inf_count} infinite values")
        return None, None, None

    # Prepare Training Data (LSTM)
    X_train, y_train = [], []
    window_size = 60

    for i in range(window_size, len(scaled_data)):
        X_train.append(scaled_data[i-window_size:i])
        y_train.append(scaled_data[i, 0])

    X_train = np.array(X_train)
    y_train = np.array(y_train)
    
    if len(X_train) == 0 or len(y_train) == 0:
        print("Error: No valid sequences created for training")
        return None, None, None
    
    print("\nFinal training data info:")
    print(f"X_train shape: {X_train.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"Number of sequences: {len(X_train)}")
        
    return X_train, y_train, scaler

def build_model(input_shape):
    model = Sequential([
        LSTM(100, return_sequences=True, input_shape=input_shape),
        Dropout(0.3),
        LSTM(100, return_sequences=False),
        Dropout(0.3),
        Dense(50),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
    return model

def predict_future_prices(model, last_sequence, scaler, days=7):
    predictions = []
    current_sequence = last_sequence.copy()
    
    for _ in range(days):
        # Get prediction
        pred = model.predict(current_sequence.reshape(1, current_sequence.shape[0], current_sequence.shape[1]))
        predictions.append(pred[0][0])
        
        current_sequence = np.roll(current_sequence, -1, axis=0)
        current_sequence[-1] = pred[0][0]  
    
    dummy_array = np.zeros((len(predictions), scaler.n_features_in_))
    dummy_array[:, 0] = np.array(predictions)  
    
    pred_original = scaler.inverse_transform(dummy_array)[:, 0]
    
    return pred_original

def main():
    print("\nIndian Stock Price Prediction")
    print("=============================")
    print("Enter a valid Indian stock symbol:")
    print("Examples:")
    print("- For NSE stocks: RELIANCE, TCS, INFY, HDFCBANK")
    print("- For BSE stocks: Add .BO suffix (e.g., 500325.BO)")
    print("- Default is NSE (.NS suffix will be added automatically)")
    
    ticker = input("\nEnter stock symbol: ").strip()
    ticker = validate_indian_stock_symbol(ticker)
    
    print(f"\nProcessing stock: {ticker}")
    df = get_stock_data(ticker)
    
    if df.empty:
        print("Error: Could not fetch data for the specified stock.")
        return
    
    print("\nPreparing data...")
    X_train, y_train, scaler = prepare_data(df)
    
    if X_train is None or y_train is None or scaler is None:
        print("Error: Failed to prepare data for training")
        return
    
    if len(X_train) < 60:  # Minimum required data points
        print("Error: Insufficient historical data for prediction")
        return
    
    # Split data
    train_size = int(len(X_train) * 0.8)
    X_test = X_train[train_size:]
    y_test = y_train[train_size:]
    X_train = X_train[:train_size]
    y_train = y_train[:train_size]
    
    print("\nBuilding and training model...")
    model = build_model((X_train.shape[1], X_train.shape[2]))
    
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )
    
    history = model.fit(
        X_train, y_train, 
        batch_size=16, 
        epochs=50, 
        validation_split=0.1,
        callbacks=[early_stopping]
    )
    
    # Make predictions
    print("\nMaking predictions...")
    y_pred = model.predict(X_test)
    
    # Validate predictions
    if np.any(np.isnan(y_pred)):
        print("Warning: NaN values detected in predictions. Attempting to fix...")
        y_pred = np.nan_to_num(y_pred, nan=np.nanmean(y_pred))
    
    # Create dummy arrays for inverse transform
    dummy_pred = np.zeros((len(y_pred), scaler.n_features_in_))
    dummy_test = np.zeros((len(y_test), scaler.n_features_in_))
    dummy_pred[:, 0] = y_pred.flatten()
    dummy_test[:, 0] = y_test
    
    # Inverse transform predictions and actual values
    try:
        y_pred_original = scaler.inverse_transform(dummy_pred)[:, 0]
        y_test_original = scaler.inverse_transform(dummy_test)[:, 0]
    except Exception as e:
        print(f"Error during inverse transform: {str(e)}")
        return
    
    # Debug information
    print(f"\nDebug Information:")
    print(f"Original test data shape: {y_test_original.shape}")
    print(f"Original prediction shape: {y_pred_original.shape}")
    print(f"Number of NaN values in test data: {np.isnan(y_test_original).sum()}")
    print(f"Number of NaN values in predictions: {np.isnan(y_pred_original).sum()}")
    print(f"Number of infinite values in test data: {np.isinf(y_test_original).sum()}")
    print(f"Number of infinite values in predictions: {np.isinf(y_pred_original).sum()}")
    
    # Handle NaN and infinite values more gracefully
    y_test_original = np.nan_to_num(y_test_original, nan=np.nanmean(y_test_original), posinf=np.nanmax(y_test_original), neginf=np.nanmin(y_test_original))
    y_pred_original = np.nan_to_num(y_pred_original, nan=np.nanmean(y_pred_original), posinf=np.nanmax(y_pred_original), neginf=np.nanmin(y_pred_original))
    
    # Ensure all values are finite
    mask = np.isfinite(y_test_original) & np.isfinite(y_pred_original)
    y_test_original = y_test_original[mask]
    y_pred_original = y_pred_original[mask]
    
    if len(y_test_original) == 0 or len(y_pred_original) == 0:
        print("\nError: No valid data points for evaluation after filtering")
        print("This might be due to:")
        print("1. Invalid predictions from the model")
        print("2. Issues with data scaling")
        print("3. Problems with the input data quality")
        return
    
    # Print data shapes for debugging
    print(f"\nFinal data shapes after filtering:")
    print(f"Test data shape: {y_test_original.shape}")
    print(f"Prediction shape: {y_pred_original.shape}")
    
    mse = mean_squared_error(y_test_original, y_pred_original)
    rmse = math.sqrt(mse)
    mae = mean_absolute_error(y_test_original, y_pred_original)
    avg_price = np.mean(y_test_original)
    accuracy = (1 - (rmse / avg_price)) * 100
    
    # Calculate additional accuracy metrics
    mape = np.mean(np.abs((y_test_original - y_pred_original) / y_test_original)) * 100
    r2_score = 1 - np.sum((y_test_original - y_pred_original) ** 2) / np.sum((y_test_original - np.mean(y_test_original)) ** 2)
    
    print("\n" + "="*50)
    print("MODEL PERFORMANCE METRICS")
    print("="*50)
    print(f"\nAccuracy Metrics:")
    print(f"1. Overall Accuracy: {accuracy:.2f}%")
    print(f"2. Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
    print(f"3. R-squared Score: {r2_score:.4f}")
    
    print(f"\nError Metrics:")
    print(f"1. Mean Squared Error (MSE): {mse:.2f}")
    print(f"2. Root Mean Squared Error (RMSE): {rmse:.2f}")
    print(f"3. Mean Absolute Error (MAE): {mae:.2f}")
    
    print(f"\nPrice Context:")
    print(f"Average Stock Price: ₹{avg_price:.2f}")
    print(f"Prediction Error Range: ±₹{rmse:.2f} (RMSE)")
    
    # Predict future prices
    print("\nPredicting future prices...")
    last_sequence = X_test[-1]
    future_prices = predict_future_prices(model, last_sequence, scaler)
    
    print("\nPredicted prices for the next 7 days:")
    current_date = datetime.now()
    for i, price in enumerate(future_prices):
        date = current_date + timedelta(days=i+1)
        print(f"{date.strftime('%Y-%m-%d')}: ₹{price:.2f}")
    
    # Save model
    model.save(f"stock_price_prediction_{ticker}.h5")
    print(f"\nModel saved as 'stock_price_prediction_{ticker}.h5'")

if __name__ == "__main__":
    main()
