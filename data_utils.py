import yfinance as yf
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def fetch_crypto_data(crypto, currency, start_date, end_date):
    """Fetch cryptocurrency data using yfinance trying various ticker patterns."""
    ticker_patterns = [
        f"{crypto}-{currency}",    # Standard format
        f"{crypto}{currency}=X",   # Alternative format for some pairs
        f"{crypto}-{currency}=X"   # Another alternative
    ]
    
    data = None
    for ticker in ticker_patterns:
        try:
            data = yf.download(ticker, start=start_date, end=end_date)
            if not data.empty:
                break
        except Exception:
            continue
            
    if data is None or data.empty:
        raise Exception(f"No data available for {crypto}-{currency}. Try a different pair.")
        
    return data

def prepare_training_data(data, prediction_days):
    """Scale data and prepare x_train, y_train sequences."""
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))

    x_train, y_train = [], []
    for i in range(prediction_days, len(scaled_data)):
        x_train.append(scaled_data[i - prediction_days:i, 0])
        y_train.append(scaled_data[i, 0])

    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    return x_train, y_train, scaled_data, scaler

def prepare_testing_data(scaled_data, actual_prices, prediction_days, train_len):
    """Prepare x_test sequences and y_test values from test data."""
    test_data = scaled_data[train_len - prediction_days:]
    
    x_test = []
    y_test = actual_prices[train_len:]
    
    for i in range(prediction_days, len(test_data)):
        x_test.append(test_data[i - prediction_days:i, 0])
    
    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
    
    return x_test, y_test
