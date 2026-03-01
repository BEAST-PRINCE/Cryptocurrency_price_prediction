import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.optimizers import Adam, RMSprop, SGD, Adagrad
from sklearn.metrics import mean_squared_error, mean_absolute_error

def get_optimizer(optimizer_name, learning_rate):
    """Return the specified Keras optimizer with the given learning rate."""
    optimizers = {
        "Adam": Adam,
        "RMSprop": RMSprop,
        "SGD": SGD,
        "Adagrad": Adagrad
    }
    if optimizer_name in optimizers:
        return optimizers[optimizer_name](learning_rate=learning_rate)
    return Adam(learning_rate=learning_rate)

def build_and_train_model(x_train, y_train, epochs, batch_size, optimizer_name, learning_rate, loss_function):
    """Build and train the LSTM sequential model."""
    model = Sequential()
    
    # Architecture based on the active Predictor code
    model.add(LSTM(units=100, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(Dropout(0.2))
    
    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(0.2))
    
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dropout(0.2))
    
    model.add(Dense(units=1))

    optimizer_instance = get_optimizer(optimizer_name, learning_rate)
    model.compile(optimizer=optimizer_instance, loss=loss_function)
    
    # Train the model
    model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1)
    
    return model

def evaluate_model(y_test, predicted_prices):
    """Calculate and return evaluation metrics."""
    if len(predicted_prices) == 0 or len(y_test) < len(predicted_prices):
        return 0, 0, 0, 0
        
    y_test_subset = y_test[:len(predicted_prices)]
    pred_flattened = predicted_prices.flatten()
    
    mse = mean_squared_error(y_test_subset, pred_flattened)
    mae = mean_absolute_error(y_test_subset, pred_flattened)
    
    accuracy = 100 - (mae / np.mean(y_test_subset) * 100)
    accuracy = float(max(0, min(accuracy, 100)))
    
    epsilon = 1e-10
    mape = np.mean(np.abs((y_test_subset - pred_flattened) / (y_test_subset + epsilon))) * 100
    mape = float(mape)
    
    return mse, mae, accuracy, mape

def predict_future(model, last_sequence, days_to_predict, scaler):
    """Predict prices for the next `days_to_predict`."""
    future_predictions = []
    current_sequence = last_sequence.copy()

    for _ in range(days_to_predict):
        current_sequence_reshaped = np.reshape(current_sequence, (1, current_sequence.shape[0], 1))
        next_day_pred = model.predict(current_sequence_reshaped, verbose=0)
        future_predictions.append(next_day_pred[0, 0])
        current_sequence = np.append(current_sequence[1:], next_day_pred[0, 0])
        
    future_prices = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))
    return future_prices.flatten()
