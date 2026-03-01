# Cryptocurrency Price Predictor

A Python application with a Tkinter Graphical User Interface (GUI) that uses deep learning (LSTM neural networks) to predict future cryptocurrency prices based on historical data.

## Table of Contents
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [How it Works](#how-it-works)

## Features
- **Data Fetching:** Automatically downloads historical cryptocurrency data via Yahoo Finance.
- **Customizable Hyperparameters:** Users can define the number of prediction days, future days to forecast, training epochs, batch sizes, learning rate, and optimizer choices directly from the GUI.
- **Deep Learning Model:** Utilizes a multi-layer Long Short-Term Memory (LSTM) neural network built with TensorFlow/Keras.
- **Metrics Dashboard:** Calculates Evaluation metrics such as Mean Squared Error (MSE), Mean Absolute Error (MAE), Mean Absolute Percentage Error (MAPE), and overall Accuracy.
- **Data Visualization:** Uses Matplotlib plotting to display Training Data, Actual Prices, Historical Predictions, and Future Price forecasts.

## Project Structure
The source code is divided cleanly into three core components:

1. `data_utils.py`: Contains functions to fetch data over the network and preprocess/scale datasets to be ML-ready.
2. `model_utils.py`: Contains the logic for constructing the LSTM neural network architecture, compiling it, executing training loops, and calculating predictions/metrics.
3. `app.py`: The entry point for the application. It constructs the interactive `tkinter` interface and orchestrates data from the util files.

## Installation

Ensure you have Python 3.8+ installed on your system.

1. Clone or download this directory.
2. It's recommended to create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

Run the main application file from your terminal:

```bash
python app.py
```

1. Select your target **Cryptocurrency** (BTC, ETH, etc.) and Base **Currency** (USD, INR, etc.).
2. Pick a **Start Date** for the historical data. (End date is always today).
3. Adjust **Model Parameters** if desired, or leave them as default.
4. Click **Predict** and wait for the model to download data and train.
5. The future prices will populate in the table, and a new window will open displaying the graphical results.

## Built With
- **TensorFlow** & **Keras** - For building and training the LSTM model
- **scikit-learn** - For data normalization (`MinMaxScaler`) and calculating metrics
- **yfinance** - For pulling realtime financial data
- **matplotlib** - For graphing results
- **tkinter** & **tkcalendar** - For the interactive desktop GUI
