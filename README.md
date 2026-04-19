# 🚀 Cryptocurrency Price Predictor

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.0%2B-orange?logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A sophisticated deep learning application built with **LSTM (Long Short-Term Memory)** neural networks to forecast future cryptocurrency prices. Featuring a clean Tkinter-based GUI, real-time data integration, and highly customizable hyperparameters.

![Project Preview](project_preview_mockup_1776577061311.png)

## ✨ Key Features

- 🔄 **Real-time Data:** Seamlessly fetches the latest historical data from **Yahoo Finance** via the `yfinance` API.
- 🧠 **Deep Learning Core:** Utilizes a multi-layered **LSTM network** designed for time-series forecasting.
- ⚙️ **Customizable Parameters:** Fine-tune your model directly from the UI:
    - Prediction lag (lookback period)
    - Future forecast horizon
    - Training Epochs & Batch Size
    - Optimizers (Adam, RMSprop, SGD, Adagrad)
    - Learning Rates & Loss Functions
- 📊 **Visual Analytics:** Interactive Matplotlib graphs showing training trends, actual vs. predicted prices, and future projections.
- 📈 **Performance Metrics:** Real-time calculation of **Accuracy**, **MSE**, **MAE**, and **MAPE** to evaluate model reliability.

## 🛠️ Built With

*   **Backend:** Python 3.8+
*   **Neural Networks:** TensorFlow / Keras
*   **Data Processing:** NumPy, Pandas, Scikit-learn (MinMaxScaler)
*   **Finance API:** yfinance
*   **GUI:** Tkinter, tkcalendar
*   **Visualization:** Matplotlib

## 📂 Project Structure

```text
Modular_Predictor/
├── app.py              # Main GUI application & entry point
├── data_utils.py       # Data fetching and preprocessing logic
├── model_utils.py      # LSTM architecture and training routines
├── requirements.txt    # Project dependencies
└── README.md           # Documentation
```

## 🚀 Getting Started

### Prerequisites
- Python 3.8 or higher installed.

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/cryptocurrency-price-prediction.git
   cd cryptocurrency-price-prediction
   ```

2. **Create a virtual environment (Recommended):**
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## 🎮 Usage

Launch the application:
```bash
python app.py
```

1. **Configure Search:** Select your cryptocurrency (e.g., BTC, ETH) and target currency (USD, INR).
2. **Select Date:** Pick a starting point for historical data analysis.
3. **Adjust Parameters:** (Optional) Modify the SLTM hyperparameters for experimentation.
4. **Predict:** Hit the **Predict** button and watch the model train and forecast in real-time.

## 📝 Analysis & Methodology

The model employs a **MinMaxScaler** to normalize historical closing prices between 0 and 1, ensuring stable LSTM training. The architecture focuses on capturing long-term dependencies in price movements, which is critical for the volatile crypto market.

> [!NOTE]
> Cryptocurrency markets are highly volatile. This tool is for educational purposes and should not be used as financial advice.

## 🤝 Contributing

Contributions are welcome! If you'd like to improve the model architecture or add new features:
1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.
