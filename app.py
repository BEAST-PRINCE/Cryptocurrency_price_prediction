import tkinter as tk
from tkinter import ttk, messagebox
from tkcalendar import DateEntry
import datetime as dt
import matplotlib.pyplot as plt
import traceback

from data_utils import fetch_crypto_data, prepare_training_data, prepare_testing_data
from model_utils import build_and_train_model, evaluate_model, predict_future

class CryptoPredictorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Crypto Price Predictor")
        self.root.geometry("900x650") 
        
        # Initialize model and data attributes
        self.data = None
        self.model = None  # Store the model for reuse
        self.scaler = None
        
        self.optimizer_options = ["Adam", "RMSprop", "SGD", "Adagrad"]
        self.loss_options = [
            "mean_squared_error", 
            "mean_absolute_error", 
            "mean_absolute_percentage_error", 
        ]
        
        self.create_widgets()

    def create_widgets(self):
        # Main frame for controls
        control_frame = ttk.Frame(self.root, padding="10")
        control_frame.grid(row=0, column=0, sticky='ew')
        self.root.columnconfigure(0, weight=1)
        
        # Date selection
        ttk.Label(control_frame, text="Start Date:").grid(row=0, column=0, padx=10, pady=10, sticky='w')
        self.start_date = DateEntry(control_frame, width=12, background='darkblue', foreground='white', borderwidth=2, 
                                   year=2022)
        self.start_date.grid(row=0, column=1, padx=10, pady=10)

        # Crypto selector
        ttk.Label(control_frame, text="Cryptocurrency:").grid(row=1, column=0, padx=10, pady=10, sticky='w')
        self.crypto_var = tk.StringVar()
        self.crypto_menu = ttk.Combobox(control_frame, textvariable=self.crypto_var, 
                                       values=["BTC", "ETH", "DOGE", "XRP", "ADA", "SOL", "DOT"], state="readonly")
        self.crypto_menu.current(0)
        self.crypto_menu.grid(row=1, column=1, padx=10, pady=10)

        # Currency selector
        ttk.Label(control_frame, text="Currency:").grid(row=2, column=0, padx=10, pady=10, sticky='w')
        self.currency_var = tk.StringVar()
        self.currency_menu = ttk.Combobox(control_frame, textvariable=self.currency_var, 
                                         values=["INR", "USD",  "EUR", "GBP", "JPY"], state="readonly")
        self.currency_menu.current(0)
        self.currency_menu.grid(row=2, column=1, padx=10, pady=10)

        # Model parameters frame
        model_frame = ttk.LabelFrame(control_frame, text="Model Parameters")
        model_frame.grid(row=0, column=2, rowspan=3, padx=20, pady=10, sticky='ns')
        
        ttk.Label(model_frame, text="Prediction Days:").grid(row=0, column=0, padx=5, pady=5, sticky='w')
        self.prediction_days_var = tk.IntVar(value=10)
        self.prediction_days_entry = ttk.Spinbox(model_frame, from_=5, to=60, textvariable=self.prediction_days_var, width=8)
        self.prediction_days_entry.grid(row=0, column=1, padx=5, pady=5)
        
        ttk.Label(model_frame, text="Future Days to Predict:").grid(row=1, column=0, padx=5, pady=5, sticky='w')
        self.future_days_var = tk.IntVar(value=7)
        self.future_days_entry = ttk.Spinbox(model_frame, from_=1, to=30, textvariable=self.future_days_var, width=8)
        self.future_days_entry.grid(row=1, column=1, padx=5, pady=5)
        
        ttk.Label(model_frame, text="Epochs:").grid(row=2, column=0, padx=5, pady=5, sticky='w')
        self.epochs_var = tk.IntVar(value=10)
        self.epochs_entry = ttk.Spinbox(model_frame, from_=1, to=100, textvariable=self.epochs_var, width=8)
        self.epochs_entry.grid(row=2, column=1, padx=5, pady=5)
        
        ttk.Label(model_frame, text="Batch Size:").grid(row=3, column=0, padx=5, pady=5, sticky='w')
        self.batch_size_var = tk.IntVar(value=32)
        self.batch_size_entry = ttk.Spinbox(model_frame, from_=8, to=128, textvariable=self.batch_size_var, width=8)
        self.batch_size_entry.grid(row=3, column=1, padx=5, pady=5)
        
        ttk.Label(model_frame, text="Optimizer:").grid(row=4, column=0, padx=5, pady=5, sticky='w')
        self.optimizer_var = tk.StringVar(value="Adam")
        self.optimizer_menu = ttk.Combobox(model_frame, textvariable=self.optimizer_var, 
                                          values=self.optimizer_options, state="readonly", width=15)
        self.optimizer_menu.grid(row=4, column=1, padx=5, pady=5)
        
        ttk.Label(model_frame, text="Learning Rate:").grid(row=5, column=0, padx=5, pady=5, sticky='w')
        self.learning_rate_var = tk.DoubleVar(value=0.001)
        self.learning_rate_entry = ttk.Spinbox(model_frame, from_=0.0001, to=0.1, 
                                              increment=0.0001, format="%.4f",
                                              textvariable=self.learning_rate_var, width=8)
        self.learning_rate_entry.grid(row=5, column=1, padx=5, pady=5)
        
        ttk.Label(model_frame, text="Loss Function:").grid(row=6, column=0, padx=5, pady=5, sticky='w')
        self.loss_var = tk.StringVar(value="mean_squared_error")
        self.loss_menu = ttk.Combobox(model_frame, textvariable=self.loss_var, 
                                     values=self.loss_options, state="readonly", width=15)
        self.loss_menu.grid(row=6, column=1, padx=5, pady=5)

        # Buttons frame
        button_frame = ttk.Frame(self.root)
        button_frame.grid(row=1, column=0, pady=10)
        
        self.predict_button = ttk.Button(button_frame, text="Predict", command=self.predict)
        self.predict_button.grid(row=0, column=0, padx=10)
        
        self.status_var = tk.StringVar(value="Ready")
        self.status_label = ttk.Label(button_frame, textvariable=self.status_var, font=("Arial", 10))
        self.status_label.grid(row=0, column=1, padx=20)

        # Results frame
        result_frame = ttk.LabelFrame(self.root, text="Prediction Results")
        result_frame.grid(row=2, column=0, padx=10, pady=10, sticky='ew')
        
        self.result_label = ttk.Label(result_frame, text="", font=("Arial", 12))
        self.result_label.pack(pady=10)
        
        # Accuracy gauge frame
        accuracy_frame = ttk.LabelFrame(self.root, text="Model Accuracy")
        accuracy_frame.grid(row=3, column=0, padx=10, pady=5, sticky='ew')
        
        gauge_frame = ttk.Frame(accuracy_frame)
        gauge_frame.pack(fill=tk.X, padx=10, pady=10)
        
        self.accuracy_var = tk.DoubleVar(value=0)
        ttk.Label(gauge_frame, text="0%").pack(side=tk.LEFT)
        self.accuracy_progress = ttk.Progressbar(gauge_frame, orient=tk.HORIZONTAL, 
                                                length=700, mode="determinate", 
                                                variable=self.accuracy_var)
        self.accuracy_progress.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        ttk.Label(gauge_frame, text="100%").pack(side=tk.LEFT)
        
        self.metrics_label = ttk.Label(accuracy_frame, text="", font=("Arial", 10))
        self.metrics_label.pack(pady=5)

        # Future predictions table frame
        self.future_frame = ttk.LabelFrame(self.root, text="Future Price Predictions")
        self.future_frame.grid(row=4, column=0, padx=10, pady=5, sticky='ew')
        
        table_frame = ttk.Frame(self.future_frame)
        table_frame.pack(fill=tk.BOTH, padx=10, pady=10, expand=True)
        
        self.pred_tree = ttk.Treeview(table_frame, columns=("date", "price", "change"), show="headings")
        self.pred_tree.heading("date", text="Date")
        self.pred_tree.heading("price", text="Predicted Price")
        self.pred_tree.heading("change", text="Change")
        self.pred_tree.column("date", width=120)
        self.pred_tree.column("price", width=120)
        self.pred_tree.column("change", width=120)
        self.pred_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        scrollbar = ttk.Scrollbar(table_frame, orient=tk.VERTICAL, command=self.pred_tree.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.pred_tree.configure(yscrollcommand=scrollbar.set)

    def plot_results(self, train_dates, train_prices, test_dates, test_prices, 
                     prediction_dates, predicted_prices, future_dates, future_prices,
                     crypto, currency):
        """Plot results in a separate matplotlib window"""
        plt.figure(figsize=(12, 7))
        
        plt.plot(train_dates, train_prices, color='blue', label='Training Data')
        plt.plot(test_dates, test_prices, color='black', label='Actual Price')
        if len(prediction_dates) > 0:
            plt.plot(prediction_dates, predicted_prices, color='green', label='Predicted Price')
        plt.plot(future_dates, future_prices, color='red', linestyle='--', marker='o', label='Future Predictions')
        
        plt.legend()
        plt.title(f'{crypto}-{currency} Price Prediction')
        plt.xlabel('Date')
        plt.ylabel(f'Price ({currency})')
        plt.grid(True, alpha=0.3)
        plt.gcf().autofmt_xdate()
        
        plt.ion()
        plt.show()

    def predict(self):
        try:
            for item in self.pred_tree.get_children():
                self.pred_tree.delete(item)
                
            self.predict_button.config(state=tk.DISABLED)
            self.status_var.set("Processing...")
            self.root.update()
            
            crypto = self.crypto_var.get()
            currency = self.currency_var.get()
            prediction_days = self.prediction_days_var.get()
            future_days = self.future_days_var.get()
            epochs = self.epochs_var.get()
            batch_size = self.batch_size_var.get()
            start_date = self.start_date.get_date()
            end_date = dt.datetime.now()
            
            optimizer_name = self.optimizer_var.get()
            learning_rate = self.learning_rate_var.get()
            loss_function = self.loss_var.get()
            
            try:
                self.status_var.set("Downloading data from Yahoo Finance...")
                self.root.update()
                self.data = fetch_crypto_data(crypto, currency, start_date, end_date)
            except Exception as e:
                messagebox.showerror("Error", str(e))
                self.predict_button.config(state=tk.NORMAL)
                self.status_var.set("Ready")
                return
            
            if self.data.empty or len(self.data) < prediction_days * 2:
                messagebox.showerror("Error", f"Not enough data to train model. Need at least {prediction_days * 2} days of data.")
                self.predict_button.config(state=tk.NORMAL)
                self.status_var.set("Ready")
                return

            x_train, y_train, scaled_data, self.scaler = prepare_training_data(self.data, prediction_days)

            self.status_var.set(f"Training model with {optimizer_name} optimizer and {loss_function} loss...")
            self.root.update()
            
            self.model = build_and_train_model(x_train, y_train, epochs, batch_size, optimizer_name, learning_rate, loss_function)
            
            actual_prices = self.data['Close'].values
            current_price = float(actual_prices[-1])
            train_len = int(len(scaled_data) * 0.8)
            
            x_test, y_test = prepare_testing_data(scaled_data, actual_prices, prediction_days, train_len)
            
            predicted_prices = self.model.predict(x_test)
            predicted_prices = self.scaler.inverse_transform(predicted_prices)
            
            mse, mae, accuracy, mape = evaluate_model(y_test, predicted_prices)
            
            if accuracy > 0:
                self.accuracy_var.set(accuracy)
                self.metrics_label.config(text=f"Accuracy: {accuracy:.2f}% | MAPE: {mape:.2f}% | MSE: {mse:.2f} | MAE: {mae:.2f}")
            else:
                self.accuracy_var.set(0)
                self.metrics_label.config(text="Insufficient data for accuracy calculation")
            
            last_sequence = scaled_data[-prediction_days:]
            future_prices = predict_future(self.model, last_sequence, future_days, self.scaler)
            future_dates = [self.data.index[-1] + dt.timedelta(days=i+1) for i in range(future_days)]
            
            for i, (date, price) in enumerate(zip(future_dates, future_prices)):
                price = float(price)
                prev_price = float(current_price if i == 0 else future_prices[i-1])
                price_change = float(price - prev_price)
                price_change_percent = float((price_change / prev_price) * 100) if prev_price > 0 else 0
                
                change_text = f"{'↑' if price_change >= 0 else '↓'}{abs(price_change):.2f} ({abs(price_change_percent):.2f}%)"
                date_str = date.strftime('%Y-%m-%d')
                
                self.pred_tree.insert("", "end", values=(date_str, f"{price:.2f} {currency}", change_text))
            
            train_dates = self.data.index[:train_len]
            train_prices = actual_prices[:train_len]
            test_dates = self.data.index[train_len:]
            test_prices = actual_prices[train_len:]
            
            prediction_dates = test_dates[:len(predicted_prices)] if len(test_dates) >= len(predicted_prices) else []
            
            self.plot_results(
                train_dates, train_prices, 
                test_dates, test_prices,
                prediction_dates, predicted_prices, 
                future_dates, future_prices,
                crypto, currency
            )
            
            next_day_price = float(future_prices[0])
            price_change = float(next_day_price - current_price)
            price_change_percent = float((price_change / current_price) * 100) if current_price > 0 else 0
            
            self.result_label.config(text=(
                f"Current price: {current_price:.2f} {currency}\n"
                f"Tomorrow's predicted price: {next_day_price:.2f} {currency} "
                f"({'↑' if price_change >= 0 else '↓'}{abs(price_change):.2f}, {abs(price_change_percent):.2f}%)\n"
                f"Prediction for next {future_days} days shown in table and graph.\n"
                f"Model: {optimizer_name} optimizer with {loss_function} loss | Epochs: {epochs}"
            ))
            
            self.status_var.set("Prediction complete - Graph opened in separate window")
            
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {str(e)}")
            traceback.print_exc()
        finally:
            self.predict_button.config(state=tk.NORMAL)


if __name__ == '__main__':
    root = tk.Tk()
    app = CryptoPredictorApp(root)
    root.mainloop()
