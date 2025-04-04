import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import datetime as dt
import yfinance as yf
import io
import base64
from flask import Flask, render_template, request, send_file, redirect, url_for, flash
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

# Initialize Flask App
app = Flask(__name__)
app.secret_key = "your_secret_key"  # Required for flash messages

# Ensure static directory exists
os.makedirs("static", exist_ok=True)

# Load Model (Handle errors)
MODEL_PATH = "C://Users//sit421//Desktop//STOCK PRICE PREDICTION USING LSTM//.ipynb_checkpoints//stock_dl_model.h5"

model = None
try:
    if os.path.exists(MODEL_PATH):
        model = load_model(MODEL_PATH)
        print("✅ Model Loaded Successfully!")
    else:
        print(f"❌ Model file not found: {MODEL_PATH}")
except Exception as e:
    print(f"❌ Error Loading Model: {e}")

def plot_to_img(fig):
    """Converts a matplotlib figure to a base64 image for rendering in HTML."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)
    return base64.b64encode(buf.getvalue()).decode()

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        stock = request.form.get("stock", "").strip().upper()

        if not stock:
            flash("⚠️ Please enter a stock symbol!", "warning")
            return redirect(url_for("index"))

        try:
            # Define date range for stock data
            start, end = dt.datetime(2000, 1, 1), dt.datetime(2024, 11, 1)
            df = yf.download(stock, start=start, end=end)
            
            if df.empty:
                flash("⚠️ Invalid stock symbol or no data available!", "danger")
                return redirect(url_for("index"))

            # Compute EMAs
            df["20EMA"] = df["Close"].ewm(span=20, adjust=False).mean()
            df["50EMA"] = df["Close"].ewm(span=50, adjust=False).mean()
            df["100EMA"] = df["Close"].ewm(span=100, adjust=False).mean()
            df["200EMA"] = df["Close"].ewm(span=200, adjust=False).mean()

            # Generate and save plots
            plot_ema_20_50 = generate_plot(df, ["Close", "20EMA", "50EMA"], "Closing Price vs Time (20 & 50 Days EMA)")
            plot_ema_100_200 = generate_plot(df, ["Close", "100EMA", "200EMA"], "Closing Price vs Time (100 & 200 Days EMA)")

            # Generate predictions if model is available
            y_test, y_predicted = generate_predictions(df) if model else (None, None)
            plot_prediction = generate_prediction_plot(y_test, y_predicted)

            # Save dataset as CSV
            csv_file_path = f"static/{stock}_dataset.csv"
            df.to_csv(csv_file_path)

            return render_template("index.html", 
                                   plot_ema_20_50=plot_ema_20_50,
                                   plot_ema_100_200=plot_ema_100_200,
                                   plot_prediction=plot_prediction,
                                   data_desc=df.describe().to_html(classes="table table-striped"), 
                                   dataset_link=csv_file_path)

        except Exception as e:
            flash(f"❌ Error: {str(e)}", "danger")
            return redirect(url_for("index"))

    return render_template("index.html")

def generate_plot(df, columns, title):
    """Generates line plot for stock data and returns a base64 image."""
    fig, ax = plt.subplots(figsize=(10, 5))
    for col in columns:
        ax.plot(df.index, df[col], label=col)
    ax.legend()
    ax.set_title(title)
    return plot_to_img(fig)

def generate_predictions(df):
    """Predicts stock price using LSTM model and returns actual & predicted values."""
    scaler = MinMaxScaler(feature_range=(0, 1))
    train_size = int(len(df) * 0.70)
    data_training, data_testing = df["Close"][:train_size], df["Close"][train_size:]
    data_training_array = scaler.fit_transform(data_training.values.reshape(-1, 1))

    # Prepare test data
    past_100_days = data_training[-100:]
    final_df = pd.concat([past_100_days, data_testing])
    input_data = scaler.transform(final_df.values.reshape(-1, 1))

    x_test, y_test = [], []
    for i in range(100, len(input_data)):
        x_test.append(input_data[i - 100:i])
        y_test.append(input_data[i, 0])

    x_test, y_test = np.array(x_test), np.array(y_test)
    
    # Predict Prices
    y_predicted = model.predict(x_test)

    # Reverse scaling
    scale_factor = 1 / scaler.scale_[0]
    return y_test * scale_factor, y_predicted.flatten() * scale_factor

def generate_prediction_plot(y_test, y_predicted):
    """Generates a prediction vs actual stock price trend graph."""
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(y_test, label="Actual Price", color="green", linewidth=1)
    ax.plot(y_predicted, label="Predicted Price", color="red", linewidth=1)
    ax.legend()
    ax.set_title("Prediction vs Original Trend")
    return plot_to_img(fig)

@app.route("/download/<filename>")
def download_file(filename):
    """ Endpoint for downloading CSV dataset """
    return send_file(f"static/{filename}", as_attachment=True)

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))


