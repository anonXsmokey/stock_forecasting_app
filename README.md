# 📈 Stock Market Forecasting App

This is a Streamlit-based web application for stock market forecasting using multiple models including:

- ARIMA
- SARIMAX
- Prophet
- Support Vector Regression (SVR)
- Random Forest
- LSTM (PyTorch)

## 🧰 Technologies Used

- Python (pandas, numpy, matplotlib, seaborn)
- Streamlit for the web interface
- Statsmodels for statistical models
- Prophet by Facebook for time series forecasting
- scikit-learn for machine learning models
- PyTorch for deep learning (LSTM)

## 📂 How to Run

1. Clone this repository or download the files.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the app:

```bash
streamlit run stock_forecasting_app.py
```

## 📁 Input Data

Ensure you place your CSV files at the specified paths or modify the code:
- `stocks.csv`: Main stock time-series data
- `historical_stocks.csv`: Optional historical data preview

## 📌 Features

- View stock data and historical patterns
- ADF stationarity test
- Correlation heatmap
- Multiple forecasting models with visual results

---

Developed as part of a stock prediction project internship.
