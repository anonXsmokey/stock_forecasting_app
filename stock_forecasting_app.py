import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from datetime import datetime
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.tsa.stattools import adfuller
import streamlit as st

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# PyTorch LSTM model
class StockLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, num_layers=2, output_size=1):
        super(StockLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

def main():
    st.title("📈 Stock Market Forecasting App")

    # Load data
    stocks_df = pd.read_csv(r"C:\Users\LENOVO\OneDrive\Documents\wholestock\stocks.csv")
    stocks_df['Date'] = pd.to_datetime(stocks_df['Date'])
    stocks_df.set_index('Date', inplace=True)

    historical_df = pd.read_csv(r"C:\Users\LENOVO\OneDrive\Documents\wholestock\historical_stocks.csv")

    st.header("Data Preview")
    st.write(stocks_df.head())
    st.write("Historical Stocks Data:")
    st.write(historical_df.head())

    st.header("📊 Visualizations")
    fig, ax = plt.subplots()
    stocks_df['Close'].plot(ax=ax)
    ax.set_title("Closing Price Over Time")
    st.pyplot(fig)

    fig2, ax2 = plt.subplots()
    stocks_df['Volume'].plot(ax=ax2)
    ax2.set_title("Trading Volume Over Time")
    st.pyplot(fig2)

    st.subheader("Correlation Heatmap")
    fig3, ax3 = plt.subplots()
    sns.heatmap(stocks_df.corr(), annot=True, cmap="coolwarm", ax=ax3)
    st.pyplot(fig3)

    st.subheader("ADF Stationarity Test")
    result = adfuller(stocks_df['Close'].dropna())
    st.write(f"ADF Statistic: {result[0]:.4f}")
    st.write(f"p-value: {result[1]:.4f}")

    st.header("🔮 Forecasting Models")
    model_choice = st.selectbox("Choose Model:", ["ARIMA", "SARIMAX", "Prophet", "SVR", "Random Forest", "LSTM"])

    train_size = int(len(stocks_df) * 0.8)
    train, test = stocks_df.iloc[:train_size], stocks_df.iloc[train_size:]

    if model_choice == "ARIMA":
        p = st.slider("p", 0, 5, 1)
        d = st.slider("d", 0, 2, 1)
        q = st.slider("q", 0, 5, 1)
        if st.button("Run ARIMA"):
            model = ARIMA(train['Close'], order=(p, d, q))
            model_fit = model.fit()
            forecast = model_fit.forecast(steps=len(test))
            st.line_chart(pd.DataFrame({"Actual": test['Close'].values, "Forecast": forecast.values}, index=test.index))

    elif model_choice == "SARIMAX":
        p = st.slider("p", 0, 5, 1)
        d = st.slider("d", 0, 2, 1)
        q = st.slider("q", 0, 5, 1)
        P = st.slider("P(seasonal)", 0, 5, 1)
        s = st.slider("Seasonal Period", 1, 30, 12)
        if st.button("Run SARIMAX"):
            model = SARIMAX(train['Close'], order=(p, d, q), seasonal_order=(P, d, q, s))
            result_model = model.fit()
            forecast = result_model.forecast(steps=len(test))
            st.line_chart(pd.DataFrame({"Actual": test['Close'].values, "Forecast": forecast.values}, index=test.index))

    elif model_choice == "Prophet":
        if st.button("Run Prophet"):
            prophet_df = stocks_df.reset_index()[['Date', 'Close']].rename(columns={'Date': 'ds', 'Close': 'y'})
            m = Prophet()
            m.fit(prophet_df.iloc[:train_size])
            future = m.make_future_dataframe(periods=len(test))
            forecast = m.predict(future)
            fig = m.plot(forecast)
            st.pyplot(fig)

    elif model_choice == "SVR":
        if st.button("Run SVR"):
            model = SVR()
            model.fit(np.arange(train_size).reshape(-1, 1), train['Close'])
            preds = model.predict(np.arange(train_size, len(stocks_df)).reshape(-1, 1))
            st.line_chart(pd.DataFrame({"Actual": test['Close'].values, "Forecast": preds}, index=test.index))

    elif model_choice == "Random Forest":
        if st.button("Run Random Forest"):
            model = RandomForestRegressor()
            model.fit(np.arange(train_size).reshape(-1, 1), train['Close'])
            preds = model.predict(np.arange(train_size, len(stocks_df)).reshape(-1, 1))
            st.line_chart(pd.DataFrame({"Actual": test['Close'].values, "Forecast": preds}, index=test.index))

    elif model_choice == "LSTM":
        if st.button("Run LSTM"):
            def create_sequences(data, seq_len=10):
                xs, ys = [], []
                for i in range(len(data)-seq_len):
                    x = data[i:i+seq_len]
                    y = data[i+seq_len]
                    xs.append(x)
                    ys.append(y)
                return torch.tensor(xs).float().unsqueeze(-1), torch.tensor(ys).float()

            seq_len = 10
            train_seq, train_labels = create_sequences(train['Close'].values, seq_len)
            test_seq, _ = create_sequences(test['Close'].values, seq_len)

            loader = DataLoader(TensorDataset(train_seq, train_labels), batch_size=16, shuffle=False)

            model = StockLSTM()
            criterion = nn.MSELoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

            for epoch in range(50):
                for seq_batch, label_batch in loader:
                    optimizer.zero_grad()
                    output = model(seq_batch)
                    loss = criterion(output.squeeze(), label_batch)
                    loss.backward()
                    optimizer.step()

            with torch.no_grad():
                preds = []
                for seq in test_seq:
                    seq = seq.unsqueeze(0)
                    pred = model(seq)
                    preds.append(pred.item())

            st.line_chart(pd.DataFrame({"Actual": test['Close'].values[seq_len:], "Forecast": preds}, index=test.index[seq_len:]))

if __name__ == "__main__":
    main()
