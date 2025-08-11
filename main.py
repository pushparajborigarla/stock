# streamlit_app.py

import streamlit as st
import pandas as pd
import requests
from prophet import Prophet
from prophet.plot import plot_plotly
import plotly.graph_objs as go

# 🔐 Your Twelve Data API key
API_KEY = "b5354a5f939342ccaffd646871fc6f7a"  # Replace this with your actual key

# Load tickers
@st.cache_data
def load_tickers():
    df = pd.read_csv("new_tickers.csv")
    df['Label'] = df['Symbol'] + " - " + df['Name']
    return df[['Symbol', 'Label']]

tickers_df = load_tickers()

# Sidebar
st.sidebar.title("📊 Stock Selector")
stock_choice = st.sidebar.selectbox("Choose a stock", tickers_df['Label'])
selected_symbol = tickers_df[tickers_df['Label'] == stock_choice]['Symbol'].values[0]

# Date range
start_date = st.date_input("Start Date", pd.to_datetime("2020-01-01"))
end_date = st.date_input("End Date", pd.to_datetime("today"))
forecast_days = st.slider("Forecast period (days)", 30, 365, 90, step=30)

# Fetch data from Twelve Data
@st.cache_data
def fetch_twelve_data(symbol, interval="1day", outputsize="5000"):
    url = f"https://api.twelvedata.com/time_series"
    params = {
        "symbol": symbol,
        "interval": interval,
        "apikey": API_KEY,
        "outputsize": outputsize,
        "format": "JSON"
    }
    response = requests.get(url, params=params)
    data = response.json()
    if "values" not in data:
        return None
    df = pd.DataFrame(data["values"])
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.rename(columns={"datetime": "ds", "close": "y"})
    df["y"] = pd.to_numeric(df["y"], errors="coerce")
    df = df.sort_values("ds")
    df = df[df["ds"].between(pd.to_datetime(start_date), pd.to_datetime(end_date))]
    df.dropna(inplace=True)
    return df

data = fetch_twelve_data(selected_symbol)

if data is None or data.empty:
    st.error("⚠️ No data found for this stock. It might be delisted or inactive.")
else:
    st.title("📈 Stock Price Forecast with Prophet")
    st.subheader(stock_choice)
    st.write("### Historical Stock Data", data.tail())

    # Plot actual data
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['ds'], y=data['y'], name="Close Price"))
    fig.update_layout(title="Historical Close Prices", xaxis_title="Date", yaxis_title="Price")
    st.plotly_chart(fig)

    # Prophet modeling
    if len(data) < 2:
        st.error("Not enough data to forecast.")
    else:
        model = Prophet()
        model.fit(data)

        future = model.make_future_dataframe(periods=forecast_days)
        forecast = model.predict(future)

        st.write("### Forecasted Stock Price")
        fig_forecast = plot_plotly(model, forecast)
        st.plotly_chart(fig_forecast)

        st.write("### Forecast Data")
        st.dataframe(forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail())
