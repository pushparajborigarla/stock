import streamlit as st
import pandas as pd
import numpy as np
import requests
import xgboost as xgb
from prophet import Prophet
from prophet.plot import plot_plotly
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import plotly.graph_objs as go
import base64
import io

st.set_page_config(layout="wide", page_title="Stock Prediction App")

st.title("📈 ML-based Stock Price Forecast App")

# Load ticker data
@st.cache_data
def load_tickers():
    df = pd.read_csv("new_tickers.csv")
    df['Label'] = df['Symbol'] + " - " + df['Name']
    return df[['Symbol', 'Label']]

tickers = load_tickers()
selected_label = st.selectbox("Select Stock", tickers["Label"])
selected_symbol = tickers[tickers["Label"] == selected_label]["Symbol"].values[0]

# Select date range
start_date = st.date_input("Start Date", pd.to_datetime("2021-01-01"))
end_date = st.date_input("End Date", pd.to_datetime("today"))

# Fetch data from Twelve Data API
@st.cache_data
def fetch_data(symbol):
    url = "https://api.twelvedata.com/time_series"
    params = {
        "symbol": symbol,
        "interval": "1day",
        "outputsize": 5000,
        "apikey": "b5354a5f939342ccaffd646871fc6f7a"
    }
    response = requests.get(url, params=params)
    data = response.json()
    if "values" not in data:
        return pd.DataFrame()
    df = pd.DataFrame(data["values"])
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.rename(columns={
        "datetime": "date",
        "close": "close",
        "open": "open",
        "high": "high",
        "low": "low",
        "volume": "volume"
    })
    df = df.astype({"open": "float", "high": "float", "low": "float", "close": "float", "volume": "float"})
    return df.sort_values("date")

df = fetch_data(selected_symbol)
if df.empty:
    st.warning("No data available for this ticker.")
    st.stop()

df = df[(df["date"] >= pd.to_datetime(start_date)) & (df["date"] <= pd.to_datetime(end_date))]
df = df[(df["date"] >= pd.to_datetime(start_date)) & (df["date"] <= pd.to_datetime(end_date))]

# Lightweight technical indicators (faster)
df["ma_10"] = df["close"].rolling(10).mean()
df["ma_30"] = df["close"].rolling(30).mean()
df["volatility"] = df["close"].rolling(10).std()

# Shift target (next-day closing price)
df["target"] = df["close"].shift(-1)
df.dropna(inplace=True)

# Feature/target split
X = df.drop(columns=["date", "target"])
y = df["target"]
split_index = int(len(df) * 0.8)
X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]

# Train XGBoost
@st.cache_resource
def train_xgb(X_train, y_train):
    model = xgb.XGBRegressor(
        objective='reg:squarederror',
        n_estimators=50,
        max_depth=4
    )
    model.fit(X_train, y_train)
    return model

model = train_xgb(X_train, y_train)
y_pred_xgb = model.predict(X_test)
# 🔮 XGBoost Future Forecast Function
def forecast_xgb_future(model, df, future_days):
    df_future = df.copy()
    predictions = []

    for _ in range(future_days):
        last_row = df_future.iloc[-1:].drop(columns=["date", "target"])
        next_pred = model.predict(last_row)[0]

        next_date = df_future["date"].iloc[-1] + pd.offsets.BDay(1)

        new_row = df_future.iloc[-1:].copy()
        new_row["date"] = next_date
        new_row["close"] = next_pred

        # Recalculate lightweight features
        new_row["ma_10"] = df_future["close"].rolling(10).mean().iloc[-1]
        new_row["ma_30"] = df_future["close"].rolling(30).mean().iloc[-1]
        new_row["volatility"] = df_future["close"].rolling(10).std().iloc[-1]

        df_future = pd.concat([df_future, new_row], ignore_index=True)
        predictions.append(next_pred)

    future_dates = pd.date_range(
        start=df["date"].iloc[-1] + pd.Timedelta(days=1),
        periods=future_days
    )

    return pd.DataFrame({
        "date": future_dates,
        "xgb_forecast": predictions
    })

# Train Prophet
df_prophet = df[["date", "close"]].rename(columns={"date": "ds", "close": "y"})
@st.cache_resource
def train_prophet(df_prophet):
    model = Prophet()
    model.fit(df_prophet)
    return model

prophet_model = train_prophet(df_prophet)

# Forecast range selector
forecast_range = st.selectbox("Forecast Range", ["Next Day", "1 Month", "3 Months", "1 Year"])
future_periods = {"Next Day": 1, "1 Month": 30, "3 Months": 90, "1 Year": 365}
future = prophet_model.make_future_dataframe(periods=future_periods[forecast_range])
forecast = prophet_model.predict(future)
# 🔮 XGBoost Future Forecast
future_days = future_periods[forecast_range]
xgb_future_df = forecast_xgb_future(model, df, future_days)

# Metrics
y_true_prophet = df_prophet["y"].iloc[-len(forecast["yhat"]):]
y_pred_prophet = forecast["yhat"][:len(df_prophet)]

mae_prophet = mean_absolute_error(y_true_prophet, y_pred_prophet)
rmse_prophet = np.sqrt(mean_squared_error(y_true_prophet, y_pred_prophet))
r2_prophet = r2_score(y_true_prophet, y_pred_prophet)

mae_xgb = mean_absolute_error(y_test, y_pred_xgb)
rmse_xgb = np.sqrt(mean_squared_error(y_test, y_pred_xgb))
r2_xgb = r2_score(y_test, y_pred_xgb)

# 📊 Display Accuracy
st.subheader("📊 Model Accuracy Comparison")

col1, col2 = st.columns(2)

with col1:
    st.markdown("### 🔮 Prophet Accuracy")
    st.metric("MAE", f"{mae_prophet:.2f}")
    st.metric("RMSE", f"{rmse_prophet:.2f}")
    st.metric("R² Score", f"{r2_prophet:.2f}")

with col2:
    st.markdown("### 🚀 XGBoost Accuracy")
    st.metric("MAE", f"{mae_xgb:.2f}")
    st.metric("RMSE", f"{rmse_xgb:.2f}")
    st.metric("R² Score", f"{r2_xgb:.2f}")

# 📈 Prophet Forecast Plot
st.subheader(f"📉 Prophet Forecast: {forecast_range}")
fig1 = plot_plotly(prophet_model, forecast)
st.plotly_chart(fig1)
# 📈 XGBoost Future Forecast Plot
st.subheader(f"🚀 XGBoost Future Forecast: {forecast_range}")

fig_xgb_future = go.Figure()
fig_xgb_future.add_trace(
    go.Scatter(
        x=xgb_future_df["date"],
        y=xgb_future_df["xgb_forecast"],
        mode="lines+markers" if future_days == 1 else "lines",
        name="XGBoost Forecast"
    )
)

st.plotly_chart(fig_xgb_future)

# 📈 Actual vs Predicted with XGBoost
st.subheader("📈 XGBoost Actual vs Predicted")
results_df = pd.DataFrame({
    "Actual": y_test.values,
    "Predicted": y_pred_xgb[:len(y_test)]
})
st.line_chart(results_df)

# 🔗 Download Excel
output = io.BytesIO()
with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
    forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].to_excel(writer, sheet_name="Prophet Forecast", index=False)
    results_df.to_excel(writer, sheet_name="XGBoost Results", index=False)
data = output.getvalue()

st.download_button(
    label="📥 Download Predictions as Excel",
    data=data,
    file_name=f"{selected_symbol}_forecast.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)
