import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from sklearn.preprocessing import MinMaxScaler

# Define chunk size for efficient memory usage
chunksize = 10**6

# Initialize lists to store processed chunks
departures_chunks = []
passengers_chunks = []

# Load and aggregate data in chunks to handle large file sizes
for chunk in pd.read_csv('International_Report_Departures.csv', chunksize=chunksize):
    chunk.rename(columns={'data_dte': 'Date'}, inplace=True)
    chunk['Date'] = pd.to_datetime(chunk['Date'], errors='coerce')
    chunk = chunk.dropna(subset=['Date'])  # Drop rows where 'Date' is NaT
    chunk = chunk.groupby('Date').sum()  # Aggregate by date (sum for numerical columns)
    departures_chunks.append(chunk)

for chunk in pd.read_csv('International_Report_Passengers.csv', chunksize=chunksize):
    chunk.rename(columns={'data_dte': 'Date'}, inplace=True)
    chunk['Date'] = pd.to_datetime(chunk['Date'], errors='coerce')
    chunk = chunk.dropna(subset=['Date'])  # Drop rows where 'Date' is NaT
    chunk = chunk.groupby('Date').sum()  # Aggregate by date (sum for numerical columns)
    passengers_chunks.append(chunk)

# Concatenate the aggregated chunks into dataframes
departures_df = pd.concat(departures_chunks, ignore_index=False)
passengers_df = pd.concat(passengers_chunks, ignore_index=False)

# Merge data on 'Date' column
merged_df = pd.merge(departures_df, passengers_df, on='Date', suffixes=('_departures', '_passengers'))

# Forward fill missing values after merging
merged_df.fillna(method='ffill', inplace=True)

# Plot time-series data
target_column = 'Total_passengers'
plt.figure(figsize=(14, 7))
plt.plot(merged_df.index, merged_df[target_column], label='Air Traffic')
plt.title('International Air Traffic over Time')
plt.xlabel('Year')
plt.ylabel('Passenger Count')
plt.legend()
plt.show()

# Prepare data for machine learning models
X = np.arange(len(merged_df)).reshape(-1, 1)
y = merged_df[target_column].values

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Linear Regression
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)
y_pred_lr = linear_model.predict(X_test)

# Random Forest Regressor
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

# LSTM Model
scaler = MinMaxScaler(feature_range=(0, 1))
y_train_scaled = scaler.fit_transform(y_train.reshape(-1, 1))
y_test_scaled = scaler.transform(y_test.reshape(-1, 1))

X_train_lstm = X_train.reshape(-1, 1, 1)
X_test_lstm = X_test.reshape(-1, 1, 1)

lstm_model = Sequential()
lstm_model.add(LSTM(50, activation='relu', input_shape=(1, 1)))
lstm_model.add(Dense(1))
lstm_model.compile(optimizer='adam', loss='mse')
lstm_model.fit(X_train_lstm, y_train_scaled, epochs=10, verbose=1)
y_pred_lstm = lstm_model.predict(X_test_lstm).flatten()
y_pred_lstm = scaler.inverse_transform(y_pred_lstm.reshape(-1, 1)).flatten()

# Calculate Mean Squared Error for model comparison
mse_lr = mean_squared_error(y_test, y_pred_lr)
mse_rf = mean_squared_error(y_test, y_pred_rf)
mse_lstm = mean_squared_error(y_test, y_pred_lstm)

print(f'Linear Regression MSE: {mse_lr}')
print(f'Random Forest MSE: {mse_rf}')
print(f'LSTM MSE: {mse_lstm}')

# Calculate residuals (errors)
residual_lr = y_test - y_pred_lr
residual_rf = y_test - y_pred_rf
residual_lstm = y_test - y_pred_lstm

# Create a DataFrame to store actual, predicted values, and residuals
results_df = pd.DataFrame({
    'Date': merged_df.index[-len(y_test):],  # Use the dates corresponding to the test set
    'Actual': y_test,
    'Predicted_LR': y_pred_lr,
    'Predicted_RF': y_pred_rf,
    'Predicted_LSTM': y_pred_lstm,
    'Residual_LR': residual_lr,
    'Residual_RF': residual_rf,
    'Residual_LSTM': residual_lstm
})

# Display the results
print(results_df.head())

# Visualize Model Predictions and Residuals
plt.figure(figsize=(14, 8))

# Plot Actual Data
plt.plot(merged_df.index[-len(y_test):], y_test, label='Actual Air Traffic', color='black')

# Plot Linear Regression Predictions
plt.plot(merged_df.index[-len(y_test):], y_pred_lr, label='Linear Regression Predictions', color='blue', linestyle='--')

# Plot Random Forest Predictions
plt.plot(merged_df.index[-len(y_test):], y_pred_rf, label='Random Forest Predictions', color='green', linestyle='--')

# Plot LSTM Predictions
plt.plot(merged_df.index[-len(y_test):], y_pred_lstm, label='LSTM Predictions', color='red', linestyle='--')

plt.xlabel('Date')
plt.ylabel('Passenger Count')
plt.title('Air Traffic Prediction using Different ML Models')
plt.legend()
plt.show()

# Plot residuals (errors) for each model
plt.figure(figsize=(14, 8))

plt.subplot(3, 1, 1)
plt.plot(results_df['Date'], results_df['Residual_LR'], label='Linear Regression Residuals', color='blue')
plt.title('Residuals for Linear Regression')
plt.xlabel('Date')
plt.ylabel('Residuals')
plt.legend()

plt.subplot(3, 1, 2)
plt.plot(results_df['Date'], results_df['Residual_RF'], label='Random Forest Residuals', color='green')
plt.title('Residuals for Random Forest')
plt.xlabel('Date')
plt.ylabel('Residuals')
plt.legend()

plt.subplot(3, 1, 3)
plt.plot(results_df['Date'], results_df['Residual_LSTM'], label='LSTM Residuals', color='red')
plt.title('Residuals for LSTM')
plt.xlabel('Date')
plt.ylabel('Residuals')
plt.legend()

plt.tight_layout()
plt.show()

# Forecast Future Air Traffic using the best model (for demonstration, we choose LSTM)
# Forecast Future Air Traffic using the best model (for demonstration, we choose LSTM)
forecast_days = 30  # Number of days to forecast
last_date = merged_df.index[-1]
forecast_dates = pd.date_range(last_date, periods=forecast_days + 1, freq='D')[1:]

# Prepare input for LSTM model to forecast
input_data = merged_df[target_column].values[-60:]  # Last 60 days for LSTM input
input_data_scaled = scaler.transform(input_data.reshape(-1, 1))

# Reshape input data to fit LSTM
X_forecast_lstm = input_data_scaled.reshape(-1, 1, 1)

# Predict future values
future_predictions_scaled = lstm_model.predict(X_forecast_lstm).flatten()
future_predictions = scaler.inverse_transform(future_predictions_scaled.reshape(-1, 1)).flatten()

# Ensure the forecast matches the forecast dates
future_predictions = future_predictions[-forecast_days:]

# Plot forecasted data
plt.figure(figsize=(14, 7))
plt.plot(merged_df.index, merged_df[target_column], label='Actual Air Traffic', color='black')
plt.plot(forecast_dates, future_predictions, label='Forecasted Air Traffic (LSTM)', color='orange', linestyle='--')
plt.title('Air Traffic Forecast for Next 30 Days using LSTM')
plt.xlabel('Date')
plt.ylabel('Passenger Count')
plt.legend()
plt.show()

