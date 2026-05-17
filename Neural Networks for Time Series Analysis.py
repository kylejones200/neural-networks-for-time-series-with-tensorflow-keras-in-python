"""Generated from Jupyter notebook: Neural Networks for Time Series Analysis

Magics and shell lines are commented out. Run with a normal Python interpreter."""


# --- code cell ---

# !pip install numpy==1.23.5  # Jupyter-only


# --- code cell ---

# !pip install --upgrade pillow  # Jupyter-only


# --- code cell ---

import numpy as np  # numpy-1.24.4
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler

# Generate synthetic data
np.random.seed(42)
time = pd.date_range(start="2023-01-01", periods=1000, freq="D")
data = (
    10
    + 0.5 * np.arange(len(time))
    + np.sin(0.2 * np.arange(len(time)))
    + np.random.normal(scale=1.0, size=len(time))
)
df = pd.DataFrame({"date": time, "value": data})


# Create lagged features
def create_features(df, lag=3):
    for i in range(1, lag + 1):
        df[f"lag_{i}"] = df["value"].shift(i)
    return df.dropna()


lag = 3
df_features = create_features(df, lag=lag)

# Prepare data for modeling
X = df_features.drop(["date", "value"], axis=1)
y = df_features["value"]

# Initialize TimeSeriesSplit
ts_cv = TimeSeriesSplit(n_splits=5)

# Initialize MinMaxScaler
scaler = MinMaxScaler()

# Define the model
model = MLPRegressor(
    hidden_layer_sizes=(64, 32),
    activation="relu",
    solver="adam",
    max_iter=500,
    random_state=42,
)

# Perform cross-validation
cv_scores = []
all_y_test = []
all_y_pred = []

for train_index, test_index in ts_cv.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    # Scale the features
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Fit the model and make predictions
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

    # Calculate the score
    score = mean_squared_error(y_test, y_pred)
    cv_scores.append(score)

    # Store actual and predicted values
    all_y_test.extend(y_test)
    all_y_pred.extend(y_pred)

# Calculate and print the average RMSE
rmse = np.sqrt(np.mean(cv_scores))
rmse_std = np.sqrt(np.std(cv_scores))
print(f"RMSE = {rmse:.3f} +/- {rmse_std:.3f}")

"""# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(df_features['date'].iloc[lag:], all_y_test, label='Actual', color='Blue')
plt.plot(df_features['date'].iloc[lag:], all_y_pred, label='Predicted', color='Red')
plt.title('Neural Network Forecast with TimeSeriesSplit')
plt.legend()
plt.xticks(rotation=45)
plt.savefig("NN_forecast_timeseries.png")
plt.show()
"""


# --- code cell ---

import plotly.graph_objs as go
from plotly.subplots import make_subplots

# ... (rest of your code remains the same until the plotting part)

# Plot the results
fig = make_subplots(rows=1, cols=1)
fig.add_trace(
    go.Scatter(
        x=df_features["date"].iloc[lag:],
        y=all_y_test,
        name="Actual",
        line=dict(color="blue"),
    )
)
fig.add_trace(
    go.Scatter(
        x=df_features["date"].iloc[lag:],
        y=all_y_pred,
        name="Predicted",
        line=dict(color="red"),
    )
)
fig.update_layout(
    title="Neural Network Forecast with TimeSeriesSplit",
    xaxis_title="Date",
    yaxis_title="Value",
)
fig.write_image("NN_forecast_timeseries.png")
fig.show()


# --- code cell ---

from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import yaml
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler


def load_config():
    with open("main.yaml") as f:
        return yaml.safe_load(f)


def fetch_fred_data(config):
    """Fetch data from FRED API and save to CSV."""
    params = {
        "series_id": config["fred"]["series_id"],
        "api_key": config["fred"]["api_key"],
        "file_type": "json",
        "observation_start": config["fred"]["start_date"],
        "observation_end": datetime.now().strftime("%Y-%m-%d"),
    }
    url = config["fred"]["api_url"]
    response = requests.get(url, params=params)

    if response.status_code == 200:
        data = response.json()
        observations = data["observations"]
        df = pd.DataFrame(observations)
        df["date"] = pd.to_datetime(df["date"])
        df["value"] = pd.to_numeric(df["value"], errors="coerce")

        df = df.dropna()
        df = df.sort_values("date")
        df = df.set_index("date")

        csv_filename = f"{config['fred']['series_id']}_data.csv"
        df.to_csv(csv_filename)
        print(f"Data saved to {csv_filename}")

        return csv_filename
    else:
        raise Exception(f"API request failed with status code {response.status_code}")


def read_fred_data(filename):
    """Read FRED data from CSV file."""
    df = pd.read_csv(filename, parse_dates=["date"])
    return df.set_index("date")


def create_features(df, lag=3):
    for i in range(1, lag + 1):
        df[f"lag_{i}"] = df["value"].shift(i)
    return df.dropna()


def display_forecast(actual_series, pred_series, forecast_type, config):
    plt.figure(figsize=(12, 6))
    actual_series.plot(label="Actual", color="blue")
    pred_series.plot(label=f"{forecast_type.capitalize()} Forecast", color="red")
    plt.title(
        f"{config['fred']['series_id']}: {forecast_type.capitalize()} Forecast\n"
        f"R2: {r2_score(actual_series, pred_series):.4f}"
    )
    plt.xlabel("Date")
    plt.ylabel(config["plot"]["y_label"])
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{config['plot']['output_dir']}/MLP_{forecast_type}_Forecast.png")
    plt.show()


if __name__ == "__main__":
    config = load_config()

    # Fetch data from FRED and save to CSV
    csv_filename = fetch_fred_data(config)

    # Read data from CSV
    df = read_fred_data(csv_filename)
    print("Data read successfully.")
    print(df.head())
    print(f"Number of rows after reading CSV: {len(df)}")

    # Create lagged features
    lag = config["data"]["lag"]
    df_features = create_features(df.reset_index(), lag=lag)

    # Prepare data for modeling
    X = df_features[["lag_1", "lag_2", "lag_3"]]
    y = df_features["value"]

    # Initialize TimeSeriesSplit
    ts_cv = TimeSeriesSplit(n_splits=min(config["model"]["n_splits"], len(X) - 1))

    # Initialize MinMaxScaler
    scaler = MinMaxScaler()

    # Define the model
    model = MLPRegressor(**config["model"]["mlp_params"])

    # Perform cross-validation and generate forecasts
    cv_scores = []
    all_predictions = []
    all_actuals = []

    for train_index, test_index in ts_cv.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        # Scale the features
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Fit the model and make predictions
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)

        # Store predictions and actuals
        all_predictions.extend(y_pred)
        all_actuals.extend(y_test)

        # Calculate the score
        score = mean_squared_error(y_test, y_pred)
        cv_scores.append(score)

    # Calculate and print the average RMSE
    rmse = np.sqrt(np.mean(cv_scores))
    rmse_std = np.sqrt(np.std(cv_scores))
    print(f"RMSE = {rmse:.3f} +/- {rmse_std:.3f}")

    # Create series for actual and predicted values
    actual_series = pd.Series(all_actuals, index=df_features.index[-len(all_actuals) :])
    pred_series = pd.Series(
        all_predictions, index=df_features.index[-len(all_predictions) :]
    )

    # Display the historical forecast
    display_forecast(actual_series, pred_series, "historical", config)


# --- code cell ---

"""
works
"""


import warnings
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Generate future predictions
future_steps = 30  # Number of future steps to predict
last_known_values = X.iloc[-1].values.reshape(1, -1)
future_predictions = []

# Context manager to ignore the specific warning
with warnings.catch_warnings():

    for _ in range(future_steps):
        # Scale the input
        scaled_input = scaler.transform(last_known_values)

        # Make prediction
        prediction = model.predict(scaled_input)
        future_predictions.append(prediction[0])

        # Update last_known_values for next prediction
        last_known_values = np.roll(last_known_values, -1)
        last_known_values[0, -1] = prediction[0]

# Create future dates
last_date = df_features["date"].iloc[-1]
future_dates = pd.date_range(
    start=last_date + pd.Timedelta(days=1), periods=future_steps
)

# Filter historical data since 2024
start_date = datetime(2024, 1, 1)
historical_data = df_features[df_features["date"] >= start_date]

# Plot historical data and future predictions
plt.figure(figsize=(12, 6))
plt.plot(
    historical_data["date"],
    historical_data["value"],
    label="Historical Data",
    color="blue",
)
plt.plot(future_dates, future_predictions, label="Future Predictions", color="red")
plt.title(f"{config['fred']['series_id']} Historical Data and Future Forecast")
plt.xlabel("Date")
plt.ylabel(config["plot"]["y_label"])
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("NN_forecast_future.png")
plt.show()

# Print future predictions
future_df = pd.DataFrame({"Date": future_dates, "Predicted Value": future_predictions})

print("Forecasting completed and visualizations saved.")


# --- code cell ---

import warnings
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm

# Generate future predictions
future_steps = 30  # Number of future steps to predict
last_known_values = X.iloc[-1].values.reshape(1, -1)
future_predictions = []
prediction_std = []

# Context manager to ignore the specific warning
with warnings.catch_warnings():

    for step in range(future_steps):
        # Scale the input
        scaled_input = scaler.transform(last_known_values)

        # Make prediction
        prediction = model.predict(scaled_input)
        future_predictions.append(prediction[0])

        # Calculate prediction standard deviation
        # Increase uncertainty as we predict further into the future
        base_std = np.std(model.predict(scaler.transform(X.iloc[-100:])))
        pred_std = base_std * np.sqrt(step + 1)  # Increase std over time
        prediction_std.append(pred_std)

        # Update last_known_values for next prediction
        lastast_known_values = np.roll(last_known_values, -1)
        last_known_values[0, -1] = prediction[0]

# Create future dates
last_date = df_features["date"].iloc[-1]
future_dates = pd.date_range(
    start=last_date + pd.Timedelta(days=1), periods=future_steps
)

# Filter historical data since 2024
start_date = datetime(2020, 1, 1)
historical_data = df_features[df_features["date"] >= start_date]

# Calculate confidence intervals
confidence_interval = 0.95
z_score = norm.ppf((1 + confidence_interval) / 2)
lower_bound = [
    pred - z_score * std for pred, std in zip(future_predictions, prediction_std)
]
upper_bound = [
    pred + z_score * std for pred, std in zip(future_predictions, prediction_std)
]

# Plot historical data and future predictions with cone of uncertainty
plt.figure(figsize=(12, 6))
plt.plot(
    historical_data["date"],
    historical_data["value"],
    label="Historical Data",
    color="blue",
)
plt.plot(future_dates, future_predictions, label="Future Predictions", color="red")
plt.fill_between(
    future_dates,
    lower_bound,
    upper_bound,
    color="red",
    alpha=0.2,
    label=f"{confidence_interval * 100}% Confidence Interval",
)
plt.title(f"{config['fred']['series_id']} Historical Data and Future Forecast")
plt.xlabel("Date")
plt.ylabel(config["plot"]["y_label"])
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("NN_forecast_future_with_uncertainty_cone.png")
plt.show()

# Print future predictions with confidence intervals
future_df = pd.DataFrame(
    {
        "Date": future_dates,
        "Predicted Value": future_predictions,
        "Lower Bound": lower_bound,
        "Upper Bound": upper_bound,
    }
)

print("Forecasting completed and visualizations saved.")
