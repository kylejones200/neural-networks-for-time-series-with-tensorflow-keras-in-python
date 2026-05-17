"""Core functions for neural networks in time series forecasting."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(message)s")


class _LSTMForecaster(nn.Module):
    """LSTM forecaster (auto-generated PyTorch replacement for Keras Sequential)."""
    def __init__(self, n_features: int, hidden: int = 64, output_size: int = 1,
                 n_layers: int = 1, dropout: float = 0.0):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden, num_layers=n_layers,
                            batch_first=True, dropout=dropout if n_layers > 1 else 0)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.fc(self.drop(out[:, -1, :]))

def _train_torch(model: nn.Module, X_train, y_train, *,
                 epochs: int = 50, batch_size: int = 32,
                 lr: float = 0.001, validation_split: float = 0.2,
                 patience: int = 15) -> nn.Module:
    """Standard training loop replacing  + model.fit()."""
    X_t = torch.FloatTensor(X_train)
    y_t = torch.FloatTensor(y_train)
    if y_t.dim() == 1:
        y_t = y_t.unsqueeze(1)
    n_val = max(1, int(len(X_t) * validation_split))
    X_val, y_val = X_t[-n_val:], y_t[-n_val:]
    X_tr, y_tr = X_t[:-n_val], y_t[:-n_val]
    loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    best, wait = float("inf"), 0
    for _ in range(epochs):
        model.train()
        for xb, yb in loader:
            optimizer.zero_grad()
            criterion(model(xb), yb).backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(X_val), y_val).item()
        if val_loss < best:
            best, wait = val_loss, 0
        else:
            wait += 1
            if wait >= patience:
                break
    return model


def _predict_torch(model: nn.Module, X_test) -> "np.ndarray":
    """Replace model.predict()."""
    model.eval()
    with torch.no_grad():
        return model(torch.FloatTensor(X_test)).numpy()

def create_lagged_features(data: np.ndarray, lag: int) -> tuple[np.ndarray, np.ndarray]:
    """Create lagged features for time series."""
    X, y = [], []
    for i in range(lag, len(data)):
        X.append(data[i - lag : i])
        y.append(data[i])
    return np.array(X), np.array(y)


def build_lstm_model(input_shape: tuple[int, int], units: int = 50) -> Sequential:
    """Build LSTM model for time series forecasting."""
    model = Sequential(
        [LSTM(units, activation="relu", input_shape=input_shape), Dense(1)]
    )
    return model


def prepare_data(
    df: pd.DataFrame, value_col: str, lag: int, train_size: float = 0.8
) -> tuple:
    """Prepare data for LSTM training."""
    scaler = MinMaxScaler()
    values = df[value_col].values.reshape(-1, 1)

    train_len = int(len(values) * train_size)
    train_data = values[:train_len]
    test_data = values[train_len:]

    _train_torch(scaler, train_data, y_train)
    train_scaled = scaler.transform(train_data)
    test_scaled = scaler.transform(test_data)

    X_train, y_train = create_lagged_features(train_scaled.flatten(), lag)
    X_test, y_test = create_lagged_features(test_scaled.flatten(), lag)

    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

    return X_train, X_test, y_train, y_test, scaler


def plot_forecast(
    actual: np.ndarray,
    predicted: np.ndarray,
    title: str,
    output_path: Path,
    plot: bool = False,
):
    """Plot forecast vs actual"""
    if plot:
        fig, ax = plt.subplots(figsize=(10, 6))

        ax.plot(actual, label="Actual", color="#4A90A4", linewidth=1.2)
        ax.plot(predicted, label="Predicted", color="#D4A574", linewidth=1.2)
        ax.set_xlabel("Time")
        ax.set_ylabel("Value")
        ax.legend(loc="best")

        plt.savefig(output_path, dpi=100, bbox_inches="tight")
        plt.close()
