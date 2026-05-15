# Description: Short example for Neural Networks for Time Series with Tensorflow Keras in Python.


import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import MinMaxScaler

np.random.seed(42)

"""
Feedforward Neural Network for Simple Forecasting
"""


# Generate synthetic data
time = np.arange(100)
data = 10 + 0.5 * time + np.sin(0.2 * time) + np.random.normal(scale=1.0, size=100)


# Create lagged features
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

class _LSTMForecaster(nn.Module):
    """LSTM forecaster (auto-generated PyTorch replacement for Keras Sequential)."""
    def __init__(self, n_features: int, hidden: int = 64, output_size: int = 1,
                 n_layers: int = 0, dropout: float = 0.0):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden, num_layers=n_layers,
                            batch_first=True, dropout=dropout if n_layers > 1 else 0)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.fc(self.drop(out[:, -1, :]))

def _train_torch(model: nn.Module, X_train, y_train, *,
                 epochs: int = 50, batch_size: int = 8,
                 lr: float = 0.001, validation_split: float = 0.1,
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

def create_features(data, lag=3):
    X, y = [], []
    for i in range(len(data) - lag):
        X.append(data[i : i + lag])
        y.append(data[i + lag])
    return np.array(X), np.array(y)



def main():
    lag = 3
    X, y = create_features(data, lag=lag)

    tscv = TimeSeriesSplit(n_splits=5)
    train_idx, test_idx = list(tscv.split(X))[-1]
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Build a simple feedforward neural network
    model = Sequential(
        [
            nn.Dense(64, activation="relu", input_shape=(lag,)),
            nn.Dense(32, activation="relu"),
            nn.Dense(1),
        ]
    )

        model.summary()

    # Train the model
    _train_torch(model, X_train, y_train)

    # Evaluate and predict
    y_pred = _predict_torch(model, X_test)
    mape = mean_absolute_percentage_error(y_test, y_pred)

    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(y_test, label="Actual", color="Blue")
    plt.plot(y_pred, label="Predicted", color="Red")
    plt.title(f"Feedforward Neural Network Forecast \n MAPE: {mape:.3f}")
    plt.legend()
    plt.savefig("NN_forecast.png")
    plt.show()

    """
    Basic RNN for Time Series
    """


    # Build an RNN model
    model = Sequential(
        [SimpleRNN(50, activation="relu", input_shape=(lag, 1)), nn.Dense(1)]
    )

        model.summary()

    # Reshape input for RNN (samples, timesteps, features)
    X_train_rnn = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test_rnn = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

    # Train the model
    _train_torch(model, X_train_rnn, y_train)

    # Predict
    y_pred_rnn = _predict_torch(model, X_test_rnn)
    mape = mean_absolute_percentage_error(y_test, y_pred_rnn)

    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(y_test, label="Actual", color="Blue")
    plt.plot(y_pred_rnn, label="Predicted", color="Red")
    plt.title(f"Recurrent Neural Network Forecast \n MAPE: {mape:.3f}")
    plt.legend()
    plt.savefig("RNN_forecast.png")
    plt.show()


    # > The graphs illustrate how neural networks can be great at matching
    # > complex time series but with that level of mimicry comes worries about
    # > overfitting.

    # I built these using Tensorflow directly. There are lots of other ways to
    # work with neural networks in time series like N-BEATS through Darts.

    # [**N-BEATS for Time Series Forecasting in Python**\
    # *N-BEATS (Neural Basis Expansion Analysis for Time Series) is a deep
    # learning model specifically designed for
    # time...*medium.com](https://medium.com/@kylejones_47003/n-beats-for-time-series-forecasting-in-python-b4a61858fe49 "https://medium.com/@kylejones_47003/n-beats-for-time-series-forecasting-in-python-b4a61858fe49")[](https://medium.com/@kylejones_47003/n-beats-for-time-series-forecasting-in-python-b4a61858fe49)
    #### Real world data: ERCOT Load Data
    # Initially I used simulated data. But what about real data? Let's use
    # data from ERCOT, the grid balancing authority in Texas.


    # <figcaption>Ercot Demand data</figcaption>


    # The RNN is much better than the basic Neural Network.


    # There is no clear benefit from the LSTM versus RNN.


    # Both the RNN and LSTM are extremely good at modeling this data.

    ### So what?
    # Neural networks are fast and can handle nonlinear patterns and
    # multivariate inputs. Tensorflow (with Keras) is really easy to use. But
    # Neural Networks can overfit the data and they are a "black box" so the
    # model itself is something we can decompose.
    # By [Kyle Jones](https://medium.com/@kyle-t-jones) on
    # [January 12, 2025](https://medium.com/p/0479918268f1).

    # [Canonical
    # link](https://medium.com/@kyle-t-jones/neural-networks-for-time-series-with-tensorflow-keras-in-python-0479918268f1)

    # Exported from [Medium](https://medium.com) on November 10, 2025.


if __name__ == "__main__":
    main()
