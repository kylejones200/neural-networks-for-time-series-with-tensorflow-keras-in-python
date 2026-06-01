import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset


class _LSTMForecaster(nn.Module):
    def __init__(self, n_features: int, hidden: int = 64, n_layers: int = 1):
        super().__init__()
        self.lstm = nn.LSTM(
            n_features, hidden, num_layers=n_layers, batch_first=True
        )
        self.fc = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


class _MLPForecaster(nn.Module):
    def __init__(self, n_features: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _predict_torch(model: nn.Module, X_test) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        return model(torch.FloatTensor(X_test)).numpy()


def _train_torch(
    model: nn.Module,
    X_train,
    y_train,
    *,
    epochs: int = 15,
    batch_size: int = 8,
) -> nn.Module:
    X_t = torch.FloatTensor(X_train)
    y_t = torch.FloatTensor(y_train)
    if y_t.dim() == 1:
        y_t = y_t.unsqueeze(1)
    loader = DataLoader(TensorDataset(X_t, y_t), batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()
    for _ in range(epochs):
        model.train()
        for xb, yb in loader:
            optimizer.zero_grad()
            criterion(model(xb), yb).backward()
            optimizer.step()
    return model


def create_features(series, lag=3):
    X, y = [], []
    for i in range(len(series) - lag):
        X.append(series[i : i + lag])
        y.append(series[i + lag])
    return np.array(X), np.array(y)


def build_an_rnn_model(X_test, X_train, lag, y_test, y_train) -> None:
    model = _LSTMForecaster(1, hidden=32, n_layers=1)
    X_train_rnn = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test_rnn = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
    _train_torch(model, X_train_rnn, y_train, epochs=10)
    y_pred_rnn = _predict_torch(model, X_test_rnn).flatten()
    mape = mean_absolute_percentage_error(y_test, y_pred_rnn)
    plt.figure(figsize=(10, 6))
    plt.plot(y_test, label="Actual")
    plt.plot(y_pred_rnn, label="Predicted")
    plt.title(f"RNN forecast MAPE: {mape:.3f}")
    plt.legend()
    plt.savefig("RNN_forecast.png")
    plt.close()


def main() -> None:
    np.random.seed(42)
    data = np.cumsum(np.random.randn(200)) + 100
    lag = 3
    X, y = create_features(data, lag=lag)
    tscv = TimeSeriesSplit(n_splits=5)
    train_idx, test_idx = list(tscv.split(X))[-1]
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    scaler = MinMaxScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    mlp = _MLPForecaster(lag)
    _train_torch(mlp, X_train_s, y_train, epochs=10)
    y_pred = _predict_torch(mlp, X_test_s).flatten()
    mape = mean_absolute_percentage_error(y_test, y_pred)
    plt.figure(figsize=(10, 6))
    plt.plot(y_test, label="Actual")
    plt.plot(y_pred, label="Predicted")
    plt.title(f"MLP forecast MAPE: {mape:.3f}")
    plt.legend()
    plt.savefig("NN_forecast.png")
    plt.close()
    build_an_rnn_model(X_test_s, X_train_s, lag, y_test, y_train)
    print("Neural network demos complete.")


if __name__ == "__main__":
    main()
