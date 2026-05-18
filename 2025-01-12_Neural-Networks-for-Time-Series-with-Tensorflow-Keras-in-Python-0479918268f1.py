import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset


class _LSTMForecaster(nn.Module):
    """LSTM forecaster (auto-generated PyTorch replacement for Keras Sequential)."""

    def __init__(
        self,
        n_features: int,
        hidden: int = 64,
        output_size: int = 1,
        n_layers: int = 0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            n_features,
            hidden,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0,
        )
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.fc(self.drop(out[:, -1, :]))


def _predict_torch(model: nn.Module, X_test) -> "np.ndarray":
    """Replace model.predict()."""
    model.eval()
    with torch.no_grad():
        return model(torch.FloatTensor(X_test)).numpy()


def _train_torch(
    model: nn.Module,
    X_train,
    y_train,
    *,
    epochs: int = 50,
    batch_size: int = 8,
    lr: float = 0.001,
    validation_split: float = 0.1,
    patience: int = 15,
) -> nn.Module:
    """Standard training loop replacing  + model.fit()."""
    X_t = torch.FloatTensor(X_train)
    y_t = torch.FloatTensor(y_train)
    if y_t.dim() == 1:
        y_t = y_t.unsqueeze(1)
    n_val = max(1, int(len(X_t) * validation_split))
    X_val, y_val = (X_t[-n_val:], y_t[-n_val:])
    X_tr, y_tr = (X_t[:-n_val], y_t[:-n_val])
    loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    best, wait = (float("inf"), 0)
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
            best, wait = (val_loss, 0)
        else:
            wait += 1
            if wait >= patience:
                break
    return model


def create_features(data, lag=3):
    X, y = ([], [])
    for i in range(len(data) - lag):
        X.append(data[i : i + lag])
        y.append(data[i + lag])
    return (np.array(X), np.array(y))


def build_an_rnn_model(X_test, X_train, lag, y_test, y_train) -> None:
    "\n    Basic RNN for Time Series\n"
    model = Sequential(
        [SimpleRNN(50, activation="relu", input_shape=(lag, 1)), nn.Dense(1)]
    )
    model.summary()
    X_train_rnn = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test_rnn = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
    _train_torch(model, X_train_rnn, y_train)
    y_pred_rnn = _predict_torch(model, X_test_rnn)
    mape = mean_absolute_percentage_error(y_test, y_pred_rnn)
    plt.figure(figsize=(10, 6))
    plt.plot(y_test, label="Actual", color="Blue")
    plt.plot(y_pred_rnn, label="Predicted", color="Red")
    plt.title(f"Recurrent Neural Network Forecast \n MAPE: {mape:.3f}")
    plt.legend()
    plt.savefig("RNN_forecast.png")
    plt.show()


def main() -> None:
    lag = 3
    X, y = create_features(data, lag=lag)
    tscv = TimeSeriesSplit(n_splits=5)
    train_idx, test_idx = list(tscv.split(X))[-1]
    X_train, X_test = (X[train_idx], X[test_idx])
    y_train, y_test = (y[train_idx], y[test_idx])
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    model = Sequential(
        [
            nn.Dense(64, activation="relu", input_shape=(lag,)),
            nn.Dense(32, activation="relu"),
            nn.Dense(1),
        ]
    )
    model.summary()
    _train_torch(model, X_train, y_train)
    y_pred = _predict_torch(model, X_test)
    mape = mean_absolute_percentage_error(y_test, y_pred)
    plt.figure(figsize=(10, 6))
    plt.plot(y_test, label="Actual", color="Blue")
    plt.plot(y_pred, label="Predicted", color="Red")
    plt.title(f"Feedforward Neural Network Forecast \n MAPE: {mape:.3f}")
    plt.legend()
    plt.savefig("NN_forecast.png")
    plt.show()
    build_an_rnn_model(X_test, X_train, lag, y_test, y_train)


if __name__ == "__main__":
    main()
