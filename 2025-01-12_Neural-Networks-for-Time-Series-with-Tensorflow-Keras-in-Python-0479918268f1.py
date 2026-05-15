# Description: Short example for Neural Networks for Time Series with Tensorflow Keras in Python.


import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import SimpleRNN

np.random.seed(42)

"""
Feedforward Neural Network for Simple Forecasting
"""


# Generate synthetic data
time = np.arange(100)
data = 10 + 0.5 * time + np.sin(0.2 * time) + np.random.normal(scale=1.0, size=100)


# Create lagged features
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
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Dense(64, activation="relu", input_shape=(lag,)),
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dense(1),
        ]
    )

    model.compile(optimizer="adam", loss="mse")
    model.summary()

    # Train the model
    model.fit(X_train, y_train, epochs=50, batch_size=8, verbose=1, validation_split=0.1)

    # Evaluate and predict
    y_pred = model.predict(X_test)
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
    model = tf.keras.Sequential(
        [SimpleRNN(50, activation="relu", input_shape=(lag, 1)), tf.keras.layers.Dense(1)]
    )

    model.compile(optimizer="adam", loss="mse")
    model.summary()

    # Reshape input for RNN (samples, timesteps, features)
    X_train_rnn = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test_rnn = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

    # Train the model
    model.fit(
        X_train_rnn, y_train, epochs=50, batch_size=8, verbose=1, validation_split=0.1
    )

    # Predict
    y_pred_rnn = model.predict(X_test_rnn)
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
