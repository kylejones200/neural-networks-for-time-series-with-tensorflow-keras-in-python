# Neural Networks for Time Series with Tensorflow Keras in Python

Neural networks excel at capturing complex temporal patterns that
traditional statistical methods struggle to model.

:::::### Neural Networks for Time Series with Tensorflow Keras in Python 

#### Neural networks excel at capturing complex temporal patterns that traditional statistical methods struggle to model.
Neural networks are good at capturing complex patterns and
relationships. They can deal with nonlinear trends and interaction
effects that traditional statistical methods can't handle. The trade off
is that the process is a black box --- we don't know why the prediction
is the way it is. We just know it works.

There are lots of ways to implement neural networks.

In this project, I build three neural networks using Tensorflow (Keras):
basic feedforward neural networks, recurrent neural networks (RNNs), and
long short-term memory (LSTM). Exponential Smoothing and ARIMA are great
for data with linear patterns, but traditional methods struggle with
n**onlinear relationships.** Neural networks can model complex
interactions between time-dependent features. Neural networks can also
handle multiple inputs (multivartiate data) which will improve
forecasting.

There are also no assumptions of stationarity or predefined trends for
Neural Networks. That means we can use them without having to run tests
like Dickey-Fuller.

### Feedforward Neural Networks for Time Series
A feedforward neural network can be used to predict future values by
using lagged observations as input features. While basic, it's a good
starting point for time series forecasting.



### Recurrent Neural Networks (RNNs)
Recurrent neural networks are designed to handle sequential data by
maintaining a hidden state that captures information about previous time
steps. This makes them ideal for time series.



### Long Short-Term Memory (LSTM) Networks
LSTMs are a type of RNN that solve the **vanishing gradient problem**
(ohh! ahh!). Basically LSTMs remember previous values which gives them a
leg up on other NNs for complex time series with long-range patterns.
:::::::```python
"""
Long Short-Term Memory (LSTM) Networks for time series
"""

from tensorflow.keras.layers import LSTM

# Build an LSTM model
model = tf.keras.Sequential([
    LSTM(50, activation='relu', input_shape=(lag, 1)),
    tf.keras.layers.Dense(1)
])

model.compile(optimizer='adam', loss='mse')
model.summary()

# Train the model
model.fit(X_train_rnn, y_train, epochs=50, batch_size=8, verbose=1, validation_split=0.1)

# Predict
y_pred_lstm = model.predict(X_test_rnn)
mape = mean_absolute_percentage_error(y_test, y_pred_lstm)
# Plot results
plt.figure(figsize=(10, 6))
plt.plot(y_test, label='Actual', color='Blue')
plt.plot(y_pred_lstm, label='Predicted', color='Red')
plt.title(f'LSTM Forecast. MAPE: {mape:.3f}')
plt.legend()
plt.savefig("LSTM_forecast.png")
plt.show()
