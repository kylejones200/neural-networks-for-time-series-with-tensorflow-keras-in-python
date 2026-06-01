# Neural Networks for Time Series with TensorFlow/Keras

This project demonstrates LSTM neural networks for time series forecasting.

## Business context

Neural networks excel at capturing complex temporal patterns that traditional statistical methods struggle to model.

Neural networks are good at capturing complex patterns and relationships. They can deal with nonlinear trends and interaction effects that traditional statistical methods can't handle. The trade off is that the process is a black box --- we don't know why the prediction is the way it is. We just know it works.

In this project, I build three neural networks using Tensorflow (Keras): basic feedforward neural networks, recurrent neural networks (RNNs), and long short-term memory (LSTM). Exponential Smoothing and ARIMA are great for data with linear patterns, but traditional methods struggle with nonlinear relationships. Neural networks can model complex interactions between time-dependent features. Neural networks can also handle multiple inputs (multivartiate data) which will improve forecasting.

## Article

Medium article: [Neural Networks for Time Series with TensorFlow/Keras in Python](https://medium.com/@kylejones_47003/neural-networks-for-time-series-with-tensorflow-keras-in-python-0479918268f1)

## Project Structure

```
.
├── README.md           # This file
├── main.py            # Main entry point
├── config.yaml        # Configuration file
├── requirements.txt   # Python dependencies
├── src/               # Core functions
│   ├── core.py        # LSTM model functions
│   └── plotting.py    # Tufte-style plotting utilities
├── tests/             # Unit tests
├── data/              # Data files
└── images/            # Generated plots and figures
```

## Configuration

Edit `config.yaml` to customize:
- Data source or synthetic generation
- Model parameters (lag, LSTM units, epochs)
- Training settings
- Output settings

## LSTM Model

Long Short-Term Memory networks:
- Capture long-term dependencies
- Handle sequential patterns
- Suitable for time series forecasting

## Caveats

- By default, generates synthetic time series data.
- LSTM training can be slow for large datasets.
- Requires sufficient data for training and validation.
- GPU recommended for faster training but not required.

## Disclaimer

Educational/demo code only. Not financial, safety, or engineering advice. Use at your own risk. Verify results independently before any production or operational use.

## License

MIT — see [LICENSE](LICENSE).