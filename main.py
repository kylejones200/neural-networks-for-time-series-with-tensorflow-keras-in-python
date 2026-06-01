#!/usr/bin/env python3
"""
Neural Networks for Time Series with PyTorch LSTM forecasting.
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from core import (  # noqa: E402
    _predict_torch,
    _train_torch,
    build_lstm_model,
    plot_forecast,
    prepare_data,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def load_config(config_path: Path | None = None) -> dict:
    if config_path is None:
        config_path = ROOT / "config.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Neural Networks for Time Series")
    parser.add_argument("--config", type=Path, default=None, help="Path to config file")
    parser.add_argument("--data-path", type=Path, default=None, help="Path to data file")
    parser.add_argument("--output-dir", type=Path, default=None, help="Output directory")
    args = parser.parse_args()
    config = load_config(args.config)
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else ROOT / config["output"]["figures_dir"]
    )
    output_dir.mkdir(exist_ok=True)

    if args.data_path and args.data_path.exists():
        df = pd.read_csv(args.data_path)
    elif config["data"]["generate_synthetic"]:
        np.random.seed(config["data"]["seed"])
        dates = pd.date_range(
            "2020-01-01", periods=config["data"]["n_periods"], freq="D"
        )
        values = np.sin(np.arange(config["data"]["n_periods"]) / 10) + np.random.normal(
            0, 0.1, config["data"]["n_periods"]
        )
        df = pd.DataFrame({"date": dates, "value": values})
    else:
        raise ValueError("No data source specified")

    X_train, X_test, y_train, y_test, scaler = prepare_data(
        df,
        config["data"]["value_column"],
        config["model"]["lag"],
        config["model"]["train_size"],
    )
    model = build_lstm_model((X_train.shape[1], 1), config["model"]["lstm_units"])
    _train_torch(
        model,
        X_train,
        y_train,
        epochs=config["model"]["epochs"],
        batch_size=config["model"]["batch_size"],
        validation_split=config["model"]["validation_split"],
    )
    y_pred = _predict_torch(model, X_test)

    y_test_inverse = scaler.inverse_transform(y_test.reshape(-1, 1))
    y_pred_inverse = scaler.inverse_transform(y_pred)
    plot_forecast(
        y_test_inverse.flatten(),
        y_pred_inverse.flatten(),
        "LSTM Forecast vs Actual",
        output_dir / "lstm_forecast.png",
    )
    logging.info("Analysis complete. Figures saved to %s", output_dir)


if __name__ == "__main__":
    main()
