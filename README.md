# Stock Forecaster

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python: 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

An LSTM-based stock price forecaster using OHLC (Open, High, Low, Close) data with TensorFlow/Keras.

## Features

- OHLC data preprocessing and normalization
- LSTM sequence building for time series forecasting
- Model training with early stopping and checkpointing
- Prediction and visualization of future price trends
- Evaluation against baseline (mean, naive)
- Command-line interface
- Comprehensive test suite
- GitHub Actions CI

## Installation

```bash
git clone https://github.com/rudrashortguy/stock-forecaster.git
cd stock-forecaster
pip install -e .
```

For development:
```bash
pip install -e ".[dev]"
```

## Usage

### Training a model

```bash
stock-forecaster train --input data/stock_data.csv --epochs 50 --output models/lstm_model.keras
```

### Making predictions

```bash
stock-forecaster predict --model models/lstm_model.keras --input data/test_data.csv --output predictions.csv
```

### Python API

```python
from stock_forecaster.preprocessor import load_and_normalize
from stock_forecaster.model import build_lstm_model, train_model
from stock_forecaster.evaluator import evaluate

# Load and preprocess data
X, y, scaler = load_and_normalize("data/stock_data.csv")

# Build and train model
model = build_lstm_model(X.shape[-1])
history = train_model(model, X, y, epochs=50)

# Evaluate
metrics = evaluate(model, X, y)
print(f"RMSE: {metrics['rmse']:.4f}")
```

## Project Structure

```
stock-forecaster/
├── data/                  # Data storage (not tracked)
├── stock_forecaster/      # Source code
│   ├── __init__.py
│   ├── preprocessor.py    # Data loading and normalization
│   ├── model.py           # LSTM model definition
│   ├── trainer.py         # Training utilities
│   └── evaluator.py       # Evaluation metrics
├── tests/                 # Unit tests
├── models/                # Trained models (not tracked)
├── README.md
├── LICENSE
├── CONTRIBUTING.md
├── pyproject.toml
└── .github/workflows/tests.yml
```

## Tech Stack

- **Language**: Python 3.8+
- **Deep Learning**: TensorFlow / Keras
- **Data Processing**: pandas, NumPy
- **Testing**: pytest
- **CI/CD**: GitHub Actions

## Roadmap

- [x] LSTM model implementation
- [x] OHLC data preprocessing
- [x] Model training and evaluation
- [ ] Add GRU and BiLSTM variants
- [ ] Integration with Yahoo Finance API
- [ ] Real-time prediction pipeline
- [ ] Hyperparameter optimization

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.