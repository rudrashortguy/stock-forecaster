"""Tests for stock forecaster."""

import pytest
import numpy as np
import pandas as pd
import tempfile
import os

from stock_forecaster.preprocessor import load_and_normalize, create_sequences
from stock_forecaster.model import build_lstm_model
from stock_forecaster.evaluator import evaluate, print_evaluation_report


# Fixtures
@pytest.fixture
def sample_data():
    """Create sample OHLC data."""
    np.random.seed(42)
    n = 200
    dates = pd.date_range("2020-01-01", periods=n, freq="D")
    close = 100 + np.cumsum(np.random.randn(n) * 0.5)
    open_ = close + np.random.randn(n) * 0.1
    high = close + abs(np.random.randn(n) * 0.2)
    low = close - abs(np.random.randn(n) * 0.2)
    volume = np.random.randint(1000, 10000, n)

    df = pd.DataFrame({
        "Date": dates,
        "Open": open_,
        "High": high,
        "Low": low,
        "Close": close,
        "Volume": volume,
    })
    return df


@pytest.fixture
def sample_csv(sample_data):
    """Write sample data to temp CSV."""
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
        sample_data.to_csv(f, index=False)
        path = f.name
    yield path
    os.unlink(path)


class TestPreprocessor:
    def test_load_and_normalize_shapes(self, sample_csv):
        X, y, scaler, prices = load_and_normalize(sample_csv, lookback=30)
        assert X.ndim == 3
        assert X.shape[0] == len(prices) - 30
        assert y.shape[0] == X.shape[0]

    def test_load_and_normalize_returns_scaler(self, sample_csv):
        _, _, scaler, _ = load_and_normalize(sample_csv, lookback=20)
        assert scaler is not None

    def test_load_and_normalize_missing_column_raises(self, sample_csv):
        with pytest.raises(ValueError, match="Target column"):
            load_and_normalize(sample_csv, target_column="MissingCol")

    def test_create_sequences(self):
        data = np.random.rand(50, 5)
        X, y = create_sequences(data, lookback=10)
        assert X.shape == (40, 10, 5)
        assert y.shape == (40,)


class TestModel:
    def test_build_lstm_model(self):
        model = build_lstm_model((60, 5))
        assert model is not None
        assert len(model.layers) == 6  # 2 LSTM + 2 Dropout + 2 Dense

    def test_build_lstm_model_compile(self):
        model = build_lstm_model((60, 5))
        assert model.optimizer is not None
        assert model.loss == "mean_squared_error"


class TestEvaluator:
    def test_evaluate_returns_metrics(self):
        model = build_lstm_model((10, 3))
        X = np.random.rand(20, 10, 3)
        y = np.random.rand(20)
        metrics = evaluate(model, X, y)
        assert "rmse" in metrics
        assert "mae" in metrics
        assert "r2" in metrics

    def test_evaluate_rmse_range(self):
        model = build_lstm_model((10, 3))
        X = np.random.rand(50, 10, 3)
        y = np.random.rand(50)
        metrics = evaluate(model, X, y)
        assert 0 <= metrics["rmse"]

    def test_print_evaluation_report(self, capsys):
        model = build_lstm_model((10, 3))
        X = np.random.rand(20, 10, 3)
        y = np.random.rand(20)
        metrics = evaluate(model, X, y)
        print_evaluation_report(metrics)
        captured = capsys.readouterr()
        assert "MODEL EVALUATION REPORT" in captured.out


class TestIntegration:
    def test_end_to_end(self, sample_csv):
        """Full pipeline test."""
        X, y, scaler, _ = load_and_normalize(sample_csv, lookback=20)

        model = build_lstm_model(X[0].shape)
        model.fit(X[:80], y[:80], epochs=5, batch_size=16, verbose=0)

        metrics = evaluate(model, X[80:], y[80:], scaler)
        assert metrics["rmse"] >= 0