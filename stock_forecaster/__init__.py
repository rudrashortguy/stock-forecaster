"""Stock Forecaster - LSTM-based stock price prediction."""

from .preprocessor import load_and_normalize, create_sequences
from .model import build_lstm_model
from .trainer import train_model, save_model, load_model
from .evaluator import evaluate, plot_predictions

__version__ = "1.0.0"
__all__ = [
    "load_and_normalize",
    "create_sequences",
    "build_lstm_model",
    "train_model",
    "save_model",
    "load_model",
    "evaluate",
    "plot_predictions",
]
