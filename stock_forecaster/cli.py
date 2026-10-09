"""Command-line interface for stock forecaster."""

import argparse
import sys
from pathlib import Path

from .preprocessor import load_and_normalize
from .model import build_lstm_model
from .trainer import train_model, save_model
from .evaluator import evaluate, print_evaluation_report


def main():
    parser = argparse.ArgumentParser(description="Stock Forecaster - LSTM prediction")
    subparsers = parser.add_subparsers(dest="command")

    # Train command
    train_p = subparsers.add_parser("train", help="Train an LSTM model")
    train_p.add_argument("--input", required=True, help="Path to OHLC CSV")
    train_p.add_argument("--epochs", type=int, default=50)
    train_p.add_argument("--output", default="models/lstm_model.keras")
    train_p.add_argument("--lookback", type=int, default=60)

    args = parser.parse_args()

    if args.command == "train":
        print("Loading data...")
        X, y, scaler, prices = load_and_normalize(args.input, lookback=args.lookback)

        print(f"Building LSTM model (X shape: {X.shape})...")
        model = build_lstm_model(X[0].shape)

        print("Training...")
        history = train_model(
            model, X, y, epochs=args.epochs, validation_split=0.2
        )

        print(f"Saving model to {args.output}...")
        save_model(model, args.output, scaler)
        print("Done!")


if __name__ == "__main__":
    main()