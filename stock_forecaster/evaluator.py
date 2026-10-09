"""Evaluation module for stock forecaster."""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def evaluate(model, X_test: np.ndarray, y_test: np.ndarray, scaler=None) -> dict:
    """
    Evaluate model performance.

    Args:
        model: Trained Keras model.
        X_test: Test sequences.
        y_test: True target values.
        scaler: Optional scaler for denormalization.

    Returns:
        Dictionary of evaluation metrics.
    """
    y_pred = model.predict(X_test, verbose=0).flatten()

    # Metrics on normalized scale
    metrics = {
        "rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
        "mae": mean_absolute_error(y_test, y_pred),
        "r2": r2_score(y_test, y_pred),
    }

    # If scaler provided, also compute on denormalized scale
    if scaler is not None:
        inv_pred = scaler.inverse_transform(
            np.zeros((len(y_pred), scaler.data_max_.shape[0]))
        )
        inv_pred[:, -1] = y_pred
        inv_true = scaler.inverse_transform(
            np.zeros((len(y_test), scaler.data_max_.shape[0]))
        )
        inv_true[:, -1] = y_test
        metrics.update({
            "rmse_denorm": np.sqrt(mean_squared_error(inv_true, inv_pred)),
            "mae_denorm": mean_absolute_error(inv_true, inv_pred),
        })

    return metrics


def print_evaluation_report(metrics: dict) -> None:
    """Print formatted evaluation report."""
    print("=" * 50)
    print("MODEL EVALUATION REPORT")
    print("=" * 50)
    print(f"RMSE:         {metrics.get('rmse', 'N/A'):.4f}")
    print(f"MAE:          {metrics.get('mae', 'N/A'):.4f}")
    print(f"R² Score:     {metrics.get('r2', 'N/A'):.4f}")
    if "rmse_denorm" in metrics:
        print(f"RMSE (denorm):{metrics['rmse_denorm']:.4f}")
    print("=" * 50)


def plot_predictions(y_true: np.ndarray, y_pred: np.ndarray, title: str = "Predictions vs Actual"):
    """Plot actual vs predicted values."""
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(y_true, label="Actual", color="blue")
    ax.plot(y_pred, label="Predicted", color="red", linestyle="--")
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Price")
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    return fig