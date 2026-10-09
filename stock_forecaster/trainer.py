"""Training utilities for stock forecaster."""

import joblib
import numpy as np
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


def train_model(
    model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    epochs: int = 50,
    batch_size: int = 32,
    validation_split: float = 0.2,
    output_path: str = None,
) -> dict:
    """
    Train the LSTM model with early stopping and checkpointing.

    Args:
        model: Compiled Keras model.
        X_train: Training sequences.
        y_train: Training targets.
        epochs: Maximum training epochs.
        batch_size: Training batch size.
        validation_split: Fraction for validation.
        output_path: Path to save best model.

    Returns:
        Training history dictionary.
    """
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
    ]

    if output_path:
        callbacks.append(
            ModelCheckpoint(output_path, save_best_only=True, monitor="val_loss")
        )

    history = model.fit(
        X_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        callbacks=callbacks,
        verbose=1,
    )

    return history.history


def save_model(model, filepath: str, scaler=None) -> None:
    """Save model and optional scaler."""
    model.save(filepath)
    if scaler is not None:
        joblib.dump(scaler, filepath.replace(".keras", "_scaler.joblib"))


def load_model(filepath: str, custom_objects: dict = None):
    """Load a saved Keras model."""
    from tensorflow.keras.models import load_model
    return load_model(filepath, custom_objects=custom_objects or {})


def load_scaler(filepath: str):
    """Load a scaler saved alongside the model."""
    scaler_path = filepath.replace(".keras", "_scaler.joblib")
    try:
        return joblib.load(scaler_path)
    except FileNotFoundError:
        return None