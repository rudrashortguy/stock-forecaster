"""Model definition for LSTM stock forecaster."""

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout


def build_lstm_model(input_shape: tuple, units: int = 50) -> Sequential:
    """
    Build an LSTM model for stock price prediction.

    Args:
        input_shape: Shape of input data (timesteps, features).
        units: Number of LSTM units.

    Returns:
        Compiled Keras model.
    """
    model = Sequential([
        LSTM(units, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(units, return_sequences=False),
        Dropout(0.2),
        Dense(25),
        Dense(1),
    ])

    model.compile(optimizer="adam", loss="mean_squared_error")
    return model


def predict_next(model, X: np.ndarray, scaler, step: int = 1) -> np.ndarray:
    """
    Predict next 'step' prices using the model.

    Args:
        model: Trained LSTM model.
        X: Input sequences.
        scaler: Fitted MinMaxScaler.
        step: Number of days to predict.

    Returns:
        Array of predicted prices (denormalized).
    """
    predictions = []
    current_seq = X[-1].reshape(1, X.shape[1], -1)

    for _ in range(step):
        pred = model.predict(current_seq, verbose=0)[0, 0]
        predictions.append(pred)
        # Shift sequence for next prediction
        new_point = np.zeros(current_seq.shape[1])
        new_point[-1] = pred
        current_seq = np.concatenate([current_seq[:, 1:], current_seq[:, -1:] + new_point.reshape(1, 1, -1)], axis=1)

    # Denormalize
    inverse = np.zeros((len(predictions), scaler.data_max_.shape[0]))
    inverse[:, -1] = predictions
    return scaler.inverse_transform(inverse)[:, -1]