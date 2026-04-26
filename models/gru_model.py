import os
import time
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, GRU, Dropout, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from config import (
    HIDDEN_DIM,
    DROPOUT,
    LR,
    BATCH_SIZE,
    MAX_EPOCHS,
    PATIENCE,
    INPUT_DIM,
    LATENCY_ITERATIONS,
    MODELS_DIR,
)


def build_model(input_shape):
    """Build and compile a single-layer GRU classification model.

    Parameters
    ----------
    input_shape : tuple
        Shape of one input sample, e.g. (60, 10) for 60 timesteps and
        10 features.

    Returns
    -------
    model : tensorflow.keras.Model
        Compiled Keras Sequential model ready for training.
    """
    model = Sequential([
        Input(shape=input_shape),
        GRU(units=HIDDEN_DIM, return_sequences=False),
        Dropout(rate=DROPOUT),
        Dense(1, activation="sigmoid"),
    ])

    model.compile(
        optimizer=Adam(learning_rate=LR),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    model.summary()
    return model


def prepare_sequences(df, feature_names, target_col, seq_len):
    """Convert a flat dataframe into sliding-window sequences for the GRU.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe containing feature columns and the target column.
    feature_names : list of str
        Names of the feature columns to use as model inputs.
    target_col : str
        Name of the column containing the raw signal labels (+1 / -1).
    seq_len : int
        Number of consecutive timesteps in each input sequence.

    Returns
    -------
    X : numpy.ndarray, shape (n_samples, seq_len, n_features), dtype float32
        Sliding-window input sequences.
    y : numpy.ndarray, shape (n_samples,), dtype float32
        Binary target array where -1 is mapped to 0 and +1 stays 1.
    """
    df = df.dropna()

    features = df[feature_names].values
    targets = df[target_col].values

    X, y = [], []
    for i in range(len(df) - seq_len):
        X.append(features[i : i + seq_len])
        y.append(targets[i + seq_len])

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)

    # Map -1 -> 0 for binary cross-entropy compatibility
    y = np.where(y == -1, 0.0, 1.0).astype(np.float32)

    return X, y


def train(model, X_train, y_train, X_val, y_val):
    """Train the GRU model with early stopping and model checkpointing.

    Parameters
    ----------
    model : tensorflow.keras.Model
        Compiled Keras model returned by build_model().
    X_train : numpy.ndarray
        Training sequences of shape (n_samples, seq_len, n_features).
    y_train : numpy.ndarray
        Binary training labels of shape (n_samples,).
    X_val : numpy.ndarray
        Validation sequences of shape (n_samples, seq_len, n_features).
    y_val : numpy.ndarray
        Binary validation labels of shape (n_samples,).

    Returns
    -------
    history : tensorflow.keras.callbacks.History
        Keras History object containing per-epoch training metrics.
    training_time : float
        Wall-clock training time in seconds.
    """
    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=PATIENCE,
        restore_best_weights=True,
    )

    checkpoint = ModelCheckpoint(
        filepath=os.path.join(MODELS_DIR, "gru_best.keras"),
        monitor="val_loss",
        save_best_only=True,
    )

    start_time = time.perf_counter()

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        batch_size=BATCH_SIZE,
        epochs=MAX_EPOCHS,
        callbacks=[early_stopping, checkpoint],
    )

    end_time = time.perf_counter()
    training_time = end_time - start_time

    print(f"Training time: {training_time:.2f} seconds")

    return history, training_time


def evaluate(model, X_test, y_test):
    """Evaluate the trained GRU model on the test set.

    Computes classification metrics, FLOPs, and inference latency.

    Parameters
    ----------
    model : tensorflow.keras.Model
        Trained Keras model.
    X_test : numpy.ndarray
        Test sequences of shape (n_samples, seq_len, n_features).
    y_test : numpy.ndarray
        Binary test labels (0 / 1) of shape (n_samples,).

    Returns
    -------
    dict
        Dictionary with the following keys:
        - accuracy : float
        - precision : float
        - recall : float
        - f1 : float
        - confusion_matrix : numpy.ndarray of shape (2, 2)
        - n_signals : int — number of test samples
        - flops : int — analytic FLOPs for one forward pass
        - latency_ms : float — mean inference latency in milliseconds
    """
    # Generate predictions and convert sigmoid output to +1 / -1 labels
    y_prob = model.predict(X_test)
    y_pred_binary = (y_prob.flatten() >= 0.5).astype(int)
    y_pred = np.where(y_pred_binary == 1, 1, -1)

    # Convert y_test from binary (0/1) back to original labels (+1 / -1)
    y_true = np.where(y_test == 0, -1, 1)

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
    rec = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
    f1 = f1_score(y_true, y_pred, pos_label=1, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    # Analytic FLOPs: 6*d^2 + 6*d*m + 9*d
    flops = 6 * HIDDEN_DIM ** 2 + 6 * HIDDEN_DIM * INPUT_DIM + 9 * HIDDEN_DIM

    # Inference latency: average over LATENCY_ITERATIONS single-sample passes
    x_single = X_test[:1]
    start = time.perf_counter()
    for _ in range(LATENCY_ITERATIONS):
        model.predict(x_single, verbose=0)
    end = time.perf_counter()
    total_us = (end - start) * 1_000_000  # seconds -> microseconds
    latency_ms = (total_us / LATENCY_ITERATIONS) / 1000  # microseconds -> milliseconds

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "confusion_matrix": cm,
        "n_signals": len(y_true),
        "flops": flops,
        "latency_ms": latency_ms,
    }
