"""Exact documented convolution/pooling/ReLU-LSTM backbone."""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np

from climate_emulator.data.preprocessing import PreparedData
from climate_emulator.models.base import Emulator


def build_backbone(shape: tuple[int, int, int, int], architecture: dict, l2: float, zero_output: bool = False):
    """Build an uncompiled Keras model; TensorFlow remains an optional import."""
    import tensorflow as tf
    layers = tf.keras.layers
    regularizer = tf.keras.regularizers.l2(l2)
    history, nlat, nlon, channels = shape
    if architecture["lstm_activation"] != "relu":
        raise ValueError("The registered ClimateBench LSTM uses ReLU")
    if nlat < 2 or nlon < 2:
        raise ValueError("Average pooling requires at least two cells per spatial axis")
    inputs = layers.Input(shape=shape)
    x = layers.TimeDistributed(layers.Conv2D(architecture["filters"], architecture["kernel_size"], padding="same", activation="relu", kernel_regularizer=regularizer))(inputs)
    x = layers.TimeDistributed(layers.AveragePooling2D(2))(x)
    x = layers.TimeDistributed(layers.GlobalAveragePooling2D())(x)
    x = layers.LSTM(architecture["lstm_units"], activation="relu", recurrent_activation="sigmoid", kernel_regularizer=regularizer, recurrent_regularizer=regularizer)(x)
    output = layers.Dense(nlat * nlon, activation="linear", kernel_initializer="zeros" if zero_output else "glorot_uniform", bias_initializer="zeros", kernel_regularizer=regularizer)(x)
    return tf.keras.Model(inputs, output, name="climatebench_cnn_lstm")


class CNNLSTMEmulator(Emulator):
    """Pure neural physical-field emulator trained through the common trainer."""

    def __init__(self, config: dict, seed: int, epochs: int | None = None, log_directory: str | Path | None = None):
        self.config, self.seed, self.epochs = config, seed, epochs
        self.log_directory = Path(log_directory) if log_directory else None
        self.network = None
        self.variant = "cnn_lstm"

    def fit(self, data: PreparedData, validation: PreparedData | None = None) -> "CNNLSTMEmulator":
        """Train direct scaled predictions, using validation only during selection."""
        from climate_emulator.training.trainer import train_neural
        from climate_emulator.utils import set_seed
        set_seed(self.seed, tensorflow=True, deterministic=self.config["training"]["deterministic"])
        self.mask = data.mask.copy()
        self.scale = 1.0 if self.config["training"]["protocol"] == "legacy" else data.scale
        self.shape = data.target.shape[1:]
        self.network = build_backbone((data.history.indices.shape[1], *self.shape, 4), self.config["models"]["cnn_lstm"], self.config["training"]["kernel_l2"])
        self.history = train_neural(self, data, validation, self.config["training"], self.epochs, self.log_directory)
        self.selected_epochs = self.history["best_epoch"]
        self.training_keys = list(data.keys)
        return self

    def training_target(self, data: PreparedData) -> np.ndarray:
        """Produce numeric scaled targets; masked placeholders have zero loss weight."""
        target = np.zeros_like(data.target, dtype=np.float32)
        target[:, self.mask] = data.target[:, self.mask] / self.scale
        return target.reshape(len(data), -1)

    def forward_scaled(self, output):
        """Differentiable output transformation used by the common trainer."""
        return output

    def predict_raw(self, data: PreparedData) -> np.ndarray:
        """Perform batched eager inference without materializing every history."""
        if self.network is None:
            raise RuntimeError("Neural network has not been fitted or restored")
        batch_size = self.config["training"]["inference_batch_size"]
        pieces = []
        for start in range(0, len(data), batch_size):
            indices = np.arange(start, min(start + batch_size, len(data)))
            pieces.append(self.network(data.history.batch(indices), training=False).numpy())
        return np.concatenate(pieces).reshape(len(data), *self.shape)

    def predict(self, data: PreparedData) -> np.ndarray:
        """Return physical anomalies and preserve invalid spatial cells as NaN."""
        result = self.predict_raw(data) * self.scale
        result[:, ~self.mask] = np.nan
        return result

    def save(self, directory: str | Path) -> None:
        """Store architecture/weights separately from trusted Python metadata."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.network.save(directory / "network.keras")
        state = {k: v for k, v in self.__dict__.items() if k != "network"}
        joblib.dump({"class": type(self).__name__, "state": state}, directory / "neural_state.joblib")

    @staticmethod
    def load(directory: str | Path) -> "CNNLSTMEmulator":
        """Restore trusted neural state and original ReLU architecture."""
        import tensorflow as tf
        from climate_emulator.models.residual import ResidualEmulator
        directory = Path(directory)
        stored = joblib.load(directory / "neural_state.joblib")
        cls = ResidualEmulator if stored["class"] == "ResidualEmulator" else CNNLSTMEmulator
        obj = cls.__new__(cls)
        obj.__dict__.update(stored["state"])
        obj.network = tf.keras.models.load_model(directory / "network.keras", compile=False)
        return obj
