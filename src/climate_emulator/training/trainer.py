"""Common eager RMSprop trainer with exact effective-batch accumulation."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from climate_emulator.data.preprocessing import PreparedData
from climate_emulator.training.callbacks import Deadline, log_epoch


def train_neural(emulator, data: PreparedData, validation: PreparedData | None, config: dict, epochs: int | None, directory: Path | None) -> dict:
    """Select best epoch on validation, or refit for a fixed selected duration."""
    import tensorflow as tf
    network = emulator.network
    target = emulator.training_target(data)
    if not np.isfinite(target[:, emulator.mask.ravel()]).all():
        raise ValueError("Nonfinite training targets on fitted mask")
    weights = np.where(emulator.mask, data.weights, 0).ravel()
    if config["protocol"] == "legacy":
        weights = emulator.mask.ravel().astype(float)
    weights = tf.constant(weights / weights.sum(), dtype=tf.float32)
    optimizer = tf.keras.optimizers.RMSprop(learning_rate=config["learning_rate"])
    rng = np.random.default_rng(emulator.seed)
    deadline = Deadline(config["fit_budget_seconds"])
    count_epochs = epochs if epochs is not None else config["max_epochs"]
    select_duration = validation is not None and config["protocol"] != "legacy"
    if count_epochs < 1:
        raise ValueError("Training duration must be positive")
    if directory:
        directory.mkdir(parents=True, exist_ok=True)
    validation_target = emulator.training_target(validation) if validation is not None else None
    validation_weights = weights
    if validation is not None:
        valid = emulator.mask & np.isfinite(validation.target).all(axis=0)
        if not valid.any():
            raise ValueError("No common training/validation target cells")
        numeric_weights = np.where(valid, validation.weights, 0).ravel()
        if config["protocol"] == "legacy":
            numeric_weights = valid.ravel().astype(float)
        validation_weights = tf.constant(numeric_weights / numeric_weights.sum(), dtype=tf.float32)
        validation_target = np.nan_to_num(validation_target, nan=0.0, posinf=0.0, neginf=0.0)
    best, best_epoch, bad_epochs, best_weights, history = np.inf, 0, 0, None, []

    def loss_value(indices: np.ndarray, source: PreparedData, y: np.ndarray, training: bool):
        output = emulator.forward_scaled(network(source.history.batch(indices), training=training))
        active_weights = weights if training else validation_weights
        per_sample = tf.reduce_sum(tf.square(output - y[indices]) * active_weights, axis=1)
        return tf.reduce_mean(per_sample)

    for epoch in range(1, count_epochs + 1):
        order = rng.permutation(len(data))
        losses, batch_counts = [], []
        effective = config["effective_batch_size"]
        micro = config["micro_batch_size"]
        for offset in range(0, len(order), effective):
            deadline.check()
            batch = order[offset:offset + effective]
            accumulators = [tf.zeros_like(v) for v in network.trainable_variables]
            batch_loss = 0.0
            for position in range(0, len(batch), micro):
                selected = batch[position:position + micro]
                fraction = len(selected) / len(batch)
                with tf.GradientTape() as tape:
                    mse = loss_value(selected, data, target, True)
                    regularization = tf.add_n(network.losses) if network.losses else tf.constant(0.0)
                    objective = fraction * (mse + regularization)
                gradients = tape.gradient(objective, network.trainable_variables)
                if any(g is None for g in gradients):
                    raise RuntimeError("Disconnected neural gradient")
                accumulators = [a + g for a, g in zip(accumulators, gradients)]
                batch_loss += fraction * float(mse.numpy())
            clipped, _ = tf.clip_by_global_norm(accumulators, config["gradient_clipnorm"])
            optimizer.apply_gradients(zip(clipped, network.trainable_variables))
            losses.append(batch_loss)
            batch_counts.append(len(batch))
        deadline.check()
        loss = float(np.average(losses, weights=batch_counts))
        val_loss = None
        if validation is not None:
            values, sizes = [], []
            for start in range(0, len(validation), micro):
                deadline.check()
                indices = np.arange(start, min(start + micro, len(validation)))
                values.append(float(loss_value(indices, validation, validation_target, False).numpy()))
                sizes.append(len(indices))
            val_loss = float(np.average(values, weights=sizes))
        score = val_loss if validation is not None else loss
        if not np.isfinite(loss) or not np.isfinite(score):
            raise FloatingPointError("Nonfinite neural loss")
        record = {"epoch": epoch, "loss": loss, "val_loss": val_loss, "runtime_seconds": deadline.elapsed}
        history.append(record)
        log_epoch(directory / "epochs.jsonl" if directory else None, record)
        if score < best - config["min_delta"]:
            best, best_epoch, bad_epochs = score, epoch, 0
            if select_duration:
                best_weights = network.get_weights()
                if directory:
                    network.save_weights(directory / "best.weights.h5")
        else:
            bad_epochs += 1
        if select_duration and bad_epochs >= config["patience"]:
            break
    if select_duration:
        if best_weights is None:
            raise RuntimeError("No valid validation checkpoint")
        network.set_weights(best_weights)
    else:
        best_epoch = count_epochs
    return {"epochs": history, "best_epoch": best_epoch, "best_validation_loss": best if select_duration else None, "runtime_seconds": deadline.elapsed, "effective_batch_size": effective, "micro_batch_size": micro}
