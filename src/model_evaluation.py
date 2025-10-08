"""Model creation, calibration and evaluation utilities."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable, List, Sequence, Tuple

import numpy as np
import tensorflow as tf
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.neighbors import LocalOutlierFactor


SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)


@dataclass
class ModelConfig:
    """Architecture of the LSTM autoencoder."""

    encoder_units: Tuple[int, ...] = (128, 64)
    decoder_units: Tuple[int, ...] = (64, 128)
    dropout: float = 0.1


@dataclass
class TrainingConfig:
    """Training hyperparameters."""

    learning_rate: float = 1e-3
    epochs: int = 100
    batch_size: int = 64
    validation_split: float = 0.1
    patience: int = 10


@dataclass
class CalibrationResult:
    threshold: float
    best_percentile: float
    precision: float
    recall: float
    f1: float


@dataclass
class ThresholdScenario:
    strategy: str
    threshold: float
    precision: float
    recall: float
    f1: float


@dataclass
class AdvancedModelResult:
    name: str
    precision: float
    recall: float
    f1: float
    roc_auc: float
    average_precision: float


@dataclass
class EvaluationResult:
    classification_report: dict
    confusion_matrix: Iterable[Iterable[int]]
    roc_auc: float
    average_precision: float


@dataclass
class PipelineArtifacts:
    calibration: CalibrationResult
    evaluation: EvaluationResult
    threshold_experiments: List[ThresholdScenario]
    advanced_results: List[AdvancedModelResult]


@dataclass
class SweepResult:
    window: int
    step: int
    model: ModelConfig
    training: TrainingConfig
    calibration: CalibrationResult
    evaluation: EvaluationResult

    def to_dict(self) -> dict:
        return {
            "window": self.window,
            "step": self.step,
            "model": asdict(self.model),
            "training": asdict(self.training),
            "calibration": asdict(self.calibration),
            "evaluation": asdict(self.evaluation),
        }


def build_autoencoder(
    timesteps: int, n_features: int, model_config: ModelConfig, learning_rate: float
) -> tf.keras.Model:
    """Create the LSTM autoencoder according to ``model_config``."""

    inputs = tf.keras.layers.Input(shape=(timesteps, n_features))
    x = inputs
    for index, units in enumerate(model_config.encoder_units):
        return_sequences = index < len(model_config.encoder_units) - 1
        x = tf.keras.layers.LSTM(
            units,
            activation="tanh",
            return_sequences=return_sequences,
            dropout=model_config.dropout,
        )(x)
    x = tf.keras.layers.RepeatVector(timesteps)(x)
    for units in model_config.decoder_units:
        x = tf.keras.layers.LSTM(
            units,
            activation="tanh",
            return_sequences=True,
            dropout=model_config.dropout,
        )(x)
    outputs = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(n_features))(x)

    model = tf.keras.Model(inputs, outputs, name="lstm_autoencoder")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss="mse"
    )
    return model


def train_autoencoder(
    model: tf.keras.Model, X_train: np.ndarray, training_config: TrainingConfig
) -> tf.keras.callbacks.History:
    """Train the autoencoder with early stopping."""

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=training_config.patience, restore_best_weights=True
    )
    history = model.fit(
        X_train,
        X_train,
        epochs=training_config.epochs,
        batch_size=training_config.batch_size,
        validation_split=training_config.validation_split,
        callbacks=[early_stopping],
        verbose=1,
    )
    return history


def reconstruction_errors(model: tf.keras.Model, X: np.ndarray) -> np.ndarray:
    """Compute the mean squared reconstruction error for each window."""

    reconstructions = model.predict(X, verbose=0)
    return np.mean(np.square(X - reconstructions), axis=(1, 2))


def calibrate_threshold(
    errors_normals: np.ndarray, errors_val: np.ndarray, labels_val: np.ndarray
) -> CalibrationResult:
    """Select the threshold that maximises F1 score on validation windows."""

    candidate_percentiles = np.linspace(80, 99, 40)
    best_threshold = 0.0
    best_metrics: Tuple[float, float, float, float] | None = None
    best_f1 = -np.inf
    for percentile in candidate_percentiles:
        threshold = float(np.percentile(errors_normals, percentile))
        preds = (errors_val > threshold).astype(int)
        precision = precision_score(labels_val, preds, zero_division=0)
        recall = recall_score(labels_val, preds, zero_division=0)
        f1 = f1_score(labels_val, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_metrics = (percentile, precision, recall, f1)
            best_threshold = threshold
    if best_metrics is None:
        raise ValueError("No se pudo calibrar el umbral: revise los datos de validación.")
    percentile, precision, recall, f1 = best_metrics
    return CalibrationResult(
        threshold=float(best_threshold),
        best_percentile=float(percentile),
        precision=float(precision),
        recall=float(recall),
        f1=float(f1),
    )


def experiment_thresholds(
    errors_normals: np.ndarray, errors_val: np.ndarray, labels_val: np.ndarray
) -> List[ThresholdScenario]:
    """Evaluate alternative fixed thresholds for comparison."""

    scenarios: List[ThresholdScenario] = []
    for percentile in [85, 90, 95, 97.5, 99]:
        threshold = float(np.percentile(errors_normals, percentile))
        preds = (errors_val > threshold).astype(int)
        scenarios.append(
            ThresholdScenario(
                strategy=f"percentil_{percentile}",
                threshold=threshold,
                precision=float(precision_score(labels_val, preds, zero_division=0)),
                recall=float(recall_score(labels_val, preds, zero_division=0)),
                f1=float(f1_score(labels_val, preds, zero_division=0)),
            )
        )

    q1, q3 = np.percentile(errors_normals, [25, 75])
    iqr_threshold = float(q3 + 1.5 * (q3 - q1))
    iqr_preds = (errors_val > iqr_threshold).astype(int)
    scenarios.append(
        ThresholdScenario(
            strategy="iqr",
            threshold=iqr_threshold,
            precision=float(precision_score(labels_val, iqr_preds, zero_division=0)),
            recall=float(recall_score(labels_val, iqr_preds, zero_division=0)),
            f1=float(f1_score(labels_val, iqr_preds, zero_division=0)),
        )
    )

    median = float(np.median(errors_normals))
    mad = float(np.median(np.abs(errors_normals - median)))
    mad_threshold = float(median + 3 * mad)
    mad_preds = (errors_val > mad_threshold).astype(int)
    scenarios.append(
        ThresholdScenario(
            strategy="mad",
            threshold=mad_threshold,
            precision=float(precision_score(labels_val, mad_preds, zero_division=0)),
            recall=float(recall_score(labels_val, mad_preds, zero_division=0)),
            f1=float(f1_score(labels_val, mad_preds, zero_division=0)),
        )
    )
    return scenarios


def evaluate_predictions(
    y_true: np.ndarray, predictions: np.ndarray, scores: np.ndarray
) -> EvaluationResult:
    """Create a metrics bundle for anomaly detection results."""

    report = classification_report(
        y_true,
        predictions,
        target_names=["Normal", "Anomalía"],
        zero_division=0,
        output_dict=True,
    )
    cm = confusion_matrix(y_true, predictions)
    roc_auc = roc_auc_score(y_true, scores)
    avg_precision = average_precision_score(y_true, scores)
    return EvaluationResult(
        classification_report=report,
        confusion_matrix=cm.tolist(),
        roc_auc=float(roc_auc),
        average_precision=float(avg_precision),
    )


def evaluate_scores(
    name: str, y_true: np.ndarray, predictions: np.ndarray, scores: np.ndarray
) -> AdvancedModelResult:
    """Package the metrics for a detector other than the autoencoder."""

    return AdvancedModelResult(
        name=name,
        precision=float(precision_score(y_true, predictions, zero_division=0)),
        recall=float(recall_score(y_true, predictions, zero_division=0)),
        f1=float(f1_score(y_true, predictions, zero_division=0)),
        roc_auc=float(roc_auc_score(y_true, scores)),
        average_precision=float(average_precision_score(y_true, scores)),
    )


def run_advanced_detectors(
    X_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray
) -> List[AdvancedModelResult]:
    """Train additional anomaly detection algorithms on flattened windows."""

    results: List[AdvancedModelResult] = []
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_test_flat = X_test.reshape(X_test.shape[0], -1)

    iso = IsolationForest(random_state=SEED, contamination="auto")
    iso.fit(X_train_flat)
    iso_scores = -iso.decision_function(X_test_flat)
    iso_preds = (iso.predict(X_test_flat) == -1).astype(int)
    results.append(evaluate_scores("IsolationForest", y_test, iso_preds, iso_scores))

    if len(X_train_flat) > 1:
        n_neighbors = min(35, len(X_train_flat) - 1)
    else:
        n_neighbors = 1
    lof = LocalOutlierFactor(n_neighbors=n_neighbors, novelty=True, contamination=0.05)
    lof.fit(X_train_flat)
    lof_scores = -lof.decision_function(X_test_flat)
    lof_preds = (lof.predict(X_test_flat) == -1).astype(int)
    results.append(evaluate_scores("LocalOutlierFactor", y_test, lof_preds, lof_scores))

    return results


__all__ = [
    "AdvancedModelResult",
    "CalibrationResult",
    "EvaluationResult",
    "ModelConfig",
    "PipelineArtifacts",
    "SweepResult",
    "ThresholdScenario",
    "TrainingConfig",
    "build_autoencoder",
    "calibrate_threshold",
    "evaluate_predictions",
    "evaluate_scores",
    "experiment_thresholds",
    "reconstruction_errors",
    "run_advanced_detectors",
    "train_autoencoder",
]
