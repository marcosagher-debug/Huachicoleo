"""Pipeline reproducible para la detección de robo de hidrocarburos."""
from __future__ import annotations

import argparse
import itertools
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, List, Sequence

import numpy as np

from src.model_evaluation import (
    AdvancedModelResult,
    CalibrationResult,
    EvaluationResult,
    ModelConfig,
    PipelineArtifacts,
    SweepResult,
    ThresholdScenario,
    TrainingConfig,
    build_autoencoder,
    calibrate_threshold,
    evaluate_predictions,
    experiment_thresholds,
    reconstruction_errors,
    run_advanced_detectors,
    train_autoencoder,
)
from src.preprocessing import (
    WindowConfig,
    build_windowed_datasets,
    fit_scaler_on_train_normals,
    load_dataset,
    temporal_split,
)


def run_pipeline(
    data_path: Path,
    window: int,
    step: int,
    train_ratio: float,
    val_ratio: float,
    model_config: ModelConfig | None = None,
    training_config: TrainingConfig | None = None,
    output_path: Path | None = None,
    analyze_thresholds: bool = True,
    evaluate_advanced: bool = False,
) -> PipelineArtifacts:
    """Execute the full training/calibration/evaluation pipeline."""

    df = load_dataset(data_path)
    df_train, df_val, _ = temporal_split(df, train_ratio, val_ratio)
    train_len = len(df_train)
    val_len = len(df_val)

    scaler = fit_scaler_on_train_normals(df, train_len)

    window_config = WindowConfig(size=window, step=step)
    (
        X_train_all,
        y_train_all,
        _ts_train,
        X_val,
        y_val,
        _ts_val,
        X_test,
        y_test,
        ts_test,
    ) = build_windowed_datasets(df, scaler, window_config, train_len, val_len)

    X_train = X_train_all[y_train_all == 0]
    if X_train.size == 0:
        raise ValueError("No hay ventanas normales en el conjunto de entrenamiento.")

    model_config = model_config or ModelConfig()
    training_config = training_config or TrainingConfig()

    model = build_autoencoder(
        timesteps=X_train.shape[1],
        n_features=X_train.shape[2],
        model_config=model_config,
        learning_rate=training_config.learning_rate,
    )
    train_autoencoder(model, X_train, training_config)

    val_errors = reconstruction_errors(model, X_val)
    val_normal_errors = val_errors[y_val == 0]
    calibration = calibrate_threshold(val_normal_errors, val_errors, y_val)

    test_errors = reconstruction_errors(model, X_test)
    y_pred = (test_errors > calibration.threshold).astype(int)
    evaluation = evaluate_predictions(y_test, y_pred, test_errors)

    threshold_experiments: List[ThresholdScenario] = []
    if analyze_thresholds:
        threshold_experiments = experiment_thresholds(
            val_normal_errors, val_errors, y_val
        )

    advanced_results: List[AdvancedModelResult] = []
    if evaluate_advanced:
        advanced_results = run_advanced_detectors(X_train, X_test, y_test)

    artifacts = PipelineArtifacts(
        calibration=calibration,
        evaluation=evaluation,
        threshold_experiments=threshold_experiments,
        advanced_results=advanced_results,
    )

    if output_path is not None:
        save_artifacts(output_path, artifacts, ts_test, test_errors, y_test, y_pred)

    return artifacts


def save_artifacts(
    output_path: Path,
    artifacts: PipelineArtifacts,
    timestamps: Sequence,
    test_errors: np.ndarray,
    y_test: np.ndarray,
    y_pred: np.ndarray,
) -> None:
    """Persist metrics and per-window errors to JSON."""

    payload = {
        "calibration": asdict(artifacts.calibration),
        "evaluation": asdict(artifacts.evaluation),
        "threshold_experiments": [asdict(scenario) for scenario in artifacts.threshold_experiments],
        "advanced_results": [asdict(result) for result in artifacts.advanced_results],
        "test": {
            "timestamps": [str(ts) for ts in timestamps],
            "errors": test_errors.tolist(),
            "labels": y_test.tolist(),
            "predictions": y_pred.tolist(),
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=_json_default)
    )


def _json_default(value: Any):
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Tipo no serializable: {type(value)!r}")


def run_sweep(
    data_path: Path,
    windows: Sequence[int],
    steps: Sequence[int],
    encoder_grid: Sequence[Sequence[int]],
    decoder_grid: Sequence[Sequence[int]],
    dropouts: Sequence[float],
    learning_rates: Sequence[float],
    epochs: Sequence[int],
    batch_sizes: Sequence[int],
    patience_values: Sequence[int],
    validation_splits: Sequence[float],
    train_ratio: float,
    val_ratio: float,
) -> List[SweepResult]:
    """Grid-search over configuration options returning ordered results."""

    results: List[SweepResult] = []
    for window, step, encoder, decoder, dropout in itertools.product(
        windows, steps, encoder_grid, decoder_grid, dropouts
    ):
        for lr, epochs_, batch, patience, val_split in itertools.product(
            learning_rates, epochs, batch_sizes, patience_values, validation_splits
        ):
            model_cfg = ModelConfig(
                encoder_units=tuple(int(u) for u in encoder),
                decoder_units=tuple(int(u) for u in decoder),
                dropout=float(dropout),
            )
            training_cfg = TrainingConfig(
                learning_rate=float(lr),
                epochs=int(epochs_),
                batch_size=int(batch),
                patience=int(patience),
                validation_split=float(val_split),
            )
            artifacts = run_pipeline(
                data_path=data_path,
                window=int(window),
                step=int(step),
                train_ratio=train_ratio,
                val_ratio=val_ratio,
                model_config=model_cfg,
                training_config=training_cfg,
                analyze_thresholds=False,
                evaluate_advanced=False,
            )
            results.append(
                SweepResult(
                    window=int(window),
                    step=int(step),
                    model=model_cfg,
                    training=training_cfg,
                    calibration=artifacts.calibration,
                    evaluation=artifacts.evaluation,
                )
            )
    results.sort(
        key=lambda item: item.evaluation.classification_report["Anomalía"]["f1-score"],
        reverse=True,
    )
    return results


def generate_report(
    report_path: Path, artifacts: PipelineArtifacts, args: argparse.Namespace
) -> None:
    """Create a Markdown summary with the most relevant metrics."""

    report_lines = [
        "# Resultados del modelo de detección de robo de hidrocarburos",
        "",
        "## Configuración",
        f"- Ventana: {args.window}",
        f"- Paso: {args.step}",
        f"- Ratio entrenamiento: {args.train_ratio}",
        f"- Ratio validación: {args.val_ratio}",
        "",
        "## Calibración del umbral",
        f"- Percentil óptimo: {artifacts.calibration.best_percentile:.2f}",
        f"- Umbral seleccionado: {artifacts.calibration.threshold:.6f}",
        f"- Precisión (val): {artifacts.calibration.precision:.3f}",
        f"- Recall (val): {artifacts.calibration.recall:.3f}",
        f"- F1 (val): {artifacts.calibration.f1:.3f}",
        "",
        "## Métricas en prueba",
        f"- F1: {artifacts.evaluation.classification_report['Anomalía']['f1-score']:.3f}",
        f"- Precisión: {artifacts.evaluation.classification_report['Anomalía']['precision']:.3f}",
        f"- Recall: {artifacts.evaluation.classification_report['Anomalía']['recall']:.3f}",
        f"- ROC-AUC: {artifacts.evaluation.roc_auc:.3f}",
        f"- Average Precision: {artifacts.evaluation.average_precision:.3f}",
        "",
    ]

    if artifacts.threshold_experiments:
        report_lines.append("## Comparativa de estrategias de umbral")
        report_lines.append("")
        report_lines.append("| Estrategia | Umbral | Precisión | Recall | F1 |")
        report_lines.append("|------------|--------|-----------|--------|----|")
        for scenario in artifacts.threshold_experiments:
            report_lines.append(
                f"| {scenario.strategy} | {scenario.threshold:.6f} | {scenario.precision:.3f} | {scenario.recall:.3f} | {scenario.f1:.3f} |"
            )
        report_lines.append("")

    if artifacts.advanced_results:
        report_lines.append("## Modelos avanzados de detección")
        report_lines.append("")
        report_lines.append("| Modelo | Precisión | Recall | F1 | ROC-AUC | AP |")
        report_lines.append("|--------|-----------|--------|----|---------|----|")
        for result in artifacts.advanced_results:
            report_lines.append(
                f"| {result.name} | {result.precision:.3f} | {result.recall:.3f} | {result.f1:.3f} | {result.roc_auc:.3f} | {result.average_precision:.3f} |"
            )
        report_lines.append("")

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(report_lines))


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Entrena y evalúa el autoencoder LSTM para detección de anomalías",
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path("data/rebombeo_huachicoleo.csv"),
        help="Ruta al archivo CSV con los datos",
    )
    parser.add_argument("--window", type=int, default=30, help="Tamaño de la ventana en minutos")
    parser.add_argument("--step", type=int, default=5, help="Paso entre ventanas en minutos")
    parser.add_argument(
        "--train-ratio", type=float, default=0.6, help="Proporción de muestras para entrenamiento"
    )
    parser.add_argument(
        "--val-ratio", type=float, default=0.2, help="Proporción de muestras para validación"
    )
    parser.add_argument("--encoder-units", nargs="+", help="Capas del codificador, e.g. 128 64")
    parser.add_argument("--decoder-units", nargs="+", help="Capas del decodificador, e.g. 64 128")
    parser.add_argument("--dropout", type=float, help="Dropout aplicado en cada capa LSTM")
    parser.add_argument("--learning-rate", type=float, help="Tasa de aprendizaje del optimizador")
    parser.add_argument("--epochs", type=int, help="Número máximo de épocas")
    parser.add_argument("--batch-size", type=int, help="Tamaño de lote")
    parser.add_argument(
        "--validation-split",
        type=float,
        help="Fracción de entrenamiento usada como validación interna",
    )
    parser.add_argument(
        "--patience",
        type=int,
        help="Número de épocas sin mejora antes de aplicar parada temprana",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Ruta donde se guardarán métricas y errores en formato JSON",
    )
    parser.add_argument(
        "--report",
        type=Path,
        help="Ruta para guardar un resumen en Markdown con los resultados",
    )
    parser.add_argument(
        "--advanced-models",
        action="store_true",
        help="Activa la evaluación de modelos avanzados (IsolationForest, LOF)",
    )
    parser.add_argument(
        "--no-threshold-comparison",
        action="store_true",
        help="Desactiva la comparación con estrategias de umbral adicionales",
    )
    parser.add_argument(
        "--sweep",
        action="store_true",
        help="Ejecuta una búsqueda exhaustiva de hiperparámetros en lugar de un solo entrenamiento",
    )
    parser.add_argument(
        "--sweep-output",
        type=Path,
        help="Ruta donde se guardará el ranking de configuraciones exploradas en JSON",
    )
    parser.add_argument(
        "--windows",
        type=int,
        nargs="+",
        default=[20, 30, 60],
        help="Ventanas a explorar cuando se activa --sweep",
    )
    parser.add_argument(
        "--steps",
        type=int,
        nargs="+",
        default=[1, 5, 10],
        help="Pasos entre ventanas a explorar cuando se activa --sweep",
    )
    parser.add_argument(
        "--encoder-grid",
        nargs="+",
        default=["128,64", "256,128"],
        help="Opciones de capas para el codificador en sweep (usa comas)",
    )
    parser.add_argument(
        "--decoder-grid",
        nargs="+",
        default=["64,128", "128,256"],
        help="Opciones de capas para el decodificador en sweep (usa comas)",
    )
    parser.add_argument(
        "--dropouts",
        type=float,
        nargs="+",
        default=[0.1, 0.2],
        help="Valores de dropout a explorar en sweep",
    )
    parser.add_argument(
        "--learning-rates",
        type=float,
        nargs="+",
        default=[1e-3, 5e-4],
        help="Tasas de aprendizaje a explorar en sweep",
    )
    parser.add_argument(
        "--epochs-grid",
        type=int,
        nargs="+",
        default=[50, 100],
        help="Épocas a explorar en sweep",
    )
    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        default=[32, 64],
        help="Tamaños de batch a explorar en sweep",
    )
    parser.add_argument(
        "--patience-grid",
        type=int,
        nargs="+",
        default=[5, 10],
        help="Valores de paciencia para parada temprana en sweep",
    )
    parser.add_argument(
        "--validation-splits",
        type=float,
        nargs="+",
        default=[0.1, 0.2],
        help="Fracciones de validación internas en sweep",
    )
    return parser


def parse_units(values: List[str] | None) -> List[int] | None:
    if values is None:
        return None
    units: List[int] = []
    for value in values:
        if "," in value:
            units.extend(int(v.strip()) for v in value.split(",") if v.strip())
        else:
            units.append(int(value))
    return units


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    if args.sweep:
        encoder_grid = [tuple(int(x) for x in option.split(",")) for option in args.encoder_grid]
        decoder_grid = [tuple(int(x) for x in option.split(",")) for option in args.decoder_grid]
        results = run_sweep(
            data_path=args.data_path,
            windows=args.windows,
            steps=args.steps,
            encoder_grid=encoder_grid,
            decoder_grid=decoder_grid,
            dropouts=args.dropouts,
            learning_rates=args.learning_rates,
            epochs=args.epochs_grid,
            batch_sizes=args.batch_sizes,
            patience_values=args.patience_grid,
            validation_splits=args.validation_splits,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
        )
        if args.sweep_output:
            args.sweep_output.parent.mkdir(parents=True, exist_ok=True)
            args.sweep_output.write_text(
                json.dumps(
                    [result.to_dict() for result in results],
                    indent=2,
                    ensure_ascii=False,
                    default=_json_default,
                )
            )
        return

    encoder_units = parse_units(args.encoder_units)
    decoder_units = parse_units(args.decoder_units)
    default_model = ModelConfig()
    model_config = ModelConfig(
        encoder_units=tuple(encoder_units) if encoder_units else default_model.encoder_units,
        decoder_units=tuple(decoder_units) if decoder_units else default_model.decoder_units,
        dropout=args.dropout if args.dropout is not None else default_model.dropout,
    )

    default_training = TrainingConfig()
    training_config = TrainingConfig(
        learning_rate=args.learning_rate if args.learning_rate is not None else default_training.learning_rate,
        epochs=args.epochs if args.epochs is not None else default_training.epochs,
        batch_size=args.batch_size if args.batch_size is not None else default_training.batch_size,
        validation_split=args.validation_split if args.validation_split is not None else default_training.validation_split,
        patience=args.patience if args.patience is not None else default_training.patience,
    )

    artifacts = run_pipeline(
        data_path=args.data_path,
        window=args.window,
        step=args.step,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        model_config=model_config,
        training_config=training_config,
        output_path=args.output,
        analyze_thresholds=not args.no_threshold_comparison,
        evaluate_advanced=args.advanced_models,
    )

    if args.report:
        generate_report(args.report, artifacts, args)


if __name__ == "__main__":
    main()
