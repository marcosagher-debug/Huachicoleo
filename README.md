# Detección de robo de hidrocarburos

Este proyecto ofrece un flujo reproducible para detectar anomalías en operaciones de rebombeo utilizando un autoencoder LSTM y modelos complementarios de `scikit-learn`. El repositorio se organiza como un paquete Python que facilita la experimentación, la calibración del umbral y la generación de reportes ejecutivos.

## Estructura del proyecto

```
Huachicoleo/
├─ .vscode/                 # Configuración recomendada para VS Code
├─ data/                    # Datos crudos (mantener privados si son sensibles)
├─ notebooks/               # Cuadernos exploratorios
├─ src/                     # Paquete con la lógica del pipeline
│  ├─ __init__.py
│  ├─ preprocessing.py      # Carga, escalado y generación de ventanas temporales
│  └─ model_evaluation.py   # Entrenamiento, calibración y métricas
├─ tests/                   # Pruebas unitarias (pytest)
├─ run_pipeline.py          # CLI reproducible para entrenar y evaluar modelos
├─ requirements.txt         # Dependencias para ejecutar el pipeline
├─ requirements-dev.txt     # Herramientas opcionales para desarrollo
├─ .gitignore
└─ README.md
```

Los datos sensibles no deberían versionarse; conserva únicamente ejemplos representativos o utiliza scripts para descargarlos bajo demanda.

## Preparación del entorno

1. **Crear un entorno virtual**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # En Windows: .venv\Scripts\activate
   ```

2. **Instalar dependencias de ejecución**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

3. *(Opcional)* **Instalar herramientas de desarrollo**
   ```bash
   pip install -r requirements-dev.txt
   ```

4. **Verificar los datos**
   Coloca el CSV con mediciones históricas en `data/`. Por defecto se espera `data/rebombeo_huachicoleo.csv` con las columnas:
   - `timestamp`: marca temporal en UTC.
   - `flow`, `pressure`, `pump_rpm`, `tank_level`, `power`: variables operativas.
   - `label`: 0 para operación normal, 1 para evento confirmado de robo.

## Opciones para ejecutar el análisis

### Cuaderno exploratorio

En `notebooks/Red_Huachicoleo.ipynb` encontrarás el flujo detallado para cargar datos, crear ventanas, entrenar el autoencoder y visualizar errores de reconstrucción. Es ideal para análisis ad-hoc y comunicación con perfiles de negocio.

### Script reproducible (`run_pipeline.py`)

El script encapsula el pipeline completo y permite automatizar experimentos:

```bash
python run_pipeline.py \
  --window 30 \
  --step 5 \
  --epochs 80 \
  --output artefactos/resultados.json \
  --report artefactos/resumen.md \
  --advanced-models
```

El comando anterior entrena el autoencoder únicamente con ventanas normales, calibra el umbral maximizando `F1` en validación, evalúa en el conjunto de prueba y genera:
- `artefactos/resultados.json`: métricas, umbral elegido, errores por ventana y predicciones.
- `artefactos/resumen.md`: resumen profesional en Markdown listo para compartir con stakeholders.

#### Parámetros clave

- `--window`, `--step`: tamaño y desplazamiento de las ventanas temporales.
- `--train-ratio`, `--val-ratio`: proporciones del split temporal (el resto se usa para prueba).
- `--encoder-units`, `--decoder-units`, `--dropout`: arquitectura del autoencoder.
- `--learning-rate`, `--epochs`, `--batch-size`, `--validation-split`, `--patience`: hiperparámetros de entrenamiento.
- `--advanced-models`: activa `IsolationForest` y `LocalOutlierFactor` para comparar detectores.
- `--no-threshold-comparison`: desactiva las comparativas con percentiles/IQR/MAD si solo necesitas el umbral calibrado.

#### Búsqueda de hiperparámetros (`--sweep`)

Activa `--sweep` para explorar automáticamente combinaciones de ventana, paso, arquitectura y entrenamiento. Por ejemplo:

```bash
python run_pipeline.py --sweep \
  --sweep-output artefactos/sweep.json \
  --windows 24 36 48 \
  --steps 1 5 10 \
  --encoder-grid 256,128 128,64 \
  --decoder-grid 64,128 128,256
```

El archivo generado ordena las configuraciones por `F1` sobre anomalías en el conjunto de prueba, facilitando la selección de parámetros para producción.

## Flujo del pipeline

1. **Carga y ordenamiento temporal**: se preserva la secuencia cronológica para evitar fuga de información.
2. **Split reproducible**: `temporal_split` divide la serie en entrenamiento/validación/prueba sin mezclar periodos.
3. **Escalado seguro**: `fit_scaler_on_train_normals` ajusta `StandardScaler` únicamente con observaciones normales del tramo de entrenamiento.
4. **Ventanas deslizantes**: `build_windowed_datasets` genera secuencias etiquetadas como anómalas si alguna muestra dentro de la ventana tiene `label = 1`.
5. **Entrenamiento del autoencoder**: `build_autoencoder` y `train_autoencoder` crean y entrenan un modelo LSTM configurable con parada temprana.
6. **Calibración del umbral**: `calibrate_threshold` explora percentiles (80–99) para maximizar `F1` en validación; `experiment_thresholds` documenta estrategias alternativas (percentiles fijos, IQR, MAD).
7. **Evaluación integral**: `evaluate_predictions` entrega `precision`, `recall`, `F1`, `ROC-AUC`, `Average Precision` y matriz de confusión; `run_advanced_detectors` compara contra Isolation Forest y LOF.
8. **Reportes auditables**: `save_artifacts` guarda métricas y errores por ventana en JSON, mientras `generate_report` produce un resumen ejecutivo en Markdown.

## Presentación de resultados

- **Reporte ejecutivo**: comparte el Markdown generado junto con visualizaciones del notebook (reconstrucciones, distribuciones de error, matrices de confusión).
- **Dashboard operativo**: integra los errores de reconstrucción y el umbral calibrado en una herramienta de BI para monitoreo en tiempo real.
- **Ficha técnica del modelo**: documenta datos utilizados, arquitectura, proceso de calibración y criterios de recalibración periódica.

## Pruebas

Las pruebas unitarias se ubican en `tests/` y validan utilidades de preprocesamiento (generación de ventanas, escalado). Ejecuta:

```bash
pytest
```

> **Nota:** las ejecuciones completas del pipeline requieren instalar dependencias de `tensorflow`, que pueden no estar disponibles en entornos sin acceso a binarios precompilados. En dichos casos se recomienda usar un entorno local con GPU/CPU compatibles.

## Próximos pasos sugeridos

- Versionar modelos entrenados y escaladores (`model.save`, `joblib.dump`) para despliegue en producción.
- Añadir validación `walk-forward` o pruebas retrospectivas con distintos periodos para evaluar robustez temporal.
- Extender el pipeline con variables contextuales (eventos de mantenimiento, clima) y técnicas de detección adicionales (VAE, Transformers) según los requerimientos operativos.

Con esta estructura podrás experimentar de forma controlada, generar reportes profesionales y preparar el modelo para integrarse en procesos de monitoreo industrial.
