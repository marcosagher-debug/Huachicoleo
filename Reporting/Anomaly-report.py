import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle

def generate_anomaly_report(pdf_path: Path, ts_test: np.ndarray, test_errors: np.ndarray, y_test: np.ndarray, y_pred: np.ndarray, window_size: int, anomaly_threshold: float) -> None:
    """Genera un reporte en PDF para las anomalías detectadas."""
    
    anomalies = np.where(y_pred == 1)[0]  # Índices donde se detectó una anomalía
    if not anomalies.size:
        print("No se detectaron anomalías.")
        return

    # Crear el documento PDF
    doc = SimpleDocTemplate(pdf_path, pagesize=letter)
    elements = []

    # Título
    title = "Reporte de Anomalías Detectadas en el Sistema de Tuberías"
    elements.append(Paragraph(title, style={'alignment': 1, 'fontSize': 18}))
    elements.append(Spacer(1, 12))

    # Descripción general
    description = (
        "Se han detectado anomalías en el flujo de hidrocarburos durante la evaluación de los datos "
        "de la tubería. A continuación se detallan las condiciones y gráficas correspondientes "
        "a las anomalías identificadas."
    )
    elements.append(Paragraph(description, style={'alignment': 0, 'fontSize': 12}))
    elements.append(Spacer(1, 12))

    # Añadir tabla con los valores de las anomalías
    anomaly_data = []
    for idx in anomalies:
        timestamp = ts_test[idx]
        error = test_errors[idx]
        label = y_test[idx]
        prediction = y_pred[idx]
        anomaly_data.append([str(timestamp), error, label, prediction])

    anomaly_table = Table(
        [["Timestamp", "Error de Reconstrucción", "Etiqueta", "Predicción"]] + anomaly_data,
        colWidths=[2*inch, 1.5*inch, 1*inch, 1*inch]
    )

    # Estilo de la tabla
    anomaly_table.setStyle(TableStyle([
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('BACKGROUND', (0, 0), (-1, 0), colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.black),
    ]))
    elements.append(anomaly_table)
    elements.append(Spacer(1, 12))

    # Gráficas de las anomalías
    for idx in anomalies:
        # Crear gráfico para cada anomalía
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.plot(ts_test[max(0, idx - window_size): idx + window_size], test_errors[max(0, idx - window_size): idx + window_size])
        ax.axvline(x=idx, color='r', linestyle='--', label='Anomalía')
        ax.set_title(f"Anomalía Detectada en {ts_test[idx]}")
        ax.set_xlabel('Tiempo')
        ax.set_ylabel('Error de Reconstrucción')
        plt.legend()

        # Guardar la imagen del gráfico
        plot_path = f"anomaly_plot_{idx}.png"
        plt.savefig(plot_path)
        plt.close()

        # Agregar imagen al reporte
        elements.append(Paragraph(f"Gráfico de la Anomalía Detectada: {ts_test[idx]}", style={'alignment': 0, 'fontSize': 12}))
        elements.append(Spacer(1, 12))
        elements.append(Spacer(1, 12))

    # Crear el PDF
    doc.build(elements)
    print(f"Reporte generado: {pdf_path}")
