"""
Validación de métricas para el pipeline de aprobación
=======================================================

Compara `model_evaluation_report.json` (generado por `model_evaluation.py`)
contra los mínimos de `umbrales_metricas` en `config.json`. Termina con
código de salida 1 si alguna métrica queda por debajo de su umbral, lo que
hace fallar la etapa de Jenkins y rechaza el cambio.
"""

import json
import sys
from pathlib import Path

DIR_SRC = Path(__file__).resolve().parent
RUTA_CONFIG = DIR_SRC / "config.json"
RUTA_REPORTE_JSON = DIR_SRC / "model_evaluation_report.json"


def metricas_por_debajo(metricas, umbrales):
    """Devuelve los nombres de las métricas que no alcanzan su umbral."""
    return [nombre for nombre, minimo in umbrales.items() if metricas[nombre] < minimo]


if __name__ == "__main__":
    umbrales = json.loads(RUTA_CONFIG.read_text(encoding="utf-8"))["umbrales_metricas"]
    metricas = json.loads(RUTA_REPORTE_JSON.read_text(encoding="utf-8"))["metricas"]

    for nombre, minimo in umbrales.items():
        print(f"  {nombre}: {metricas[nombre]:.3f} (mínimo {minimo:.3f})")

    rechazadas = metricas_por_debajo(metricas, umbrales)
    if rechazadas:
        print(f"Modelo rechazado. Métricas por debajo del umbral: {', '.join(rechazadas)}")
        sys.exit(1)

    print("Modelo aprobado: todas las métricas cumplen su umbral.")
