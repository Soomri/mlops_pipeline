"""
Evaluación del modelo desplegado
==================================

Mide el desempeño real del modelo ya desplegado (`model_artifact.joblib`,
servido por `model_deploy.py`) contra `holdout_test.csv`: el 20% de datos
que `model_training.py` dejó aparte y que el modelo desplegado nunca vio
durante su entrenamiento.

No se reutilizan las métricas de `model_training_resumen.json` a propósito
aunque vengan del mismo holdout: aquí se pasa por el mismo camino que
seguiría una petición real al endpoint (`model_deploy.predecir_batch`),
para evaluar el artefacto tal como quedó empaquetado y detectar cualquier
diferencia entre "el modelo que se seleccionó" y "el modelo que
efectivamente quedó desplegado".

Genera `model_evaluation_report.json` — el contenido de la "pestaña de
métricas" que un dashboard mostraría sobre el modelo en producción — y un
gráfico de la matriz de confusión.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

from model_deploy import cargar_modelo, predecir_batch
from model_training import summarize_classification

DIR_SRC = Path(__file__).resolve().parent
RUTA_HOLDOUT_TEST = DIR_SRC / "holdout_test.csv"
RUTA_REPORTE_JSON = DIR_SRC / "model_evaluation_report.json"
RUTA_MATRIZ_PNG = DIR_SRC / "model_evaluation_matriz_confusion.png"


def evaluar_modelo_desplegado(modelo_artefacto, holdout):
    """Corre el holdout completo a través del mismo camino de inferencia
    del endpoint (`predecir_batch`) y calcula las métricas resultantes.

    Devuelve (metricas, matriz_confusion, n_descartadas). Filas descartadas
    por el pipeline de features (edad fuera de rango, duplicados) se
    excluyen del cálculo de métricas -- no hay una predicción real que
    comparar contra su verdadero valor -- pero se reportan aparte.
    """
    y_true = holdout["Pago_atiempo"]
    X = holdout.drop(columns=["Pago_atiempo"])

    predicciones, n_descartadas = predecir_batch(modelo_artefacto, X)
    pred_series = pd.Series(predicciones, index=X.index)

    mask_validas = pred_series.notna()
    y_true_validas = y_true[mask_validas]
    y_pred_validas = pred_series[mask_validas].astype(int)

    metricas = summarize_classification(y_true_validas, y_pred_validas, "modelo_desplegado")
    matriz = confusion_matrix(y_true_validas, y_pred_validas).tolist()
    return metricas, matriz, n_descartadas


def graficar_matriz_confusion(matriz, ruta_salida):
    matriz = np.array(matriz)
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(matriz, cmap="Blues")
    for i in range(matriz.shape[0]):
        for j in range(matriz.shape[1]):
            ax.text(j, i, str(matriz[i, j]), ha="center", va="center")
    ax.set_xlabel("Predicción")
    ax.set_ylabel("Real")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_title("Matriz de confusión — modelo desplegado")
    fig.tight_layout()
    fig.savefig(ruta_salida, dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    if not RUTA_HOLDOUT_TEST.exists():
        raise FileNotFoundError(
            f"No se encontró {RUTA_HOLDOUT_TEST}. Corre model_training.py primero "
            f"para generar el holdout de evaluación."
        )

    modelo = cargar_modelo()
    holdout = pd.read_csv(RUTA_HOLDOUT_TEST)

    metricas, matriz, n_descartadas = evaluar_modelo_desplegado(modelo, holdout)

    print("Métricas del modelo desplegado (holdout no visto en entrenamiento):")
    for k, v in metricas.items():
        if k != "modelo":
            print(f"  {k}: {v:.3f}")
    print(f"\nMatriz de confusión: {matriz}")
    if n_descartadas:
        print(f"Registros del holdout descartados por el pipeline de features: {n_descartadas}")

    graficar_matriz_confusion(matriz, RUTA_MATRIZ_PNG)
    print(f"Gráfico de matriz de confusión guardado en {RUTA_MATRIZ_PNG}")

    reporte = {
        "generado_en": datetime.now(timezone.utc).isoformat(),
        "n_registros_evaluados": int(len(holdout) - n_descartadas),
        "n_registros_descartados": n_descartadas,
        "metricas": metricas,
        "matriz_confusion": matriz,
    }
    RUTA_REPORTE_JSON.write_text(json.dumps(reporte, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Reporte de métricas guardado en {RUTA_REPORTE_JSON}")