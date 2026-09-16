"""
Entrenamiento de modelos — versión base
=========================================

Primera iteración de `model_training.py`: entrena un modelo de aprendizaje
automático real y lo compara contra las dos líneas base ya establecidas
(`heuristic_model.HeuristicModel` y el baseline trivial que siempre predice
la clase mayoritaria), usando las mismas métricas (`accuracy`,
`balanced_accuracy`, `precision`, `recall`, `f1`) para que la comparación
sea directa.

Se entrena un único candidato (regresión logística, con
`class_weight='balanced'` por el desbalance del target ~95%/5%) sobre las
features de `ft_engineering.pipeline_ml`. Ampliar a más modelos (árboles,
boosting, comparación de hiperparámetros, gráficos) queda para una
siguiente iteración de este archivo.

Sigue el mismo patrón de `heuristic_model.py` para evitar data leakage:
el split va ANTES de ajustar `pipeline_ml` (winsorización, imputación,
escalado y encoding se calibran solo con datos de entrenamiento).
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)


def build_model(estimator, X_train, y_train):
    """Entrena un estimador de sklearn (interfaz fit/predict) sobre datos ya
    procesados y lo devuelve ajustado. Uniforma cómo se instancian/entrenan
    los distintos modelos candidatos al compararlos."""
    return estimator.fit(X_train, y_train)


def summarize_classification(y_true, y_pred, nombre_modelo):
    """Resume las métricas de clasificación de un modelo, con el mismo
    criterio usado para el heurístico en `heuristic_model.evaluar_heuristico`:
    incluye `balanced_accuracy` explícitamente porque el target está
    desbalanceado (~95%/5%) y `accuracy`/`f1` solos pueden hacer ver fuerte
    a un modelo que discrimina poco mejor que el azar.
    """
    return {
        "modelo": nombre_modelo,
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }


if __name__ == "__main__":
    import json
    from pathlib import Path

    from sklearn.model_selection import train_test_split

    from ft_engineering import pipeline_basemodel, pipeline_ml
    from heuristic_model import HeuristicModel

    DIR_SRC = Path(__file__).resolve().parent
    config = json.loads((DIR_SRC / "config.json").read_text(encoding="utf-8"))

    df = pd.read_csv(DIR_SRC / config["raw_data_path"], **config["csv_read_options"]["raw"])
    y = df["Pago_atiempo"]
    X = df.drop(columns=["Pago_atiempo"])

    # Split ANTES de ajustar cualquier pipeline: mismo motivo documentado en
    # heuristic_model.py (evitar que la winsorización/imputación/escalado
    # se calibren viendo datos del test set).
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Features para el modelo de ML: pipeline_ml (incluye escalado/encoding).
    pipeline_ml.fit(X_train, y_train)
    X_train_ml = pipeline_ml.transform(X_train)
    y_train_ml = y_train.loc[X_train_ml.index]
    X_test_ml = pipeline_ml.transform(X_test)
    y_test_ml = y_test.loc[X_test_ml.index]

    # class_weight='balanced': el target está desbalanceado (~95%/5%). Sin
    # esto, LogisticRegression converge casi al baseline trivial (siempre
    # predice la clase mayoritaria) igual que le pasó a HeuristicModel con
    # accuracy simple — balanced_accuracy pasa de ~0.51 a un valor que sí
    # refleja discriminación real entre clases.
    modelo_lr = build_model(
        LogisticRegression(max_iter=1000, class_weight="balanced"), X_train_ml, y_train_ml
    )
    pred_lr = modelo_lr.predict(X_test_ml)

    # Heurístico: usa pipeline_basemodel (sin escalar/codificar), no pipeline_ml.
    pipeline_basemodel.fit(X_train, y_train)
    X_train_base = pipeline_basemodel.transform(X_train)
    y_train_base = y_train.loc[X_train_base.index]
    X_test_base = pipeline_basemodel.transform(X_test)
    y_test_base = y_test.loc[X_test_base.index]

    heuristico = HeuristicModel().fit(X_train_base, y_train_base)
    pred_heuristico = heuristico.predict(X_test_base)

    baseline_trivial = np.full_like(y_test_ml, fill_value=y_train_ml.mode().iloc[0])

    resultados = [
        summarize_classification(y_test_ml, pred_lr, "LogisticRegression"),
        summarize_classification(y_test_base, pred_heuristico, "HeuristicModel"),
        summarize_classification(y_test_ml, baseline_trivial, "Baseline trivial (clase mayoritaria)"),
    ]

    tabla = pd.DataFrame(resultados).set_index("modelo")
    print("Tabla comparativa (test set):")
    print(tabla.round(3))