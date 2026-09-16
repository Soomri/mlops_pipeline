"""
Entrenamiento y selección de modelos
=====================================

Entrena varios modelos de clasificación candidatos, los compara entre sí y
contra las dos líneas base ya establecidas (`heuristic_model.HeuristicModel`
y el baseline trivial que siempre predice la clase mayoritaria), y
selecciona el mejor para desplegar (`model_deploy.py` carga el artefacto
que este script produce).

Candidatos (todos con `class_weight='balanced'`: el target está
desbalanceado ~95%/5%, y sin esto los modelos colapsan casi al baseline
trivial, como se documentó en la primera iteración de este archivo):
  - LogisticRegression: rápida de entrenar y de aplicar en producción,
    interpretable (coeficientes), buena línea base de ML.
  - DecisionTreeClassifier: captura no linealidades simples, también
    interpretable, pero más inestable entre folds (mayor varianza).
  - RandomForestClassifier: promedia muchos árboles, más costosa de
    entrenar/servir que las dos anteriores, pero típicamente más estable.

Selección del mejor modelo (se busca balancear las tres dimensiones que
pide el enunciado, no solo la métrica de test):
  - Performance: `balanced_accuracy` en el test set (métrica principal,
    por el desbalance del target).
  - Consistency: desviación estándar de `balanced_accuracy` en validación
    cruzada (5 folds) sobre el set de entrenamiento — un modelo con buen
    desempeño promedio pero muy inestable entre folds es una elección
    riesgosa para producción.
  - Scalability: proxy simple = tiempo de entrenamiento. Con este tamaño
    de dataset (~10K filas) no es un factor decisivo, pero queda medido y
    documentado para cuando el volumen de datos crezca.

El modelo con mejor `balanced_accuracy` gana salvo que su `cv_std` sea
notablemente peor (>0.03 por encima del mínimo observado) que el de otro
candidato con métrica de test muy cercana (diferencia < 0.01): en ese caso
se prefiere el más estable. Con los candidatos y el dataset actuales esto
no cambia el ganador (se documenta el criterio para cuando sí importe).

El artefacto desplegado (`model_artifact.joblib`) usa el modelo ganador
entrenado SOLO con el 80% de entrenamiento, no reentrenado sobre el 100%
de los datos. Es una decisión deliberada: el 20% restante (`holdout_test.csv`)
queda como datos genuinamente no vistos por el modelo desplegado, para que
`model_evaluation.py` mida su desempeño real (no una re-estimación) y
`model_monitoring.py` tenga una población de referencia limpia contra la
cual comparar datos nuevos al calcular datadrift.

Sigue el mismo patrón de `heuristic_model.py` para evitar data leakage:
el split va ANTES de ajustar cualquier pipeline (winsorización,
imputación, escalado y encoding se calibran solo con datos de
entrenamiento).
"""

import json
import time
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

DIR_SRC = Path(__file__).resolve().parent
RUTA_MODELO_ARTEFACTO = DIR_SRC / "model_artifact.joblib"
RUTA_COMPARATIVO_PNG = DIR_SRC / "model_training_comparativo.png"
RUTA_RESUMEN_JSON = DIR_SRC / "model_training_resumen.json"
RUTA_HOLDOUT_TEST = DIR_SRC / "holdout_test.csv"

CANDIDATOS = {
    "LogisticRegression": LogisticRegression(max_iter=1000, class_weight="balanced"),
    "DecisionTree": DecisionTreeClassifier(max_depth=6, class_weight="balanced", random_state=42),
    "RandomForest": RandomForestClassifier(
        n_estimators=200, max_depth=10, class_weight="balanced", random_state=42
    ),
}


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


def graficar_comparativo(tabla_metricas, ruta_salida):
    """Gráfico de barras agrupadas con las métricas de cada modelo, para
    comparar visualmente performance entre candidatos y líneas base."""
    metricas = ["accuracy", "balanced_accuracy", "precision", "recall", "f1"]
    modelos = tabla_metricas.index.tolist()
    x = np.arange(len(metricas))
    ancho = 0.8 / len(modelos)

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, modelo in enumerate(modelos):
        valores = tabla_metricas.loc[modelo, metricas].to_numpy(dtype=float)
        ax.bar(x + i * ancho, valores, width=ancho, label=modelo)

    ax.set_xticks(x + ancho * (len(modelos) - 1) / 2)
    ax.set_xticklabels(metricas)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Valor de la métrica")
    ax.set_title("Comparación de modelos (test set)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(ruta_salida, dpi=120)
    plt.close(fig)


def seleccionar_mejor_modelo(resultados_cv, tabla_metricas):
    """Aplica el criterio documentado en el docstring del módulo:
    balanced_accuracy de test como métrica principal, con desempate hacia
    el modelo más consistente (menor cv_std) cuando dos modelos quedan muy
    cerca en test. Devuelve el nombre del modelo elegido."""
    candidatos_ml = [n for n in tabla_metricas.index if n in CANDIDATOS]
    mejor = max(candidatos_ml, key=lambda n: tabla_metricas.loc[n, "balanced_accuracy"])

    for nombre in candidatos_ml:
        if nombre == mejor:
            continue
        diff_test = tabla_metricas.loc[mejor, "balanced_accuracy"] - tabla_metricas.loc[nombre, "balanced_accuracy"]
        diff_consistencia = resultados_cv[nombre]["cv_std"] - resultados_cv[mejor]["cv_std"]
        if diff_test < 0.01 and diff_consistencia < -0.03:
            mejor = nombre  # empate técnico en test, pero claramente más consistente

    return mejor


if __name__ == "__main__":
    from ft_engineering import pipeline_basemodel, pipeline_ml
    from heuristic_model import HeuristicModel

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

    # Features para los modelos de ML: pipeline_ml (incluye escalado/encoding).
    pipeline_ml.fit(X_train, y_train)
    X_train_ml = pipeline_ml.transform(X_train)
    y_train_ml = y_train.loc[X_train_ml.index]
    X_test_ml = pipeline_ml.transform(X_test)
    y_test_ml = y_test.loc[X_test_ml.index]

    resultados = []
    resultados_cv = {}
    modelos_entrenados = {}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for nombre, estimador in CANDIDATOS.items():
        inicio = time.perf_counter()
        modelo = build_model(clone(estimador), X_train_ml, y_train_ml)
        tiempo_entrenamiento = time.perf_counter() - inicio
        modelos_entrenados[nombre] = modelo

        pred = modelo.predict(X_test_ml)
        resultados.append(summarize_classification(y_test_ml, pred, nombre))

        scores_cv = cross_val_score(
            clone(estimador), X_train_ml, y_train_ml, cv=cv, scoring="balanced_accuracy"
        )
        resultados_cv[nombre] = {
            "cv_balanced_accuracy_media": scores_cv.mean(),
            "cv_std": scores_cv.std(),
            "tiempo_entrenamiento_seg": tiempo_entrenamiento,
        }

    # Heurístico: usa pipeline_basemodel (sin escalar/codificar), no pipeline_ml.
    pipeline_basemodel.fit(X_train, y_train)
    X_train_base = pipeline_basemodel.transform(X_train)
    y_train_base = y_train.loc[X_train_base.index]
    X_test_base = pipeline_basemodel.transform(X_test)
    y_test_base = y_test.loc[X_test_base.index]

    heuristico = HeuristicModel().fit(X_train_base, y_train_base)
    pred_heuristico = heuristico.predict(X_test_base)
    resultados.append(summarize_classification(y_test_base, pred_heuristico, "HeuristicModel"))

    baseline_trivial = np.full_like(y_test_ml, fill_value=y_train_ml.mode().iloc[0])
    resultados.append(
        summarize_classification(y_test_ml, baseline_trivial, "Baseline trivial (clase mayoritaria)")
    )

    tabla = pd.DataFrame(resultados).set_index("modelo")
    print("Tabla comparativa (test set):")
    print(tabla.round(3))

    print("\nConsistencia (validación cruzada, 5 folds) y tiempo de entrenamiento:")
    for nombre, r in resultados_cv.items():
        print(
            f"  {nombre}: balanced_accuracy cv = {r['cv_balanced_accuracy_media']:.3f} "
            f"± {r['cv_std']:.3f} | tiempo = {r['tiempo_entrenamiento_seg']:.2f}s"
        )

    graficar_comparativo(tabla, RUTA_COMPARATIVO_PNG)
    print(f"\nGráfico comparativo guardado en {RUTA_COMPARATIVO_PNG}")

    nombre_mejor = seleccionar_mejor_modelo(resultados_cv, tabla)
    print(f"\nModelo seleccionado: {nombre_mejor}")

    # El artefacto que se despliega usa el pipeline de features y el modelo
    # YA entrenados solo con X_train/y_train (no se reentrena sobre el 100%
    # de los datos): X_test/y_test quedan como holdout genuinamente no
    # visto por el modelo desplegado, para que model_evaluation.py pueda
    # medir su desempeño real (no una re-estimación) y model_monitoring.py
    # tenga una población de referencia limpia para detectar datadrift.
    artefacto = Pipeline(steps=[("features", pipeline_ml), ("modelo", modelos_entrenados[nombre_mejor])])
    joblib.dump(artefacto, RUTA_MODELO_ARTEFACTO)
    print(f"Artefacto del modelo ganador guardado en {RUTA_MODELO_ARTEFACTO}")

    holdout = X_test.copy()
    holdout["Pago_atiempo"] = y_test
    holdout.to_csv(RUTA_HOLDOUT_TEST, index=False)
    print(f"Holdout de evaluación guardado en {RUTA_HOLDOUT_TEST} ({len(holdout)} filas)")

    resumen = {
        "modelo_seleccionado": nombre_mejor,
        "metricas_test": tabla.loc[nombre_mejor].to_dict(),
        "consistencia_cv": resultados_cv,
        "tabla_comparativa": tabla.reset_index().to_dict(orient="records"),
    }
    RUTA_RESUMEN_JSON.write_text(json.dumps(resumen, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Resumen guardado en {RUTA_RESUMEN_JSON}")