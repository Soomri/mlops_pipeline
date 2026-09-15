"""
Modelo heurístico — línea base sin aprendizaje automático
==========================================================

Antes de entrenar modelos de ML (`model_training.py`), se necesita un punto
de comparación: ¿cuánto valor agrega realmente el aprendizaje automático
frente a una regla de negocio simple que cualquier analista de crédito
aplicaría a ojo?

Este módulo implementa esa línea base: clasifica `Pago_atiempo` usando
únicamente `puntaje_datacredito` (el score de la central de riesgo), que en
el EDA (`comprension_eda.ipynb`, sección de correlaciones) mostró ser la
variable individual más asociada al target (r punto-biserial = 0.069,
Mann-Whitney p < 0.001): a mayor puntaje, menor riesgo de incumplimiento.

La regla es un único umbral sobre `puntaje_datacredito`:
    puntaje_datacredito >= umbral  ->  Pago_atiempo = 1 (paga a tiempo)
    puntaje_datacredito <  umbral  ->  Pago_atiempo = 0 (no paga a tiempo)

El umbral no se fija a mano: `fit` prueba cada valor observado en el set de
entrenamiento y se queda con el que maximiza balanced accuracy (no accuracy
simple: el target está desbalanceado ~95%/5%, y accuracy simple colapsa
trivialmente a predecir siempre la clase mayoritaria), para tener la mejor
versión posible de esta regla simple como referencia justa.

Se espera recibir los datos ya procesados por `ft_engineering.py`
(`pipeline_basemodel`), donde `puntaje_datacredito` ya tiene sus valores
fuera de rango corregidos e imputados.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


class HeuristicModel(BaseEstimator, ClassifierMixin):
    """Clasificador de referencia (baseline) basado en un único umbral de
    negocio sobre `puntaje_datacredito`. No aprende relaciones entre
    variables ni pesos: solo calibra el punto de corte que mejor separa
    las clases sobre los datos de entrenamiento.
    """

    def __init__(self, columna="puntaje_datacredito"):
        self.columna = columna
        self.umbral_ = None

    def fit(self, X, y):
        col = pd.Series(X[self.columna]).reset_index(drop=True)
        y = pd.Series(y).reset_index(drop=True)

        candidatos = np.unique(col.dropna())
        if len(candidatos) == 0:
            raise ValueError(f"'{self.columna}' no tiene valores no nulos para calibrar el umbral")

        # Se calibra maximizando balanced accuracy, no accuracy simple: el
        # target está desbalanceado (~95% paga a tiempo), así que accuracy
        # simple se maximiza trivialmente prediciendo siempre la clase
        # mayoritaria (umbral = mínimo posible), lo cual no representa una
        # regla de negocio real ni sirve como referencia útil.
        mejor_umbral, mejor_score = candidatos[0], -1.0
        for umbral in candidatos:
            pred = (col >= umbral).astype(int)
            score = balanced_accuracy_score(y, pred)
            if score > mejor_score:
                mejor_score, mejor_umbral = score, umbral

        self.umbral_ = mejor_umbral
        return self

    def predict(self, X):
        if self.umbral_ is None:
            raise RuntimeError("El modelo no ha sido entrenado: llama a fit(X, y) primero")
        col = pd.Series(X[self.columna])
        return (col >= self.umbral_).astype(int).to_numpy()


def evaluar_heuristico(y_true, y_pred):
    """Métricas de clasificación estándar para comparar la heurística
    contra los modelos de ML entrenados en `model_training.py`."""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "matriz_confusion": confusion_matrix(y_true, y_pred).tolist(),
    }


if __name__ == "__main__":
    from sklearn.model_selection import train_test_split

    from ft_engineering import pipeline_basemodel

    df = pd.read_csv("../BD_creditos.csv", sep=";", decimal=",", encoding="utf-8-sig")
    y = df["Pago_atiempo"]
    X = df.drop(columns=["Pago_atiempo"])

    X_procesado = pipeline_basemodel.fit_transform(X, y)
    y_procesado = y.loc[X_procesado.index]

    X_train, X_test, y_train, y_test = train_test_split(
        X_procesado, y_procesado, test_size=0.2, random_state=42, stratify=y_procesado
    )

    modelo = HeuristicModel().fit(X_train, y_train)
    y_pred = modelo.predict(X_test)

    print(f"Umbral calibrado (puntaje_datacredito): {modelo.umbral_}")
    for metrica, valor in evaluar_heuristico(y_test, y_pred).items():
        print(f"{metrica}: {valor}")