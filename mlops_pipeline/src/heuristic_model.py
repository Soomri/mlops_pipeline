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

El umbral no se fija a mano: `fit` prueba cada valor observado (excluyendo
el sentinela, ver más abajo) en el set de entrenamiento y se queda con el
que maximiza balanced accuracy (no accuracy simple: el target está
desbalanceado ~95%/5%, y accuracy simple colapsa trivialmente a predecir
siempre la clase mayoritaria).

`puntaje_datacredito == 0` es el sentinela de "sin score" (ver
`ReglasNegocio` en `ft_engineering.py`, que lo conserva intacto a
propósito). No es un score real ni el más bajo posible: tratarlo como tal
metería esas filas siempre bajo el umbral, aunque en la práctica la
mayoría de esos clientes sí paga a tiempo. Por eso se excluyen del
cálculo del umbral y se predicen aparte, con la clase mayoritaria
observada en entrenamiento para ese subgrupo.

Se espera recibir los datos ya procesados por `ft_engineering.py`
(`pipeline_basemodel`), donde `puntaje_datacredito` ya tiene sus valores
fuera de rango corregidos e imputados (sin NaN).
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
from sklearn.utils.validation import check_is_fitted


class HeuristicModel(BaseEstimator, ClassifierMixin):
    """Clasificador de referencia (baseline) basado en un único umbral de
    negocio sobre `puntaje_datacredito`. No aprende relaciones entre
    variables ni pesos: solo calibra el punto de corte que mejor separa
    las clases sobre los datos de entrenamiento.
    """

    def __init__(self, columna="puntaje_datacredito", sentinela=0):
        self.columna = columna
        self.sentinela = sentinela
        # 'umbral_' y 'clase_sentinela_' NO se inicializan aquí: deben
        # existir solo después de fit(), para que check_is_fitted (y
        # cualquier herramienta de sklearn aguas abajo) detecte
        # correctamente un modelo sin entrenar.

    def fit(self, X, y):
        col = pd.Series(X[self.columna]).reset_index(drop=True)
        y = pd.Series(y).reset_index(drop=True)

        if col.isna().any():
            raise ValueError(
                f"'{self.columna}' tiene valores nulos; imputa antes de entrenar "
                f"(este modelo espera la salida de ft_engineering.pipeline_basemodel)"
            )

        mask_sentinela = col == self.sentinela
        col_validas = col[~mask_sentinela]
        y_validas = y[~mask_sentinela]

        candidatos = np.unique(col_validas)
        if len(candidatos) == 0:
            raise ValueError(f"'{self.columna}' no tiene valores no nulos para calibrar el umbral")

        # Se calibra maximizando balanced accuracy, no accuracy simple: el
        # target está desbalanceado (~95% paga a tiempo), así que accuracy
        # simple se maximiza trivialmente prediciendo siempre la clase
        # mayoritaria (umbral = mínimo posible), lo cual no representa una
        # regla de negocio real ni sirve como referencia útil.
        mejor_umbral, mejor_score = candidatos[0], -1.0
        for umbral in candidatos:
            pred = (col_validas >= umbral).astype(int)
            score = balanced_accuracy_score(y_validas, pred)
            if score > mejor_score:
                mejor_score, mejor_umbral = score, umbral

        self.umbral_ = mejor_umbral
        # Para las filas con sentinela (sin score real), se predice la
        # clase mayoritaria observada en entrenamiento para ese subgrupo,
        # en vez de dejarlas competir como si el 0 fuera un score bajo real.
        if mask_sentinela.any():
            self.clase_sentinela_ = int(y[mask_sentinela].mode().iloc[0])
        else:
            self.clase_sentinela_ = int(y_validas.mode().iloc[0])
        return self

    def predict(self, X):
        check_is_fitted(self, ["umbral_", "clase_sentinela_"])
        col = pd.Series(X[self.columna])

        if col.isna().any():
            raise ValueError(
                f"'{self.columna}' tiene valores nulos: predict() no puede evaluarlos "
                f"(NaN >= umbral siempre es False y devolvería clase 0 en silencio)"
            )

        mask_sentinela = col == self.sentinela
        pred = (col >= self.umbral_).astype(int)
        pred[mask_sentinela] = self.clase_sentinela_
        return pred.to_numpy()


def evaluar_heuristico(y_true, y_pred):
    """Métricas de clasificación estándar para comparar la heurística
    contra los modelos de ML entrenados en `model_training.py`.

    Incluye `balanced_accuracy` explícitamente: con un target tan
    desbalanceado (~95%/5%), `accuracy`/`f1` por sí solos pueden hacer ver
    fuerte a un modelo que en realidad discrimina poco mejor que el azar
    (un clasificador trivial que siempre predice la clase mayoritaria ya
    saca accuracy/f1 muy altos aquí). `balanced_accuracy` es la métrica
    real de referencia para juzgar esta heurística.
    """
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "matriz_confusion": confusion_matrix(y_true, y_pred).tolist(),
    }


if __name__ == "__main__":
    import json
    from pathlib import Path

    from sklearn.model_selection import train_test_split

    from ft_engineering import pipeline_basemodel

    # Rutas ancladas a este archivo, no al directorio de trabajo actual:
    # así el script corre igual desde src/ o desde la raíz del repo.
    DIR_SRC = Path(__file__).resolve().parent
    config = json.loads((DIR_SRC / "config.json").read_text(encoding="utf-8"))

    df = pd.read_csv(DIR_SRC / config["raw_data_path"], **config["csv_read_options"]["raw"])
    y = df["Pago_atiempo"]
    X = df.drop(columns=["Pago_atiempo"])

    # Split ANTES de ajustar el pipeline: ajustar sobre todo el dataset
    # (incluyendo el futuro test set) filtraría información del test hacia
    # el entrenamiento vía la winsorización y las medianas/modas de
    # imputación, que quedarían calibradas viendo datos que se supone no
    # conoce. Con el split actual (80/20, random_state=42) esto no cambiaba
    # las métricas del heurístico, pero sí afecta a columnas como
    # saldo_total (el límite de winsorización varía ~6.4% según qué datos
    # ve el fit) y hay que evitarlo aquí antes de que se repita en
    # model_training.py.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipeline_basemodel.fit(X_train, y_train)
    X_train_proc = pipeline_basemodel.transform(X_train)
    y_train_proc = y_train.loc[X_train_proc.index]
    X_test_proc = pipeline_basemodel.transform(X_test)
    y_test_proc = y_test.loc[X_test_proc.index]

    modelo = HeuristicModel().fit(X_train_proc, y_train_proc)
    y_pred = modelo.predict(X_test_proc)

    baseline_trivial = np.full_like(y_test_proc, fill_value=y_train_proc.mode().iloc[0])

    print(f"Umbral calibrado (puntaje_datacredito): {modelo.umbral_}")
    print("\nHeuristicModel:")
    for metrica, valor in evaluar_heuristico(y_test_proc, y_pred).items():
        print(f"  {metrica}: {valor}")

    print("\nBaseline trivial (siempre predice la clase mayoritaria), para comparar:")
    for metrica, valor in evaluar_heuristico(y_test_proc, baseline_trivial).items():
        print(f"  {metrica}: {valor}")
