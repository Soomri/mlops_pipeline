"""
Despliegue del modelo — endpoint de predicciones por batch
============================================================

Toma el artefacto serializado por `model_training.py`
(`model_artifact.joblib`: pipeline de features + el modelo ganador) y lo
expone como un endpoint HTTP (`Flask`) para predicciones por batch.

Manejo de filas descartadas por el pipeline de features
---------------------------------------------------------
`pipeline_ml` (dentro del artefacto) incluye pasos que descartan filas
(`Outliers` filtra `edad_cliente` fuera de [18, 100]; `EliminarDuplicados`
quita registros idénticos). Si se llamara `artefacto.predict(X)`
directamente sobre un batch con alguna de esas filas, el array de
predicciones resultante tendría MENOS elementos que filas de entrada, sin
ningún aviso — quien consuma el endpoint no tendría forma de saber a cuál
registro le falta predicción. `predecir_batch()` evita esto: aplica el
paso de features y el modelo por separado, y devuelve un resultado
alineado al índice original, con `None` explícito (no una predicción
inventada) para las filas que el pipeline descartó, más el conteo de
cuántas fueron.
"""

import json
from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, jsonify, request

DIR_SRC = Path(__file__).resolve().parent
RUTA_MODELO_ARTEFACTO = DIR_SRC / "model_artifact.joblib"


def cargar_modelo(ruta=RUTA_MODELO_ARTEFACTO):
    if not Path(ruta).exists():
        raise FileNotFoundError(
            f"No se encontró el artefacto del modelo en {ruta}. "
            f"Corre model_training.py primero para generarlo."
        )
    return joblib.load(ruta)


def predecir_batch(modelo_artefacto, registros):
    """Aplica el pipeline de features y el modelo por separado (en vez de
    `modelo_artefacto.predict(X)` directo) para poder alinear las
    predicciones al índice original, incluso si `Outliers`/
    `EliminarDuplicados` descartan filas.

    `registros`: lista de dicts o DataFrame con las columnas crudas del
    dataset (las mismas que `BD_creditos.csv`, sin la columna target).

    Devuelve (predicciones, n_descartadas):
      - predicciones: lista de 0/1/None, una por registro de entrada, en
        el mismo orden. None donde el pipeline descartó la fila.
      - n_descartadas: cuántas filas del batch no llegaron a predecirse.
    """
    X = pd.DataFrame(registros) if not isinstance(registros, pd.DataFrame) else registros.copy()
    X = X.reset_index(drop=True)

    features = modelo_artefacto.named_steps["features"]
    clasificador = modelo_artefacto.named_steps["modelo"]

    X_proc = features.transform(X)
    pred_series = pd.Series(clasificador.predict(X_proc), index=X_proc.index)
    pred_alineadas = pred_series.reindex(X.index)

    n_descartadas = int(pred_alineadas.isna().sum())
    predicciones = [None if pd.isna(v) else int(v) for v in pred_alineadas]
    return predicciones, n_descartadas


app = Flask(__name__)
_modelo_cache = {}


def _get_modelo():
    if "modelo" not in _modelo_cache:
        _modelo_cache["modelo"] = cargar_modelo()
    return _modelo_cache["modelo"]


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/predict", methods=["POST"])
def predict():
    payload = request.get_json(force=True, silent=True)
    registros = payload.get("registros") if isinstance(payload, dict) else payload

    if not registros:
        return jsonify({"error": "Se espera {'registros': [...]} con al menos un registro"}), 400

    try:
        modelo = _get_modelo()
        predicciones, n_descartadas = predecir_batch(modelo, registros)
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 503
    except Exception as e:
        return jsonify({"error": f"No se pudo generar la predicción: {e}"}), 400

    respuesta = {"predicciones": predicciones}
    if n_descartadas:
        respuesta["advertencia"] = (
            f"{n_descartadas} de {len(registros)} registros no recibieron predicción "
            f"(filtrados por reglas de calidad del pipeline: edad fuera de rango o "
            f"duplicados); aparecen como null en 'predicciones', en la misma posición "
            f"del registro enviado."
        )
    return jsonify(respuesta)


if __name__ == "__main__":
    # Aquí sirve directamente con el servidor de desarrollo de Flask, para
    # correrlo local o dentro del contenedor (ver Dockerfile en la raíz del
    # repo). En un despliegue real se pondría detrás de un WSGI server
    # (gunicorn/uwsgi) y un proxy, pero eso es infraestructura fuera del
    # alcance de este ejercicio.
    app.run(host="0.0.0.0", port=5000)