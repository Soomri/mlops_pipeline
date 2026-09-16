"""
Monitoreo del modelo desplegado — datadrift
==============================================

Registra los datos enviados al endpoint (`model_deploy.py`) junto con los
pronósticos entregados, y usa esa tabla, con una periodicidad definida,
para muestrear y comparar la distribución de las variables de entrada
contra la población de referencia (la que vio el modelo al entrenar) —
detectando cambios que puedan degradar su desempeño (datadrift).

Métrica de datadrift: PSI (Population Stability Index) por columna
numérica, el estándar en la industria de riesgo crediticio para esto
(la misma industria de este dataset):
    PSI < 0.10            -> sin cambio relevante
    0.10 <= PSI < 0.20     -> cambio moderado, a vigilar
    PSI >= 0.20            -> cambio significativo (alerta)

`job_monitoreo()` está separado de CUÁNDO se ejecuta: en producción un
scheduler (cron/Airflow/el orquestador de la plataforma de MLOps) lo
llamaría con la periodicidad definida (p. ej. semanal), pasándole los
datos acumulados en `monitoring_log.csv` desde la última corrida. Aquí se
deja como una función pura para que conectarla a un scheduler real sea
solo eso: decidir cuándo llamarla, no cómo calcula el drift.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from model_deploy import cargar_modelo, predecir_batch

DIR_SRC = Path(__file__).resolve().parent
RUTA_LOG_MONITOREO = DIR_SRC / "monitoring_log.csv"
RUTA_REPORTE_DRIFT = DIR_SRC / "monitoring_datadrift_report.json"

COLUMNAS_NUMERICAS_DRIFT = [
    "capital_prestado", "plazo_meses", "edad_cliente", "salario_cliente",
    "total_otros_prestamos", "cuota_pactada", "puntaje_datacredito",
    "cant_creditosvigentes", "huella_consulta", "saldo_total",
]

UMBRAL_PSI_ALERTA = 0.2


def registrar_llamadas(registros, predicciones, ruta_log=RUTA_LOG_MONITOREO):
    """Agrega a `ruta_log` los registros enviados al endpoint junto con sus
    predicciones y una marca de tiempo. En producción esta tabla la
    alimentaría directamente el logging del endpoint; aquí se llama
    explícitamente después de cada batch para simularlo."""
    registros_df = pd.DataFrame(registros) if not isinstance(registros, pd.DataFrame) else registros.copy()
    registros_df = registros_df.reset_index(drop=True)
    registros_df["prediccion"] = predicciones
    registros_df["timestamp"] = datetime.now(timezone.utc).isoformat()

    registros_df.to_csv(ruta_log, mode="a", header=not ruta_log.exists(), index=False)
    return registros_df


def calcular_psi(referencia, actual, bins=10):
    """PSI de una variable numérica entre dos poblaciones. Se calcula
    dividiendo la referencia en `bins` cuantiles (bordes fijos, calculados
    una sola vez sobre la referencia) y comparando qué proporción de cada
    población cae en cada bin."""
    referencia = pd.Series(referencia).dropna()
    actual = pd.Series(actual).dropna()
    if referencia.empty or actual.empty:
        return np.nan

    cortes = np.quantile(referencia, np.linspace(0, 1, bins + 1))
    cortes[0], cortes[-1] = -np.inf, np.inf
    cortes = np.unique(cortes)  # evita bins duplicados si hay muchos valores repetidos

    dist_ref = pd.cut(referencia, bins=cortes).value_counts(normalize=True, sort=False)
    dist_actual = pd.cut(actual, bins=cortes).value_counts(normalize=True, sort=False)
    dist_actual = dist_actual.reindex(dist_ref.index, fill_value=0)

    # Suaviza bins con 0 observaciones: evita log(0) y división por 0 sin
    # distorsionar bins con proporciones normales.
    eps = 1e-4
    p_ref = dist_ref.to_numpy() + eps
    p_actual = dist_actual.to_numpy() + eps

    return float(np.sum((p_actual - p_ref) * np.log(p_actual / p_ref)))


def calcular_datadrift(referencia_df, actual_df, columnas=COLUMNAS_NUMERICAS_DRIFT, umbral=UMBRAL_PSI_ALERTA):
    """PSI de cada columna en `columnas`, marcando cuáles superan `umbral`."""
    resultado = {}
    for col in columnas:
        if col not in referencia_df.columns or col not in actual_df.columns:
            continue
        psi = calcular_psi(referencia_df[col], actual_df[col])
        resultado[col] = {"psi": psi, "alerta": bool(psi > umbral)}
    return resultado


def job_monitoreo(referencia_df, actual_df, ruta_reporte=RUTA_REPORTE_DRIFT):
    """Compara `actual_df` (una muestra reciente de `monitoring_log.csv`)
    contra `referencia_df` (población de entrenamiento) y guarda el
    reporte de datadrift resultante."""
    drift = calcular_datadrift(referencia_df, actual_df)
    columnas_en_alerta = [c for c, r in drift.items() if r["alerta"]]

    reporte = {
        "generado_en": datetime.now(timezone.utc).isoformat(),
        "n_referencia": len(referencia_df),
        "n_actual": len(actual_df),
        "datadrift_por_columna": drift,
        "columnas_en_alerta": columnas_en_alerta,
        "drift_detectado": len(columnas_en_alerta) > 0,
    }
    Path(ruta_reporte).write_text(json.dumps(reporte, indent=2, ensure_ascii=False), encoding="utf-8")
    return reporte


if __name__ == "__main__":
    from sklearn.model_selection import train_test_split

    from ft_engineering import CorreccionTipos

    config = json.loads((DIR_SRC / "config.json").read_text(encoding="utf-8"))
    df = pd.read_csv(DIR_SRC / config["raw_data_path"], **config["csv_read_options"]["raw"])
    y = df["Pago_atiempo"]
    X = df.drop(columns=["Pago_atiempo"])

    # Misma partición (random_state/test_size/stratify) que model_training.py:
    # reconstruye X_train como población de referencia sin persistir un
    # archivo aparte para eso.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    modelo = cargar_modelo()

    # calcular_psi necesita columnas numéricas de verdad: el CSV crudo trae
    # decimales con coma (str). predecir_batch() sí hace esta conversión
    # internamente (vía CorreccionTipos, dentro del pipeline), pero el
    # cálculo de drift es independiente de la predicción y necesita su
    # propia copia ya tipada.
    corrector_tipos = CorreccionTipos()
    X_train_num = corrector_tipos.transform(X_train)

    # Escenario 1: batch "reciente" sin drift real (muestra del propio
    # holdout de test -> misma población que entrenamiento).
    muestra_sin_drift = X_test.sample(n=min(300, len(X_test)), random_state=1)
    pred_sin_drift, _ = predecir_batch(modelo, muestra_sin_drift)
    registrar_llamadas(muestra_sin_drift, pred_sin_drift)

    reporte_sin_drift = job_monitoreo(X_train_num, corrector_tipos.transform(muestra_sin_drift))
    print("Escenario sin drift (muestra del holdout, misma población de entrenamiento):")
    print(f"  drift_detectado: {reporte_sin_drift['drift_detectado']}")
    print(f"  columnas en alerta: {reporte_sin_drift['columnas_en_alerta']}")

    # Escenario 2: batch con drift simulado (salario_cliente desplazado
    # x1.6, como si cambiara el segmento de clientes que llega al modelo),
    # para verificar que el job SÍ lo detecta.
    muestra_con_drift = muestra_sin_drift.copy()
    muestra_con_drift["salario_cliente"] = muestra_con_drift["salario_cliente"] * 1.6
    pred_con_drift, _ = predecir_batch(modelo, muestra_con_drift)
    registrar_llamadas(muestra_con_drift, pred_con_drift)

    reporte_con_drift = job_monitoreo(
        X_train_num,
        corrector_tipos.transform(muestra_con_drift),
        ruta_reporte=DIR_SRC / "monitoring_datadrift_report_simulado.json",
    )
    print("\nEscenario con drift simulado (salario_cliente x1.6):")
    print(f"  drift_detectado: {reporte_con_drift['drift_detectado']}")
    print(f"  columnas en alerta: {reporte_con_drift['columnas_en_alerta']}")
    psi_salario = reporte_con_drift["datadrift_por_columna"]["salario_cliente"]["psi"]
    print(f"  PSI salario_cliente: {psi_salario:.3f}")

    print(f"\nLog de monitoreo acumulado en {RUTA_LOG_MONITOREO}")
    print(f"Reporte de datadrift (sin drift) guardado en {RUTA_REPORTE_DRIFT}")