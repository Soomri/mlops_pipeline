"""Pipeline custom en Vertex AI Pipelines (KFP v2) para el modelo de creditos.

Etapas:
  1. cargar_datos   -> lee el CSV de GCS y separa TRAIN+VALIDATE / TEST
  2. entrenar_evaluar (x3, en paralelo) -> LogisticRegression, RandomForest,
     HistGradientBoosting con class_weight='balanced'; reporta metricas de
     test, curva ROC, matriz de confusion y estabilidad (CV 5 folds y 5 semillas)
  3. comparar_modelos -> elige el mejor por ROC-AUC y guarda el resumen
"""
import subprocess
import sys

from kfp import compiler, dsl
from kfp.dsl import ClassificationMetrics, Dataset, Input, Metrics, Model, Output

IMAGEN = "python:3.11-slim"
PAQUETES = ["pandas==2.2.3", "scikit-learn==1.6.1", "numpy==2.2.6"]


@dsl.component(base_image=IMAGEN, packages_to_install=PAQUETES)
def cargar_datos(ruta_csv: str, train: Output[Dataset], test: Output[Dataset]):
    import pandas as pd

    df = pd.read_csv(ruta_csv)
    df[df["data_split"] != "TEST"].to_csv(train.path, index=False)
    df[df["data_split"] == "TEST"].to_csv(test.path, index=False)
    print("train:", (df["data_split"] != "TEST").sum(), "test:", (df["data_split"] == "TEST").sum())


@dsl.component(base_image=IMAGEN, packages_to_install=PAQUETES)
def entrenar_evaluar(
    algoritmo: str,
    train: Input[Dataset],
    test: Input[Dataset],
    modelo: Output[Model],
    predicciones: Output[Dataset],
    metricas: Output[Metrics],
    clasificacion: Output[ClassificationMetrics],
) -> float:
    import json
    import pickle
    import time

    import numpy as np
    import pandas as pd
    from sklearn.base import clone
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score,
                                 confusion_matrix, f1_score, log_loss, mean_squared_error,
                                 precision_score, recall_score, roc_auc_score, roc_curve)
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

    target = "Pago_atiempo"
    nominales = ["tipo_credito", "tipo_laboral", "mes_desembolso"]
    ordinales = ["tendencia_ingresos"]
    tr = pd.read_csv(train.path)
    te = pd.read_csv(test.path)
    numericas = [c for c in tr.columns if c not in nominales + ordinales + ["row_id", target, "data_split"]]
    for d in (tr, te):
        d[nominales] = d[nominales].astype(str)

    prep = ColumnTransformer([
        ("num", StandardScaler(), numericas),
        ("nom", OneHotEncoder(handle_unknown="ignore"), nominales),
        ("ord", OrdinalEncoder(categories=[["Decreciente", "Estable", "Creciente"]],
                               handle_unknown="use_encoded_value", unknown_value=-1), ordinales),
    ], sparse_threshold=0)
    estimadores = {
        "LogisticRegression": lambda s: LogisticRegression(max_iter=1000, class_weight="balanced"),
        "RandomForest": lambda s: RandomForestClassifier(n_estimators=200, max_depth=10,
                                                         class_weight="balanced", random_state=s),
        "HistGradientBoosting": lambda s: HistGradientBoostingClassifier(
            max_iter=200, learning_rate=0.05, max_depth=4, class_weight="balanced", random_state=s),
    }
    construir = lambda s: Pipeline([("prep", clone(prep)), ("modelo", estimadores[algoritmo](s))])

    X, y = tr.drop(columns=[target, "row_id", "data_split"]), tr[target]
    Xt, yt = te.drop(columns=[target, "row_id", "data_split"]), te[target]

    t0 = time.perf_counter()
    pipe = construir(42).fit(X, y)
    t_ent = time.perf_counter() - t0
    p = pipe.predict_proba(Xt)[:, 1]
    yhat = (p >= 0.5).astype(int)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_auc = cross_val_score(construir(42), X, y, cv=cv, scoring="roc_auc")
    semillas = [roc_auc_score(yt, construir(s).fit(X, y).predict_proba(Xt)[:, 1]) for s in range(5)]

    res = {
        "accuracy": accuracy_score(yt, yhat), "balanced_accuracy": balanced_accuracy_score(yt, yhat),
        "precision": precision_score(yt, yhat, zero_division=0), "recall": recall_score(yt, yhat),
        "f1": f1_score(yt, yhat), "roc_auc": roc_auc_score(yt, p), "pr_auc": average_precision_score(yt, p),
        "log_loss": log_loss(yt, p), "rmse": float(np.sqrt(mean_squared_error(yt, p))),
        "cv_roc_auc_media": float(cv_auc.mean()), "cv_roc_auc_std": float(cv_auc.std()),
        "semillas_roc_auc_std": float(np.std(semillas)), "tiempo_entrenamiento_seg": t_ent,
    }
    for k, v in res.items():
        metricas.log_metric(k, round(float(v), 4))
    metricas.metadata["algoritmo"] = algoritmo

    fpr, tpr, thr = roc_curve(yt, p)
    paso = max(1, len(fpr) // 200)
    thr = np.clip(thr, 0, 1)
    clasificacion.log_roc_curve(fpr[::paso].tolist(), tpr[::paso].tolist(), thr[::paso].tolist())
    clasificacion.log_confusion_matrix(["0", "1"], confusion_matrix(yt, yhat).tolist())

    with open(modelo.path, "wb") as f:
        pickle.dump(pipe, f)
    modelo.metadata["framework"] = "scikit-learn 1.6.1"
    pd.DataFrame({"row_id": te["row_id"], "y_true": yt, "p": p}).to_csv(predicciones.path, index=False)
    print(json.dumps(res, indent=2))
    return float(res["roc_auc"])


@dsl.component(base_image=IMAGEN, packages_to_install=PAQUETES)
def comparar_modelos(auc_lr: float, auc_rf: float, auc_hgb: float, resumen: Output[Metrics]) -> str:
    aucs = {"LogisticRegression": auc_lr, "RandomForest": auc_rf, "HistGradientBoosting": auc_hgb}
    mejor = max(aucs, key=aucs.get)
    for k, v in aucs.items():
        resumen.log_metric(f"roc_auc_{k}", round(v, 4))
    resumen.metadata["modelo_seleccionado"] = mejor
    print("Mejor modelo:", mejor, aucs)
    return mejor


@dsl.pipeline(name="pipeline-custom-creditos",
              description="Entrenamiento y comparacion de modelos de pago a tiempo (custom sklearn)")
def pipeline_creditos(ruta_csv: str = "/gcs/nifty-realm-511821-g3-vertex/data/creditos_vertex.csv"):
    datos = cargar_datos(ruta_csv=ruta_csv).set_display_name("Cargar datos")
    tareas = {}
    for alg in ("LogisticRegression", "RandomForest", "HistGradientBoosting"):
        tareas[alg] = (entrenar_evaluar(algoritmo=alg, train=datos.outputs["train"], test=datos.outputs["test"])
                       .set_display_name(f"Entrenar {alg}").set_cpu_limit("4").set_memory_limit("8G"))
    comparar_modelos(auc_lr=tareas["LogisticRegression"].outputs["Output"],
                     auc_rf=tareas["RandomForest"].outputs["Output"],
                     auc_hgb=tareas["HistGradientBoosting"].outputs["Output"]).set_display_name("Comparar modelos")


if __name__ == "__main__":
    compiler.Compiler().compile(pipeline_creditos, "pipeline_creditos.json")
    if "--run" in sys.argv:
        from google.cloud import aiplatform
        from google.oauth2.credentials import Credentials

        token = subprocess.check_output("gcloud auth print-access-token", shell=True, text=True).strip()
        aiplatform.init(project="nifty-realm-511821-g3", location="us-central1", credentials=Credentials(token))
        job = aiplatform.PipelineJob(display_name="pipeline-custom-creditos",
                                     template_path="pipeline_creditos.json",
                                     pipeline_root="gs://nifty-realm-511821-g3-vertex/pipeline_root",
                                     enable_caching=False)
        job.submit()
        print("PIPELINE_JOB", job.resource_name)
