"""Prepara el dataset para Vertex AI y recalcula la linea base local en Python.

Usa el mismo split de model_training.py (test_size=0.2, random_state=42,
stratify) para que AutoML, Vertex Pipelines y Python se evaluen sobre las
mismas filas de prueba.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score, log_loss,
                             precision_score, recall_score, roc_auc_score, average_precision_score,
                             mean_squared_error)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.tree import DecisionTreeClassifier

SRC = Path(sys.argv[1])          # carpeta mlops_pipeline/src del repo
RAW = Path(sys.argv[2])          # CSV crudo (version del commit)
OUT = Path(sys.argv[3])          # carpeta de salida
sys.path.insert(0, str(SRC))
from ft_engineering import pipeline_basemodel, pipeline_ml  # noqa: E402

config = json.loads((SRC / "config.json").read_text(encoding="utf-8"))
df = pd.read_csv(RAW, **config["csv_read_options"]["raw"])
y = df["Pago_atiempo"]
X = df.drop(columns=["Pago_atiempo"])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# ---- Dataset para Vertex: features limpias (sin escalar ni codificar) + columna de split
pipeline_basemodel.fit(X_train, y_train)
tr = pipeline_basemodel.transform(X_train)
te = pipeline_basemodel.transform(X_test)
tr["Pago_atiempo"] = y_train.loc[tr.index].astype(int)
te["Pago_atiempo"] = y_test.loc[te.index].astype(int)
idx_tr, idx_val = train_test_split(tr.index, test_size=0.1, random_state=42, stratify=tr["Pago_atiempo"])
tr["data_split"] = "TRAIN"
tr.loc[idx_val, "data_split"] = "VALIDATE"
te["data_split"] = "TEST"
full = pd.concat([tr, te])
full.insert(0, "row_id", full.index.astype(int))
for c in ("tipo_credito", "tipo_laboral", "tendencia_ingresos"):
    full[c] = full[c].astype(str)
full.to_csv(OUT / "creditos_vertex.csv", index=False)
print("filas:", full["data_split"].value_counts().to_dict(), "clase 1:", round(full["Pago_atiempo"].mean(), 4))

# ---- Linea base Python (mismo codigo y semillas que model_training.py)
pipeline_ml.fit(X_train, y_train)
Xtr = pipeline_ml.transform(X_train); ytr = y_train.loc[Xtr.index]
Xte = pipeline_ml.transform(X_test); yte = y_test.loc[Xte.index]
cands = {
    "LogisticRegression": LogisticRegression(max_iter=1000, class_weight="balanced"),
    "DecisionTree": DecisionTreeClassifier(max_depth=6, class_weight="balanced", random_state=42),
    "RandomForest": RandomForestClassifier(n_estimators=200, max_depth=10, class_weight="balanced", random_state=42),
}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
res = {}
preds = pd.DataFrame({"row_id": Xte.index.astype(int), "y_true": yte.values})
for n, est in cands.items():
    t0 = time.perf_counter(); m = clone(est).fit(Xtr, ytr); t = time.perf_counter() - t0
    p = m.predict_proba(Xte)[:, 1]; yhat = (p >= 0.5).astype(int)
    cvs = cross_val_score(clone(est), Xtr, ytr, cv=cv, scoring="roc_auc")
    res[n] = {
        "accuracy": accuracy_score(yte, yhat), "balanced_accuracy": balanced_accuracy_score(yte, yhat),
        "precision": precision_score(yte, yhat, zero_division=0), "recall": recall_score(yte, yhat),
        "f1": f1_score(yte, yhat), "roc_auc": roc_auc_score(yte, p), "pr_auc": average_precision_score(yte, p),
        "log_loss": log_loss(yte, p), "rmse": float(np.sqrt(mean_squared_error(yte, p))),
        "cv_roc_auc_media": cvs.mean(), "cv_roc_auc_std": cvs.std(), "tiempo_entrenamiento_seg": t,
    }
    preds[f"p_{n}"] = p
    print(n, {k: round(v, 4) for k, v in res[n].items()})
(OUT / "python_metricas.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
preds.to_csv(OUT / "python_predicciones_test.csv", index=False)
