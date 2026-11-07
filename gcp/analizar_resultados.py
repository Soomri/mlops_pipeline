"""Reune predicciones de Python, Vertex Pipelines y AutoML sobre el mismo test,
calcula metricas, intervalos bootstrap y genera los graficos comparativos."""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score, f1_score,
                             log_loss, mean_squared_error, precision_score, recall_score, roc_auc_score,
                             roc_curve)

D = Path(sys.argv[1])   # carpeta fase2/data
G = Path(sys.argv[2])   # carpeta de graficos
G.mkdir(exist_ok=True)

py = pd.read_csv(D / "python_predicciones_test.csv")
base = py[["row_id", "y_true"]].copy()
modelos = {"Python · LogisticRegression": py.set_index("row_id")["p_LogisticRegression"],
           "Python · RandomForest": py.set_index("row_id")["p_RandomForest"]}
for alg in ("LogisticRegression", "RandomForest", "HistGradientBoosting"):
    f = D / f"vertex_pred_{alg}.csv"
    modelos[f"Vertex Pipelines · {alg}"] = pd.read_csv(f).set_index("row_id")["p"]
modelos["AutoML Tabular"] = pd.read_csv(D / "automl_pred.csv").set_index("row_id")["p"]

y = base.set_index("row_id")["y_true"]
P = pd.DataFrame({k: v.reindex(y.index) for k, v in modelos.items()})
assert not P.isna().any().any(), P.isna().sum()


def metricas(yt, p):
    yh = (p >= 0.5).astype(int)
    return {"accuracy": accuracy_score(yt, yh), "balanced_accuracy": balanced_accuracy_score(yt, yh),
            "precision": precision_score(yt, yh, zero_division=0), "recall": recall_score(yt, yh),
            "f1": f1_score(yt, yh), "roc_auc": roc_auc_score(yt, p), "pr_auc": average_precision_score(yt, p),
            "log_loss": log_loss(yt, np.clip(p, 1e-6, 1 - 1e-6)), "rmse": float(np.sqrt(mean_squared_error(yt, p)))}


tabla = pd.DataFrame({k: metricas(y.values, P[k].values) for k in P}).T
rng = np.random.default_rng(42)
boot = {k: [] for k in P}
bootf1 = {k: [] for k in P}
yv = y.values
for _ in range(1000):
    i = rng.integers(0, len(yv), len(yv))
    if yv[i].min() == yv[i].max():
        continue
    for k in P:
        p = P[k].values[i]
        boot[k].append(roc_auc_score(yv[i], p))
        bootf1[k].append(f1_score(yv[i], (p >= 0.5).astype(int)))
for k in P:
    b = np.array(boot[k])
    tabla.loc[k, "boot_auc_std"] = b.std()
    tabla.loc[k, "boot_auc_ic_inf"], tabla.loc[k, "boot_auc_ic_sup"] = np.percentile(b, [2.5, 97.5])
    tabla.loc[k, "boot_f1_std"] = np.std(bootf1[k])
tabla.to_csv(D / "tabla_comparativa.csv")
print(tabla.round(4).to_string())

# ---------- graficos
corto = {k: k.replace("Vertex Pipelines · ", "Vertex · ").replace("Python · ", "Python · ")
         .replace("LogisticRegression", "LogReg").replace("RandomForest", "RF").replace("HistGradientBoosting", "HGB")
         for k in P}
colores = {"Python": "#7A869A", "Vertex": "#2E6FBA", "AutoML": "#E07A1F"}
col = [colores[k.split(" ")[0] if not k.startswith("AutoML") else "AutoML"] for k in P]
plt.rcParams.update({"font.size": 10, "font.family": "DejaVu Sans"})

fig, axes = plt.subplots(1, 3, figsize=(11, 3.6))
for ax, m, tit in zip(axes, ["roc_auc", "balanced_accuracy", "f1"], ["ROC-AUC", "Balanced accuracy", "F1 (clase 1)"]):
    v = tabla[m].values
    ax.barh([corto[k] for k in P], v, color=col)
    for j, x in enumerate(v):
        ax.text(x + 0.005, j, f"{x:.3f}", va="center", fontsize=8)
    ax.set_title(tit); ax.set_xlim(0.4 if m != "f1" else 0.6, 1.0); ax.invert_yaxis()
    if ax is not axes[0]:
        ax.set_yticklabels([])
    ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout(); fig.savefig(G / "g1_metricas_comparativas.png", dpi=200); plt.close(fig)

fig, ax = plt.subplots(figsize=(6, 5))
paleta = {"LogReg": "#2E6FBA", "RF": "#2A9D8F", "HGB": "#8E5BB5"}
for k in P:
    fpr, tpr, _ = roc_curve(yv, P[k].values)
    c = "#E07A1F" if k.startswith("AutoML") else paleta[corto[k].split(" · ")[1]]
    ls = "--" if k.startswith("Python") else "-"
    ax.plot(fpr, tpr, ls, color=c, lw=1.6 if k.startswith("AutoML") else 1.1,
            label=f"{corto[k]} (AUC {tabla.loc[k, 'roc_auc']:.3f})", alpha=0.95)
ax.plot([0, 1], [0, 1], ":", color="#999")
ax.set_xlabel("Tasa de falsos positivos"); ax.set_ylabel("Tasa de verdaderos positivos")
ax.set_title("Curvas ROC sobre el mismo conjunto de prueba"); ax.legend(fontsize=7.5, loc="lower right")
fig.tight_layout(); fig.savefig(G / "g2_curvas_roc.png", dpi=200); plt.close(fig)

fig, ax = plt.subplots(figsize=(9, 3.8))
ax.boxplot([boot[k] for k in P], vert=False, labels=[corto[k] for k in P], showfliers=False,
           patch_artist=True, boxprops=dict(facecolor="#DCE6F2"))
ax.invert_yaxis(); ax.set_xlabel("ROC-AUC (1.000 remuestreos bootstrap del test)")
ax.set_title("Estabilidad del ROC-AUC frente a la muestra de prueba")
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout(); fig.savefig(G / "g3_bootstrap_auc.png", dpi=200); plt.close(fig)

cvj = D / "cv_resumen.json"
if cvj.exists():
    cvd = json.loads(cvj.read_text())
    fig, ax = plt.subplots(figsize=(7, 3.4))
    names = list(cvd); med = [cvd[n]["media"] for n in names]; sd = [cvd[n]["std"] for n in names]
    ax.errorbar(med, range(len(names)), xerr=sd, fmt="o", color="#2E6FBA", capsize=4)
    ax.set_ylim(len(names) - 0.5, -0.9)
    ax.set_yticks(range(len(names))); ax.set_yticklabels(names)
    for j, (m_, s_) in enumerate(zip(med, sd)):
        ax.text(m_, j - 0.25, f"{m_:.3f} ± {s_:.3f}", ha="center", fontsize=8)
    ax.set_xlabel("ROC-AUC en validación"); ax.set_title("Consistencia entre particiones (media ± desv. estándar)")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(G / "g4_consistencia_cv.png", dpi=200); plt.close(fig)
print("graficos en", G)
