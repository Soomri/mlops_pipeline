# Fase 2: entrenamiento en GCP (Vertex AI)

Scripts usados para entrenar el modelo de pago a tiempo en Vertex AI y compararlo
con el modelo de `mlops_pipeline/src/model_training.py`.

- Proyecto: `nifty-realm-511821-g3`
- Región: `us-central1`
- Bucket: `gs://nifty-realm-511821-g3-vertex`

| Script | Qué hace |
|---|---|
| `preparar_datos.py` | Limpia los datos con `pipeline_basemodel`, crea `creditos_vertex.csv` con la columna `data_split` (TRAIN, VALIDATE y TEST, este último igual al test de `model_training.py`) y recalcula las métricas del modelo de Python |
| `lanzar_automl.py` | Crea el dataset tabular en Vertex AI y lanza AutoML Tabular (objetivo `maximize-au-roc`, 1 hora-nodo) |
| `pipeline_creditos.py` | Pipeline de Kubeflow (KFP v2) para Vertex Pipelines: carga los datos, entrena LogisticRegression, RandomForest y HistGradientBoosting en paralelo y elige el mejor por ROC-AUC |
| `analizar_resultados.py` | Junta las predicciones de los tres enfoques sobre el mismo test, calcula las métricas y el bootstrap, y genera los gráficos comparativos |

## Requisitos

```bash
pip install google-cloud-aiplatform kfp pandas==2.2.3 scikit-learn==1.6.1 matplotlib
gcloud auth login
```

La cuenta de servicio por defecto de Compute necesita `roles/storage.objectAdmin` sobre el bucket
para que Vertex Pipelines pueda leer los datos.

## Pasos

Usa el `BD_creditos.csv` original, separado por `;` y con coma decimal. Los comandos se ejecutan desde la raíz del repositorio y escriben sus salidas en `gcp/salidas/`, que está ignorada por git.

1. Preparar los datos y subirlos al bucket:

   ```bash
   mkdir -p gcp/salidas
   python gcp/preparar_datos.py mlops_pipeline/src mlops_pipeline/BD_creditos.csv gcp/salidas
   gcloud storage cp gcp/salidas/creditos_vertex.csv gs://nifty-realm-511821-g3-vertex/data/
   ```

2. Lanzar AutoML Tabular (tarda unas 2,5 horas):

   ```bash
   python gcp/lanzar_automl.py gs://nifty-realm-511821-g3-vertex/data/creditos_vertex.csv
   ```

3. Compilar y ejecutar el pipeline custom (unos 10 minutos):

   ```bash
   cd gcp && python pipeline_creditos.py --run
   ```

4. Descargar las predicciones a `gcp/salidas/`:
   - `vertex_pred_<algoritmo>.csv`: el artefacto `predicciones` de cada paso de entrenamiento del pipeline.
   - `automl_pred.csv`: las columnas `row_id` y `p`, donde `p` es `Pago_atiempo_1_scores` de una predicción por lotes de AutoML sobre las filas TEST.

   Después, generar la comparación:

   ```bash
   python gcp/analizar_resultados.py gcp/salidas gcp/salidas/graficos
   ```

   El gráfico de consistencia solo se genera si existe `gcp/salidas/cv_resumen.json`, con la media y la desviación del ROC-AUC en validación cruzada de cada modelo.

> Nota: la evaluación que muestra la consola de Vertex para AutoML promedia las dos clases
> (ROC-AUC 0,972). Para compararla con los demás modelos se usan las probabilidades de la
> predicción por lotes (ROC-AUC de la clase 1: 0,805).
