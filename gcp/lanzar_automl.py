"""Crea el dataset tabular en Vertex AI y lanza el entrenamiento AutoML Tabular."""
import subprocess
import sys

from google.cloud import aiplatform
from google.oauth2.credentials import Credentials

PROJECT, REGION = "nifty-realm-511821-g3", "us-central1"
GCS_CSV = sys.argv[1]
token = subprocess.check_output("gcloud auth print-access-token", shell=True, text=True).strip()
aiplatform.init(project=PROJECT, location=REGION, credentials=Credentials(token),
                staging_bucket="gs://nifty-realm-511821-g3-vertex")

ds = aiplatform.TabularDataset.create(display_name="creditos_pago_atiempo", gcs_source=[GCS_CSV])
print("DATASET", ds.resource_name)

features = [c for c in ds.column_names if c not in ("row_id", "Pago_atiempo", "data_split")]
column_specs = {c: "auto" for c in features}
for c in ("tipo_credito", "tipo_laboral", "tendencia_ingresos", "mes_desembolso"):
    column_specs[c] = "categorical"

job = aiplatform.AutoMLTabularTrainingJob(
    display_name="automl_creditos_pago_atiempo",
    optimization_prediction_type="classification",
    optimization_objective="maximize-au-roc",
    column_specs=column_specs,
)
job.run(
    dataset=ds,
    target_column="Pago_atiempo",
    predefined_split_column_name="data_split",
    budget_milli_node_hours=1000,
    model_display_name="automl_creditos_modelo",
    disable_early_stopping=False,
    sync=False,
)
job.wait_for_resource_creation()
print("TRAINING_PIPELINE", job.resource_name)
