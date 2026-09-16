FROM python:3.11-slim

# Se replica la misma estructura relativa del repo (mlops_pipeline/src/ un
# nivel bajo mlops_pipeline/BD_creditos.csv) para que las rutas relativas
# de config.json y los scripts (p. ej. '../BD_creditos.csv') funcionen
# idénticas dentro del contenedor, sin rutas especiales para Docker.
WORKDIR /app/mlops_pipeline

COPY mlops_pipeline/src/requirements.txt src/requirements.txt
RUN pip install --no-cache-dir -r src/requirements.txt flask==3.1.0 joblib==1.5.1

# model_artifact.joblib debe existir antes del build (lo genera
# `python model_training.py`); se commitea junto al resto de artefactos
# generados del proyecto (mismo criterio que BD_creditos_modelo.csv/.pkl).
COPY mlops_pipeline/BD_creditos.csv BD_creditos.csv
COPY mlops_pipeline/src/ src/

WORKDIR /app/mlops_pipeline/src

EXPOSE 5000

CMD ["python", "model_deploy.py"]