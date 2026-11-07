FROM jenkins/jenkins:lts-jdk17

# La imagen oficial no trae Python, y el pipeline lo necesita para evaluar el modelo.
USER root
RUN apt-get update \
    && apt-get install -y --no-install-recommends python3 python3-venv \
    && rm -rf /var/lib/apt/lists/*
