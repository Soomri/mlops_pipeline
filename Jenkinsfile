pipeline {
    agent any

    options {
        // El clon es explícito en la primera etapa; sin esto se clonaría dos veces.
        skipDefaultCheckout(true)
    }

    // Jenkins corre en local y GitHub no puede enviarle webhooks, así que se
    // revisa master cada minuto: un merge a master dispara el flujo.
    triggers {
        pollSCM('* * * * *')
    }

    stages {
        stage('Clonar Repositorio') {
            steps {
                echo 'Clonando repositorio desde GitHub...'
                git branch: 'master', url: 'https://github.com/Soomri/mlops_pipeline.git'
            }
        }

        stage('Validar Estructura de Carpetas') {
            steps {
                echo 'Validando estructura del proyecto...'
                sh '''
                    for carpeta in mlops_pipeline mlops_pipeline/src; do
                        test -d "$carpeta" || { echo "Falta la carpeta: $carpeta"; exit 1; }
                    done
                    for archivo in \
                        Dockerfile \
                        mlops_pipeline/BD_creditos.csv \
                        mlops_pipeline/src/config.json \
                        mlops_pipeline/src/requirements.txt \
                        mlops_pipeline/src/ft_engineering.py \
                        mlops_pipeline/src/model_training.py \
                        mlops_pipeline/src/model_deploy.py \
                        mlops_pipeline/src/model_evaluation.py \
                        mlops_pipeline/src/model_monitoring.py \
                        mlops_pipeline/src/validar_metricas.py \
                        mlops_pipeline/src/model_artifact.joblib \
                        mlops_pipeline/src/holdout_test.csv; do
                        test -f "$archivo" || { echo "Falta el archivo: $archivo"; exit 1; }
                    done
                    echo "Estructura del proyecto validada."
                '''
            }
        }

        stage('Instalar Dependencias') {
            steps {
                echo 'Instalando dependencias en un entorno virtual...'
                // Mismas dependencias que el Dockerfile del modelo.
                sh '''
                    python3 -m venv .venv
                    .venv/bin/pip install --quiet -r mlops_pipeline/src/requirements.txt flask==3.1.0 joblib==1.5.1
                '''
            }
        }

        stage('Validar Métricas de Evaluación') {
            steps {
                dir('mlops_pipeline/src') {
                    echo 'Evaluando el modelo contra el holdout...'
                    sh '../../.venv/bin/python model_evaluation.py'
                    echo 'Comparando las métricas contra los umbrales de config.json...'
                    sh '../../.venv/bin/python validar_metricas.py'
                }
            }
        }
    }

    post {
        always {
            archiveArtifacts artifacts: 'mlops_pipeline/src/model_evaluation_report.json, mlops_pipeline/src/model_evaluation_matriz_confusion.png',
                             allowEmptyArchive: true
            script {
                // Un fallo del correo no debe cambiar el veredicto del pipeline.
                try {
                    mail to: 'sofia.mejiar@upb.edu.co',
                         subject: "Pipeline ${env.JOB_NAME} #${env.BUILD_NUMBER}: ${currentBuild.currentResult}",
                         body: """El flujo de Jenkins terminó para el proyecto mlops_pipeline.

Resultado: ${currentBuild.currentResult}
Duración: ${currentBuild.durationString}
Detalle: ${env.BUILD_URL}console
"""
                    echo 'Notificación enviada por correo.'
                } catch (Exception e) {
                    echo "ADVERTENCIA: no se pudo enviar el correo (revisa el SMTP en Manage Jenkins > System): ${e.getMessage()}"
                }
            }
        }
    }
}