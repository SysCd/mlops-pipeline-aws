FROM python:3.10-slim

WORKDIR /mlflow

RUN pip install --no-cache-dir mlflow boto3

RUN mkdir -p /mlflow/artifacts

EXPOSE 8000

CMD ["mlflow", "server", "--host", "0.0.0.0", "--port", "8000", "--backend-store-uri", "sqlite:///mlflow.db", "--default-artifact-root", "/mlflow/artifacts"]
