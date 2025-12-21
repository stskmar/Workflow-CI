import time
import mlflow.pyfunc
from fastapi import FastAPI
from prometheus_client import Counter, Histogram, Gauge, generate_latest
from prometheus_client import CONTENT_TYPE_LATEST
import pandas as pd
import os
import mlflow.pyfunc

mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "file:/mlruns"))

# Load the model from MLflow Model Registry
model = mlflow.pyfunc.load_model(
    model_uri="models:/workflow-ci-mlflow/Production"
)

app = FastAPI()

REQUEST_COUNT = Counter("inference_requests_total", "Total inference requests")
REQUEST_LATENCY = Histogram("inference_latency_seconds", "Inference latency")
PREDICTION_COUNT = Counter("prediction_total", "Total predictions", ["label"])
ERROR_COUNT = Counter("inference_errors_total", "Total inference errors")

CPU_USAGE = Gauge("cpu_usage_percent", "CPU usage")
MEMORY_USAGE = Gauge("memory_usage_percent", "Memory usage")

@app.post("/predict")
def predict(data: dict):
    start_time = time.time()
    REQUEST_COUNT.inc()
    try:
        df = pd.DataFrame([data])
        preds = model.predict(df)[0]
        PREDICTION_COUNT.labels(label=str(preds)).inc()
        REQUEST_LATENCY.observe(time.time() - start_time)
        return {"prediction": int(preds)}
    except Exception as e:
        ERROR_COUNT.inc()
        return {"error": str(e)}

@app.get("/metrics")
def metrics():
    return generate_latest(), 200, {"Content-Type": CONTENT_TYPE_LATEST}
