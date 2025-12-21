import os
import time
import mlflow
import mlflow.pyfunc
import pandas as pd
from fastapi import FastAPI
from contextlib import asynccontextmanager
from prometheus_client import Counter, Histogram, generate_latest
from prometheus_client import CONTENT_TYPE_LATEST

model = None  # global, diisi saat startup

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model

    run_id = os.getenv("MLFLOW_RUN_ID")
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "file:/mlruns")

    if not run_id:
        raise RuntimeError("MLFLOW_RUN_ID environment variable is not set")

    mlflow.set_tracking_uri(tracking_uri)

    print(f"[LIFESPAN] Loading model from run_id={run_id}")
    model = mlflow.pyfunc.load_model(f"runs:/{run_id}/model")
    print("[LIFESPAN] Model loaded successfully")

    yield  # ⬅️ app runs here

    print("[LIFESPAN] Application shutdown")

app = FastAPI(lifespan=lifespan)

# ===== PROMETHEUS METRICS =====
REQUEST_COUNT = Counter("inference_requests_total", "Total inference requests")
REQUEST_LATENCY = Histogram("inference_latency_seconds", "Inference latency")
ERROR_COUNT = Counter("inference_errors_total", "Total inference errors")
# =================================


@app.post("/predict")
def predict(data: dict):
    start = time.time()
    REQUEST_COUNT.inc()

    try:
        df = pd.DataFrame([data])
        preds = model.predict(df)

        REQUEST_LATENCY.observe(time.time() - start)
        return {"prediction": preds.tolist()}

    except Exception as e:
        ERROR_COUNT.inc()
        return {"error": str(e)}


@app.get("/metrics")
def metrics():
    return generate_latest(), 200, {"Content-Type": CONTENT_TYPE_LATEST}
