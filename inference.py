import os
import time
import mlflow
import mlflow.pyfunc
import pandas as pd
from fastapi import FastAPI
from contextlib import asynccontextmanager
from prometheus_client import Counter, Histogram, generate_latest
from prometheus_client import CONTENT_TYPE_LATEST
from pathlib import Path

model = None  # global, diisi saat startup

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "file:/mlruns")
    mlflow.set_tracking_uri(tracking_uri)

    latest_file = Path("/mlruns/run_id.txt")
    if not latest_file.exists():
        raise RuntimeError("run_id.txt not found")

    run_id = latest_file.read_text().strip()
    print(f"[LIFESPAN] Using latest run_id={run_id}")

    model = mlflow.pyfunc.load_model(f"runs:/{run_id}/model")
    print("[LIFESPAN] Model loaded successfully")

    yield

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
