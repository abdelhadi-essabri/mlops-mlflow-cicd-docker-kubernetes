from fastapi import FastAPI
import os
import socket

app = FastAPI()

# Simule un état "modèle chargé" pour tester les probes plus tard
model_loaded = True

@app.get("/")
def root():
    return {
        "message": "Fraud API demo",
        "pod_hostname": socket.gethostname(),  # utile pour voir quel Pod répond
        "version": os.getenv("APP_VERSION", "v1")
    }

@app.get("/health/live")
def liveness():
    return {"status": "alive"}

@app.get("/health/ready")
def readiness():
    if not model_loaded:
        return {"status": "not ready"}, 503
    return {"status": "ready"}

@app.get("/predict")
def predict():
    return {"prediction": "fraud", "score": 0.87}