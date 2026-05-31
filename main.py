from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import joblib
import os

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
LABELS_DIR = os.path.join(os.path.dirname(__file__), "labels")

MODEL_FILES = {
    "random_forest": "random_forest_binary.pkl",
    "decision_tree": "decision_tree_binary.pkl",
    "knn": "knn_binary.pkl",
    "logistic_regression": "logistic_regressor_binary.pkl",
    "svm": "lsvm_binary.pkl",
    "mlp": "mlp_binary.pkl",
    "linear_regression": "linear_regressor_binary.pkl",
}

state: dict = {"models": {}, "scaler": None, "le1_classes": None}


@asynccontextmanager
async def lifespan(app: FastAPI):
    scaler_path = os.path.join(MODELS_DIR, "scaler.pkl")
    if os.path.exists(scaler_path):
        state["scaler"] = joblib.load(scaler_path)

    le1_path = os.path.join(LABELS_DIR, "le1_classes.npy")
    if os.path.exists(le1_path):
        state["le1_classes"] = np.load(le1_path, allow_pickle=True)

    for name, filename in MODEL_FILES.items():
        path = os.path.join(MODELS_DIR, filename)
        if os.path.exists(path):
            state["models"][name] = joblib.load(path)

    yield


app = FastAPI(lifespan=lifespan)


class FlowFeatures(BaseModel):
    rate: float
    sttl: float
    sload: float
    dload: float
    ct_srv_src: int
    ct_state_ttl: int
    ct_dst_ltm: int
    ct_src_dport_ltm: int
    ct_dst_sport_ltm: int
    ct_dst_src_ltm: int
    ct_src_ltm: int
    ct_srv_dst: int
    state_CON: int
    state_INT: int


class PredictRequest(BaseModel):
    flows: list[FlowFeatures]


@app.get("/health")
def health():
    return {
        "status": "ok",
        "models_loaded": len(state["models"]),
        "scaler_ready": state["scaler"] is not None,
    }


@app.post("/predict")
def predict(request: PredictRequest):
    models = state["models"]
    scaler = state["scaler"]
    le1_classes = state["le1_classes"]

    if not models:
        raise HTTPException(status_code=503, detail="Models not loaded")
    if scaler is None:
        raise HTTPException(
            status_code=503,
            detail="Scaler not loaded — commit models/scaler.pkl to the repository",
        )

    X = np.array(
        [
            [
                f.rate, f.sttl, f.sload, f.dload,
                f.ct_srv_src, f.ct_state_ttl, f.ct_dst_ltm, f.ct_src_dport_ltm,
                f.ct_dst_sport_ltm, f.ct_dst_src_ltm, f.ct_src_ltm, f.ct_srv_dst,
                f.state_CON, f.state_INT,
            ]
            for f in request.flows
        ]
    )

    X_scaled = scaler.transform(X)

    def decode_label(pred) -> str:  # type: ignore[no-untyped-def]
        if le1_classes is not None:
            idx = int(round(float(pred))) if isinstance(pred, (float, np.floating)) else int(pred)
            if 0 <= idx < len(le1_classes):
                return str(le1_classes[idx])
        return "abnormal" if int(round(float(pred))) == 0 else "normal"

    results = []
    for i in range(len(request.flows)):
        row = X_scaled[i : i + 1]
        votes: dict[str, str] = {}

        for name, model in models.items():
            try:
                raw = model.predict(row)[0]
                votes[name] = decode_label(raw)
            except Exception:
                votes[name] = "error"

        rf_label = votes.get("random_forest", "unknown")
        rf_confidence = 0.5
        rf = models.get("random_forest")
        if rf is not None:
            try:
                rf_confidence = float(np.max(rf.predict_proba(row)[0]))
            except Exception:
                pass

        results.append(
            {
                "binary_label": rf_label,
                "confidence": rf_confidence,
                "model_votes": votes,
                "model_count": len(votes),
            }
        )

    return {"predictions": results}
