from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sklearn.preprocessing import MinMaxScaler
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

# MinMaxScaler parameters fitted on all 175 341 rows of UNSW-NB15.
# Feature order matches the 14 binary features exactly:
#   rate, sttl, sload, dload, ct_srv_src, ct_state_ttl, ct_dst_ltm,
#   ct_src_dport_ltm, ct_dst_sport_ltm, ct_dst_src_ltm, ct_src_ltm,
#   ct_srv_dst, state_CON, state_INT
_DATA_MIN = np.array(
    [0., 0., 0., 0., 1., 0., 1., 1., 1., 1., 1., 1., 0., 0.],
    dtype=np.float64,
)
_DATA_MAX = np.array(
    [1_000_000., 255., 5_988_000_256., 22_422_730.,
     63., 6., 51., 51., 46., 65., 60., 62., 1., 1.],
    dtype=np.float64,
)


def _build_scaler() -> MinMaxScaler:
    s = MinMaxScaler(feature_range=(0, 1))
    s.data_min_ = _DATA_MIN
    s.data_max_ = _DATA_MAX
    s.data_range_ = _DATA_MAX - _DATA_MIN
    s.scale_ = 1.0 / s.data_range_
    s.min_ = -_DATA_MIN * s.scale_
    s.n_features_in_ = 14
    s.n_samples_seen_ = 175_341
    return s


state: dict = {"models": {}, "scaler": None, "le1_classes": None}


@asynccontextmanager
async def lifespan(app: FastAPI):
    state["scaler"] = _build_scaler()

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
        raise HTTPException(status_code=503, detail="Scaler not ready")

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
