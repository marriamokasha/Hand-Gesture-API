import os
from fastapi.middleware.cors import CORSMiddleware

from fastapi import FastAPI
import joblib
import numpy as np
from pydantic import BaseModel
import uvicorn
from prometheus_fastapi_instrumentator import Instrumentator

# Local imports
try:
    try:
        from .utils import normalize_landmarks, gesture_map  # Run as module
    except:
        from utils import normalize_landmarks, gesture_map   # Run directly
except:
    from app.utils import normalize_landmarks, gesture_map   # Run directly

app = FastAPI()
Instrumentator().instrument(app).expose(app)

# Dynamically resolve model path
model = None
model_path = os.path.join(os.path.dirname(__file__), "../models/optim_xgboost_model.pkl")
if os.path.exists(model_path):
    model = joblib.load(model_path)
    print(f"✅ Model loaded from: {model_path}")
else:
    raise FileNotFoundError(f"❌ Model not found at {model_path}")


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (or specify exact frontend URL for better security)
    allow_methods=["*"],
    allow_headers=["*"],
)
# Input schema
class InputFeatures(BaseModel):
    landmarks: list[float]

@app.get("/")
def read_root():
    return {"message": "Hand Gesture Classification API is running!"}

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/predict")
def predict(data: InputFeatures):
    try:
        normalized_df = normalize_landmarks(data.landmarks)
        features = normalized_df.values  # shape (1, 63)
        prediction = model.predict(features)[0]
        gesture_name = gesture_map.get(prediction, "unknown")

        return {
            "prediction": int(prediction),
            "gesture": gesture_name
        }

    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
