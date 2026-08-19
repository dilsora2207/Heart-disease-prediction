import pickle
import numpy as np
import pandas as pd
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from typing import Optional

# =====================================================
#  App Setup
# =====================================================

app = FastAPI(title="Heart Disease Prediction")
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# =====================================================
#  Load Model, Scaler, Column Order
# =====================================================

MODEL_PATH   = r"C:\Users\ofniz\Desktop\Heart Disease Prediction\Models\Gradient Boosting(optimization)\model.pkl"
SCALER_PATH  = r"C:\Users\ofniz\Desktop\Heart Disease Prediction\Data\processed\scaler.pkl"
COLUMNS_PATH = r"C:\Users\ofniz\Desktop\Heart Disease Prediction\Data\processed\columns.pkl"

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

with open(SCALER_PATH, "rb") as f:
    scaler = pickle.load(f)

with open(COLUMNS_PATH, "rb") as f:
    feature_columns = pickle.load(f)

print("Model loaded successfully")
print("Scaler loaded successfully")
print("Feature columns:", feature_columns)

# =====================================================
#  Risk Level Logic
# =====================================================

def get_risk_level(probability: float) -> dict:
    if probability < 0.35:
        return {"label": "Low Risk",      "color": "#22c55e", "icon": "✅"}
    elif probability < 0.60:
        return {"label": "Moderate Risk", "color": "#f59e0b", "icon": "⚠️"}
    else:
        return {"label": "High Risk",     "color": "#ef4444", "icon": "🚨"}

# =====================================================
#  Routes
# =====================================================

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/predict", response_class=HTMLResponse)
async def predict(
    request:    Request,
    patient_id: str   = Form(...),
    age:        float = Form(...),
    sex:        float = Form(...),
    cp:         float = Form(...),
    trestbps:   float = Form(...),
    chol:       float = Form(...),
    fbs:        float = Form(...),
    restecg:    float = Form(...),
    thalach:    float = Form(...),
    exang:      float = Form(...),
    oldpeak:    float = Form(...),
    slope:      float = Form(...),
    ca:         float = Form(...),
):
    input_dict = {
        feature_columns[0]:  age,
        feature_columns[1]:  sex,
        feature_columns[2]:  cp,
        feature_columns[3]:  trestbps,
        feature_columns[4]:  chol,
        feature_columns[5]:  fbs,
        feature_columns[6]:  restecg,
        feature_columns[7]:  thalach,
        feature_columns[8]:  exang,
        feature_columns[9]:  oldpeak,
        feature_columns[10]: slope,
        feature_columns[11]: ca,
    }

    input_df     = pd.DataFrame([input_dict])
    input_scaled = scaler.transform(input_df)

    probability = float(model.predict_proba(input_scaled)[0][1])
    prediction  = int(model.predict(input_scaled)[0])
    risk        = get_risk_level(probability)

    print(f"Patient: {patient_id} | Probability: {probability:.4f} → {risk['label']}")

    return templates.TemplateResponse("index.html", {
        "request":      request,
        "result":       True,
        "patient_id":   patient_id,
        "probability":  round(probability * 100, 1),
        "prediction":   prediction,
        "risk_label":   risk["label"],
        "risk_color":   risk["color"],
        "risk_icon":    risk["icon"],
        # Pass back form values
        "age": age, "sex": sex, "cp": cp, "trestbps": trestbps,
        "chol": chol, "fbs": fbs, "restecg": restecg, "thalach": thalach,
        "exang": exang, "oldpeak": oldpeak, "slope": slope, "ca": ca,
    })