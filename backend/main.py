
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI,HTTPException
from pydantic import BaseModel, Field

from soil.assessment import assess_soil_ph, assess_nutrients
from soil.fertilizer import recommend_fertilizer
from pydantic import BaseModel, Field, field_validator


# Project paths
ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "crop_model.joblib"


# Initialize FastAPI
app = FastAPI(
    title="SoilHealthAI",
    description="AI-driven soil assessment and crop suitability",
    version="0.3.0"
)


# Load the trained crop model
saved = joblib.load(MODEL_PATH)
model = saved["model"]
features = saved["features"]


# Request models
class CropInput(BaseModel):
    N: float = Field(ge=0, le=1000)
    P: float = Field(ge=0, le=1000)
    K: float = Field(ge=0, le=1000)
    temperature: float = Field(ge=-10, le=60)
    humidity: float = Field(ge=0, le=100)
    ph: float = Field(ge=0, le=14)
    rainfall: float = Field(ge=0, le=5000)


class SoilPHInput(BaseModel):
    ph: float = Field(ge=0, le=14)


class SoilNPKInput(BaseModel):
    n: float = Field(ge=0)
    p: float = Field(ge=0)
    k: float = Field(ge=0)
    
class FertilizerInput(BaseModel):
    crop: str = Field(min_length=1)
    nitrogen: float = Field(ge=0, allow_inf_nan=False)
    phosphorus: float = Field(ge=0, allow_inf_nan=False)
    potassium: float = Field(ge=0, allow_inf_nan=False)

    @field_validator("crop")
    @classmethod
    def validate_crop(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Crop name cannot be empty.")
        return value


# Home endpoint
@app.get("/")
def home():
    return {
        "project": "SoilHealthAI",
        "status": "Backend running successfully"
    }


# Health check endpoint
@app.get("/health")
def health_check():
    return {"status": "healthy"}


# Crop prediction endpoint
@app.post("/predict-crop")
def predict_crop(data: CropInput):
    values = data.model_dump()
    input_df = pd.DataFrame([values], columns=features)

    prediction = model.predict(input_df)[0]
    probabilities = model.predict_proba(input_df)[0]

    top_indices = probabilities.argsort()[::-1][:3]

    return {
        "predicted_crop": str(prediction),
        "top_3_candidates": [
            {
                "crop": str(model.classes_[i]),
                "model_score": round(float(probabilities[i]), 4)
            }
            for i in top_indices
        ],
        "notice": (
            "Dataset-based crop suitability prediction only. "
            "Not a validated farm-specific recommendation."
        )
    }


# Soil pH assessment endpoint
@app.post("/assess-soil-ph")
def assess_soil_ph_endpoint(data: SoilPHInput):
    return assess_soil_ph(data.ph)


# Soil NPK assessment endpoint
@app.post("/assess-soil-npk")
def assess_soil_npk(data: SoilNPKInput):
    return assess_nutrients(data.n, data.p, data.k)

@app.post("/recommend-fertilizer")
def fertilizer_endpoint(data: FertilizerInput):
    try:
        return recommend_fertilizer(
            crop=data.crop,
            nitrogen=data.nitrogen,
            phosphorus=data.phosphorus,
            potassium=data.potassium
        )
    except ValueError as error:
        raise HTTPException(
            status_code=422,
            detail=str(error)
        )