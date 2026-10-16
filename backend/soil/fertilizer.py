from pathlib import Path
import pandas as pd
import math
ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "raw" / "Crop_recommendation.csv"

CROP_NAMES = set(
    pd.read_csv(DATA_PATH)["label"].str.strip().str.lower().unique()
)
def recommend_fertilizer(
    crop: str,
    nitrogen: float,
    phosphorus: float,
    potassium: float
) -> dict:
    """
    Prepare soil nutrient information for future
    crop-specific fertilizer recommendations.
    """

    if not crop.strip():
        raise ValueError("Crop name cannot be empty.")
    
    crop = crop.strip().lower()
    if crop not in CROP_NAMES:
        raise ValueError(
            f"Unsupported crop '{crop}'. "
            "Please use a crop supported by the dataset."
        )

    nutrients = {
        "nitrogen": nitrogen,
        "phosphorus": phosphorus,
        "potassium": potassium
    }

    for name, value in nutrients.items():
        if not math.isfinite(value) or value < 0:
            raise ValueError(
                f"{name} must be a finite, non-negative number."
            )

    return {
        "crop": crop.strip(),
        "soil_nutrients": nutrients,
        "fertilizer_recommendation": None,
        "status": "Reference data required",
        "notice": (
            "A fertilizer dosage cannot be recommended safely "
            "without suitable soil-test units, crop-specific "
            "calibration, and regional reference data."
        )
    }
if __name__ == "__main__":
    result = recommend_fertilizer(
        crop="rice",
        nitrogen=90,
        phosphorus=42,
        potassium=43
    )

    print(result)