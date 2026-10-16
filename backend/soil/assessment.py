import math
def assess_soil_ph(ph: float) -> dict:
    """Provide a basic pH-based soil assessment."""

    if not math.isfinite(ph) or not 0 <= ph <= 14:
        raise ValueError("Soil pH must be between 0 and 14.")

    if ph < 5.5:
        status = "Strongly acidic"
    elif ph < 6.5:
        status = "Moderately acidic"
    elif ph <= 7.5:
        status = "Near neutral"
    elif ph <= 8.5:
        status = "Moderately alkaline"
    else:
        status = "Strongly alkaline"

    return {
        "ph": ph,
        "assessment": status,
        "interpretation": "Soil pH can affect nutrient availability; suitability depends on the crop and local soil conditions.",
        "notice": (
            "This is a preliminary pH classification, "
            "not a complete soil-health diagnosis."
        )
    }


def assess_nutrients(n: float, p: float, k: float) -> dict:
    """Validate and return the supplied NPK values."""

    if not all(math.isfinite(value) and value >= 0 for value in (n, p, k)):
        raise ValueError("NPK values cannot be negative.")

    return {
        "nitrogen": n,
        "assessment_type": "NPK measurement recording only",
        "phosphorus": p,
        "potassium": k,
        "units": "Not specified by the dataset",
        "notice": (
            "NPK values are recorded, but nutrient status "
            "requires crop- and region-specific reference ranges."
        )
    }
