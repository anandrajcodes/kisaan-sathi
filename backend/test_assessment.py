
import math
import pytest

from soil.assessment import assess_soil_ph, assess_nutrients
from soil.fertilizer import recommend_fertilizer


# ---------- Soil pH tests ----------

@pytest.mark.parametrize(
    "ph, expected",
    [
        (0, "Strongly acidic"),
        (5.2, "Strongly acidic"),
        (5.5, "Moderately acidic"),
        (6.5, "Near neutral"),
        (7.0, "Near neutral"),
        (7.5, "Near neutral"),
        (7.6, "Moderately alkaline"),
        (8.5, "Moderately alkaline"),
        (8.6, "Strongly alkaline"),
        (14, "Strongly alkaline"),
    ],
)
def test_ph_classification(ph, expected):
    result = assess_soil_ph(ph)
    assert result["assessment"] == expected
    assert result["ph"] == ph
    assert "notice" in result


@pytest.mark.parametrize(
    "ph",
    [-1, 15, float("nan"), float("inf"), float("-inf")],
)
def test_invalid_ph(ph):
    with pytest.raises(ValueError):
        assess_soil_ph(ph)


# ---------- NPK assessment tests ----------

def test_valid_npk():
    result = assess_nutrients(90, 42, 43)

    assert result["nitrogen"] == 90
    assert result["phosphorus"] == 42
    assert result["potassium"] == 43
    assert result["units"] == "Not specified by the dataset"
    assert result["assessment_type"] == "NPK measurement recording only"


@pytest.mark.parametrize(
    "n, p, k",
    [
        (-1, 42, 43),
        (90, -1, 43),
        (90, 42, -1),
        (float("nan"), 42, 43),
        (90, float("inf"), 43),
        (90, 42, float("-inf")),
    ],
)
def test_invalid_npk(n, p, k):
    with pytest.raises(ValueError):
        assess_nutrients(n, p, k)


# ---------- Fertilizer module tests ----------

def test_valid_fertilizer_request():
    result = recommend_fertilizer(
        crop="rice",
        nitrogen=90,
        phosphorus=42,
        potassium=43,
    )

    assert result["crop"] == "rice"
    assert result["soil_nutrients"]["nitrogen"] == 90
    assert result["fertilizer_recommendation"] is None
    assert result["status"] == "Reference data required"


@pytest.mark.parametrize("crop", ["dragonfruit", "unknown_crop"])
def test_unsupported_crop(crop):
    with pytest.raises(ValueError, match="Unsupported crop"):
        recommend_fertilizer(crop, 90, 42, 43)


@pytest.mark.parametrize("crop", ["", "   "])
def test_empty_crop(crop):
    with pytest.raises(ValueError, match="Crop name cannot be empty"):
        recommend_fertilizer(crop, 90, 42, 43)


@pytest.mark.parametrize(
    "n, p, k",
    [
        (-1, 42, 43),
        (90, -1, 43),
        (90, 42, -1),
        (float("nan"), 42, 43),
        (90, float("inf"), 43),
        (90, 42, float("-inf")),
    ],
)
def test_invalid_fertilizer_nutrients(n, p, k):
    with pytest.raises(ValueError):
        recommend_fertilizer("rice", n, p, k)


def test_crop_name_is_normalized():
    result = recommend_fertilizer("  RICE  ", 90, 42, 43)
    assert result["crop"] == "rice"
