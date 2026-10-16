# Dataset Notes

## Dataset
Crop_recommendation.csv

## Records
2200

## Input Features
- N: Nitrogen
- P: Phosphorus
- K: Potassium
- temperature
- humidity
- ph
- rainfall

## Target
label (recommended crop class)

## Current Use
Training a crop classification model.

## Limitations
- The dataset does not establish laboratory soil-test units.
- It does not contain fertilizer dosage labels.
- It must not be used alone to prescribe fertilizer quantities.
- Recommendations for Indian farmers require appropriate regional
  soil-test data and crop-specific fertilizer guidelines.

  ## Fertilizer Recommendation Methodology

The Government of India's Methods Manual: Soil Testing in India
(2011) describes several approaches:

1. Generalized fertilizer recommendations
2. Soil-test rating and fertilizer adjustment
3. Critical nutrient level or sufficiency concept
4. Fertilizer recommendation for a targeted yield
5. Fertilizer recommendation for a crop sequence

The project will not prescribe fertilizer quantities using arbitrary
NPK thresholds. Recommendations will require suitable soil-test
methods, crop-specific calibration, and regional reference data.

The existing Crop_recommendation.csv dataset is used for crop
classification only, not fertilizer dosage prediction.