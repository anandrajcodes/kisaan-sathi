
from pathlib import Path
import pandas as pd

DATA_PATH = (
    Path(__file__).resolve().parents[2]
    / "data" / "raw" / "Crop_recommendation.csv"
)

if not DATA_PATH.exists():
    raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

df = pd.read_csv(DATA_PATH)

print("\nDataset shape:", df.shape)
print("\nColumn names:")
print(df.columns.tolist())

print("\nFirst five rows:")
print(df.head())

print("\nMissing values:")
print(df.isnull().sum())

print("\nData types:")
print(df.dtypes)
print(df["label"].value_counts())
print(df[["N", "P", "K"]].describe())
print(sorted(df["label"].unique()))