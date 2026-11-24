import os
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

warnings.filterwarnings("ignore")

DATA_FILE = "BlinkIT Grocery Data.xlsx"
OUTPUT_DIR = "blinkit_outputs"
MODEL_FILE = os.path.join(OUTPUT_DIR, "blinkit_sales_model.joblib")
RANDOM_STATE = 42

os.makedirs(OUTPUT_DIR, exist_ok=True)
sns.set_theme(style="whitegrid")


def load_data():
    """Load the Excel dataset and check required columns."""
    if not os.path.exists(DATA_FILE):
        raise FileNotFoundError(
            f"Cannot find '{DATA_FILE}'. Put it in the same folder as this script."
        )

    df = pd.read_excel(DATA_FILE)

    required = [
        "Item Fat Content", "Item Type", "Outlet Establishment Year",
        "Outlet Location Type", "Outlet Size", "Outlet Type",
        "Item Visibility", "Item Weight", "Sales", "Rating"
    ]
    missing_columns = [col for col in required if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Dataset is missing these columns: {missing_columns}")

    print("\n========== DATASET OVERVIEW ==========")
    print("Rows and columns:", df.shape)
    print("\nColumn names:")
    print(df.columns.tolist())
    print("\nFirst five records:")
    print(df.head())
    return df


def data_quality_report(df):
    """Report missing values, duplicate records and IQR outliers."""
    print("\n========== DATA QUALITY REPORT ==========")

    missing = df.isnull().sum().sort_values(ascending=False)
    print("\nMissing values:")
    print(missing[missing > 0] if (missing > 0).any() else "No missing values")

    duplicate_count = int(df.duplicated().sum())
    print("\nExact duplicate rows:", duplicate_count)

    # Save a quality report. Duplicates are reported, not automatically removed:
    # repeated products/outlets can represent legitimate separate records.
    quality = pd.DataFrame({
        "missing_values": df.isnull().sum(),
        "missing_percent": (df.isnull().mean() * 100).round(2),
        "data_type": df.dtypes.astype(str)
    })
    quality.to_csv(os.path.join(OUTPUT_DIR, "data_quality_report.csv"))

    numeric_cols = df.select_dtypes(include=np.number).columns
    outlier_rows = []
    for col in numeric_cols:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        count = int(((df[col] < lower) | (df[col] > upper)).sum())