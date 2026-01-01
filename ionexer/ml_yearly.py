from __future__ import annotations
from datetime import date, timedelta
from typing import List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from .file_manager import FileManager
from .parser import Parser



# Load TEC for entire year at one Iran grid point

def load_year_tec(year: int, lat: float = 32.0, lon: float = 53.0) -> pd.DataFrame:
    fm = FileManager()
    rows = []
    first_lat_idx = None
    first_lon_idx = None

    start = date(year, 1, 1)
    end = date(year, 12, 2)
    current = start

    while current <= end:
        path = fm.get_file_for_date(current)
        if path is None:
            current += timedelta(days=1)
            continue

        parser = Parser(path)
        try:
            tec_3d, lats, lons, epochs = parser.parse_all_epochs()
        except Exception as e:
            print(f"[WARN] Failed to parse {current}: {e}")
            current += timedelta(days=1)
            continue

        if first_lat_idx is None:
            lons_wrapped = lons.copy()
            if np.max(lons_wrapped) > 180:
                lons_wrapped = (lons_wrapped + 180) % 360 - 180
            first_lat_idx = int(np.argmin(np.abs(lats - lat)))
            first_lon_idx = int(np.argmin(np.abs(lons_wrapped - lon)))
            print(f"[INFO] Using Iran grid point lat={lats[first_lat_idx]:.2f}, "
                  f"lon={lons_wrapped[first_lon_idx]:.2f}")

        for i, ts in enumerate(epochs):
            value = tec_3d[i, first_lat_idx, first_lon_idx]
            if np.isfinite(value):
                rows.append({"datetime": ts, "tec": float(value)})

        current += timedelta(days=1)

    df = pd.DataFrame(rows).sort_values("datetime").reset_index(drop=True)
    return df



# Feature engineering

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    dt = pd.to_datetime(df["datetime"])
    df = df.copy()

    df["day_of_year"] = dt.dt.dayofyear
    df["hour"] = dt.dt.hour + dt.dt.minute / 60.0

    df["sin_hour"] = np.sin(2 * np.pi * df["hour"] / 24.0)
    df["cos_hour"] = np.cos(2 * np.pi * df["hour"] / 24.0)

    df["sin_doy"] = np.sin(2 * np.pi * df["day_of_year"] / 365.0)
    df["cos_doy"] = np.cos(2 * np.pi * df["day_of_year"] / 365.0)

    return df



# Train-test split: 80/20


def split_80_20(df: pd.DataFrame):
    feature_cols = ["day_of_year", "hour", "sin_hour", "cos_hour", "sin_doy", "cos_doy"]
    X = df[feature_cols].values
    y = df["tec"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, shuffle=True
    )
    return X_train, X_test, y_train, y_test, feature_cols



# Train model


def train_model(X_train, y_train):
    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=None,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model



# Evaluate

def evaluate(model, X, y, label="Test"):
    pred = model.predict(X)
    mse = mean_squared_error(y, pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y, pred)
    r2 = r2_score(y, pred)

    print(f"\n=== {label} Evaluation ===")
    print(f"RMSE = {rmse:.3f}")
    print(f"MAE  = {mae:.3f}")
    print(f"R²   = {r2:.3f}")

    return pred


# Predict specific day

def predict_specific_day(model, df, feature_cols: List[str], target_day: date):
    dt = pd.to_datetime(df["datetime"])
    mask = dt.dt.date == target_day
    df_day = df[mask].copy()

    if df_day.empty:
        print(f"[WARN] No data for {target_day}")
        return

    X = df_day[feature_cols].values
    y_true = df_day["tec"].values
    y_pred = model.predict(X)
    hours = df_day["hour"].values

    plt.figure(figsize=(10, 5))
    plt.plot(hours, y_true, "o-", label="True TEC")
    plt.plot(hours, y_pred, "s--", label="Predicted TEC")
    plt.title(f"True vs Predicted TEC — {target_day}")
    plt.xlabel("UTC Hour")
    plt.ylabel("TEC")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()





def run_yearly_ml_for_10_january():
    year = 2025

    print(f"[INFO] Loading TEC for full year {year} over Iran...")
    df = load_year_tec(year=year)

    # Add time features (day_of_year, hour, sin/cos, ...)
    df = add_features(df)

    # 80/20 train-test split
    X_train, X_test, y_train, y_test, feature_cols = split_80_20(df)
    print(f"[INFO] Training samples: {len(y_train)}, Test samples: {len(y_test)}")

    # Train Random Forest
    model = train_model(X_train, y_train)

    # Evaluate on train & test
    evaluate(model, X_train, y_train, label="Train")
    evaluate(model, X_test,  y_test,  label="Test")

    # ---- Plot 10 January 2025 ----
    target_day = date(2025, 1, 24)
    print(f"[INFO] Plotting true vs predicted TEC for {target_day}")
    predict_specific_day(model, df, feature_cols, target_day)


if __name__ == "__main__":
    run_yearly_ml_for_10_january()
