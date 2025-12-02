# ml_baseline.py

from __future__ import annotations

from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Tuple, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from .file_manager import FileManager
from .parser import Parser, IRAN_BBOX



def find_iran_grid_index(
    lats: np.ndarray,
    lons: np.ndarray,
    target_lat: float = 32.0,
    target_lon: float = 53.0,
) -> Tuple[int, int]:
    """
    Default location ~central Iran (around Isfahan/Yazd).
    """
   
    lons_wrapped = lons.copy()
    if np.max(lons_wrapped) > 180.0:
        lons_wrapped = (lons_wrapped + 180.0) % 360.0 - 180.0

    lat_idx = int(np.argmin(np.abs(lats - target_lat)))
    lon_idx = int(np.argmin(np.abs(lons_wrapped - target_lon)))
    return lat_idx, lon_idx



# Build time series TEC over Iran for a date range


def build_tec_timeseries_for_point(
    start_date: date,
    end_date: date,
    target_lat: float = 32.0,
    target_lon: float = 53.0,
) -> pd.DataFrame:
    
    fm = FileManager()
    rows: List[dict] = []

    current = start_date
    first_lat_idx = None
    first_lon_idx = None

    while current <= end_date:
        path = fm.get_file_for_date(current)
        if path is None:
            print(f"[WARN] No file for {current}, skipping.")
            current += timedelta(days=1)
            continue

        parser = Parser(path)
        try:
            tec_3d, lats, lons, epochs = parser.parse_all_epochs()
        except Exception as e:
            print(f"[WARN] Failed to parse all epochs for {current}: {e}")
            current += timedelta(days=1)
            continue

        # Find grid index once, then reuse for all days (to ensure consistency)
        if first_lat_idx is None or first_lon_idx is None:
            lat_idx, lon_idx = find_iran_grid_index(lats, lons, target_lat, target_lon)
            first_lat_idx, first_lon_idx = lat_idx, lon_idx
            print(f"[INFO] Using grid index (lat_idx={lat_idx}, lon_idx={lon_idx}) "
                  f"at lat={lats[lat_idx]:.2f}, lon={lons[lon_idx]:.2f}")
        else:
            lat_idx, lon_idx = first_lat_idx, first_lon_idx

        # For each epoch, take TEC at that grid point
        for t_idx, ts in enumerate(epochs):
            tec_value = float(tec_3d[t_idx, lat_idx, lon_idx])
            if not np.isfinite(tec_value):
                continue

            rows.append({
                "datetime": ts,
                "tec": tec_value,
            })

        current += timedelta(days=1)

    if not rows:
        raise RuntimeError("No TEC data collected for the given date range.")

    df = pd.DataFrame(rows)
    df = df.sort_values("datetime").reset_index(drop=True)
    return df




def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    
    dt = pd.to_datetime(df["datetime"])

    df = df.copy()
    df["day_of_year"] = dt.dt.dayofyear
    df["hour"] = dt.dt.hour + dt.dt.minute / 60.0

    # Encode daily cycle
    df["sin_hour"] = np.sin(2 * np.pi * df["hour"] / 24.0)
    df["cos_hour"] = np.cos(2 * np.pi * df["hour"] / 24.0)

    # Encode annual cycle
    df["sin_doy"] = np.sin(2 * np.pi * df["day_of_year"] / 365.0)
    df["cos_doy"] = np.cos(2 * np.pi * df["day_of_year"] / 365.0)

    return df




def train_test_split_by_date(
    df: pd.DataFrame,
    split_date: date,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split dataframe into train/test based on a calendar date.
    Everything before split_date goes to train,
    everything on/after split_date goes to test.
    """
    dt = pd.to_datetime(df["datetime"])
    mask_train = dt.dt.date < split_date
    df_train = df[mask_train].copy()
    df_test = df[~mask_train].copy()
    return df_train, df_test


def train_random_forest_model(df_train: pd.DataFrame):
    """
    Train a RandomForestRegressor on the given training dataframe.
    """
    feature_cols = ["day_of_year", "hour", "sin_hour", "cos_hour", "sin_doy", "cos_doy"]
    X_train = df_train[feature_cols].values
    y_train = df_train["tec"].values

    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=None,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model, feature_cols


def evaluate_model(
    model,
    feature_cols: List[str],
    df_test: pd.DataFrame,
    label: str = "Test",
):
   
    if df_test.empty:
        print(f"[WARN] No data in {label} set.")
        return

    X_test = df_test[feature_cols].values
    y_true = df_test["tec"].values
    y_pred = model.predict(X_test)

    # ✅ RMSE به‌صورت دستی: sqrt(MSE)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)

    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    print(f"=== {label} evaluation ===")
    print(f"RMSE = {rmse:.3f}")
    print(f"MAE  = {mae:.3f}")
    print(f"R^2  = {r2:.3f}")

    return y_true, y_pred





#Plot true vs predicted for a single day


def plot_day_prediction(
    model,
    feature_cols: List[str],
    df: pd.DataFrame,
    target_day: date,
):
    """
    For a specific calendar date, extract all rows, predict TEC, and
    plot true vs predicted as a function of time (UTC hour).
    """
    dt = pd.to_datetime(df["datetime"])
    mask = dt.dt.date == target_day
    df_day = df[mask].copy()

    if df_day.empty:
        print(f"[WARN] No data for day {target_day}.")
        return

    X = df_day[feature_cols].values
    y_true = df_day["tec"].values
    y_pred = model.predict(X)

    hours = dt[mask].dt.hour + dt[mask].dt.minute / 60.0

    plt.figure(figsize=(10, 5))
    plt.plot(hours, y_true, "o-", label="True TEC")
    plt.plot(hours, y_pred, "s--", label="Predicted TEC")
    plt.xlabel("UTC hour")
    plt.ylabel("TEC (TECU)")
    plt.title(f"TEC over Iran – True vs Predicted ({target_day.isoformat()})")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()




def run_ml_for_month(
    year: int = 2025,
    month: int = 1,
    split_day: int = 20,
    target_day_for_plot: int = 24,
):
    # Determine start/end of month
    start = date(year, month, 1)
    if month == 12:
        end = date(year, 12, 31)
    else:
        end = date(year, month + 1, 1) - timedelta(days=1)

    print(f"[INFO] Building TEC time series for {start} to {end} over Iran...")
    df = build_tec_timeseries_for_point(start, end)
    df = add_time_features(df)

    # Train/test split
    split_date = date(year, month, split_day)
    df_train, df_test = train_test_split_by_date(df, split_date)
    print(f"[INFO] Train samples: {len(df_train)}, Test samples: {len(df_test)}")

    
    model, feature_cols = train_random_forest_model(df_train)


    evaluate_model(model, feature_cols, df_train, label="Train")
    y_true_test, y_pred_test = evaluate_model(model, feature_cols, df_test, label="Test")

    # Plot one day
    target_day = date(year, month, target_day_for_plot)
    plot_day_prediction(model, feature_cols, df, target_day)


if __name__ == "__main__":
    # Example usage:
    #   python -m ionexer.ml_baseline
    run_ml_for_month(year=2025, month=1, split_day=20, target_day_for_plot=24)