from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Tuple, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from ionexer.file_manager import FileManager
from ionexer.parser import Parser
from ionexer.space_weather import OmniSpaceWeather


# Find nearest grid cell over Iran for a chosen reference point

def find_iran_grid_index(
    lats: np.ndarray,
    lons: np.ndarray,
    target_lat: float = 32.0,
    target_lon: float = 53.0,
) -> Tuple[int, int]:
    """
    Given latitude/longitude arrays from IONEX,
    return the index of the grid cell closest to a selected Iran coordinate.

    This ensures consistency across days when loading TEC maps.
    """
    lons_wrapped = lons.copy()
    if np.max(lons_wrapped) > 180.0:
        # Convert 0–360 → -180–180
        lons_wrapped = (lons_wrapped + 180.0) % 360.0 - 180.0

    lat_idx = int(np.argmin(np.abs(lats - target_lat)))
    lon_idx = int(np.argmin(np.abs(lons_wrapped - target_lon)))
    return lat_idx, lon_idx



# Build a TEC time series for one grid point during a date range

def build_tec_timeseries_for_point(
    start_date: date,
    end_date: date,
    target_lat: float = 32.0,
    target_lon: float = 53.0,
) -> pd.DataFrame:
    """
    For each day in the selected date range:
        - Load the IONEX file
        - Read all epochs (typically every 2 hours)
        - Extract TEC at one gridcell over Iran
        - Append (timestamp, TEC) to a dataframe

    Output dataframe columns:
        datetime : datetime of TEC map
        tec      : TECU value at selected grid cell
    """
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
            # tec_3d shape: (time_steps, n_lat, n_lon)
            tec_3d, lats, lons, epochs = parser.parse_all_epochs()
        except Exception as e:
            print(f"[WARN] Failed to parse {current}: {e}")
            current += timedelta(days=1)
            continue

        # Determine grid index once for consistency across days
        if first_lat_idx is None:
            lat_idx, lon_idx = find_iran_grid_index(lats, lons, target_lat, target_lon)
            first_lat_idx, first_lon_idx = lat_idx, lon_idx
            print(
                f"[INFO] Using gridpoint lat={lats[lat_idx]:.2f}, lon={lons[lon_idx]:.2f}"
            )
        else:
            lat_idx, lon_idx = first_lat_idx, first_lon_idx

        # Extract TEC for each epoch
        for t_idx, ts in enumerate(epochs):
            tec_value = float(tec_3d[t_idx, lat_idx, lon_idx])
            if np.isfinite(tec_value):
                rows.append({"datetime": ts, "tec": tec_value})

        current += timedelta(days=1)

    if not rows:
        raise RuntimeError("No TEC data collected for the selected range.")

    df = pd.DataFrame(rows).sort_values("datetime").reset_index(drop=True)
    return df





def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add temporal features for machine learning:
        - day_of_year   (seasonal pattern)
        - hour          (diurnal pattern)
        - sin/cos hour  (smooth periodic 24h cycle)
        - sin/cos doy   (annual periodic cycle)

    These features allow Random Forest to learn TEC patterns through time.
    """
    dt = pd.to_datetime(df["datetime"])
    df = df.copy()

    df["day_of_year"] = dt.dt.dayofyear
    df["hour"] = dt.dt.hour + dt.dt.minute / 60.0

    df["sin_hour"] = np.sin(2 * np.pi * df["hour"] / 24.0)
    df["cos_hour"] = np.cos(2 * np.pi * df["hour"] / 24.0)

    df["sin_doy"] = np.sin(2 * np.pi * df["day_of_year"] / 365.0)
    df["cos_doy"] = np.cos(2 * np.pi * df["day_of_year"] / 365.0)

    return df



#  Splitting train/test using a calendar date

def train_test_split_by_date(
    df: pd.DataFrame,
    split_date: date,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    A clean and interpretable time-based split:
        - All data before split_date → training set
        - All data on/after split_date → test set
    """
    dt = pd.to_datetime(df["datetime"])
    mask_train = dt.dt.date < split_date
    df_train = df[mask_train].copy()
    df_test = df[~mask_train].copy()
    return df_train, df_test




# Train a Random Forest regression model

def train_random_forest_model(df_train: pd.DataFrame):
    base_features = ["day_of_year", "hour", "sin_hour", "cos_hour", "sin_doy", "cos_doy"]

    # Add any OMNI lag features automatically:
    omni_features = [c for c in df_train.columns if c.endswith("h") and "_lag" in c]

    feature_cols = base_features + omni_features

    X = df_train[feature_cols].values
    y = df_train["tec"].values

    model = RandomForestRegressor(
        n_estimators=400,
        random_state=42,
        n_jobs=-1,
        max_depth=None,
        min_samples_leaf=2,
    )
    model.fit(X, y)
    return model, feature_cols




def evaluate_model(
    model,
    feature_cols: List[str],
    df_subset: pd.DataFrame,
    label: str = "Set",
):
    """
    Compute standard regression metrics:
        - RMSE (root mean square error)
        - MAE  (mean absolute error)
        - R² score

    Returns:
        (true_values, predicted_values)
    """
    if df_subset.empty:
        print(f"[WARN] No data in {label} set.")
        return

    X = df_subset[feature_cols].values
    y_true = df_subset["tec"].values
    y_pred = model.predict(X)

    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    print(f"\n=== {label} evaluation ===")
    print(f"RMSE = {rmse:.3f}")
    print(f"MAE  = {mae:.3f}")
    print(f"R^2  = {r2:.3f}")

    return y_true, y_pred



#  Plot predicted vs. true TEC

def plot_day_prediction(
    model,
    feature_cols: List[str],
    df: pd.DataFrame,
    target_day: date,
):
    """
    Extract all TEC samples belonging to the target date,
    predict them, and plot a comparison curve.

    This visually shows how well the ML model follows the real diurnal TEC behavior.
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

    hours = df_day["hour"].values

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




# monthly ML baseline experiment

def run_ml_for_month(
    year: int = 2025,
    month: int = 4,
    split_day: int = 20,
    target_day_for_plot: int = 24,
):
    """
    A simple monthly ML baseline.
    Steps:
        1. Load TEC for the selected month
        2. Build temporal features
        3. Train on days < split_day
        4. Test on days >= split_day
        5. Plot model performance for a specific target day

    This provides a baseline for comparison against:
        - robust TEC anomaly detection
        - full-year ML models
    """
    start = date(year, month, 1)


    if month == 12:
        end = date(year, 12, 31)
    else:
        end = date(year, month + 1, 1) - timedelta(days=1)

    print(f"[INFO] Building TEC time series for {start} to {end} ...")
    df = build_tec_timeseries_for_point(start, end)
    df = add_time_features(df)

    sw = OmniSpaceWeather()
    df = sw.add_to_tec_df(
        df,
        datetime_col="datetime",
        join_method="ffill",
        lags_hours=(1, 3, 6),
        include_current=False,
        drop_rows_with_any_nan_in_sw=False
    )

    split_date = date(year, month, split_day)
    df_train, df_test = train_test_split_by_date(df, split_date)
    print(f"[INFO] Train samples: {len(df_train)}, Test samples: {len(df_test)}")

    model, feature_cols = train_random_forest_model(df_train)

    evaluate_model(model, feature_cols, df_train, label="Train")
    evaluate_model(model, feature_cols, df_test,  label="Test")

    target_day = date(year, month, target_day_for_plot)
    print(f"[INFO] Plotting prediction for {target_day} ...")
    plot_day_prediction(model, feature_cols, df, target_day)


if __name__ == "__main__":
    print("[DEBUG] Running ml_baseline from file:", __file__)
    run_ml_for_month(year=2025, month=4, split_day=20, target_day_for_plot=24)
