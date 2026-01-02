# ml_iran_map_yearly.py
from __future__ import annotations
from datetime import date, timedelta
from typing import Optional, Tuple, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from ionexer.file_manager import FileManager
from ionexer.parser import Parser
from ionexer.space_weather import OmniSpaceWeather

IRAN_BBOX = (24.0, 40.0, 44.0, 64.0)


def _wrap_lons(lons: np.ndarray) -> np.ndarray:
    l = lons.copy()
    if np.max(l) > 180.0:
        l = (l + 180.0) % 360.0 - 180.0
    return l


def _crop_indices(lats: np.ndarray, lons_wrapped: np.ndarray, bbox=IRAN_BBOX):
    lat_min, lat_max, lon_min, lon_max = bbox
    lat_idx = np.where((lats >= lat_min) & (lats <= lat_max))[0]
    lon_idx = np.where((lons_wrapped >= lon_min) & (lons_wrapped <= lon_max))[0]
    if lat_idx.size == 0 or lon_idx.size == 0:
        lat_idx = np.arange(len(lats))
        lon_idx = np.arange(len(lons_wrapped))
    return lat_idx, lon_idx


def _time_features(ts: pd.Timestamp):
    doy = ts.dayofyear
    hour = ts.hour + ts.minute / 60.0
    sin_hour = np.sin(2 * np.pi * hour / 24.0)
    cos_hour = np.cos(2 * np.pi * hour / 24.0)
    sin_doy = np.sin(2 * np.pi * doy / 365.0)
    cos_doy = np.cos(2 * np.pi * doy / 365.0)
    return doy, hour, sin_hour, cos_hour, sin_doy, cos_doy


def build_year_dataset_iran(
        year: int,
        end: Optional[date] = None, sample_frac: float = 0.25,
        add_lag: bool = True,
        include_space_weather: bool = True,
        sw_lags_hours: Tuple[int, ...] = (1, 3, 6),
        sw_include_current: bool = False,
) -> pd.DataFrame:
    fm = FileManager()
    rows = []

    sw = OmniSpaceWeather() if include_space_weather else None

    start = date(year, 1, 1)
    end = end or date(year, 12, 31)

    current = start
    while current <= end:
        path = fm.get_file_for_date(current)
        if path is None:
            current += timedelta(days=1)
            continue

        try:
            tec_3d, lats, lons, epochs = Parser(path).parse_all_epochs()
        except Exception:
            current += timedelta(days=1)
            continue

        lons_w = _wrap_lons(lons)
        lat_idx, lon_idx = _crop_indices(lats, lons_w)

        tec_ir = tec_3d[:, lat_idx][:, :, lon_idx]  # (T, Lat, Lon)
        lats_ir = lats[lat_idx]
        lons_ir = lons_w[lon_idx]

        # آماده‌سازی شبکه lat/lon برای تبدیل سریع به رکورد
        lon_grid, lat_grid = np.meshgrid(lons_ir, lats_ir)  # (Lat, Lon)

        prev = None
        for t_i, ts in enumerate(epochs):
            ts_pd = pd.Timestamp(ts)
            doy, hour, sin_h, cos_h, sin_d, cos_d = _time_features(ts_pd)

            Z = tec_ir[t_i]  # (Lat, Lon)
            if add_lag:
                tec_prev = prev
                prev = Z.copy()
            else:
                tec_prev = None

            # Flatten
            y = Z.reshape(-1)
            lat_flat = lat_grid.reshape(-1)
            lon_flat = lon_grid.reshape(-1)

            mask = np.isfinite(y)
            if add_lag and tec_prev is not None:
                prev_flat = tec_prev.reshape(-1)
                mask = mask & np.isfinite(prev_flat)
            else:
                prev_flat = None

            idx = np.where(mask)[0]
            if idx.size == 0:
                continue

            # subsample for speed
            if 0 < sample_frac < 1.0:
                k = max(1, int(idx.size * sample_frac))
                idx = np.random.choice(idx, size=k, replace=False)

            batch = {
                "datetime": np.repeat(ts_pd, len(idx)),
                "day_of_year": np.repeat(doy, len(idx)),
                "hour": np.repeat(hour, len(idx)),
                "sin_hour": np.repeat(sin_h, len(idx)),
                "cos_hour": np.repeat(cos_h, len(idx)),
                "sin_doy": np.repeat(sin_d, len(idx)),
                "cos_doy": np.repeat(cos_d, len(idx)),
                "lat": lat_flat[idx],
                "lon": lon_flat[idx],
                "tec": y[idx],
            }
            if add_lag and prev_flat is not None:
                batch["tec_prev"] = prev_flat[idx]

            rows.append(pd.DataFrame(batch))

        current += timedelta(days=1)

    if not rows:
        raise RuntimeError("No rows collected for dataset.")
    df = pd.concat(rows, ignore_index=True)
    df = df.sort_values("datetime").reset_index(drop=True)
    # df = pd.DataFrame(rows)

    if include_space_weather and sw is not None and len(df) > 0:
        df = sw.add_to_tec_df(
            df,
            datetime_col="datetime",
            join_method="ffill",
            lags_hours=sw_lags_hours,
            include_current=sw_include_current,
            drop_rows_with_any_nan_in_sw=False,
        )

    return df


def split_timewise(df: pd.DataFrame, split_date: date):
    dt = pd.to_datetime(df["datetime"])
    train = df[dt.dt.date < split_date].copy()
    test = df[dt.dt.date >= split_date].copy()
    return train, test


def train_rf(train_df: pd.DataFrame, use_lag: bool = True):
    feature_cols = ["day_of_year","hour","sin_hour","cos_hour","sin_doy","cos_doy","lat","lon"]
    if use_lag and "tec_prev" in train_df.columns:
        feature_cols.append("tec_prev")

    feature_cols += [c for c in train_df.columns if "_lag" in c and c.endswith("h")]

    X = train_df[feature_cols].values
    y = train_df["tec"].values

    model = RandomForestRegressor(
        n_estimators=400,
        random_state=42,
        n_jobs=-1,
        max_depth=None,
        min_samples_leaf=2,  # کمی smoother → بهتر برای نقشه
    )
    model.fit(X, y)
    return model, feature_cols


def eval_model(model, feature_cols, df, label="Test"):
    X = df[feature_cols].values
    y = df["tec"].values
    pred = model.predict(X)
    rmse = np.sqrt(mean_squared_error(y, pred))
    mae = mean_absolute_error(y, pred)
    r2 = r2_score(y, pred)
    print(f"\n=== {label} ===")
    print(f"RMSE = {rmse:.3f}")
    print(f"MAE  = {mae:.3f}")
    print(f"R²   = {r2:.3f}")
    return pred


def predict_map_for_time(
        model,
        feature_cols,
        lats_ir: np.ndarray,
        lons_ir: np.ndarray,
        ts: pd.Timestamp,
        tec_prev_map: Optional[np.ndarray] = None,
):
    lon_grid, lat_grid = np.meshgrid(lons_ir, lats_ir)
    doy, hour, sin_h, cos_h, sin_d, cos_d = _time_features(ts)

    N = lat_grid.size
    data = {
        "day_of_year": np.full(N, doy),
        "hour": np.full(N, hour),
        "sin_hour": np.full(N, sin_h),
        "cos_hour": np.full(N, cos_h),
        "sin_doy": np.full(N, sin_d),
        "cos_doy": np.full(N, cos_d),
        "lat": lat_grid.reshape(-1),
        "lon": lon_grid.reshape(-1),
    }
    if "tec_prev" in feature_cols:
        if tec_prev_map is None:
            raise ValueError("tec_prev_map is required because model uses tec_prev.")
        data["tec_prev"] = tec_prev_map.reshape(-1)

    sw_lag_cols = [c for c in feature_cols if "_lag" in c and c.endswith("h")]
    if sw_lag_cols:
        sw = OmniSpaceWeather()

        ts_utc = pd.to_datetime(ts, utc=True)
        # small window is enough; just needs to cover plot time
        start = (ts_utc - pd.Timedelta(hours=24)).date()
        end   = ts_utc.date()

        sw_hourly = sw.load_hourly(start, end)
        sw_hourly = sw_hourly.copy()
        sw_hourly["Epoch"] = pd.to_datetime(sw_hourly["Epoch"], utc=True)
        sw_hourly = (
            sw_hourly.sort_values("Epoch")
            .drop_duplicates(subset="Epoch", keep="last")
            .set_index("Epoch")
        )

        # get OMNI values at the plot time (causal fill)
        row = sw_hourly.reindex([ts_utc], method="ffill").iloc[0]

        # IMPORTANT:
        # Your current training pipeline's "lag" construction is row-shifted on a grid dataset,
        # so these lag columns behave almost like "current" values for most rows.
        # To stay consistent (and avoid retraining right now), fill lag cols using the current value.
        for col in sw_lag_cols:
            base = col.split("_lag")[0]  # e.g. "kp" from "kp_lag1h"
            v = row.get(base, np.nan)
            if pd.isna(v):
                v = 0.0
            data[col] = np.full(N, float(v))

    X = pd.DataFrame(data)[feature_cols].values
    pred = model.predict(X).reshape(lat_grid.shape)
    return pred


def run_year_model_and_plot_one_time(
        year: int = 2025,
        train_until: date = date(2025, 11, 1),  # train < Nov 1, test >= Nov 1
        plot_day: date = date(2025, 11, 27),
        plot_hour_utc: int = 2,
):
    print("[INFO] Building yearly Iran dataset (this can take time)...")
    df = build_year_dataset_iran(year=year, end=date(2025, 12, 2), sample_frac=0.25, add_lag=True)

    train_df, test_df = split_timewise(df, split_date=train_until)
    print(f"[INFO] Train rows: {len(train_df)}, Test rows: {len(test_df)}")

    model, feature_cols = train_rf(train_df, use_lag=True)
    eval_model(model, feature_cols, train_df, "Train")
    eval_model(model, feature_cols, test_df, "Test")

    # For plotting maps you also need lats/lons Iran grid for that day:
    fm = FileManager()
    p = fm.get_file_for_date(plot_day)
    tec_3d, lats, lons, epochs = Parser(p).parse_all_epochs()
    lons_w = _wrap_lons(lons)
    lat_idx, lon_idx = _crop_indices(lats, lons_w)

    lats_ir = lats[lat_idx]
    lons_ir = lons_w[lon_idx]
    tec_ir = tec_3d[:, lat_idx][:, :, lon_idx]

    # find epoch closest to plot_hour_utc
    target_ts = pd.Timestamp(datetime(plot_day.year, plot_day.month, plot_day.day, plot_hour_utc, 0))
    epoch_list = [pd.Timestamp(e) for e in epochs]
    t_i = int(np.argmin([abs((e - target_ts).total_seconds()) for e in epoch_list]))

    true_map = tec_ir[t_i]
    prev_map = tec_ir[t_i - 1] if t_i > 0 else tec_ir[t_i]  # simple fallback
    pred_map = predict_map_for_time(model, feature_cols, lats_ir, lons_ir, epoch_list[t_i], tec_prev_map=prev_map)
    err_map = true_map - pred_map

    # plot
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    im0 = axes[0].imshow(true_map, origin="lower", aspect="auto")
    axes[0].set_title(f"True TEC\n{epoch_list[t_i]} UTC")
    plt.colorbar(im0, ax=axes[0])

    im1 = axes[1].imshow(pred_map, origin="lower", aspect="auto")
    axes[1].set_title("Predicted TEC (RF yearly)")
    plt.colorbar(im1, ax=axes[1])

    im2 = axes[2].imshow(err_map, origin="lower", aspect="auto")
    axes[2].set_title("Error (True - Pred)")
    plt.colorbar(im2, ax=axes[2])

    for ax in axes:
        ax.set_xlabel("Lon index")
        ax.set_ylabel("Lat index")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    from datetime import datetime

    run_year_model_and_plot_one_time()
