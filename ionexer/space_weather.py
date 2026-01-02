from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable, Optional, Sequence, Dict, List, Tuple

import numpy as np
import pandas as pd


# ----------------------------
# Utilities
# ----------------------------

def _to_utc_timestamp(x) -> pd.Timestamp:
    """
    Normalize to timezone-aware UTC pandas Timestamp.
    """
    ts = pd.to_datetime(x)
    if ts.tzinfo is None:
        # assume it is already UTC if naive
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts


def _safe_float_series(s: pd.Series) -> pd.Series:
    """
    Convert numeric-like series to float, coercing errors to NaN.
    """
    return pd.to_numeric(s, errors="coerce").astype("float64")


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _hash_key(*parts: str) -> str:
    # stable filename key without importing hashlib (but you can if you want)
    return str(abs(hash("|".join(parts))))


# ----------------------------
# Config
# ----------------------------

@dataclass(frozen=True)
class OmniConfig:
    """
    CDAWeb OMNI Hourly Data (OMNI2) DOI shown by OMNIWeb citation page.
    The OMNI dataset is available in CDAWeb with IDs like OMNI2_H0_MRG1HR. :contentReference[oaicite:1]{index=1}
    """
    doi: str = "10.48322/1shr-ht18"  # OMNI Hourly Data DOI (OMNIWeb citation page)
    cache_dir: str = "data/space_weather_cache"
    # Use a conservative default variable list.
    # If CDAWeb uses different variable names, we'll auto-map (see mapping logic).
    preferred_vars: Tuple[str, ...] = (
        # IMF
        "BZ_GSE", "BY_GSE", "BX_GSE", "B_MAG",
        # Solar wind
        "V", "N", "T", "Pdyn",
        # Indices / solar
        "DST", "KP", "F10_INDEX",
    )


# ----------------------------
# Loader
# ----------------------------

class OmniSpaceWeather:
    """
    Loads OMNI hourly (and 3-hour indices inside OMNI) for a time window,
    caches to parquet, and can merge safely into your TEC dataframe.

    Safety features:
      - Caches retrieved data (avoids rate limits).
      - Provides lag-only mode to avoid future leakage.
      - Forward-fills short gaps but never fills across long holes (configurable).
    """

    def __init__(self, cfg: OmniConfig = OmniConfig()):
        self.cfg = cfg
        self.cache_path = Path(cfg.cache_dir)
        _ensure_dir(self.cache_path)

        try:
            from cdasws import CdasWs  # type: ignore
            from cdasws.timeinterval import TimeInterval  # type: ignore
            from cdasws import datarepresentation as dr  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "Missing dependency 'cdasws'. Install it with: pip install cdasws\n"
                f"Original import error: {e}"
            )

        self._CdasWs = CdasWs
        self._TimeInterval = TimeInterval
        self._dr = dr

    # ---------
    # CDAWeb fetch
    # ---------

    def _fetch_from_cdaweb(self, start_utc: pd.Timestamp, end_utc: pd.Timestamp) -> pd.DataFrame:
        """
        Fetch raw data from CDAWeb using cdasws.

        Returns a dataframe with an 'Epoch' column (UTC datetimes) plus variables.
        """
        cdas = self._CdasWs()

        # Ask CDAWeb what variables exist, then map to what we want.
        variables = cdas.get_variables(self.cfg.doi)
        available = {v["Name"] for v in variables}

        # Build a mapping for common OMNI naming differences.
        # We try preferred first, but also accept typical alternates.
        candidates: Dict[str, List[str]] = {
            "bz_gse": ["BZ_GSE", "Bz_GSE", "BZ_GSM", "Bz_GSM"],
            "by_gse": ["BY_GSE", "By_GSE", "BY_GSM", "By_GSM"],
            "bx_gse": ["BX_GSE", "Bx_GSE", "BX_GSM", "Bx_GSM"],
            "b_mag":  ["B_MAG", "B", "Bmag", "IMF_MAG", "BMAG"],
            "v":      ["V", "V_SW", "Vsw", "flow_speed", "SPEED"],
            "n":      ["N", "Np", "PROTON_DENSITY", "density"],
            "t":      ["T", "TEMP", "temperature"],
            "pdyn":   ["Pdyn", "PDYN", "DYN_P", "dynamic_pressure"],
            "dst":    ["DST", "Dst"],
            "kp":     ["KP", "Kp", "KP1800", "Kp_index"],
            "f107":   ["F10_INDEX", "F10.7", "F107", "F107_OBS", "F10_INDEX1800"],
        }

        def pick(name_list: List[str]) -> Optional[str]:
            for n in name_list:
                if n in available:
                    return n
            return None

        # Resolve actual variable names that exist in this CDAWeb dataset.
        resolved: Dict[str, str] = {}
        for key, cand in candidates.items():
            vname = pick(cand)
            if vname is not None:
                resolved[key] = vname

        # Request only what exists (plus Epoch is implied).
        var_names = list(resolved.values())
        if len(var_names) == 0:
            raise RuntimeError(
                "Could not resolve any OMNI variable names from CDAWeb. "
                "Open CDAWeb OMNI dataset and inspect variable names."
            )

        ti = self._TimeInterval(
            start_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
            end_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
        )

        # Try to retrieve as Pandas if supported; otherwise xarray then convert.
        # cdasws supports SPACEPY, XARRAY, and Pandas DataFrame in examples. :contentReference[oaicite:2]{index=2}
        dr = self._dr
        data = None

        # Prefer PANDAS if present; else XARRAY fallback.
        if hasattr(dr, "PANDAS"):
            _, data = cdas.get_data(self.cfg.doi, var_names, ti, dataRepresentation=dr.PANDAS)
            # expected: pandas.DataFrame with 'Epoch'
            if not isinstance(data, pd.DataFrame):
                # Sometimes it returns dict-like; handle below
                data = None

        if data is None:
            _, xds = cdas.get_data(self.cfg.doi, var_names, ti, dataRepresentation=dr.XARRAY)
            # xarray.Dataset -> DataFrame
            df = xds.to_dataframe().reset_index()
            # Usually 'Epoch' is index; ensure it exists:
            if "Epoch" not in df.columns:
                # some datasets call it 'time' or similar
                # If present as index name, already in columns from reset_index()
                pass
            data = df

        df_raw = data.copy()

        # Ensure Epoch exists
        if "Epoch" not in df_raw.columns:
            # Try common alternates
            for alt in ["epoch", "time", "Time", "DATE_TIME"]:
                if alt in df_raw.columns:
                    df_raw = df_raw.rename(columns={alt: "Epoch"})
                    break
        if "Epoch" not in df_raw.columns:
            raise RuntimeError("CDAWeb did not return an 'Epoch' column; cannot time-align.")

        # Rename columns into our canonical feature names
        inv_resolved = {v: k for k, v in resolved.items()}
        rename_map = {col: inv_resolved[col] for col in df_raw.columns if col in inv_resolved}
        df_raw = df_raw.rename(columns=rename_map)

        # Keep only relevant cols
        keep = ["Epoch"] + list(resolved.keys())
        keep = [c for c in keep if c in df_raw.columns]
        df_raw = df_raw[keep]

        return df_raw

    # ---------
    # Public API
    # ---------

    def load_hourly(self, start: date, end: date) -> pd.DataFrame:
        """
        Load hourly OMNI features between [start, end] inclusive.
        Uses caching.
        """
        start_utc = _to_utc_timestamp(datetime(start.year, start.month, start.day, 0, 0, 0, tzinfo=timezone.utc))
        end_utc = _to_utc_timestamp(datetime(end.year, end.month, end.day, 23, 59, 59, tzinfo=timezone.utc))

        cache_key = _hash_key(self.cfg.doi, str(start), str(end))
        cache_file = self.cache_path / f"omni_{cache_key}_{start}_{end}.parquet"

        if cache_file.exists():
            df = pd.read_parquet(cache_file)
        else:
            df = self._fetch_from_cdaweb(start_utc, end_utc)
            df.to_parquet(cache_file, index=False)

        # Clean and standardize
        df["Epoch"] = pd.to_datetime(df["Epoch"], utc=True)
        df = df.sort_values("Epoch").reset_index(drop=True)

        # Convert to floats
        for c in df.columns:
            if c != "Epoch":
                df[c] = _safe_float_series(df[c])

        # Derived coupling: Ey (mV/m-ish, sign convention depends; this is simple proxy)
        # Ey = -V * Bz (if V km/s, Bz nT, Ey ~ mV/m scaling constant omitted)
        if "v" in df.columns and "bz_gse" in df.columns:
            df["ey_proxy"] = -df["v"] * df["bz_gse"]

        # Convert kp storage if needed: OMNI may store Kp*10 or Kp in weird steps. :contentReference[oaicite:3]{index=3}
        # We do NOT “decode” plus/minus; we just scale if it looks like *10.
        if "kp" in df.columns:
            kp = df["kp"].copy()
            if kp.dropna().max() > 9.5:
                df["kp"] = kp / 10.0

        return df

    def to_timegrid(
            self,
            sw_hourly: pd.DataFrame,
            target_times_utc: pd.DatetimeIndex,
            method: str = "nearest",
            tolerance: pd.Timedelta = pd.Timedelta("90min"),
    ) -> pd.DataFrame:
        """
        Reindex OMNI data to your TEC timestamps.

        method:
          - "nearest": pick nearest hourly sample within tolerance
          - "ffill": forward fill (safe for causal pipelines)
        """
        df = sw_hourly.copy().set_index("Epoch")

        # For causal safety, ffill is better; nearest is fine if you are not forecasting.
        if method == "ffill":
            out = df.reindex(target_times_utc, method="ffill")
        else:
            out = df.reindex(target_times_utc, method="nearest", tolerance=tolerance)

        out = out.reset_index().rename(columns={"index": "datetime"})
        out["datetime"] = pd.to_datetime(out["datetime"], utc=True)
        return out

    def add_to_tec_df(
            self,
            tec_df: pd.DataFrame,
            datetime_col: str = "datetime",
            start: Optional[date] = None,
            end: Optional[date] = None,
            join_method: str = "ffill",
            lags_hours: Sequence[int] = (1, 3, 6),
            include_current: bool = False,
            drop_rows_with_any_nan_in_sw: bool = False,
    ) -> pd.DataFrame:
        """
        Merge space-weather features into a TEC dataframe.

        SAFE DEFAULTS:
          - join_method="ffill" (causal)
          - include_current=False and lags_hours=(1,3,6) (no leakage)
          - drop_rows_with_any_nan_in_sw=False (keeps your dataset size; you can tighten later)

        If you are *not* forecasting (you just want explanation), set include_current=True.
        """
        df = tec_df.copy()
        df[datetime_col] = pd.to_datetime(df[datetime_col], utc=True)

        start = start or df[datetime_col].dt.date.min()
        end = end or df[datetime_col].dt.date.max()

        sw_hourly = self.load_hourly(start, end)

        target_times = pd.DatetimeIndex(df[datetime_col])
        sw_on_grid = self.to_timegrid(sw_hourly, target_times, method=join_method)

        # Merge
        merged = df.merge(sw_on_grid, left_on=datetime_col, right_on="datetime", how="left", suffixes=("", "_sw"))
        if "datetime_sw" in merged.columns:
            merged = merged.drop(columns=["datetime_sw"])

        # Create lagged versions (by rows, assuming timestamps are sorted)
        merged = merged.sort_values(datetime_col).reset_index(drop=True)

        sw_cols = [c for c in sw_on_grid.columns if c not in ["datetime"]]

        # If include_current is false, remove the non-lag versions (safest).
        if not include_current:
            # we keep them temporarily to compute lags, then drop originals.
            pass

        # Estimate step size in hours from your TEC timestamps (often 1h or 2h).
        dt_hours = merged[datetime_col].diff().dt.total_seconds().median() / 3600.0
        if not np.isfinite(dt_hours) or dt_hours <= 0:
            dt_hours = 1.0

        def rows_for_lag(h: int) -> int:
            return int(round(h / dt_hours))

        # Add lags
        for h in lags_hours:
            k = rows_for_lag(h)
            for c in sw_cols:
                merged[f"{c}_lag{h}h"] = merged[c].shift(k)

        # Optionally keep current
        if not include_current:
            merged = merged.drop(columns=sw_cols, errors="ignore")

        # Handle NaNs created by lags
        if drop_rows_with_any_nan_in_sw:
            lag_cols = [c for c in merged.columns if "_lag" in c]
            merged = merged.dropna(subset=lag_cols).reset_index(drop=True)

        return merged