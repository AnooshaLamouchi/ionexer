from pathlib import Path
from datetime import date, timedelta
import numpy as np
import logging
from typing import Tuple, Optional, List

from .file_manager import FileManager
from .config import DEFAULT_WINDOW_DAYS, DEFAULT_THRESHOLD, EPS


def _safe_scale(arr: np.ndarray, eps: float = EPS) -> np.ndarray:
    s = np.where(np.isfinite(arr), arr, np.nan)
    med_abs = np.nanmedian(np.abs(s))
    tiny = max(eps, 1e-6 * (med_abs if np.isfinite(med_abs) else 1.0))
    out = np.where((~np.isfinite(arr)) | (arr <= 0), tiny, arr)
    out = np.where(arr == 0, tiny, out)
    return out


def _load_first_epoch(path: Path):
    from .parser import Parser
    return Parser(path).parse()


def _load_all_epochs(path: Path):
    from .parser import Parser
    return Parser(path).parse_all_epochs()


class AnomalyDetector:
    def __init__(self, base_dir: Path, window_days: int = DEFAULT_WINDOW_DAYS):
        self.fm = FileManager(base_dir)
        self.window_days = int(window_days)

    def detect_zscore(
        self,
        target_date: date,
        threshold: float = 3.0
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        start = target_date - timedelta(days=self.window_days)
        end = target_date - timedelta(days=1)

        stack = []
        current = start
        latlons = None
        while current <= end:
            fpath = self.fm.get_file_for_date(current)
            if fpath:
                tec, lat, lon = _load_first_epoch(fpath)
                if latlons is None:
                    latlons = (lat, lon)
                elif lat.shape != latlons[0].shape or lon.shape != latlons[1].shape:
                    logging.warning(f"Skipping {current} due to grid mismatch.")
                    current += timedelta(days=1)
                    continue
                stack.append(tec)
            current += timedelta(days=1)

        if not stack:
            logging.warning("No historical TEC maps for z-score baseline.")
            return None, None

        hist_stack = np.stack(stack)
        mean_map = np.nanmean(hist_stack, axis=0)
        std_map = np.nanstd(hist_stack, axis=0)
        std_map = _safe_scale(std_map)

        today_path = self.fm.get_file_for_date(target_date)
        if not today_path:
            logging.warning(f"No TEC file for {target_date}")
            return None, None

        today_tec, *_ = _load_first_epoch(today_path)
        z_map = (today_tec - mean_map) / std_map
        anomalies = np.abs(z_map) > threshold
        return z_map, anomalies

    def detect_robust(
        self,
        target_date: date,
        threshold: float = DEFAULT_THRESHOLD,
        method: str = "MAD"
    ):
        today_path = self.fm.get_file_for_date(target_date)
        if not today_path:
            logging.warning(f"No TEC file for {target_date}")
            return None, None, None

        today_tec_3d, lats, lons, epochs = _load_all_epochs(today_path)

        hist = []
        for d in range(1, self.window_days + 1):
            fpath = self.fm.get_file_for_date(target_date - timedelta(days=d))
            if not fpath:
                continue
            try:
                tec_3d, *_ = _load_all_epochs(fpath)
            except Exception as e:
                logging.warning(f"Failed parsing {fpath.name}: {e}")
                continue
            if tec_3d.shape != today_tec_3d.shape:
                logging.warning(f"Grid mismatch on {fpath.name}, skipping.")
                continue
            hist.append(tec_3d)

        if len(hist) < max(7, self.window_days // 3):
            logging.warning("Not enough history for robust baseline.")
            return None, None, None

        H = np.stack(hist, axis=0)
        median = np.nanmedian(H, axis=0)
        if method.upper() == "MAD":
            mad = np.nanmedian(np.abs(H - median[None, ...]), axis=0) * 1.4826
            scale = _safe_scale(mad)
        else:
            q25 = np.nanpercentile(H, 25, axis=0)
            q75 = np.nanpercentile(H, 75, axis=0)
            iqr = (q75 - q25) * 0.7413
            scale = _safe_scale(iqr)

        z = (today_tec_3d - median) / scale
        mask = np.abs(z) > threshold

        return z[0], mask[0], (lats, lons)

    def detect_robust_all(
        self,
        target_date: date,
        threshold: float = DEFAULT_THRESHOLD,
        method: str = "MAD",
    ) -> Tuple[
        Optional[np.ndarray],
        Optional[np.ndarray],
        Optional[Tuple[np.ndarray, np.ndarray]],
        Optional[List]
    ]:
        today_path = self.fm.get_file_for_date(target_date)
        if not today_path:
            logging.warning(f"No TEC file for {target_date}")
            return None, None, None, None

        today_tec_3d, lats, lons, epochs = _load_all_epochs(today_path)

        hist = []
        for d in range(1, self.window_days + 1):
            fpath = self.fm.get_file_for_date(target_date - timedelta(days=d))
            if not fpath:
                continue
            try:
                tec_3d, *_ = _load_all_epochs(fpath)
            except Exception as e:
                logging.warning(f"Failed parsing {fpath.name}: {e}")
                continue
            if tec_3d.shape != today_tec_3d.shape:
                logging.warning(f"Grid mismatch on {fpath.name}, skipping.")
                continue
            hist.append(tec_3d)

        if len(hist) < max(7, self.window_days // 3):
            logging.warning("Not enough history for robust baseline.")
            return None, None, None, None

        H = np.stack(hist, axis=0)
        median = np.nanmedian(H, axis=0)
        if method.upper() == "MAD":
            mad = np.nanmedian(np.abs(H - median[None, ...]), axis=0) * 1.4826
            scale = _safe_scale(mad)
        else:
            q25 = np.nanpercentile(H, 25, axis=0)
            q75 = np.nanpercentile(H, 75, axis=0)
            iqr = (q75 - q25) * 0.7413
            scale = _safe_scale(iqr)

        z = (today_tec_3d - median) / scale
        mask = np.abs(z) > threshold
        return z, mask, (lats, lons), epochs