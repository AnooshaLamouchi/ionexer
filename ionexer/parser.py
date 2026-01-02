from datetime import date, datetime, timedelta
from pathlib import Path
import re
import numpy as np
import gzip
import unlzw3
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.widgets import TextBox, Button, Slider
from typing import List, Tuple, Optional

from .file_manager import FileManager
from .anomaly import AnomalyDetector
from .config import DEFAULT_THRESHOLD, DEFAULT_WINDOW_DAYS

IRAN_BBOX = (24.0, 40.0, 44.0, 64.0)

def crop_to_bbox(
    z_map: np.ndarray,
    latitudes: np.ndarray,
    longitudes: np.ndarray,
    bbox: tuple[float, float, float, float],
):
    """
    Crop a global z_map(lat, lon) to a bounding box.

    bbox = (lat_min, lat_max, lon_min, lon_max)

    Returns:
        z_sub, lats_sub, lons_sub
    """
    lat_min, lat_max, lon_min, lon_max = bbox

    # Handle lon range 0–360 vs -180–180
    lons_wrapped = longitudes.copy()
    if np.max(lons_wrapped) > 180.0:  # probably 0–360
        lons_wrapped = (lons_wrapped + 180.0) % 360.0 - 180.0

    lat_mask = (latitudes >= lat_min) & (latitudes <= lat_max)
    lon_mask = (lons_wrapped >= lon_min) & (lons_wrapped <= lon_max)

    # If something went wrong, don't crash – just return original arrays
    if not lat_mask.any() or not lon_mask.any():
        return z_map, latitudes, lons_wrapped

    z_sub = z_map[np.ix_(lat_mask, lon_mask)]
    lats_sub = latitudes[lat_mask]
    lons_sub = lons_wrapped[lon_mask]

    return z_sub, lats_sub, lons_sub


class Parser:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.lines = self._open_ionex_file(self.path)
        self.latitudes: Optional[np.ndarray] = None
        self.longitudes: Optional[np.ndarray] = None
        self.tec_map: Optional[np.ndarray] = None

    @staticmethod
    def _open_ionex_file(p: Path) -> list[str]:
        data = p.read_bytes()

        def _to_lines(b: bytes) -> list[str]:
            try:
                return b.decode("utf-8", errors="ignore").splitlines()
            except Exception:
                return b.decode("latin-1", errors="ignore").splitlines()

        is_gzip = data[:2] == b"\x1f\x8b"
        is_lzw  = data[:2] == b"\x1f\x9d"

        strategies: list[tuple[str, callable]] = []

        if p.suffix.lower() == ".gz":
            strategies.append(("gzip", lambda: gzip.decompress(data)))
        if p.suffix.lower() == ".z" or is_lzw:
            strategies.append(("lzw",  lambda: unlzw3.unlzw(data)))
        if is_gzip:
            strategies.append(("gzip", lambda: gzip.decompress(data)))

        strategies.append(("plain", lambda: data))
        last_err = None
        for name, fn in strategies:
            try:
                raw = fn()
                lines = _to_lines(raw)
                if not any("IONEX VERSION / TYPE" in ln for ln in lines[:80]):
                    pass
                return lines
            except Exception as e:
                last_err = e
        raise RuntimeError(f"Failed to open IONEX file {p} with all strategies. Last error: {last_err}")

    def _parse_header(self):
        float_re = re.compile(r"[-+]?\d*\.\d+|[-+]?\d+")
        lat1 = lat2 = dlat = lon1 = lon2 = dlon = None

        for ln in self.lines:
            if "LAT1 / LAT2 / DLAT" in ln:
                nums = list(map(float, float_re.findall(ln)))
                if len(nums) >= 3:
                    lat1, lat2, dlat = nums[0], nums[1], abs(nums[2])
            elif "LON1 / LON2 / DLON" in ln:
                nums = list(map(float, float_re.findall(ln)))
                if len(nums) >= 3:
                    lon1, lon2, dlon = nums[0], nums[1], abs(nums[2])
            if all(v is not None for v in (lat1, lat2, dlat, lon1, lon2, dlon)):
                break

        if None in (lat1, lat2, dlat, lon1, lon2, dlon):
            sample = "\n".join(self.lines[:40])
            raise ValueError("Header LAT/LON info missing in IONEX file.\n"
                             f"First lines:\n{sample}")

        if lat1 >= lat2:
            self.latitudes = np.round(np.arange(lat1, lat2 - 1e-9, -dlat), 6)
        else:
            self.latitudes = np.round(np.arange(lat1, lat2 + 1e-9,  dlat), 6)
        n_steps = int(round((lon2 - lon1) / dlon))
        self.longitudes = np.round(lon1 + np.arange(n_steps + 1) * dlon, 6)

    def parse(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        self._parse_header()
        float_re = re.compile(r"[-+]?\d*\.\d+|[-+]?\d+")
        n_lon = len(self.longitudes)

        start = end = None
        for i, ln in enumerate(self.lines):
            if "START OF TEC MAP" in ln and start is None:
                start = i
            elif "END OF TEC MAP" in ln and start is not None:
                end = i
                break
        if start is None or end is None:
            raise ValueError("No 'START OF TEC MAP' / 'END OF TEC MAP' block found.")
        tec_fixed = self._parse_block(self.lines[start + 1:end], n_lon)

        med = np.nanmedian(tec_fixed)
        if np.isfinite(med) and med > 50:
            tec_fixed = tec_fixed / 10.0

        self.tec_map = tec_fixed
        return self.tec_map, self.latitudes, self.longitudes

    def parse_all_epochs(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[datetime]]:
        self._parse_header()
        n_lon = len(self.longitudes)
        starts, ends = [], []
        for i, ln in enumerate(self.lines):
            if "START OF TEC MAP" in ln:
                starts.append(i)
            elif "END OF TEC MAP" in ln:
                ends.append(i)
        if not starts or not ends or len(starts) != len(ends):
            raise ValueError("TEC MAP blocks are malformed in the IONEX file.")

        tecs, epochs = [], []
        for s, e in zip(starts, ends):
            block = self.lines[s + 1 : e]
            tec_fixed = self._parse_block(block, n_lon)
            med = np.nanmedian(tec_fixed)
            if np.isfinite(med) and med > 50:
                tec_fixed = tec_fixed / 10.0
            tecs.append(tec_fixed)
            ts = self._extract_epoch_from_block(block)
            epochs.append(ts)
        tec_3d = np.stack(tecs, axis=0)
        return tec_3d, self.latitudes, self.longitudes, epochs

    def _parse_block(self, lines: List[str], n_lon: int) -> np.ndarray:
        float_re = re.compile(r"[-+]?\d*\.\d+|[-+]?\d+")
        parsed_rows: List[List[float]] = []
        parsed_lats: List[float] = []
        current_row: List[float] = []
        current_lat: Optional[float] = None

        for ln in lines:
            if "LAT/LON1/LON2/DLON/H" in ln:
                if current_lat is not None:
                    if len(current_row) > n_lon:
                        current_row = current_row[:n_lon]
                    elif len(current_row) < n_lon:
                        current_row += [np.nan] * (n_lon - len(current_row))
                    parsed_rows.append(current_row)
                nums = re.findall(r"[-+]?\d*\.\d+|[-+]?\d+", ln)
                if len(nums) >= 5:
                    current_lat = float(nums[0])
                    parsed_lats.append(current_lat)
                current_row = []
            else:
                vals = [float(x) for x in re.findall(r"[-+]?\d*\.\d+|[-+]?\d+", ln)]
                if vals:
                    current_row.extend(vals)

        if current_lat is not None:
            if len(current_row) > n_lon:
                current_row = current_row[:n_lon]
            elif len(current_row) < n_lon:
                current_row += [np.nan] * (n_lon - len(current_row))
            parsed_rows.append(current_row)

        tec = np.array(parsed_rows, dtype=float)
        parsed_lats = np.array(parsed_lats, dtype=float)

        tec_fixed = np.full((len(self.latitudes), n_lon), np.nan, float)
        if len(self.latitudes) < 2:
            lat_step = 1.0
        else:
            lat_step = abs(self.latitudes[1] - self.latitudes[0])
        for i, el in enumerate(self.latitudes):
            diffs = np.abs(parsed_lats - el)
            if diffs.size == 0:
                continue
            idxs = np.where(diffs <= (lat_step / 2.0 + 1e-9))[0]
            if idxs.size == 0:
                idx = int(np.argmin(diffs))
                row = tec[idx]
            else:
                row = np.nanmean(tec[idxs], axis=0)
            tec_fixed[i, :] = row
        return tec_fixed

    def _extract_epoch_from_block(self, block: List[str]) -> datetime:
        float_re = re.compile(r"[-+]?\d*\.\d+|[-+]?\d+")
        for ln in block[:8]:
            if "EPOCH OF CURRENT MAP" in ln or "TIME OF MAP" in ln or "EPOCH" in ln:
                nums = [int(float(x)) for x in float_re.findall(ln)[:5]]
                if len(nums) >= 5:
                    y, m, d, H, M = nums[:5]
                    try:
                        return datetime(y, m, d, H, M)
                    except Exception:
                        pass
        try:
            m = re.search(r"(\d{4})(\d{3})", self.path.name)
            if m:
                year = int(m.group(1))
                doy  = int(m.group(2))
                return datetime.strptime(f"{year} {doy}", "%Y %j")
        except Exception:
            pass
        return datetime.utcnow()

    @staticmethod
    def create_interactive_viewer(
        initial_date: date | None = None,
        threshold: float = DEFAULT_THRESHOLD,
        window_days: int = DEFAULT_WINDOW_DAYS,
    ):
        fm = FileManager()
        detector = AnomalyDetector(fm.base_dir, window_days=window_days)

        fig = plt.figure(figsize=(13, 7))
        ax_map = plt.axes(projection=ccrs.PlateCarree())
        ax_map.set_global()
        ax_map.add_feature(cfeature.LAND, facecolor="lightgray", alpha=0.25)
        ax_map.add_feature(cfeature.COASTLINE, linewidth=0.6)
        ax_map.add_feature(cfeature.BORDERS, linewidth=0.4)

        ax_start = plt.axes([0.12, 0.05, 0.22, 0.05])
        ax_end   = plt.axes([0.36, 0.05, 0.22, 0.05])
        ax_load  = plt.axes([0.60, 0.05, 0.08, 0.05])
        ax_prev  = plt.axes([0.70, 0.05, 0.06, 0.05])
        ax_next  = plt.axes([0.77, 0.05, 0.06, 0.05])
        ax_slider = plt.axes([0.85, 0.055, 0.10, 0.035])

        tb_start = TextBox(ax_start, "Start: ", initial=(initial_date or date.today()).isoformat())
        tb_end   = TextBox(ax_end,   "End:   ", initial=(initial_date or date.today()).isoformat())
        btn_load = Button(ax_load, "Load")
        btn_prev = Button(ax_prev, "Prev")
        btn_next = Button(ax_next, "Next")

        s_idx = Slider(ax_slider, "Idx", 0, 0, valinit=0, valstep=1)

        frames: list[dict] = [] 
        current_idx = {"i": 0}
        cbar = {"obj": None}

        def _decorate_world(ax):
            ax.set_global()
            ax.add_feature(cfeature.LAND, facecolor="lightgray", alpha=0.25)
            ax.add_feature(cfeature.COASTLINE, linewidth=0.6)
            ax.add_feature(cfeature.BORDERS, linewidth=0.4)

        def update_map(i: int):
            if not frames:
                return
            i = max(0, min(i, len(frames) - 1))
            current_idx["i"] = i
            fr = frames[i]
            z = fr["z"]
            lats = fr["lats"]
            lons = fr["lons"]
            epoch = fr["epoch"]

            # ---- CROP TO IRAN ----
            z_ir, lats_ir, lons_ir = crop_to_bbox(
                z_map=z,
                latitudes=lats,
                longitudes=lons,
                bbox=IRAN_BBOX,
            )

            # Clear and decorate map
            ax_map.clear()
            _decorate_world(ax_map)

            # Zoom to Iran
            lat_min, lat_max, lon_min, lon_max = IRAN_BBOX
            ax_map.set_extent([lon_min, lon_max, lat_min, lat_max],
                              crs=ccrs.PlateCarree())

            # Meshgrid for cropped region
            lon_grid, lat_grid = np.meshgrid(lons_ir, lats_ir)

            # Plot Iran-only anomalies
            pcm = ax_map.pcolormesh(
                lon_grid,
                lat_grid,
                np.clip(z_ir, -5, 5),
                cmap="RdBu_r",
                vmin=-5,
                vmax=5,
                shading="auto",
                transform=ccrs.PlateCarree(),
            )
            ax_map.contour(
                lon_grid,
                lat_grid,
                (np.abs(z_ir) > threshold).astype(float),
                levels=[0.5],
                colors="black",
                linewidths=0.7,
                transform=ccrs.PlateCarree(),
            )

            # Colorbar (unchanged)
            if cbar["obj"] is None:
                cbar["obj"] = fig.colorbar(
                    pcm,
                    ax=ax_map,
                    orientation="vertical",
                    pad=0.02,
                    label="Robust z (clipped ±5)",
                )
            else:
                cbar["obj"].update_normal(pcm)

            # Title: EXACTLY like worldmap (date+time from epoch)
            ax_map.set_title(
                f"TEC Anomalies (|z| ≥ {threshold:.1f}) — "
                f"{epoch.strftime('%a, %d %b %Y %H:%M UTC')}"
            )
            fig.canvas.draw_idle()


        def build_frames(start_d: date, end_d: date):
            nonlocal s_idx
            frames.clear()
            cur = start_d
            while cur <= end_d:
                z3d, _, latlon, epochs = detector.detect_robust_all(cur, threshold=threshold, method="MAD")
                if z3d is not None and latlon is not None and epochs:
                    lats, lons = latlon
                    for t in range(z3d.shape[0]):
                        frames.append({
                            "z": z3d[t], "lats": lats, "lons": lons, "epoch": epochs[t]
                        })
                cur += timedelta(days=1)

            ax_slider.cla()
            if frames:
                s_idx = Slider(ax_slider, "Idx", 0, len(frames)-1, valinit=0, valstep=1)
                s_idx.on_changed(on_slide)
                update_map(0)
            else:
                s_idx = Slider(ax_slider, "Idx", 0, 0, valinit=0, valstep=1)
                ax_map.set_title("No data found in selected range.")
                fig.canvas.draw_idle()

        def on_load(event):
            try:
                start_d = datetime.strptime(tb_start.text.strip(), "%Y-%m-%d").date()
                end_d   = datetime.strptime(tb_end.text.strip(), "%Y-%m-%d").date()
                if end_d < start_d:
                    start_d, end_d = end_d, start_d
                build_frames(start_d, end_d)
            except Exception as e:
                ax_map.set_title(f"Invalid date(s): {e}")
                fig.canvas.draw_idle()

        def on_prev(event):
            if not frames:
                return
            new_i = max(0, current_idx["i"] - 1)
            s_idx.set_val(new_i)

        def on_next(event):
            if not frames:
                return
            new_i = min(len(frames)-1, current_idx["i"] + 1)
            s_idx.set_val(new_i)

        def on_slide(val):
            update_map(int(val))

        btn_load.on_clicked(on_load)
        btn_prev.on_clicked(on_prev)
        btn_next.on_clicked(on_next)
        s_idx.on_changed(on_slide)

        init_day = initial_date or date.today()
        build_frames(init_day, init_day)
        plt.show()

    @staticmethod
    def plot_anomaly(z_map, latitudes, longitudes, threshold=DEFAULT_THRESHOLD):
        plt.figure(figsize=(12,6))
        plt.title(f"TEC Anomaly Map (|z| ≥ {threshold})")
        lon_grid, lat_grid = np.meshgrid(longitudes, latitudes)
        plt.pcolormesh(lon_grid, lat_grid, np.clip(z_map, -5, 5), cmap="RdBu_r", vmin=-5, vmax=5, shading="auto")
        plt.contour(lon_grid, lat_grid, (np.abs(z_map) > threshold).astype(float), levels=[0.5], colors='black', linewidths=0.7)
        plt.colorbar(label="Robust z (clipped ±5)")
        plt.xlabel("Longitude"); plt.ylabel("Latitude")
        plt.tight_layout(); plt.show()
    
    @staticmethod
    def plot_anomaly_iran(
        z_map: np.ndarray,
        latitudes: np.ndarray,
        longitudes: np.ndarray,
        threshold: float = DEFAULT_THRESHOLD,
    ):
        """
        Same as plot_anomaly, but cropped and zoomed to Iran only.
        """
        # Crop global grid to Iran bbox
        z_ir, lats_ir, lons_ir = crop_to_bbox(
            z_map=z_map,
            latitudes=latitudes,
            longitudes=longitudes,
            bbox=IRAN_BBOX,
        )

        # Prepare lon/lat mesh
        lon_grid, lat_grid = np.meshgrid(lons_ir, lats_ir)

        # Create figure + Cartopy geo-axes
        fig = plt.figure(figsize=(10, 6))
        ax = plt.axes(projection=ccrs.PlateCarree())

        # Zoom to Iran area
        lat_min, lat_max, lon_min, lon_max = IRAN_BBOX
        ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())

        # Add basic map features
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
        ax.add_feature(cfeature.BORDERS, linewidth=0.5)
        ax.add_feature(cfeature.LAND, facecolor="0.9")
        ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)

        # Plot anomaly map (clipped to ±5 for readability)
        pcm = ax.pcolormesh(
            lon_grid,
            lat_grid,
            np.clip(z_ir, -5, 5),
            cmap="RdBu_r",
            vmin=-5,
            vmax=5,
            shading="auto",
            transform=ccrs.PlateCarree(),
        )

        # Black contour where |z| exceeds threshold (anomalies)
        ax.contour(
            lon_grid,
            lat_grid,
            (np.abs(z_ir) > threshold).astype(float),
            levels=[0.5],
            colors="black",
            linewidths=0.7,
            transform=ccrs.PlateCarree(),
        )

        # Colorbar + labels
        cbar = fig.colorbar(pcm, ax=ax, orientation="vertical", pad=0.02)
        cbar.set_label("Robust z (clipped ±5)")

        ax.set_title(f"TEC Anomaly Map over Iran (|z| ≥ {threshold})")
        plt.tight_layout()
        plt.show()