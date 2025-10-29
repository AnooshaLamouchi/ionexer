from datetime import date, datetime
from pathlib import Path
from matplotlib.widgets import TextBox
import re
import numpy as np
import gzip
import unlzw3
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.widgets import TextBox, Button
import matplotlib.animation as animation

from ionexer.file_manager import FileManager


class Parser:

    def __init__(self, file_path: Path):
        self.file_path = Path(file_path)
        self.lines = self._load_file()
        self.latitudes = None
        self.longitudes = None
        self.tec_map = None

    def _load_file(self) -> list[str]:
        p = self.file_path
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
            strategies.append(("lzw",  lambda: unlzw3.unlzw(data)))
        elif p.suffix.lower() == ".z":
            if is_lzw:
                strategies.append(("lzw",  lambda: unlzw3.unlzw(data)))
                strategies.append(("gzip", lambda: gzip.decompress(data)))
            elif is_gzip:
                strategies.append(("gzip", lambda: gzip.decompress(data)))
                strategies.append(("lzw",  lambda: unlzw3.unlzw(data)))
            else:
                strategies.append(("lzw",  lambda: unlzw3.unlzw(data)))
                strategies.append(("gzip", lambda: gzip.decompress(data)))
        else:
            if is_gzip:
                strategies.append(("gzip", lambda: gzip.decompress(data)))
            elif is_lzw:
                strategies.append(("lzw",  lambda: unlzw3.unlzw(data)))

        strategies.append(("plain", lambda: data))

        last_err = None
        for name, fn in strategies:
            try:
                raw = fn()
                lines = _to_lines(raw)
                if not any("IONEX VERSION / TYPE" in ln for ln in lines[:50]):
                    pass
                print(f"Opened as {name}: {p.name}")
                return lines
            except Exception as e:
                last_err = e

        raise RuntimeError(
            f"Failed to open IONEX file {p} with all strategies. Last error: {last_err}"
        )

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
        
    def parse(self):
        self._parse_header()

        float_re = re.compile(r"[-+]?\d*\.\d+|[-+]?\d+")
        n_lon = len(self.longitudes)

        start = end = None
        for i, ln in enumerate(self.lines):
            if "START OF TEC MAP" in ln and start is None:
                start = i
            if start is not None and "END OF TEC MAP" in ln:
                end = i
                break
        if start is None or end is None:
            raise ValueError("No 'START OF TEC MAP' / 'END OF TEC MAP' block found.")

        parsed_rows = []
        parsed_lats = []
        current_row = []
        current_lat = None

        for ln in self.lines[start + 1:end]:
            if "LAT/LON1/LON2/DLON/H" in ln:
                if current_lat is not None:
                    if len(current_row) > n_lon:
                        current_row = current_row[:n_lon]
                    elif len(current_row) < n_lon:
                        current_row += [np.nan] * (n_lon - len(current_row))
                    parsed_rows.append(current_row)
                nums = float_re.findall(ln)
                if len(nums) >= 5:
                    current_lat = float(nums[0])
                    parsed_lats.append(current_lat)
                current_row = []
            else:
                vals = [float(x) for x in float_re.findall(ln)]
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
        for i, el in enumerate(self.latitudes):
            diffs = np.abs(parsed_lats - el)
            if diffs.size == 0:
                continue
            idxs = np.where(diffs <= (abs(self.latitudes[1]-self.latitudes[0]) / 2.0 + 1e-9))[0]
            if idxs.size == 0:
                idx = int(np.argmin(diffs))
                row = tec[idx]
            else:
                row = np.nanmean(tec[idxs], axis=0)
            tec_fixed[i, :] = row

        # Auto-scale if in 0.1 TECU
        med = np.nanmedian(tec_fixed)
        if np.isfinite(med) and med > 50:
            tec_fixed = tec_fixed / 10.0

        self.tec_map = tec_fixed
        print(f"Parsed TEC map: shape {tec_fixed.shape}  (lat x lon)")
        return self.tec_map, self.latitudes, self.longitudes

    @staticmethod
    def plot_heatmap(cmap="plasma"):
        fm = FileManager(auto_download=True)
        current_date = date.today()

        fig = plt.figure(figsize=(12, 6))
        ax = plt.axes(projection=ccrs.PlateCarree())
        ax.set_global()
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
        ax.add_feature(cfeature.BORDERS, linewidth=0.3)
        ax.add_feature(cfeature.LAND, facecolor="lightgray", alpha=0.3)

        pcm = None
        cbar = None

        def update_plot(new_date: date):
            nonlocal pcm, cbar
            fpath = fm.get_file_for_date(new_date)
            if not fpath:
                print("File not found for", new_date)
                return

            parser = Parser(fpath)
            tec, lats, lons = parser.parse()
            ax.clear()
            ax.set_global()
            ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
            ax.add_feature(cfeature.BORDERS, linewidth=0.3)
            ax.add_feature(cfeature.LAND, facecolor="lightgray", alpha=0.3)

            lon_grid, lat_grid = np.meshgrid(lons, lats)
            pcm = ax.pcolormesh(
                lon_grid, lat_grid, tec, cmap=cmap, transform=ccrs.PlateCarree()
            )

            if cbar:
                cbar.update_normal(pcm)
            else:
                cbar = fig.colorbar(pcm, ax=ax, label="TEC (TECU)", pad=0.02)

            ax.set_title(f"Global IONEX TEC — {new_date.isoformat()}")

            fig.canvas.draw_idle()

        box_y = 0.05
        text_box_date = TextBox(
            plt.axes([0.25, box_y, 0.2, 0.05]),
            "Date (YYYY-MM-DD): ",
            initial=current_date.isoformat(),
        )
        text_box_start = TextBox(
            plt.axes([0.5, box_y, 0.15, 0.05]), "Start: ", initial=current_date.isoformat()
        )
        text_box_end = TextBox(
            plt.axes([0.68, box_y, 0.15, 0.05]), "End: ", initial=current_date.isoformat()
        )

        ax_button = plt.axes([0.85, box_y, 0.08, 0.05])
        play_button = Button(ax_button, "▶ Play")

        playing = {"state": False}
        anim = {"obj": None}

        def submit_date(text):
            try:
                new_date = datetime.strptime(text.strip(), "%Y-%m-%d").date()
                update_plot(new_date)
            except Exception as e:
                print("Invalid date:", e)

        text_box_date.on_submit(submit_date)

        def toggle_play(event):
            if not playing["state"]:
                try:
                    start = datetime.strptime(text_box_start.text.strip(), "%Y-%m-%d").date()
                    end = datetime.strptime(text_box_end.text.strip(), "%Y-%m-%d").date()
                except Exception as e:
                    print("Invalid start/end date:", e)
                    return

                files = fm.get_files_in_range(start, end)
                if not files:
                    print("No files in range.")
                    return

                play_button.label.set_text("⏸ Pause")
                playing["state"] = True

                def frame_gen():
                    for p in files:
                        yield p

                def animate(file_path):
                    nonlocal cbar
                    parser = Parser(file_path)
                    tec, lats, lons = parser.parse()
                    ax.clear()
                    ax.set_global()
                    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
                    ax.add_feature(cfeature.BORDERS, linewidth=0.3)
                    ax.add_feature(cfeature.LAND, facecolor="lightgray", alpha=0.3)
                    lon_grid, lat_grid = np.meshgrid(lons, lats)
                    pcm = ax.pcolormesh(
                        lon_grid, lat_grid, tec, cmap=cmap, transform=ccrs.PlateCarree()
                    )
                    if cbar:
                        cbar.update_normal(pcm)
                    else:
                        cbar = fig.colorbar(pcm, ax=ax, label="TEC (TECU)", pad=0.02)
                    ax.set_title(f"Global IONEX TEC — {file_path.stem}")
                    return [pcm]

                anim["obj"] = animation.FuncAnimation(
                    fig, animate, frames=frame_gen, interval=1500, repeat=False
                )

            else:
                if anim["obj"]:
                    anim["obj"].event_source.stop()
                play_button.label.set_text("▶ Play")
                playing["state"] = False

        play_button.on_clicked(toggle_play)

        update_plot(current_date)
        plt.show()