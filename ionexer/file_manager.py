from pathlib import Path
from datetime import date, timedelta
import logging
from typing import Optional, List

from .config import DOWNLOAD_DIR
from .downloader import Downloader


class FileManager:

    def __init__(self, base_dir: Path = DOWNLOAD_DIR, auto_download: bool = True):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.downloader = Downloader(download_dir=self.base_dir)
        self.auto_download = auto_download

    @staticmethod
    def _date_to_doy(dt: date) -> str:
        return f"{dt.timetuple().tm_yday:03d}"

    def _expected_filenames(self, dt: date) -> List[str]:
        year = dt.year
        doy = self._date_to_doy(dt)
        yy = str(year)[2:]

        return [
            f"c1pg{doy}0.{yy}i.Z",
            f"c2pg{doy}0.{yy}i.Z",
            f"IGS0OPSFIN_{year}{doy}0000_01D_02H_GIM.INX.gz",  
            f"IGS0OPSRAP_{year}{doy}0000_01D_02H_GIM.INX.gz"
        ]

    def get_file_for_date(self, dt: date) -> Optional[Path]:
        possible_files = self._expected_filenames(dt)

        for fileName in possible_files:
            path = self.base_dir / fileName
            if path.exists():
                logging.info(f"Found local file for {dt}: {path.name}")
                return path

        if self.auto_download:
            logging.info(f"No local file for {dt}. Attempting download...")
            try:
                path = self.downloader.download_for_date(dt)
                return path
            except Exception as e:
                logging.warning(f"Download failed for {dt}: {e}")
                return None

        logging.warning(f"File for {dt} not found locally and auto-download disabled.")
        return None

    def get_files_in_range(self, start: date, end: date) -> List[Path]:
        current = start
        results: List[Path] = []

        while current <= end:
            path = self.get_file_for_date(current)
            if path:
                results.append(path)
            else:
                logging.warning(f"No file found for {current}")
            current += timedelta(days=1)

        logging.info(f"Retrieved {len(results)} files between {start} and {end}.")
        return results