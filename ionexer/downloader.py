import requests
from pathlib import Path
from datetime import date, timedelta
import logging
import time
from typing import List, Optional
from .config import DOWNLOAD_DIR

CDDIS_BASE_URL = "https://cddis.nasa.gov/archive/gnss/products/ionex"
EARTHDATA_TOKEN = "eyJ0eXAiOiJKV1QiLCJvcmlnaW4iOiJFYXJ0aGRhdGEgTG9naW4iLCJzaWciOiJlZGxqd3RwdWJrZXlfb3BzIiwiYWxnIjoiUlMyNTYifQ.eyJ0eXBlIjoiVXNlciIsInVpZCI6ImFudXNoYWxhbW91Y2hpIiwiZXhwIjoxNzcyNDY4NDEzLCJpYXQiOjE3NjcyODQ0MTMsImlzcyI6Imh0dHBzOi8vdXJzLmVhcnRoZGF0YS5uYXNhLmdvdiIsImlkZW50aXR5X3Byb3ZpZGVyIjoiZWRsX29wcyIsImFjciI6ImVkbCIsImFzc3VyYW5jZV9sZXZlbCI6M30.pak60Rum8MG6-3lC2DJDGI3XGhZUOiAk_ZLJ68egwc1A4lQeexHBQSHk7W9ZB6LTwglXkA_c0xoF31JRmarH7RFm5AyB_1mXf6BWhSfHe6DvcvHCFq4x48_UOA2I4p2FhxZiCgg8X5egGY-TwwlzjxrFS6ntXeVkStriUmvhiUv-ZZPKnrmzze2QeS07YgTajXPi5iid8RyO9fEpzfteU4BlYfH1intBMWqyHWijE0_PahMfEecLLy5QAoe0wMS5A2O27nreXA72uOSvZ6CK7uotfPa0-MphjRNTp1yWcA02vq7hpFajZD0J4yJLD3IsbwyLyvAE96HqxTTTACiwgw"


class Downloader:
    def __init__(
        self,
        download_dir: Path = DOWNLOAD_DIR,
        session: Optional[requests.Session] = None,
        verbose: bool = True,
    ):
        self.download_dir = Path(download_dir)
        self.download_dir.mkdir(parents=True, exist_ok=True)

        self.session = session or requests.Session()
        self.session.headers.update({
            "User-Agent": "ionex-downloader/1.0 (+https://cddis.nasa.gov/)",
            "Authorization": f"Bearer {EARTHDATA_TOKEN}",
            "Connection": "keep-alive",
        })
        self.session.max_redirects = 5

        if verbose:
            logging.basicConfig(
                level=logging.INFO,
                format="%(asctime)s [%(levelname)s] %(message)s",
                datefmt="%H:%M:%S",
            )

    @staticmethod
    def _date_to_doy(dt: date) -> str:
        return f"{dt.timetuple().tm_yday:03d}"

    def _construct_urls(self, dt: date) -> List[str]:
        year = dt.year
        doy = self._date_to_doy(dt)
        yy = str(year)[2:]
        folder = f"{year}/{doy}"
        return [
            f"{CDDIS_BASE_URL}/{folder}/c1pg{doy}0.{yy}i.Z",
            f"{CDDIS_BASE_URL}/{folder}/c2pg{doy}0.{yy}i.Z",
            f"{CDDIS_BASE_URL}/{folder}/IGS0OPSFIN_{year}{doy}0000_01D_02H_GIM.INX.gz",
            f"{CDDIS_BASE_URL}/{folder}/IGS0OPSRAP_{year}{doy}0000_01D_02H_GIM.INX.gz",
        ]

    def _fetch(self, url: str):
        r = self.session.get(url, allow_redirects=True, stream=True, timeout=120)
        if b"<title>Earthdata Login" in r.content[:300]:
            raise PermissionError(
                "Earthdata token invalid or expired. Regenerate it in your profile."
            )
        return r

    def download_for_date(self, dt: date, overwrite=False, max_retries=3, delay_sec=5) -> Path:
        urls = self._construct_urls(dt)

        for url in urls:
            local_fname = self.download_dir / Path(url).name
            if local_fname.exists() and not overwrite:
                logging.info(f"File already exists: {local_fname.name}")
                return local_fname

            for attempt in range(1, max_retries + 1):
                try:
                    logging.info(f"Downloading {url} (attempt {attempt}/{max_retries})")
                    r = self._fetch(url)

                    if r.status_code == 200:
                        with open(local_fname, "wb") as f:
                            for chunk in r.iter_content(chunk_size=8192):
                                f.write(chunk)
                        logging.info(f"✅ Downloaded {local_fname.name}")
                        return local_fname

                    elif r.status_code == 404:
                        logging.warning(f"❌ Not found: {url}")
                        break

                    else:
                        logging.warning(f"Unexpected HTTP {r.status_code} for {url}")

                except PermissionError as e:
                    logging.error(str(e))
                    return None
                except requests.RequestException as e:
                    logging.warning(f"Network error: {e}")

                if attempt < max_retries:
                    logging.info(f"Retrying in {delay_sec}s...")
                    time.sleep(delay_sec)

            logging.warning(f"Failed to fetch {url}")
        raise FileNotFoundError(f"No IONEX file found for date {dt}")

    def download_range(self, start: date, end: date, overwrite=False, **kwargs) -> List[Path]:
        current = start
        results = []
        while current <= end:
            try:
                path = self.download_for_date(current, overwrite=overwrite, **kwargs)
                if path:
                    results.append(path)
            except FileNotFoundError:
                logging.warning(f"No file for {current}")
            current += timedelta(days=1)
        logging.info(f"Finished downloading {len(results)} files.")
        return results