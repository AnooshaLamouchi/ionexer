from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DOWNLOAD_DIR = BASE_DIR / "raw"

DEFAULT_WINDOW_DAYS = 27
DEFAULT_THRESHOLD = 3.5
EPS = 1e-6  