# 🌐 IONEXER — Global Ionospheric TEC Downloader & Visualizer

**IONEXER** is a Python toolkit for automatically downloading, parsing, and visualizing **IONEX (IONosphere EXchange Format)** maps published by NASA’s CDDIS archive.
It provides a robust workflow for **Total Electron Content (TEC)** analysis, space-weather monitoring, and geophysics research.

<p align="center">
  <img src="https://res.cloudinary.com/superdarncanada/image/upload/v1614024790/superDARN/tutorials/t-iono1_xa19yj.jpg" width="500">
</p>

---

## 🚀 Features

✅ **Automated Downloading**
Fetches daily or multi-day global IONEX files directly from NASA’s CDDIS archive using an authenticated Earthdata token.

✅ **Smart Caching & Retry**
Skips existing files, retries failed downloads, and optionally auto-downloads missing data when parsing.

✅ **Flexible Parsing**
Reads and decompresses `.gz`, `.Z` (LZW), or plain `.IONEX` files using a fallback strategy (`gzip`, `unlzw3`).

✅ **Interactive Visualization**
Generates **global TEC heatmaps** using `matplotlib` + `cartopy`, with date pickers and animation playback between date ranges.

✅ **Command-Line Interface (CLI)**
Run directly from terminal:

```bash
python -m ionexer.main 2025-10-15
```

✅ **Modular & Extensible**
Clean architecture with independent modules (`Downloader`, `FileManager`, `Parser`) for integration in your own research pipelines.

---

## 🧱 Repository Structure

```
IONOSPHERE/
│
├── ionexer/
│   ├── config.py           # Base directory paths
│   ├── downloader.py       # Handles NASA CDDIS downloads (token-auth)
│   ├── file_manager.py     # Finds or downloads missing files
│   ├── parser.py           # Parses IONEX headers, TEC grids, and visualizes them
│   ├── main.py             # CLI entry point and orchestrator
│   └── raw/                # Cached downloaded IONEX/TEC files
│
├── .gitignore
├── README.md
└── requirements.txt
```

---

## ⚙️ Installation

### 1️⃣ Clone the repository

```bash
git clone https://github.com/AnooshaLamouchi/ionexer.git
cd ionexer
```

### 2️⃣ Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate      # (Linux/macOS)
# or
.venv\Scripts\activate         # (Windows)
```

### 3️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

Typical dependencies:

```text
requests
numpy
matplotlib
cartopy
unlzw3
```

---

## 🔐 NASA Earthdata Login Setup

This tool uses your **Earthdata bearer token** to download IONEX files.

1. Go to [https://urs.earthdata.nasa.gov/profile](https://urs.earthdata.nasa.gov/profile)
2. Generate a new **“Application Token”**
3. Copy the token and paste it in [`downloader.py`](ionexer/downloader.py):

   ```python
   EARTHDATA_TOKEN = "your_new_token_here"
   ```

---

## 💻 Usage

### ▶ Run for a specific date

```bash
python -m ionexer.main 2025-10-27
```

### ▶ Run without arguments (defaults to today)

```bash
python -m ionexer.main
```

### ▶ Example output

```
📅 Target date: 2025-10-27
✅ File ready: ionexer/raw/IGS0OPSFIN_2025300000_01D_02H_GIM.INX.gz
Parsed TEC map: shape (71, 361)
🌍 Displaying heatmap...
```

---

## 📊 Visualization

The interactive plot includes:

* **Global map projection (PlateCarree)**
* **Date input box** for direct selection
* **Range inputs (Start / End)** for animation
* **Play/Pause** button for sequential playback of TEC evolution

---

## 🧩 Example: Programmatic Use

```python
from datetime import date
from ionexer.file_manager import FileManager
from ionexer.parser import Parser

fm = FileManager(auto_download=True)
file_path = fm.get_file_for_date(date(2025, 10, 28))

parser = Parser(file_path)
tec, lats, lons = parser.parse()
parser.plot_heatmap()
```

---

## 🧠 Architecture Overview

| Component     | Description                                                                                |
| ------------- | ------------------------------------------------------------------------------------------ |
| `Downloader`  | Connects to NASA CDDIS, constructs URLs, manages retries, downloads `.Z`/`.gz` IONEX files |
| `FileManager` | Locates local files or triggers download if missing                                        |
| `Parser`      | Decompresses, parses LAT/LON grids, extracts TEC maps, auto-scales, and visualizes         |
| `main.py`     | Command-line entrypoint for user interaction                                               |
| `config.py`   | Defines local storage paths and constants                                                  |

---

## 🧰 Logging & Error Handling

* Automatic retry logic for unstable connections
* Warnings for missing files or invalid Earthdata tokens
* Graceful fallback between decompression methods (`gzip` → `unlzw3`)
* Logging enabled by default in `Downloader(verbose=True)`

---

## 🗂️ Data Storage

Downloaded files are stored in:

```
ionexer/raw/
```

Each file name follows NASA’s IONEX convention:

```
IGS0OPSFIN_YYYYDDD0000_01D_02H_GIM.INX.gz
```

where `YYYY` = year, `DDD` = day of year.

---

## 🧾 License

MIT License © 2025 Anoosha Lamouchi

---

## 🧑‍💻 Author

Developed by **Anoosha Lamouchi**
Atmosphere Physics Research • Tehran / K. N. Toosi University of Technology

Contact: [GitHub Profile](https://github.com/AnooshaLamouchi) | [LinkedIn](https://linkedin.com/in/anoosha-lamouchi)
