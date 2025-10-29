import sys
from datetime import date, datetime
from pathlib import Path

from .file_manager import FileManager
from .parser import Parser


def main():
    if len(sys.argv) > 1:
        try:
            target_date = datetime.strptime(sys.argv[1], "%Y-%m-%d").date()
        except ValueError:
            print("❌ Invalid date format. Use YYYY-MM-DD (e.g., 2025-10-15).")
            return
    else:
        target_date = date.today()
    
    print(f"📅 Target date: {target_date}")

    fm = FileManager(auto_download=True)
    file_path = fm.get_file_for_date(target_date)

    if not file_path or not Path(file_path).exists():
        print("❌ File not found or could not be downloaded.")
        return

    print(f"✅ File ready: {file_path}")

    try:
        reader = Parser(file_path)
        reader.parse()
        reader.plot_heatmap()
    except Exception as e:
        print(f"❌ Failed to parse or visualize: {e}")


if __name__ == "__main__":
    main()
