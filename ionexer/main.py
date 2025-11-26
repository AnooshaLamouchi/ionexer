import sys
from datetime import date, datetime

from .file_manager import FileManager
from .parser import Parser
from .anomaly import AnomalyDetector
from .config import DEFAULT_WINDOW_DAYS, DEFAULT_THRESHOLD


def _parse_args(argv):
    args = {
        "date": None,
        "method": "robust",
        "window": DEFAULT_WINDOW_DAYS,
        "threshold": DEFAULT_THRESHOLD,
        "viewer": False,
    }
    it = iter(argv[1:])
    if len(argv) > 1 and not str(argv[1]).startswith("--"):
        try:
            args["date"] = datetime.strptime(argv[1], "%Y-%m-%d").date()
            next(it)
        except Exception:
            pass
    for tok in it:
        if tok == "--method":
            args["method"] = next(it, "robust")
        elif tok == "--window":
            args["window"] = int(next(it, str(DEFAULT_WINDOW_DAYS)))
        elif tok in ("--threshold", "--k"):
            args["threshold"] = float(next(it, str(DEFAULT_THRESHOLD)))
        elif tok == "--viewer":
            args["viewer"] = True
    return args


def main():
    args = _parse_args(sys.argv)
    try:
        target_date = args["date"] or date.today()

        if args["viewer"]:
            Parser.create_interactive_viewer(
                initial_date=target_date,
                threshold=args["threshold"],
                window_days=args["window"],
            )
            return

        fm = FileManager()
        file_path = fm.get_file_for_date(target_date)
        if not file_path:
            print(f"⚠️  Could not locate or download a file for {target_date}.")
            return

        reader = Parser(file_path)
        tec, lats, lons = reader.parse()

        detector = AnomalyDetector(file_path.parent, window_days=args["window"])
        if args["method"].lower() == "zscore":
            z_map, anomalies = detector.detect_zscore(target_date, threshold=args["threshold"])
        else:
            z_map, anomalies, _ = detector.detect_robust(target_date, threshold=args["threshold"])
            if z_map is None:
                z_map, anomalies = detector.detect_zscore(target_date, threshold=args["threshold"])

        if z_map is not None:
            Parser.plot_anomaly_iran(z_map, lats, lons, threshold=args["threshold"])
        else:
            print("⚠️  Could not compute anomalies.")

    except Exception as e:
        print(f"❌ Failed to parse or visualize: {e}")


if __name__ == "__main__":
    main()