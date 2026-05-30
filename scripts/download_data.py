"""Download the Home Credit Default Risk competition data from Kaggle.

Requirements:
  - ~/.kaggle/kaggle.json with a valid API token (chmod 600)
  - You must have ACCEPTED the competition rules on the Kaggle website first,
    otherwise the API returns 403 Forbidden.

Usage:
    python scripts/download_data.py            # download + unzip all files
    python scripts/download_data.py --files application_train.csv bureau.csv
"""

from __future__ import annotations

import argparse
import sys
import zipfile

from creditlens.config import KAGGLE_COMPETITION, RAW_DIR


def main() -> int:
    parser = argparse.ArgumentParser(description="Download Home Credit data from Kaggle")
    parser.add_argument(
        "--files",
        nargs="*",
        default=None,
        help="Specific files to download (default: whole competition archive)",
    )
    args = parser.parse_args()

    RAW_DIR.mkdir(parents=True, exist_ok=True)

    # Import here so a missing 'kaggle' package gives a clear message, not an import crash.
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except OSError as exc:  # raised at import if credentials are missing/invalid
        print(f"Kaggle credentials error: {exc}", file=sys.stderr)
        return 1
    except ImportError:
        print("The 'kaggle' package is not installed. Run: pip install kaggle", file=sys.stderr)
        return 1

    api = KaggleApi()
    api.authenticate()

    if args.files:
        for fname in args.files:
            print(f"Downloading {fname} ...")
            api.competition_download_file(KAGGLE_COMPETITION, fname, path=str(RAW_DIR))
    else:
        print(f"Downloading full competition archive: {KAGGLE_COMPETITION} ...")
        api.competition_download_files(KAGGLE_COMPETITION, path=str(RAW_DIR))

    # Decompress archives. The whole-competition download lands as *.zip; the
    # per-file API returns a ZIP archive but saves it with a *.csv name, so we
    # check every file's magic bytes rather than trusting the extension.
    for path in list(RAW_DIR.iterdir()):
        if not path.is_file() or not zipfile.is_zipfile(path):
            continue
        print(f"Unzipping {path.name} ...")
        with zipfile.ZipFile(path) as zf:
            names = zf.namelist()
            zf.extractall(RAW_DIR)
        # If a *.csv was actually a zip wrapping the same-named CSV, the extract
        # above already overwrote it with the real CSV. Only remove leftover *.zip.
        if path.suffix == ".zip":
            path.unlink()
        elif names == [path.name]:
            pass  # extracted in place, nothing to clean

    print(f"Done. Files in {RAW_DIR}:")
    for p in sorted(RAW_DIR.glob("*.csv")):
        print(f"  {p.name}  ({p.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
