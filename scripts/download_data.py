"""Re-download the IBM HR Analytics Employee Attrition CSV from its public mirror
and verify it against the known SHA-256 before saving to data/.

Usage: python scripts/download_data.py
"""
from __future__ import annotations

import hashlib
import sys
import urllib.request
from pathlib import Path

MIRROR_URL = "https://raw.githubusercontent.com/netsatsawat/HR-Analytics/master/data/WA_Fn-UseC_-HR-Employee-Attrition.csv"
EXPECTED_SHA256 = "a5c31e38bd7fafc9bc333884eb181b06b41b8e5e488e8f7ccb27199fb3be7659"
EXPECTED_ROWS = 1470
EXPECTED_COLS = 35

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "data" / "WA_Fn-UseC_-HR-Employee-Attrition.csv"


def main() -> None:
    print(f"Downloading {MIRROR_URL}")
    urllib.request.urlretrieve(MIRROR_URL, DEST)
    digest = hashlib.sha256(DEST.read_bytes()).hexdigest()
    if digest != EXPECTED_SHA256:
        DEST.unlink(missing_ok=True)
        print(f"SHA-256 mismatch: got {digest}, expected {EXPECTED_SHA256}")
        raise SystemExit(1)
    print(f"Saved {DEST} ({DEST.stat().st_size} bytes, sha256 verified)")


if __name__ == "__main__":
    main()
