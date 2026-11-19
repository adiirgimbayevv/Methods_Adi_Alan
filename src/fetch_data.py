"""Download the public, de-identified ProgFeed classroom dataset."""
from __future__ import annotations

import hashlib
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
URL = (
    "https://raw.githubusercontent.com/umass-ml4ed/"
    "progFeed-dataset-public/e020c7c013187dba7b4130991eadc539d115526a/"
    "all_submissions_consolidated.csv"
)
DEST = ROOT / "data" / "raw" / "progfeed" / "all_submissions_consolidated.csv"
EXPECTED_SHA256 = "228124c59a5b5a87dec983b43099ec31abe6ca1dcfba45e3aadcc3d83382895b"


def fetch() -> Path:
    DEST.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(URL, headers={"User-Agent": "Methods-course-research/1.0"})
    with urllib.request.urlopen(request, timeout=60) as response, DEST.open("wb") as output:
        output.write(response.read())
    digest = hashlib.sha256(DEST.read_bytes()).hexdigest()
    if digest != EXPECTED_SHA256:
        DEST.unlink(missing_ok=True)
        raise ValueError(f"Dataset checksum mismatch: expected {EXPECTED_SHA256}, got {digest}")
    print(f"Downloaded {DEST.relative_to(ROOT)} ({DEST.stat().st_size:,} bytes)")
    print(f"SHA-256: {digest}")
    return DEST


if __name__ == "__main__":
    fetch()
