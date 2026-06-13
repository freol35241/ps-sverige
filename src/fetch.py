"""Download Riksdag bulk dumps to data/raw/, idempotent."""
from __future__ import annotations

from pathlib import Path
import sys

import requests

RAW = Path(__file__).resolve().parent.parent / "data" / "raw"
BASE = "https://data.riksdagen.se/dataset"

# Current mandate, elected September 2022.
RIKSMOTEN = ["202223", "202324", "202425", "202526"]


def fetch(dataset: str, riksmote: str) -> Path:
    """Download dataset-riksmote.json.zip to data/raw/ if missing."""
    RAW.mkdir(parents=True, exist_ok=True)
    name = f"{dataset}-{riksmote}.json.zip"
    dest = RAW / name
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    url = f"{BASE}/{dataset}/{name}"
    print(f"  downloading {url}")
    r = requests.get(url, timeout=120, stream=True)
    r.raise_for_status()
    with dest.open("wb") as f:
        for chunk in r.iter_content(chunk_size=1 << 16):
            f.write(chunk)
    return dest


def fetch_personlista() -> Path:
    """Sitting MPs (rdlstatus=tjanst => currently serving)."""
    RAW.mkdir(parents=True, exist_ok=True)
    dest = RAW / "personlista-tjanst.json"
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    url = "https://data.riksdagen.se/personlista/?utformat=json&rdlstatus=tjanst"
    print(f"  downloading {url}")
    r = requests.get(url, timeout=120)
    r.raise_for_status()
    dest.write_bytes(r.content)
    return dest


def main() -> None:
    print("votering bulk dumps:")
    for rm in RIKSMOTEN:
        p = fetch("votering", rm)
        print(f"  {p.name}: {p.stat().st_size:>10,} bytes")
    print("anforande bulk dumps:")
    for rm in RIKSMOTEN:
        p = fetch("anforande", rm)
        print(f"  {p.name}: {p.stat().st_size:>10,} bytes")
    print("personlista (sitting MPs):")
    p = fetch_personlista()
    print(f"  {p.name}: {p.stat().st_size:>10,} bytes")


if __name__ == "__main__":
    sys.exit(main())
