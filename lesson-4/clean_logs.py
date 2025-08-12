from __future__ import annotations

"""Utility script to convert the `lesson-4/nurtureboss_logs.csv` file into a
flattened conversation traces dataset compatible with the *traces.json* format
used elsewhere in the repository.

The CSV contains an `input` column where each cell is itself a JSON-encoded list
of message dictionaries (one per turn of the conversation).  This script:

1. Reads the CSV file.
2. Parses the JSON in each `input` cell.
3. Extracts **only** the `role` and `content` keys from each message to keep the
   output lightweight and model-agnostic.
4. Writes the resulting list of conversations to
   `lesson-4/nurtureboss_traces.json`.

Run this file once whenever `nurtureboss_logs.csv` is updated.

Usage (from repository root):

    python lesson-4/clean_logs.py

The script is idempotent – running it multiple times will simply overwrite the
output file with the latest conversion.
"""

from pathlib import Path
import csv
import json
import uuid
from typing import List, Dict, Any

# Constants – adjust only if the file locations change
CSV_PATH = Path(__file__).with_name("nurtureboss_logs.csv")
OUTPUT_PATH = Path(__file__).with_name("nurtureboss_traces.json")  # keeps same name for backward-compat

# Columns containing JSON strings
JSON_MESSAGES_COLUMN = "input"
JSON_METADATA_COLUMN = "metadata"


def parse_json_cell(raw: str) -> Any:
    """Safely ``json.loads`` a cell that contains JSON data.

    Returns ``None`` if the cell cannot be parsed.
    """
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return None


# ---------------------------------------------------------------------------
# Messages helpers
# ---------------------------------------------------------------------------


def parse_messages(raw: str) -> List[Dict[str, str]] | None:
    """Parse the JSON `input` column into a list of simplified messages.

    Each message keeps only the ``role`` and ``content`` fields. Returns ``None``
    if the payload cannot be parsed or is not a list.
    """
    data = parse_json_cell(raw)
    if not isinstance(data, list):
        return None

    cleaned: List[Dict[str, str]] = []
    for msg in data:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role")
        content = msg.get("content")
        if role is None or content is None:
            continue
        cleaned.append({"role": role, "content": content})

    return cleaned or None


def convert_csv_to_records(csv_path: Path) -> List[Dict[str, Any]]:
    """Read *csv_path* and extract the JSON payload from ``JSON_COLUMN``.

    Each resulting record is the parsed JSON object **augmented** with an "id"
    field sourced from the ``name`` column (or a generated fallback).
    """
    records: List[Dict[str, Any]] = []
    with csv_path.open(newline="", encoding="utf-8") as fp:
        reader = csv.DictReader(fp)
        for idx, row in enumerate(reader, start=1):
            # --- Metadata ----------------------------------------------------
            raw_meta: str | None = row.get(JSON_METADATA_COLUMN)
            meta_payload = parse_json_cell(raw_meta or "") if raw_meta else None

            # --- Messages ----------------------------------------------------
            raw_input: str | None = row.get(JSON_MESSAGES_COLUMN)
            messages = parse_messages(raw_input) if raw_input else None

            # Skip rows that have neither payload
            if meta_payload is None and messages is None:
                continue

            record: Dict[str, Any] = {}

            # Merge metadata (if dict) or store under "metadata" key
            if isinstance(meta_payload, dict):
                record.update(meta_payload)
            elif meta_payload is not None:
                record["metadata"] = meta_payload

            if messages is not None:
                record["messages"] = messages

            # Always assign a unique identifier
            record["id"] = row.get("name") or f"row_{idx:04d}_{uuid.uuid4().hex[:8]}"

            records.append(record)
    return records


def main() -> None:
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"CSV file not found at {CSV_PATH.relative_to(Path.cwd())}")

    records = convert_csv_to_records(CSV_PATH)
    if not records:
        print("No records extracted – output file not written.")
        return

    with OUTPUT_PATH.open("w", encoding="utf-8") as fp:
        json.dump(records, fp, ensure_ascii=False, indent=2)

    print(f"Wrote {len(records)} records → {OUTPUT_PATH.relative_to(Path.cwd())}")


if __name__ == "__main__":
    main()
