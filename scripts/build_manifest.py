"""Phase 3 of the validation-dataset pipeline: build the final manifest.

Pure, deterministic merge of the three earlier-phase outputs
(`raw_manifest.csv`, `partition.json`, `augmented_manifest.csv`) into
`data/manifest.csv` — the single file the (separate, future) CLIP
embedding/scoring pipeline will read. Since it only reads already-computed
inputs, it's safe to rerun any time, e.g. after hand-editing
`partition.json`.

Usage:
    python scripts/build_manifest.py
"""

import csv
import json
import sys

import config

FIELDNAMES = [
    "clip_id",
    "partner_id",
    "arrival_index",
    "source_original_id",
    "is_synthetic_duplicate",
    "local_path",
    "pexels_id",
    "search_term",
    "source_url",
    "augmentation_type",
]


def main() -> None:
    """Merges the phase 1/1.5/2 outputs into data/manifest.csv.

    Builds one row per clip with a stable `clip_id` (`A01`..`A18`,
    `B01`..`B16`), a per-partner `arrival_index`, and ground-truth labels
    (`source_original_id`, `is_synthetic_duplicate`) — see
    `docs/DATA_CONSTRUCTION.md` for the full manifest schema.

    Raises:
        SystemExit: If any of the three required input files is missing.
    """
    for path in (config.RAW_MANIFEST_PATH, config.PARTITION_PATH, config.AUGMENTED_MANIFEST_PATH):
        if not path.exists():
            sys.exit(f"Missing {path} — run the earlier phase scripts first.")

    with open(config.RAW_MANIFEST_PATH, newline="", encoding="utf-8") as f:
        raw_by_id = {row["pexels_id"]: row for row in csv.DictReader(f)}
    with open(config.PARTITION_PATH, encoding="utf-8") as f:
        partition = json.load(f)
    with open(config.AUGMENTED_MANIFEST_PATH, newline="", encoding="utf-8") as f:
        augmented_rows = list(csv.DictReader(f))

    rows = []

    for idx, pexels_id in enumerate(partition["partner_a"], start=1):
        raw = raw_by_id[pexels_id]
        clip_id = f"A{idx:02d}"
        rows.append(
            {
                "clip_id": clip_id,
                "partner_id": "partner_a",
                "arrival_index": idx,
                "source_original_id": clip_id,
                "is_synthetic_duplicate": False,
                "local_path": raw["local_path"],
                "pexels_id": pexels_id,
                "search_term": raw["search_term"],
                "source_url": raw["source_url"],
                "augmentation_type": "",
            }
        )

    b_genuine = partition["partner_b_genuine"]
    pexels_to_clip_id = {}
    for idx, pexels_id in enumerate(b_genuine, start=1):
        raw = raw_by_id[pexels_id]
        clip_id = f"B{idx:02d}"
        pexels_to_clip_id[pexels_id] = clip_id
        rows.append(
            {
                "clip_id": clip_id,
                "partner_id": "partner_b",
                "arrival_index": idx,
                "source_original_id": clip_id,
                "is_synthetic_duplicate": False,
                "local_path": raw["local_path"],
                "pexels_id": pexels_id,
                "search_term": raw["search_term"],
                "source_url": raw["source_url"],
                "augmentation_type": "",
            }
        )

    dup_source_clip_id = pexels_to_clip_id[partition["duplication_source"]]
    start_idx = len(b_genuine) + 1
    for offset, aug_row in enumerate(augmented_rows):
        idx = start_idx + offset
        clip_id = f"B{idx:02d}"
        rows.append(
            {
                "clip_id": clip_id,
                "partner_id": "partner_b",
                "arrival_index": idx,
                "source_original_id": dup_source_clip_id,
                "is_synthetic_duplicate": True,
                "local_path": aug_row["local_path"],
                "pexels_id": "",
                "search_term": "",
                "source_url": "",
                "augmentation_type": aug_row["augmentation_type"],
            }
        )

    rows.sort(key=lambda r: (r["partner_id"], r["arrival_index"]))

    with open(config.FINAL_MANIFEST_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    a_count = sum(1 for r in rows if r["partner_id"] == "partner_a")
    b_count = sum(1 for r in rows if r["partner_id"] == "partner_b")
    print(f"Partner A: {a_count} clips, Partner B: {b_count} clips ({len(augmented_rows)} duplicates)")
    print(f"-> {config.FINAL_MANIFEST_PATH}")


if __name__ == "__main__":
    main()
