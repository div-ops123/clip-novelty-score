"""Phase 1.5 of the validation-dataset pipeline: assign clips to partners.

Deterministically splits the pool built by `download_pexels.py` into two
synthetic partners and an arrival order, and designates one of Partner B's
clips as the source that `augment_duplicates.py` will clone into
near-duplicates. Writes `data/partition.json`, which is intentionally plain
and hand-editable so a specific auto-picked clip can be swapped for another
unused pool clip after a quick visual check, without touching any code.

Usage:
    python scripts/partition_pool.py
"""

import csv
import json
import random
import sys

import config


def main() -> None:
    """Splits the raw clip pool into Partner A/B assignments and writes partition.json.

    Reads `data/raw_manifest.csv`, deterministically shuffles the pool with
    a fixed seed (`config.PARTITION_SEED`), and takes the first
    `config.PARTNER_A_COUNT` clips for Partner A and the next
    `config.PARTNER_B_GENUINE_COUNT` for Partner B's genuine submissions.
    The first of Partner B's genuine clips is designated the
    `duplication_source` for the later augmentation phase.

    Raises:
        SystemExit: If `raw_manifest.csv` is missing, or the pool is too
            small to cover both partners' clip counts.
    """
    if not config.RAW_MANIFEST_PATH.exists():
        sys.exit("raw_manifest.csv not found — run download_pexels.py discover first.")

    with open(config.RAW_MANIFEST_PATH, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    pool = [row["pexels_id"] for row in rows]

    needed = config.PARTNER_A_COUNT + config.PARTNER_B_GENUINE_COUNT
    if len(pool) < needed:
        sys.exit(f"Pool too small: have {len(pool)}, need {needed}. Run discover again or add search terms.")

    rng = random.Random(config.PARTITION_SEED)
    shuffled = pool[:]
    rng.shuffle(shuffled)

    partner_a = shuffled[: config.PARTNER_A_COUNT]
    partner_b_genuine = shuffled[config.PARTNER_A_COUNT : needed]
    duplication_source = partner_b_genuine[0]

    partition = {
        "partner_a": partner_a,
        "partner_b_genuine": partner_b_genuine,
        "duplication_source": duplication_source,
    }

    with open(config.PARTITION_PATH, "w", encoding="utf-8") as f:
        json.dump(partition, f, indent=2)

    print(f"Partner A: {len(partner_a)} clips")
    print(f"Partner B (genuine): {len(partner_b_genuine)} clips")
    print(f"Duplication source (Partner B): {duplication_source}")
    print(f"-> {config.PARTITION_PATH}")
    print("\nEdit this file by hand if a clip looks like a poor POV example, then rerun build_manifest.py.")


if __name__ == "__main__":
    main()
