"""Shared constants for the validation-dataset construction pipeline.

Centralizes every tunable value used across the pipeline scripts
(`download_pexels.py`, `partition_pool.py`, `augment_duplicates.py`,
`build_manifest.py`, `verify_dataset.py`) so none of them hardcode counts,
paths, or transform parameters. See `docs/DATA_CONSTRUCTION.md` for the
design rationale behind these specific values.

Grouped by concern:
    - Paths: repo-relative locations for raw/augmented/inspection data and
      the four manifest files produced across the pipeline's phases.
    - Pexels API: search/video endpoint URLs.
    - Pool discovery: search terms, target pool size, per-term cap, and
      filters (duration, preferred rendition quality) used by
      `download_pexels.py`.
    - Partition sizes: how many clips go to each synthetic partner, and how
      many of Partner B's submissions are synthetic duplicates, used by
      `partition_pool.py` and `build_manifest.py`.
    - Augmentation: fixed, non-randomized parameters for the four
      near-duplicate transforms, used by `augment_duplicates.py`.
"""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
AUGMENTED_DIR = DATA_DIR / "augmented"
INSPECTION_DIR = DATA_DIR / "inspection"

RAW_MANIFEST_PATH = DATA_DIR / "raw_manifest.csv"
PARTITION_PATH = DATA_DIR / "partition.json"
AUGMENTED_MANIFEST_PATH = DATA_DIR / "augmented_manifest.csv"
FINAL_MANIFEST_PATH = DATA_DIR / "manifest.csv"

PEXELS_SEARCH_URL = "https://api.pexels.com/videos/search"
PEXELS_VIDEO_URL = "https://api.pexels.com/videos/videos/{video_id}"

SEARCH_TERMS = [
    "pov hands tool",
    "first person cooking",
    "pov mechanic",
    "pov woodworking",
    "pov assembly",
    "pov gardening",
    "pov cleaning",
    "pov painting",
    "pov crafting",
    "pov typing",
    "pov baking",
]

TARGET_POOL_SIZE = 28
MAX_PER_TERM = 4
RESULTS_PER_PAGE = 15
MAX_DURATION_SEC = 30
PREFERRED_QUALITY = "sd"

PARTNER_A_COUNT = 18
PARTNER_B_GENUINE_COUNT = 9
PARTNER_B_TOTAL = 16
PARTNER_B_DUP_COUNT = PARTNER_B_TOTAL - PARTNER_B_GENUINE_COUNT  # 7

PARTITION_SEED = 42

# Augmentation constants (fixed, deterministic — not randomized)
CROP_PCT = 0.10          # trim 10% off each edge, then resize back
BRIGHTNESS_ALPHA = 1.2   # gain
BRIGHTNESS_BETA = 25     # offset
SPEED_FACTOR = 1.15      # resample frames faster by this factor

BASE_TRANSFORMS = ["crop", "brightness", "flip", "speed"]
