"""Shared constants for the dataset-construction and scoring pipelines.

Centralizes every tunable value used across both pipeline's scripts
(`download_pexels.py`, `partition_pool.py`, `augment_duplicates.py`,
`build_manifest.py`, `verify_dataset.py`, `embed_clips.py`,
`compute_scores.py`, `plot_validation.py`) so none of them hardcode counts,
paths, or model/transform parameters. See `docs/DATA_CONSTRUCTION.md` and
`docs/JOURNAL.md` for the design rationale behind these specific values.

Grouped by concern:
    - Paths: repo-relative locations for raw/augmented/inspection/plot data
      and the manifest/score files produced across both pipelines' phases.
    - Pexels API: search/video endpoint URLs.
    - Pool discovery: search terms, target pool size, per-term cap, and
      filters (duration, preferred rendition quality) used by
      `download_pexels.py`.
    - Partition sizes: how many clips go to each synthetic partner, and how
      many of Partner B's submissions are synthetic duplicates, used by
      `partition_pool.py` and `build_manifest.py`.
    - Augmentation: fixed, non-randomized parameters for the four
      near-duplicate transforms, used by `augment_duplicates.py`.
    - Scoring: the CLIP checkpoint, frame-sampling count, novelty/trend
      window size, and warmup threshold used by `embed_clips.py`,
      `compute_scores.py`, and `plot_validation.py`.
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

# --- Scoring pipeline (embed_clips.py, compute_scores.py, plot_validation.py) ---

EMBEDDINGS_PATH = DATA_DIR / "embeddings.npz"
SCORES_PATH = DATA_DIR / "scores.csv"
PLOTS_DIR = DATA_DIR / "plots"
VALIDATION_PLOT_PATH = PLOTS_DIR / "novelty_validation.png"

# See docs/JOURNAL.md "Why CLIP, not a video-native model"
CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
CLIP_EMBEDDING_DIM = 512

# See docs/JOURNAL.md "Frame sampling: fixed count, not fixed rate"
FRAMES_PER_CLIP = 8

# See docs/JOURNAL.md "Per-clip novelty: max similarity, not centroid
# similarity" and "Window size". W and N are equal by deliberate choice,
# not coincidence — kept as one source value with two named aliases so the
# two formulas that use it stay self-documenting if they're ever decoupled.
WINDOW_SIZE = 10
NOVELTY_COMPARISON_WINDOW = WINDOW_SIZE  # W: prior clips a new clip is compared against
TREND_WINDOW = WINDOW_SIZE               # N: novelty scores averaged into the rolling trend

# See docs/JOURNAL.md "Cold start" — a placeholder within the stated 15-20
# range, not tuned on data. Used only by plot_validation.py as an
# annotation of where a real system would start surfacing the trend;
# compute_scores.py computes every clip's score unconditionally.
WARMUP_THRESHOLD = 15
