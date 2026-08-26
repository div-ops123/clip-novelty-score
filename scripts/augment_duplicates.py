"""Phase 2 of the validation-dataset pipeline: generate near-duplicates.

Loads Partner B's `duplication_source` clip (chosen by `partition_pool.py`)
and generates `config.PARTNER_B_DUP_COUNT` near-duplicate variants of it
using deterministic combinations of four transforms: crop, brightness
shift, horizontal flip, and speed change. These variants are Partner B's
synthetic "quota-farming" submissions with known ground truth — each is
provably a perturbed copy of `duplication_source`.

Usage:
    python scripts/augment_duplicates.py
"""

import csv
import itertools
import json
import sys

import cv2
import numpy as np

import config


def _read_raw_manifest() -> dict[str, dict]:
    """Loads raw_manifest.csv as a dict keyed by Pexels ID."""
    with open(config.RAW_MANIFEST_PATH, newline="", encoding="utf-8") as f:
        return {row["pexels_id"]: row for row in csv.DictReader(f)}


def _load_frames(path) -> tuple[list, float, int, int]:
    """Reads every frame of a video into memory via OpenCV.

    Full in-memory buffering is acceptable here since source clips are
    short (a few hundred frames); this keeps the augmentation logic simple.

    Args:
        path: Path to the source video file.

    Returns:
        A tuple of (frames, fps, width, height).

    Raises:
        SystemExit: If the video can't be opened or contains no frames.
    """
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        sys.exit(f"Could not open source video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    if not frames:
        sys.exit(f"No frames read from: {path}")
    return frames, fps, width, height


def _apply_crop(frame: np.ndarray) -> np.ndarray:
    """Center-crops a frame by `config.CROP_PCT` per edge, then resizes back to the original size."""
    h, w = frame.shape[:2]
    dx, dy = int(w * config.CROP_PCT), int(h * config.CROP_PCT)
    cropped = frame[dy : h - dy, dx : w - dx]
    return cv2.resize(cropped, (w, h))


def _apply_brightness(frame: np.ndarray) -> np.ndarray:
    """Applies a fixed gain/offset brightness-contrast shift to a frame."""
    return cv2.convertScaleAbs(frame, alpha=config.BRIGHTNESS_ALPHA, beta=config.BRIGHTNESS_BETA)


def _apply_flip(frame: np.ndarray) -> np.ndarray:
    """Mirrors a frame horizontally."""
    return cv2.flip(frame, 1)


def _apply_speed(frames: list) -> list:
    """Resamples a frame sequence to play back at `config.SPEED_FACTOR`x speed.

    Output frame `i` pulls source frame `round(i * SPEED_FACTOR)`, so the
    result is written at the source fps but plays shorter/faster — this is
    genuine frame resampling, not a metadata-only fps relabel.

    Args:
        frames: The full source frame sequence.

    Returns:
        A shorter list of frames sampled at the sped-up rate.
    """
    n = len(frames)
    out = []
    i = 0
    while True:
        src_idx = round(i * config.SPEED_FACTOR)
        if src_idx >= n:
            break
        out.append(frames[src_idx])
        i += 1
    return out


TRANSFORM_ORDER = ["crop", "brightness", "flip", "speed"]
"""list[str]: Canonical application order for the four base transforms.

`speed` (a frame-count-changing operation) always runs first when present,
then the remaining per-frame pixel ops run in this fixed order — this
guarantees identical output for a given transform combination regardless
of how that combination happens to be listed elsewhere.
"""


def _build_recipes(count: int) -> list[tuple[str, ...]]:
    """Builds `count` deterministic transform-combination recipes.

    Generates non-empty subsets of `TRANSFORM_ORDER` ordered by subset size
    (all 4 singles, then all 6 pairs, ...) via `itertools.combinations`, so
    recipes are reproducible and no two duplicate submissions ever use an
    identical transform combination.

    Args:
        count: Number of recipes to return.

    Returns:
        A list of `count` transform-name tuples.
    """
    recipes = []
    for r in range(1, len(TRANSFORM_ORDER) + 1):
        for combo in itertools.combinations(TRANSFORM_ORDER, r):
            recipes.append(combo)
        if len(recipes) >= count:
            break
    return recipes[:count]


def _apply_recipe(frames: list, recipe: tuple[str, ...]) -> list:
    """Applies one transform-combination recipe to a frame sequence.

    Args:
        frames: The source frame sequence.
        recipe: Transform names to apply, e.g. `("crop", "flip")`.

    Returns:
        The transformed frame sequence.
    """
    working = frames
    if "speed" in recipe:
        working = _apply_speed(working)
    per_frame_ops = [t for t in TRANSFORM_ORDER if t in recipe and t != "speed"]
    if per_frame_ops:
        new_frames = []
        for frame in working:
            f = frame
            for op in per_frame_ops:
                if op == "crop":
                    f = _apply_crop(f)
                elif op == "brightness":
                    f = _apply_brightness(f)
                elif op == "flip":
                    f = _apply_flip(f)
            new_frames.append(f)
        working = new_frames
    return working


def main() -> None:
    """Generates all near-duplicate variants for Partner B's duplication_source clip.

    Loads `data/partition.json` to find the designated source clip, buffers
    its frames, applies `config.PARTNER_B_DUP_COUNT` deterministic transform
    recipes (see `_build_recipes`), writes each variant to
    `config.AUGMENTED_DIR`, and records the exact transform parameters used
    for each in `data/augmented_manifest.csv`.

    Raises:
        SystemExit: If `partition.json` is missing.
    """
    if not config.PARTITION_PATH.exists():
        sys.exit("partition.json not found — run partition_pool.py first.")

    with open(config.PARTITION_PATH, encoding="utf-8") as f:
        partition = json.load(f)

    raw_manifest = _read_raw_manifest()
    source_id = partition["duplication_source"]
    source_row = raw_manifest[source_id]
    source_path = config.REPO_ROOT / source_row["local_path"]

    print(f"Loading source clip {source_id} ({source_path})...")
    frames, fps, width, height = _load_frames(source_path)
    print(f"  {len(frames)} frames @ {fps:.1f}fps, {width}x{height}")

    recipes = _build_recipes(config.PARTNER_B_DUP_COUNT)
    config.AUGMENTED_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    for idx, recipe in enumerate(recipes, start=1):
        recipe_name = "+".join(recipe)
        out_frames = _apply_recipe(frames, recipe)
        out_path = config.AUGMENTED_DIR / f"B_dup{idx:02d}_{recipe_name.replace('+', '-')}.mp4"

        writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
        for frame in out_frames:
            writer.write(frame)
        writer.release()

        print(f"  [{idx}/{len(recipes)}] {recipe_name} -> {out_path.name} ({len(out_frames)} frames)")

        transform_params = {
            "crop_pct": config.CROP_PCT if "crop" in recipe else None,
            "brightness_alpha": config.BRIGHTNESS_ALPHA if "brightness" in recipe else None,
            "brightness_beta": config.BRIGHTNESS_BETA if "brightness" in recipe else None,
            "flip": "horizontal" if "flip" in recipe else None,
            "speed_factor": config.SPEED_FACTOR if "speed" in recipe else None,
        }
        rows.append(
            {
                "dup_index": idx,
                "source_pexels_id": source_id,
                "local_path": str(out_path.relative_to(config.REPO_ROOT)).replace("\\", "/"),
                "augmentation_type": recipe_name,
                "transform_params_json": json.dumps({k: v for k, v in transform_params.items() if v is not None}),
            }
        )

    with open(config.AUGMENTED_MANIFEST_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["dup_index", "source_pexels_id", "local_path", "augmentation_type", "transform_params_json"]
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n{len(rows)} duplicate variants -> {config.AUGMENTED_MANIFEST_PATH}")


if __name__ == "__main__":
    main()
