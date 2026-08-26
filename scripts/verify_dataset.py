"""Phase 4 of the validation-dataset pipeline: verify the built dataset.

Sanity-checks `data/manifest.csv` before it's handed to the (separate,
future) scoring pipeline: structural invariants (clip counts, contiguous
arrival order, correct duplicate/original ground-truth flags, no reused
Pexels IDs) and file integrity (every video opens and is readable via
OpenCV). Also writes a side-by-side middle-frame comparison PNG per
duplicate to `data/inspection/`, so a human can visually confirm a
duplicate really does look like a perturbed version of its source. Exits
with a non-zero status if any check fails, so it can double as a CI gate.

Usage:
    python scripts/verify_dataset.py
"""

import sys

import cv2
import numpy as np
import pandas as pd

import config


def _fail(msg: str, failures: list[str]) -> None:
    """Records and prints a single verification failure."""
    failures.append(msg)
    print(f"FAIL: {msg}")


def check_structure(df: pd.DataFrame, failures: list[str]) -> None:
    """Checks manifest-level invariants against ground truth expectations.

    Verifies, per partner: clip count is in [15, 20] and `arrival_index` is
    contiguous from 1. Verifies Partner A has no rows flagged as synthetic
    duplicates. Verifies Partner B's duplicate rows all arrive after its
    genuine rows and all reference the same `source_original_id`. Verifies
    no Pexels ID is reused across manifest rows.

    Args:
        df: The loaded `data/manifest.csv` as a DataFrame.
        failures: Accumulator list; failure messages are appended to it.
    """
    for partner_id, group in df.groupby("partner_id"):
        n = len(group)
        if not (15 <= n <= 20):
            _fail(f"{partner_id}: {n} clips, expected 15-20", failures)

        indices = sorted(group["arrival_index"].tolist())
        if indices != list(range(1, n + 1)):
            _fail(f"{partner_id}: arrival_index not contiguous 1..{n}: {indices}", failures)

    a = df[df["partner_id"] == "partner_a"]
    if a["is_synthetic_duplicate"].any():
        _fail("partner_a has rows marked as synthetic duplicates", failures)
    if not (a["source_original_id"] == a["clip_id"]).all():
        _fail("partner_a has rows where source_original_id != clip_id", failures)

    b = df[df["partner_id"] == "partner_b"].sort_values("arrival_index")
    genuine = b[~b["is_synthetic_duplicate"]]
    dup = b[b["is_synthetic_duplicate"]]
    if len(genuine) == 0 or len(dup) == 0:
        _fail("partner_b missing genuine or duplicate rows", failures)
    else:
        max_genuine_idx = genuine["arrival_index"].max()
        min_dup_idx = dup["arrival_index"].min()
        if min_dup_idx <= max_genuine_idx:
            _fail("partner_b duplicates are not all after genuine clips in arrival order", failures)
        if dup["source_original_id"].nunique() != 1:
            _fail("partner_b duplicates reference more than one source_original_id", failures)

    pexels_ids = df.loc[df["pexels_id"] != "", "pexels_id"]
    dupes = pexels_ids[pexels_ids.duplicated()]
    if not dupes.empty:
        _fail(f"pexels_id reused across manifest rows: {dupes.tolist()}", failures)


def check_files(df: pd.DataFrame, failures: list[str]) -> list[dict]:
    """Checks that every manifest clip's video file exists and is readable.

    Opens each clip with OpenCV and confirms both the first and last frame
    can be read, which catches truncated downloads or incomplete writer
    flushes that a mere file-exists check would miss.

    Args:
        df: The loaded `data/manifest.csv` as a DataFrame.
        failures: Accumulator list; failure messages are appended to it.

    Returns:
        A list of per-clip summary dicts (frame count, fps, resolution) for
        clips that opened successfully, for the printed sanity table.
    """
    summaries = []
    for _, row in df.iterrows():
        path = config.REPO_ROOT / row["local_path"]
        if not path.exists():
            _fail(f"{row['clip_id']}: file missing at {path}", failures)
            continue

        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            _fail(f"{row['clip_id']}: could not open {path}", failures)
            continue

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        ok_first, first_frame = cap.read()
        if frame_count > 1:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count - 1)
        ok_last, _ = cap.read()
        cap.release()

        if frame_count <= 0 or not ok_first:
            _fail(f"{row['clip_id']}: unreadable/empty video ({path})", failures)
            continue
        if not ok_last:
            _fail(f"{row['clip_id']}: could not read last frame ({path})", failures)

        summaries.append(
            {
                "clip_id": row["clip_id"],
                "partner_id": row["partner_id"],
                "frames": frame_count,
                "fps": round(fps, 1),
                "resolution": f"{width}x{height}",
                "mid_frame": first_frame if frame_count <= 1 else None,
            }
        )
    return summaries


def write_comparisons(df: pd.DataFrame) -> None:
    """Writes a side-by-side source-vs-duplicate PNG for every duplicate clip.

    For each manifest row flagged `is_synthetic_duplicate`, grabs the
    middle frame of both it and its `source_original_id` clip and writes
    them side by side to `data/inspection/{dup_id}_vs_{source_id}.png` —
    the concrete artifact a human opens to confirm the augmentation
    actually looks like a perturbed copy of the source.

    Args:
        df: The loaded `data/manifest.csv` as a DataFrame.
    """
    config.INSPECTION_DIR.mkdir(parents=True, exist_ok=True)
    by_clip_id = {row["clip_id"]: row for _, row in df.iterrows()}

    dup_rows = df[df["is_synthetic_duplicate"]]
    for _, dup in dup_rows.iterrows():
        source = by_clip_id.get(dup["source_original_id"])
        if source is None:
            continue

        dup_frame = _middle_frame(config.REPO_ROOT / dup["local_path"])
        src_frame = _middle_frame(config.REPO_ROOT / source["local_path"])
        if dup_frame is None or src_frame is None:
            continue

        h = min(dup_frame.shape[0], src_frame.shape[0])
        dup_r = cv2.resize(dup_frame, (int(dup_frame.shape[1] * h / dup_frame.shape[0]), h))
        src_r = cv2.resize(src_frame, (int(src_frame.shape[1] * h / src_frame.shape[0]), h))
        combined = np.hstack([src_r, dup_r])

        out_path = config.INSPECTION_DIR / f"{dup['clip_id']}_vs_{source['clip_id']}.png"
        cv2.imwrite(str(out_path), combined)
        print(f"  wrote {out_path.name}")


def _middle_frame(path):
    """Reads the middle frame of a video, or None if it can't be read."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return None
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(frame_count // 2 - 1, 0))
    ok, frame = cap.read()
    cap.release()
    return frame if ok else None


def main() -> None:
    """Runs structural + file-integrity checks and writes comparison images.

    Raises:
        SystemExit: With status 1 if any check fails (status 0 on pass), or
            if `data/manifest.csv` is missing.
    """
    if not config.FINAL_MANIFEST_PATH.exists():
        sys.exit("manifest.csv not found — run build_manifest.py first.")

    df = pd.read_csv(config.FINAL_MANIFEST_PATH, dtype={"pexels_id": str})
    df["pexels_id"] = df["pexels_id"].fillna("")
    failures: list[str] = []

    print("== structural checks ==")
    check_structure(df, failures)

    print("\n== file integrity checks ==")
    summaries = check_files(df, failures)

    print("\n== summary ==")
    for s in summaries:
        print(f"  {s['clip_id']:6s} {s['partner_id']:10s} {s['frames']:4d}f  {s['fps']:5.1f}fps  {s['resolution']}")

    print("\n== visual comparisons ==")
    write_comparisons(df)

    print(f"\n{'PASS' if not failures else 'FAIL'}: {len(failures)} failure(s)")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
