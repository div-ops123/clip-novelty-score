"""Phase 1 of the scoring pipeline: compute a CLIP embedding per clip.

Loads every clip in `data/manifest.csv`, samples `config.FRAMES_PER_CLIP`
evenly-spaced frames per clip via sequential OpenCV decoding, embeds each
frame with CLIP (Hugging Face `transformers`, `config.CLIP_MODEL_NAME`),
and mean+L2-normalize-pools the per-frame embeddings into one unit vector
per clip. Writes `data/embeddings.npz` so `compute_scores.py` never needs
to load the model. See `docs/JOURNAL.md` for why CLIP and why fixed-count
frame sampling were chosen.

Usage:
    python scripts/embed_clips.py
"""

import sys

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from transformers import CLIPImageProcessor, CLIPModel

import config


def sample_frames(path, n: int = config.FRAMES_PER_CLIP) -> list[np.ndarray]:
    """Sequentially decodes a clip and returns n evenly-spaced BGR frames.

    Never seeks (`cap.set(CAP_PROP_POS_FRAMES, ...)`) — buffers every
    decoded frame in order instead, then selects target indices from the
    actual decoded frame count. This sidesteps the container/codec metadata
    mismatch that made random seeking unreliable on at least one clip
    during dataset construction (see docs/DATA_CONSTRUCTION.md).

    Args:
        path: Path to the clip's video file.
        n: Number of frames to sample.

    Returns:
        A list of up to n BGR frames (numpy arrays), evenly spaced across
        the clip's actual decoded length.

    Raises:
        RuntimeError: If the video can't be opened or no frames decode.
    """
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open {path}")

    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()

    total = len(frames)
    if total == 0:
        raise RuntimeError(f"No frames decoded from {path}")

    if total <= n:
        indices = range(total)
    else:
        indices = sorted({round(x) for x in np.linspace(0, total - 1, n)})

    return [frames[i] for i in indices]


def embed_clip(frames_bgr: list[np.ndarray], model: CLIPModel, processor: CLIPImageProcessor) -> np.ndarray:
    """Embeds a clip's sampled frames with CLIP and mean+L2-normalize-pools them.

    Each frame is normalized to unit length individually before pooling,
    not just once at the end, so that a frame's raw feature magnitude
    (which can shift with a brightness change, one of the four duplicate
    transforms) doesn't unevenly weight its contribution to the pooled
    vector.

    Args:
        frames_bgr: Sampled frames from `sample_frames`, in OpenCV BGR order.
        model: A loaded `CLIPModel` in eval mode.
        processor: The matching `CLIPImageProcessor`.

    Returns:
        A single unit-length embedding vector for the clip.
    """
    pil_frames = [Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)) for f in frames_bgr]
    inputs = processor(images=pil_frames, return_tensors="pt")
    with torch.no_grad():
        # get_image_features returns a BaseModelOutputWithPooling whose
        # pooler_output holds the projected 512-dim image embedding (see
        # CLIPModel.get_image_features source), not a raw tensor.
        feats = model.get_image_features(**inputs).pooler_output.numpy()

    per_frame_unit = feats / np.linalg.norm(feats, axis=1, keepdims=True)
    pooled = per_frame_unit.mean(axis=0)
    pooled = pooled / np.linalg.norm(pooled)
    return pooled.astype(np.float32)


def main() -> None:
    """Embeds every clip in the manifest and writes data/embeddings.npz.

    Raises:
        SystemExit: If `data/manifest.csv` does not exist yet.
    """
    if not config.FINAL_MANIFEST_PATH.exists():
        sys.exit("manifest.csv not found — run the dataset construction pipeline first.")

    df = pd.read_csv(config.FINAL_MANIFEST_PATH)

    print(f"Loading {config.CLIP_MODEL_NAME}...")
    model = CLIPModel.from_pretrained(config.CLIP_MODEL_NAME)
    model.eval()
    processor = CLIPImageProcessor.from_pretrained(config.CLIP_MODEL_NAME)

    clip_ids = []
    vectors = []
    for _, row in df.iterrows():
        path = config.REPO_ROOT / row["local_path"]
        frames = sample_frames(path)
        vector = embed_clip(frames, model, processor)
        clip_ids.append(row["clip_id"])
        vectors.append(vector)
        print(f"  {row['clip_id']}: embedded from {len(frames)} frames")

    np.savez(config.EMBEDDINGS_PATH, clip_ids=np.array(clip_ids), embeddings=np.stack(vectors))
    print(f"\n{len(clip_ids)} clip embeddings ({config.CLIP_EMBEDDING_DIM}-dim) -> {config.EMBEDDINGS_PATH}")


if __name__ == "__main__":
    main()
