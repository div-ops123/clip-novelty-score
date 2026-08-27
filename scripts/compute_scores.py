"""Phase 2 of the scoring pipeline: per-clip novelty and rolling trend.

Pure, deterministic numpy/pandas computation reading `data/embeddings.npz`
and `data/manifest.csv`. For each partner, sorted by `arrival_index`,
computes a per-clip novelty score (1 - max cosine similarity to that
partner's last `config.NOVELTY_COMPARISON_WINDOW` prior submissions) and a
rolling trend (mean of the last `config.TREND_WINDOW` novelty scores).
Writes `data/scores.csv`. Never loads the CLIP model, so it's safe to rerun
instantly while iterating on this file alone.

Note: this script has no notion of `config.WARMUP_THRESHOLD` — cold-start
is a reporting-layer concern applied only in `plot_validation.py`, not a
computation-layer one. See docs/JOURNAL.md "Cold start" and "Proactive vs.
reactive" for why score computation and reporting cadence are kept separate.

Usage:
    python scripts/compute_scores.py
"""

import sys

import numpy as np
import pandas as pd

import config


def compute_partner_scores(vectors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Computes per-clip novelty and rolling trend for one partner's clips.

    Args:
        vectors: Unit-length embeddings for one partner's clips, ordered by
            arrival_index, shape (k, embedding_dim).

    Returns:
        A tuple (novelty, trend), each shape (k,), with NaN at a partner's
        first arrival_index (no prior clips to compare against).
    """
    k = len(vectors)
    novelty = np.full(k, np.nan)
    for i in range(k):
        start = max(0, i - config.NOVELTY_COMPARISON_WINDOW)
        if start == i:
            continue
        sims = vectors[start:i] @ vectors[i]
        novelty[i] = 1.0 - sims.max()

    trend = np.full(k, np.nan)
    for i in range(k):
        start = max(0, i - config.TREND_WINDOW + 1)
        window = novelty[start : i + 1]
        valid = window[~np.isnan(window)]
        if valid.size:
            trend[i] = valid.mean()

    return novelty, trend


def main() -> None:
    """Computes novelty scores and rolling trends for every partner and writes data/scores.csv.

    Raises:
        SystemExit: If `manifest.csv` or `embeddings.npz` is missing.
    """
    if not config.FINAL_MANIFEST_PATH.exists():
        sys.exit("manifest.csv not found — run the dataset construction pipeline first.")
    if not config.EMBEDDINGS_PATH.exists():
        sys.exit("embeddings.npz not found — run embed_clips.py first.")

    df = pd.read_csv(config.FINAL_MANIFEST_PATH)
    emb = np.load(config.EMBEDDINGS_PATH, allow_pickle=True)
    clip_to_vec = dict(zip(emb["clip_ids"], emb["embeddings"]))

    rows = []
    for partner_id, group in df.groupby("partner_id"):
        group = group.sort_values("arrival_index").reset_index(drop=True)
        vectors = np.stack([clip_to_vec[cid] for cid in group["clip_id"]])
        novelty, trend = compute_partner_scores(vectors)

        for i, row in group.iterrows():
            rows.append(
                {
                    "clip_id": row["clip_id"],
                    "partner_id": partner_id,
                    "arrival_index": row["arrival_index"],
                    "novelty_score": novelty[i],
                    "rolling_trend": trend[i],
                    "is_synthetic_duplicate": row["is_synthetic_duplicate"],
                }
            )

    scores = pd.DataFrame(rows).sort_values(["partner_id", "arrival_index"])
    scores.to_csv(config.SCORES_PATH, index=False)
    print(f"{len(scores)} scored clips -> {config.SCORES_PATH}")


if __name__ == "__main__":
    main()
