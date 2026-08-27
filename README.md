# clip-novelty-score

A small prototype that scores how novel an incoming clip is **relative to the same data partner's own recent submissions** — not against the whole dataset.

## Why

DeepReach's data network runs on partners submitting footage largely independently. Existing described quality mechanisms check whether a clip *looks clean* (technical quality) and whether a batch's *category mix* stays within contracted limits (concentration caps). Neither checks whether a clip is actually new **relative to what that same partner already submitted** — so a partner could repeat near-identical footage and still pass both gates.

This project scores that specific gap: a rolling, per-partner novelty trend, not a per-clip duplicate flag.

Full problem framing: [`docs/PROBLEM.md`](docs/PROBLEM.md)
Design reasoning and tradeoffs considered along the way: [`docs/JOURNAL.md`](docs/JOURNAL.md)

## Important: this is a prototype, not a claim about DeepReach's systems

- Built on a small, deliberately constructed dataset with known ground truth (a few distinct public clips plus intentionally created near-duplicate variants) — this exact scenario doesn't exist pre-labeled anywhere, so it's constructed rather than found.
- No access to DeepReach's real clip data, partner metadata, or submission cadence. Where a real-world detail is unknown (e.g. how often partners actually submit), the assumption used is stated explicitly in `docs/PROBLEM.md`, not presented as fact.
- Not a claim that DeepReach lacks this — only that it's not addressed by the two mechanisms publicly described.

## How to run (reproducing this from a fresh clone)

**Setup**
```
pip install -r requirements.txt
cp .env.example .env      # then fill in PEXELS_API_KEY (free key: pexels.com/api)
```

**Rebuild the dataset.** The manifests (`data/raw_manifest.csv`, `data/partition.json`, `data/augmented_manifest.csv`, `data/manifest.csv`) are committed and pin the exact clips/partner assignments/duplicate recipes used — only the actual video files are gitignored (too large to commit), so a fresh clone needs to re-fetch and re-generate them:
```
python scripts/download_pexels.py fetch   # re-downloads the pool videos raw_manifest.csv already lists
python scripts/augment_duplicates.py      # deterministically regenerates Partner B's near-duplicate clips
python scripts/build_manifest.py          # rebuilds manifest.csv (no-op if nothing changed)
python scripts/verify_dataset.py          # sanity-checks the rebuilt dataset end-to-end
```
This reproduces the *same* dataset already committed. To build a fresh, differently-sampled dataset instead (new random pool from Pexels), run `python scripts/download_pexels.py discover` first, then `python scripts/partition_pool.py`, then the four commands above — see `docs/DATA_CONSTRUCTION.md` for what each phase does and why.

**Run the scoring pipeline** (reads `data/manifest.csv`; safe to rerun any time):
```
python scripts/embed_clips.py       # CLIP-embeds every clip -> data/embeddings.npz (gitignored, derived)
python scripts/compute_scores.py    # -> data/scores.csv
python scripts/plot_validation.py   # -> data/plots/novelty_validation.png
```

## What's here

- `docs/PROBLEM.md` — precise problem definition and scope boundaries
- `docs/JOURNAL.md` — the reasoning trail behind each design decision
- `docs/DATA_CONSTRUCTION.md` — how the validation dataset (`data/manifest.csv`) was built, and why
- `scripts/` — the dataset construction pipeline (download → partition → augment → build manifest → verify) and the scoring pipeline (embed → compute scores → plot)
- `data/scores.csv` — per-clip novelty score and rolling trend for every clip
- `data/plots/novelty_validation.png` — the validation plot: high/flat novelty for Partner A's genuine submissions, a visible drop for Partner B once its planted near-duplicates begin
