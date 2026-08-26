# Data Construction — Reference

How the validation dataset in `data/` was built, and why. For the problem this
dataset exists to validate, see `PROBLEM.md`; for the design reasoning behind
the scoring approach itself, see `JOURNAL.md`. This document covers only the
dataset construction pipeline in `scripts/`.

## Why this dataset had to be built, not found

The failure mode under test — a partner submitting near-duplicate footage
over time — doesn't exist pre-labeled in any public dataset (see JOURNAL.md,
"The data problem"). To validate a novelty score, you need ground truth: for
each clip, whether it actually is a near-duplicate of a specific earlier
clip. That only exists if you construct the duplicates yourself.

## Dataset design

Two synthetic partners, each with a known, deliberately constructed
submission history:

| Partner | Clips | Composition | Expected novelty signal |
|---|---|---|---|
| A (control) | 18 | All genuinely distinct POV clips, different task categories | Stays high/flat the entire sequence — this is genuine diversity, never redundancy |
| B (farming case) | 16 | 9 genuinely distinct clips, then 7 near-duplicate variants of one of its own earlier clips | High for the first 9, then a visible drop once duplicates start |

Both partners' clip counts fall inside 15–20, the cold-start / warm-up
threshold stated in JOURNAL.md — below that count a partner's trend isn't
evaluated at all, so both need to actually clear the threshold for the
validation plot to show anything.

Partner A exists specifically to rule out false positives: without it, a
score that flags *all* redundancy-looking similarity — including the
legitimate kind (two different environments producing similar footage of a
similar task) — would look identical to a score that correctly flags only
within-partner farming. Partner A never repeats anything, so its trend must
stay flat for the score to be considered correct.

## Why Pexels stock footage instead of a research egocentric dataset

Real egocentric datasets (Ego4D, EPIC-KITCHENS) require license applications
and multi-GB downloads for access that isn't actually needed here — the
realism of the base footage doesn't matter for validating the score
computation, only that the "distinct" clips are genuinely visually distinct
and the "duplicate" clips are genuinely, verifiably derived from a specific
source. Pexels' free video API provides real POV/first-person footage of
physical tasks with an instant API key and no licensing friction, which is
sufficient for that purpose.

## Pipeline

Five scripts, run in order, each consuming only the outputs of earlier
phases so any later phase can be rerun independently:

```
download_pexels.py  -->  partition_pool.py  -->  augment_duplicates.py  -->  build_manifest.py  -->  verify_dataset.py
   (phase 1)              (phase 1.5)              (phase 2)                  (phase 3)               (phase 4)
```

### Phase 1 — `download_pexels.py discover`

Searches Pexels using the task-category terms in `config.SEARCH_TERMS`
("pov hands tool", "first person cooking", "pov mechanic", etc.), skipping
clips over `config.MAX_DURATION_SEC` and picking the smallest available
`sd` mp4 rendition per clip (keeps downloads light; resolution doesn't
matter for the future CLIP embedding step, which will downsample further
anyway).

**`config.MAX_PER_TERM` caps clips taken from any single search term.**
This was added after the first run pulled all 28 clips from one term
("pov hands tool") before the pool size was reached — technically valid
(28 distinct Pexels videos) but a weak "genuine diversity" story, since the
resulting pool ended up narrowly one flavor of hand-and-tool footage rather
than spanning task categories. Capping per term forces the search to spread
across categories before any one term is exhausted.

Writes `data/raw/{pexels_id}.mp4` + `data/raw_manifest.csv` (one row per
pool clip: pexels_id, search_term, source_url, rendition metadata,
local_path). Resumable — rerunning skips clips already in the manifest.

`download_pexels.py fetch` re-downloads any manifest clip missing from disk
(e.g. after cloning the repo, since video files are gitignored) by looking
up a fresh rendition link — the original CDN link can expire. A clip Pexels
has since removed is skipped with a warning rather than aborting the run.

### Phase 1.5 — `partition_pool.py`

Deterministically shuffles the pool (fixed seed, `config.PARTITION_SEED`)
and splits it: first `PARTNER_A_COUNT` clips to Partner A, next
`PARTNER_B_GENUINE_COUNT` to Partner B's genuine submissions. The first of
Partner B's genuine clips becomes `duplication_source` — the clip that
phase 2 will clone.

Writes `data/partition.json` as plain, hand-editable JSON on purpose: if a
specific auto-picked clip turns out to be a poor example on inspection (bad
framing, wrong content, or — as happened once during construction — a file
whose container metadata OpenCV can't reliably seek to the end of), you can
swap its Pexels ID for another id from the pool that wasn't assigned to
either partner, and rerun `build_manifest.py`. No code change needed.

### Phase 2 — `augment_duplicates.py`

Loads `duplication_source`'s frames into memory (clips are short, so this
is simpler than streaming) and generates `PARTNER_B_DUP_COUNT` (7) variants
using deterministic combinations of four transforms, applied in a fixed
order (crop → brightness → flip → speed) regardless of which subset a given
recipe uses:

| Transform | Implementation | Constant |
|---|---|---|
| Crop | Center-crop 10% off each edge, resize back to original dimensions | `CROP_PCT = 0.10` |
| Brightness | `cv2.convertScaleAbs(alpha, beta)` gain + offset | `BRIGHTNESS_ALPHA = 1.2`, `BRIGHTNESS_BETA = 25` |
| Flip | Horizontal mirror | — |
| Speed | Resample frames at 1.15x — output frame `i` pulls source frame `round(i * 1.15)`, written at the same fps | `SPEED_FACTOR = 1.15` |

Recipes are built from `itertools.combinations` over the four transforms,
ordered by subset size (all 4 singles first, then pairs) — deterministic,
not randomized, so the dataset is exactly reproducible from the same pool.
Using combinations rather than 7 copies of one transform also better
represents a real quota-farming partner, who would apply cheap edits
inconsistently rather than resubmitting byte-identical files.

**Why OpenCV instead of moviepy:** OpenCV's Windows wheel bundles its own
FFmpeg-based video I/O, so `cv2.VideoCapture`/`VideoWriter` read and write
mp4 with no separate system FFmpeg install. moviepy doesn't actually avoid
that dependency either — it pulls in `imageio-ffmpeg`, which downloads a
static ffmpeg binary at install time — while adding a heavier API on top.
OpenCV is also almost certainly what the future scoring script will use for
frame sampling, so this keeps both parts of the codebase on one library.
**Known limitation:** OpenCV's `VideoWriter` drops audio entirely. This is
harmless here — CLIP embeddings are visual-only — but is a real limitation
if this pipeline is ever reused somewhere audio matters.

Writes `data/augmented/B_dup{NN}_{recipe}.mp4` + `data/augmented_manifest.csv`
(one row per variant: which source it came from, which transforms were
applied, and their exact numeric parameters).

### Phase 3 — `build_manifest.py`

Pure merge of the three earlier outputs into `data/manifest.csv` — the
single file the future scoring pipeline consumes. Assigns stable
`clip_id`s (`A01..A18`, `B01..B16`) and per-partner `arrival_index` (1-based
submission order). This script does no I/O beyond reading its three inputs
and writing one output, so it's always safe to rerun, including right after
hand-editing `partition.json`.

**Manifest schema** (`data/manifest.csv`):

| Column | Meaning |
|---|---|
| `clip_id` | Stable id, e.g. `A01`, `B14` |
| `partner_id` | `partner_a` or `partner_b` |
| `arrival_index` | 1-based submission order within that partner |
| `source_original_id` | The `clip_id` this clip is a variant of (itself, if genuine) — the ground-truth join key |
| `is_synthetic_duplicate` | `True` for phase-2-generated variants, `False` for real downloaded clips |
| `local_path` | Repo-relative path to the video file |
| `pexels_id`, `search_term`, `source_url` | Attribution metadata (blank for duplicates — resolved via `source_original_id` instead of copied) |
| `augmentation_type` | e.g. `crop+flip` (blank for genuine clips) |

### Phase 4 — `verify_dataset.py`

Checks the manifest actually satisfies its own ground-truth claims before
anything downstream trusts it:

- **Structural**: clip counts per partner in [15, 20]; `arrival_index`
  contiguous from 1; Partner A has zero duplicate-flagged rows; Partner B's
  duplicates all arrive after its genuine clips and all reference one
  `source_original_id`; no Pexels ID reused across rows.
- **File integrity**: every `local_path` exists and opens in OpenCV, with
  both its first and last frame readable (catches truncated downloads or
  incomplete writer flushes that a file-exists check alone would miss).
- **Visual**: writes `data/inspection/{dup_id}_vs_{source_id}.png` — a
  side-by-side middle-frame comparison for every duplicate, so a human can
  actually look and confirm the augmentation resembles its source rather
  than trusting the label alone.

Exits non-zero on any failure, so it can gate later automation even without
CI configured yet.

## Reproducing the dataset

```bash
pip install -r requirements.txt
# put a Pexels API key in .env (see .env.example)
python scripts/download_pexels.py discover
python scripts/partition_pool.py
python scripts/augment_duplicates.py
python scripts/build_manifest.py
python scripts/verify_dataset.py
```

Video files (`data/raw/*.mp4`, `data/augmented/*.mp4`) are gitignored — they
regenerate deterministically from the committed `raw_manifest.csv` +
`partition.json` given the same Pexels pool. If a specific Pexels video is
later removed from Pexels entirely, `download_pexels.py fetch` will skip it
with a warning rather than failing the whole run; that one row's clip would
need to be replaced by hand.

## What's explicitly out of scope here

This document covers dataset construction only. It says nothing about how
novelty is actually computed from these clips (embedding model, rolling
window, warm-up gating) — that's the separate scoring pipeline that
consumes `data/manifest.csv`, not yet built as of this writing.
