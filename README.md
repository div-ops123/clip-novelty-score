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

## What's here

- `docs/PROBLEM.md` — precise problem definition and scope boundaries
- `docs/JOURNAL.md` — the reasoning trail behind each design decision
- `docs/DATA_CONSTRUCTION.md` — how the validation dataset (`data/manifest.csv`) was built, and why
- `scripts/` — the dataset construction pipeline (download → partition → augment → build manifest → verify)
- *(score computation and validation experiment — in progress)*
