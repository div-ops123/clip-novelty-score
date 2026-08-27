# Journal — clip-novelty-score

## Why this project

DeepReach's whole moat is access and diversity of real-world data collection at scale. So I asked myself: where could something go wrong that's specific to *how* they collect, not just what they collect?

They already have two gates I could see from their public materials:
1. EgoView — checks if a clip is technically clean (sharp, stable, hands visible).
2. Concentration caps — checks if a category (environment type) is over-represented in a batch.

Neither one asks the question I kept coming back to: what if a partner submits clips that pass both gates, but are basically the same five seconds repeated? Right tags, clean footage, but no new information for the model. That's not a quality problem and not a category problem. It's a redundancy problem, and it's invisible to both of their described gates.

I'm calling this quota-farming risk. It's the kind of failure that gets worse, not better, as DeepReach scales — more partners, more countries, less direct oversight per partner. That's the exact dimension they're trying to grow.

## Defining what "novelty" actually means here

First instinct was: compare every new clip against the whole dataset. Wrong move. Two different mechanics tightening a bolt in two different countries will genuinely look similar in embedding space — that's not farming, that's the diversity DeepReach is selling working as intended. A dataset-wide similarity check would punish the exact thing they want more of.

So the score has to be relative, not global: does this clip look like *this same partner's* own recent submissions? Not "does this look like other bolt-tightening clips out there." Comparing within-partner only is what keeps genuine category overlap from getting flagged.

## No hard duplicate/not-duplicate cutoff

I didn't want a binary "duplicate: yes/no" rule, because a couple of genuine repeats shouldn't be punished — some tasks just look similar by nature, and DeepReach still benefits from that footage. So instead of a cutoff, it's a continuous novelty score per clip, rolled up into a trend per partner (mean novelty over their last N submissions). Occasional repeats barely move a rolling average. A sustained drop is the actual signal. That gives the grace I wanted without needing to hand-tune a magic similarity threshold.

## Cold start

A partner with 2 submitted clips has no meaningful trend yet — averaging 2 numbers is noise, not a pattern. So there's a warm-up policy: a partner's novelty trend isn't evaluated until they've hit some minimum number of clips (I used 15–20 as a placeholder). Below that, the system just accumulates silently. Normal behavior for a new partner is silence by design, not a false "all clear."

## Proactive vs. reactive — and where I was wrong

I originally assumed real-time meant more infra and batch-based meant cheaper and more accurate math. Neither is true. The score itself (embedding one clip) costs the same whether you compute it the second a clip lands or two weeks later. What actually changes between the two designs is detection latency, not accuracy — the math behind a rolling average doesn't care about cadence, only window size.

So I split the design into two separate decisions:
- Score computation: incremental, per clip, cheap, can piggyback on whatever step already extracts frames for EgoView.
- Reporting cadence: separate call, driven by what's useful to a human, not compute cost. Flagging every single clip is noise. A daily/weekly digest per partner is the right granularity — early enough to catch drift before a batch ships, coarse enough not to spam.

I don't actually know DeepReach's real submission cadence (batched uploads vs. continuous vs. daily syncs), so I'm stating that as an assumption rather than pretending I know their system.

## compute_scores.py is batch, not incremental

Worth being explicit about a gap between the design above and what I actually built. `compute_partner_scores` loops `for i in range(k)` over a partner's *entire* clip history every time it runs — starting at their first-ever clip, not from a saved checkpoint. Every novelty and trend value gets recomputed from scratch on every run, even ones that haven't changed since the last one.

That's the right tradeoff for this prototype specifically: it's validating against a small, fixed, already-complete 34-clip dataset, not a live stream, and the whole computation is cheap numpy over 512-dim vectors — recomputing everything costs nothing. But it means this script does not implement the "incremental, per clip" production design described above. A real deployed version would score just the newest clip against its already-known last-10-vector window and update the trend, not replay a partner's whole history on every new submission. If I ever build that version, it's a different script with different state-handling, not a small edit to this one.

## Limitation: novelty score doesn't identify which prior clip matched

`novelty[i] = 1 - sims.max()` only keeps the *value* of the closest match in the comparison window — it never records *which* prior clip produced it. So `scores.csv` can say a clip scored 0.005 novelty (clearly matched something closely), but not which specific prior submission it matched. For the validation plot that's enough, since the point there is proving the pattern exists at all. For a real flagging workflow it isn't: a human investigating a flagged clip would want to know exactly which prior submission it's nearly identical to, not just that one exists somewhere in the last 10.

Fix, if/when this needs to be actionable: track `sims.argmax()` alongside `sims.max()`, map that index back to `start + argmax` to get the matching clip's actual position, and look up its `clip_id` to add a `most_similar_to` column to `scores.csv`. Not implemented — noting it here as a known gap, not a silent one.

## The data problem

This is the part I got stuck on. This exact scenario — a partner submitting near-duplicate footage over time — isn't something I could go find in a public dataset. It doesn't exist pre-labeled anywhere.

So instead of searching for it, I decided to construct it deliberately, with known ground truth: pull a small set of genuinely different public egocentric clips, then deliberately generate near-duplicate variants of a few of them myself (crop, brightness shift, slight speed change, flip). Assign synthetic partner IDs and synthetic arrival order. Because I made the duplicates myself, I know exactly which clips are supposed to score low novelty against which — so I can actually validate the score instead of just asserting it works.

## Why CLIP, not a video-native model

A video file isn't a tensor a model can consume — it's compressed bytes (a codec-encoded sequence of frames) that has to be decoded first. So no matter what encoder I pick, step one is always: decode the clip into individual frames, each just a plain pixel array.

The real choice is what happens after that. Two options:

1. Sample a handful of frames and run each through an **image** encoder (CLIP), then pool the per-frame vectors into one embedding for the clip.
2. Feed a whole sequence of frames into a **video-native** encoder (VideoMAE, X-CLIP, etc.) that's trained to understand motion and temporal order, producing one embedding directly.

I went with option 1. The reasoning: quota-farming, as I've scoped it, is fundamentally a *content/appearance* similarity question — "does this clip show substantially the same thing this partner already submitted?" — not a *motion* question. The near-duplicates I'm constructing to test this (crop, brightness shift, flip, slight speed change) all preserve nearly all of the original visual content; none of them are attacks that only show up in motion patterns. A frame-level appearance embedding catches exactly the signal this problem is about, without paying for temporal modeling I don't have a use for yet.

CLIP specifically because it's pretrained on a huge corpus of image-text pairs and ships as an off-the-shelf general-purpose semantic image embedding — no training or fine-tuning required to get a space where visually/semantically similar frames land close together. That's exactly the property this score needs, and it's a well-established, easy-to-run choice.

Stated assumption, explicitly: this assumes a farming partner is reusing visual content, not manufacturing novel-looking motion around otherwise-repeated footage. If that assumption turns out to be wrong — someone finds a way to make near-duplicate footage read as motion-diverse — that's a real blind spot in this design, not something the current approach happens to cover.

## Frame sampling: fixed count, not fixed rate

Clips vary in length (a few seconds to 30s), so sampling at a fixed rate (e.g. 1fps) would give long clips far more frames — and more say in the pooled embedding — than short ones, for no reason related to content. Sampling a fixed count (8) evenly across each clip's duration instead gives every clip the same representational budget regardless of length. 8 is a round default, not a tuned value — similar in spirit to the 15–20 warm-up threshold.

## Per-clip novelty: max similarity, not centroid similarity

Comparing a new clip to a partner's recent history means collapsing "similarity to each of the last W clips" into one number. Two ways to do that: similarity to the *centroid* (average) of those W clips, or the *max* similarity to any single one of them.

I went with max. The reasoning: this system already has a smoothing layer built in — the rolling trend (mean novelty over the last N submissions, see "No hard duplicate/not-duplicate cutoff" above) — which is where the tolerance for an occasional one-off repeat should live, since that's exactly why I built it. Smoothing again at the per-clip level, via centroid, would stack a second layer of forgiveness on top of the first, and centroid similarity specifically risks diluting a real near-duplicate: a clone of one specific prior clip doesn't necessarily look close to the *average* of several visually unrelated prior clips. Max similarity asks the sharper, more direct question — "does this match any one specific thing I've submitted before" — which is what a near-duplicate actually is. The trend layer still absorbs a single low score without overreacting; it just does it once, at the right layer, instead of twice.

## What I'm actually shipping

Not a pipeline — I want to be honest about that. It's:
1. A memo — the failure mode, why it's invisible to EgoView and the concentration cap, the within-partner design choice, the cold-start policy, and every assumption labeled clearly as an assumption.
2. The score computation itself — load clips, sample frames, embed with an off-the-shelf encoder, maintain a rolling window per simulated partner, respect the warm-up threshold.
3. One validation plot — proving the score does what I claim: high novelty for distinct clips, visible drop for the duplicates I planted on purpose.
