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

## The data problem

This is the part I got stuck on. This exact scenario — a partner submitting near-duplicate footage over time — isn't something I could go find in a public dataset. It doesn't exist pre-labeled anywhere.

So instead of searching for it, I decided to construct it deliberately, with known ground truth: pull a small set of genuinely different public egocentric clips, then deliberately generate near-duplicate variants of a few of them myself (crop, brightness shift, slight speed change, flip). Assign synthetic partner IDs and synthetic arrival order. Because I made the duplicates myself, I know exactly which clips are supposed to score low novelty against which — so I can actually validate the score instead of just asserting it works.

## What I'm actually shipping

Not a pipeline — I want to be honest about that. It's:
1. A memo — the failure mode, why it's invisible to EgoView and the concentration cap, the within-partner design choice, the cold-start policy, and every assumption labeled clearly as an assumption.
2. The score computation itself — load clips, sample frames, embed with an off-the-shelf encoder, maintain a rolling window per simulated partner, respect the warm-up threshold.
3. One validation plot — proving the score does what I claim: high novelty for distinct clips, visible drop for the duplicates I planted on purpose.
