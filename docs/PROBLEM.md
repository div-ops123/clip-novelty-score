# Problem Definition — Clip Novelty Scoring

## Context

DeepReach's core moat is access and diversity of real-world data collection at scale: a network of local data partners, each equipped with a wearable capture device, submitting first-person footage of physical work across many independent environments. The business value of this network depends on one property holding at scale: that footage volume translates into genuine coverage of real-world variation, not repeated instances of the same narrow activity.

Two quality mechanisms are publicly described as part of this pipeline:

1. **Per-clip technical quality gating** (e.g. EgoView) — evaluates whether an individual clip meets a visual quality standard: hand visibility, sharpness, camera stability, framing.
2. **Per-category concentration limits** — contractual acceptance specs that cap how much of a delivered batch can originate from any single environment type, preventing over-representation of one setting.

Both mechanisms operate on **metadata or per-clip visual properties**. Neither evaluates a clip's content **relative to what the same partner has already submitted**.

## The gap

A data partner is paid to produce and submit footage, typically under limited direct supervision, since the network model depends on partners operating independently rather than under centralized oversight. This creates a structural incentive: a partner under time or volume pressure can satisfy submission requirements by recording minor variations of the same task repeatedly, rather than capturing genuinely new activity.

Such submissions can pass both existing gates:
- Each clip can independently meet EgoView's technical quality bar (sharp, stable, hands visible).
- Each clip can carry a correct, permitted environment/category tag, so it does not violate concentration limits.

Neither gate inspects whether a clip is **substantively new relative to that partner's own prior output**. This is a distinct failure mode from either gate's stated purpose, not a stricter version of one of them.

## Why this matters

- **It is invisible at the point of failure.** A rejected batch under the concentration cap is a known, bounded, and presumably already-priced cost. Repetitive-but-compliant footage is not rejected; it ships as if it were genuinely diverse data.
- **It undermines the core claim being sold, not just one delivery.** DeepReach's value proposition to frontier labs and robotics companies rests on diversity producing generalization. Undetected redundancy degrades that property silently, with the cost surfacing later, at the level of model performance, rather than at the point of data intake.
- **It scales adversely with growth.** The stated trajectory is toward far more partners and far less centralized oversight per partner. Whatever rate of this behavior exists today is more likely to increase, not decrease, as the network grows in exactly the dimension the risk depends on.

## Problem statement

Given a stream of clips submitted by a data partner over time, determine whether that partner's submissions are trending toward redundancy — i.e., whether recent clips are substantively similar to that same partner's own recent prior submissions — early enough to allow intervention before the pattern compounds across many clips or is packaged into a delivered batch.

## Explicit scope boundaries

**In scope:**
- Detecting redundancy *within a single partner's own submission history* over time.
- A continuous, trend-based signal (rolling novelty over a partner's recent submissions), not a binary per-clip duplicate flag.
- A stated cold-start policy: a partner's trend is not evaluated below a minimum submission count, since a trend requires a baseline.

**Out of scope, and deliberately so:**
- Cross-partner similarity. Two different partners producing similar footage of the same task is expected diversity, not a defect — this is not being flagged.
- Replacing or duplicating EgoView's technical quality function.
- Replacing or duplicating the concentration-cap / category-diversity accounting function.
- Any claim about DeepReach's actual internal tooling or submission cadence — none of this document assumes those systems are absent, only that this specific failure mode is not addressed by the two mechanisms publicly described.

## Working assumptions (stated explicitly, not verified)

- Partner submission cadence is unknown; this work assumes irregular, incremental submission per partner rather than a specific batch schedule.
- No access to DeepReach's real clip metadata or footage; all validation is performed on a small, deliberately constructed dataset with known ground truth (see `docs/JOURNAL.md` for the reasoning behind this choice).