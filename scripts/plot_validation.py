"""Phase 3 of the scoring pipeline: the validation plot.

Reads `data/scores.csv` and produces `data/plots/novelty_validation.png` —
the one validation artifact proving the score behaves as claimed: high,
stable novelty for Partner A's genuinely distinct clips, and a visible drop
for Partner B once its planted near-duplicates begin arriving at
arrival_index 10. Two panels (one per partner, never one overlaid axis) so
the plot never implies a cross-partner comparison, which is explicitly out
of scope (see docs/PROBLEM.md).

Usage:
    python scripts/plot_validation.py
"""

import sys

import matplotlib.pyplot as plt
import pandas as pd

import config

# Light-mode palette (see dataviz skill references/palette.md)
SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"
SERIES_NOVELTY = "#2a78d6"  # categorical slot 1 (blue)
SERIES_TREND = "#eb6834"  # categorical slot 2 (orange)
STATUS_CRITICAL = "#d03b3b"  # reserved status color, used only as a low-opacity band


def plot_partner(ax, group: pd.DataFrame, title: str) -> None:
    """Draws one partner's novelty/trend panel onto the given axes.

    Args:
        ax: Matplotlib axes to draw on.
        group: This partner's rows from scores.csv, sorted by arrival_index.
        title: Panel title (partner label).
    """
    ax.set_facecolor(SURFACE)
    x = group["arrival_index"]

    dup = group[group["is_synthetic_duplicate"]]
    if not dup.empty:
        ax.axvspan(dup["arrival_index"].min() - 0.5, dup["arrival_index"].max() + 0.5, color=STATUS_CRITICAL, alpha=0.08, zorder=0)
        ax.text(
            dup["arrival_index"].min(),
            1.04,
            "planted duplicates",
            color=STATUS_CRITICAL,
            fontsize=8,
            ha="left",
            va="bottom",
        )

    if config.WARMUP_THRESHOLD <= x.max():
        ax.axvline(config.WARMUP_THRESHOLD, color=INK_MUTED, linestyle="--", linewidth=1, zorder=1)
        ax.text(
            config.WARMUP_THRESHOLD + 0.15,
            1.04,
            f"warmup threshold ({config.WARMUP_THRESHOLD})",
            color=INK_MUTED,
            fontsize=7,
            rotation=90,
            va="bottom",
            ha="left",
        )

    ax.plot(
        x,
        group["novelty_score"],
        color=SERIES_NOVELTY,
        linewidth=2,
        marker="o",
        markersize=5,
        markeredgecolor=SURFACE,
        markeredgewidth=1.2,
        solid_capstyle="round",
        solid_joinstyle="round",
        label="Per-clip novelty",
        zorder=2,
    )
    ax.plot(
        x,
        group["rolling_trend"],
        color=SERIES_TREND,
        linewidth=2,
        marker="o",
        markersize=5,
        markeredgecolor=SURFACE,
        markeredgewidth=1.2,
        solid_capstyle="round",
        solid_joinstyle="round",
        label="Rolling trend",
        zorder=3,
    )

    ax.set_title(title, color=INK_PRIMARY, fontsize=12, loc="left")
    ax.set_xlabel("Arrival index", color=INK_SECONDARY, fontsize=9)
    ax.set_ylabel("Novelty score", color=INK_SECONDARY, fontsize=9)
    ax.set_ylim(-0.05, 1.1)
    ax.set_xticks(range(1, int(x.max()) + 1))
    ax.tick_params(colors=INK_MUTED, labelsize=8)
    ax.grid(True, color=GRIDLINE, linewidth=1, zorder=-1)
    for spine in ax.spines.values():
        spine.set_color(BASELINE)


def main() -> None:
    """Builds the two-panel validation plot and writes it to disk.

    Raises:
        SystemExit: If `data/scores.csv` does not exist yet.
    """
    if not config.SCORES_PATH.exists():
        sys.exit("scores.csv not found — run compute_scores.py first.")

    scores = pd.read_csv(config.SCORES_PATH)

    config.PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(9, 9), facecolor=SURFACE)

    partner_titles = {
        "partner_a": "Partner A — control (genuinely distinct submissions)",
        "partner_b": "Partner B — quota-farming case (planted near-duplicates)",
    }
    for ax, (partner_id, group) in zip(axes, scores.groupby("partner_id")):
        group = group.sort_values("arrival_index")
        plot_partner(ax, group, partner_titles.get(partner_id, partner_id))

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, labelcolor=INK_SECONDARY, bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout()
    fig.savefig(config.VALIDATION_PLOT_PATH, dpi=150, facecolor=SURFACE, bbox_inches="tight")
    print(f"-> {config.VALIDATION_PLOT_PATH}")


if __name__ == "__main__":
    main()
