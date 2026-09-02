"""Generate vector diagrams for representative Kronecker-delta contractions.

Each column represents one of the four membership regions determined by two
subsystems A and A'.  Equal fills within a column identify replica indices
that are constrained to agree at every qubit in that region.  The labels below
the grid list the same equality classes algebraically, so the figures remain
unambiguous in grayscale.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


ROOT = Path(__file__).resolve().parents[1]
FIGURE_DIR = ROOT / "figures"

PINK = "#d99aa3"
BLUE = "#94b8d7"
YELLOW = "#e8d28d"
GRAY = "#a8a8a8"
WHITE = "#ffffff"

REGION_WIDTHS = (1.35, 1.05, 1.35, 1.35)
REGION_LABELS = (
    r"$A\cap \bar A'$",
    r"$A\cap A'$",
    r"$\bar A\cap A'$",
    r"$\bar A\cap\bar A'$",
)


def _bar(ax: plt.Axes, x0: float, x1: float, y: float, label: str) -> None:
    """Draw a simple subsystem span with end ticks and a centered label."""

    ax.plot([x0, x1], [y, y], color="black", linewidth=0.9, clip_on=False)
    ax.plot([x0, x0], [y - 0.06, y + 0.06], color="black", linewidth=0.9)
    ax.plot([x1, x1], [y - 0.06, y + 0.06], color="black", linewidth=0.9)
    ax.text((x0 + x1) / 2, y + 0.09, label, ha="center", va="bottom")


def _draw_diagram(
    filename: str,
    colors_by_region: tuple[tuple[str, ...], ...],
    equality_labels: tuple[str, ...],
) -> None:
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 9,
            "mathtext.fontset": "dejavuserif",
            "pdf.fonttype": 42,
        }
    )

    edges = [0.0]
    for width in REGION_WIDTHS:
        edges.append(edges[-1] + width)
    total_width = edges[-1]

    figure, ax = plt.subplots(figsize=(6.2, 4.35), constrained_layout=True)
    for region, (x0, x1) in enumerate(zip(edges[:-1], edges[1:])):
        for replica in range(4):
            y0 = 3 - replica
            ax.add_patch(
                Rectangle(
                    (x0, y0),
                    x1 - x0,
                    1,
                    facecolor=colors_by_region[region][replica],
                    edgecolor="black",
                    linewidth=0.65,
                )
            )
        ax.text(
            (x0 + x1) / 2,
            4.10,
            REGION_LABELS[region],
            ha="center",
            va="bottom",
            fontsize=8.4,
        )
        ax.text(
            (x0 + x1) / 2,
            -0.16,
            equality_labels[region],
            ha="center",
            va="top",
            fontsize=7.8,
        )

    for replica in range(4):
        ax.text(
            total_width + 0.12,
            3.5 - replica,
            rf"$u_{replica + 1}$",
            ha="left",
            va="center",
        )

    split = edges[2]
    _bar(ax, edges[0], split, 4.70, r"$A$")
    _bar(ax, split, edges[4], 4.70, r"$\bar A$")

    ax.text(
        total_width / 2,
        -0.62,
        r"braces group equal replica labels; $\mid$ separates independent classes",
        ha="center",
        va="top",
        fontsize=7.5,
        color="0.25",
    )

    _bar(ax, edges[0], edges[1], -1.22, r"$\bar A'$" )
    _bar(ax, edges[1], edges[3], -1.22, r"$A'$" )
    _bar(ax, edges[3], edges[4], -1.22, r"$\bar A'$" )

    ax.set_xlim(-0.04, total_width + 0.52)
    ax.set_ylim(-1.50, 5.02)
    ax.set_aspect("equal")
    ax.axis("off")

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    figure.savefig(FIGURE_DIR / f"{filename}.pdf", bbox_inches="tight")
    figure.savefig(FIGURE_DIR / f"{filename}.png", dpi=240, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    _draw_diagram(
        "delta_contraction_three_replica",
        (
            (PINK, PINK, PINK, WHITE),
            (PINK, PINK, PINK, PINK),
            (WHITE, BLUE, BLUE, BLUE),
            (WHITE, BLUE, BLUE, GRAY),
        ),
        (
            r"$\{1,2,3\}\mid\{4\}$",
            r"$\{1,2,3,4\}$",
            r"$\{1\}\mid\{2,3,4\}$",
            r"$\{1\}\mid\{2,3\}\mid\{4\}$",
        ),
    )
    _draw_diagram(
        "delta_contraction_crossed",
        (
            (PINK, PINK, PINK, PINK),
            (PINK, YELLOW, YELLOW, PINK),
            (BLUE, BLUE, BLUE, BLUE),
            (BLUE, GRAY, BLUE, GRAY),
        ),
        (
            r"$\{1,2,3,4\}$",
            r"$\{1,4\}\mid\{2,3\}$",
            r"$\{1,2,3,4\}$",
            r"$\{1,3\}\mid\{2,4\}$",
        ),
    )


if __name__ == "__main__":
    main()
