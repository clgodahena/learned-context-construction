"""Evaluation dashboard plots.

Renders the two views the proposal calls for: the prequential metric
trajectory over the stream, and the evolving composition of the selected
context (class balance and temporal span). The composition view is the
diagnostic instrument -- it is what reveals *why* a policy outperforms a window.
"""
from __future__ import annotations

from typing import Optional

import matplotlib

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt  # noqa: E402


def plot_run(trajectory: list, composition: list, out_path: str, title: str = "") -> None:
    import numpy as np

    have_comp = bool(composition)
    nrows = 2 if have_comp else 1
    fig, axes = plt.subplots(nrows, 1, figsize=(10, 4 * nrows), squeeze=False)

    # --- metric trajectory ---
    ax = axes[0][0]
    t = [r["t"] for r in trajectory]
    ax.plot(t, [r["windowed_accuracy"] for r in trajectory], label="windowed acc")
    ax.plot(t, [r["balanced_accuracy"] for r in trajectory], label="balanced acc")
    ax.plot(t, [r["kappa_temporal"] for r in trajectory], label="kappa (temporal)")
    ax.set_xlabel("stream position")
    ax.set_ylabel("metric")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc="lower right", fontsize=8)
    ax.set_title(title or "Prequential metrics")

    # --- context composition ---
    if have_comp:
        ax2 = axes[1][0]
        ct = [c["t"] for c in composition]
        ax2.plot(ct, [c["ctx_pos_rate"] for c in composition], color="tab:red",
                 label="context positive rate")
        ax2.set_xlabel("stream position")
        ax2.set_ylabel("context positive rate")
        ax2.set_ylim(-0.02, 1.02)
        ax2.legend(loc="upper right", fontsize=8)
        ax2.set_title("Selected-context composition")

    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
