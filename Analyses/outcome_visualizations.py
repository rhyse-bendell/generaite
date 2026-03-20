from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


GOLD_DARK = "#B08D00"
GOLD_LIGHT = "#E6C65C"
BLACK = "#111111"


def _safe_filename(value: str) -> str:
    keep = []
    for ch in str(value):
        if ch.isalnum() or ch in {"_", "-"}:
            keep.append(ch)
        elif ch.isspace():
            keep.append("_")
    out = "".join(keep).strip("_")
    return out or "unknown"


def _preferred_order(series: pd.Series, preferred: Sequence[str]) -> List[str]:
    observed = [str(x) for x in series.dropna().astype("string").unique().tolist()]
    ordered = [lvl for lvl in preferred if lvl in observed]
    ordered.extend([lvl for lvl in observed if lvl not in ordered])
    return ordered


def _color_map(levels: Sequence[str]) -> Dict[str, str]:
    palette = {
        "NoLabel": GOLD_DARK,
        "Human": BLACK,
        "AI": GOLD_LIGHT,
        "Accurate": GOLD_DARK,
        "Deceptive": BLACK,
    }
    colors: Dict[str, str] = {}
    fallback = [GOLD_DARK, BLACK, GOLD_LIGHT, "#7A6A2F", "#2B2B2B"]
    for idx, lvl in enumerate(levels):
        colors[lvl] = palette.get(lvl, fallback[idx % len(fallback)])
    return colors


def _hatch_map(levels: Sequence[str]) -> Dict[str, str]:
    hatches = ["", "///", "xx", "..", "++", "\\\\"]
    return {lvl: hatches[idx % len(hatches)] for idx, lvl in enumerate(levels)}


def create_outcome_boxplots(
    df: pd.DataFrame,
    outcomes: Iterable[str],
    output_dir: Path,
    study_label: str,
    grouping_cols: Sequence[str],
) -> pd.DataFrame:
    """Create publication-oriented boxplots from row-level processed data."""
    output_dir.mkdir(parents=True, exist_ok=True)
    if len(grouping_cols) not in {1, 2}:
        raise ValueError("grouping_cols must have length 1 or 2")

    g1 = grouping_cols[0]
    g2 = grouping_cols[1] if len(grouping_cols) == 2 else None

    if g1 not in df.columns:
        return pd.DataFrame()
    if g2 and g2 not in df.columns:
        return pd.DataFrame()

    lvl1 = _preferred_order(df[g1], ["NoLabel", "Human", "AI"])
    if not lvl1:
        return pd.DataFrame()

    lvl2 = _preferred_order(df[g2], ["Accurate", "Deceptive", "Original", "Neutral", "Swapped"]) if g2 else []
    if g2 and not lvl2:
        return pd.DataFrame()

    colors = _color_map(lvl1)
    hatches = _hatch_map(lvl2)

    rows: List[Dict[str, str]] = []

    for outcome in outcomes:
        if outcome not in df.columns:
            continue

        plot_df = df[[outcome, g1] + ([g2] if g2 else [])].copy()
        plot_df[outcome] = pd.to_numeric(plot_df[outcome], errors="coerce")
        plot_df = plot_df.dropna(subset=[outcome, g1] + ([g2] if g2 else []))
        if plot_df.empty:
            continue

        fig, ax = plt.subplots(figsize=(11, 6))
        ax.grid(axis="y", color="#D8D8D8", linestyle="--", linewidth=0.8, alpha=0.8)

        if not g2:
            data = [plot_df.loc[plot_df[g1].astype("string") == lvl, outcome].values for lvl in lvl1]
            bp = ax.boxplot(
                data,
                labels=lvl1,
                widths=0.55,
                patch_artist=True,
                medianprops={"color": GOLD_LIGHT, "linewidth": 2},
                whiskerprops={"color": BLACK, "linewidth": 1.5},
                capprops={"color": BLACK, "linewidth": 1.5},
                flierprops={"marker": "o", "markerfacecolor": BLACK, "markeredgecolor": BLACK, "markersize": 3, "alpha": 0.45},
            )
            for i, patch in enumerate(bp["boxes"]):
                patch.set_facecolor(colors[lvl1[i]])
                patch.set_edgecolor(BLACK)
                patch.set_alpha(0.9)

            color_handles = [
                Patch(facecolor=colors[lvl], edgecolor=BLACK, label=str(lvl)) for lvl in lvl1
            ]
            legend1 = ax.legend(handles=color_handles, title=f"Color = {g1}", loc="upper left")
            ax.add_artist(legend1)
            filename = f"{study_label}_{_safe_filename(outcome)}_boxplot_by_{_safe_filename(g1)}.png"
        else:
            n2 = len(lvl2)
            group_gap = n2 + 1
            positions = []
            data = []
            facecolors = []
            box_hatches = []
            for i, l1 in enumerate(lvl1):
                for j, l2 in enumerate(lvl2):
                    vals = plot_df.loc[
                        (plot_df[g1].astype("string") == l1) & (plot_df[g2].astype("string") == l2),
                        outcome,
                    ].values
                    if vals.size == 0:
                        continue
                    data.append(vals)
                    positions.append(i * group_gap + (j + 1))
                    facecolors.append(colors[l1])
                    box_hatches.append(hatches[l2])

            if not data:
                plt.close(fig)
                continue

            bp = ax.boxplot(
                data,
                positions=positions,
                widths=0.75,
                patch_artist=True,
                medianprops={"color": GOLD_LIGHT, "linewidth": 2},
                whiskerprops={"color": BLACK, "linewidth": 1.3},
                capprops={"color": BLACK, "linewidth": 1.3},
                flierprops={"marker": "o", "markerfacecolor": BLACK, "markeredgecolor": BLACK, "markersize": 2.8, "alpha": 0.4},
            )
            for i, patch in enumerate(bp["boxes"]):
                patch.set_facecolor(facecolors[i])
                patch.set_hatch(box_hatches[i])
                patch.set_edgecolor(BLACK)
                patch.set_alpha(0.9)

            centers = [(i * group_gap) + (n2 + 1) / 2 for i in range(len(lvl1))]
            ax.set_xticks(centers)
            ax.set_xticklabels(lvl1)

            color_handles = [Patch(facecolor=colors[lvl], edgecolor=BLACK, label=str(lvl)) for lvl in lvl1]
            hatch_handles = [Patch(facecolor="white", edgecolor=BLACK, hatch=hatches[lvl], label=str(lvl)) for lvl in lvl2]
            legend1 = ax.legend(handles=color_handles, title=f"Color = {g1}", loc="upper left")
            ax.add_artist(legend1)
            ax.legend(handles=hatch_handles, title=f"Pattern = {g2}", loc="upper right")

            filename = (
                f"{study_label}_{_safe_filename(outcome)}_boxplot_by_"
                f"{_safe_filename(g1)}_x_{_safe_filename(g2)}.png"
            )

        ax.set_title(f"{study_label.replace('_', ' ').title()} — {outcome} by {' × '.join(grouping_cols)}", fontsize=13, fontweight="bold")
        ax.set_xlabel(" / ".join(grouping_cols), fontsize=11, fontweight="bold")
        ax.set_ylabel(outcome, fontsize=11, fontweight="bold")
        fig.tight_layout()

        plot_path = output_dir / filename
        fig.savefig(plot_path, dpi=220)
        plt.close(fig)

        rows.append({
            "study": study_label,
            "outcome": outcome,
            "plot_type": "boxplot",
            "grouping": " x ".join(grouping_cols),
            "filename": filename,
            "path": str(plot_path),
        })

    manifest = pd.DataFrame(rows)
    if not manifest.empty:
        manifest.to_csv(output_dir / "visualization_manifest.csv", index=False)
    return manifest
