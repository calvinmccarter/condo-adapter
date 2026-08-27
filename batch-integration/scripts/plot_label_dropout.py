"""Plot label-dropout robustness curves.

Two panels (affine | location-scale). Each curve = composite score normalized to
its 0%-dropout baseline, vs fraction of cell-type labels hidden. Color encodes the
unlabeled-cell handling (excluded vs bucket, the most important contrast); marker
encodes dataset (secondary). Reads the per-run means dumped by the sweep analysis.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

BIO = ["nmi", "ari", "isolated_label_asw", "clisi"]
BATCH = ["asw_batch", "graph_connectivity", "ilisi"]

# color = handling (most salient): slots 1 & 2 of the reference palette
HANDLING_COLOR = {"excluded": "#2a78d6", "bucket": "#eb6834"}
HANDLING_LABEL = {"excluded": "Excluded from fit", "bucket": '"Unknown" bucket'}

# marker = dataset (secondary)
DS_ORDER = ["dkd", "gtex_v9", "immune_cell_atlas", "mouse_pancreas_atlas", "hypomap", "tabula_sapiens"]
DS_MARKER = {"dkd": "o", "gtex_v9": "s", "immune_cell_atlas": "^",
             "mouse_pancreas_atlas": "D", "hypomap": "v", "tabula_sapiens": "P"}
DS_LABEL = {"dkd": "DKD", "gtex_v9": "GTEx", "immune_cell_atlas": "Immune",
            "mouse_pancreas_atlas": "Mouse panc.", "hypomap": "HypoMap",
            "tabula_sapiens": "Tabula Sapiens"}

FRACS = [0.0, 0.25, 0.5, 0.75, 1.0]
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e1e0d9"


def composite(path):
    m = json.load(open(path))
    d = {x["metric"]: x["score"] for x in m if isinstance(x.get("score"), (int, float))}
    if any(k not in d for k in BIO + BATCH):
        return None
    return 0.6 * sum(d[k] for k in BIO) / 4 + 0.4 * sum(d[k] for k in BATCH) / 3


def load_means(results_dir):
    from collections import defaultdict
    rx = re.compile(r"^(location_scale|affine)__([a-z_0-9]+?)__f([0-9.]+)__(bucket|excluded|full)__s(\d+)$")
    acc = defaultdict(list)
    for f in Path(results_dir).glob("*.json"):
        if f.name.endswith(".timing.json"):
            continue
        mm = rx.match(f.stem)
        if not mm:
            continue
        t, ds, fr, h = mm.group(1), mm.group(2), float(mm.group(3)), mm.group(4)
        c = composite(f)
        if c is not None:
            acc[(t, ds, h, fr)].append(c)
    return {k: sum(v) / len(v) for k, v in acc.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir",
                    default="work/label_dropout/results")
    ap.add_argument("--out", default="work/label_dropout/dropout_curves.png")
    ap.add_argument("--exclude", nargs="*", default=["tabula_sapiens"],
                    help="datasets to omit (e.g. still-running ones)")
    args = ap.parse_args()

    mean = load_means(args.results_dir)
    present = [ds for ds in DS_ORDER if ds not in args.exclude
               and any((t, ds, "full", 0.0) in mean for t in ("affine", "location_scale"))]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4), sharey=True)
    for ax, t, title in zip(axes, ["affine", "location_scale"],
                            ["Affine", "Location-scale"]):
        for ds in present:
            base = mean.get((t, ds, "full", 0.0))
            if base is None:
                continue
            for h in ["excluded", "bucket"]:
                xs, ys = [0.0], [1.0]  # anchor at baseline
                for fr in [0.25, 0.5, 0.75, 1.0]:
                    v = mean.get((t, ds, h, fr))
                    if v is not None:
                        xs.append(fr); ys.append(v / base)
                if len(xs) < 2:
                    continue
                ax.plot(xs, ys, color=HANDLING_COLOR[h], marker=DS_MARKER[ds],
                        markersize=6, linewidth=2, markeredgecolor="white",
                        markeredgewidth=0.6, alpha=0.9, zorder=3)
        ax.axhline(1.0, color=GRID, linewidth=1, zorder=1)
        ax.set_title(title, fontsize=12, color=INK, pad=8)
        ax.set_xlabel("Fraction of cell-type labels hidden", fontsize=10, color=MUTED)
        ax.set_xticks(FRACS)
        ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        for spine in ("left", "bottom"):
            ax.spines[spine].set_color("#c3c2b7")
        ax.tick_params(colors=MUTED, labelsize=9)
    axes[0].set_ylabel("Composite (relative to 0% dropout)", fontsize=10, color=MUTED)

    color_leg = [Line2D([0], [0], color=HANDLING_COLOR[h], lw=3, label=HANDLING_LABEL[h])
                 for h in ["excluded", "bucket"]]
    ds_leg = [Line2D([0], [0], color=MUTED, marker=DS_MARKER[ds], lw=0, markersize=7,
                     markeredgecolor="white", markeredgewidth=0.6, label=DS_LABEL[ds])
              for ds in present]
    leg1 = axes[1].legend(handles=color_leg, title="Unlabeled handling",
                          loc="lower left", fontsize=9, title_fontsize=9, frameon=False)
    axes[1].add_artist(leg1)
    axes[0].legend(handles=ds_leg, title="Dataset", loc="lower left",
                   fontsize=8.5, title_fontsize=9, frameon=False, ncol=1)

    fig.suptitle("ConDo integration quality vs. cell-type label availability",
                 fontsize=13, color=INK, y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(args.out, dpi=150, facecolor="#fcfcfb")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
