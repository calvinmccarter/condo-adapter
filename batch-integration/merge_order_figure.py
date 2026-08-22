"""Render the merge-order ablation figure for the condo-scib paper.

Two panels (affine | location-scale), datasets on the y-axis, all-methods
composite on the x-axis; each of the 7 ranking strategies is a colored dot,
the deployed cell-type-silhouette default drawn as an emphasized star, and a
per-dataset tick marks scANVI as an embedding reference. Colors are the
paper's 7-color palette, one per strategy.

Run with the clonebo env python (has matplotlib + pyyaml):
  ~/miniconda3/envs/clonebo/bin/python batch-integration/merge_order_figure.py
"""
import sys
from pathlib import Path

REPO = Path("/Users/cmccarter/sandbox/condo-adapter")
sys.path.insert(0, str(REPO / "batch-integration"))
import leaderboard as L

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

OUT_PDF = Path("/Users/cmccarter/sandbox/condo-scib/figures/merge_order.pdf")
OUT_PNG = Path("/private/tmp/claude-501/-Users-cmccarter-sandbox/"
               "672d4b4b-eeb6-4d6f-95b1-3b1c12610a48/scratchpad/merge_order.png")

DATASETS = ["dkd", "gtex_v9", "hypomap", "immune_cell_atlas",
            "mouse_pancreas_atlas", "tabula_sapiens"]
DS_LABEL = {"dkd": "DKD", "gtex_v9": "GTEx", "hypomap": "Hypomap",
            "immune_cell_atlas": "ImmuneCell", "mouse_pancreas_atlas": "MousePancreas",
            "tabula_sapiens": "TabulaSapiens"}

# strategy: (affine suffix, display label, palette hex); default first.
STRATS = [
    ("celltype_silhouette", "cell-type silhouette, high (default)", "#7AA4CA", True),
    ("celltype_sil_low",    "cell-type silhouette, low",           "#F3B06F", False),
    ("batch_sil_low",       "batch silhouette, low",               "#A3B478", False),
    ("batch_sil_high",      "batch silhouette, high",              "#A894C1", False),
    ("biggest",             "batch size, biggest",                 "#D6897C", False),
    ("smallest",            "batch size, smallest",                "#74B2AF", False),
    ("random_rs42",         "random priority",                     "#E6B8D2", False),
]

PROFILE = "all"
S = L.PROFILES[PROFILE]["metrics"]

bl, methods = L.load_baselines(REPO / "paper-results" / "baselines" / "score_uns.yaml")
# fill the scANVI/scVI tabula_sapiens gap (isolated_label_asw), as the paper does
EXTRAS = REPO / "batch-integration" / "leaderboard_extras"
bl, methods = L.inject_local_results(bl, methods, [
    f"scanvi:tabula_sapiens:{EXTRAS/'scanvi_tabula_sapiens.json'}",
    f"scvi:tabula_sapiens:{EXTRAS/'scvi_tabula_sapiens.json'}",
])
SWEEPS = REPO / "paper-results" / "sweeps"


def strat_score(results_dir, ds, kind):
    """condo score (kind = composite|bio|batch) for one strategy dir on one dataset
    (own pool, matches the ablation tables)."""
    condo = L.load_condo(results_dir, ds)
    pool = [bl[(m, ds)] for m in methods] + [condo]
    mm = L.metric_minmax(pool, S)
    return L.composite(condo, S, mm, kind)


def scanvi_score(ds, kind):
    """scANVI reference score, scaled with a stable pool (baselines + default condo)."""
    condo = L.load_condo(SWEEPS / "abl_celltype_silhouette", ds)
    pool = [bl[(m, ds)] for m in methods] + [condo]
    mm = L.metric_minmax(pool, S)
    return L.composite(bl[("scanvi", ds)], S, mm, kind)


KINDS = [("composite", "Overall composite"),
         ("bio", "Bio-conservation"),
         ("batch", "Batch-correction")]

# data[kind][transform][suf] = list over datasets; scanvi[kind] = list over datasets
data = {}
scanvi = {}
for kind, _ in KINDS:
    aff = {suf: [strat_score(SWEEPS / f"abl_{suf}", ds, kind) for ds in DATASETS]
           for suf, *_ in STRATS}
    ls = {suf: [strat_score(SWEEPS / f"abl_locscale_{suf}", ds, kind) for ds in DATASETS]
          for suf, *_ in STRATS}
    data[kind] = {"Affine": aff, "Location-scale": ls}
    scanvi[kind] = [scanvi_score(ds, kind) for ds in DATASETS]

# echo composite numbers for sanity-check against the tables
print(f"{'strategy':22s}  " + "  ".join(DS_LABEL[d][:6] for d in DATASETS))
for suf, lab, _, _ in STRATS:
    print(f"AFF {lab[:18]:18s}  " +
          "  ".join(f"{v:.3f}" for v in data['composite']['Affine'][suf]))
print("scANVI ref            " + "  ".join(f"{v:.3f}" for v in scanvi['composite']))

# ---- plot: 3 rows (component) x 2 cols (transform) ----
plt.rcParams.update({"font.size": 8.5, "font.family": "sans-serif",
                     "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"]})
ny = len(DATASETS)
ybase = list(range(ny))[::-1]  # dataset 0 at top
# vertical anti-overlap slots; default (index 0) centered so the star draws the eye
offsets = [0.0, -0.31, 0.09, -0.09, 0.20, -0.20, 0.31]
TRANSFORMS = ["Affine", "Location-scale"]

fig, axall = plt.subplots(len(KINDS), 2, figsize=(7.2, 8.4), sharey=True)

for r, (kind, klabel) in enumerate(KINDS):
    # shared x-range for this component's row (both transforms + scANVI)
    vals = [v for tf in TRANSFORMS for suf, *_ in STRATS
            for v in data[kind][tf][suf]] + scanvi[kind]
    lo, hi = min(vals), max(vals)
    pad = 0.03 * (hi - lo)
    for c, tf in enumerate(TRANSFORMS):
        ax = axall[r, c]
        ax.set_axisbelow(True)
        ax.grid(axis="x", color="0.85", lw=0.6)
        for row, ds in enumerate(DATASETS):
            y0 = ybase[row]
            ax.plot([scanvi[kind][row], scanvi[kind][row]], [y0 - 0.42, y0 + 0.42],
                    color="0.45", lw=1.0, ls=(0, (2, 1.5)), zorder=1)
            for i, (suf, lab, hex_, is_def) in enumerate(STRATS):
                x = data[kind][tf][suf][row]
                y = y0 + offsets[i]
                if is_def:
                    ax.scatter([x], [y], s=95, marker="*", color=hex_,
                               edgecolor="black", linewidth=0.6, zorder=4)
                else:
                    ax.scatter([x], [y], s=34, marker="o", color=hex_,
                               edgecolor="white", linewidth=0.5, zorder=3)
        ax.set_xlim(lo - pad, hi + pad)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        if r == 0:                       # transform titles on the top row only
            ax.set_title(tf, fontsize=10)
        if r == len(KINDS) - 1:          # x-label on the bottom row only
            ax.set_xlabel("all-methods score (7-metric)")
    # component label on the left of each row
    axall[r, 0].set_yticks(ybase)
    axall[r, 0].set_yticklabels([DS_LABEL[d] for d in DATASETS])
    axall[r, 0].annotate(klabel, xy=(0, 0.5), xytext=(-80, 0),
                         xycoords="axes fraction", textcoords="offset points",
                         ha="center", va="center", rotation=90, fontsize=10)

# shared legend beneath all panels
handles = []
for suf, lab, hex_, is_def in STRATS:
    m = "*" if is_def else "o"
    handles.append(Line2D([0], [0], marker=m, color="none", markerfacecolor=hex_,
                          markeredgecolor="black" if is_def else "white",
                          markersize=11 if is_def else 7, label=lab))
handles.append(Line2D([0], [0], color="0.45", lw=1.0, ls=(0, (2, 1.5)),
                      label="scANVI (embedding reference)"))
fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False,
           fontsize=7.6, handletextpad=0.4, columnspacing=1.2,
           bbox_to_anchor=(0.5, 0.0))

fig.subplots_adjust(left=0.20, right=0.98, top=0.95, bottom=0.13, wspace=0.08, hspace=0.22)
OUT_PDF.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT_PDF, bbox_inches="tight")
fig.savefig(OUT_PNG, dpi=170, bbox_inches="tight")
print(f"\nwrote {OUT_PDF}\nwrote {OUT_PNG}")
