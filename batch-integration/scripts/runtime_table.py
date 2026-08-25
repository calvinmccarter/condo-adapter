"""Assemble a fit/eval runtime table from sweep summary.jsonl files.

For each method, reads per-dataset ``fit_dt`` / ``eval_dt`` from each sweep
dir's ``summary.jsonl``. Where a run reused a prior fit (``fit_dt`` logged as
0), it recovers the fit wall-clock from a ``>> CONDO_TIMING read=.. fit=..
write=..`` line in ``logs/<dataset>.fit.log`` (emitted by the instrumented
condo_runner) by summing read+fit+write. Multiple dirs can be merged per
method, so a table can be built from runs split across sweeps/boxes.

Run with no args to reproduce the paper's affine-vs-location-scale official
runtime table (celltype_silhouette, wd=1e-4). Override with --method LABEL=dirs
(comma-separated, repeatable) and --format {plain,markdown,csv}.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

DATASETS = [
    "dkd", "gtex_v9", "mouse_pancreas_atlas",
    "immune_cell_atlas", "hypomap", "tabula_sapiens",
]

# The official runtime table: affine data is scattered across the completion
# rerun (dkd/gtex fit+eval + hypomap fit-only) and the two-box ablation dirs
# (mouse, immune, tabula, hypomap-eval); location-scale is the single wd=1e-4 run.
DEFAULT_METHODS = {
    "affine": [
        "work/affine_official_completion",
        "work/ablation_hypomap_mouse/abl_celltype_silhouette",
        "work/ablation_immune_tabula/abl_celltype_silhouette",
    ],
    "location-scale": ["work/abl_locscale_celltype_silhouette_wd1e-4"],
}

_TIMING_RE = re.compile(r"read=([\d.]+)s\s+fit=([\d.]+)s\s+write=([\d.]+)s")


def _load_dir(d: Path) -> dict[str, dict]:
    """Return {dataset: {'fit': sec|None, 'eval': sec|None}} for one sweep dir."""
    rows: dict[str, dict] = {}
    summ = d / "summary.jsonl"
    if summ.exists():
        for line in summ.read_text().splitlines():
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("status") != "ok":
                continue
            cur = rows.setdefault(r["dataset"], {})
            if r.get("eval_dt") is not None:
                cur["eval"] = r["eval_dt"]
            if r.get("fit_dt"):  # non-zero
                cur["fit"] = r["fit_dt"]
    logs = d / "logs"
    if logs.is_dir():
        for f in logs.glob("*.fit.log"):
            ds = f.name[: -len(".fit.log")]
            wall = None
            for line in f.read_text(errors="ignore").splitlines():
                if "CONDO_TIMING" in line:
                    m = _TIMING_RE.search(line)
                    if m:
                        wall = sum(float(x) for x in m.groups())
            if wall is not None:
                rows.setdefault(ds, {}).setdefault("fit", wall)
    return rows


def _merge(dirs: list[str]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for d in dirs:
        for ds, vals in _load_dir(Path(d)).items():
            cur = out.setdefault(ds, {})
            for k in ("fit", "eval"):
                if vals.get(k) is not None:
                    cur.setdefault(k, vals[k])
    return out


def _mins(sec):
    return f"{sec / 60:.1f}" if sec else "n/a"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--method", action="append", default=None,
        help="LABEL=dir1,dir2,...  (repeatable). Defaults to the paper table.",
    )
    ap.add_argument("--format", choices=["plain", "markdown", "csv"], default="plain")
    args = ap.parse_args()

    if args.method:
        methods = {}
        for spec in args.method:
            label, _, dirs = spec.partition("=")
            methods[label] = [d for d in dirs.split(",") if d]
    else:
        methods = DEFAULT_METHODS

    data = {label: _merge(dirs) for label, dirs in methods.items()}
    cols = [(label, phase) for label in methods for phase in ("fit", "eval")]
    header = ["dataset"] + [f"{label} {phase} (min)" for label, phase in cols]

    def row(ds):
        return [ds] + [_mins(data[label].get(ds, {}).get(phase)) for label, phase in cols]

    totals = ["total"]
    for label, phase in cols:
        tot = sum(data[label].get(ds, {}).get(phase) or 0 for ds in DATASETS)
        totals.append(f"{tot / 60:.1f}")

    table = [header] + [row(ds) for ds in DATASETS] + [totals]

    if args.format == "csv":
        print("\n".join(",".join(r) for r in table))
    elif args.format == "markdown":
        print("| " + " | ".join(table[0]) + " |")
        print("| " + " | ".join("---" for _ in table[0]) + " |")
        for r in table[1:]:
            print("| " + " | ".join(r) + " |")
    else:
        w = [max(len(r[i]) for r in table) for i in range(len(header))]
        for i, r in enumerate(table):
            print("  ".join(c.rjust(w[j]) for j, c in enumerate(r)))
            if i == 0 or i == len(table) - 2:
                print("  ".join("-" * w[j] for j in range(len(header))))


if __name__ == "__main__":
    main()
