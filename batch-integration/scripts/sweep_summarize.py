"""Aggregate a sweep into a single table with hyperparams + scores.

Reads:
- ``<sweep_dir>/results/*.json``   per-config metric scores (from eval_variant.py)
- ``<sweep_dir>/grid.json``        list of configs the sweep was launched with
- ``<sweep_dir>/sweep_summary.jsonl`` per-fit status + dt

Joins on the per-config tag and prints a sorted leaderboard. Use this
instead of summarize.py when results carry hyperparameter columns that
should be shown.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean


BIO_METRICS = (
    "asw_label",
    "nmi",
    "ari",
    "isolated_label_f1",
    "isolated_label_asw",
    "clisi",
    "hvg_overlap",
)
BATCH_METRICS = (
    "asw_batch",
    "graph_connectivity",
    "pcr",
    "ilisi",
    "kbet",
)


def tag_for(cfg: dict) -> str:
    base = (
        "ne{n_epochs}_lr{learning_rate:.0e}"
        "_ms{mmd_size}_bs{batch_size}"
        "_wd{weight_decay:.0e}_seed{random_state}"
    ).format(**cfg)
    return base


def composite(scores: dict[str, float]) -> tuple[float, float, float]:
    bio_vals = [scores[m] for m in BIO_METRICS if m in scores]
    batch_vals = [scores[m] for m in BATCH_METRICS if m in scores]
    bio = mean(bio_vals) if bio_vals else float("nan")
    batch = mean(batch_vals) if batch_vals else float("nan")
    return bio, batch, 0.6 * bio + 0.4 * batch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep-dir", required=True)
    parser.add_argument(
        "--group-by-config",
        action="store_true",
        help="Aggregate seeds: mean ± std per (n_epochs, lr, mmd_size, batch_size, weight_decay)",
    )
    args = parser.parse_args()

    sweep_dir = Path(args.sweep_dir)
    grid = json.loads((sweep_dir / "grid.json").read_text())

    rows: list[dict] = []
    for cfg in grid:
        tag = tag_for(cfg)
        res_path = sweep_dir / "results" / f"{tag}.json"
        if not res_path.exists() or res_path.stat().st_size == 0:
            continue
        rows_for = json.loads(res_path.read_text())
        scores = {r["metric"]: r["score"] for r in rows_for if "score" in r}
        bio, batch, comp = composite(scores)
        rows.append({**cfg, **scores, "bio": bio, "batch": batch, "composite": comp})

    if not rows:
        print("(no completed configs)")
        return

    if args.group_by_config:
        # average across seeds
        keys = ("n_epochs", "learning_rate", "mmd_size", "batch_size", "weight_decay")
        from collections import defaultdict

        groups: dict[tuple, list[dict]] = defaultdict(list)
        for r in rows:
            groups[tuple(r[k] for k in keys)].append(r)

        out_rows = []
        for k, items in groups.items():
            comp_vals = [it["composite"] for it in items]
            bio_vals = [it["bio"] for it in items]
            batch_vals = [it["batch"] for it in items]
            out_rows.append(
                {
                    **dict(zip(keys, k)),
                    "n_seeds": len(items),
                    "composite_mean": mean(comp_vals),
                    "composite_std": (
                        (
                            sum((c - mean(comp_vals)) ** 2 for c in comp_vals)
                            / max(1, len(comp_vals) - 1)
                        )
                        ** 0.5
                    ),
                    "bio_mean": mean(bio_vals),
                    "batch_mean": mean(batch_vals),
                }
            )
        out_rows.sort(key=lambda r: r["composite_mean"], reverse=True)

        cols = [
            "n_epochs",
            "learning_rate",
            "mmd_size",
            "batch_size",
            "weight_decay",
            "n_seeds",
            "bio_mean",
            "batch_mean",
            "composite_mean",
            "composite_std",
        ]
        widths = {
            c: max(len(c), *(len(str(_fmt(r.get(c)))) for r in out_rows)) for c in cols
        }
        print(" | ".join(c.ljust(widths[c]) for c in cols))
        print("-+-".join("-" * widths[c] for c in cols))
        for r in out_rows:
            print(" | ".join(_fmt(r.get(c)).ljust(widths[c]) for c in cols))
    else:
        rows.sort(key=lambda r: r["composite"], reverse=True)
        cols = [
            "n_epochs",
            "learning_rate",
            "mmd_size",
            "batch_size",
            "weight_decay",
            "random_state",
            *BIO_METRICS,
            *BATCH_METRICS,
            "bio",
            "batch",
            "composite",
        ]
        widths = {c: max(len(c), *(len(str(_fmt(r.get(c)))) for r in rows)) for c in cols}
        print(" | ".join(c.ljust(widths[c]) for c in cols))
        print("-+-".join("-" * widths[c] for c in cols))
        for r in rows:
            print(" | ".join(_fmt(r.get(c)).ljust(widths[c]) for c in cols))


def _fmt(v) -> str:
    if v is None:
        return "-"
    if isinstance(v, float):
        if abs(v) >= 0.01 or v == 0:
            return f"{v:.3f}"
        return f"{v:.1e}"
    return str(v)


if __name__ == "__main__":
    main()
