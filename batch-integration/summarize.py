"""Aggregate per-method JSON results into a single table.

scIB-style composite: 0.4 * batch + 0.6 * bio, where
  bio   = mean(asw_label, nmi, ari)
  batch = mean(asw_batch, graph_connectivity, pcr, ari_batch, nmi_batch)
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


def load_results(results_dir: Path) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for p in sorted(results_dir.glob("*.json")):
        rows = json.loads(p.read_text())
        out[p.stem] = {r["metric"]: r["score"] for r in rows if "score" in r}
    return out


def composite(scores: dict[str, float]) -> tuple[float, float, float]:
    bio_vals = [scores[m] for m in BIO_METRICS if m in scores]
    batch_vals = [scores[m] for m in BATCH_METRICS if m in scores]
    bio = mean(bio_vals) if bio_vals else float("nan")
    batch = mean(batch_vals) if batch_vals else float("nan")
    return bio, batch, 0.6 * bio + 0.4 * batch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", required=True)
    args = parser.parse_args()

    methods = load_results(Path(args.results_dir))
    if not methods:
        print("(no results)")
        return

    headers = (
        ["method"]
        + list(BIO_METRICS)
        + list(BATCH_METRICS)
        + ["bio", "batch", "composite"]
    )
    rows = []
    for name, scores in methods.items():
        bio, batch, comp = composite(scores)
        row = [name]
        for m in BIO_METRICS + BATCH_METRICS:
            v = scores.get(m, float("nan"))
            row.append(f"{v:.3f}" if isinstance(v, float) else str(v))
        row.extend([f"{bio:.3f}", f"{batch:.3f}", f"{comp:.3f}"])
        rows.append((comp, row))

    rows.sort(key=lambda r: r[0], reverse=True)
    rows = [r[1] for r in rows]

    widths = [
        max(len(str(c)) for c in [h] + [row[i] for row in rows])
        for i, h in enumerate(headers)
    ]
    print(" | ".join(h.ljust(w) for h, w in zip(headers, widths)))
    print("-+-".join("-" * w for w in widths))
    for row in rows:
        print(" | ".join(c.ljust(w) for c, w in zip(row, widths)))


if __name__ == "__main__":
    main()
