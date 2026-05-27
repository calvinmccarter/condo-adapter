"""Backfill the cell_cycle_conservation metric for condo result JSONs.

openproblems only runs ccc on immune_cell_atlas, mouse_pancreas_atlas, and
tabula_sapiens (it's disabled on dkd/gtex_v9/hypomap). Our eval didn't
compute it, so this driver runs the standalone compute_ccc.py against an
already-written output h5ad and appends/patches the ``cell_cycle_conservation``
entry of the result JSON in place.

No re-fit; reads existing outputs/<ds>.h5ad and the benchmark solution.

Usage:
    python backfill_ccc.py --sweep-dir work/fullbench_v3_agglomerative \\
        --datasets immune_cell_atlas mouse_pancreas_atlas tabula_sapiens
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
CCC_SCRIPT = REPO / "batch-integration" / "scripts" / "compute_ccc.py"
BENCH = REPO.parent / "fullbench"
METRIC = "cell_cycle_conservation"


def _entry(items: list, name: str) -> dict | None:
    for m in items:
        if (m.get("metric_id") or m.get("metric") or m.get("name")) == name:
            return m
    return None


def backfill_one(ds: str, sweep_dir: Path, python: str, timeout: int) -> dict:
    out_h5 = sweep_dir / "outputs" / f"{ds}.h5ad"
    res_json = sweep_dir / "results" / f"{ds}.json"
    solution = BENCH / "datasets" / ds / "solution.h5ad"
    for p in (out_h5, res_json, solution):
        if not p.exists():
            return {"dataset": ds, "status": "missing", "path": str(p)}

    items = json.loads(res_json.read_text())
    entry = _entry(items, METRIC)
    if entry is not None and entry.get("score") is not None:
        return {"dataset": ds, "status": "already_present", "score": entry["score"]}

    t0 = time.time()
    print(f"[{ds}] running ccc (timeout={timeout}s) ...", flush=True)
    try:
        proc = subprocess.run(
            [python, str(CCC_SCRIPT),
             "--integrated", str(out_h5), "--solution", str(solution)],
            capture_output=True, text=True, timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return {"dataset": ds, "status": "timeout", "dt": time.time() - t0}

    dt = time.time() - t0
    if proc.returncode != 0:
        return {"dataset": ds, "status": "failed", "rc": proc.returncode,
                "stderr": proc.stderr[-1500:], "dt": dt}

    last = next(ln for ln in reversed(proc.stdout.splitlines()) if ln.strip())
    score = float(json.loads(last)["score"])

    new_entry = {"metric": METRIC, "score": score, "dt": dt}
    if entry is None:
        items.append(new_entry)
    else:
        entry.clear()
        entry.update(new_entry)
    res_json.write_text(json.dumps(items, indent=2))
    return {"dataset": ds, "status": "ok", "score": score, "dt": dt}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep-dir", required=True)
    ap.add_argument("--datasets", nargs="+",
                    default=["immune_cell_atlas", "mouse_pancreas_atlas",
                             "tabula_sapiens"])
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--timeout", type=int, default=3 * 60 * 60)
    args = ap.parse_args()

    sweep_dir = Path(args.sweep_dir)
    for ds in args.datasets:
        res = backfill_one(ds, sweep_dir, args.python, args.timeout)
        print(f"[{res['status']}] {ds}  score={res.get('score')}  "
              f"dt={res.get('dt', 0):.0f}s", flush=True)
        if res["status"] == "failed":
            print(f"  stderr: {res.get('stderr', '')[-600:]}", flush=True)
    print("ccc backfill complete", flush=True)


if __name__ == "__main__":
    main()
