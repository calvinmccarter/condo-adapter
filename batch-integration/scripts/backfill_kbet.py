"""Backfill the kbet metric for datasets whose eval timed out.

kBET (R/rpy2) scales poorly with cell count and exceeds eval_variant.py's
3600s subprocess cap on the large datasets. This driver re-runs the
standalone ``compute_kbet.py`` against an already-written output h5ad with
a generous timeout, then patches the ``kbet`` entry of the result JSON in
place (replacing the recorded error with ``{"metric","score","dt"}``).

No re-fit; reads existing outputs/<ds>.h5ad and the benchmark solution.

Usage:
    python backfill_kbet.py --sweep-dir work/fullbench_v3_agglomerative \\
        --datasets hypomap immune_cell_atlas mouse_pancreas_atlas \\
        --timeout 21600
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
KBET_PY = REPO.parent / ".venv-kbet" / "bin" / "python"
KBET_SCRIPT = REPO / "batch-integration" / "scripts" / "compute_kbet.py"
BENCH = REPO.parent / "fullbench"


def _kbet_entry(items: list) -> dict | None:
    for m in items:
        if (m.get("metric_id") or m.get("metric") or m.get("name")) == "kbet":
            return m
    return None


def backfill_one(ds: str, sweep_dir: Path, timeout: int) -> dict:
    out_h5 = sweep_dir / "outputs" / f"{ds}.h5ad"
    res_json = sweep_dir / "results" / f"{ds}.json"
    solution = BENCH / "datasets" / ds / "solution.h5ad"
    for p in (out_h5, res_json, solution):
        if not p.exists():
            return {"dataset": ds, "status": "missing", "path": str(p)}

    items = json.loads(res_json.read_text())
    entry = _kbet_entry(items)
    if entry is not None and entry.get("score") is not None:
        return {"dataset": ds, "status": "already_has_kbet", "score": entry["score"]}

    t0 = time.time()
    print(f"[{ds}] running kbet (timeout={timeout}s) ...", flush=True)
    try:
        proc = subprocess.run(
            [str(KBET_PY), str(KBET_SCRIPT),
             "--integrated", str(out_h5), "--solution", str(solution)],
            capture_output=True, text=True, env={"R_HOME": "/usr/lib/R",
                                                  "PATH": "/usr/bin:/bin"},
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return {"dataset": ds, "status": "timeout", "dt": time.time() - t0}

    dt = time.time() - t0
    if proc.returncode != 0:
        return {"dataset": ds, "status": "failed", "rc": proc.returncode,
                "stderr": proc.stderr[-1500:], "dt": dt}

    last = next(ln for ln in reversed(proc.stdout.splitlines()) if ln.strip())
    score = float(json.loads(last)["score"])

    new_entry = {"metric": "kbet", "score": score, "dt": dt}
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
    ap.add_argument("--datasets", nargs="+", required=True)
    ap.add_argument("--timeout", type=int, default=6 * 60 * 60)
    args = ap.parse_args()

    sweep_dir = Path(args.sweep_dir)
    for ds in args.datasets:
        res = backfill_one(ds, sweep_dir, args.timeout)
        print(f"[{res['status']}] {ds}  "
              f"score={res.get('score')}  dt={res.get('dt', 0):.0f}s", flush=True)
        if res["status"] == "failed":
            print(f"  stderr: {res.get('stderr', '')[-500:]}", flush=True)
    print("kbet backfill complete", flush=True)


if __name__ == "__main__":
    main()
