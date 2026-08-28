"""Wait for the running label-dropout sweep (the 144 original runs) to finish,
then launch the f=0.90 cells on the same sweep dir.

f=0.90 gets both handlings (excluded is defined for f<1) with the usual seed
policy (3 on dkd/gtex, 1 on the atlases). Re-invoking the sweep is resumable:
the already-complete f={0,0.25,0.5,0.75,1.0} runs are skipped, only the new
f=0.90 runs execute. Launch durably:

    setsid nohup .venv-condo-bench/bin/python \
        batch-integration/scripts/run_dropout90_after.py > LOG 2>&1 &
"""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import label_dropout_sweep as L  # noqa: E402

REPO = HERE.parents[1]
SWEEP = REPO / "work" / "label_dropout"
PY = sys.executable
POLL = 120


def original_keys():
    runs = L.build_runs(["affine", "location-scale"], L.DATASETS,
                        [0.25, 0.5, 0.75, 1.0], ["bucket", "excluded"], 3, 1)
    return [L._key(*r) for r in runs]


def done_count(keys):
    return sum((SWEEP / "results" / f"{k}.json").exists() for k in keys)


def orch_running():
    r = subprocess.run(["pgrep", "-f", f"label_dropout_sweep.py .*{SWEEP}"],
                       capture_output=True, text=True)
    return bool(r.stdout.strip())


def main():
    keys = original_keys()
    tot = len(keys)
    last, stable = -1, 0
    while True:
        d = done_count(keys)
        running = orch_running()
        print(f"[wait] original sweep {d}/{tot} done, orchestrator_running={running}",
              flush=True)
        if d >= tot:
            print("[wait] all original runs complete", flush=True)
            break
        if not running:
            stable = stable + 1 if d == last else 0
            if stable >= 3:
                miss = [k for k in keys if not (SWEEP / "results" / f"{k}.json").exists()]
                print(f"[wait] orchestrator gone, count stable at {d}/{tot}; "
                      f"proceeding anyway. missing={miss}", flush=True)
                break
        last = d
        time.sleep(POLL)

    print("[launch] starting f=0.90 sweep", flush=True)
    cmd = [PY, str(HERE / "label_dropout_sweep.py"),
           "--sweep-dir", str(SWEEP), "--python", PY,
           "--fractions", "0.9", "--n-gpus", "2", "--parallel", "2"]
    print("  " + " ".join(cmd), flush=True)
    rc = subprocess.call(cmd)
    print(f"[done] f=0.90 sweep exited rc={rc}", flush=True)


if __name__ == "__main__":
    main()
