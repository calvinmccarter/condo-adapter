#!/usr/bin/env bash
# Fit + eval a single condo variant, detached. Designed to survive SSH
# disconnects: launch via
#     setsid nohup batch-integration/scripts/run_one.sh \
#         <divergence> <transform-type> [rep] [hvg_only] [work_dir] > LOG 2>&1 &
# where:
#     <divergence>     ∈ {kld, mmd}
#     <transform-type> ∈ {location-scale, affine}
#     [rep]            ∈ {features, pca}  (default: features)
#     [hvg_only]       ∈ {0, 1}           (default: 0)
#     [work_dir]       directory containing outputs/, results/, logs/ subdirs
#                      (default: <repo>/work)
#
# Environment overrides (all optional):
#     CONDO_BENCH_PY        path to python interpreter (default: python3)
#     CONDO_BENCH_DATASET   path to dataset.h5ad
#     CONDO_BENCH_SOLUTION  path to solution.h5ad

set -uo pipefail

divergence="$1"
transform="$2"
rep="${3:-features}"
hvg_only="${4:-0}"

HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
WORK_DIR="${5:-$REPO/work}"
mkdir -p "$WORK_DIR/outputs" "$WORK_DIR/results" "$WORK_DIR/logs"

PY="${CONDO_BENCH_PY:-python3}"
DATASET="${CONDO_BENCH_DATASET:?CONDO_BENCH_DATASET must be set or passed via env}"
SOLUTION="${CONDO_BENCH_SOLUTION:?CONDO_BENCH_SOLUTION must be set or passed via env}"

tag="condo_${divergence}_${transform//-/_}"
[ "$rep" = "pca" ] && tag="${tag}_pca"
[ "$hvg_only" = "1" ] && tag="${tag}_hvg"

out="$WORK_DIR/outputs/${tag}.h5ad"
res="$WORK_DIR/results/${tag}.json"
fitlog="$WORK_DIR/logs/${tag}.fit.log"
evallog="$WORK_DIR/logs/${tag}.eval.log"

log() { echo "[$(date '+%F %T')] [$tag] $*"; }

hvg_args=()
[ "$hvg_only" = "1" ] && hvg_args+=(--hvg-only)

log "fitting"
if ! "$PY" "$REPO/batch-integration/run_local.py" \
        --divergence "$divergence" \
        --transform-type "$transform" \
        --rep "$rep" \
        "${hvg_args[@]}" \
        --input "$DATASET" \
        --output "$out" >"$fitlog" 2>&1; then
    log "FIT FAILED (see $fitlog)"
    exit 1
fi
log "fit ok ($out)"

log "evaluating"
if ! "$PY" "$REPO/batch-integration/eval_variant.py" \
        --integrated "$out" \
        --dataset "$DATASET" \
        --solution "$SOLUTION" \
        --output "$res" >"$evallog" 2>&1; then
    log "EVAL FAILED (see $evallog)"
    exit 1
fi
log "eval ok ($res)"
