#!/usr/bin/env bash
# Sweep across condo variants. Loops over the cross-product of
# (divergence x transform x rep x hvg_only), spawning each as a detached
# child of run_one.sh.
#
# Usage:
#     setsid nohup batch-integration/scripts/run_sweep.sh > sweep.log 2>&1 &
#
# Required env (same as run_one.sh):
#     CONDO_BENCH_PY, CONDO_BENCH_DATASET, CONDO_BENCH_SOLUTION
#
# Optional env:
#     CONDO_BENCH_WORK_DIR  base directory for outputs/results/logs
#     CONDO_BENCH_VARIANTS  space-separated list of "divergence transform rep hvg_only"
#                           tuples; defaults to the cartesian product of all axes.
#     CONDO_BENCH_PARALLEL  if "1", launch all variants concurrently
#                           (detached). Default sequential.

set -uo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
WORK_DIR="${CONDO_BENCH_WORK_DIR:-$(cd "$HERE/../.." && pwd)/work}"

if [ -n "${CONDO_BENCH_VARIANTS:-}" ]; then
    variants=()
    while read -r line; do
        [ -z "$line" ] && continue
        variants+=("$line")
    done <<< "$CONDO_BENCH_VARIANTS"
else
    variants=(
        "kld location-scale features 0"
        "kld affine          features 0"
        "mmd location-scale features 0"
        "mmd affine          features 0"
    )
fi

mkdir -p "$WORK_DIR/logs"
SWEEP_LOG="$WORK_DIR/logs/sweep.log"

log() { echo "[$(date '+%F %T')] $*" | tee -a "$SWEEP_LOG"; }

log "Sweep start (PID $$, work_dir=$WORK_DIR)"
for spec in "${variants[@]}"; do
    # shellcheck disable=SC2206
    args=($spec)
    if [ "${CONDO_BENCH_PARALLEL:-0}" = "1" ]; then
        setsid nohup "$HERE/run_one.sh" "${args[@]}" "$WORK_DIR" \
            > "$WORK_DIR/logs/runner_${args[0]}_${args[1]//-/_}.log" 2>&1 < /dev/null &
        disown
        log "launched ${args[*]} detached PID=$!"
    else
        "$HERE/run_one.sh" "${args[@]}" "$WORK_DIR" \
            || log "[${args[*]}] continuing past failure"
    fi
done

if [ "${CONDO_BENCH_PARALLEL:-0}" = "1" ]; then
    log "all variants launched in background"
else
    log "Sweep done"
fi
