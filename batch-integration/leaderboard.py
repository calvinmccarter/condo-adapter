"""Rank condo against the openproblems baselines on a fixed metric set.

Methodology (per the project decision): pick ONE metric set S used for
*every* method on *every* dataset, then average each method over exactly S.
S is the set of metrics that are present and finite (NaN counts as absent)
for all methods in a representative roster + condo, on all datasets in
scope. This keeps the comparison apples-to-apples instead of crediting any
method for a metric others structurally lack.

Scoring matches the openproblems/scIB aggregation: every metric is min-max
scaled across the full method pool (all baselines incl. controls + condo)
per dataset BEFORE averaging, so each metric's contribution is normalized
to its across-method spread. The negative controls (no_integration,
shuffle) and the embed_cell_types oracle anchor the scale. Then:
    bio   = mean(scaled bio metrics in S)
    batch = mean(scaled batch metrics in S)
    composite = 0.6*bio + 0.4*batch
Scaling uses the full pool regardless of --include-controls, so condo's
scaled scores are stable; the controls are only hidden from the displayed
ranking by default.

Two profiles are emitted:

* "all" — every method (embedding + feature). The largest set common to
  embedding and feature methods, so it necessarily drops feature-only
  metrics (hvg_overlap) plus those NaN/missing somewhere:
      bio   = nmi, ari, isolated_label_asw, clisi
      batch = asw_batch, graph_connectivity, ilisi          (7 metrics)
  NB: embed_cell_types[_jittered] top this board because they are ORACLE
  controls that embed from ground-truth labels — an upper bound, not a
  competitor. Always read condo's rank among *real* methods too.

* "feature" — feature-space methods only (those producing corrected
  expression, i.e. with a finite hvg_overlap). With the embedding methods
  gone the common set grows to include hvg_overlap, asw_label, and pcr:
      bio   = asw_label, nmi, ari, isolated_label_asw, clisi, hvg_overlap
      batch = asw_batch, graph_connectivity, pcr, ilisi    (10 metrics)
  Derived NaN-aware over {combat, scanorama} + condo (scalex excluded from
  derivation as it failed entirely on mouse_pancreas_atlas, but is still
  scored on the datasets where it has the full set).

* "per_dataset" — like "all" (every method, oracle caveat applies), but the
  metric set S is derived independently per dataset rather than forced
  uniform across datasets. S_ds = metrics finite for every method in the
  complete-coverage roster + condo on that dataset. This lets metrics that
  only exist on some datasets count where everyone has them — notably
  cell_cycle_conservation, which openproblems only runs on immune_cell_atlas,
  mouse_pancreas_atlas, and tabula_sapiens. Most faithful to openproblems'
  own per-dataset aggregation.

The global "all"/"feature" sets drop isolated_label_f1 (NaN on degenerate
dkd), kbet (missing on the large datasets for everyone), and
cell_cycle_conservation (only on 3/6 datasets). "per_dataset" recovers each
of these on the specific datasets where every method has them.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean

REPO = Path(__file__).resolve().parents[1]
BASELINES_YAML = REPO / "work" / "fullbench" / "baselines" / "score_uns.yaml"

DEFAULT_DATASETS = [
    "dkd", "gtex_v9", "hypomap", "immune_cell_atlas", "mouse_pancreas_atlas",
    "tabula_sapiens",
]

# Which pool each metric belongs to, for the 0.6*bio + 0.4*batch composite.
BIO_POOL = {
    "asw_label", "nmi", "ari", "isolated_label_f1", "isolated_label_asw",
    "clisi", "hvg_overlap", "cell_cycle_conservation",
}
BATCH_POOL = {"asw_batch", "graph_connectivity", "pcr", "ilisi", "kbet"}
ALL_METRICS = [
    "asw_label", "nmi", "ari", "isolated_label_f1", "isolated_label_asw",
    "clisi", "hvg_overlap", "cell_cycle_conservation",
    "asw_batch", "graph_connectivity", "pcr", "ilisi", "kbet",
]
# Methods that ran on every dataset — used to DERIVE a metric set NaN-aware
# without a partial-coverage method (bbknn/uce) or a total failure
# (mnnpy/batchelor_mnn_correct on most) collapsing the intersection.
DERIVE_ROSTER = [
    "combat", "harmony", "harmonypy", "pyliger",
    "scanvi", "scvi", "batchelor_fastmnn",
]
# Not real integration methods: oracle upper bounds that embed from the
# ground-truth labels, plus the negative/positive shuffle & no-integration
# controls. --real-only drops these from the ranking.
CONTROLS = {
    "embed_cell_types", "embed_cell_types_jittered",
    "no_integration", "no_integration_batch",
    "shuffle_integration", "shuffle_integration_by_batch",
    "shuffle_integration_by_cell_type",
}

PROFILES = {
    "all": {
        "metrics": ["nmi", "ari", "isolated_label_asw", "clisi",
                    "asw_batch", "graph_connectivity", "ilisi"],
        "roster": None,  # None => all methods
    },
    "feature": {
        "metrics": ["asw_label", "nmi", "ari", "isolated_label_asw", "clisi",
                    "hvg_overlap", "asw_batch", "graph_connectivity",
                    "pcr", "ilisi"],
        "roster": "feature",  # only feature-space methods
    },
    "per_dataset": {
        "metrics": "per_dataset",  # derived per dataset (see derive_set)
        "roster": None,
    },
}


def _finite(v) -> bool:
    return isinstance(v, (int, float)) and not (
        isinstance(v, float) and math.isnan(v)
    )


def load_baselines(path: Path):
    import yaml

    data = yaml.safe_load(path.read_text())
    bl: dict = defaultdict(dict)
    for e in data:
        ds = e["dataset_id"].split("/")[-1]
        for m, v in zip(e["metric_ids"], e["metric_values"]):
            bl[(e["method_id"], ds)][m] = v
    methods = sorted({e["method_id"] for e in data})
    return bl, methods


def inject_local_results(bl, methods, injections):
    """Override or add per-(method, dataset) entries with locally-computed
    result JSONs. Each injection is "method:dataset:path" -> the result JSON
    at `path` is loaded and replaces bl[(method, dataset)] entirely. Adds
    `method` to the methods list if it wasn't already present.

    Used to patch in metrics that are missing from the published baselines
    (e.g. scvi/scanvi on tabula_sapiens, which lack isolated_label_asw)."""
    for spec in injections:
        parts = spec.split(":", 2)
        if len(parts) != 3:
            raise ValueError(
                f"--inject expects 'method:dataset:path', got {spec!r}"
            )
        method, dataset, path = parts
        rows = json.loads(Path(path).read_text())
        sc = {r["metric"]: r["score"] for r in rows
              if "score" in r and r["score"] is not None}
        bl[(method, dataset)] = sc
        if method not in methods:
            methods.append(method)
            methods.sort()
        print(f"[inject] {method} on {dataset}: {len(sc)} metrics from {path}",
              flush=True)
    return bl, methods


def load_condo(results_dir: Path, ds: str) -> dict:
    p = results_dir / f"{ds}.json"
    if not p.exists():
        return {}
    rows = json.loads(p.read_text())
    return {r["metric"]: r["score"] for r in rows if "score" in r}


def is_feature_method(bl: dict, me: str, datasets: list[str]) -> bool:
    """Feature-space methods produce corrected expression, so they have a
    finite hvg_overlap on at least one dataset. condo is always feature."""
    return any(_finite(bl[(me, ds)].get("hvg_overlap")) for ds in datasets)


def metric_minmax(pool_scores: list[dict], S: list[str]) -> dict:
    """Per-metric (min, max) over the full method pool (finite values only).
    Mirrors openproblems: each metric is min-max scaled across all methods
    evaluated on the dataset before being averaged into bio/batch."""
    mm = {}
    for m in S:
        vals = [s[m] for s in pool_scores if m in s and _finite(s[m])]
        mm[m] = (min(vals), max(vals)) if vals else (0.0, 0.0)
    return mm


def _scaled(v: float, lo: float, hi: float) -> float:
    # No spread => metric carries no discriminative info; contribute 0 so it
    # neither helps nor hurts ranking (affects every method equally).
    return (v - lo) / (hi - lo) if hi > lo else 0.0


def composite(scores: dict, S: list[str], mm: dict,
              kind: str = "composite") -> float | None:
    if not all(m in scores and _finite(scores[m]) for m in S):
        return None
    bio = [_scaled(scores[m], *mm[m]) for m in S if m in BIO_POOL]
    batch = [_scaled(scores[m], *mm[m]) for m in S if m in BATCH_POOL]
    if not bio or not batch:
        return None
    if kind == "bio":
        return mean(bio)
    if kind == "batch":
        return mean(batch)
    return 0.6 * mean(bio) + 0.4 * mean(batch)


def derive_set(bl, results_dir, ds: str) -> list[str]:
    """S_ds = metrics finite for every DERIVE_ROSTER method + condo on ds."""
    condo = load_condo(results_dir, ds)

    def has(me, m):
        sc = condo if me == "condo" else bl[(me, ds)]
        return m in sc and _finite(sc[m])

    return [m for m in ALL_METRICS
            if all(has(me, m) for me in DERIVE_ROSTER + ["condo"])]


def run_profile(name, profile, bl, methods, results_dir, datasets, condo_label,
                real_only=False, kind="composite"):
    per_dataset = profile["metrics"] == "per_dataset"
    print(f"\n{'#' * 70}")
    print(f"# PROFILE: {name}" +
          ("" if real_only else "  [INCLUDING controls/oracles]"))
    if not per_dataset:
        S = profile["metrics"]
        nbio = sum(1 for m in S if m in BIO_POOL)
        nbatch = sum(1 for m in S if m in BATCH_POOL)
        print(f"#   {len(S)} metrics: {nbio} bio + {nbatch} batch")
        print(f"#   bio  = {[m for m in S if m in BIO_POOL]}")
        print(f"#   batch= {[m for m in S if m in BATCH_POOL]}")
    else:
        print(f"#   metric set derived per dataset (NaN-aware) over "
              f"{len(DERIVE_ROSTER)} complete-coverage methods + condo")
    score_desc = {
        "composite": "0.6*bio + 0.4*batch",
        "bio": "bio only = mean(scaled bio metrics)",
        "batch": "batch only = mean(scaled batch metrics)",
    }[kind]
    print(f"#   score = {score_desc}, metrics min-max scaled "
          f"across the full method pool per dataset")
    print(f"{'#' * 70}")

    if profile["roster"] == "feature":
        roster = [me for me in methods if is_feature_method(bl, me, datasets)]
    else:
        roster = methods
    if real_only:
        roster_display = [me for me in roster if me not in CONTROLS]
    else:
        roster_display = roster

    condo_ranks = []
    for ds in datasets:
        S = derive_set(bl, results_dir, ds) if per_dataset else profile["metrics"]
        condo_scores = load_condo(results_dir, ds)
        # Scaling pool: every method in the (feature-filtered) roster + condo,
        # controls INCLUDED — they anchor the min-max scale as in openproblems.
        pool = [bl[(me, ds)] for me in roster] + [condo_scores]
        mm = metric_minmax(pool, S)

        table = []
        for me in roster_display:
            c = composite(bl[(me, ds)], S, mm, kind)
            if c is not None:
                table.append((me, c))
        cc = composite(condo_scores, S, mm, kind)
        if cc is not None:
            table.append((condo_label, cc))
        table.sort(key=lambda x: -x[1])
        rank = next((i for i, (n, _) in enumerate(table, 1)
                     if n == condo_label), None)
        if rank:
            condo_ranks.append((ds, rank, len(table), cc))
        extra = ""
        if per_dataset:
            extra = f"  |S|={len(S)}: {S}"
        print(f"\n=== {ds}  (condo rank {rank}/{len(table)}){extra} ===")
        for i, (n, c) in enumerate(table, 1):
            mark = "  <== condo" if n == condo_label else ""
            print(f"  {i:2d}. {n:34s} {c:.4f}{mark}")

    if condo_ranks:
        print(f"\n=== {name}: condo summary ===")
        for ds, r, n, c in condo_ranks:
            print(f"  {ds:22s} rank {r}/{n}   composite={c:.4f}")


def aggregate_across_datasets(
    bl, methods, results_dir, datasets, condo_label,
    real_only=True, profile_names=None,
):
    """For each profile, summarise per-method performance across the given
    datasets, restricted to methods scoreable (finite composite) on *all*
    datasets. For composite/bio/batch, prints:
      - avg rank within the full-coverage pool, per dataset, averaged
      - improvability = mean over datasets of (best_score - this_score) * 100,
        where best_score is the top of the full-coverage pool for that
        dataset+kind. Lower is better; 0 means this method is the top on
        every dataset for that kind.
    """
    kinds = ("composite", "bio", "batch")
    if profile_names is None:
        profile_names = list(PROFILES)

    for pname in profile_names:
        prof = PROFILES[pname]
        feature = prof["roster"] == "feature"
        roster = [me for me in methods
                  if (not feature or is_feature_method(bl, me, datasets))]
        if real_only:
            roster_display = [me for me in roster if me not in CONTROLS]
        else:
            roster_display = roster

        # scores[method][dataset][kind] = scaled value (or None)
        method_names = list(roster_display) + [condo_label]
        scores = {m: {} for m in method_names}

        for ds in datasets:
            S = (derive_set(bl, results_dir, ds)
                 if prof["metrics"] == "per_dataset" else prof["metrics"])
            condo_scores = load_condo(results_dir, ds)
            # full scaling pool (incl. controls as anchors) + condo
            pool = [bl[(me, ds)] for me in roster] + [condo_scores]
            mm = metric_minmax(pool, S)
            for me in roster_display:
                scores[me][ds] = {
                    k: composite(bl[(me, ds)], S, mm, k) for k in kinds
                }
            scores[condo_label][ds] = {
                k: composite(condo_scores, S, mm, k) for k in kinds
            }

        # Methods with finite composite on every dataset.
        full = [m for m in method_names
                if all(scores[m].get(ds, {}).get("composite") is not None
                       for ds in datasets)]
        if not full:
            print(f"\n  [{pname}] no methods with full coverage across "
                  f"{len(datasets)} datasets")
            continue

        # Per-(dataset, kind) best score among full-coverage pool.
        best = {ds: {k: max(scores[m][ds][k] for m in full
                            if scores[m][ds][k] is not None)
                     for k in kinds}
                for ds in datasets}

        # Per-method, per-kind: ranks (within full pool) and improvability.
        agg = []
        for m in full:
            row = {"method": m}
            for k in kinds:
                vals = sorted(
                    [(mm_, scores[mm_][ds][k]) for ds in datasets
                     for mm_ in full],
                    key=lambda x: x[0],
                )
                # rank per dataset within full pool
                ranks_per_ds = []
                gaps_per_ds = []
                for ds in datasets:
                    rows = sorted(
                        [(mm_, scores[mm_][ds][k]) for mm_ in full
                         if scores[mm_][ds][k] is not None],
                        key=lambda x: -x[1],
                    )
                    rk = next((i for i, (mm_, _) in enumerate(rows, 1)
                               if mm_ == m), None)
                    if rk is not None:
                        ranks_per_ds.append(rk)
                    if (scores[m][ds][k] is not None
                            and best[ds][k] is not None):
                        gaps_per_ds.append(best[ds][k] - scores[m][ds][k])
                row[f"rank_{k}"] = (sum(ranks_per_ds) / len(ranks_per_ds)
                                    if ranks_per_ds else float("nan"))
                row[f"imp_{k}"] = (sum(gaps_per_ds) / len(gaps_per_ds) * 100
                                   if gaps_per_ds else float("nan"))
            agg.append(row)

        agg.sort(key=lambda r: r["rank_composite"])

        print(f"\n{'#' * 80}")
        print(f"# Cross-dataset summary ({pname}) over {len(datasets)} "
              f"datasets — {len(full)} methods with full coverage")
        print("#   rk_X  = mean rank within the full-coverage pool on metric X")
        print("#   imp_X = mean (best_score - this_score) * 100 on metric X "
              "(0 = always top; lower = better)")
        print(f"{'#' * 80}")
        header = (f"  {'method':28s}  "
                  f"{'rk_cmp':>6} {'imp_cmp':>7}   "
                  f"{'rk_bio':>6} {'imp_bio':>7}   "
                  f"{'rk_bat':>6} {'imp_bat':>7}")
        print(header)
        for row in agg:
            mark = " <==" if row["method"] == condo_label else ""
            print(
                f"  {row['method']:28s}  "
                f"{row['rank_composite']:>6.2f} {row['imp_composite']:>7.2f}   "
                f"{row['rank_bio']:>6.2f} {row['imp_bio']:>7.2f}   "
                f"{row['rank_batch']:>6.2f} {row['imp_batch']:>7.2f}"
                f"{mark}"
            )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", required=True,
                    help="condo sweep results dir, e.g. "
                         "work/fullbench_v3_agglomerative/results")
    ap.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    ap.add_argument("--condo-label", default="condo_v3")
    ap.add_argument("--profile", choices=list(PROFILES) + ["both"],
                    default="both")
    ap.add_argument("--include-controls", action="store_true",
                    help="keep the oracle/shuffle/no-integration controls "
                         "(dropped by default)")
    ap.add_argument("--score", choices=["composite", "bio", "batch"],
                    default="composite",
                    help="rank by full composite (default), bio only, "
                         "or batch only")
    ap.add_argument("--no-aggregate", dest="aggregate", action="store_false",
                    help="skip the cross-dataset summary block")
    ap.add_argument("--inject", action="append", default=[],
                    metavar="METHOD:DATASET:JSON_PATH",
                    help="override the baseline entry for (method, dataset) "
                         "with metrics from a local result JSON. Repeatable. "
                         "Useful for filling gaps in the published baselines "
                         "(e.g. scvi/scanvi on tabula_sapiens).")
    args = ap.parse_args()

    bl, methods = load_baselines(BASELINES_YAML)
    if args.inject:
        bl, methods = inject_local_results(bl, methods, args.inject)
    results_dir = Path(args.results_dir)

    profiles = list(PROFILES) if args.profile == "both" else [args.profile]
    for name in profiles:
        run_profile(name, PROFILES[name], bl, methods, results_dir,
                    args.datasets, args.condo_label,
                    real_only=not args.include_controls, kind=args.score)

    if args.aggregate:
        aggregate_across_datasets(
            bl, methods, results_dir, args.datasets, args.condo_label,
            real_only=not args.include_controls,
            profile_names=profiles,
        )


if __name__ == "__main__":
    main()
