"""Frozen ConDo runner for the paper's cell-type-label-availability experiment.

PROVENANCE / STATUS -- read before reusing:
  This is a frozen copy of the ConDo agglomerative runner that was contributed
  to and merged into the OpenProblems ``task_batch_integration`` repository
  (``src/methods/.../condo_runner.py``). On top of that upstreamed version it
  carries two additions used only for this paper:
    1. read / fit / write timing (``uns["condo_timing"]`` + a ``CONDO_TIMING``
       log line), and
    2. label dropout -- ``_apply_label_dropout`` plus its call site in
       ``run_condo`` (params ``label_dropout`` / ``dropout_handling`` /
       ``dropout_seed``), which hides a fraction of cell-type labels from the
       fit to probe robustness to missing annotation.

  It is kept HERE, in condo-adapter, for reference and reproducibility only:
    * The production method upstream is deliberately left unchanged (the
      dropout is a paper probe, not method behaviour).
    * The actual paper runs executed from the fork checkout, not this file;
      ``run_local.py`` is NOT wired to import it. ``read_anndata_partial.py``
      is vendored alongside so this file's dependencies are self-contained.
  Not part of the production method; do not import into the shipped pipeline.

Runs the agglomerative batch integrator: pick the seed batch with the
highest per-batch pre-integration silhouette of cell_type on X_pca, then
iteratively merge each next-best compatible (cell-type-overlapping)
neighbour by fitting a ConDo adapter conditioned on cell_type. Batches
in disconnected components of the compatibility graph are passed through
untouched.

Supports two orthogonal axes (set via the ``par`` dict in the viash
script):

* ``transform_type`` : 'location-scale' | 'affine'
* ``rep``            : 'features' | 'pca'
    - 'features': fit on normalized expression, write corrected_counts.
    - 'pca':      fit on obsm['X_pca'], write obsm['X_emb'] (embedding).
* ``hvg_only`` (features only): restrict fit to var['hvg'] columns; pass
  through non-HVGs at source values.
"""
from __future__ import annotations

import sys
import time
from typing import Any

import anndata as ad
import numpy as np
from scipy.sparse import csr_matrix, issparse


def _to_dense(x: Any) -> np.ndarray:
    return x.toarray() if issparse(x) else np.asarray(x)


def _apply_label_dropout(cell_types, batches, frac, handling, seed):
    """Randomly hide a fraction of cell-type labels from the ConDo fit.

    Only the labels condo sees at fit time are perturbed; evaluation still uses
    the true labels. Two handlings:

    * ``bucket``  : hidden cells share one ``"Unknown"`` value, which condo
      conditions on like any other cell type (frac=1.0 => a single category =>
      unconditioned).
    * ``excluded``: hidden cells get a per-batch-unique sentinel, so
      product_prior gives them zero weight (never sampled, never a shared
      condition) and they do not link batches in the compatibility graph --
      but transform still applies to them. Undefined at frac=1.0.
    """
    n = cell_types.shape[0]
    k = int(round(frac * n))
    if k <= 0:
        return cell_types, 0
    rng = np.random.default_rng(seed)
    mask = np.zeros(n, dtype=bool)
    mask[rng.choice(n, size=k, replace=False)] = True
    ct = cell_types.astype(object).copy()
    if handling == "bucket":
        ct[mask] = "Unknown"
    elif handling == "excluded":
        bidx = np.searchsorted(np.unique(batches), batches).astype("U")
        ct[mask] = np.char.add("__drop", bidx)[mask]
    else:
        raise ValueError(f"unknown dropout_handling {handling!r}")
    return np.asarray(ct.tolist(), dtype="U"), int(k)


def _build_adapter(par: dict[str, Any]):
    from condo import ConDoAdapterMMD

    return ConDoAdapterMMD(
        transform_type=par["transform_type"],
        bootstrap_fraction=float(par.get("bootstrap_fraction", 1.0)),
        n_epochs=int(par.get("n_epochs", 5)),
        learning_rate=float(par.get("learning_rate", 1e-3)),
        mmd_size=int(par.get("mmd_size", 40)),
        batch_size=int(par.get("batch_size", 8)),
        weight_decay=float(par.get("weight_decay", 1e-4)),
        patience=int(par.get("patience", 3)),
        random_state=int(par.get("random_state", 42)),
        verbose=0,
        device=par.get("device", "cpu"),
    )


def _read_input(par: dict, meta: dict) -> ad.AnnData:
    sys.path.append(meta["resources_dir"])
    from read_anndata_partial import read_anndata

    rep = par.get("rep", "features")
    # The agglomerative seed selection needs obsm['X_pca'] for per-batch
    # pre-integration silhouette; pull it in both rep modes.
    if rep == "features":
        return read_anndata(
            par["input"], X="layers/normalized",
            obs="obs", obsm="obsm", var="var", uns="uns",
        )
    if rep == "pca":
        return read_anndata(
            par["input"], obs="obs", obsm="obsm", var="var", uns="uns"
        )
    raise ValueError(f"Unknown rep: {rep!r}")


def _pick_target_by_pre_asw(
    adata: ad.AnnData, batches: np.ndarray, cell_types: np.ndarray
) -> tuple[str, dict]:
    """Compute per-batch pre-integration silhouette of cell_type on
    ``obsm['X_pca']``. Returns ``(argmax_batch_label, per_batch_asw_table)``.
    Used as the seed/scoring criterion for the agglomerative integrator."""
    from scib.metrics import silhouette

    if "X_pca" not in adata.obsm:
        raise ValueError(
            "agglomerative seed selection requires obsm['X_pca']; "
            f"obsm keys present: {list(adata.obsm.keys())}"
        )
    batch_labels = np.unique(batches)
    per_batch: dict[str, float] = {}
    for b in batch_labels:
        mask = batches == b
        if mask.sum() < 4:
            per_batch[b] = float("-inf")
            continue
        sub = adata[mask].copy()
        cts = sub.obs["cell_type"].astype(str)
        keep_cts = cts.value_counts()[cts.value_counts() >= 2].index
        keep_mask = cts.isin(keep_cts).values
        if keep_mask.sum() < 4 or len(keep_cts) < 2:
            per_batch[b] = float("-inf")
            continue
        sub2 = sub[keep_mask].copy()
        try:
            s = float(silhouette(sub2, label_key="cell_type", embed="X_pca"))
        except Exception:
            s = float("-inf")
        per_batch[b] = s
    best = max(per_batch.items(), key=lambda kv: kv[1])
    return best[0], per_batch


def _per_batch_batch_silhouette(
    adata: ad.AnnData,
    batches: np.ndarray,
    random_state: int,
    max_cells: int = 20000,
) -> dict:
    """Per-batch mean silhouette of the *batch* label on ``obsm['X_pca']``.

    Symmetric analog of the cell-type silhouette used for the baseline
    ranking: instead of "how well separated are cell types within a
    batch", this measures "how well separated is each batch from the
    others" in the pre-integration embedding. Batch silhouette is
    undefined within a single batch, so it is computed globally on a
    batch-stratified subsample (silhouette is O(n^2); the atlases have up
    to ~500k cells). High score => the batch stands apart from the rest
    (strong batch effect); low score => the batch already overlaps the
    others.
    """
    from sklearn.metrics import silhouette_samples

    if "X_pca" not in adata.obsm:
        raise ValueError("batch_silhouette ranking requires obsm['X_pca']")
    X = np.asarray(adata.obsm["X_pca"])
    n = X.shape[0]
    rng = np.random.default_rng(random_state)
    unique = np.unique(batches)

    # Batch-stratified subsample to ~max_cells so every batch is
    # represented and the silhouette stays tractable.
    if n > max_cells:
        per = max(2, max_cells // len(unique))
        idx_parts = []
        for b in unique:
            bidx = np.flatnonzero(batches == b)
            if bidx.size > per:
                bidx = rng.choice(bidx, size=per, replace=False)
            idx_parts.append(bidx)
        sub_idx = np.concatenate(idx_parts)
    else:
        sub_idx = np.arange(n)

    Xs = X[sub_idx]
    bs = batches[sub_idx]
    per_batch: dict = {b: float("nan") for b in unique}
    if len(np.unique(bs)) < 2:
        # Degenerate: only one batch present -> no separation to speak of.
        return {b: 0.0 for b in unique}
    sil = silhouette_samples(Xs, bs)
    for b in unique:
        m = bs == b
        if m.sum() > 0:
            per_batch[b] = float(sil[m].mean())
    # Any batch missing from the subsample falls back to the global mean.
    finite = [v for v in per_batch.values() if np.isfinite(v)]
    fallback = float(np.mean(finite)) if finite else 0.0
    for b in unique:
        if not np.isfinite(per_batch[b]):
            per_batch[b] = fallback
    return per_batch


def _compute_batch_score(
    adata: ad.AnnData,
    batches: np.ndarray,
    cell_types: np.ndarray,
    strategy: str,
    random_state: int,
) -> tuple[dict, str]:
    """Return ``(batch_score, description)`` for an agglomerative ranking
    strategy.

    ``batch_score`` is consumed by :func:`agglomerative_integrate` as BOTH
    the seed criterion (``argmax``) and the compatible-neighbour ranking
    (``argmax`` over graph neighbours of the current target set). So the
    chosen strategy fully determines the seed *and* the merge order.

    Strategies (ablations around the ``celltype_silhouette`` baseline):

    * ``celltype_silhouette`` : per-batch cell-type silhouette on X_pca,
      highest first. This is the ``v3_baseline_seeded`` behaviour.
    * ``celltype_silhouette_low`` : same score, LOWEST first (mirror).
    * ``random``              : a fixed per-batch random priority seeded by
      ``random_state`` (reproducible random seed + random neighbour order).
    * ``biggest``             : batch size in cells, biggest first.
    * ``smallest``            : batch size in cells, smallest first (mirror).
    * ``batch_silhouette_low``: per-batch batch silhouette, LOWEST first
      (merge already-mixed batches earliest).
    * ``batch_silhouette_high``: per-batch batch silhouette, HIGHEST first
      (mirrors the cell-type-silhouette-highest baseline).
    """
    unique = np.unique(batches)
    if strategy in ("celltype_silhouette", "celltype_silhouette_low"):
        _, per_batch = _pick_target_by_pre_asw(adata, batches, cell_types)
        if strategy == "celltype_silhouette_low":
            # argmax over negated score -> lowest cell-type silhouette first.
            # -inf sentinels (degenerate batches) become +inf and would be
            # picked first; clamp them to the worst finite so they stay last.
            finite = [v for v in per_batch.values() if np.isfinite(v)]
            worst = min(finite) if finite else 0.0
            per_batch = {b: -(v if np.isfinite(v) else worst)
                         for b, v in per_batch.items()}
            return per_batch, "per-batch cell-type silhouette on X_pca (lowest first)"
        return per_batch, "per-batch cell-type silhouette on X_pca (highest first)"
    if strategy in ("biggest", "smallest"):
        counts = {b: float((batches == b).sum()) for b in unique}
        if strategy == "smallest":
            per_batch = {b: -c for b, c in counts.items()}
            return per_batch, "batch size in cells (smallest first)"
        return counts, "batch size in cells (biggest first)"
    if strategy == "random":
        rng = np.random.default_rng(random_state)
        vals = rng.random(len(unique))
        per_batch = {b: float(v) for b, v in zip(unique, vals)}
        return per_batch, f"fixed random priority (random_state={random_state})"
    if strategy in ("batch_silhouette_low", "batch_silhouette_high"):
        raw = _per_batch_batch_silhouette(adata, batches, random_state)
        sign = -1.0 if strategy == "batch_silhouette_low" else 1.0
        per_batch = {b: sign * raw[b] for b in unique}
        direction = "lowest" if sign < 0 else "highest"
        return (
            per_batch,
            f"per-batch batch silhouette on X_pca ({direction} first)",
        )
    raise ValueError(
        f"Unknown ranking_strategy {strategy!r}; expected one of: "
        "celltype_silhouette, celltype_silhouette_low, random, biggest, "
        "smallest, batch_silhouette_low, batch_silhouette_high"
    )


def _select_feature_columns(adata: ad.AnnData, hvg_only: bool) -> np.ndarray | None:
    """Return a boolean column mask if HVG-only is requested, else None."""
    if not hvg_only:
        return None
    if "hvg" not in adata.var.columns:
        raise ValueError(
            "--hvg_only requires a boolean 'hvg' column in var; "
            f"available columns: {list(adata.var.columns)}"
        )
    mask = adata.var["hvg"].astype(bool).values
    if mask.sum() == 0:
        raise ValueError("--hvg_only requested but no var['hvg'] entries are True")
    return mask


def run_condo(par: dict, meta: dict) -> None:
    rep = par.get("rep", "features")
    hvg_only = bool(par.get("hvg_only", False)) and rep == "features"

    print(f">> Read input (rep={rep}, hvg_only={hvg_only})", flush=True)
    _t_read = time.time()
    adata = _read_input(par, meta)
    read_seconds = time.time() - _t_read
    _t_fit = time.time()

    # condo's product_prior dispatches on dtype.kind ∈ {'U','S'} for discrete
    # confounders; pandas .astype(str).values returns object dtype which
    # slips past that check.
    batches = np.asarray(adata.obs["batch"].astype(str).values, dtype="U")
    cell_types = np.asarray(adata.obs["cell_type"].astype(str).values, dtype="U")

    label_dropout = float(par.get("label_dropout", 0.0))
    if label_dropout > 0:
        handling = par.get("dropout_handling", "bucket")
        seed = int(par.get("dropout_seed", 0))
        cell_types, n_hidden = _apply_label_dropout(
            cell_types, batches, label_dropout, handling, seed
        )
        print(
            f">> label dropout: frac={label_dropout} handling={handling} "
            f"seed={seed} -> {n_hidden}/{len(cell_types)} labels hidden",
            flush=True,
        )

    # Per-batch ranking score consumed by agglomerative_integrate as both
    # the seed criterion (argmax) and the neighbor-ranking score at each
    # merge step. The default 'celltype_silhouette' strategy reproduces
    # v3_baseline_seeded; the other strategies are the paper ablations.
    strategy = par.get("ranking_strategy", "celltype_silhouette")
    random_state = int(par.get("random_state", 42))
    batch_score, score_desc = _compute_batch_score(
        adata, batches, cell_types, strategy, random_state
    )
    print(f">> Agglomerative ranking strategy = {strategy}", flush=True)
    print(f">>   score = {score_desc}; seed = argmax(score)", flush=True)
    for b, s in sorted(batch_score.items(), key=lambda kv: -kv[1]):
        print(f"    score[{b}] = {s:.4f}", flush=True)

    if rep == "features":
        Y_full = _to_dense(adata.X).astype(np.float64)
        hvg_mask = _select_feature_columns(adata, hvg_only)
        if hvg_mask is not None:
            print(
                f">> HVG-only: fitting on {int(hvg_mask.sum())}/{Y_full.shape[1]} genes",
                flush=True,
            )
            Y = Y_full[:, hvg_mask]
        else:
            Y = Y_full
    else:  # pca
        Y_full = None
        hvg_mask = None
        Y = np.asarray(adata.obsm["X_pca"], dtype=np.float64)

    from condo.batch_integration import agglomerative_integrate

    def _adapter_factory():
        return _build_adapter(par)

    result = agglomerative_integrate(
        Y=Y,
        batches=batches,
        confounders=cell_types,
        batch_score=batch_score,
        adapter_factory=_adapter_factory,
        verbose=True,
    )
    Y_out = result.Y_out
    fit_seconds = time.time() - _t_fit
    timing_uns = {"read_seconds": read_seconds, "fit_seconds": fit_seconds}

    print(">> Build output", flush=True)
    if rep == "features":
        if hvg_mask is None:
            corrected = Y_out
        else:
            # Start from the original full-feature matrix; overwrite the HVG
            # columns with the adapted values. Non-HVG columns are pass-through.
            corrected = Y_full.copy()
            corrected[:, hvg_mask] = Y_out
        output = ad.AnnData(
            obs=adata.obs[[]],
            var=adata.var[[]],
            layers={"corrected_counts": csr_matrix(corrected)},
            uns={
                "dataset_id": adata.uns["dataset_id"],
                "normalization_id": adata.uns["normalization_id"],
                "method_id": meta["name"],
                "condo_timing": timing_uns,
            },
        )
    else:  # pca -> embedding output
        output = ad.AnnData(
            obs=adata.obs[[]],
            var=adata.var[[]],
            obsm={"X_emb": Y_out},
            uns={
                "dataset_id": adata.uns["dataset_id"],
                "normalization_id": adata.uns["normalization_id"],
                "method_id": meta["name"],
                "condo_timing": timing_uns,
            },
        )

    print(">> Write output", flush=True)
    _t_write = time.time()
    output.write_h5ad(par["output"], compression="gzip")
    write_seconds = time.time() - _t_write
    print(
        f">> CONDO_TIMING read={read_seconds:.1f}s fit={fit_seconds:.1f}s "
        f"write={write_seconds:.1f}s",
        flush=True,
    )
