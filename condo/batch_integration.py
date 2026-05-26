"""Agglomerative graph-walking batch integration via ConDo adapters.

This module provides a single high-level function,
:func:`agglomerative_integrate`, which integrates multi-batch data by
walking a *compatibility graph* over batches and greedily merging them
into a growing target pool.

The graph has one node per batch and an edge between two batches iff
they share at least one confounder value (e.g. at least one cell type
present in both). Starting from a seed batch (default: highest
``batch_score``), at each iteration we pick the highest-``batch_score``
batch among the compatibility-graph neighbors of the current target
*set*, fit a ConDo adapter mapping that source's cells onto the current
target distribution, and merge the transformed source cells into the
target pool. Subsequent fits therefore see a richer reference
distribution.

Two practical benefits over fixed-target ConDo:

1. **No zero-overlap fits.** Source batches are only ever paired with a
   target that shares confounder values with them, because we only
   merge across graph edges. ``condo.product_prior`` never sees a
   degenerate input.
2. **Reference grows with every merge.** Later sources are fit against
   a larger and more diverse pool than just the original target,
   instead of every source being adapted to a single (possibly small)
   batch.

Batches in connected components of the graph disjoint from the seed
are *unreachable*; their cells stay raw in the output and are listed
in :attr:`AgglomerativeIntegrationResult.unreachable`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import numpy as np


@dataclass
class AgglomerativeIntegrationResult:
    """Output of :func:`agglomerative_integrate`.

    Attributes:
        Y_out: integrated feature matrix, same shape and order as the
            input ``Y``. Cells from the seed batch are unchanged. Cells
            from each merged source batch are replaced with their
            ConDo-transformed values. Cells from unreachable batches
            are unchanged.
        merge_order: batch labels in the order they were merged into
            the target set. Does not include the seed.
        initial_target: the seed batch label.
        unreachable: batches that were never merged because they have
            no graph path to the seed.
        compatibility: adjacency dict, ``batch -> list of compatible
            batches``, computed once at the start.
        batch_score: the per-batch scores used for ranking.
    """

    Y_out: np.ndarray
    merge_order: List[str]
    initial_target: str
    unreachable: List[str]
    compatibility: Dict[str, List[str]]
    batch_score: Dict[str, float]


def build_compatibility_graph(
    batches: np.ndarray, confounders: np.ndarray
) -> Dict[str, List[str]]:
    """Return an adjacency dict over distinct batch labels.

    Two batches are connected iff their confounder-value sets intersect.

    Args:
        batches: (n,) array of batch labels (any hashable dtype).
        confounders: (n,) or (n, 1) array of confounder labels (any hashable
            dtype). Must align with ``batches`` row-by-row.

    Returns:
        ``{batch_label: [list of compatible batch labels]}`` for every
        unique batch label.
    """
    batches_1d = np.asarray(batches).reshape(-1)
    confounders_1d = np.asarray(confounders).reshape(-1)
    if batches_1d.shape[0] != confounders_1d.shape[0]:
        raise ValueError("batches and confounders must align row-wise")

    unique_batches = np.unique(batches_1d).tolist()
    per_batch_confounders: Dict[Any, set] = {}
    for b in unique_batches:
        mask = batches_1d == b
        per_batch_confounders[b] = set(np.unique(confounders_1d[mask]).tolist())

    adj: Dict[str, List[str]] = {b: [] for b in unique_batches}
    for i, b1 in enumerate(unique_batches):
        for b2 in unique_batches[i + 1 :]:
            if per_batch_confounders[b1] & per_batch_confounders[b2]:
                adj[b1].append(b2)
                adj[b2].append(b1)
    return adj


def agglomerative_integrate(
    Y: np.ndarray,
    batches: np.ndarray,
    confounders: np.ndarray,
    *,
    batch_score: Dict[Any, float],
    adapter_factory: Callable[[], Any],
    initial_target: Optional[Any] = None,
    verbose: bool = True,
) -> AgglomerativeIntegrationResult:
    """Integrate multi-batch data by greedy compatibility-graph walking.

    Args:
        Y: (n_obs, n_features) array. Row ``i`` is cell ``i`` in batch
            ``batches[i]`` with confounder ``confounders[i]``.
        batches: (n_obs,) array of batch labels.
        confounders: (n_obs,) array of confounder values; two batches
            are compatible iff their confounder sets intersect.
        batch_score: mapping from batch label to a float used for
            ranking. The seed (if ``initial_target`` is None) is the
            argmax; at each iteration the next source is the argmax over
            compatible neighbors of the current target set.
        adapter_factory: zero-arg callable returning a fresh adapter
            instance, e.g. ``lambda: ConDoAdapterKLD()``. Each merge
            step calls ``adapter.fit(Ys, Yt, Zs, Zt)`` and
            ``adapter.transform(Ys)``; ConDo's adapters expect
            confounders shaped as ``(n, 1)``.
        initial_target: batch label to seed with. Default: ``argmax
            batch_score``.
        verbose: print each merge step.

    Returns:
        :class:`AgglomerativeIntegrationResult` carrying the integrated
        matrix and provenance information.

    Raises:
        ValueError: if shapes mismatch or ``initial_target`` is not a
            valid batch label.
        KeyError: if ``batch_score`` lacks a score for some batch.
    """
    batches_1d = np.asarray(batches).reshape(-1)
    confounders_1d = np.asarray(confounders).reshape(-1)
    if Y.shape[0] != batches_1d.shape[0]:
        raise ValueError(
            f"Y has {Y.shape[0]} rows but batches has {batches_1d.shape[0]}"
        )
    if confounders_1d.shape[0] != batches_1d.shape[0]:
        raise ValueError(
            "batches and confounders must align row-wise"
        )

    unique_batches = np.unique(batches_1d).tolist()
    missing = [b for b in unique_batches if b not in batch_score]
    if missing:
        raise KeyError(f"batch_score missing entries for: {missing}")

    adj = build_compatibility_graph(batches_1d, confounders_1d)

    seed = initial_target if initial_target is not None else max(
        unique_batches, key=lambda b: batch_score[b]
    )
    if seed not in unique_batches:
        raise ValueError(
            f"initial_target {seed!r} not in batches: {unique_batches}"
        )

    if verbose:
        print(
            f">> agglomerative seed = {seed} "
            f"(score={batch_score[seed]:.4f})",
            flush=True,
        )

    target_set: set = {seed}
    Y_out = Y.copy()
    merge_order: List[str] = []

    # ConDo's product_prior expects (n, 1) confounder arrays.
    Z_col = confounders_1d.reshape(-1, 1)

    while True:
        candidates: set = set()
        for b in target_set:
            for n in adj[b]:
                if n not in target_set:
                    candidates.add(n)
        if not candidates:
            break

        next_source = max(candidates, key=lambda b: batch_score[b])

        target_mask = np.isin(batches_1d, list(target_set))
        src_mask = batches_1d == next_source

        Yt = Y_out[target_mask]
        Zt = Z_col[target_mask]
        # Source uses raw Y (it has not been transformed before — each
        # batch is merged at most once).
        Ys = Y[src_mask]
        Zs = Z_col[src_mask]

        if verbose:
            shared = set(np.unique(Zt.ravel()).tolist()) & set(
                np.unique(Zs.ravel()).tolist()
            )
            print(
                f">> merge {next_source} -> target "
                f"(target_n={int(target_mask.sum())}, "
                f"src_n={int(src_mask.sum())}, "
                f"shared_confounders={len(shared)})",
                flush=True,
            )

        adapter = adapter_factory()
        adapter.fit(Ys, Yt, Zs, Zt)
        Y_out[src_mask] = adapter.transform(Ys)
        target_set.add(next_source)
        merge_order.append(next_source)

    unreachable = [b for b in unique_batches if b not in target_set]
    if verbose and unreachable:
        print(
            f">> unreachable batches (left raw): {unreachable}",
            flush=True,
        )

    return AgglomerativeIntegrationResult(
        Y_out=Y_out,
        merge_order=merge_order,
        initial_target=seed,
        unreachable=unreachable,
        compatibility=adj,
        batch_score=batch_score,
    )
