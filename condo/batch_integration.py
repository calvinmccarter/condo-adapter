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
class BestFirstIntegrationResult:
    """Output of :func:`bestfirst_integrate`.

    Attributes:
        Y_out: integrated feature matrix, same shape/order as input ``Y``.
            Cells are replaced by their (possibly repeatedly) ConDo-
            transformed values; cells that were only ever in an anchor
            agglomeration are unchanged.
        merge_order: list of ``(target_batches, source_batches,
            target_asw, source_asw)`` tuples, one per merge, in order.
        components: final agglomerations as lists of original batch
            labels. More than one component => some batches were never
            type-compatible and stayed in separate frames.
        n_merges: number of merge steps performed.
    """

    Y_out: np.ndarray
    merge_order: List[tuple]
    components: List[List[Any]]
    n_merges: int


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


def _global_pca(features: np.ndarray, n_pcs: int) -> np.ndarray:
    """Fit a single PCA over *all* cells and return the (n_obs, n_comps)
    coordinates. Replicates the openproblems ``process_dataset`` X_pca
    recipe (PCA on the normalized, batch-aware-HVG matrix) applied to the
    current corrected feature matrix -- the caller is expected to pass that
    matrix (the benchmark dataset is already subset to batch-aware HVGs)."""
    from sklearn.decomposition import PCA

    n_comp = min(n_pcs, features.shape[1], features.shape[0] - 1)
    return PCA(n_components=n_comp, random_state=0).fit_transform(features)


def _silhouette_in_coords(
    coords: np.ndarray,
    labels: np.ndarray,
    *,
    subsample: int,
    rng: np.random.Generator,
) -> float:
    """Uniform-subsample silhouette of ``labels`` on ``coords`` (already a
    shared embedding). Returns ``-inf`` when undefined for ranking (fewer
    than two distinct labels, or too few cells)."""
    from sklearn.metrics import silhouette_score

    labels = np.asarray(labels).reshape(-1)
    n = coords.shape[0]
    if n > subsample:
        idx = rng.choice(n, size=subsample, replace=False)
        coords = coords[idx]
        labels = labels[idx]

    uniq = np.unique(labels)
    # silhouette_score requires 2 <= n_labels <= n_samples - 1.
    if uniq.shape[0] < 2 or coords.shape[0] <= uniq.shape[0]:
        return float("-inf")
    try:
        return float(silhouette_score(coords, labels))
    except ValueError:
        return float("-inf")


def bestfirst_integrate(
    Y: np.ndarray,
    batches: np.ndarray,
    confounders: np.ndarray,
    *,
    adapter_factory: Callable[[], Any],
    asw_subsample: int = 10000,
    n_pcs: int = 50,
    random_state: int = 0,
    verbose: bool = True,
) -> BestFirstIntegrationResult:
    """Best-first ("competitive forest") agglomerative integration.

    Unlike :func:`agglomerative_integrate`, which grows a single fixed
    target pool seeded once with a static per-batch score, here *every*
    current agglomeration competes to be the anchor each round based on
    its **live** asw (cell-type silhouette), recomputed after every merge:

    1. Start with one agglomeration per batch. Score each by asw.
    2. Each round, among agglomerations that have at least one
       type-compatible neighbor (sharing >=1 confounder value), pick the
       **target** = highest asw. Pick the **source** = highest-asw
       compatible neighbor of that target.
    3. Fit a fresh adapter mapping the source's cells onto the target,
       transform them, and merge into a new agglomeration (cells =
       target U transformed-source; types = union).
    4. **Refit a single global PCA** over all cells' current corrected
       features and recompute **every** agglomeration's asw in that shared
       basis. (The basis shifts each merge, so all asws change -- and a
       shared basis is what makes asws comparable across the competing
       agglomerations.)
    5. Repeat until no two remaining agglomerations share a cell type
       (disconnected components stay in separate frames).

    asw is a uniform random subsample (``asw_subsample`` cells) silhouette
    of the confounder labels, computed in the global ``n_pcs``-component
    PCA of the current corrected feature matrix. This replicates the
    openproblems ``process_dataset`` X_pca recipe (PCA on the normalized,
    batch-aware-HVG matrix) -- the caller passes that matrix (the benchmark
    dataset is already subset to batch-aware HVGs) -- but refit on the
    evolving corrected data. Subsampling keeps the repeated O(n^2)
    silhouette tractable; the PCA itself is fit on all cells.

    Args:
        Y: (n_obs, n_features) feature matrix (normalized, batch-aware-HVG
            restricted, to replicate the benchmark's X_pca).
        batches: (n_obs,) batch labels.
        confounders: (n_obs,) confounder values (e.g. cell type).
        adapter_factory: zero-arg callable returning a fresh adapter;
            each merge calls ``adapter.fit(Ys, Yt, Zs, Zt)`` and
            ``adapter.transform(Ys)`` with confounders shaped ``(n, 1)``.
        asw_subsample: max cells used to estimate an agglomeration's asw.
        n_pcs: components for the global PCA asw embedding.
        random_state: seed for the subsampling RNG.
        verbose: print each merge step.

    Returns:
        :class:`BestFirstIntegrationResult`.
    """
    batches_1d = np.asarray(batches).reshape(-1)
    confounders_1d = np.asarray(confounders).reshape(-1)
    if Y.shape[0] != batches_1d.shape[0]:
        raise ValueError(
            f"Y has {Y.shape[0]} rows but batches has {batches_1d.shape[0]}"
        )
    if confounders_1d.shape[0] != batches_1d.shape[0]:
        raise ValueError("batches and confounders must align row-wise")

    rng = np.random.default_rng(random_state)
    Y_work = np.array(Y, dtype=float, copy=True)
    Z_col = confounders_1d.reshape(-1, 1)

    def recompute_all_asw(aggloms: Dict[int, Dict[str, Any]]) -> None:
        """Refit one global PCA on the current corrected features and set
        every agglomeration's asw in that shared basis."""
        coords = _global_pca(Y_work, n_pcs)
        for a in aggloms.values():
            a["asw"] = _silhouette_in_coords(
                coords[a["members"]], confounders_1d[a["members"]],
                subsample=asw_subsample, rng=rng,
            )

    # One agglomeration per batch; score all in the initial global PCA.
    aggloms: Dict[int, Dict[str, Any]] = {}
    next_id = 0
    for b in np.unique(batches_1d).tolist():
        members = np.flatnonzero(batches_1d == b)
        aggloms[next_id] = {
            "members": members,
            "types": set(np.unique(confounders_1d[members]).tolist()),
            "batches": {b},
            "asw": float("-inf"),
        }
        next_id += 1
    recompute_all_asw(aggloms)

    if verbose:
        print(
            f">> bestfirst: {len(aggloms)} batches; asw = uniform subsample"
            f"={asw_subsample} silhouette in a global {n_pcs}-PC PCA, "
            "refit on corrected features every merge",
            flush=True,
        )

    merge_order: List[tuple] = []

    while True:
        ids = list(aggloms)

        def neighbors(i: int) -> List[int]:
            ti = aggloms[i]["types"]
            return [j for j in ids if j != i and (ti & aggloms[j]["types"])]

        eligible = [(i, neighbors(i)) for i in ids]
        eligible = [(i, nb) for i, nb in eligible if nb]
        if not eligible:
            break

        target, nbrs = max(eligible, key=lambda x: aggloms[x[0]]["asw"])
        source = max(nbrs, key=lambda j: aggloms[j]["asw"])

        t, s = aggloms[target], aggloms[source]
        Yt, Zt = Y_work[t["members"]], Z_col[t["members"]]
        Ys, Zs = Y_work[s["members"]], Z_col[s["members"]]

        src_asw, tgt_asw = s["asw"], t["asw"]  # scores that drove selection

        adapter = adapter_factory()
        adapter.fit(Ys, Yt, Zs, Zt)
        Y_work[s["members"]] = adapter.transform(Ys)

        new = {
            "members": np.concatenate([t["members"], s["members"]]),
            "types": t["types"] | s["types"],
            "batches": t["batches"] | s["batches"],
            "asw": float("-inf"),
        }
        del aggloms[target]
        del aggloms[source]
        aggloms[next_id] = new
        # Refit the global PCA on the updated corrected features and rescore
        # every agglomeration in the new shared basis.
        recompute_all_asw(aggloms)

        if verbose:
            print(
                f">> merge source={sorted(s['batches'])} "
                f"(asw={src_asw:.4f}, n={s['members'].size}) "
                f"-> target={sorted(t['batches'])} "
                f"(asw={tgt_asw:.4f}, n={t['members'].size}); "
                f"merged asw={new['asw']:.4f}",
                flush=True,
            )

        merge_order.append(
            (sorted(t["batches"]), sorted(s["batches"]), tgt_asw, src_asw)
        )
        next_id += 1

    components = [sorted(a["batches"]) for a in aggloms.values()]
    if verbose and len(components) > 1:
        print(
            f">> {len(components)} disconnected components left in "
            f"separate frames: {components}",
            flush=True,
        )

    return BestFirstIntegrationResult(
        Y_out=Y_work,
        merge_order=merge_order,
        components=components,
        n_merges=len(merge_order),
    )
