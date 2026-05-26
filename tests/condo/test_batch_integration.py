"""Tests for the agglomerative graph-walking integrator."""
import numpy as np
import pytest

from condo import (
    ConDoAdapterKLD,
    agglomerative_integrate,
    build_compatibility_graph,
)


def _make_dataset(rng, batch_specs, n_per_celltype=40):
    """Build a synthetic dataset.

    batch_specs: dict batch_label -> {celltype_label -> (mu_vec, sigma_vec, shift_vec)}
    where shift_vec is a per-batch additive shift (so batches differ by a
    constant additive offset per cell-type cluster).

    Returns: (Y, batches, celltypes) numpy arrays.
    """
    rows_Y = []
    rows_b = []
    rows_c = []
    for batch, cts in batch_specs.items():
        for ct, (mu, sigma, shift) in cts.items():
            x = rng.normal(mu, sigma, size=(n_per_celltype, mu.shape[0]))
            x = x + shift
            rows_Y.append(x)
            rows_b += [batch] * n_per_celltype
            rows_c += [ct] * n_per_celltype
    Y = np.vstack(rows_Y).astype(np.float32)
    return Y, np.asarray(rows_b, dtype="U"), np.asarray(rows_c, dtype="U")


def test_build_compatibility_graph_basic():
    batches = np.array(["A", "A", "B", "B", "C", "C", "D", "D"], dtype="U")
    cell_types = np.array(
        ["x", "y", "y", "z", "z", "w", "q", "q"], dtype="U"
    )
    # A covers {x,y}; B {y,z}; C {z,w}; D {q}
    # A-B share y; B-C share z; A-C: none; D: none.
    adj = build_compatibility_graph(batches, cell_types)
    assert set(adj["A"]) == {"B"}
    assert set(adj["B"]) == {"A", "C"}
    assert set(adj["C"]) == {"B"}
    assert adj["D"] == []  # disconnected


def test_agglomerative_seed_default_argmax_and_merges_only_compatible():
    """Three batches in a path A-B-C with scores A<B<C. Seed should be
    C; first merge should be B (its only compatible neighbor); second
    merge should be A (now compatible with B in the target set)."""
    rng = np.random.RandomState(0)
    mu_x = np.array([0.0, 0.0])
    mu_y = np.array([5.0, 5.0])
    sigma = np.array([0.4, 0.4])
    no_shift = np.zeros(2)
    shift_b = np.array([0.0, 0.0])  # we'll let pre_asw scores drive ordering

    Y, batches, cts = _make_dataset(
        rng,
        {
            "A": {"x": (mu_x, sigma, no_shift)},  # only has x
            "B": {"x": (mu_x, sigma, no_shift), "y": (mu_y, sigma, no_shift)},
            "C": {"y": (mu_y, sigma, no_shift)},  # only has y
        },
    )
    scores = {"A": 0.5, "B": 0.7, "C": 0.9}

    factory = lambda: ConDoAdapterKLD(transform_type="location-scale", verbose=0)
    res = agglomerative_integrate(
        Y, batches, cts, batch_score=scores, adapter_factory=factory, verbose=False,
    )
    assert res.initial_target == "C"
    # B is C's only neighbor (shared y); A becomes reachable after B joins.
    assert res.merge_order == ["B", "A"]
    assert res.unreachable == []
    assert res.Y_out.shape == Y.shape


def test_agglomerative_skips_disconnected_components():
    """D is isolated (no shared cell type with A,B,C); must end up
    unreachable and left at raw values."""
    rng = np.random.RandomState(0)
    mu_x = np.array([0.0, 0.0])
    mu_y = np.array([5.0, 5.0])
    mu_q = np.array([10.0, 10.0])
    sigma = np.array([0.4, 0.4])
    z = np.zeros(2)

    Y, batches, cts = _make_dataset(
        rng,
        {
            "A": {"x": (mu_x, sigma, z), "y": (mu_y, sigma, z)},
            "B": {"y": (mu_y, sigma, z)},
            "D": {"q": (mu_q, sigma, z)},  # disconnected
        },
    )
    scores = {"A": 0.9, "B": 0.5, "D": 0.99}  # D has highest score but isolated

    factory = lambda: ConDoAdapterKLD(transform_type="location-scale", verbose=0)
    res = agglomerative_integrate(
        Y, batches, cts, batch_score=scores, adapter_factory=factory, verbose=False,
    )
    # Seed must be argmax. D is the argmax but it's its own connected
    # component — that's fine; only A,B are unreachable.
    assert res.initial_target == "D"
    assert res.merge_order == []
    assert set(res.unreachable) == {"A", "B"}

    # Confirm D's cells are unchanged.
    d_mask = batches == "D"
    np.testing.assert_array_equal(res.Y_out[d_mask], Y[d_mask])
    a_mask = batches == "A"
    np.testing.assert_array_equal(res.Y_out[a_mask], Y[a_mask])


def test_agglomerative_with_explicit_initial_target():
    """Force the seed to be a non-argmax batch and verify the walk uses
    the explicit seed."""
    rng = np.random.RandomState(0)
    mu_x = np.array([0.0, 0.0])
    mu_y = np.array([5.0, 5.0])
    sigma = np.array([0.4, 0.4])
    z = np.zeros(2)

    Y, batches, cts = _make_dataset(
        rng,
        {
            "A": {"x": (mu_x, sigma, z), "y": (mu_y, sigma, z)},
            "B": {"y": (mu_y, sigma, z)},
            "C": {"x": (mu_x, sigma, z)},
        },
    )
    scores = {"A": 0.9, "B": 0.5, "C": 0.7}

    factory = lambda: ConDoAdapterKLD(transform_type="location-scale", verbose=0)
    res = agglomerative_integrate(
        Y, batches, cts,
        batch_score=scores,
        adapter_factory=factory,
        initial_target="B",
        verbose=False,
    )
    assert res.initial_target == "B"
    # From B: only A is compatible (shares y).
    # After {B, A}: A also opens up C via x.
    assert res.merge_order == ["A", "C"]


def test_agglomerative_actually_corrects_batch_shift():
    """A simple sanity check: with a large per-batch additive shift,
    after agglomerative integration the per-cell-type cluster centroids
    should be closer across batches than before."""
    rng = np.random.RandomState(0)
    mu_x = np.array([0.0, 0.0])
    mu_y = np.array([5.0, 5.0])
    sigma = np.array([0.3, 0.3])
    no_shift = np.zeros(2)
    big_shift = np.array([10.0, -10.0])  # batch B is far from A
    bigger_shift = np.array([-8.0, 7.0])  # batch C is also far

    Y, batches, cts = _make_dataset(
        rng,
        {
            "A": {"x": (mu_x, sigma, no_shift), "y": (mu_y, sigma, no_shift)},
            "B": {"x": (mu_x, sigma, big_shift), "y": (mu_y, sigma, big_shift)},
            "C": {"x": (mu_x, sigma, bigger_shift), "y": (mu_y, sigma, bigger_shift)},
        },
        n_per_celltype=80,
    )
    # Score: A best (centroids near canonical positions); B and C lower.
    scores = {"A": 0.9, "B": 0.5, "C": 0.4}

    factory = lambda: ConDoAdapterKLD(transform_type="location-scale", verbose=0)
    res = agglomerative_integrate(
        Y, batches, cts, batch_score=scores, adapter_factory=factory, verbose=False,
    )
    assert res.initial_target == "A"
    assert set(res.merge_order) == {"B", "C"}

    def per_batch_x_centroid(arr, batches, cts, b):
        m = (batches == b) & (cts == "x")
        return arr[m].mean(axis=0)

    pre_dist_AB = np.linalg.norm(
        per_batch_x_centroid(Y, batches, cts, "A")
        - per_batch_x_centroid(Y, batches, cts, "B")
    )
    post_dist_AB = np.linalg.norm(
        per_batch_x_centroid(res.Y_out, batches, cts, "A")
        - per_batch_x_centroid(res.Y_out, batches, cts, "B")
    )
    assert post_dist_AB < 0.5 * pre_dist_AB, (
        f"x-cluster centroid distance A→B should shrink dramatically; "
        f"pre={pre_dist_AB:.2f} post={post_dist_AB:.2f}"
    )


def test_agglomerative_validates_shapes_and_scores():
    Y = np.zeros((10, 3))
    batches = np.array(["A"] * 10, dtype="U")
    cts = np.array(["x"] * 9, dtype="U")  # wrong length
    with pytest.raises(ValueError):
        agglomerative_integrate(
            Y, batches, cts,
            batch_score={"A": 1.0}, adapter_factory=lambda: None, verbose=False,
        )

    batches = np.array(["A", "B"] * 5, dtype="U")
    cts = np.array(["x"] * 10, dtype="U")
    with pytest.raises(KeyError):
        # Missing score for B
        agglomerative_integrate(
            Y, batches, cts,
            batch_score={"A": 1.0}, adapter_factory=lambda: None, verbose=False,
        )
