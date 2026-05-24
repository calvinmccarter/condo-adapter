"""ConDoAdapterMMD with the Muon optimizer route.

Smoke tests that the Muon path:
1. Constructs and steps without errors for both transform_types.
2. Splits parameters correctly into Muon (2D) and AdamW (1D) groups via
   ``make_param_groups``.
3. Actually fits a toy categorical-confounder problem better than the
   initialization (MSE strictly decreases).
"""
import numpy as np
import pytest
import torch

from condo import ConDoAdapterMMD
from condo.utils import LinearAdapter
from condo.muon import Muon, make_param_groups


def _toy(rng, n_per_class=60, batch_m=2.0, batch_b=5.0):
    classes = np.array([["a"] * n_per_class + ["b"] * n_per_class]).reshape(-1, 1)
    mu = {"a": np.array([10.0, -2.0]), "b": np.array([3.0, 7.0])}
    sigma = {"a": np.array([1.0, 1.5]), "b": np.array([2.0, 1.0])}

    def _draw(Z):
        out = np.zeros((Z.shape[0], 2))
        for k, m in mu.items():
            mask = (Z[:, 0] == k)
            out[mask] = rng.normal(m, sigma[k], size=(mask.sum(), 2))
        return out

    T = _draw(classes)
    Strue = _draw(classes)
    Sbatch = batch_m * Strue + batch_b
    return Sbatch.astype(np.float32), T.astype(np.float32), classes, classes


def test_make_param_groups_splits_by_ndim():
    # location-scale: 1D M + 1D b → both in AdamW group.
    adapter = LinearAdapter("location-scale", 4, 4)
    groups = make_param_groups(adapter)
    assert len(groups) == 1
    assert groups[0]["use_muon"] is False
    assert all(p.ndim < 2 for p in groups[0]["params"])

    # square affine: 2D M + 1D b → Muon group and AdamW group.
    adapter = LinearAdapter("affine", 4, 4)
    groups = make_param_groups(adapter)
    assert len(groups) == 2
    by_flag = {g["use_muon"]: g for g in groups}
    assert any(p.ndim >= 2 for p in by_flag[True]["params"])
    assert all(p.ndim < 2 for p in by_flag[False]["params"])


def test_muon_step_runs():
    """One full optimizer step on a minimal model; mainly checking that
    the Newton-Schulz path and the AdamW fallback both execute without
    shape errors."""
    adapter = LinearAdapter("affine", 4, 4)
    opt = Muon(make_param_groups(adapter), lr=1e-2, weight_decay=1e-2)
    x = torch.randn(2, 3, 4)
    y = adapter(x).sum()
    y.backward()
    opt.step()  # would raise if NS reshape or AdamW path were broken


@pytest.mark.parametrize("transform_type", ["location-scale", "affine"])
def test_condo_mmd_with_muon_fits_toy(transform_type):
    rng = np.random.RandomState(0)
    Sbatch, T, Z_S, Z_T = _toy(rng)
    Strue = (Sbatch - 5.0) / 2.0

    cder = ConDoAdapterMMD(
        transform_type=transform_type,
        n_bootstraps=4,
        batch_size=4,
        n_epochs=10,
        learning_rate=2e-2,  # Muon's recommended default ~10x AdamW
        verbose=0,
        optimizer="muon",
    )
    cder.fit(Sbatch, T, Z_S, Z_T)
    Sadapted = cder.transform(Sbatch)
    pre = np.mean((Sbatch - Strue) ** 2)
    post = np.mean((Sadapted - Strue) ** 2)
    assert post < pre, (
        f"Muon-trained {transform_type} adapter should reduce MSE: "
        f"pre={pre} post={post}"
    )


def test_condo_mmd_optimizer_choice_validated():
    with pytest.raises(ValueError, match="optimizer must be adamw or muon"):
        ConDoAdapterMMD(optimizer="lookahead")
