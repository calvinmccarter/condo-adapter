"""Tier-1 GPU support: device kwarg on KLD-affine and MMD adapters.

These tests run on CPU unconditionally; the GPU variants are skipped if
``torch.cuda.is_available()`` is False so the suite stays green on
CPU-only CI.
"""
import numpy as np
import pytest
import torch

from condo import ConDoAdapterKLD, ConDoAdapterMMD


def _toy_categorical_dataset(rng, n_per_class=80, batch_m=2.0, batch_b=5.0):
    """A 2d-feature, 2-class categorical confounder problem."""
    classes = np.array([["a"] * n_per_class + ["b"] * n_per_class]).reshape(-1, 1)
    Z_T = classes.copy()
    Z_S = classes.copy()

    mu = {"a": np.array([10.0, -2.0]), "b": np.array([3.0, 7.0])}
    sigma = {"a": np.array([1.0, 1.5]), "b": np.array([2.0, 1.0])}

    def _draw(Z):
        out = np.zeros((Z.shape[0], 2))
        for k, m in mu.items():
            mask = (Z[:, 0] == k)
            out[mask] = rng.normal(m, sigma[k], size=(mask.sum(), 2))
        return out

    T = _draw(Z_T)
    Strue = _draw(Z_S)
    Sbatch = batch_m * Strue + batch_b
    return Sbatch.astype(np.float32), T.astype(np.float32), Z_S, Z_T


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="no CUDA available"
            ),
        ),
    ],
)
def test_kld_affine_device(device):
    """ConDoAdapterKLD with transform_type='affine' fits successfully on
    the requested device and produces a finite M, b that reduce the
    distance from Sbatch to Strue."""
    rng = np.random.RandomState(0)
    Sbatch, T, Z_S, Z_T = _toy_categorical_dataset(rng)
    Strue = (Sbatch - 5.0) / 2.0  # inverse of batch_m=2, batch_b=5

    cder = ConDoAdapterKLD(transform_type="affine", verbose=0, device=device)
    cder.fit(Sbatch, T, Z_S, Z_T)
    Sadapted = cder.transform(Sbatch)

    assert np.isfinite(cder.M_).all()
    assert np.isfinite(cder.b_).all()
    pre = np.mean((Sbatch - Strue) ** 2)
    post = np.mean((Sadapted - Strue) ** 2)
    assert post < pre, f"adapter on {device} should reduce MSE: pre={pre} post={post}"


@pytest.mark.parametrize("transform_type", ["location-scale", "affine"])
@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="no CUDA available"
            ),
        ),
    ],
)
def test_mmd_device(device, transform_type):
    """ConDoAdapterMMD fits successfully on the requested device."""
    rng = np.random.RandomState(0)
    Sbatch, T, Z_S, Z_T = _toy_categorical_dataset(rng, n_per_class=60)
    Strue = (Sbatch - 5.0) / 2.0

    cder = ConDoAdapterMMD(
        transform_type=transform_type,
        n_bootstraps=4,
        batch_size=4,
        n_epochs=10,
        learning_rate=5e-3,
        verbose=0,
        device=device,
    )
    cder.fit(Sbatch, T, Z_S, Z_T)
    Sadapted = cder.transform(Sbatch)
    if transform_type == "affine":
        assert np.isfinite(cder.M_).all()
    else:
        assert np.isfinite(cder.m_).all()
    assert np.isfinite(cder.b_).all()
    pre = np.mean((Sbatch - Strue) ** 2)
    post = np.mean((Sadapted - Strue) ** 2)
    assert post < pre, f"adapter on {device} should reduce MSE: pre={pre} post={post}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA available")
def test_kld_affine_cpu_cuda_agreement():
    """Same seed + same data should give finite, comparable solutions on
    CPU and CUDA. We don't require bitwise equality (BLAS differences
    between devices are expected) — just that both reduce MSE and end
    up in roughly the same neighborhood."""
    rng = np.random.RandomState(0)
    Sbatch, T, Z_S, Z_T = _toy_categorical_dataset(rng)
    Strue = (Sbatch - 5.0) / 2.0

    def _fit(device):
        cder = ConDoAdapterKLD(transform_type="affine", verbose=0, device=device)
        cder.fit(Sbatch, T, Z_S, Z_T)
        return cder

    cpu = _fit("cpu")
    gpu = _fit("cuda")
    np.testing.assert_allclose(cpu.M_, gpu.M_, atol=1e-2, rtol=1e-2)
    np.testing.assert_allclose(cpu.b_, gpu.b_, atol=1e-2, rtol=1e-2)
