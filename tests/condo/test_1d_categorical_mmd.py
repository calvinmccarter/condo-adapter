import pytest
import numpy as np
from condo import ConDoAdapter


@pytest.mark.parametrize(
    "sampling", ["source", "sum-proportional", "target", "product"]
)
@pytest.mark.parametrize(
    "transform_type",
    ["location-scale", "affine"],
)
def test_1d_categorical_mmd(sampling, transform_type):
    """Test MMD on 1d variable with 1d categorical confounder."""

    rtol = 0.3

    rng = np.random.RandomState(0)
    N = 200
    N_T = 100
    N_S = 100

    # How confounder X affects the distribution of T and S
    mu_hotdog = 10.0
    sigma_hotdog = 1.0
    mu_not = 5.0
    sigma_not = 2.0

    # How batch effect affects S
    batch_m = 2
    batch_b = 5
    # The true batch correction from Sbatch to S
    true_m = 1.0 / batch_m
    true_b = -1 * batch_b / batch_m

    n_hotdogT = 75
    n_notT = 25
    n_hotdogS = 25
    n_notS = 75
    X_T = np.array([["hotdog"] * n_hotdogT + ["not"] * n_notT]).reshape((N_T, 1))
    X_S = np.array([["hotdog"] * n_hotdogS + ["not"] * n_notS]).reshape((N_S, 1))

    Strue = np.nan * np.ones((N_S, 1))
    T = np.nan * np.ones((N_T, 1))
    Strue[np.where(X_S[:, 0] == "hotdog"), 0] = rng.normal(
        mu_hotdog, sigma_hotdog, size=(n_hotdogS)
    )
    T[np.where(X_T[:, 0] == "hotdog"), 0] = rng.normal(
        mu_hotdog, sigma_hotdog, size=(n_hotdogT)
    )
    Strue[np.where(X_S[:, 0] == "not"), 0] = rng.normal(
        mu_not, sigma_not, size=(n_notS)
    )
    T[np.where(X_T[:, 0] == "not"), 0] = rng.normal(mu_not, sigma_not, size=(n_notT))

    Sbatch = batch_m * Strue + batch_b

    cder = ConDoAdapter(
        sampling=sampling,
        transform_type=transform_type,
        model_type="empirical",
        divergence="mmd",
    )

    cder.fit(Sbatch, T, X_S, X_T)
    Sadapted = cder.transform(Sbatch)
    reldiff_pre = 2 * np.abs(Sbatch - Strue) / (np.abs(Sbatch) + np.abs(Strue))
    reldiff_post = 2 * np.abs(Sadapted - Strue) / (np.abs(Sadapted) + np.abs(Strue))
    np.testing.assert_array_less(reldiff_post.mean(axis=0), reldiff_pre.mean(axis=0))
    np.testing.assert_allclose(Strue, Sadapted, atol=0.1, rtol=rtol)
