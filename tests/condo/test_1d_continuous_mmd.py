import pytest
import numpy as np
from condo import ConDoAdapterMMD
#from condo import MMDAdapter


@pytest.mark.parametrize(
    "transform_type",
    ["location-scale", "affine"],
)
def test_1d_continuous_condo_mmd(transform_type):
    """Test MMD on 1d variable with 1d continuous confounder."""

    rng = np.random.RandomState(0)
    N = 100
    N_T = 60
    N_S = 40

    # How confounder X affects the distribution of T and S
    theta_m = 4
    theta_b = 1
    phi_m = 1
    phi_b = 1

    # How batch effect affects S
    batch_m = 2
    batch_b = -5
    # The true batch correction from Sbatch to S
    true_m = 1.0 / batch_m
    true_b = -1 * batch_b / batch_m

    X_T = np.sort(rng.uniform(0, 8, size=(N_T, 1)))
    X_S = np.sort(rng.uniform(4, 8, size=(N_S, 1)))

    mu_T = theta_m * X_T + theta_b
    sigma_T = phi_m * X_T + phi_b
    mu_S = theta_m * X_S + theta_b
    sigma_S = phi_m * X_S + phi_b

    T = rng.normal(mu_T, sigma_T)
    Strue = rng.normal(mu_S, sigma_S)
    Sbatch = batch_m * Strue + batch_b
    print(f"true_m:{true_m:.3f} true_b:{true_b:.3f}")

    cder = ConDoAdapterMMD(
        transform_type=transform_type,
        n_epochs=10,
        learning_rate=1e-2,
    )

    cder.fit(Sbatch, T, X_S, X_T)
    Sadapted = cder.transform(Sbatch)
    reldiff_pre = 2 * np.abs(Sbatch - Strue) / (np.abs(Sbatch) + np.abs(Strue))
    reldiff_post = 2 * np.abs(Sadapted - Strue) / (np.abs(Sadapted) + np.abs(Strue))
    np.testing.assert_array_less(reldiff_post.mean(axis=0), reldiff_pre.mean(axis=0))
    np.testing.assert_allclose(Strue, Sadapted, atol=0.1, rtol=0.2)


@pytest.mark.parametrize(
    "transform_type",
    []#["location-scale", "affine"],
)
def test_1d_continuous_mmd(transform_type):
    """Test MMD on 1d variable with 1d continuous confounder."""

    rtol = 0.3

    rng = np.random.RandomState(0)
    N = 100
    N_T = 60
    N_S = 40

    # How confounder X affects the distribution of T and S
    theta_m = 4
    theta_b = 1
    phi_m = 1
    phi_b = 1

    # How batch effect affects S
    batch_m = 2
    batch_b = -5
    # The true batch correction from Sbatch to S
    true_m = 1.0 / batch_m
    true_b = -1 * batch_b / batch_m

    X_T = np.sort(rng.uniform(0, 8, size=(N_T, 1)))
    X_S = np.sort(rng.uniform(4, 8, size=(N_S, 1)))

    mu_T = theta_m * X_T + theta_b
    sigma_T = phi_m * X_T + phi_b
    mu_S = theta_m * X_S + theta_b
    sigma_S = phi_m * X_S + phi_b

    T = rng.normal(mu_T, sigma_T)
    Strue = rng.normal(mu_S, sigma_S)
    Sbatch = batch_m * Strue + batch_b

    cder = MMDAdapter(
        transform_type=transform_type,
        optim_kwargs={"epochs": 10, "learning_rate": 1e-4},
    )

    cder.fit(Sbatch, T)
    Sadapted = cder.transform(Sbatch)
    reldiff_pre = 2 * np.abs(Sbatch - Strue) / (np.abs(Sbatch) + np.abs(Strue))
    reldiff_post = 2 * np.abs(Sadapted - Strue) / (np.abs(Sadapted) + np.abs(Strue))
    np.testing.assert_array_less(reldiff_pre.mean(axis=0), reldiff_post.mean(axis=0))
