import pytest
import numpy as np
from condo import ConDoAdapter

@pytest.mark.parametrize(
    "sampling", ["proportional", "source", "target"]
)
@pytest.mark.parametrize(
    "model", ["joint", "independent"],
)
@pytest.mark.parametrize(
    "kld_direction", ["forward", "reverse"],
)
@pytest.mark.parametrize(
    "heteroscedastic", ["heteroscedastic", "homoscedastic"],
)
def test_1d_continuous(sampling, model, kld_direction, heteroscedastic):
    """Test 1d variable with 1d continuous confounder."""
    joint = model == "joint"
    heteroscedastic = heteroscedastic == "heteroscedastic"

    np.random.seed(0)
    N = 300
    N_T = 200
    N_S = 100

    # How confounder X affects the distribution of T and S
    theta_m = 4
    theta_b = 1
    phi_m = 1
    phi_b = 1

    # How batch effect affects S
    batch_m = 2
    batch_b = -5
    # The true batch correction from Sbatch to S
    true_m = 1. / batch_m
    true_b = -1 * batch_b / batch_m

    X_T = np.sort(np.random.uniform(1, 8, size=(N_T, 1)))
    X_S = np.sort(np.random.uniform(4, 8, size=(N_S, 1)))

    mu_T = theta_m * X_T + theta_b
    sigma_T = phi_m * X_T + phi_b
    mu_S = theta_m * X_S + theta_b
    sigma_S = phi_m * X_S + phi_b

    T = np.random.normal(mu_T, sigma_T)
    Strue = np.random.normal(mu_S, sigma_S)
    Sbatch = batch_m * Strue + batch_b

    cder = ConDoAdapter(
        sampling=sampling,
        joint=joint,
        kld_direction=kld_direction,
        heteroscedastic=heteroscedastic,
    )

    cder.fit(Sbatch, T, X_S, X_T)
    Sadapted = cder.transform(Sbatch)
    reldiff_pre = 2 * np.abs(Sbatch - Strue) / (np.abs(Sbatch) + np.abs(Strue))
    reldiff_post = 2 * np.abs(Sadapted - Strue) / (np.abs(Sadapted) + np.abs(Strue))
    np.testing.assert_array_less(reldiff_post.mean(axis=0), reldiff_pre.mean(axis=0))
    np.testing.assert_allclose(np.array([true_m]), cder.m_, atol=1.0, rtol=1.0)
    np.testing.assert_allclose(np.array([true_b]), cder.b_, atol=1.0, rtol=1.0)
    np.testing.assert_allclose(Strue, Sadapted, atol=0.1, rtol=1.0)