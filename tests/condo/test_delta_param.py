"""LinearAdapter is delta-parameterized when the transform is square.

Verifies three properties:

1. **Initial-state identity.** After construction the effective transform
   maps any input to itself (modulo the zero bias) — i.e. ``M_eff = I``
   for square affine and ``m_eff = 1`` for location-scale.
2. **Weight decay pulls toward identity, not zero**, for square affine
   and location-scale. (For non-square affine we don't claim this — the
   notion of identity isn't defined; the legacy zero-target behavior is
   preserved.)
3. **Non-square affine still uses the legacy eye_ init** (delta-param
   off), so we don't regress the rectangular-mapping case.
"""
import numpy as np
import pytest
import torch

from condo.utils import LinearAdapter


def _identity_input(batch=2, n_mice=3, d=4, dtype=torch.float32):
    rng = torch.Generator()
    rng.manual_seed(0)
    return torch.randn(batch, n_mice, d, generator=rng, dtype=dtype)


@pytest.mark.parametrize("transform_type", ["location-scale", "affine"])
def test_initial_state_is_identity_for_square(transform_type):
    adapter = LinearAdapter(transform_type=transform_type, in_features=4, out_features=4)
    assert adapter.is_square is True
    x = _identity_input(d=4)
    y = adapter(x)
    np.testing.assert_allclose(y.detach().numpy(), x.numpy(), atol=1e-6)


def test_initial_state_for_nonsquare_affine_uses_eye():
    # 4 -> 6: not square; legacy zero-target parameterization stays on.
    adapter = LinearAdapter(transform_type="affine", in_features=4, out_features=6)
    assert adapter.is_square is False
    # M is the rectangular eye (1s on the diagonal, 0s elsewhere).
    M = adapter.M.detach().numpy()
    expected = np.zeros((6, 4))
    np.testing.assert_array_equal(np.diag(M[:4, :4]), np.ones(4))
    np.testing.assert_array_equal(M[4:, :], expected[4:, :])


def _identity_target(transform_type, M):
    if transform_type == "location-scale":
        return np.ones_like(M)
    return np.eye(M.shape[0], dtype=M.dtype)


@pytest.mark.parametrize("transform_type", ["location-scale", "affine"])
def test_weight_decay_pulls_effective_toward_identity(transform_type):
    """With no gradient signal, pure weight decay on ΔM moves ΔM toward 0,
    which means M_eff = I + ΔM moves back toward identity. We assert
    monotonic shrinking of ``||M_eff - I||`` rather than full convergence —
    AdamW with no gradient signal decays multiplicatively, so reaching
    machine-zero takes many more steps than needed to demonstrate the
    direction."""
    adapter = LinearAdapter(transform_type=transform_type, in_features=3, out_features=3)
    with torch.no_grad():
        adapter.M.add_(torch.tensor(0.5))   # ΔM ← 0.5 (so M_eff = I + 0.5)
        adapter.b.add_(torch.tensor(0.5))

    M_pre, b_pre = (np.copy(a) for a in adapter.get_M_b())
    target = _identity_target(transform_type, M_pre)
    dist_pre = np.linalg.norm(M_pre - target)

    opt = torch.optim.AdamW(adapter.parameters(), lr=1e-1, weight_decay=1e-1)
    for _ in range(50):
        opt.zero_grad()
        # Synthetic zero loss — gradient is zero, only weight decay acts.
        # Touch both M and b so AdamW initializes .grad on both (AdamW
        # silently skips parameters whose grad is None).
        loss = (adapter.M.sum() + adapter.b.sum()) * 0.0
        loss.backward()
        opt.step()
    M_post, b_post = (np.copy(a) for a in adapter.get_M_b())
    dist_post = np.linalg.norm(M_post - target)
    # The effective transform moved closer to identity.
    assert dist_post < dist_pre, (
        f"||M_eff - I|| should shrink under pure weight decay; "
        f"pre={dist_pre:.4f} post={dist_post:.4f}"
    )
    # b → 0 (identity translation) as well.
    assert np.linalg.norm(b_post) < np.linalg.norm(b_pre)


def test_nonsquare_affine_weight_decay_pulls_toward_zero_M():
    """For non-square affine, parameter equals effective M, weight decay
    pulls it toward zero (legacy semantics) — we don't claim identity
    regularization here since identity isn't defined for non-square maps.
    """
    adapter = LinearAdapter(transform_type="affine", in_features=3, out_features=5)
    assert adapter.is_square is False

    opt = torch.optim.AdamW(adapter.parameters(), lr=1e-1, weight_decay=1e-1)
    initial_M, _ = (np.copy(a) for a in adapter.get_M_b())
    for _ in range(50):
        opt.zero_grad()
        loss = (adapter.M.sum() * 0.0)
        loss.backward()
        opt.step()
    final_M, _ = (np.copy(a) for a in adapter.get_M_b())
    # ||final|| < ||initial||  (weight decay shrunk parameters toward 0)
    assert np.linalg.norm(final_M) < np.linalg.norm(initial_M)
