"""Muon optimizer (MomentUm Orthogonalized by Newton-schulz).

Single-GPU adaptation copied from sda/sda/optim/muon.py (which in turn
was adapted from
https://github.com/toothacher17/Megatron-LM/blob/moonshot/distributedmuon-impl/
megatron/core/optimizer/muon.py and Keller Jordan's reference at
https://github.com/KellerJordan/Muon).

The default lr=2e-2 is roughly 10x the AdamW analogue; the orthogonalization
step normalizes the update's spectral norm, so the lr's units differ.

Parameters with ndim < 2 go through the AdamW path inside this same
optimizer — see `step()`. Use that to put e.g. bias vectors in their own
group with `use_muon=False`.
"""
from typing import Iterable
import math

import torch


def zeropower_via_newtonschulz5(G: torch.Tensor, steps: int) -> torch.Tensor:
    """Newton-Schulz iteration that approximates the zero'th power /
    orthogonalization of G. Returns something near US'V^T where USV^T = G
    and S' is diagonal with entries ~ Uniform(0.5, 1.5).
    """
    assert G.ndim == 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G
    if G.size(0) > G.size(1):
        X = X.T

    # Ensure spectral norm is at most 1
    X = X / (X.norm() + 1e-7)
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X

    if G.size(0) > G.size(1):
        X = X.T
    return X


def adjust_lr_wd_for_muon(lr: float, matched_adamw_rms: float, param_shape) -> float:
    """Scale the per-step update so its expected RMS matches AdamW's at
    the same nominal lr. See https://github.com/MoonshotAI/Moonlight."""
    A, B = param_shape
    return lr * math.sqrt(max(A, B)) * matched_adamw_rms


class Muon(torch.optim.Optimizer):
    """Muon with an AdamW fallback path for sub-2D parameters.

    Set ``use_muon=True`` on a parameter group whose tensors are all >= 2D
    (the Newton-Schulz iteration requires 2D). Sub-2D params (biases,
    location-scale scales) should live in a group with ``use_muon=False``
    — those take the internal AdamW path.

    Arguments default to the original Muon recipe: lr=0.02, weight_decay=0.1,
    momentum=0.95, Nesterov, 5 NS steps, AdamW betas (0.95, 0.95).
    """

    def __init__(
        self,
        param_groups: Iterable,
        lr: float = 2e-2,
        weight_decay: float = 0.1,
        matched_adamw_rms: float = 0.2,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 5,
        adamw_betas=(0.95, 0.95),
        adamw_eps: float = 1e-8,
    ):
        defaults = dict(
            lr=lr,
            weight_decay=weight_decay,
            matched_adamw_rms=matched_adamw_rms,
            momentum=momentum,
            nesterov=nesterov,
            ns_steps=ns_steps,
            adamw_betas=adamw_betas,
            adamw_eps=adamw_eps,
        )
        super().__init__(param_groups, defaults)

    def step(self):
        # Muon path: orthogonalized SGD-momentum for 2D params.
        for group in self.param_groups:
            if not group.get("use_muon", False):
                continue
            lr = group["lr"]
            ns_steps = group["ns_steps"]
            weight_decay = group["weight_decay"]
            momentum = group["momentum"]
            matched_adamw_rms = group["matched_adamw_rms"]
            for p in group["params"]:
                g = p.grad
                assert g is not None
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(g)

                ns_input = g.add(buf, alpha=momentum) if group["nesterov"] else buf
                ns_input = ns_input.reshape(len(g), -1)
                assert ns_input.ndim >= 2
                update = zeropower_via_newtonschulz5(
                    ns_input, steps=ns_steps
                ).view(g.shape)

                # weight decay → pull p toward zero (same form as AdamW
                # in PyTorch: p ← p · (1 − lr · wd))
                p.data.mul_(1 - lr * weight_decay)
                adjusted_lr = adjust_lr_wd_for_muon(
                    lr, matched_adamw_rms, ns_input.shape
                )
                p.data.add_(update, alpha=-adjusted_lr)

        # AdamW path: fallback for sub-2D params and any group without use_muon.
        for group in self.param_groups:
            if group.get("use_muon", False):
                continue
            group["step"] = group.get("step", 0) + 1
            step = group["step"]
            lr = group["lr"]
            weight_decay = group["weight_decay"]
            beta1, beta2 = group["adamw_betas"]
            eps = group["adamw_eps"]
            for p in group["params"]:
                g = p.grad
                assert g is not None
                state = self.state[p]
                if "adamw_exp_avg" not in state:
                    state["adamw_exp_avg"] = torch.zeros_like(g)
                    state["adamw_exp_avg_sq"] = torch.zeros_like(g)
                buf1 = state["adamw_exp_avg"]
                buf2 = state["adamw_exp_avg_sq"]
                buf1.lerp_(g, 1 - beta1)
                buf2.lerp_(g.square(), 1 - beta2)
                gh = buf1 / (eps + buf2.sqrt())
                bc1 = 1 - beta1 ** step
                bc2 = 1 - beta2 ** step
                scale = bc1 / (bc2 ** 0.5)
                p.data.mul_(1 - lr * weight_decay)
                p.data.add_(gh, alpha=-lr / scale)


def make_param_groups(model: torch.nn.Module) -> list[dict]:
    """Split a module's parameters into a Muon group (2D+) and an AdamW
    group (sub-2D) suitable for passing to ``Muon(...)``.

    Empty groups are omitted so the optimizer doesn't iterate over them.
    """
    muon_params = [p for p in model.parameters() if p.requires_grad and p.ndim >= 2]
    adamw_params = [p for p in model.parameters() if p.requires_grad and p.ndim < 2]
    groups: list[dict] = []
    if muon_params:
        groups.append({"params": muon_params, "use_muon": True})
    if adamw_params:
        groups.append({"params": adamw_params, "use_muon": False})
    return groups
