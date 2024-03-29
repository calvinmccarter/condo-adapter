from copy import deepcopy

import numpy as np
import torch


class EarlyStopping:
    def __init__(self, patience, model=None):
        self.patience = patience
        self.counter = 0
        self.early_stop = False
        self.loss_min = np.Inf
        self.state_dict = None
        if model is not None:
            self.state_dict = deepcopy(model.state_dict())

    def __call__(self, loss, model, epoch):
        if loss < self.loss_min:
            self.loss_min = loss
            self.epoch_min = epoch
            self.state_dict = deepcopy(model.state_dict())
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True


class MMDLinearAdapterModule(torch.nn.Module):
    def __init__(
        self,
        transform_type: str,
        num_feats: int,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.transform_type = transform_type
        self.num_feats = num_feats

        if transform_type == "location-scale":
            self.M = torch.nn.Parameter(torch.empty(num_feats, **factory_kwargs))
            self.b = torch.nn.Parameter(torch.empty(num_feats, **factory_kwargs))

        elif transform_type == "affine":
            self.M = torch.nn.Parameter(
                torch.empty((num_feats, num_feats), **factory_kwargs)
            )
            self.b = torch.nn.Parameter(torch.empty(num_feats, **factory_kwargs))
        else:
            raise ValueError(f"invalid transform_type:{transform_type}")
        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.transform_type == "location-scale":
            torch.nn.init.ones_(self.M)
            torch.nn.init.zeros_(self.b)
        elif self.transform_type == "affine":
            torch.nn.init.eye_(self.M)
            torch.nn.init.zeros_(self.b)

    def forward(self, S: torch.Tensor) -> torch.Tensor:
        if self.transform_type == "location-scale":
            adaptedSsample = S * self.M.reshape(1, -1) + self.b.reshape(1, -1)
        elif self.transform_type == "affine":
            adaptedSsample = S @ M.T + b.reshape(1, -1)
        return adaptedSsample

    def extra_repr(self) -> str:
        return "transform_type={}, num_feats={}".format(
            self.transform_type,
            self.num_feats,
        )


def MMDLoss(
    adaptedSsample,
    Tsample,
    length_scale: np.ndarray,
    multiscale: bool = False,
    reduction: str = "sum",
):
    if reduction == "mean":
        assert adaptedSsample.shape[0] == Tsample.shape[0]
        reduction_factor = 1.0 / adaptedSsample.shape[0]
    elif reduction == "sum":
        reduction_factor = 1.0
    elif reduction == "product":
        reduction_factor = 1.0 / (adaptedSsample.shape[0] * Tsample.shape[0])
    else:
        raise ValueError(f"MMDLoss invalid reduction={reduction}")

    if multiscale:
        mults = [0.1, 1.0, 10.0]
    else:
        mults = [1.0]
    invroot_length_scale = 1.0 / np.sqrt(length_scale)

    obj = torch.tensor(0.0)
    for mult in mults:
        lscaler = torch.from_numpy(mult * invroot_length_scale).view(1, -1)
        scaled_Tsample = Tsample * lscaler
        scaled_adaptedSsample = adaptedSsample * lscaler

        T_Tt = scaled_Tsample @ scaled_Tsample.T
        T_aSt = scaled_Tsample @ scaled_adaptedSsample.T
        aS_aSt = scaled_adaptedSsample @ scaled_adaptedSsample.T
        obj += -2 * reduction_factor * torch.sum(
            torch.exp(
                -0.5
                * (T_Tt.diag().unsqueeze(1) - 2 * T_aSt + aS_aSt.diag().unsqueeze(0))
            )
        ) + reduction_factor * torch.sum(
            torch.exp(
                -0.5
                * (aS_aSt.diag().unsqueeze(1) - 2 * aS_aSt + aS_aSt.diag().unsqueeze(0))
            )
        )
    return obj
