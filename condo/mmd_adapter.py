from copy import deepcopy
from typing import Union

import math
import numpy as np
import pandas as pd
import torch

from torch.utils.data import DataLoader

from condo.utils import (
    EarlyStopping,
    MMDDataset,
    MMDLinearAdapterModule,
    MMDLoss,
)


def run_mmd(
    S: np.ndarray,
    T: np.ndarray,
    transform_type: str,
    verbose: Union[bool, int],
    epochs: int = 100,
    batch_size: int = 16,
    learning_rate: float = 1e-2,
    weight_decay: float = 1e-3,
    patience: int = 5,
    length_scale_method: str = "target",
    multiscale: bool = False,
):
    """
    Args:
        epochs: number of times to pass through all observed confounder values.

        batch_size: number of samples to draw from S and T per confounder value.
            kernel matrix will be of size (batch_size, batch_size).
            The number of batches per epoch is num_S*num_T / (batch_size ** 2).

        learning_rate: AdamW learning rate (ie step size).

        weight_decay: AdamW weight_decay

        patience: early-stopping patience in epochs

    Returns:
        M: (num_feats, num_feats)
        b: (num_feats,)
        debug_dict: dict with keys "train_loss, "val_loss"
    """
    S = S.astype(np.float32)
    T = T.astype(np.float32)
    num_feats = S.shape[1]
    train_dataset = MMDDataset(S, T)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    model = MMDLinearAdapterModule(transform_type=transform_type, num_feats=num_feats)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay, eps=1e-5
    )
    early_stopping = EarlyStopping(patience=patience)

    if length_scale_method == "target":
        length_scale = np.var(T, axis=0)
    elif length_scale_method == "source":
        length_scale = np.var(S, axis=0)
    elif length_scale_method in ("initialMSE", "dynamic"):
        n_samples = max(T.shape[0], S.shape[0])
        rng = np.random.RandomState(42)
        T_ls_ixs = rng.choice(T.shape[0], size=n_samples, replace=True)
        S_ls_ixs = rng.choice(S.shape[0], size=n_samples, replace=True)
        length_scale = np.mean((T[T_ls_ixs, :] - S[S_ls_ixs, :]) ** 2, axis=0)
    else:
        raise ValueError(f"Invalid length_scale_method: {length_scale_method}")

    debug_dict = {"train_loss": [], "val_loss": []}
    for epoch in range(epochs):
        n_samples = len(train_loader.dataset)
        n_batches = len(train_loader)
        model.train()
        train_loss = 0.0
        for batch_ix, (Ssample, Tsample) in enumerate(train_loader):
            adaptedSsample = model(Ssample)
            loss = MMDLoss(adaptedSsample, Tsample, length_scale, multiscale=multiscale)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            if verbose >= 2 and batch_ix % (max(n_batches, 5) // 5) == 0:
                # print progress ~5 times per epoch
                loss = loss.item()
                print(f"epoch:{epoch} loss: {loss:>7f}  [{batch_ix}/{n_batches}]")
        debug_dict["train_loss"].append(train_loss)

        model.eval()
        adaptedS = model(torch.from_numpy(S))
        val_loss = MMDLoss(
            adaptedS, torch.from_numpy(T), length_scale, multiscale=multiscale
        )
        debug_dict["val_loss"].append(val_loss.item())
        early_stopping(val_loss.item(), model, epoch)
        if verbose and early_stopping.early_stop:
            (best_loss, best_epoch) = early_stopping.loss_min, early_stopping.epoch_min
            print(
                f"Early stopping: {val_loss:.2f}@{epoch} vs {best_loss:.2f}@{best_epoch}"
            )
            break
        if length_scale_method == "dynamic":
            adaptedSnp = adaptedS.detach().numpy()
            length_scale = np.mean(
                (T[T_ls_ixs, :] - adaptedSnp[S_ls_ixs, :]) ** 2, axis=0
            )

    model.load_state_dict(early_stopping.state_dict)
    (best_M, best_b) = model.get_M_b()
    return (best_M, best_b, debug_dict)


class MMDAdapter:
    def __init__(
        self,
        transform_type: str = "location-scale",
        optim_kwargs: dict = None,
        verbose: Union[bool, int] = 1,
    ):
        """
        Args:
            transform_type: Whether to jointly transform all features ("affine"),
                or to transform each feature independently ("location-scale").
                Modeling the joint distribution is slower and allows each
                adapted feature to be a function of all other features,
                rather than purely based on that particular feature
                observation.

            optim_kwargs: Dict containing args for optimization.
                Valid keys are "epochs", "batch_size", "learning_rate",
                "weight_decay", "patience", and "length_scale_method".

            verbose: Bool or integer that indicates the verbosity.
        """
        if transform_type not in ("affine", "location-scale"):
            raise ValueError(f"invalid transform_type: {transform_type}")

        self.transform_type = transform_type
        self.optim_kwargs = deepcopy(optim_kwargs) or {}
        self.verbose = verbose

    def fit(
        self,
        S: np.ndarray,
        T: np.ndarray,
    ):
        """
        Modifies M_, b_, M_inv_, self.debug_dict_
        """
        num_S = S.shape[0]
        num_T = T.shape[0]
        assert S.shape[1] == T.shape[1]

        num_feats = S.shape[1]
        self.M_, self.b_, self.debug_dict_ = run_mmd(
            S=S,
            T=T,
            transform_type=self.transform_type,
            verbose=self.verbose,
            **self.optim_kwargs,
        )

        if self.transform_type == "location-scale":
            self.M_inv_ = np.diag(1.0 / np.diag(self.M_))
        elif self.transform_type == "affine":
            self.M_inv_ = np.linalg.inv(self.M_)

        return self

    def transform(
        self,
        S,
    ):
        adaptedS = S @ self.M_.T + self.b_.reshape(1, -1)
        # same as adaptedS = (self.M_ @ S.T).T + self.b_.reshape(1, -1)
        # self.b_.reshape(1, -1) has shape (1, num_feats)
        return adaptedS

    def inverse_transform(
        self,
        T,
    ):
        adaptedT = (T - self.b_.reshape(1, -1)) @ self.M_inv_.T
        return adaptedT
