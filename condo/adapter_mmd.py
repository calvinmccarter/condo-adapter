from copy import deepcopy
from typing import Union

import numpy as np
import sklearn.utils as skut
import torch

from torch.utils.data import DataLoader

from condo.utils import (
    AdapterDataset,
    BatchMMDLoss,
    EarlyStopping,
    LinearAdapter,
)


class AdapterMMD:
    def __init__(
        self,
        transform_type: str = 'affine',
        bootstrap_fraction: float = 1.,
        n_bootstraps: int = None,
        n_epochs: int = 5,
        batch_size: int = 8,
        mmd_size: int = 20,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        random_state=42,
        verbose: Union[bool, int] = 1,
    ):
        transforms = {'location-scale', 'affine'}
        if transform_type not in transforms:
            raise NotImplementedError(f'transform_type {transform_type}')
        assert bootstrap_fraction <= 1
        self.transform_type = transform_type
        self.bootstrap_fraction = bootstrap_fraction
        self.n_bootstraps = n_bootstraps
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.mmd_size = mmd_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.random_state = random_state
        self.verbose = verbose
        # bootsize = n_test * bootstrap_fraction sampled with replacement
        # so total dataset is of size n_test * n_bootstraps * bootstrap_fraction * n_impute

    def fit(
        self,
        Xs: np.ndarray,
        Xt: np.ndarray,
    ):
        #assert Xs.shape[1] == Xt.shape[1]
        ds = Xs.shape[1]
        dt = Xt.shape[1]
        n = min(Xs.shape[0], Xt.shape[0])

        rng = skut.check_random_state(self.random_state)
        bootsize = max(1, int(n * self.bootstrap_fraction))
        if self.n_bootstraps is None:
            n_bootstraps = int(np.ceil(self.batch_size / bootsize))
        else:
            n_bootstraps = self.n_bootstraps
        assert self.batch_size <= n_bootstraps * bootsize

        # Each list has len n_bootstraps * bootsize, with elts shape=(mmd_size, d)
        S_list = np.zeros((n_bootstraps * bootsize, self.mmd_size, ds), dtype=Xs.dtype)
        T_list = np.zeros((n_bootstraps * bootsize, self.mmd_size, dt), dtype=Xt.dtype)
        for list_ix in range(n_bootstraps * bootsize):
            Xs_ixs = rng.choice(Xs.shape[0], size=self.mmd_size)
            S_list[list_ix, :, :] = Xs[Xs_ixs, :]
            Xt_ixs = rng.choice(Xt.shape[0], size=self.mmd_size)
            T_list[list_ix, :, :] = Xt[Xt_ixs, :]

        dataset = AdapterDataset(S_list, T_list)
        train_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        model = LinearAdapter(
            transform_type=self.transform_type,
            in_features=ds, out_features=dt, dtype=dataset.dtype())
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay,
        )
        early_stopping = EarlyStopping(patience=3, model=model)
        loss_fn = BatchMMDLoss()
        n_batches = len(train_loader)
        if self.verbose:
            print(f'n_batches: {n_batches} dataset_size:{S_list.shape}')

        for epoch in range(self.n_epochs):
            model.train()
            train_loss = 0.
            for bix, (Ssample, Tsample) in enumerate(train_loader):
                if (epoch == 0) and (bix == 0) and self.verbose:
                    print("MMD sample shapes", Ssample.shape, Tsample.shape)
                optimizer.zero_grad()
                adaptedSsample = model(Ssample)
                loss = loss_fn(adaptedSsample, Tsample)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                if self.verbose >= 2 and bix % (max(n_batches, 5) // 5) == 0:
                    # print progress ~5 times per epoch
                    tl = train_loss * n_batches / (bix + 1)
                    print(f'    epoch:{epoch} train_loss:{tl:.5f}  [{bix}/{n_batches}]')
            early_stopping(train_loss, model, epoch)
            if early_stopping.early_stop:
                break

        model.load_state_dict(early_stopping.state_dict)
        (M, b) = model.get_M_b()
        (M, b) = (M.astype(Xs.dtype), b.astype(Xs.dtype))
        if self.transform_type == 'location-scale':
            self.m_ = M
            self.m_inv_ = 1 / self.m_
        elif self.transform_type == 'affine':
            self.M_ = M
            if M.shape[0] == M.shape[1]:
                self.M_inv_ = np.linalg.inv(self.M_)
        self.b_ = b

    def transform(
        self,
        Xs,
    ):
        if self.transform_type == 'location-scale':
            adaptedS = Xs * self.m_.reshape(1, -1) + self.b_.reshape(1, -1)
        elif self.transform_type == 'affine':
            adaptedS = Xs @ self.M_.T + self.b_.reshape(1, -1)
        return adaptedS

    def inverse_transform(
        self,
        Xt,
    ):
        if self.transform_type == 'location-scale':
            adaptedT = (Xt - self.b_.reshape(1, -1)) * self.m_inv_.reshape(1, -1)
        elif self.transform_type == 'affine':
            adaptedT = (Xt - self.b_.reshape(1, -1)) @ self.M_inv_.T
        return adaptedT
