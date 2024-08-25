from copy import deepcopy
from typing import Union

import miceforest as mf
import numpy as np
import sklearn.utils as skut
import torch

from torch.utils.data import DataLoader

from condo.product_prior import product_prior
from condo.utils import (
    AdapterDataset,
    AdapterDatasetConDo,
    BatchMMDLoss,
    EarlyStopping,
    LinearAdapter,
)


class ConDoAdapterMMD:
    def __init__(
        self,
        transform_type: str = 'affine',
        use_mice_discrete_confounder: bool = False,
        mmd_size: int = 20,
        n_mice_iters: int = 2,
        bootstrap_fraction: float = 1.,
        n_bootstraps: int = None,  # if None, smallest possible given batch_size
        n_epochs: int = 5,
        batch_size: int = 8,
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
        self.use_mice_discrete_confounder = use_mice_discrete_confounder
        self.mmd_size = mmd_size
        self.n_mice_iters = n_mice_iters
        self.bootstrap_fraction = bootstrap_fraction
        self.n_bootstraps = n_bootstraps
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.random_state = random_state
        self.verbose = verbose
        # bootsize = n_test * bootstrap_fraction sampled with replacement
        # each is then given n_imp impute samples
        # so total dataset is of size n_test * n_bootstraps * bootstrap_fraction * n_impute

    def fit(
        self,
        Xs: np.ndarray,
        Xt: np.ndarray,
        Zs: np.ndarray,
        Zt: np.ndarray,
    ):
        assert Xs.shape[0] == Zs.shape[0]
        assert Xt.shape[0] == Zt.shape[0]
        #assert Xs.shape[1] == Xt.shape[1]
        assert Zs.shape[1] == Zt.shape[1]
        ds = Xs.shape[1]
        dt = Xt.shape[1]
        n = min(Xs.shape[0], Xt.shape[0])
        assert Xs.dtype == Xt.dtype
        dtype = Xs.dtype
        rng = skut.check_random_state(self.random_state)

        Z_test, W_test, encoder = product_prior(Zs, Zt)
        W_test = W_test.astype(dtype)
        n_test = Z_test.shape[0]
        discrete_confounder = encoder is not None
        use_mice = not discrete_confounder or self.use_mice_discrete_confounder
        if discrete_confounder:
            Z_test_ = encoder.transform(Z_test)
            Zs_ = encoder.transform(Zs)
            Zt_ = encoder.transform(Zt)
        else:
            Z_test_ = Z_test
            Zs_ = Zs
            Zt_ = Zt

        bootsize = max(1, int(n * self.bootstrap_fraction))
        if self.n_bootstraps is None:
            n_bootstraps = int(np.ceil(self.batch_size / bootsize))
        else:
            n_bootstraps = self.n_bootstraps
        assert self.batch_size <= n_bootstraps * bootsize

        # Each list has len n_bootstraps * bootsize, with elts shape=(mmd_size, d)
        S_list = np.zeros((n_bootstraps * bootsize, self.mmd_size, ds), dtype=dtype)
        T_list = np.zeros((n_bootstraps * bootsize, self.mmd_size, dt), dtype=dtype)

        if use_mice:
            """
            # This works and reduces memory usage but is slower
            dataset = AdapterDatasetConDo(
                Xs, Xt, Zs_, Zt_, Z_test_, W_test, self.mmd_size, self.n_mice_iters,
                n_samples=n_bootstraps * bootsize, batch_size=self.batch_size,
            )
            train_loader = DataLoader(dataset, batch_size=1, shuffle=True)
            """
            list_ix = 0
            for bix in range(n_bootstraps):
                Z_testixs = rng.choice(n_test, size=bootsize, p=W_test.ravel())
                bZ_test_ = Z_test_[Z_testixs, :]
            
                S_dataset = np.concatenate([
                    np.concatenate([Xs, Zs_], axis=1),
                    np.concatenate([np.full((bootsize, ds), np.nan), bZ_test_], axis=1),
                ])
                S_imputer = mf.ImputationKernel(
                    S_dataset,
                    datasets=self.mmd_size,
                    save_all_iterations=False,
                    random_state=self.random_state
                )
                S_imputer.mice(self.n_mice_iters)
                S_complete = np.zeros((self.mmd_size, bootsize, ds), dtype=dtype)
                for imp in range(self.mmd_size):
                    S_complete[imp, :, :] = S_imputer.complete_data(dataset=imp)[Xs.shape[0]:, :ds]

                T_dataset = np.concatenate([
                    np.concatenate([Xt, Zt_], axis=1),
                    np.concatenate([np.full((bootsize, dt), np.nan), bZ_test_], axis=1),
                ])
                T_imputer = mf.ImputationKernel(
                    T_dataset,
                    datasets=self.mmd_size,
                    save_all_iterations=False,
                    random_state=self.random_state
                )
                T_imputer.mice(self.n_mice_iters)
                T_complete = np.zeros((self.mmd_size, bootsize, dt), dtype=dtype)
                for imp in range(self.mmd_size):
                    T_complete[imp, :, :] = T_imputer.complete_data(dataset=imp)[Xt.shape[0]:, :dt]

                for i in range(bootsize):
                    S_list[list_ix, :, :] = S_complete[:, i, :]
                    T_list[list_ix, :, :] = T_complete[:, i, :]
                    list_ix += 1
            dataset = AdapterDataset(S_list, T_list)
            train_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        else:
            # If not use_mice, we use the data directly
            Z_testixs = rng.choice(n_test, size=n_bootstraps * bootsize, p=W_test.ravel())
            for list_ix in range(n_bootstraps * bootsize):
                i = Z_testixs[list_ix]
                Zs_ixs, = (Zs == Z_test[i, :]).ravel().nonzero()
                Zt_ixs, = (Zt == Z_test[i, :]).ravel().nonzero()
                S_list[list_ix, :, :] = Xs[rng.choice(Zs_ixs, size=self.mmd_size), :]
                T_list[list_ix, :, :] = Xt[rng.choice(Zt_ixs, size=self.mmd_size), :]
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
                if Ssample.ndim == 4:
                    # Needed for AdapterDatasetConDo
                    assert Tsample.ndim == 4
                    assert Ssample.shape[0] == 1
                    assert Tsample.shape[0] == 1
                    Ssample = Ssample.reshape(Ssample.shape[1], Ssample.shape[2], Ssample.shape[3])
                    Tsample = Tsample.reshape(Tsample.shape[1], Tsample.shape[2], Tsample.shape[3])
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
            if self.verbose >= 2:
                print(f'    epoch:{epoch} train_loss:{train_loss:.5f}')
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
