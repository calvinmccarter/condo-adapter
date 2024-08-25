from copy import deepcopy

import miceforest as mf
import numpy as np
import sklearn.utils as skut
import torch


class AdapterDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        S_list: np.ndarray,
        T_list: np.ndarray,
    ):
        # Each list has len n_bootstraps * bootsize, with elts shape=(n_mice_impute, d)
        #assert S_list.shape == T_list.shape
        assert S_list.shape[0] == T_list.shape[0]
        self.S_list = torch.from_numpy(S_list)
        self.T_list = torch.from_numpy(T_list)

    def __len__(self):
        return self.S_list.shape[0]

    def __getitem__(self, idx):
        # Returns a pair of (n_mice_impute, d) matrices as a single "sample"
        # We will compute the MMD between these two matrices
        # And the loss for a batch will be the sum over a batch of "samples"
        return self.S_list[idx, :, :], self.T_list[idx, :, :]

    def dtype(self):
        return self.S_list.dtype


class AdapterDatasetConDo(torch.utils.data.Dataset):
    def __init__(
        self,
        Xs,
        Xt,
        Zs_,
        Zt_,
        Z_test_,
        W_test,
        n_mice_impute,
        n_mice_iters,
        n_samples,
        batch_size,
    ):
        self.Xs = Xs
        self.Xt = Xt
        self.Zs_ = Zs_
        self.Zt_ = Zt_
        self.Z_test_ = Z_test_
        self.W_test = W_test
        self.n_mice_impute = n_mice_impute
        self.n_mice_iters = n_mice_iters
        self.n_samples = n_samples
        self.batch_size = batch_size
        self.mydtype = torch.from_numpy(Xs).dtype

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        Xs = self.Xs
        Zs_ = self.Zs_
        Xt = self.Xt
        Zt_ = self.Zt_
        Z_test_ = self.Z_test_
        W_test = self.W_test
        batch_size = self.batch_size
        dtype = Xs.dtype
        rng = skut.check_random_state(idx)
        d = Xs.shape[1]

        Z_testixs = rng.choice(Z_test_.shape[0], size=batch_size, p=W_test.ravel())
        bZ_test_ = Z_test_[Z_testixs, :]

        S_dataset = np.concatenate([
            np.concatenate([Xs, Zs_], axis=1),
            np.concatenate([np.full((batch_size, d), np.nan), bZ_test_], axis=1),
        ])
        S_imputer = mf.ImputationKernel(
            S_dataset,
            datasets=self.n_mice_impute,
            save_all_iterations=False,
            random_state=idx,
        )
        S_imputer.mice(self.n_mice_iters)
        S_complete = np.zeros((batch_size, self.n_mice_impute, d), dtype=dtype)
        for imp in range(self.n_mice_impute):
            S_complete[:, imp, :] = S_imputer.complete_data(dataset=imp)[Xs.shape[0]:, :d]

        T_dataset = np.concatenate([
            np.concatenate([Xt, Zt_], axis=1),
            np.concatenate([np.full((batch_size, d), np.nan), bZ_test_], axis=1),
        ])
        T_imputer = mf.ImputationKernel(
            T_dataset,
            datasets=self.n_mice_impute,
            save_all_iterations=False,
            random_state=idx+1234,
        )
        T_imputer.mice(self.n_mice_iters)
        T_complete = np.zeros((batch_size, self.n_mice_impute, d), dtype=dtype)
        for imp in range(self.n_mice_impute):
            T_complete[:, imp, :] = T_imputer.complete_data(dataset=imp)[Xt.shape[0]:, :d]

        return torch.from_numpy(S_complete), torch.from_numpy(T_complete)

    def dtype(self):
        return self.mydtype


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


class LinearAdapter(torch.nn.Module):
    def __init__(
        self,
        transform_type: str,
        in_features: int,
        out_features: int,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.transform_type = transform_type
        self.in_features = in_features
        self.out_features = out_features

        if transform_type == "location-scale":
            assert in_features == out_features
            num_feats = in_features
            self.M = torch.nn.Parameter(torch.empty(num_feats, **factory_kwargs))
            self.b = torch.nn.Parameter(torch.empty(num_feats, **factory_kwargs))

        elif transform_type == "affine":
            self.M = torch.nn.Parameter(
                torch.empty((out_features, in_features), **factory_kwargs)
            )
            self.b = torch.nn.Parameter(torch.empty(out_features, **factory_kwargs))
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
        (batch_size, n_mice_impute, ds) = S.shape
        S_ = S.reshape(-1, ds)
        if self.transform_type == "location-scale":
            adaptedSsample = S_ * self.M.reshape(1, -1) + self.b.reshape(1, -1)
        elif self.transform_type == "affine":
            adaptedSsample = S_ @ self.M.T + self.b.reshape(1, -1)
        adaptedSsample = adaptedSsample.reshape(batch_size, n_mice_impute, -1)
        return adaptedSsample

    def extra_repr(self) -> str:
        return "transform_type={}, in_features={}, out_features={}".format(
            self.transform_type,
            self.in_features,
            self.out_features,
        )

    def get_M_b(self):
        best_M = self.M.detach().numpy()
        best_b = self.b.detach().numpy()
        return (best_M, best_b)


"""
class LinearAdapter(torch.nn.Module):
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
            torch.nn.init.zeros_(self.M)
            torch.nn.init.zeros_(self.b)
        elif self.transform_type == "affine":
            torch.nn.init.zeros_(self.M)
            torch.nn.init.zeros_(self.b)

    def forward(self, S: torch.Tensor) -> torch.Tensor:
        if self.transform_type == "location-scale":
            adaptedSsample = S * self.M.reshape(1, -1) + self.b.reshape(1, -1) + S
        elif self.transform_type == "affine":
            adaptedSsample = S @ self.M.T + self.b.reshape(1, -1) + S
        return adaptedSsample

    def extra_repr(self) -> str:
        return "transform_type={}, num_feats={}".format(
            self.transform_type,
            self.num_feats,
        )

    def get_M_b(self):
        best_M = self.M.detach().numpy()
        best_b = self.b.detach().numpy()
        if best_M.ndim == 1:
            best_M = best_M + 1.
        else:
            best_M = best_M + np.eye(self.num_feats, dtype=best_M.dtype)
        return (best_M, best_b)
"""


class RBF(torch.nn.Module):
    """https://github.com/yiftachbeer/mmd_loss_pytorch"""
    def __init__(self, n_kernels=1, mul_factor=2.0, bandwidth=None):
        super().__init__()
        # XXX n_kernels > 1 causes a segfault at torch.exp with torch==2.1.2 and numpy==1.26.3
        self.bandwidth_multipliers = mul_factor ** (torch.arange(n_kernels) - n_kernels // 2)
        self.bandwidth = bandwidth

    def get_bandwidth(self, L2_distances):
        if self.bandwidth is None:
            n_samples = L2_distances.shape[0]
            return L2_distances.data.sum() / (n_samples ** 2 - n_samples)

        return self.bandwidth

    def forward(self, X):
        L2_distances = torch.cdist(X, X) ** 2
        bws = (self.get_bandwidth(L2_distances.detach()) * self.bandwidth_multipliers)[:, None, None]
        beforeexp = -L2_distances[None, ...] / bws
        afterexp = torch.exp(beforeexp)
        return afterexp.sum(dim=0)


class BatchMMDLoss(torch.nn.Module):
    """https://github.com/yiftachbeer/mmd_loss_pytorch"""
    def __init__(self, kernel=RBF()):
        super().__init__()
        self.kernel = kernel

    def forward(self, allX, allY):
        batch_size = allX.shape[0]
        mmd = torch.tensor(0.)

        for i in range(batch_size):
            X = allX[i, :, :]
            Y = allY[i, :, :]
            K = self.kernel(torch.vstack([X, Y]))

            X_size = X.shape[0]
            XX = K[:X_size, :X_size].mean()
            XY = K[:X_size, X_size:].mean()
            YY = K[X_size:, X_size:].mean()
            mmd = mmd + XX - 2 * XY + YY
        return mmd
 
