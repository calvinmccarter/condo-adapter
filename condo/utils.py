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
        # NumPy 2.0 removed `np.Inf`; use `np.inf`.
        self.loss_min = np.inf
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
    """Linear adapter parameterized so AdamW weight decay regularizes
    toward the identity transform whenever that is meaningful.

    When the transform is square (always the case for ``location-scale``;
    only when ``in_features == out_features`` for ``affine``), ``self.M``
    holds a *delta from identity*: at initialization it is zero, and the
    effective transform is ``(I + ΔM)`` (or ``(1 + Δm)`` element-wise for
    location-scale). AdamW's pull-toward-zero then translates into a pull
    toward identity in the effective transform.

    For non-square affine maps, identity isn't defined, so we fall back to
    the legacy parameterization: ``self.M`` is initialized via
    ``torch.nn.init.eye_`` (rectangular identity-like) and weight decay
    pulls it toward zero. The non-square case isn't used by the
    batch-integration runner, but is preserved here for parity with the
    earlier API.
    """

    def __init__(
        self,
        transform_type: str,
        in_features: int,
        out_features: int,
        rank: int = 16,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.transform_type = transform_type
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        # Delta-from-identity parameterization is used iff the transform is
        # square (always so for location-scale). Recorded once so
        # forward / get_M_b can branch cheaply.
        self.is_square = in_features == out_features

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

        elif transform_type == "diagonal-plus-low-rank":
            assert in_features == out_features, (
                "diagonal-plus-low-rank requires square (in_features == out_features)"
            )
            num_feats = in_features
            # M holds the diagonal delta-from-identity: effective d = 1 + ΔM.
            self.M = torch.nn.Parameter(torch.empty(num_feats, **factory_kwargs))
            self.U = torch.nn.Parameter(
                torch.empty(num_feats, rank, **factory_kwargs)
            )
            self.V = torch.nn.Parameter(
                torch.empty(num_feats, rank, **factory_kwargs)
            )
            self.b = torch.nn.Parameter(torch.empty(num_feats, **factory_kwargs))
        else:
            raise ValueError(f"invalid transform_type:{transform_type}")
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # b always starts at zero (identity translation).
        torch.nn.init.zeros_(self.b)
        if self.transform_type == "diagonal-plus-low-rank":
            torch.nn.init.zeros_(self.M)
            # Small symmetric random init for U, V breaks the saddle at
            # U=V=0 while keeping initial UV^T tiny (~sqrt(rank)*1e-3 per entry).
            torch.nn.init.normal_(self.U, mean=0.0, std=1e-3)
            torch.nn.init.normal_(self.V, mean=0.0, std=1e-3)
        elif self.is_square:
            torch.nn.init.zeros_(self.M)
        else:
            # Non-square affine: legacy eye_ init, regularizes toward zero.
            torch.nn.init.eye_(self.M)

    def forward(self, S: torch.Tensor) -> torch.Tensor:
        (batch_size, n_mice_impute, ds) = S.shape
        S_ = S.reshape(-1, ds)
        if self.transform_type == "location-scale":
            # Effective scale is 1 + ΔM, applied elementwise.
            adaptedSsample = (
                S_ * (1.0 + self.M).reshape(1, -1) + self.b.reshape(1, -1)
            )
        elif self.transform_type == "affine":
            if self.is_square:
                # Effective M is (I + ΔM); compute S @ (I + ΔM)^T = S + S @ ΔM^T
                # to avoid materializing the d×d identity each forward.
                adaptedSsample = S_ + S_ @ self.M.T + self.b.reshape(1, -1)
            else:
                adaptedSsample = S_ @ self.M.T + self.b.reshape(1, -1)
        elif self.transform_type == "diagonal-plus-low-rank":
            # y = S * (1 + ΔM) + (S @ V) @ U^T + b. Never materializes the
            # dense d×d matrix; cost is O(N d r) instead of O(N d^2).
            adaptedSsample = (
                S_ * (1.0 + self.M).reshape(1, -1)
                + (S_ @ self.V) @ self.U.T
                + self.b.reshape(1, -1)
            )
        adaptedSsample = adaptedSsample.reshape(batch_size, n_mice_impute, -1)
        return adaptedSsample

    def extra_repr(self) -> str:
        return (
            "transform_type={}, in_features={}, out_features={}, delta_param={}"
        ).format(
            self.transform_type,
            self.in_features,
            self.out_features,
            self.is_square,
        )

    def perturbation_sq(self):
        """Return ||diag(ΔM) + U V^T||²_F as a scalar tensor.

        Equals the squared Frobenius norm of the effective matrix's
        deviation from identity. Used as the explicit regularizer for the
        diagonal-plus-low-rank transform (replaces AdamW weight decay on
        the individual factors). Uses the r×r-trace identity to avoid
        materializing the dense d×d perturbation:
            ||UV^T||²_F = trace(U^T U V^T V) = sum(elementwise(U^T U, V^T V))
            trace(diag(ΔM) UV^T) = sum_i ΔM_i <U_i, V_i>
        """
        assert self.transform_type == "diagonal-plus-low-rank"
        diag_sq = (self.M ** 2).sum()
        UTU = self.U.T @ self.U
        VTV = self.V.T @ self.V
        prod_sq = (UTU * VTV).sum()
        cross = (self.M * (self.U * self.V).sum(dim=-1)).sum()
        return diag_sq + prod_sq + 2 * cross

    def get_M_b(self):
        # `.cpu()` is a no-op when the tensor is already on CPU, so this
        # path is safe regardless of where the module lives.
        best_b = self.b.detach().cpu().numpy()
        if self.transform_type == "diagonal-plus-low-rank":
            # Materialize the dense effective matrix diag(1 + ΔM) + U V^T
            # so downstream callers can treat it as a square affine M.
            delta_d = self.M.detach().cpu().numpy()
            U = self.U.detach().cpu().numpy()
            V = self.V.detach().cpu().numpy()
            best_M = np.diag(1.0 + delta_d) + U @ V.T
            return (best_M, best_b)
        best_M = self.M.detach().cpu().numpy()
        if self.is_square:
            # Recover the effective transform that downstream numpy code
            # (transform / inverse_transform) expects.
            if self.transform_type == "location-scale":
                best_M = best_M + 1.0
            else:  # square affine
                best_M = best_M + np.eye(best_M.shape[0], dtype=best_M.dtype)
        return (best_M, best_b)


class RBF(torch.nn.Module):
    """https://github.com/yiftachbeer/mmd_loss_pytorch"""
    def __init__(self, n_kernels=1, mul_factor=2.0, bandwidth=None):
        super().__init__()
        # XXX n_kernels > 1 causes a segfault at torch.exp with torch==2.1.2 and numpy==1.26.3
        # Register as a buffer so .to(device) on the loss/parent module
        # propagates it; otherwise CPU/CUDA mixing in forward() raises.
        self.register_buffer(
            'bandwidth_multipliers',
            mul_factor ** (torch.arange(n_kernels) - n_kernels // 2),
        )
        self.bandwidth = bandwidth

    def get_bandwidth(self, L2_distances):
        if self.bandwidth is None:
            n_samples = L2_distances.shape[0]
            return L2_distances.data.sum() / (n_samples ** 2 - n_samples)

        return self.bandwidth

    def forward(self, X):
        L2_distances = torch.cdist(X, X) ** 2
        # Track the input's device in case the loss wasn't explicitly .to()'d.
        bw_mul = self.bandwidth_multipliers.to(X.device)
        bws = (self.get_bandwidth(L2_distances.detach()) * bw_mul)[:, None, None]
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
        # Initialize accumulator on the input's device so the loss runs
        # cross-device cleanly without callers needing to .to() the loss.
        mmd = torch.tensor(0., device=allX.device)

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
 
