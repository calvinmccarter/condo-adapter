from copy import deepcopy
from typing import Union

import math
import miceforest as mf
import numpy as np
import sklearn.utils as skut

from condo.product_prior import product_prior
from condo.ot import wasserstein_procrustes


class ConDoAdapterWD:
    def __init__(
        self,
        transform_type: str = 'orthogonal',
        center: bool = True,
        use_mice_discrete_confounder: bool = False,
        wd_size: int = 20,
        n_mice_iters: int = 2,
        bootstrap_fraction: float = 1.,
        n_bootstraps: int = 1,
        init_gromov: bool = True,
        n_iters: int = 5,  # wasserstein-procrustes iterations
        random_state=42,
        verbose: Union[bool, int] = 1,
    ):
        transforms = {'orthogonal', 'orthogonal-scale'}
        if transform_type not in transforms:
            raise NotImplementedError(f'transform_type {transform_type}')
        assert bootstrap_fraction <= 1
        self.transform_type = transform_type
        self.center = center
        self.use_mice_discrete_confounder = use_mice_discrete_confounder
        self.wd_size = wd_size
        self.n_mice_iters = n_mice_iters
        self.bootstrap_fraction = bootstrap_fraction
        self.n_bootstraps = n_bootstraps
        self.init_gromov = init_gromov
        self.n_iters = n_iters
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
        assert Xs.shape[1] == Xt.shape[1]
        assert Zs.shape[1] == Zt.shape[1]
        d = Xs.shape[1]
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
        # Each list has len n_bootstraps * bootsize, with elts shape=(wd_size, d)
        S_list = []  # all sampled Xs
        T_list = []  # all sampled Xt
        if use_mice:
            for bix in range(self.n_bootstraps):
                Z_testixs = rng.choice(n_test, size=bootsize, p=W_test.ravel())
                bZ_test_ = Z_test_[Z_testixs, :]
            
                S_dataset = np.concatenate([
                    np.concatenate([Xs, Zs_], axis=1),
                    np.concatenate([np.full((bootsize, d), np.nan), bZ_test_], axis=1),
                ])
                S_imputer = mf.ImputationKernel(
                    S_dataset,
                    datasets=self.wd_size,
                    save_all_iterations=False,
                    random_state=self.random_state
                )
                S_imputer.mice(self.n_mice_iters)
                S_complete = np.zeros((self.wd_size, bootsize, d), dtype=dtype)
                for imp in range(self.wd_size):
                    S_complete[imp, :, :] = S_imputer.complete_data(dataset=imp)[Xs.shape[0]:, :d]

                T_dataset = np.concatenate([
                    np.concatenate([Xt, Zt_], axis=1),
                    np.concatenate([np.full((bootsize, d), np.nan), bZ_test_], axis=1),
                ])
                T_imputer = mf.ImputationKernel(
                    T_dataset,
                    datasets=self.wd_size,
                    save_all_iterations=False,
                    random_state=self.random_state
                )
                T_imputer.mice(self.n_mice_iters)
                T_complete = np.zeros((self.wd_size, bootsize, d), dtype=dtype)
                for imp in range(self.wd_size):
                    T_complete[imp, :, :] = T_imputer.complete_data(dataset=imp)[Xt.shape[0]:, :d]

                for i in range(bootsize):
                    S_list.append(S_complete[:, i, :])
                    T_list.append(T_complete[:, i, :])
        else:
            # If not use_mice, we use the data directly
            Z_testixs = rng.choice(n_test, size=self.n_bootstraps * bootsize, p=W_test.ravel())
            for list_ix in range(self.n_bootstraps * bootsize):
                i = Z_testixs[list_ix]
                Zs_ixs,  = (Zs == Z_test[i, :]).ravel().nonzero()
                Zt_ixs,  = (Zt == Z_test[i, :]).ravel().nonzero()
                S_list.append(Xs[rng.choice(Zs_ixs, size=self.wd_size), :])
                T_list.append(Xt[rng.choice(Zt_ixs, size=self.wd_size), :])

        if self.center:
            meanS = np.mean(np.concatenate(S_list, axis=0), axis=0, keepdims=True)
            meanT = np.mean(np.concatenate(T_list, axis=0), axis=0, keepdims=True)
            S_list = [curS - meanS for curS in S_list]
            T_list = [curT - meanT for curT in T_list]

        # Now we solve || S M - P T ||^2, with block-diagonal P
        P_list = []
        for curS, curT in zip(S_list, T_list):
            curP, _ = wasserstein_procrustes(
                X=curS,
                Y=curT,
                init_gromov=self.init_gromov,
                max_iter=1,
            )
            P_list.append(curP)

        M = np.eye(d)

        for wpix in range(self.n_iters):
            X = np.concatenate(S_list, axis=0)
            Y = np.concatenate([curP @ curT for curP, curT in zip(P_list, T_list)])

            U, sigma, Vh = np.linalg.svd(X.T @ Y)
            if self.transform_type == 'orthogonal':
                Q = U @ Vh
            elif self.transform_type == 'orthogonal-scale':
                # Manifold Alignment using Procrustes Analysis (ICML2008)
                # argmin_{k, Q} || X - kY Q || = (trace(Sigma)/trace(Y.T@Y), U@V.T)
                # where Y.T @ X = U @ Sigma @ V.T.
                Q = (np.sum(sigma) / np.sum(X ** 2)) * U @ Vh
            M = M @ Q

            S_list = [curS @ Q for curS in S_list]
            P_list = []
            for curS, curT in zip(S_list, T_list):
                curP, _ = wasserstein_procrustes(
                    X=curS,
                    Y=curT,
                    init_gromov=False,
                    max_iter=1,
                )
                P_list.append(curP)

        if self.center:
            # (S - meanS) @ M + meanT = S @ M + (meanT - meanS @ M)
            B = meanT - meanS @ M
        else:
            B = np.zeros((1, d))

        self.M_ = M
        self.b_ = B.flatten()
        self.B_ = B
        self.M_inv_ = np.linalg.inv(self.M_)
        return self

    def transform(
        self,
        Xs,
    ):
        adaptedS = Xs @ self.M_.T + self.B_
        return adaptedS

    def inverse_transform(
        self,
        Xt,
    ):
        adaptedT = (Xt - self.B_) @ self.M_inv_.T
        return adaptedT
