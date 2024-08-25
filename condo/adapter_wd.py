from copy import deepcopy
from typing import Union

import math
import numpy as np
import sklearn.utils as skut

from condo.ot import wasserstein_procrustes


class AdapterWD:
    def __init__(
        self,
        transform_type: str = 'orthogonal',
        center: bool = True,
        wd_size: int = 20,
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
        self.wd_size = wd_size
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
    ):
        assert Xs.shape[1] == Xt.shape[1]
        d = Xs.shape[1]
        n = min(Xs.shape[0], Xt.shape[0])

        rng = skut.check_random_state(self.random_state)
        bootsize = int(n * self.bootstrap_fraction)

        # Each list has len n_bootstraps * bootsize, with elts shape=(wd_size, d)
        S_list = []  # all sampled Xs
        T_list = []  # all sampled Xt
        for list_ix in range(self.n_bootstraps * bootsize):
            Xs_ixs = rng.choice(Xs.shape[0], size=self.wd_size)
            Xt_ixs = rng.choice(Xt.shape[0], size=self.wd_size)
            S_list.append(Xs[Xs_ixs, :])
            T_list.append(Xt[Xt_ixs, :])

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
