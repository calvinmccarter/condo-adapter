from copy import deepcopy
from typing import Union

import numpy as np
import pandas as pd
import torch
import torchmin as tm
import sklearn.utils as skut

from sklearn.cluster import KMeans
from sklearn.compose import make_column_transformer
from sklearn.compose import make_column_selector
from sklearn.covariance import GraphicalLassoCV
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    ConstantKernel,
    RBF,
    WhiteKernel,
)
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import OneHotEncoder

from condo.heteroscedastic_kernel import HeteroscedasticKernel
from condo.cat_kernels import (
    CatKernel,
    HeteroscedasticCatKernel,
)


def run_mmd_independent(
    S,
    T,
    X_S,
    X_T,
    Xtest,
    debug: bool = False,
    verbose: Union[bool, int] = 0,
    epochs: int = 10,
    batch_size: int = 16,
    alpha: float = 1e-2,
    beta: float = 0.9,
):
    """
    Args:
        epochs: number of times to pass through all observed confounder values.

        batch_size: number of samples to draw from S and T per confounder value.
            kernel matrix will be of size (batch_size, batch_size).
            The number of batches per epoch is num_S*num_T / (batch_size ** 2).

        alpha: gradient descent step size

        beta: Nesterov momentum

    Returns:
        M: (num_feats, num_feats)
        b: (num_feats,)
    """
    num_S = S.shape[0]
    num_T = T.shape[0]
    num_test = Xtest.shape[0]
    num_feats = S.shape[1]
    num_confounders = X_S.shape[1]
    M_ = np.eye(num_feats, num_feats)
    b_ = np.zeros(num_feats)
    confounder_is_cat = (Xtest.dtype == bool) or not np.issubdtype(
        Xtest.dtype, np.number
    )
    assert num_confounders == 1
    # TODO: handle multiple confounders

    if confounder_is_cat:
        target_kernel = CatKernel()
        source_kernel = CatKernel()
        (Xtestu, Xtestu_counts) = np.unique(Xtest, axis=0, return_counts=True)
        num_testu = Xtestu.shape[0]
        # print(Xtestu_counts)
        # print(Xtestu)
    else:
        target_kernel = 1.0 * RBF(length_scale=np.std(X_S))
        source_kernel = 1.0 * RBF(length_scale=np.std(X_T))
        Xtestu = Xtest
        num_testu = num_test
        Xtestu_counts = np.ones(num_testu)
    target_similarities = target_kernel(X_T, Xtestu)  # (num_T, num_testu)
    T_weights = target_similarities / np.sum(target_similarities, axis=0, keepdims=True)
    source_similarities = source_kernel(X_S, Xtestu)  # (num_T, num_testu)
    S_weights = source_similarities / np.sum(source_similarities, axis=0, keepdims=True)
    # print(T_weights)
    # print(S_weights)
    # return
    T_torch = torch.from_numpy(T)
    S_torch = torch.from_numpy(S)

    batches = round(num_S * num_T / (batch_size * batch_size))
    terms_per_batch = num_testu * batch_size * batch_size
    recip = 1.0 / terms_per_batch
    debug_tuple = None
    rbf_factor = -1 / (2 * np.var(np.vstack([T, S])))
    print(f"rbf: {rbf_factor}")
    for fix in range(num_feats):
        M = torch.eye(1, 1, dtype=torch.float64, requires_grad=True)
        M = torch.tensor(1.0, dtype=torch.float64, requires_grad=True)
        b = torch.zeros(1, dtype=torch.float64, requires_grad=True)
        b = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
        for epoch in range(epochs):
            Mz = torch.zeros(1, 1)
            bz = torch.zeros(1)
            Mz = torch.tensor(1.0, dtype=torch.float64, requires_grad=True)
            bz = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)

            objs = np.zeros(batches)
            for batch in range(batches):
                tgtsample_ixs = [
                    np.random.choice(
                        num_T, size=batch_size, replace=True, p=T_weights[:, cix]
                    ).tolist()
                    for cix in range(num_testu)
                ]
                srcsample_ixs = [
                    np.random.choice(
                        num_S, size=batch_size, replace=True, p=S_weights[:, cix]
                    ).tolist()
                    for cix in range(num_testu)
                ]
                obj = torch.tensor(0.0, requires_grad=True)
                for cix in range(num_testu):
                    for tix in range(batch_size):
                        T_cur = T_torch[tgtsample_ixs[cix][tix], fix]
                        for six in range(batch_size):
                            S_cur = S_torch[srcsample_ixs[cix][six], fix]
                            obj = obj - 2 * recip * Xtestu_counts[cix] * torch.exp(
                                rbf_factor * torch.sum((T_cur - (M * S_cur + b)) ** 2)
                            )
                    for six1 in range(batch_size):
                        S_cur1 = S_torch[srcsample_ixs[cix][six1], fix]
                        for six2 in range(batch_size):
                            S_cur2 = S_torch[srcsample_ixs[cix][six2], fix]
                            obj = obj + recip * Xtestu_counts[cix] * torch.exp(
                                rbf_factor * torch.sum(((M * (S_cur1 - S_cur2)) ** 2))
                            )

                obj.backward()
                # print(f"before update M:{M} b:{b} M.grad:{M.grad} b.grad:{b.grad}")
                with torch.no_grad():
                    Mz = beta * Mz + M.grad
                    bz = beta * bz + b.grad
                    M -= alpha * Mz
                    b -= alpha * bz
                # print(f"updated M:{M} b:{b}")

                M.grad.zero_()
                b.grad.zero_()
                if verbose >= 2:
                    print(
                        f"epoch:{epoch}/{epochs} batch:{batch}/{batches} obj:{obj:.5f}"
                    )
                objs[batch] = obj.detach().numpy()
            if verbose >= 1:
                print(
                    f"epoch:{epoch} {objs[0]:.5f}->{objs[-1]:.5f} avg:{np.mean(objs):.5f}"
                )
        M_[fix, fix] = M.detach().numpy()
        b_[fix] = b.detach().numpy()
        if debug and fix == 0 and False:

            def mmd_obj(cur_m, cur_b):
                obj = 0.0
                for cix in range(num_testu):
                    for tix in range(num_T):
                        T_cur = T[tix, fix]
                        # print(f"cix:{cix}/{num_testu} tix:{tix}")
                        for six in range(num_S):
                            S_cur = S[six, fix]
                            obj -= (
                                2
                                * T_weights[tix, cix]
                                * S_weights[six, cix]
                                * Xtestu_counts[cix]
                            ) * np.exp(
                                -0.5 * np.sum((T_cur - (cur_m * S_cur + cur_b)) ** 2)
                            )
                    for six1 in range(num_S):
                        S_cur1 = S[six1, fix]
                        # print(f"cix:{cix}/{num_testu} six1:{six1}")
                        for six2 in range(num_S):
                            S_cur2 = S[six2, fix]
                            obj += (
                                S_weights[six1, cix]
                                * S_weights[six2, cix]
                                * Xtestu_counts[cix]
                            ) * np.exp(
                                -0.5 * np.sum(((cur_m * (S_cur1 - S_cur2)) ** 2))
                            )

            m_plot = np.geomspace(M_[fix, fix] / 5, M_[fix, fix] * 5, 7)
            b_plot = np.linspace(b_[fix] - 10, b_[fix] + 10, 4)
            mb_objs = np.zeros((7, 4))
            for mix in range(7):
                for bix in range(4):
                    with torch.no_grad():
                        mb_objs[mix, bix] = mmd_obj(m_plot[mix], b_plot[bix])
            debug_tuple = (m_plot, b_plot, mb_objs)

    return (M_, b_, debug_tuple)


def run_mmd_affine(
    S,
    T,
    X_S,
    X_T,
    Xtest,
    debug: bool = False,
    verbose: Union[bool, int] = 0,
    epochs: int = 10,
    batch_size: int = 16,
    alpha: float = 1e-2,
    beta: float = 0.9,
):
    """
    Args:
        debug: is ignored

        epochs: number of times to pass through all observed confounder values.

        batch_size: number of samples to draw from S and T per confounder value.
            kernel matrix will be of size (batch_size, batch_size).
            The number of batches per epoch is num_S*num_T / (batch_size ** 2).

        alpha: gradient descent step size

        beta: Nesterov momentum

    Returns:
        M: (num_feats, num_feats)
        b: (num_feats,)
    """
    num_S = S.shape[0]
    num_T = T.shape[0]
    num_test = Xtest.shape[0]
    num_feats = S.shape[1]
    num_confounders = X_S.shape[1]
    M_ = np.eye(num_feats, num_feats)
    b_ = np.zeros((1, num_feats))
    confounder_is_cat = (Xtest.dtype == bool) or not np.issubdtype(
        Xtest.dtype, np.number
    )
    assert num_confounders == 1
    # TODO: handle multiple confounders

    if confounder_is_cat:
        target_kernel = CatKernel()
        source_kernel = CatKernel()
        (Xtestu, Xtestu_counts) = np.unique(Xtest, axis=0, return_counts=True)
        num_testu = Xtestu.shape[0]
    else:
        target_kernel = 1.0 * RBF(length_scale=np.std(X_S))
        source_kernel = 1.0 * RBF(length_scale=np.std(X_T))
        Xtestu = Xtest
        num_testu = num_test
        Xtestu_counts = np.ones(num_testu)
    target_similarities = target_kernel(X_T, Xtestu)  # (num_T, num_testu)
    T_weights = target_similarities / np.sum(target_similarities, axis=0, keepdims=True)
    source_similarities = source_kernel(X_S, Xtestu)  # (num_T, num_testu)
    S_weights = source_similarities / np.sum(source_similarities, axis=0, keepdims=True)
    T_torch = torch.from_numpy(T)
    S_torch = torch.from_numpy(S)

    M = torch.eye(num_feats, num_feats, dtype=torch.float64, requires_grad=True)
    b = torch.zeros(num_feats, dtype=torch.float64, requires_grad=True)
    batches = round(num_S * num_T / (batch_size * batch_size))
    terms_per_batch = num_testu * batch_size * batch_size
    recip = 1.0 / terms_per_batch
    for epoch in range(epochs):
        Mz = torch.zeros(num_feats, num_feats)
        bz = torch.zeros(num_feats)
        objs = np.zeros(batches)
        for batch in range(batches):
            tgtsample_ixs = [
                np.random.choice(
                    num_T, size=batch_size, replace=True, p=T_weights[:, cix]
                ).tolist()
                for cix in range(num_testu)
            ]
            srcsample_ixs = [
                np.random.choice(
                    num_S, size=batch_size, replace=True, p=S_weights[:, cix]
                ).tolist()
                for cix in range(num_testu)
            ]
            obj = torch.tensor(0.0, requires_grad=True)
            for cix in range(num_testu):
                for tix in range(batch_size):
                    T_cur = T_torch[tgtsample_ixs[cix][tix], :]
                    for six in range(batch_size):
                        S_cur = S_torch[srcsample_ixs[cix][six], :]
                        obj = obj - 2 * recip * torch.exp(
                            -0.5 * torch.sum((T_cur - (M @ S_cur + b)) ** 2)
                        )
                for six1 in range(batch_size):
                    S_cur1 = S_torch[srcsample_ixs[cix][six1], :]
                    for six2 in range(batch_size):
                        S_cur2 = S_torch[srcsample_ixs[cix][six2], :]
                        obj = obj + recip * torch.exp(
                            -0.5 * torch.sum(((M @ (S_cur1 - S_cur2)) ** 2))
                        )
            obj.backward()
            with torch.no_grad():
                Mz = beta * Mz + M.grad
                bz = beta * bz + b.grad
                M -= alpha * Mz
                b -= alpha * bz
            M.grad.zero_()
            b.grad.zero_()
            if verbose >= 2:
                print(f"epoch:{epoch}/{epochs} batch:{batch}/{batches} obj:{obj:.5f}")
            objs[batch] = obj.detach().numpy()
        if verbose >= 1:
            print(
                f"epoch:{epoch} {objs[0]:.5f}->{objs[-1]:.5f} avg:{np.mean(objs):.5f}"
            )

    M_ = M.detach().numpy()
    b_ = b.detach().numpy()
    return (M_, b_, None)


def run_mmd_original(
    S,
    T,
    X_S,
    X_T,
    Xtest,
    debug: bool = False,
    verbose: Union[bool, int] = 0,
):
    num_S = S.shape[0]
    num_T = T.shape[0]
    num_test = Xtest.shape[0]
    num_feats = S.shape[1]
    num_confounders = X_S.shape[1]
    M_ = np.eye(num_feats, num_feats)
    b_ = np.zeros(num_feats)
    assert num_confounders == 1

    M_ = np.eye(num_feats, num_feats)
    b_ = np.zeros((1, num_feats))
    num_consample = min(10, num_test)
    num_srcsample = min(20, num_S)
    num_tgtsample = min(20, num_T)
    consample_ixs = np.random.choice(num_test, size=num_consample, replace=False)
    X_consample = Xtest[consample_ixs, :]
    confounder_is_cat = (Xtest.dtype == bool) or not np.issubdtype(
        Xtest.dtype, np.number
    )
    print(confounder_is_cat)
    if confounder_is_cat:
        target_kernel = CatKernel()
        source_kernel = CatKernel()
    else:
        target_kernel = 1.0 * RBF(length_scale=np.std(X_S))
        source_kernel = 1.0 * RBF(length_scale=np.std(X_T))

    target_similarities = target_kernel(X_T, X_consample)  # (num_T, num_consample)
    target_weights = target_similarities / np.sum(
        target_similarities, axis=0, keepdims=True
    )
    tgtsample_ixs = [
        np.random.choice(
            num_T, size=num_tgtsample, replace=False, p=target_weights[:, cix]
        ).tolist()
        for cix in range(num_consample)
    ]
    source_similarities = source_kernel(X_S, X_consample)  # (num_T, num_consample)
    source_weights = source_similarities / np.sum(
        source_similarities, axis=0, keepdims=True
    )
    srcsample_ixs = [
        np.random.choice(
            num_S, size=num_srcsample, replace=False, p=source_weights[:, cix]
        ).tolist()
        for cix in range(num_consample)
    ]
    T_torch = torch.from_numpy(T)
    S_torch = torch.from_numpy(S)

    def joint_mmd_obj(mb):
        M = mb[0:num_feats, :]  # (num_feats, num_feats)
        b = mb[num_feats, :]  # (num_feats,)

        obj = torch.tensor(0.0)
        for cix in range(num_consample):
            for tix in range(num_tgtsample):
                T_cur = T_torch[tgtsample_ixs[cix][tix], :]
                for six in range(num_srcsample):
                    S_cur = S_torch[srcsample_ixs[cix][six], :]
                    obj -= 2 * torch.exp(
                        -0.5 * torch.sum((T_cur - (M @ S_cur + b)) ** 2)
                    )
            for six1 in range(num_srcsample):
                S_cur1 = S_torch[srcsample_ixs[cix][six1], :]
                for six2 in range(num_srcsample):
                    S_cur2 = S_torch[srcsample_ixs[cix][six2], :]
                    obj += torch.exp(-0.5 * torch.sum(((M @ (S_cur1 - S_cur2)) ** 2)))
        return obj

    mb_init = torch.from_numpy(np.vstack([M_, b_]))
    res = tm.minimize(
        joint_mmd_obj,
        mb_init,
        method="l-bfgs",
        max_iter=50,
        disp=verbose,
    )
    mb_opt = res.x.numpy()
    M_ = mb_opt[0:num_feats, :]  # (num_feats, num_feats)
    b_ = mb_opt[num_feats, :]  # (num_feats,)
    return (M_, b_, None)


def joint_linear_distr(
    D: np.ndarray,
    X: np.ndarray,
    Xtest: np.ndarray,
    verbose: Union[bool, int] = 0,
):
    """
    Args:
        D: (num_train, num_feats)
        X: (num_train, num_confounders)
        Xtest: (num_test, num_confounders)

    Returns:
        est_mus: (num_test, num_feats)
        est_Sigma: (num_feats, num_feats)
        predictor: sklearn model predicting D given X
    """
    num_test = Xtest.shape[0]
    num_feats = D.shape[1]

    oher = make_column_transformer(
        (
            OneHotEncoder(sparse=False, handle_unknown="ignore"),
            make_column_selector(dtype_include="category"),
        ),
        remainder="passthrough",
    )
    XandXtest_df = pd.DataFrame(np.vstack([X, Xtest]))
    oher.fit(XandXtest_df)
    encodedX = oher.transform(pd.DataFrame(X))
    encodedXtest = oher.transform(pd.DataFrame(Xtest))

    ridger = RidgeCV(alpha_per_target=True)
    ridger.fit(encodedX, D)
    predD = ridger.predict(encodedX)
    residD = predD - D
    glassoer = GraphicalLassoCV(verbose=verbose)
    glassoer.fit(residD)

    est_Sigma = glassoer.covariance_
    est_mus = ridger.predict(Xtest)
    predictor = ridger
    return (est_mus, est_Sigma, predictor)


def independent_linear_distr(
    D: np.ndarray,
    X: np.ndarray,
    Xtest: np.ndarray,
    verbose: Union[bool, int] = 0,
):
    """
    Args:
        D: (num_train, num_feats)
        X: (num_train, num_confounders)
        Xtest: (num_test, num_confounders)

    Returns:
        est_mus: (num_test, num_feats)
        est_sigmas: (num_test, num_feats)
        predictor: sklearn model predicting D given X
    """
    num_test = Xtest.shape[0]
    num_feats = D.shape[1]

    oher = make_column_transformer(
        (
            OneHotEncoder(sparse=False, handle_unknown="ignore"),
            make_column_selector(dtype_include="category"),
        ),
        remainder="passthrough",
    )
    XandXtest_df = pd.DataFrame(np.vstack([X, Xtest]))
    oher.fit(XandXtest_df)
    encodedX = oher.transform(pd.DataFrame(X))
    encodedXtest = oher.transform(pd.DataFrame(Xtest))

    ridger = RidgeCV(alpha_per_target=True)
    ridger.fit(encodedX, D)
    predD = ridger.predict(encodedX)
    predDtest = ridger.predict(encodedXtest)
    est_mus = predDtest

    residD = predD - D
    est_sigmas = np.std(residD, axis=0, keepdims=True)
    est_sigmas = np.tile(est_sigmas, (num_test, 1))
    predictor = ridger

    return (est_mus, est_sigmas, predictor)


def heteroscedastic_gp_distr(
    D: np.ndarray,
    X: np.ndarray,
    Xtest: np.ndarray,
    multi_confounder_kernel: str = "sum",
    verbose: Union[bool, int] = 1,
):
    """
    Args:
        D: (num_train, num_feats)
        X: (num_train, num_confounders)
        Xtest: (num_test, num_confounders)
        multi_confounder_kernel:
        verbose:

    Returns:
        est_mus: (num_test, num_feats)
        est_sigmas: (num_test, num_feats)
        predictor: sklearn model predicting D given X
    """
    num_test = Xtest.shape[0]
    num_feats = D.shape[1]
    num_confounders = X.shape[1]
    if num_confounders > 1:
        raise NotImplementedError(f"num_confounders {num_confounders}")
    confounder_is_cat = (X.dtype == bool) or not np.issubdtype(X.dtype, np.number)

    est_mus = np.zeros((num_test, num_feats))
    est_sigmas = np.zeros((num_test, num_feats))
    for fix in range(num_feats):
        if confounder_is_cat:
            # Assumes X has single confounder
            noise_dict = dict(
                [
                    (catname, np.var(D[np.where(X[:, 0] == catname), fix]))
                    for catname in list(set(list(X[:, 0])))
                ]
            )
            kernel = CatKernel() + HeteroscedasticCatKernel(noise_dict)
            alpha = 0.0
            gper = GaussianProcessRegressor(
                kernel=kernel,
                alpha=alpha,
                normalize_y=False,
                n_restarts_optimizer=9,
            )
        else:
            prototypes = KMeans(n_clusters=10).fit(X).cluster_centers_
            # hyperparams for HeteroscedasticKernel are from gp_extras-examples
            kernel = ConstantKernel(1.0, (1e-3, 1e3)) * RBF(
                1, (1e-2, 1e2)
            ) + HeteroscedasticKernel.construct(
                prototypes,
                1e-3,
                (1e-10, 50.0),
                gamma=5.0,
                gamma_bounds="fixed",
            )
            alpha = 0.0
            gper = GaussianProcessRegressor(
                kernel=kernel,
                alpha=alpha,
                normalize_y=False,
                n_restarts_optimizer=9,
            )

        gper.fit(X, D[:, fix])

        # TODO: make faster when Xtest rows are not unique
        (est_mu, est_sigma) = gper.predict(Xtest, return_std=True)
        est_mus[:, fix] = est_mu
        est_sigmas[:, fix] = est_sigma

    return (est_mus, est_sigmas, gper)


def homoscedastic_gp_distr(
    D: np.ndarray,
    X: np.ndarray,
    Xtest: np.ndarray,
    multi_confounder_kernel: str = "sum",
    verbose: Union[bool, int] = 1,
):
    """
    Args:
        D: (num_train, num_feats)
        X: (num_train, num_confounders)
        Xtest: (num_test, num_confounders)
        multi_confounder_kernel:
        verbose:

    Returns:
        est_mus: (num_test, num_feats)
        est_sigmas: (num_test, num_feats)
    """
    num_test = Xtest.shape[0]
    num_feats = D.shape[1]
    num_confounders = X.shape[1]
    if num_confounders > 1:
        raise NotImplementedError(f"num_confounders {num_confounders}")
    confounder_is_cat = (X.dtype == bool) or not np.issubdtype(X.dtype, np.number)
    if confounder_is_cat:
        print("warning: homoscedastic with categorical not recommended")

    est_mus = np.zeros((num_test, num_feats))
    est_sigmas = np.zeros((num_test, num_feats))
    for fix in range(num_feats):
        if confounder_is_cat:
            kernel = CatKernel()
            noise_dict = dict(
                [(catname, np.var(D[:, fix])) for catname in list(set(list(X[:, 0])))]
            )
            kernel = CatKernel() + HeteroscedasticCatKernel(noise_dict)

            alpha = 1e-3
            gper = GaussianProcessRegressor(
                kernel=kernel,
                alpha=alpha,
                normalize_y=False,
                n_restarts_optimizer=9,
            )
        else:
            kernel = ConstantKernel(1.0, (1e-3, 1e3)) * RBF(
                length_scale=1, length_scale_bounds=(1e-2, 1e2)
            )
            alpha = 100.0
            # Maybe use the default kernel from gp_extras-examples below:
            """
            kernel = (
                ConstantKernel(1.0, (1e-10, 1000)) * RBF(1, (0.01, 100.0))
                + WhiteKernel(1e-3, (1e-10, 50.0))
            )
            alpha = 0.0
            """
            gper = GaussianProcessRegressor(
                alpha=alpha,
                kernel=kernel,
                normalize_y=False,
                random_state=0,
                n_restarts_optimizer=9,
            )

        gper.fit(X, D[:, fix])

        # TODO: make faster when Xtest rows are not unique
        (est_mu, est_sigma) = gper.predict(Xtest, return_std=True)
        est_mus[:, fix] = est_mu
        est_sigmas[:, fix] = est_sigma

    return (est_mus, est_sigmas, gper)


class ConDoAdapter:
    def __init__(
        self,
        sampling: str = "source",
        transform_type: str = "location-scale",
        model_type: str = "linear",
        multi_confounder_kernel: str = "sum",
        divergence: Union[None, str] = "mmd",
        mmd_kwargs: dict = None,
        verbose: Union[bool, int] = 1,
        debug: bool = False,
    ):
        """
        Args:
            sampling: How to sample from dataset
                ("source", "target", "proportional", "equal").

            transform_type: Whether to jointly transform all features ("affine"),
                or to transform each feature independently ("location-scale").
                Modeling the joint distribution is slower and allows each
                adapted feature to be a function of all other features,
                rather than purely based on that particular feature
                observation.

            model_type: Model for features given confounders.
                ("linear", "homoscedastic-gp", "heteroscedastic-gp", "empirical").

            divergence: Distance / divergence between distributions to minimize.
                Valid options are "mmd", "forward", "reverse".
                The option "forward" corresponds to D_KL(target || source).

            multi_confounder_kernel: How to construct a kernel from multiple
                confounding variables ("sum", "product"). The default
                is "sum" because this is less likely to overfit.

            mmd_kwargs: Args for MMD objective.

            verbose: Bool or integer that indicates the verbosity.

            debug: Whether to save state for debugging.
        """
        if sampling not in ("source", "target", "proportional", "equal", "optimum"):
            raise ValueError(f"invalid sampling: {sampling}")
        if transform_type not in ("affine", "location-scale"):
            raise ValueError(f"invalid transform_type: {transform_type}")
        if model_type not in (
            "linear",
            "homoscedastic-gp",
            "heteroscedastic-gp",
            "empirical",
        ):
            raise ValueError(f"invalid model_type: {model_type}")
        if divergence not in (None, "forward", "reverse", "mmd"):
            raise ValueError(f"invalid divergence: {divergence}")
        if multi_confounder_kernel not in ("sum", "product"):
            raise ValueError(
                f"invalid multi_confounder_kernel: {multi_confounder_kernel}"
            )
        if transform_type == "joint" and model_type not in ("linear", "empirical"):
            raise ValueError(
                f"incompatible (transform_type, model_type): {(transform_type, model_type)}"
            )
        if (model_type == "empirical") != (divergence == "mmd"):
            raise ValueError(
                f"incompatible (model_type, divergence): {(model_type, divergence)}"
            )

        self.sampling = sampling
        self.transform_type = transform_type
        self.model_type = model_type
        self.multi_confounder_kernel = multi_confounder_kernel
        self.divergence = divergence
        self.mmd_kwargs = deepcopy(mmd_kwargs) or {}
        self.verbose = verbose
        self.debug = debug

    def fit(
        self,
        S: np.ndarray,
        T: np.ndarray,
        X_S: np.ndarray,
        X_T: np.ndarray,
    ):
        """

        Modifies M_, b_,

        """
        num_S = S.shape[0]
        num_T = T.shape[0]
        assert not (X_S == X_S[0]).all() or not np.isclose(X_S, X_S[0]).all()
        assert not (X_T == X_T[0]).all() or not np.isclose(X_T, X_T[0]).all()
        S, X_S = skut.check_X_y(
            S,
            X_S,
            accept_sparse=False,
            dtype=None,
            force_all_finite=True,
            ensure_2d=True,
            allow_nd=False,
            multi_output=True,
            ensure_min_samples=2,
            ensure_min_features=1,
            y_numeric=False,
        )
        T, X_T = skut.check_X_y(
            T,
            X_T,
            accept_sparse=False,
            dtype=None,
            force_all_finite=True,
            ensure_2d=True,
            allow_nd=False,
            multi_output=True,
            ensure_min_samples=2,
            ensure_min_features=1,
            y_numeric=False,
        )
        X_S = skut.check_array(X_S, dtype=None, ensure_2d=True)
        X_T = skut.check_array(X_T, dtype=None, ensure_2d=True)
        assert S.shape[1] == T.shape[1]
        assert X_S.shape[1] == X_T.shape[1]

        num_feats = S.shape[1]
        num_confounders = X_S.shape[1]

        if self.sampling == "source":
            Xtest = X_S
        elif self.sampling == "target":
            Xtest = X_T
        elif self.sampling == "proportional":
            Xtest = np.vstack([X_T, X_S])
        elif self.sampling == "equal":
            raise NotImplementedError(f"sampling: {self.sampling}")
        elif self.sampling == "optimum":
            raise NotImplementedError(f"sampling: {self.sampling}")
        else:
            raise ValueError(f"sampling: {self.sampling}")
        num_test = Xtest.shape[0]
        if self.divergence == "mmd":
            if self.transform_type == "affine":
                """
                self.M_, self.b_, debug_tuple = run_mmd_original(
                    S=S,
                    T=T,
                    X_S=X_S,
                    X_T=X_T,
                    Xtest=Xtest,
                )
                """
                self.M_, self.b_, debug_tuple = run_mmd_affine(
                    S=S,
                    T=T,
                    X_S=X_S,
                    X_T=X_T,
                    Xtest=Xtest,
                    debug=self.debug,
                    verbose=self.verbose,
                    **self.mmd_kwargs,
                )
            else:
                self.M_, self.b_, debug_tuple = run_mmd_independent(
                    S=S,
                    T=T,
                    X_S=X_S,
                    X_T=X_T,
                    Xtest=Xtest,
                    debug=self.debug,
                    verbose=self.verbose,
                    **self.mmd_kwargs,
                )
            if debug_tuple is not None:
                (self.m_plot_, self.b_plot_, self.mb_objs_) = debug_tuple
            return self

        if self.transform_type == "joint" and num_feats > 1:
            assert self.model_type == "linear"
            self.m_ = None
            self.M_ = np.eye(num_feats, num_feats)
            self.b_ = np.zeros((1, num_feats))
            (est_mu_T_all, est_Sigma_T, predictor_T) = joint_linear_distr(
                D=T,
                X=X_T,
                Xtest=Xtest,
                verbose=self.verbose,
            )
            Est_mu_T_all = [
                torch.from_numpy(est_mu_T_all[[i], :].T) for i in range(num_test)
            ]
            Est_Sigma_T = torch.from_numpy(est_Sigma_T)
            Est_inv_Sigma_T = torch.from_numpy(np.linalg.inv(est_Sigma_T))
            (est_mu_S_all, est_Sigma_S, predictor_S) = joint_linear_distr(
                D=S,
                X=X_S,
                Xtest=Xtest,
                verbose=self.verbose,
            )
            Est_mu_S_all = [
                torch.from_numpy(est_mu_S_all[[i], :].T) for i in range(num_test)
            ]
            Est_Sigma_S = torch.from_numpy(est_Sigma_S)

            if self.divergence == "forward":

                def joint_forward_kl_obj(mb):
                    M = mb[0:num_feats, :]  # (num_feats, num_feats)
                    b = (mb[num_feats, :]).view(-1, 1)  # (num_feats, 1)

                    MSMTinv = torch.linalg.inv(M @ Est_Sigma_S @ M.T)

                    obj = num_test * torch.logdet(M @ Est_Sigma_S @ M.T)
                    obj += num_test * torch.einsum(
                        "ij,ji->",
                        Est_Sigma_T,
                        MSMTinv,
                    )
                    # obj += num_test * 1e-8 * torch.sum(M ** 2)
                    for n in range(num_test):
                        # err_n has size (num_feats, 1)
                        err_n = M @ Est_mu_S_all[n] + b - Est_mu_T_all[n]
                        obj += (err_n.T @ MSMTinv @ err_n).squeeze()
                    return obj

                mb_init = torch.from_numpy(np.vstack([self.M_, self.b_]))
                res = tm.minimize(
                    joint_forward_kl_obj,
                    mb_init,
                    method="l-bfgs",
                    max_iter=50,
                    disp=self.verbose,
                )
                mb_opt = res.x.numpy()
                self.M_ = mb_opt[0:num_feats, :]  # (num_feats, num_feats)
                self.b_ = mb_opt[num_feats, :]  # (num_feats,)

            elif self.divergence == "reverse":
                # TODO: speedup via explicit gradient torchmin trick
                # TODO: or speedup by vectorizing the for-loop
                # TODO: simplify the logdet(M @ Sigma_S @ M) term
                def joint_reverse_kl_obj(mb):
                    M = mb[0:num_feats, :]  # (num_feats, num_feats)
                    b = (mb[num_feats, :]).view(-1, 1)  # (num_feats, 1)

                    obj = num_test * -1.0 * torch.logdet(M @ Est_Sigma_S @ M.T)
                    obj += num_test * torch.einsum(
                        "ij,ji->",
                        Est_inv_Sigma_T @ M,
                        Est_Sigma_S @ M.T,
                    )
                    # obj += num_test * 1e-8 * torch.sum(M ** 2)
                    # obj += -10 * torch.sum(torch.clamp(M, min=float('-inf'), max=0))
                    for n in range(num_test):
                        # err_n has size (num_feats, 1)
                        err_n = M @ Est_mu_S_all[n] + b - Est_mu_T_all[n]
                        obj += (err_n.T @ Est_inv_Sigma_T @ err_n).squeeze()
                    return obj

                mb_init = torch.from_numpy(np.vstack([self.M_, self.b_]))
                res = tm.minimize(
                    joint_reverse_kl_obj,
                    mb_init,
                    method="l-bfgs",
                    max_iter=50,
                    disp=self.verbose,
                )
                mb_opt = res.x.numpy()
                self.M_ = mb_opt[0:num_feats, :]  # (num_feats, num_feats)
                self.b_ = mb_opt[num_feats, :]  # (num_feats,)
        else:
            # location-scale transformation treats features independently
            self.M_ = np.zeros((num_feats, num_feats))
            self.m_ = np.zeros(num_feats)
            self.b_ = np.zeros(num_feats)
            if self.model_type == "linear":
                (est_mu_T_all, est_sigma_T_all, predictor_T) = independent_linear_distr(
                    D=T,
                    X=X_T,
                    Xtest=Xtest,
                    verbose=self.verbose,
                )
                (est_mu_S_all, est_sigma_S_all, predictor_S) = independent_linear_distr(
                    D=S,
                    X=X_S,
                    Xtest=Xtest,
                    verbose=self.verbose,
                )
            elif self.model_type == "homoscedastic-gp":
                (est_mu_T_all, est_sigma_T_all, predictor_T) = homoscedastic_gp_distr(
                    D=T,
                    X=X_T,
                    Xtest=Xtest,
                    multi_confounder_kernel=self.multi_confounder_kernel,
                    verbose=self.verbose,
                )
                (est_mu_S_all, est_sigma_S_all, predictor_S) = homoscedastic_gp_distr(
                    D=S,
                    X=X_S,
                    Xtest=Xtest,
                    multi_confounder_kernel=self.multi_confounder_kernel,
                    verbose=self.verbose,
                )
            elif self.model_type == "heteroscedastic-gp":
                (est_mu_T_all, est_sigma_T_all, predictor_T) = heteroscedastic_gp_distr(
                    D=T,
                    X=X_T,
                    Xtest=Xtest,
                    multi_confounder_kernel=self.multi_confounder_kernel,
                    verbose=self.verbose,
                )
                (est_mu_S_all, est_sigma_S_all, predictor_S) = heteroscedastic_gp_distr(
                    D=S,
                    X=X_S,
                    Xtest=Xtest,
                    multi_confounder_kernel=self.multi_confounder_kernel,
                    verbose=self.verbose,
                )

            est_var_T_all = est_sigma_T_all**2
            est_var_S_all = est_sigma_S_all**2
            if self.divergence == "forward":
                F_0 = np.mean(
                    est_var_S_all * np.log(est_sigma_S_all / est_sigma_T_all), axis=0
                )
                F_1 = np.mean(est_var_S_all, axis=0)
                F_2 = np.mean(est_mu_T_all * est_mu_S_all, axis=0)
                F_3 = np.mean(est_mu_T_all, axis=0)
                F_4 = np.mean(est_mu_S_all**2, axis=0)
                F_5 = np.mean(est_mu_S_all, axis=0)
                F_6 = np.ones(num_feats)

                # Loop over features since independent not joint
                for i in range(num_feats):
                    (f_0, f_1, f_2, f_3, f_4, f_5, f_6) = (
                        F_0[i],
                        F_1[i],
                        F_2[i],
                        F_3[i],
                        F_4[i],
                        F_5[i],
                        F_6[i],
                    )

                    def forward_kl_obj(mb):
                        m, b = mb[0], mb[1]
                        obj = (
                            +2 * (m**2) * f_0
                            + 2 * (m**2) * torch.log(m) * f_1
                            - 2 * m * f_2
                            - 2 * b * f_3
                            + (m**2) * f_4
                            + 2 * m * b * f_5
                            + (b**2) * f_6
                        )
                        return obj

                    mb_init = torch.tensor([1.0, 0.0])
                    res = tm.minimize(
                        forward_kl_obj,
                        mb_init,
                        method="l-bfgs",
                        max_iter=50,
                        disp=self.verbose,
                    )
                    (self.m_[i], self.b_[i]) = res.x.numpy()

                    if self.debug and i == 0:
                        m_plot = np.geomspace(self.m_[i] / 10, self.m_[i] * 10, 500)
                        b_plot = np.linspace(self.b_[i] - 10, self.b_[i] + 10, 200)
                        self.mb_objs_ = np.zeros((500, 200))
                        for mix in range(500):
                            for bix in range(200):
                                with torch.no_grad():
                                    self.mb_objs_[mix, bix] = forward_kl_obj(
                                        torch.tensor([m_plot[mix], b_plot[bix]])
                                    ).numpy()
                        self.m_plot_ = m_plot
                        self.b_plot_ = b_plot
                self.M_ = np.diag(self.m_)

            elif self.divergence == "reverse":
                R_1 = 2 * np.mean(est_var_T_all, axis=0)
                R_2 = np.mean(est_var_S_all, axis=0)
                R_3 = np.mean(est_mu_S_all**2, axis=0)
                R_4 = 2 * np.mean(est_mu_S_all, axis=0)
                R_5 = 2 * np.mean(est_mu_S_all * est_mu_T_all, axis=0)
                R_6 = np.ones(num_feats)
                R_7 = 2 * np.mean(est_mu_T_all, axis=0)

                # Loop over features since independent not joint
                for i in range(num_feats):
                    (r_1, r_2, r_3, r_4, r_5, r_6, r_7) = (
                        R_1[i],
                        R_2[i],
                        R_3[i],
                        R_4[i],
                        R_5[i],
                        R_6[i],
                        R_7[i],
                    )

                    def reverse_kl_obj(mb):
                        m, b = mb[0], mb[1]
                        obj = (
                            -2 * r_1 * torch.log(m)
                            + r_2 * (m**2)
                            + r_3 * (m**2)
                            + r_4 * m * b
                            - r_5 * m
                            + r_6 * (b**2)
                            - r_7 * b
                        )
                        return obj

                    mb_init = torch.tensor([1.0, 0.0])
                    res = tm.minimize(
                        reverse_kl_obj,
                        mb_init,
                        method="l-bfgs",
                        max_iter=50,
                        disp=self.verbose,
                    )
                    (self.m_[i], self.b_[i]) = res.x.numpy()
                    if self.debug and i == 0:
                        m_plot = np.geomspace(self.m_[i] / 10, self.m_[i] * 10, 500)
                        b_plot = np.linspace(self.b_[i] - 10, self.b_[i] + 10, 200)
                        self.mb_objs_ = np.zeros((500, 200))
                        for mix in range(500):
                            for bix in range(200):
                                with torch.no_grad():
                                    self.mb_objs_[mix, bix] = reverse_kl_obj(
                                        torch.tensor([m_plot[mix], b_plot[bix]])
                                    ).numpy()
                        self.m_plot_ = m_plot
                        self.b_plot_ = b_plot
                self.M_ = np.diag(self.m_)
            else:
                raise ValueError(f"divergence: {self.divergence}")

        return self

    def transform(
        self,
        S,
    ):

        # same as adaptedS = (self.M_ @ S.T).T + self.b_.reshape(1, -1)
        # self.b_.reshape(1, -1) has shape (1, num_feats)
        adaptedS = S @ self.M_.T + self.b_.reshape(1, -1)
        return adaptedS
