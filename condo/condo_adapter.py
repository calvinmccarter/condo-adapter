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
    S: np.ndarray,
    T: np.ndarray,
    X_S: np.ndarray,
    X_T: np.ndarray,
    Xtest: np.ndarray,
    debug: bool,
    verbose: Union[bool, int],
    epochs: int = 100,
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

        alpha: gradient descent learning rate (ie step size).

        beta: Nesterov momentum

    Returns:
        M: (num_feats, num_feats)
        b: (num_feats,)
        debug_tuple: (m_plot, b_plot, mb_objs) or None
    """
    rng = np.random.RandomState(42)
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
    if num_confounders != 1:
        # TODO: handle multiple confounders
        raise NotImplementedError(f"MMD affine num_confounders:{num_confounders}")

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

    batches = round(num_S * num_T / (batch_size * batch_size))
    terms_per_batch = num_testu * batch_size * batch_size
    debug_tuple = None

    for fix in range(num_feats):
        M = torch.eye(1, 1, dtype=torch.float64, requires_grad=True)
        b = torch.zeros(1, dtype=torch.float64, requires_grad=True)

        obj_history = []
        best_M = np.eye(1, 1)
        best_b = np.zeros(1)

        for epoch in range(epochs):
            epoch_start_M = M.detach().numpy()
            epoch_start_b = b.detach().numpy()
            Mz = torch.zeros(1, 1)
            bz = torch.zeros(1)

            objs = np.zeros(batches)
            for batch in range(batches):
                tgtsample_ixs = [
                    rng.choice(
                        num_T, size=batch_size, replace=True, p=T_weights[:, cix]
                    ).tolist()
                    for cix in range(num_testu)
                ]
                srcsample_ixs = [
                    rng.choice(
                        num_S, size=batch_size, replace=True, p=S_weights[:, cix]
                    ).tolist()
                    for cix in range(num_testu)
                ]
                obj = torch.tensor(0.0, requires_grad=True)
                for cix in range(num_testu):
                    Tsample = (T_torch[tgtsample_ixs[cix], fix]).reshape(-1, 1)
                    adaptedSsample = (S_torch[srcsample_ixs[cix], fix]).reshape(
                        -1, 1
                    ) @ M.T + b.reshape(1, -1)
                    length_scale = (
                        torch.mean((Tsample - adaptedSsample) ** 2).detach().numpy()
                    )
                    factor = Xtestu_counts[cix] / terms_per_batch
                    obj = obj - 2 * factor * torch.sum(
                        torch.exp(
                            -1.0
                            / (2 * length_scale)
                            * (
                                (Tsample @ Tsample.T).diag().unsqueeze(1)
                                - 2 * Tsample @ adaptedSsample.T
                                + (adaptedSsample @ adaptedSsample.T)
                                .diag()
                                .unsqueeze(0)
                            )
                        )
                    )
                    obj = obj + factor * torch.sum(
                        torch.exp(
                            -1.0
                            / (2 * length_scale)
                            * (
                                (adaptedSsample @ adaptedSsample.T).diag().unsqueeze(1)
                                - 2 * adaptedSsample @ adaptedSsample.T
                                + (adaptedSsample @ adaptedSsample.T)
                                .diag()
                                .unsqueeze(0)
                            )
                        )
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
                    print(
                        f"epoch:{epoch}/{epochs} batch:{batch}/{batches} obj:{obj:.5f}"
                    )
                objs[batch] = obj.detach().numpy()
            last_obj = np.mean(objs)
            if verbose >= 1:
                print(f"epoch:{epoch} {objs[0]:.5f}->{objs[-1]:.5f} avg:{last_obj:.5f}")
            if epoch > 0 and last_obj < np.min(np.array(obj_history)):
                best_M = epoch_start_M
                best_b = epoch_start_b
            if len(obj_history) >= 10:
                if last_obj > np.max(np.array(obj_history[-10:])):
                    # Terminate early if worse than all previous 10 iterations
                    if verbose >= 1:
                        print(f"Terminating {fix} after epoch {epoch}: {last_obj:.5f}")
                    break
            obj_history.append(last_obj)

        M_[fix, fix] = best_M
        b_[fix] = best_b
        if debug and fix == 0:

            def mmd_obj(cur_m, cur_b):
                tgtsample_ixs = [
                    rng.choice(
                        num_T, size=100, replace=True, p=T_weights[:, cix]
                    ).tolist()
                    for cix in range(num_testu)
                ]
                srcsample_ixs = [
                    rng.choice(
                        num_S, size=100, replace=True, p=S_weights[:, cix]
                    ).tolist()
                    for cix in range(num_testu)
                ]
                obj = torch.tensor(0.0)
                for cix in range(num_testu):
                    Tsample = (T_torch[tgtsample_ixs[cix], fix]).reshape(-1, 1)
                    adaptedSsample = (S_torch[srcsample_ixs[cix], fix]).reshape(
                        -1, 1
                    ) @ M.T + b.reshape(1, -1)
                    length_scale = (
                        torch.mean((Tsample - adaptedSsample) ** 2).detach().numpy()
                    )
                    factor = Xtestu_counts[cix] / terms_per_batch
                    obj = obj - 2 * factor * torch.sum(
                        torch.exp(
                            -1.0
                            / (2 * length_scale)
                            * (
                                (Tsample @ Tsample.T).diag().unsqueeze(1)
                                - 2 * Tsample @ adaptedSsample.T
                                + (adaptedSsample @ adaptedSsample.T)
                                .diag()
                                .unsqueeze(0)
                            )
                        )
                    )
                    obj = obj + factor * torch.sum(
                        torch.exp(
                            -1.0
                            / (2 * length_scale)
                            * (
                                (adaptedSsample @ adaptedSsample.T).diag().unsqueeze(1)
                                - 2 * adaptedSsample @ adaptedSsample.T
                                + (adaptedSsample @ adaptedSsample.T)
                                .diag()
                                .unsqueeze(0)
                            )
                        )
                    )
                return obj.detach().numpy()

            m_plot = np.geomspace(M_[fix, fix] / 10, M_[fix, fix] * 10, 70)
            b_plot = np.linspace(b_[fix] - 10, b_[fix] + 10, 40)
            mb_objs = np.zeros((70, 40))
            for mix in range(70):
                for bix in range(40):
                    with torch.no_grad():
                        mb_objs[mix, bix] = mmd_obj(m_plot[mix], b_plot[bix])
            debug_tuple = (m_plot, b_plot, mb_objs)

    return (M_, b_, debug_tuple)


def run_mmd_affine(
    S: np.ndarray,
    T: np.ndarray,
    X_S: np.ndarray,
    X_T: np.ndarray,
    Xtest: np.ndarray,
    debug: bool,
    verbose: Union[bool, int],
    epochs: int = 100,
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

        alpha: gradient descent learning rate (ie step size).

        beta: Nesterov momentum

    Returns:
        M: (num_feats, num_feats)
        b: (num_feats,)
        debug_tuple: (m_plot, b_plot, mb_objs) or None
    """
    rng = np.random.RandomState(42)
    num_S = S.shape[0]
    num_T = T.shape[0]
    num_test = Xtest.shape[0]
    num_feats = S.shape[1]
    num_confounders = X_S.shape[1]
    confounder_is_cat = (Xtest.dtype == bool) or not np.issubdtype(
        Xtest.dtype, np.number
    )
    if num_confounders != 1:
        # TODO: handle multiple confounders
        raise NotImplementedError(f"MMD affine num_confounders:{num_confounders}")

    if confounder_is_cat:
        # TODO- this did not work, not sure why
        """
        S_noise_dict = dict(
            [
                (catname, np.var(S[np.where(X_S[:, 0] == catname), :]))
                for catname in list(set(list(X_S[:, 0])))
            ]
        )
        source_kernel = CatKernel() + HeteroscedasticCatKernel(S_noise_dict)
        T_noise_dict = dict(
            [
                (catname, np.var(T[np.where(X_T[:, 0] == catname), :]))
                for catname in list(set(list(X_T[:, 0])))
            ]
        )
        target_kernel = CatKernel() + HeteroscedasticCatKernel(T_noise_dict)
        """
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
    obj_history = []
    best_M = np.eye(num_feats, num_feats)
    best_b = np.zeros(num_feats)
    for epoch in range(epochs):
        epoch_start_M = M.detach().numpy()
        epoch_start_b = b.detach().numpy()
        Mz = torch.zeros(num_feats, num_feats)
        bz = torch.zeros(num_feats)
        objs = np.zeros(batches)
        for batch in range(batches):
            tgtsample_ixs = [
                rng.choice(
                    num_T, size=batch_size, replace=True, p=T_weights[:, cix]
                ).tolist()
                for cix in range(num_testu)
            ]
            srcsample_ixs = [
                rng.choice(
                    num_S, size=batch_size, replace=True, p=S_weights[:, cix]
                ).tolist()
                for cix in range(num_testu)
            ]
            obj = torch.tensor(0.0, requires_grad=True)
            for cix in range(num_testu):
                Tsample = T_torch[tgtsample_ixs[cix], :]
                adaptedSsample = S_torch[srcsample_ixs[cix], :] @ M.T + b.reshape(1, -1)
                length_scale = (
                    torch.mean((Tsample - adaptedSsample) ** 2).detach().numpy()
                )
                factor = Xtestu_counts[cix] / terms_per_batch
                obj = obj - 2 * factor * torch.sum(
                    torch.exp(
                        -1.0
                        / (2 * length_scale)
                        * (
                            (Tsample @ Tsample.T).diag().unsqueeze(1)
                            - 2 * Tsample @ adaptedSsample.T
                            + (adaptedSsample @ adaptedSsample.T).diag().unsqueeze(0)
                        )
                    )
                )
                obj = obj + factor * torch.sum(
                    torch.exp(
                        -1.0
                        / (2 * length_scale)
                        * (
                            (adaptedSsample @ adaptedSsample.T).diag().unsqueeze(1)
                            - 2 * adaptedSsample @ adaptedSsample.T
                            + (adaptedSsample @ adaptedSsample.T).diag().unsqueeze(0)
                        )
                    )
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

        last_obj = np.mean(objs)
        if verbose >= 1:
            print(f"epoch:{epoch} {objs[0]:.5f}->{objs[-1]:.5f} avg:{last_obj:.5f}")

        if epoch > 0 and last_obj < np.min(np.array(obj_history)):
            best_M = epoch_start_M
            best_b = epoch_start_b
        if len(obj_history) >= 10:
            if last_obj > np.max(np.array(obj_history[-10:])):
                # Terminate early if worse than all previous 10 iterations
                if verbose >= 1:
                    print(
                        f"Terminating {(alpha, beta)} after epoch {epoch}: {last_obj:.5f}"
                    )
                break
        obj_history.append(last_obj)
    return (best_M, best_b, None)


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
    if num_confounders != 1:
        # TODO: handle multiple confounders
        raise NotImplementedError(f"MMD affine num_confounders:{num_confounders}")

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
            # TODO: X - X.T will be incorrect with multiple confounders
            min_ls = np.sqrt(np.mean((X - X.T) ** 2))
            kernel = ConstantKernel(1, (1e-3, 1e3)) * RBF(
                10 * min_ls, (min_ls, 100 * min_ls)
            ) + HeteroscedasticKernel.construct(
                prototypes,
                1e-3,
                (1e-10, 50.0),
                gamma=5.0,
                gamma_bounds="fixed",
            )
            alpha = 100.0
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
                n_restarts_optimizer=9,
            )

        gper.fit(X, D[:, fix])

        # TODO: make faster when Xtest rows are not unique
        (est_mu, est_sigma) = gper.predict(Xtest, return_std=True)
        est_mus[:, fix] = est_mu
        est_sigmas[:, fix] = est_sigma

    return (est_mus, est_sigmas, gper)


def run_kl_linear_affine(
    S: np.ndarray,
    T: np.ndarray,
    X_S: np.ndarray,
    X_T: np.ndarray,
    Xtest: np.ndarray,
    divergence: str,
    debug: bool,
    verbose: Union[bool, int],
    max_iter: int = 50,
):
    m_ = None
    M_ = np.eye(num_feats, num_feats)
    b_ = np.zeros((1, num_feats))
    (est_mu_T_all, est_Sigma_T, predictor_T) = joint_linear_distr(
        D=T,
        X=X_T,
        Xtest=Xtest,
        verbose=verbose,
    )
    Est_mu_T_all = [torch.from_numpy(est_mu_T_all[[i], :].T) for i in range(num_test)]
    Est_Sigma_T = torch.from_numpy(est_Sigma_T)
    Est_inv_Sigma_T = torch.from_numpy(np.linalg.inv(est_Sigma_T))
    (est_mu_S_all, est_Sigma_S, predictor_S) = joint_linear_distr(
        D=S,
        X=X_S,
        Xtest=Xtest,
        verbose=verbose,
    )
    Est_mu_S_all = [torch.from_numpy(est_mu_S_all[[i], :].T) for i in range(num_test)]
    Est_Sigma_S = torch.from_numpy(est_Sigma_S)

    if divergence == "forward":

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

        mb_init = torch.from_numpy(np.vstack([M_, b_]))
        res = tm.minimize(
            joint_forward_kl_obj,
            mb_init,
            method="l-bfgs",
            max_iter=max_iter,
            disp=verbose,
        )
        mb_opt = res.x.numpy()
        M_ = mb_opt[0:num_feats, :]  # (num_feats, num_feats)
        b_ = mb_opt[num_feats, :]  # (num_feats,)

    elif divergence == "reverse":
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

        mb_init = torch.from_numpy(np.vstack([M_, b_]))
        res = tm.minimize(
            joint_reverse_kl_obj,
            mb_init,
            method="l-bfgs",
            max_iter=max_iter,
            disp=verbose,
        )
        mb_opt = res.x.numpy()
        M_ = mb_opt[0:num_feats, :]  # (num_feats, num_feats)
        b_ = mb_opt[num_feats, :]  # (num_feats,)
    debug_dict = {}
    debug_dict["predictor_T"] = predictor_T
    debug_dict["predictor_S"] = predictor_S
    return (M_, b_, debug_dict)


def run_kl_independent(
    S: np.ndarray,
    T: np.ndarray,
    X_S: np.ndarray,
    X_T: np.ndarray,
    Xtest: np.ndarray,
    model_type: str,
    divergence: str,
    multi_confounder_kernel: str,
    debug: bool,
    verbose: Union[bool, int],
    method: str = "l-bfgs",
    max_iter: int = 50,
):
    num_feats = S.shape[1]
    num_confounders = X_S.shape[1]

    M_ = np.zeros((num_feats, num_feats))
    m_ = np.zeros(num_feats)
    b_ = np.zeros(num_feats)
    debug_dict = {}
    if model_type == "linear":
        (est_mu_T_all, est_sigma_T_all, predictor_T) = independent_linear_distr(
            D=T,
            X=X_T,
            Xtest=Xtest,
            verbose=verbose,
        )
        (est_mu_S_all, est_sigma_S_all, predictor_S) = independent_linear_distr(
            D=S,
            X=X_S,
            Xtest=Xtest,
            verbose=verbose,
        )
    elif model_type == "homoscedastic-gp":
        (est_mu_T_all, est_sigma_T_all, predictor_T) = homoscedastic_gp_distr(
            D=T,
            X=X_T,
            Xtest=Xtest,
            multi_confounder_kernel=multi_confounder_kernel,
            verbose=verbose,
        )
        (est_mu_S_all, est_sigma_S_all, predictor_S) = homoscedastic_gp_distr(
            D=S,
            X=X_S,
            Xtest=Xtest,
            multi_confounder_kernel=multi_confounder_kernel,
            verbose=verbose,
        )
    elif model_type == "heteroscedastic-gp":
        (est_mu_T_all, est_sigma_T_all, predictor_T) = heteroscedastic_gp_distr(
            D=T,
            X=X_T,
            Xtest=Xtest,
            multi_confounder_kernel=multi_confounder_kernel,
            verbose=verbose,
        )
        (est_mu_S_all, est_sigma_S_all, predictor_S) = heteroscedastic_gp_distr(
            D=S,
            X=X_S,
            Xtest=Xtest,
            multi_confounder_kernel=multi_confounder_kernel,
            verbose=verbose,
        )
    debug_dict["predictor_T"] = predictor_T
    debug_dict["predictor_S"] = predictor_S

    est_var_T_all = est_sigma_T_all**2
    est_var_S_all = est_sigma_S_all**2
    if divergence == "forward":
        F_0 = np.mean(est_var_S_all * np.log(est_sigma_S_all / est_sigma_T_all), axis=0)
        F_1 = np.mean(est_var_S_all, axis=0)
        F_2 = np.mean(est_mu_T_all * est_mu_S_all, axis=0)
        F_3 = np.mean(est_mu_T_all, axis=0)
        F_4 = np.mean(est_mu_S_all**2, axis=0)
        F_5 = np.mean(est_mu_S_all, axis=0)
        F_6 = np.ones(num_feats)

        # Loop over features since independent not joint
        for fix in range(num_feats):
            (f_0, f_1, f_2, f_3, f_4, f_5, f_6) = (
                F_0[fix],
                F_1[fix],
                F_2[fix],
                F_3[fix],
                F_4[fix],
                F_5[fix],
                F_6[fix],
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
                method=method,
                max_iter=max_iter,
                disp=verbose,
            )
            (m_[fix], b_[fix]) = res.x.numpy()

            if debug and fix == 0:
                m_plot = np.geomspace(m_[fix] / 10, m_[fix] * 10, 500)
                b_plot = np.linspace(b_[fix] - 10, b_[fix] + 10, 200)
                mb_objs = np.zeros((500, 200))
                for mix in range(500):
                    for bix in range(200):
                        with torch.no_grad():
                            mb_objs[mix, bix] = forward_kl_obj(
                                torch.tensor([m_plot[mix], b_plot[bix]])
                            ).numpy()
                debug_dict["m_plot"] = m_plot
                debug_dict["b_plot"] = b_plot
                debug_dict["mb_objs"] = mb_objs
        M_ = np.diag(m_)
        return (M_, b_, debug_dict)

    elif divergence == "reverse":
        R_1 = -2 * np.mean(est_var_T_all, axis=0)
        R_2 = np.mean(est_var_S_all, axis=0)
        R_3 = np.mean(est_mu_S_all**2, axis=0)
        R_4 = 2 * np.mean(est_mu_S_all, axis=0)
        R_5 = -2 * np.mean(est_mu_S_all * est_mu_T_all, axis=0)
        R_6 = np.ones(num_feats)
        R_7 = -2 * np.mean(est_mu_T_all, axis=0)

        # Loop over features since independent not joint
        for fix in range(num_feats):
            (r_1, r_2, r_3, r_4, r_5, r_6, r_7) = (
                R_1[fix],
                R_2[fix],
                R_3[fix],
                R_4[fix],
                R_5[fix],
                R_6[fix],
                R_7[fix],
            )

            def reverse_kl_obj(mb):
                m, b = mb[0], mb[1]
                obj = (
                    r_1 * torch.log(m)
                    + r_2 * (m**2)
                    + r_3 * (m**2)
                    + r_4 * m * b
                    + r_5 * m
                    + r_6 * (b**2)
                    + r_7 * b
                )
                return obj

            mb_init = torch.tensor([1.0, 0.0])
            res = tm.minimize(
                reverse_kl_obj,
                mb_init,
                method=method,
                max_iter=max_iter,
                disp=verbose,
            )
            (m_[fix], b_[fix]) = res.x.numpy()
            if debug and fix == 0:
                m_plot = np.geomspace(m_[fix] / 10, m_[fix] * 10, 500)
                b_plot = np.linspace(b_[fix] - 10, b_[fix] + 10, 200)
                mb_objs = np.zeros((500, 200))
                for mix in range(500):
                    for bix in range(200):
                        with torch.no_grad():
                            mb_objs[mix, bix] = reverse_kl_obj(
                                torch.tensor([m_plot[mix], b_plot[bix]])
                            ).numpy()
                debug_dict["m_plot"] = m_plot
                debug_dict["b_plot"] = b_plot
                debug_dict["mb_objs"] = mb_objs

        M_ = np.diag(m_)
        return (M_, b_, debug_dict)
    else:
        assert False


class ConDoAdapter:
    def __init__(
        self,
        sampling: str = "source",
        transform_type: str = "location-scale",
        model_type: str = "linear",
        multi_confounder_kernel: str = "sum",
        divergence: Union[None, str] = "mmd",
        optim_kwargs: dict = None,
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

            optim_kwargs: Dict containing args for optimization.
                If mmd, valid keys are "epochs", "batch_size",
                "alpha" (learning rate), and "beta" (momentum).
                If forward or reverse KL, valid keys are "method" (eg "l-bfgs")
                and "max_iter".

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
        if transform_type == "affine" and model_type not in ("linear", "empirical"):
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
        self.optim_kwargs = deepcopy(optim_kwargs) or {}
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

        Modifies M_, b_, possibly m_plot_, b_plot_, mb_objs_

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
                self.M_, self.b_, self.debug_dict_ = run_mmd_affine(
                    S=S,
                    T=T,
                    X_S=X_S,
                    X_T=X_T,
                    Xtest=Xtest,
                    debug=self.debug,
                    verbose=self.verbose,
                    **self.optim_kwargs,
                )
            else:
                self.M_, self.b_, self.debug_dict_ = run_mmd_independent(
                    S=S,
                    T=T,
                    X_S=X_S,
                    X_T=X_T,
                    Xtest=Xtest,
                    debug=self.debug,
                    verbose=self.verbose,
                    **self.optim_kwargs,
                )
            return self

        assert self.divergence in ("forward", "reverse")

        if self.transform_type == "affine" and num_feats > 1:
            assert self.model_type == "linear"
            self.M_, self.b_, self.debug_dict_ = run_kl_linear_affine(
                S=S,
                T=T,
                X_S=X_S,
                X_T=X_T,
                Xtest=Xtest,
                divergence=self.divergence,
                debug=self.debug,
                verbose=self.verbose,
                **self.optim_kwargs,
            )
            return self

        elif self.transform_type == "location-scale":
            # location-scale transformation treats features independently
            assert self.model_type in (
                "linear",
                "homoscedastic-gp",
                "heteroscedastic-gp",
            )
            self.M_, self.b_, self.debug_dict_ = run_kl_independent(
                S=S,
                T=T,
                X_S=X_S,
                X_T=X_T,
                Xtest=Xtest,
                model_type=self.model_type,
                divergence=self.divergence,
                multi_confounder_kernel=self.multi_confounder_kernel,
                debug=self.debug,
                verbose=self.verbose,
                **self.optim_kwargs,
            )
            return self

        return self

    def transform(
        self,
        S,
    ):

        # same as adaptedS = (self.M_ @ S.T).T + self.b_.reshape(1, -1)
        # self.b_.reshape(1, -1) has shape (1, num_feats)
        adaptedS = S @ self.M_.T + self.b_.reshape(1, -1)
        return adaptedS
