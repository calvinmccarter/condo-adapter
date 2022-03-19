from copy import deepcopy
from typing import Union

import math
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
    Kernel,
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
    custom_kernel,
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
    # TODO: handle confounders of different dtypes
    if confounder_is_cat:
        (Xtestu, Xtestu_counts) = np.unique(Xtest, axis=0, return_counts=True)
        num_testu = Xtestu.shape[0]
    else:
        Xtestu = Xtest
        num_testu = num_test
        Xtestu_counts = np.ones(num_testu)

    if custom_kernel is not None:
        target_kernel = custom_kernel()
        source_kernel = custom_kernel()
    else:
        if confounder_is_cat:
            target_kernel = CatKernel()
            source_kernel = CatKernel()
        else:
            target_kernel = 1.0 * RBF(length_scale=np.std(X_T, axis=0))
            source_kernel = 1.0 * RBF(length_scale=np.std(X_S, axis=0))

    target_similarities = target_kernel(X_T, Xtestu)  # (num_T, num_testu)
    T_weights = target_similarities / np.sum(target_similarities, axis=0, keepdims=True)
    source_similarities = source_kernel(X_S, Xtestu)  # (num_T, num_testu)
    S_weights = source_similarities / np.sum(source_similarities, axis=0, keepdims=True)
    T_torch = torch.from_numpy(T)
    S_torch = torch.from_numpy(S)

    batches = math.ceil(num_S * num_T / (batch_size * batch_size))
    full_epochs = math.floor(epochs)
    frac_epochs = epochs % 1
    terms_per_batch = num_testu * batch_size * batch_size
    debug_dict = {}
    debug_dict["mbos"] = []

    for fix in range(num_feats):
        if fix % (1 + num_feats // 100) == 0:
            print(f"fix:{fix}/{num_feats}")
        M = torch.eye(1, 1, dtype=torch.float64, requires_grad=True)
        b = torch.zeros(1, dtype=torch.float64, requires_grad=True)

        obj_history = []
        best_M = np.eye(1, 1)
        best_b = np.zeros(1)

        for epoch in range(full_epochs + 1):
            epoch_start_M = M.detach().numpy()
            epoch_start_b = b.detach().numpy()
            Mz = torch.zeros(1, 1)
            bz = torch.zeros(1)

            objs = np.zeros(batches)
            if epoch == full_epochs:
                cur_batches = round(frac_epochs * batches)
            else:
                cur_batches = batches
            if debug:
                mbos = np.zeros((cur_batches, 3))
            for batch in range(cur_batches):

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
                if debug:
                    mbos[batch, :] = (
                        M.detach().numpy()[0, 0],
                        b.detach().numpy()[0],
                        obj.detach().numpy(),
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
            if debug:
                debug_dict["mbos"].append(mbos)
            if verbose >= 1:
                print(f"epoch:{epoch} {objs[0]:.5f}->{objs[-1]:.5f} avg:{last_obj:.5f}")
            if epoch > 0 and last_obj < np.min(np.array(obj_history)):
                best_M = epoch_start_M
                best_b = epoch_start_b
            if epoch == full_epochs and full_epochs == 0:
                best_M = M.detach().numpy()
                best_b = b.detach().numpy()
            if len(obj_history) >= 10:
                if last_obj > np.max(np.array(obj_history[-10:])):
                    # Terminate early if worse than all previous 10 iterations
                    if verbose >= 1:
                        print(f"Terminating {fix} after epoch {epoch}: {last_obj:.5f}")
                    break
            obj_history.append(last_obj)

        M_[fix, fix] = best_M
        b_[fix] = best_b

    return (M_, b_, debug_dict)


def run_mmd_independent_fast(
    S: np.ndarray,
    T: np.ndarray,
    X_S: np.ndarray,
    X_T: np.ndarray,
    Xtest: np.ndarray,
    custom_kernel,
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
    print("running fast")
    rng = np.random.RandomState(42)
    num_S = S.shape[0]
    num_T = T.shape[0]
    num_test = Xtest.shape[0]
    num_feats = S.shape[1]
    num_confounders = X_S.shape[1]
    confounder_is_cat = (Xtest.dtype == bool) or not np.issubdtype(
        Xtest.dtype, np.number
    )
    # TODO: handle confounders of different dtypes
    if confounder_is_cat:
        (Xtestu, Xtestu_counts) = np.unique(Xtest, axis=0, return_counts=True)
        num_testu = Xtestu.shape[0]
    else:
        Xtestu = Xtest
        num_testu = num_test
        Xtestu_counts = np.ones(num_testu)

    if custom_kernel is not None:
        target_kernel = custom_kernel()
        source_kernel = custom_kernel()
    else:
        if confounder_is_cat:
            target_kernel = CatKernel()
            source_kernel = CatKernel()
        else:
            target_kernel = 1.0 * RBF(length_scale=np.std(X_T, axis=0))
            source_kernel = 1.0 * RBF(length_scale=np.std(X_S, axis=0))

    target_similarities = target_kernel(X_T, Xtestu)  # (num_T, num_testu)
    T_weights = target_similarities / np.sum(target_similarities, axis=0, keepdims=True)
    source_similarities = source_kernel(X_S, Xtestu)  # (num_T, num_testu)
    S_weights = source_similarities / np.sum(source_similarities, axis=0, keepdims=True)
    T_torch = torch.from_numpy(T)
    S_torch = torch.from_numpy(S)

    M = torch.ones(num_feats, dtype=torch.float64, requires_grad=True)
    b = torch.zeros(num_feats, dtype=torch.float64, requires_grad=True)
    batches = round(num_S * num_T / (batch_size * batch_size))
    full_epochs = math.floor(epochs)
    frac_epochs = epochs % 1
    terms_per_batch = num_testu * batch_size * batch_size
    obj_history = []
    best_M = np.ones(num_feats)
    best_b = np.zeros(num_feats)
    for epoch in range(full_epochs + 1):
        epoch_start_M = M.detach().numpy()
        epoch_start_b = b.detach().numpy()
        Mz = torch.zeros(num_feats)
        bz = torch.zeros(num_feats)
        objs = np.zeros(batches)
        if epoch == full_epochs:
            cur_batches = round(frac_epochs * batches)
        else:
            cur_batches = batches
        for batch in range(cur_batches):
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
                adaptedSsample = S_torch[srcsample_ixs[cix], :] * M.view(
                    1, -1
                ) + b.view(1, -1)
                length_scale_np = (
                    torch.mean((Tsample - adaptedSsample) ** 2, axis=0).detach().numpy()
                )
                invroot_length_scale = 1.0 / np.sqrt(length_scale_np)
                lscaler = torch.from_numpy(invroot_length_scale).view(1, -1)
                scaled_Tsample = Tsample * lscaler
                scaled_adaptedSsample = adaptedSsample * lscaler

                factor = Xtestu_counts[cix] / terms_per_batch
                obj = obj - 2 * factor * torch.sum(
                    torch.exp(
                        -1.0
                        / 2.0
                        * (
                            (scaled_Tsample @ scaled_Tsample.T).diag().unsqueeze(1)
                            - 2 * scaled_Tsample @ scaled_adaptedSsample.T
                            + (scaled_adaptedSsample @ scaled_adaptedSsample.T)
                            .diag()
                            .unsqueeze(0)
                        )
                    )
                )
                obj = obj + factor * torch.sum(
                    torch.exp(
                        -1.0
                        / 2.0
                        * (
                            (scaled_adaptedSsample @ scaled_adaptedSsample.T)
                            .diag()
                            .unsqueeze(1)
                            - 2 * scaled_adaptedSsample @ scaled_adaptedSsample.T
                            + (scaled_adaptedSsample @ scaled_adaptedSsample.T)
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
                print(torch.mean(torch.abs(M.grad)))
            M.grad.zero_()
            b.grad.zero_()
            if verbose >= 2:
                print(
                    f"epoch:{epoch}/{epochs} batch:{batch}/{cur_batches} obj:{obj:.5f}"
                )
            objs[batch] = obj.detach().numpy()

        last_obj = np.mean(objs)
        if verbose >= 1:
            print(f"epoch:{epoch} {objs[0]:.5f}->{objs[-1]:.5f} avg:{last_obj:.5f}")
        if epoch > 0 and last_obj < np.min(np.array(obj_history)):
            best_M = epoch_start_M
            print(best_M)
            best_b = epoch_start_b
        if epoch == full_epochs and full_epochs == 0:
            best_M = M.detach().numpy()
            best_b = b.detach().numpy()
        if len(obj_history) >= 10:
            if last_obj > np.max(np.array(obj_history[-10:])):
                # Terminate early if worse than all previous 10 iterations
                if verbose >= 1:
                    print(
                        f"Terminating {(alpha, beta)} after epoch {epoch}: {last_obj:.5f}"
                    )
                break
        obj_history.append(last_obj)
    best_M = np.diag(best_M)
    return (best_M, best_b, None)


def run_mmd_affine(
    S: np.ndarray,
    T: np.ndarray,
    X_S: np.ndarray,
    X_T: np.ndarray,
    Xtest: np.ndarray,
    custom_kernel,
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
    # TODO: handle confounders of different dtypes
    if confounder_is_cat:
        (Xtestu, Xtestu_counts) = np.unique(Xtest, axis=0, return_counts=True)
        num_testu = Xtestu.shape[0]
    else:
        Xtestu = Xtest
        num_testu = num_test
        Xtestu_counts = np.ones(num_testu)

    if custom_kernel is not None:
        target_kernel = custom_kernel()
        source_kernel = custom_kernel()
    else:
        if confounder_is_cat:
            target_kernel = CatKernel()
            source_kernel = CatKernel()
        else:
            target_kernel = 1.0 * RBF(length_scale=np.std(X_T, axis=0))
            source_kernel = 1.0 * RBF(length_scale=np.std(X_S, axis=0))

    target_similarities = target_kernel(X_T, Xtestu)  # (num_T, num_testu)
    T_weights = target_similarities / np.sum(target_similarities, axis=0, keepdims=True)
    source_similarities = source_kernel(X_S, Xtestu)  # (num_T, num_testu)
    S_weights = source_similarities / np.sum(source_similarities, axis=0, keepdims=True)
    T_torch = torch.from_numpy(T)
    S_torch = torch.from_numpy(S)

    M = torch.eye(num_feats, num_feats, dtype=torch.float64, requires_grad=True)
    b = torch.zeros(num_feats, dtype=torch.float64, requires_grad=True)
    batches = round(num_S * num_T / (batch_size * batch_size))
    full_epochs = math.floor(epochs)
    frac_epochs = epochs % 1
    terms_per_batch = num_testu * batch_size * batch_size
    obj_history = []
    best_M = np.eye(num_feats, num_feats)
    best_b = np.zeros(num_feats)
    for epoch in range(full_epochs + 1):
        epoch_start_M = M.detach().numpy()
        epoch_start_b = b.detach().numpy()
        Mz = torch.zeros(num_feats, num_feats)
        bz = torch.zeros(num_feats)
        objs = np.zeros(batches)
        if epoch == full_epochs:
            cur_batches = round(frac_epochs * batches)
        else:
            cur_batches = batches
        for batch in range(cur_batches):
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
                print(
                    f"epoch:{epoch}/{epochs} batch:{batch}/{cur_batches} obj:{obj:.5f}"
                )
            objs[batch] = obj.detach().numpy()

        last_obj = np.mean(objs)
        if verbose >= 1:
            print(f"epoch:{epoch} {objs[0]:.5f}->{objs[-1]:.5f} avg:{last_obj:.5f}")
        if epoch > 0 and last_obj < np.min(np.array(obj_history)):
            best_M = epoch_start_M
            best_b = epoch_start_b
        if epoch == full_epochs and full_epochs == 0:
            best_M = M.detach().numpy()
            best_b = b.detach().numpy()
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
    num_D = D.shape[0]
    num_feats = D.shape[1]

    oher = make_column_transformer(
        (
            OneHotEncoder(sparse=False, handle_unknown="ignore"),
            make_column_selector(dtype_include="category"),
        ),
        remainder="passthrough",
    )
    XandXtest_df = pd.DataFrame(np.vstack([X, Xtest]))
    X_df = pd.DataFrame(X)
    Xtest_df = pd.DataFrame(Xtest)
    XandXtest_df = XandXtest_df.convert_dtypes()
    X_df = X_df.convert_dtypes()
    Xtest_df = Xtest_df.convert_dtypes()
    # TODO: classify bool as categorical
    cat_columns = [
        col
        for col in XandXtest_df.columns
        if not pd.api.types.is_numeric_dtype(XandXtest_df[col])
    ]
    XandXtest_df[cat_columns] = XandXtest_df[cat_columns].astype("category")
    X_df[cat_columns] = X_df[cat_columns].astype("category")
    Xtest_df[cat_columns] = Xtest_df[cat_columns].astype("category")
    oher.fit(XandXtest_df)
    encodedX = oher.transform(X_df)
    encodedXtest = oher.transform(Xtest_df)

    ridger = RidgeCV(alpha_per_target=True)
    ridger.fit(encodedX, D)
    predD = ridger.predict(encodedX)
    residD = predD - D
    if num_D < 20 * (num_feats**2):
        glassoer = GraphicalLassoCV(verbose=verbose)
        glassoer.fit(residD)
        est_Sigma = glassoer.covariance_
    else:
        est_Sigma = np.cov(residD, rowvar=False) + 1e-4 * np.eye(num_feats)
    predDtest = ridger.predict(encodedXtest)
    est_mus = predDtest
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
    X_df = pd.DataFrame(X)
    Xtest_df = pd.DataFrame(Xtest)
    XandXtest_df = XandXtest_df.convert_dtypes()
    X_df = X_df.convert_dtypes()
    Xtest_df = Xtest_df.convert_dtypes()
    # TODO: classify bool as categorical
    cat_columns = [
        col
        for col in XandXtest_df.columns
        if not pd.api.types.is_numeric_dtype(XandXtest_df[col])
    ]
    XandXtest_df[cat_columns] = XandXtest_df[cat_columns].astype("category")
    X_df[cat_columns] = X_df[cat_columns].astype("category")
    Xtest_df[cat_columns] = Xtest_df[cat_columns].astype("category")
    oher.fit(XandXtest_df)
    encodedX = oher.transform(X_df)
    encodedXtest = oher.transform(Xtest_df)

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
    custom_kernel,
    verbose: Union[bool, int] = 1,
):
    """
    Args:
        D: (num_train, num_feats)
        X: (num_train, num_confounders)
        Xtest: (num_test, num_confounders)
        custom_kernel: None or Kernel
        verbose:

    Returns:
        est_mus: (num_test, num_feats)
        est_sigmas: (num_test, num_feats)
        predictor: sklearn model predicting D given X
    """
    num_test = Xtest.shape[0]
    num_feats = D.shape[1]
    num_confounders = X.shape[1]
    confounder_is_cat = (X.dtype == bool) or not np.issubdtype(X.dtype, np.number)

    est_mus = np.zeros((num_test, num_feats))
    est_sigmas = np.zeros((num_test, num_feats))
    for fix in range(num_feats):
        if fix % (1 + num_feats // 100) == 0:
            print(f"fix:{fix}/{num_feats}")

        if custom_kernel is not None:
            first_kernel = custom_kernel()
        elif confounder_is_cat:
            first_kernel = CatKernel()
        else:
            min_ls = np.sqrt(np.mean((X - X.T) ** 2))
            first_kernel = ConstantKernel(1, (1e-3, 1e3)) * RBF(
                10 * min_ls, (min_ls, 100 * min_ls)
            )
        if confounder_is_cat:
            noise_dict = {}
            Xulist = list(np.unique(X, axis=0))
            for Xuval in Xulist:
                noise_dict[tuple(list(Xuval))] = (
                    np.var(D[np.where(X == Xuval), :]) + 1e-2
                )
            kernel = first_kernel + HeteroscedasticCatKernel(
                noise_dict, np.var(D[:, fix])
            )
            alpha = 1e-3
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
            kernel = first_kernel + HeteroscedasticKernel.construct(
                prototypes,
                1e-3,
                (1e-10, 50.0),
                gamma=5.0,
                gamma_bounds="fixed",
            )
            alpha = 1e-3
            gper = GaussianProcessRegressor(
                kernel=kernel,
                alpha=alpha,
                normalize_y=False,
                n_restarts_optimizer=9,
            )

        gper.fit(X, D[:, fix])

        (est_mu, est_sigma) = gper.predict(Xtest, return_std=True)
        est_mus[:, fix] = est_mu
        est_sigmas[:, fix] = est_sigma

    return (est_mus, est_sigmas, gper)


def homoscedastic_gp_distr(
    D: np.ndarray,
    X: np.ndarray,
    Xtest: np.ndarray,
    custom_kernel,
    verbose: Union[bool, int] = 1,
):
    """
    Args:
        D: (num_train, num_feats)
        X: (num_train, num_confounders)
        Xtest: (num_test, num_confounders)
        custom_kernel: None or Kernel
        verbose:

    Returns:
        est_mus: (num_test, num_feats)
        est_sigmas: (num_test, num_feats)
    """
    num_test = Xtest.shape[0]
    num_feats = D.shape[1]
    num_confounders = X.shape[1]
    confounder_is_cat = (X.dtype == bool) or not np.issubdtype(X.dtype, np.number)
    if confounder_is_cat:
        print("warning: homoscedastic with categorical not recommended")

    est_mus = np.zeros((num_test, num_feats))
    est_sigmas = np.zeros((num_test, num_feats))
    for fix in range(num_feats):
        if custom_kernel is not None:
            kernel = custom_kernel()
            alpha = 1e-3
            gper = GaussianProcessRegressor(
                kernel=kernel,
                alpha=alpha,
                normalize_y=False,
                n_restarts_optimizer=9,
            )
        elif confounder_is_cat:
            kernel = CatKernel() + HeteroscedasticCatKernel({}, np.var(D[:, fix]))
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
    num_test = Xtest.shape[0]
    num_feats = S.shape[1]
    num_confounders = X_S.shape[1]

    m_ = None
    M_ = np.eye(num_feats, num_feats)
    b_ = np.zeros((1, num_feats))

    (Xtestu, Xtestu_counts) = np.unique(Xtest, axis=0, return_counts=True)
    num_testu = Xtestu.shape[0]

    (est_mu_T_all, est_Sigma_T, predictor_T) = joint_linear_distr(
        D=T,
        X=X_T,
        Xtest=Xtestu,
        verbose=verbose,
    )
    Est_mu_T_all = [torch.from_numpy(est_mu_T_all[[i], :].T) for i in range(num_testu)]
    Est_Sigma_T = torch.from_numpy(est_Sigma_T)
    Est_inv_Sigma_T = torch.from_numpy(np.linalg.inv(est_Sigma_T))
    (est_mu_S_all, est_Sigma_S, predictor_S) = joint_linear_distr(
        D=S,
        X=X_S,
        Xtest=Xtestu,
        verbose=verbose,
    )
    Est_mu_S_all = [torch.from_numpy(est_mu_S_all[[i], :].T) for i in range(num_testu)]
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
            for n in range(num_testu):
                # err_n has size (num_feats, 1)
                err_n = M @ Est_mu_S_all[n] + b - Est_mu_T_all[n]
                obj += Xtestu_counts[n] * (err_n.T @ MSMTinv @ err_n).squeeze()
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
            for n in range(num_testu):
                # err_n has size (num_feats, 1)
                err_n = M @ Est_mu_S_all[n] + b - Est_mu_T_all[n]
                obj += Xtestu_counts[n] * (err_n.T @ Est_inv_Sigma_T @ err_n).squeeze()
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
    custom_kernel,
    debug: bool,
    verbose: Union[bool, int],
    method: str = "l-bfgs",
    max_iter: int = 50,
):
    num_feats = S.shape[1]
    num_confounders = X_S.shape[1]
    (Xtestu, Xtestu_ixs, Xtestu_counts) = np.unique(
        Xtest, axis=0, return_inverse=True, return_counts=True
    )
    num_testu = Xtestu.shape[0]

    M_ = np.zeros((num_feats, num_feats))
    m_ = np.zeros(num_feats)
    b_ = np.zeros(num_feats)
    debug_dict = {}
    if model_type == "linear":
        (est_mu_T_all, est_sigma_T_all, predictor_T) = independent_linear_distr(
            D=T,
            X=X_T,
            Xtest=Xtestu,
            verbose=verbose,
        )
        (est_mu_S_all, est_sigma_S_all, predictor_S) = independent_linear_distr(
            D=S,
            X=X_S,
            Xtest=Xtestu,
            verbose=verbose,
        )
    elif model_type == "homoscedastic-gp":
        (est_mu_T_all, est_sigma_T_all, predictor_T) = homoscedastic_gp_distr(
            D=T,
            X=X_T,
            Xtest=Xtestu,
            custom_kernel=custom_kernel,
            verbose=verbose,
        )
        (est_mu_S_all, est_sigma_S_all, predictor_S) = homoscedastic_gp_distr(
            D=S,
            X=X_S,
            Xtest=Xtestu,
            custom_kernel=custom_kernel,
            verbose=verbose,
        )
    elif model_type == "heteroscedastic-gp":
        (est_mu_T_all, est_sigma_T_all, predictor_T) = heteroscedastic_gp_distr(
            D=T,
            X=X_T,
            Xtest=Xtestu,
            custom_kernel=custom_kernel,
            verbose=verbose,
        )
        (est_mu_S_all, est_sigma_S_all, predictor_S) = heteroscedastic_gp_distr(
            D=S,
            X=X_S,
            Xtest=Xtestu,
            custom_kernel=custom_kernel,
            verbose=verbose,
        )
    est_mu_T_all = est_mu_T_all[Xtestu_ixs, :]
    est_mu_S_all = est_mu_S_all[Xtestu_ixs, :]
    est_sigma_T_all = est_sigma_T_all[Xtestu_ixs, :]
    est_sigma_S_all = est_sigma_S_all[Xtestu_ixs, :]

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
            try:
                res = tm.minimize(
                    reverse_kl_obj,
                    mb_init,
                    method=method,
                    max_iter=max_iter,
                    disp=verbose,
                )
                (m_[fix], b_[fix]) = res.x.numpy()
            except:
                print("l-bfgs failure, trying cg")
                print(r_1, r_2, r_3, r_4, r_5, r_6, r_7)
                res = tm.minimize(
                    reverse_kl_obj,
                    mb_init,
                    method="cg",
                    max_iter=max_iter,
                    disp=verbose,
                )
                (m_[fix], b_[fix]) = res.x.numpy()

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
        divergence: Union[None, str] = "mmd",
        custom_kernel=None,
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

            custom_kernel: None or a subclass of sklearn's Kernel.

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
        if transform_type == "affine" and model_type not in ("linear", "empirical"):
            raise ValueError(
                f"incompatible (transform_type, model_type): {(transform_type, model_type)}"
            )
        if (model_type == "empirical") != (divergence == "mmd"):
            raise ValueError(
                f"incompatible (model_type, divergence): {(model_type, divergence)}"
            )
        if custom_kernel is not None:
            if not issubclass(custom_kernel, Kernel):
                raise ValueError("custom_kernel {custom_kernel} is not Kernel")
        if model_type == "linear" and custom_kernel is not None:
            print(f"Ignoring custom_kernel {custom_kernel} since linear model_type")

        self.sampling = sampling
        self.transform_type = transform_type
        self.model_type = model_type
        self.divergence = divergence
        self.custom_kernel = custom_kernel
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
        if num_confounders == 0:
            raise ValueError(f"called fit with num_confounders:{num_confounders}")
        if self.model_type != "linear":
            if num_confounders > 1 and self.custom_kernel is not None:
                # TODO: handle multiple confounders even without custom_kernel
                raise NotImplementedError(
                    f"num_confounders:{num_confounders} requires linear model_type"
                    f"or custom_kernel"
                )

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
                    custom_kernel=self.custom_kernel,
                    debug=self.debug,
                    verbose=self.verbose,
                    **self.optim_kwargs,
                )
            elif self.transform_type == "location-scale":
                self.M_, self.b_, self.debug_dict_ = run_mmd_independent(
                    S=S,
                    T=T,
                    X_S=X_S,
                    X_T=X_T,
                    Xtest=Xtest,
                    custom_kernel=self.custom_kernel,
                    debug=self.debug,
                    verbose=self.verbose,
                    **self.optim_kwargs,
                )
            else:
                assert False
        else:
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
                    custom_kernel=self.custom_kernel,
                    debug=self.debug,
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

        # same as adaptedS = (self.M_ @ S.T).T + self.b_.reshape(1, -1)
        # self.b_.reshape(1, -1) has shape (1, num_feats)
        adaptedS = S @ self.M_.T + self.b_.reshape(1, -1)
        return adaptedS

    def inverse_transform(
        self,
        T,
    ):
        adaptedT = (T - self.b_.reshape(1, -1)) @ self.M_inv_.T
        return adaptedT
