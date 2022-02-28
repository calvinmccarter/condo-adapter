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
    return (est_mus, est_Sigma)


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
    """
    num_test = Xtest.shape[0]
    num_feats = D.shape[1]
    """
    oher = make_column_transformer(
        (
            OneHotEncoder(sparse=False, handle_unknown="ignore"),
            make_column_selector(dtype_include="category"),
        ),
        remainder="passthrough",
    )
    XandXtest = np.vstack([X, Xtest])
    oher.fit(XandXtest)
    encodedX = oher.transform(X)
    encodedXtest = oher.transform(Xtest)
    """
    encodedX = X
    encodedXtest = Xtest

    ridger = RidgeCV(alpha_per_target=True)
    ridger.fit(encodedX, D)
    predD = ridger.predict(encodedX)
    predDtest = ridger.predict(encodedXtest)
    est_mus = predDtest

    residD = predD - D
    est_sigmas = np.std(residD, axis=0, keepdims=True)
    est_sigmas = np.tile(est_sigmas, (num_test, 1))

    return (est_mus, est_sigmas)


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

    return (est_mus, est_sigmas)


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
            # TODO: not sure about normalize_y
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

    return (est_mus, est_sigmas)


class ConDoAdapter:
    def __init__(
        self,
        sampling: str = "source",
        transform_type: str = "independent",
        model_type: str = "linear",
        multi_confounder_kernel: str = "sum",
        kld_direction: Union[None, str] = None,
        verbose: Union[bool, int] = 1,
        debug: bool = False,
    ):
        """
        Args:
            sampling: How to sample from dataset
                ("source", "target", "proportional", "equal").

            transform_type: Whether to jointly transform all features ("joint"),
                or to transform each feature independently ("independent").
                Modeling the joint distribution is slower and allows each
                adapted feature to be a function of all other features,
                rather than purely based on that particular feature
                observation.

            model_type: Model for features given confounders.
                ("linear", "homoscedastic-gp", "heteroscedastic-gp").

            kld_direction: Direction of the KL-divergence to minimize.
                Valid options are None, "forward", "reverse".
                The option "forward" corresponds to D_KL(target || source).
                By default (with None), uses "reverse" if performing joint transform,
                but uses "forward" if performing independent transform.


            multi_confounder_kernel: How to construct a kernel from multiple
                confounding variables ("sum", "product"). The default
                is "sum" because this is less likely to overfit.

            verbose: Bool or integer that indicates the verbosity.

            debug: Whether to save state for debugging.
        """
        if sampling not in ("source", "target", "proportional", "equal", "optimum"):
            raise ValueError(f"invalid sampling: {sampling}")
        if transform_type not in ("joint", "independent"):
            raise ValueError(f"invalid transform_type: {transform_type}")
        if model_type not in ("linear", "homoscedastic-gp", "heteroscedastic-gp"):
            raise ValueError(f"invalid model_type: {model_type}")
        if kld_direction not in (None, "forward", "reverse"):
            raise ValueError(f"invalid kld_direction: {kld_direction}")
        if multi_confounder_kernel not in ("sum", "product"):
            raise ValueError(
                f"invalid multi_confounder_kernel: {multi_confounder_kernel}"
            )
        if transform_type == "joint" and model_type != "linear":
            raise ValueError(f"incompatible (joint, model_type): {(joint, model_type)}")

        self.sampling = sampling
        self.transform_type = transform_type
        self.model_type = model_type
        self.multi_confounder_kernel = multi_confounder_kernel
        self.kld_direction = kld_direction
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

        Modifies m_, b_, M_, num_feats_

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

        if self.transform_type == "joint" and num_feats > 1:
            assert self.model_type == "linear"
            self.m_ = None
            self.M_ = np.eye(num_feats, num_feats)
            self.b_ = np.zeros((1, num_feats))
            (est_mu_T_all, est_Sigma_T) = joint_linear_distr(
                D=T,
                X=X_T,
                Xtest=Xtest,
                verbose=self.verbose,
            )
            Est_mu_T_all = [
                torch.from_numpy(est_mu_T_all[[i], :].T) for i in range(num_test)
            ]
            Est_inv_Sigma_T = torch.from_numpy(np.linalg.inv(est_Sigma_T))
            (est_mu_S_all, est_Sigma_S) = joint_linear_distr(
                D=S,
                X=X_S,
                Xtest=Xtest,
                verbose=self.verbose,
            )
            Est_mu_S_all = [
                torch.from_numpy(est_mu_S_all[[i], :].T) for i in range(num_test)
            ]
            Est_Sigma_S = torch.from_numpy(est_Sigma_S)

            if self.kld_direction == "forward":
                raise NotImplementedError(
                    f"(joint, kld_direction): {(self.joint, self.kld_direction)}"
                )
            elif self.kld_direction == "reverse":
                # TODO: speedup via explicit gradient torchmin trick
                # TODO: or speedup using fact that homoscedastic
                def joint_reverse_kl_obj(mb):
                    M = mb[0:num_feats, :]  # (num_feats, num_feats)
                    b = (mb[num_feats, :]).view(-1, 1)  # (num_feats, 1)

                    obj = num_test * -1.0 * torch.logdet(M @ Est_Sigma_S @ M)
                    obj += num_test * torch.einsum(
                        "ij,ji->",
                        Est_inv_Sigma_T @ M,
                        Est_Sigma_S @ M.T,
                    )
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
                    disp=0,
                )
                mb_opt = res.x.numpy()
                self.M_ = mb_opt[0:num_feats, :]  # (num_feats, num_feats)
                self.b_ = mb_opt[num_feats, :]  # (num_feats,)
        else:
            # Not joint: treating features independently
            self.M_ = None
            self.m_ = np.zeros(num_feats)
            self.b_ = np.zeros(num_feats)
            if self.model_type == "linear":
                (est_mu_T_all, est_sigma_T_all) = independent_linear_distr(
                    D=T,
                    X=X_T,
                    Xtest=Xtest,
                    verbose=self.verbose,
                )
                (est_mu_S_all, est_sigma_S_all) = independent_linear_distr(
                    D=S,
                    X=X_S,
                    Xtest=Xtest,
                    verbose=self.verbose,
                )
            elif self.model_type == "homoscedastic-gp":
                (est_mu_T_all, est_sigma_T_all) = homoscedastic_gp_distr(
                    D=T,
                    X=X_T,
                    Xtest=Xtest,
                    multi_confounder_kernel=self.multi_confounder_kernel,
                    verbose=self.verbose,
                )
                (est_mu_S_all, est_sigma_S_all) = homoscedastic_gp_distr(
                    D=S,
                    X=X_S,
                    Xtest=Xtest,
                    multi_confounder_kernel=self.multi_confounder_kernel,
                    verbose=self.verbose,
                )
            elif self.model_type == "heteroscedastic-gp":
                (est_mu_T_all, est_sigma_T_all) = heteroscedastic_gp_distr(
                    D=T,
                    X=X_T,
                    Xtest=Xtest,
                    multi_confounder_kernel=self.multi_confounder_kernel,
                    verbose=self.verbose,
                )
                (est_mu_S_all, est_sigma_S_all) = heteroscedastic_gp_distr(
                    D=S,
                    X=X_S,
                    Xtest=Xtest,
                    multi_confounder_kernel=self.multi_confounder_kernel,
                    verbose=self.verbose,
                )

            est_var_T_all = est_sigma_T_all**2
            est_var_S_all = est_sigma_S_all**2
            if self.kld_direction == "forward":
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

            elif self.kld_direction == "reverse":
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
            else:
                raise ValueError(f"kld_direction: {self.kld_direction}")

        self.num_feats_ = num_feats

        return self

    def transform(
        self,
        S,
    ):
        if self.transform_type == "joint" and self.num_feats_ > 1:
            adaptedS = (self.M_ @ S.T).T + self.b_.reshape(1, -1)
        else:
            adaptedS = self.m_ * S + self.b_
        return adaptedS
