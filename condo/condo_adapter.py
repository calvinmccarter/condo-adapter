from typing import Union

import numpy as np
import pandas as pd
import torch
import torchmin as tm
import sklearn.utils as skut

from sklearn.cluster import KMeans
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    ConstantKernel,
    RBF,
    WhiteKernel,
)

from condo.heteroscedastic_kernel import HeteroscedasticKernel
from condo.cat_kernels import (
    CatKernel,
    HeteroscedasticCatKernel,
)

def joint_conditional_distr(
    D,
    X,
    Xtest,
    multi_confounder: str = "sum",
    heteroscedastic: bool = True,
    verbose: Union[bool, int] = 1,
):
    raise NotImplementedError("joint_conditional_distr")


def independent_conditional_distr(
    D,
    X,
    Xtest,
    multi_confounder: str = "sum",
    heteroscedastic: bool = True,
    verbose: Union[bool, int] = 1,
):
    """
    Args:
        asdf

    Returns:
        est_mus:
            (Xtest.shape[0], D.shape[1])

        est_sigmas:
            (Xtest.shape[0], D.shape[1])
    """
    num_test = Xtest.shape[0]
    num_feats = D.shape[1]
    num_confounders = X.shape[1]
    if num_confounders > 1:
        raise NotImplementedError(f"num_confounders {num_confounders}")
    confounder_is_cat = (X.dtype == bool) or not np.issubdtype(X.dtype, np.number)
    if confounder_is_cat and not heteroscedastic:
        print("warning: homoscedastic with categorical not recommended")

    est_mus = np.zeros((num_test, num_feats))
    est_sigmas = np.zeros((num_test, num_feats))
    for fix in range(num_feats):
        if confounder_is_cat:
            if heteroscedastic:
                kernel = CatKernel() + HeteroscedasticCatKernel()
                alpha = 0.0
                gper = GaussianProcessRegressor(
                    kernel=kernel, alpha=alpha, normalize_y=True,
                )
            else:
                kernel = CatKernel()
                alpha = 0.0
                gper = GaussianProcessRegressor(
                    kernel=kernel, alpha=alpha, normalize_y=True,
                )
        else:
            if heteroscedastic:
                prototypes = KMeans(n_clusters=10).fit(X[:, [fix]]).cluster_centers_
                kernel = (
                    ConstantKernel(1.0, (1e-10, 1000)) * RBF(1, (0.01, 100.0))
                    + HeteroscedasticKernel.construct(
                        prototypes, 1e-3, (1e-10, 50.0),
                        gamma=5.0, gamma_bounds="fixed",
                    )
                )
                alpha = 0.0
                gper = GaussianProcessRegressor(
                    kernel=kernel, alpha=alpha, normalize_y=False)
            else:
                kernel = 1.0 * RBF(length_scale=1, length_scale_bounds=(1, 3e1))
                alpha = 100.0
                # XXX or use gp_extras example below:
                """
                kernel = (
                    ConstantKernel(1.0, (1e-10, 1000)) * RBF(1, (0.01, 100.0))
                    + WhiteKernel(1e-3, (1e-10, 50.0))
                )
                alpha = 0.0
                """
                gper = GaussianProcessRegressor(
                    alpha=alpha, kernel=kernel, normalize_y=False,
                    random_state=0, n_restarts_optimizer=9,
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
        joint: bool = False,
        multi_confounder: str = "sum",
        kld_direction: Union[None, str] = None,
        heteroscedastic: bool = True,
        verbose: Union[bool, int] = 1,
        debug: bool = False,
    ):
        """
        Args:
            sampling: How to sample from dataset
                ("source", "target", "proportional", "equal").

            joint: Whether to model joint distribution over all features.
                Modeling the joint distribution is slower and allows each
                adapted feature to be a function of all other features,
                rather than purely based on that particular feature
                observation.

            multi_confounder: How to construct a kernel from multiple
                confounding variables ("sum", "product"). The default
                is "sum" because this is less likely to overfit.

            kld_direction: Direction of the KL-divergence to minimize.
                Valid options are None, "forward", "reverse".
                The option "forward" corresponds to D_KL(target || source).
                By default (with None), uses "reverse" iff joint is True.

            verbose: Bool or integer that indicates the verbosity.

            debug: Whether to save state for debugging.
        """
        if sampling not in ("source", "target", "proportional", "equal", "optimum"):
            raise ValueError(f"invalid sampling: {sampling}")
        if multi_confounder not in ("sum", "product"):
            raise ValueError(f"invalid multi_confounder: {multi_confounder}")
        if kld_direction not in (None, "forward", "reverse"):
            raise ValueError(f"invalid kld_direction: {kld_direction}")
        
        self.sampling = sampling
        self.joint = joint
        self.multi_confounder = multi_confounder
        self.kld_direction = kld_direction
        self.heteroscedastic = heteroscedastic
        self.verbose = verbose
        self.debug = debug

    def fit(
        self,
        S,
        T,
        X_S: Union[np.ndarray, pd.DataFrame] = None,
        X_T: Union[np.ndarray, pd.DataFrame] = None,
    ):
        """

        Modifies m_, b_, M_, num_feats_

        """
        num_S = S.shape[0]
        num_T = T.shape[0]
        if X_S is None:
            X_S = np.tile(np.array(['dummy']), (num_S, 1))
        if X_T is None:
            X_T = np.tile(np.array(['dummy']), (num_T, 1))
        S, X_S = skut.check_X_y(
            S, X_S, accept_sparse=False, dtype=None, force_all_finite=True,
            ensure_2d=True, allow_nd=False, multi_output=True,
            ensure_min_samples=2, ensure_min_features=1, y_numeric=False,
        )
        T, X_T = skut.check_X_y(
            T, X_T, accept_sparse=False, dtype=None, force_all_finite=True,
            ensure_2d=True, allow_nd=False, multi_output=True,
            ensure_min_samples=2, ensure_min_features=1, y_numeric=False,
        )
        X_S = skut.check_array(
            X_S, dtype=None, ensure_2d=True)
        X_T = skut.check_array(
            X_T, dtype=None, ensure_2d=True)
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
        

        if self.joint and num_feats > 1:
            self.m_ = np.eye((num_feats, num_feats))
            self.b_ = np.zeros((1, num_feats))
            (est_mu_T_all, est_sigma_T_all) = joint_conditional_distr(
                D=T,
                X=X_T,
                Xtest=Xtest,
                multi_confounder=self.multi_confounder,
                heteroscedastic=self.heteroscedastic,
                verbose=self.verbose,
            )
            Est_mu_T_all = [
                torch.from_numpy(est_mu_T_all[[i], :].T)
                for i in range(num_test)
            ]
            Est_inv_sigma_T_all = [
                torch.from_numpy(np.linalg.inv(est_sigma_T_all[i, :, :]))
                for i in range(num_test)
            ]
            (est_mu_S_all, est_sigma_S_all) = joint_conditional_distr(
                D=S,
                X=X_S,
                Xtest=Xtest,
                multi_confounder=self.multi_confounder,
                heteroscedastic=self.heteroscedastic,
                verbose=self.verbose,
            )
            Est_sigma_S_all = [
                torch.from_numpy(est_sigma_S_all[i, :, :])
                for i in range(num_test)
            ]
            Est_mu_S_all = [
                torch.from_numpy(est_mu_S_all[[i], :].T)
                for i in range(num_test)
            ]

            if self.kld_direction == "forward":
                raise NotImplementedError(
                    f"(joint, kld_direction): {(self.joint, self.kld_direction)}")
            elif self.kld_direction == "reverse":
                # TODO: speedup via explicit gradient torchmin trick
                def joint_reverse_kl_obj(mb):
                    M = mb[0:num_feats, :] # (num_feats, num_feats)
                    b = mb[num_feats, :] # (num_feats,)
                    
                    obj = torch.tensor(0.0)
                    for n in range(num_test):
                        # err_n has size (num_feats, 1)
                        err_n = M @ Est_mu_S_all[n] + b - Est_mu_T_all[n]
                        obj += (
                            -1. * torch.logdet(M @ Est_sigma_S_all[n] @ M)
                            + torch.einsum(
                                "ij,ji->",
                                Est_inv_sigma_T_all[n] @ M,
                                Est_sigma_S_all[n] @ M.T,
                            )
                            + (err_n.T @ Est_inv_sigma_T_all[n] @ err_n).squeeze()
                        )
                    return obj
                        
                mb_init = torch.from_numpy(np.hstack([self.m_, self.b_]))
                res = tm.minimize(
                    joint_reverse_kl_obj, mb_init, method="l-bfgs",
                    max_iter=50, disp=0,
                )
                mb_opt = res.x.numpy()
                self.M_ = mb[0:num_feats, :]  # (num_feats, num_feats)
                self.b_ = mb[num_feats, :]  # (num_feats,)
        else:
            print("not joint")
            self.m_ = np.zeros(num_feats)
            self.b_ = np.zeros(num_feats)
            (est_mu_T_all, est_sigma_T_all, gpT) = independent_conditional_distr(
                D=T,
                X=X_T,
                Xtest=Xtest,
                multi_confounder=self.multi_confounder,
                heteroscedastic=self.heteroscedastic,
                verbose=self.verbose,
            )
            est_var_T_all = est_sigma_T_all ** 2
            (est_mu_S_all, est_sigma_S_all, gpS) = independent_conditional_distr(
                D=S,
                X=X_S,
                Xtest=Xtest,
                multi_confounder=self.multi_confounder,
                heteroscedastic=self.heteroscedastic,
                verbose=self.verbose,
            )
            est_var_S_all = est_sigma_S_all ** 2
            if self.debug:
                self.gpS_ = gpS
                self.gpT_ = gpT
            if self.kld_direction == "forward":
                F_1 = np.mean(est_var_T_all / est_var_S_all, axis=0)
                F_2 = np.mean((est_mu_T_all ** 2) / est_var_S_all, axis=0)
                F_3 = np.mean((est_mu_T_all * est_mu_S_all) / est_var_S_all, axis=0)
                F_4 = np.mean(est_mu_T_all / est_var_S_all, axis=0)
                F_5 = np.mean((est_mu_S_all ** 2) / est_var_S_all, axis=0)
                F_6 = np.mean(est_mu_S_all / est_var_S_all, axis=0)
                F_7 = np.mean(1 / est_var_S_all, axis=0)
                for i in range(num_feats):
                    (f_1, f_2, f_3, f_4, f_5, f_6, f_7) = (
                        F_1[i], F_2[i], F_3[i], F_4[i], F_5[i], F_6[i], F_7[i]
                    )
                    def forward_kl_obj(mb):
                        m, b = mb[0], mb[1]
                        obj = (
                            torch.log(m ** 2) + 0.5 * (m ** -2) * (
                                f_1 + f_2 + f_3 * (-2 * m) + f_4 * (-2 * b)
                                + f_5 * (m ** 2) + f_6 * (2 * m * b) + f_7 * b * b
                            )
                        )
                        return obj
                    mb_init = torch.tensor([1.0, 0.0])
                    res = tm.minimize(
                        forward_kl_obj, mb_init, method="l-bfgs",
                        max_iter=50, disp=2,
                    )
                    (self.m_[i], self.b_[i]) = res.x.numpy()
            elif self.kld_direction == "reverse":
                R_1 = np.mean(est_var_S_all / est_var_T_all, axis=0)
                R_2 = np.mean(est_mu_S_all ** 2 / est_var_T_all, axis=0)
                R_3 = np.mean(2 * est_mu_S_all / est_var_T_all, axis=0)
                R_4 = np.mean(2 * est_mu_S_all *  est_mu_T_all / est_var_T_all, axis=0)
                R_5 = np.mean(1 / est_var_T_all, axis=0)
                R_6 = np.mean(2 * est_mu_S_all / est_var_T_all, axis=0)
                # TODO- closed form expression
                for i in range(num_feats):
                    (r_1, r_2, r_3, r_4, r_5, r_6) = (
                        R_1[i], R_2[i], R_3[i], R_4[i], R_5[i], R_6[i]
                    )
                    def reverse_kl_obj(mb):
                        m, b = mb[0], mb[1]
                        obj = (
                            -2 * torch.log(m) + (
                                (m ** 2) * r_1 + (m ** 2) * r_2 + (m * b) * r_3
                                - m * r_4 + (b ** 2) * r_5 - b * r_6
                            )
                        )
                        return obj
                    mb_init = torch.tensor([1.0, 0.0])
                    res = tm.minimize(
                        reverse_kl_obj, mb_init, method="l-bfgs",
                        max_iter=50, disp=1,
                    )
                    (self.m_[i], self.b_[i]) = res.x.numpy()
            else:
                raise ValueError(f"kld_direction: {self.kld_direction}")

        self.num_feats_ = num_feats

        return self


    def transform(
        self,
        S,
    ):
        if self.joint and self.num_feats_ > 1:
            adaptedS = (self.M_ @ S.T).T + self.b_.reshape(1, -1)
        else:
            adaptedS = self.m_ * S + self.b_
        return adaptedS


