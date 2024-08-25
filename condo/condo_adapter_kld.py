from copy import deepcopy
from typing import Union

import miceforest as mf
import numpy as np
import torch
import torchmin as tm

from condo.product_prior import product_prior


class ConDoAdapterKLD:
    def __init__(
        self,
        transform_type: str = 'location-scale',
        use_mice_discrete_confounder: bool = False,
        reg: float = 1e-8,  # regularization for covariance
        n_mice_impute: int = 20,
        n_mice_iters: int = 2,
        random_state: int = 42,
        verbose: Union[bool, int] = 1,
    ):
        if transform_type not in {'location-scale', 'affine'}:
            raise NotImplementedError(f'transform_type {transform_type}')
        self.transform_type = transform_type
        self.use_mice_discrete_confounder = use_mice_discrete_confounder
        self.reg = reg
        self.n_mice_impute = n_mice_impute
        self.n_mice_iters = n_mice_iters
        self.random_state = random_state
        self.verbose = verbose

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
        assert Xs.dtype == Xt.dtype
        dtype = Xs.dtype

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

        if use_mice:
            S_dataset = np.concatenate([
                np.concatenate([Xs, Zs_], axis=1),
                np.concatenate([np.full((n_test, d), np.nan), Z_test_], axis=1),
            ])
            S_imputer = mf.ImputationKernel(
                S_dataset,
                datasets=self.n_mice_impute,
                save_all_iterations=False,
                random_state=self.random_state
            )
            S_imputer.mice(self.n_mice_iters)
            S_complete = np.zeros((self.n_mice_impute, n_test, d))
            for imp in range(self.n_mice_impute):
                S_complete[imp, :, :] = S_imputer.complete_data(dataset=imp)[Xs.shape[0]:, :d]

            T_dataset = np.concatenate([
                np.concatenate([Xt, Zt_], axis=1),
                np.concatenate([np.full((n_test, d), np.nan), Z_test_], axis=1),
            ])
            T_imputer = mf.ImputationKernel(
                T_dataset,
                datasets=self.n_mice_impute,
                save_all_iterations=False,
                random_state=self.random_state
            )
            T_imputer.mice(self.n_mice_iters)
            T_complete = np.zeros((self.n_mice_impute, n_test, d))
            for imp in range(self.n_mice_impute):
                T_complete[imp, :, :] = T_imputer.complete_data(dataset=imp)[Xt.shape[0]:, :d]

            est_mu_S_all = np.mean(S_complete, axis=0)  # (n_test, d)
            est_var_S_all = np.var(S_complete, axis=0) + self.reg  # (n_test, d)
            est_mu_T_all = np.mean(T_complete, axis=0)  # (n_test, d)
            est_var_T_all = np.var(T_complete, axis=0) + self.reg  # (n_test, d)
            if self.transform_type == 'affine':
                est_Sigma_S_all = np.zeros((n_test, d, d), dtype=dtype)
                est_invSigma_T_all = np.zeros((n_test, d, d), dtype=dtype)
                lamdaI = self.reg * np.eye(d, dtype=dtype)
                for i in range(n_test):
                    est_Sigma_S_all[i, :, :] = (
                        np.cov(S_complete[:, i, :], rowvar=False) + lamdaI
                    )
                    est_invSigma_T_all[i, :, :] = np.linalg.inv(
                        np.cov(T_complete[:, i, :], rowvar=False) + lamdaI
                    )
        else:
            # If not use_mice, we use the data directly
            est_mu_S_all = np.zeros((n_test, d), dtype=dtype)
            est_mu_T_all = np.zeros((n_test, d), dtype=dtype)
            est_var_S_all = np.zeros((n_test, d), dtype=dtype)
            est_var_T_all = np.zeros((n_test, d), dtype=dtype)
            est_Sigma_S_all = np.zeros((n_test, d, d), dtype=dtype)
            est_invSigma_T_all = np.zeros((n_test, d, d), dtype=dtype)
            lamdaI = self.reg * np.eye(d, dtype=dtype)
            for i in range(n_test):
                Zs_ixs = (Zs == Z_test[i, :]).ravel()
                Zt_ixs = (Zt == Z_test[i, :]).ravel()
                if Zs_ixs.sum() < 1 or Zt_ixs.sum() < 1:
                    est_mu_S_all[i, :] = 0.
                    est_mu_T_all[i, :] = 0.
                    est_var_S_all[i, :] = self.reg
                    est_var_T_all[i, :] = self.reg
                    est_Sigma_S_all[i, :, :] = lamdaI
                    est_invSigma_T_all[i, :, :] = (1 / self.reg) * np.eye(d, dtype=dtype)
                    continue
                est_mu_S_all[i, :] = np.mean(Xs[Zs_ixs, :], axis=0)
                est_mu_T_all[i, :] = np.mean(Xt[Zt_ixs, :], axis=0)
                est_var_S_all[i, :] = np.var(Xs[Zs_ixs, :], axis=0) + self.reg
                est_var_T_all[i, :] = np.var(Xt[Zt_ixs, :], axis=0) + self.reg
                if self.transform_type == 'affine':
                    est_Sigma_S_all[i, :, :] = (
                        np.cov(Xs[Zs_ixs, :], rowvar=False) + lamdaI
                    )
                    est_invSigma_T_all[i, :, :] = np.linalg.inv(
                        np.cov(Xt[Zt_ixs, :], rowvar=False) + lamdaI
                    )

        C_1 = np.mean(W_test * est_mu_T_all, axis=0, keepdims=True)
        C_2 = np.mean(W_test * est_mu_S_all, axis=0, keepdims=True)
        R_A = (
            2 * np.sum(W_test * est_var_S_all, axis=0)
            + 2 * np.sum(W_test * (est_mu_S_all - C_2) ** 2, axis=0)
            + 1e-8
        )
        R_B = 2 * np.sum(W_test * (C_1 - est_mu_T_all) * (est_mu_S_all - C_2), axis=0)
        R_C = -2 * np.sum(W_test * est_var_T_all, axis=0)
        m_ = (-1 * R_B + np.sqrt(R_B**2 - 4 * R_A * R_C)) / (2 * R_A)
        b_ = C_1.squeeze() - m_ * C_2.squeeze()

        self.m_ = m_
        self.m_inv_ = 1 / m_
        self.b_ = b_

        if self.transform_type == 'location-scale':
            return self

        Est_mu_S = []
        Est_mu_T = []
        Est_Sigma_S = []
        Est_invSigma_T = []
        for n in range(n_test):
            Est_mu_S.append(torch.from_numpy(est_mu_S_all[n, :].reshape(d, 1)))
            Est_mu_T.append(torch.from_numpy(est_mu_T_all[n, :].reshape(d, 1)))
            Est_Sigma_S.append(torch.from_numpy(est_Sigma_S_all[n, :, :]))
            Est_invSigma_T.append(torch.from_numpy(est_invSigma_T_all[n, :, :]))

        def joint_reverse_kl_obj(mb):
            M = mb[0:d, :]  # (num_feats, num_feats)
            b = (mb[d, :]).view(d, 1)  # (num_feats, 1)

            obj = torch.tensor(0.0, requires_grad=True)
            for n in range(n_test):
                # err_n has size (num_feats, 1)
                err_n = M @ Est_mu_S[n] + b - Est_mu_T[n]
                obj = obj + W_test[n, 0] * (
                    (err_n.T @ Est_invSigma_T[n] @ err_n).squeeze()
                    - torch.logdet(M @ Est_Sigma_S[n] @ M.T)
                    + torch.einsum('ij,ji->', Est_invSigma_T[n] @ M, Est_Sigma_S[n] @ M.T)
                )
            return obj

        mb_init = torch.from_numpy(np.vstack([np.diag(m_), b_]))
        res = tm.minimize(
            joint_reverse_kl_obj,
            mb_init,
            method='trust-ncg',
            max_iter=10,
            disp=self.verbose,
        )
        mb_opt = res.x.numpy()
        M_ = mb_opt[0:d, :]  # (num_feats, num_feats)
        b_ = mb_opt[d, :]  # (num_feats,)

        self.M_ = M_
        self.b_ = b_
        self.M_inv_ = np.linalg.inv(self.M_)

        return self

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
