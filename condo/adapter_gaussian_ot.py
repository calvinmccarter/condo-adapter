import numpy as np
import scipy.linalg as spla

# Modified from Python Optimal Transport (ot.da.LinearTransport)
# which has the below info:
"""
Domain adaptation with optimal transport
"""
# Author: Remi Flamary <remi.flamary@unice.fr>
#         Nicolas Courty <ncourty@irisa.fr>
#         Michael Perrot <michael.perrot@univ-st-etienne.fr>
#         Nathalie Gayraud <nat.gayraud@gmail.com>
#         Ievgen Redko <ievgen.redko@univ-st-etienne.fr>
#         Eloi Tanguy <eloi.tanguy@u-paris.fr>
#
# License: MIT License


class AdapterGaussianOT:
    def __init__(
        self,
        transform_type: str = 'location-scale',
        reg=1e-8,
    ):
        if transform_type not in ('location-scale', 'affine'):
            raise ValueError(f'transform_type: {transform_type}')
        self.transform_type = transform_type
        self.reg = reg


    def fit(
        self,
        Xs: np.ndarray,
        Xt: np.ndarray,
    ):
        assert Xs.shape[1] == Xt.shape[1]
        n, d = Xs.shape

        if self.transform_type == 'affine':
            mXs = np.mean(Xs, axis=0)[None, :]
            mXt = np.mean(Xt, axis=0)[None, :]
            Xs = Xs - mXs
            Xt = Xt - mXt
           
            Cs = np.cov(Xs, rowvar=False) + self.reg * np.eye(d, dtype=Xs.dtype)
            Ct = np.cov(Xt, rowvar=False) + self.reg * np.eye(d, dtype=Xt.dtype)
            
            Cs12 = spla.sqrtm(Cs)
            Cs12inv = np.linalg.inv(Cs12)
            M0 = spla.sqrtm(Cs12 @ Ct @ Cs12)
            A = Cs12inv @ M0 @ Cs12inv
            b = mXt - np.dot(mXs, A)
            
            # Unlike POT, we follow the convention X @ M + b, not A @ X.T + b
            M = A.T
            self.M_ = M
            self.b_ = b.flatten()
            self.M_inv_ = np.linalg.inv(self.M_)

        elif self.transform_type == 'location-scale':
            # Simplifies to m = sigma_t / sigma_s, b = mu_t - m*mu_s
            sigma_s = np.std(Xs, axis=0) + self.reg
            sigma_t = np.std(Xt, axis=0) + self.reg
            mu_s = np.mean(Xs, axis=0)
            mu_t = np.mean(Xt, axis=0)
            self.m_ = sigma_t / sigma_s
            self.b_ = mu_t - self.m_ * mu_s
            self.m_inv_ = 1 / self.m_

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
