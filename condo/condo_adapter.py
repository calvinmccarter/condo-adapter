from typing import Union

import numpy as np
import pandas as pd
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

def independent_conditional_distr(
    Y,
    X,
    Ytest,
    multi_confounder,
    kld_direction,
    heteroscedastic,
    verbose,
):
    """
    Args:
        asdf

    Returns:
        est_mus:
            same size as S

        est_sigmas:
            same size as S
    """
    num_test = Ytest.shape[0]
    num_feats = Y.shape[1]
    num_confounders = X.shape[1]
    if num_confounders > 1:
        raise NotImplementedError(f"num_confounders {num_confounders}")
    confounder_is_cat = (
        np.issubdtype(X.dtype, np.number) and not (X.type == bool)
    )
    if confounder_is_cat and not heteroscedastic:
        print("warning: homoscedastic with categorical not recommended")

    est_mus = np.zeros((num_test, num_feats))
    est_sigmas = np.zeros((num_test, num_feats))
    for fix in range(num_feats):
        if confounder_is_cat:
            if heteroscedastic:
                kernel = CatKernel() + HeteroscedasticKernel()
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
                prototypes = KMeans(n_clusters=10).fit(Y[:, [fix]]).cluster_centers_
                kernel = (
                    ConstantKernel(1.0, (1e-10, 1000)) * RBF(1, (0.01, 100.0))
                    + HeteroscedasticKernel.construct(
                        prototypes, 1e-3, (1e-10, 50.0),
                        gamma=5.0, gamma_bounds="fixed",
                    )
                )
                alpha = 0.0
                gper = GaussianProcessRegressor(kernel=kernel, alpha=alpha)
            else:
                kernel = RBF(length_scale=1, length_scale_bounds=(1, 3e1))
                alpha = 100.0
                # XXX or use gp_extras example below:
                kernel = (
                    ConstantKernel(1.0, (1e-10, 1000)) * RBF(1, (0.01, 100.0))
                    + WhiteKernel(1e-3, (1e-10, 50.0))
                )
                alpha = 0.0
                gper = GaussianProcessRegressor(
                    alpha=alpha, kernel=kernel, random_state=0, 
                    n_restarts_optimizer=9,
                )

        gper.fit(Y[:, [fix]], X)
        est_mu, est_sigma = gper.predict(Ytest[:, [fix]], return_std=True)
        est_mus[:, fix] = est_mu
        est_sigmas[:, fix] = est_sigma

    return (est_mus, est_sigmas)
        
        

class ConDoAdapter(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        sampling: str = "source",
        joint: bool = False,
        multi_confounder: str = "sum",
        kld_direction: Union[NoneType, str] = None,
        heteroscedastic: bool = True,
        verbose: Union[bool, int] = 1,
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
                The option "forward" corresponds to D_KL(target || source).
                By default (with None), uses "forward" iff joint is True.

            verbose: Bool or integer that indicates the verbosity.
        """
        if sampling not in ("source", "target", "proportional", "equal"):
            raise ValueError(f"invalid sampling: {sampling}")
        if multi_confounder not in ("sum", "product"):
            raise ValueError(f"invalid multi_confounder: {multi_confounder}")
        if kld_direction is not in (None, "forward", "reverse"):
            raise ValueError(f"invalid kld_direction: {kld_direction}")
        
        self.sampling = sampling
        self.joint = joint
        self.multi_confounder = multi_confounder
        self.kld_direction = kld_direction
        self.verbose = verbose

    def fit(
        self,
        S,
        T,
        X_S: Union[np.ndarray, pd.DataFrame] = None,
        X_T: Union[np.ndarray, pd.DataFrame] = None,
    ):
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
        

        if self.joint:
        kernel_hetero = CatKernel() + HeteroscedasticKernel()
        return self


    def predict(
        self,
        S,
    ):
        return S


