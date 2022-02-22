from typing import Union

import numpy as np
import torchmin as tm

from sklearn.gaussian_process import GaussianProcessRegressor

class ConDoAdapter(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        sampling: str = "source",
        joint: bool = False,
        multi_confounder: str = "sum",
        kld_direction: Union[NoneType, str] = None,
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
        X,
    ):
        return self


    def predict(
        self,
        S,
    ):
        return S


