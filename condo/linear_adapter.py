import numpy as np
import ot


class LinearAdapter:
    def __init__(
        self,
        transform_type: str = "location-scale",
    ):
        if transform_type not in ("location-scale", "affine"):
            raise ValueError(f"transform_type: {transform_type}")
        self.transform_type = transform_type

        if self.transform_type == "affine":
            self.otda = ot.da.LinearTransport()
        else:
            self.otdas = []

    def fit(
        self,
        Xs,
        Xt,
    ):
        if self.transform_type == "affine":
            self.otda.fit(Xs=Xs, Xt=Xt)
        elif self.transform_type == "location-scale":
            num_feats = Xs.shape[1]

            self.otdas = []
            for fix in range(num_feats):
                cur_otda = ot.da.LinearTransport()
                cur_otda.fit(Xs=Xs[:, [fix]], Xt=Xt[:, [fix]])
                self.otdas.append(cur_otda)
        return self

    def transform(
        self,
        Xs,
    ):
        if self.transform_type == "affine":
            return self.otda.transform(Xs=Xs)
        elif self.transform_type == "location-scale":
            Xs2t = np.zeros_like(Xs)
            num_feats = Xs.shape[1]

            for fix in range(num_feats):
                cur_otda = self.otdas[fix]
                Xs2t[:, [fix]] = cur_otda.transform(Xs=Xs[:, [fix]])

            return Xs2t

    def inverse_transform(
        self,
        Xt,
    ):
        if self.transform_type == "affine":
            return self.otda.inverse_transform(Xt=Xt)
        elif self.transform_type == "location-scale":
            Xt2s = np.zeros_like(Xt)
            num_feats = Xt.shape[1]

            for fix in range(num_feats):
                cur_otda = self.otdas[fix]
                Xt2s[:, [fix]] = cur_otda.inverse_transform(Xt=Xt[:, [fix]])

            return Xt2s
