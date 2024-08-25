import numpy as np

from sklearn.gaussian_process.kernels import (
    GenericKernelMixin,
    Kernel,
    _approx_fprime,
    Hyperparameter,
    RBF,
)


class CatKernel(GenericKernelMixin, Kernel):
    def __init__(self):
        pass

    def _f(self, s1, s2):
        """
        kernel value between a pair of categories
        """
        return 1.0 if np.array_equal(s1, s2) else 1e-8

    def _g(self, s1, s2):
        """
        kernel derivative between a pair of categories
        """
        return 0.0 if np.array_equal(s1, s2) else 1.0

    def __call__(self, X, Y=None, eval_gradient=False):
        if Y is None:
            Y = X

        if eval_gradient:
            return (
                np.array([[self._f(x, y) for y in Y] for x in X]),
                np.array([[[self._g(x, y)] for y in Y] for x in X]),
            )
        else:
            return np.array([[self._f(x, y) for y in Y] for x in X])

    def diag(self, X):
        return np.array([self._f(x, x) for x in X])

    def is_stationary(self):
        return False


class AdditiveCatKernel(GenericKernelMixin, Kernel):
    def __init__(self):
        # TODO- implement
        pass

    def _f(self, s1, s2):
        """
        kernel value between a pair of categories
        """
        return 1.0 if np.array_equal(s1, s2) else 0.0

    def _g(self, s1, s2):
        """
        kernel derivative between a pair of categories
        """
        return 0.0 if np.array_equal(s1, s2) else 1.0

    def __call__(self, X, Y=None, eval_gradient=False):
        if Y is None:
            Y = X

        if eval_gradient:
            return (
                np.array([[self._f(x, y) for y in Y] for x in X]),
                np.array([[[self._g(x, y)] for y in Y] for x in X]),
            )
        else:
            return np.array([[self._f(x, y) for y in Y] for x in X])

    def diag(self, X):
        return np.array([self._f(x, x) for x in X])

    def is_stationary(self):
        return False


class HeteroscedasticCatKernel(GenericKernelMixin, Kernel):
    def __init__(self, noise_dict, missing_noise):
        # XXX - make these hyperparameters fixed
        self.noise_dict = noise_dict
        self.missing_noise = missing_noise

    def __call__(self, X, Y=None, eval_gradient=False):
        if Y is not None and eval_gradient:
            raise ValueError("Gradient can only be evaluated when Y is None.")

        if Y is None:
            K = np.eye(X.shape[0]) * self.diag(X)
            return K
        else:
            # Setting K to 0 makes the output have a mean of 0
            # Thus, we'll have to sum with CatKernel.
            # But this gives us direct control over the output variance,
            # as is clear from Eqs 2.25 and 2.26 of GP4ML.
            # f_* = [k_*]^T (K + \sigma^2 I)^{-1} y
            # becomes 0 since we set k_* to 0.
            # V[f_*] = k(x_*, x_*) - [k_*]^T (K + \sigma^2 I)^{-1} k_*
            # becomes k(x_*, x_*), the output of self.diag(X).
            K = np.zeros((X.shape[0], Y.shape[0]))
            return K

    def is_stationary(self):
        return False

    def diag(self, X):
        def get_noise(x):
            try:
                xitem = x.item()
                return self.noise_dict.get(xitem, self.missing_noise)
            except:
                return self.missing_noise

        return np.array([get_noise(x) for x in X])


class TRBF(StationaryKernelMixin, NormalizedKernelMixin, Kernel):

    def __init__(self, length_scale=1.0, length_scale_bounds=(1e-5, 1e5)):
        self.length_scale = length_scale
        self.length_scale_bounds = length_scale_bounds

    @property
    def anisotropic(self):
        return np.iterable(self.length_scale) and len(self.length_scale) > 1

    @property
    def hyperparameter_length_scale(self):
        if self.anisotropic:
            return Hyperparameter(
                "length_scale",
                "numeric",
                self.length_scale_bounds,
                len(self.length_scale),
            )
        return Hyperparameter("length_scale", "numeric", self.length_scale_bounds)

    def __call__(self, X, Y=None, eval_gradient=False):
        """Return the kernel k(X, Y) and optionally its gradient.

        Parameters
        ----------
        X : ndarray of shape (n_samples_X, n_features)
            Left argument of the returned kernel k(X, Y)

        Y : ndarray of shape (n_samples_Y, n_features), default=None
            Right argument of the returned kernel k(X, Y). If None, k(X, X)
            if evaluated instead.

        eval_gradient : bool, default=False
            Determines whether the gradient with respect to the log of
            the kernel hyperparameter is computed.
            Only supported when Y is None.

        Returns
        -------
        K : ndarray of shape (n_samples_X, n_samples_Y)
            Kernel k(X, Y)

        K_gradient : ndarray of shape (n_samples_X, n_samples_X, n_dims), \
                optional
            The gradient of the kernel k(X, X) with respect to the log of the
            hyperparameter of the kernel. Only returned when `eval_gradient`
            is True.
        """
        # TODO - implement!!!!!!
        X = np.atleast_2d(X)
        length_scale = _check_length_scale(X, self.length_scale)
        if Y is None:
            dists = pdist(X / length_scale, metric="sqeuclidean")
            K = np.exp(-0.5 * dists)
            # convert from upper-triangular matrix to square matrix
            K = squareform(K)
            np.fill_diagonal(K, 1)
        else:
            if eval_gradient:
                raise ValueError("Gradient can only be evaluated when Y is None.")
            dists = cdist(X / length_scale, Y / length_scale, metric="sqeuclidean")
            K = np.exp(-0.5 * dists)

        if eval_gradient:
                raise ValueError("Gradient not implemented yet.")
        else:
            return K

    def __repr__(self):
        if self.anisotropic:
            return "{0}(length_scale=[{1}])".format(
                self.__class__.__name__,
                ", ".join(map("{0:.3g}".format, self.length_scale)),
            )
        else:  # isotropic
            return "{0}(length_scale={1:.3g})".format(
                self.__class__.__name__, np.ravel(self.length_scale)[0]
            )
