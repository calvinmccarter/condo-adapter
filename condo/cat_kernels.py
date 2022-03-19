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
