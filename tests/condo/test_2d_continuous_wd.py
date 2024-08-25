import pytest
import numpy as np
import sklearn.datasets as skd
from sklearn.utils import check_random_state

from condo import ConDoAdapterWD

@pytest.mark.parametrize(
    "transform_type",
    ["orthogonal-scale", "orthogonal"],
)
@pytest.mark.parametrize(
    "rescale",
    [1.0, 4.0],
)
def test_2d_categorical_wd(transform_type, rescale):
    if transform_type == "orthogonal" and rescale != 1.0:
        return
    rng = check_random_state(0)    
    n_x, n_y = 200, 200
    data, labels = skd.make_moons(
    (n_x, n_y), shuffle=False, noise=0.1, random_state=0)
    X = data[:n_x, :]
    rng.shuffle(X)
    Q = np.array([[1, 0], [0, -1]]) * rescale
    b = np.array([[0., 0.]])
    Y = X @ Q + rng.normal(0, 0.01, size=X.shape)
    cder = ConDoAdapterWD(transform_type=transform_type)
    Z = np.array(['a', 'b'] * (n_x // 2)).reshape(n_x, 1)
    Z = rng.normal(size=(n_x,1))
    cder.fit(Xs=X, Xt=Y, Zs=Z, Zt=Z)
    np.testing.assert_allclose(Q, cder.M_, atol=0.1, rtol=0.1)
