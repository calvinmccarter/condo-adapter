import numpy as np
import scipy.stats as spst

from sklearn.preprocessing import OneHotEncoder

def product_prior_float(Z_S, Z_T, bw_method='silverman'):
    n_S, zds = Z_S.shape
    n_T, zdt = Z_T.shape
    assert zds == zdt

    zskder = spst.gaussian_kde(Z_S.T, bw_method=bw_method)
    ztkder = spst.gaussian_kde(Z_T.T, bw_method=bw_method)
    P_SunderT = ztkder.pdf(Z_S.T) # (n_S,)
    P_SunderT = P_SunderT / np.sum(P_SunderT)
    P_TunderS = zskder.pdf(Z_T.T) # (n_T,)
    P_TunderS = P_TunderS / np.sum(P_TunderS)

    Z_test = np.concatenate([Z_S, Z_T], axis=0)  # (n_test, zd)
    P_test = np.concatenate([0.5*P_SunderT, 0.5*P_TunderS], axis=0)
    P_test = P_test.reshape(-1, 1)  # (n_test, 1)
    return Z_test, P_test, None


def product_prior_str(Z_S, Z_T):
    n_S, zd = Z_S.shape
    assert zd == 1
    n_T, zd = Z_T.shape
    assert zd == 1
    Z_test = np.unique(np.concatenate([Z_S, Z_T], axis=0)).reshape(1, -1)  # (1, numu_test)
    p_source_test = np.mean(Z_S == Z_test, axis=0)  # (numu_test,)
    p_target_test = np.mean(Z_T == Z_test, axis=0)  # (numu_test,)
    P_test = np.sqrt(p_source_test * p_target_test)  # (numu_test,)
    P_test = P_test / np.sum(P_test)
    
    Z_test = Z_test.reshape(-1, 1)  # (numu_test, 1)
    P_test = P_test.reshape(-1, 1)  # (numu_test, 1)
    oher = OneHotEncoder(drop=None, sparse_output=False)
    oher.fit(Z_test)
    return Z_test, P_test, oher


def product_prior(Z_S, Z_T):
    assert Z_S.shape[1] == Z_T.shape[1]
    if Z_S.dtype.kind in {'U', 'S'}:
        # str or unicode type
        assert Z_T.dtype.kind in {'U', 'S'}
        assert Z_S.shape[1] == 1
        assert Z_T.shape[1] == 1
        return product_prior_str(Z_S, Z_T)
    else:
        assert Z_T.dtype.kind not in {'U', 'S'}
        return product_prior_float(Z_S, Z_T)


def product_prior_float_old(Z_S, Z_T, bw_method='silverman'):
    n_S, zds = Z_S.shape
    n_T, zdt = Z_T.shape
    assert zds == zdt

    Z_test = np.concatenate([Z_S, Z_T], axis=0)  # (n_test, zd)
    zskder = spst.gaussian_kde(Z_S.T, bw_method=bw_method)
    ztkder = spst.gaussian_kde(Z_T.T, bw_method=bw_method)
    P_test = np.sqrt(zskder.pdf(Z_test.T) * ztkder.pdf(Z_test.T))  # (n_test,)
    P_test = P_test / np.sum(P_test)
    P_test = P_test.reshape(-1, 1)  # (n_test, 1)
    return Z_test, P_test


def get_kernel(Z, custom_kernel):
    num_confounders = Z.shape[1]
    if custom_kernel is not None:
        kernel = custom_kernel()
    else:
        confounder_is_cat = (Z.dtype == bool) or not np.issubdtype(Z.dtype, np.number)
        # TODO: handle confounders of different dtypes
        if confounder_is_cat and num_confounders > 1:
            raise ValueError(
                f"Requires custom_kernel since "
                f"num_confounder = {num_confounders}, with at least "
                f"1 categorical confounder"
            )
        elif confounder_is_cat:
            kernel = CatKernel()
        else:
            (n, d) = Z.shape
            stddev = np.std(Z, axis=0)
            iqr = (spst.scoreatpercentile(Z, 75) - spst.scoreatpercentile(Z, 25)) / 1.349
            silverman = (0.9 * (n * (d + 2) / 4.0) ** (-1.0 / (d + 4))) * np.minimum(
                stddev, iqr
            ) + 1e-8
            kernel = RBF(silverman)

    return kernel


def tsne_kernel(X, Y):
    dist = skmx.pairwise_distances(X, Y, metric='sqeuclidean')
    degrees_of_freedom = max(X.shape[1]-1, 1)
    dist /= degrees_of_freedom
    dist += 1.0
    dist **= (degrees_of_freedom + 1.0) / -2.0
    sims = dist / np.sum(dist, axis=1, keepdims=True)
    return sims


def product_prior_float_old(Z_S, Z_T):
    """
    Returns all unique values in {Z_T, Z_S}, with weights (summing to 1) from
    the product of D_{Z_S} and D_{Z_T}.
    """
    Z_test = np.concatenate([Z_S, Z_T], axis=0)
    source_kernel = get_kernel(Z_S, custom_kernel=None)
    target_kernel = get_kernel(Z_T, custom_kernel=None)
    #source_kernel = tsne_kernel
    #target_kernel = tsne_kernel

    p_source2test = source_kernel(Z_S, Z_test)  # (num_S, num_test)
    p_target2test = target_kernel(Z_T, Z_test)  # (num_T, num_test)
    p_source2test = p_source2test / np.sum(p_source2test, axis=1, keepdims=True)
    p_target2test = p_target2test / np.sum(p_target2test, axis=1, keepdims=True)
    # source_probs[i, j] can be considered a transition probability
    # from source sample i to test sample j.


    # The probability of a test sample (according to source) is
    # proportional to the sum of all transition probabilities to
    # that given test sample. And similarly for target.
    p_test_source = np.sum(p_source2test, axis=0)  # (num_test,)
    p_test_target = np.sum(p_target2test, axis=0)  # (num_test,)
    # Normalize to probabilities:
    p_test_source = p_test_source / np.sum(p_test_source)  # (num_test,)
    p_test_target = p_test_target / np.sum(p_test_target)  # (num_test,)

    # Probability of a test sample is proportional to the root of product:
    P_test = np.sqrt(p_test_source * p_test_target)  # (num_test,)
    P_test = P_test / np.sum(P_test)  # (num_test,)
    np.testing.assert_almost_equal(np.sum(P_test), 1.0, decimal=4)
    assert P_test.shape == (Z_test.shape[0],)
    return Z_test, P_test.reshape(-1, 1)



def product_prior_str_old(Z_S, Z_T):
    """
    Returns all unique values in {Z_T, Z_S}, with weights (summing to 1) from
    the product of D_{Z_S} and D_{Z_T}.
    """
    Z_test = np.concatenate([Z_S, Z_T], axis=0)
    Zu_test, Zu_test_counts = np.unique(Z_test, return_counts=True)
    source_kernel = CatKernel()
    target_kernel = CatKernel()

    p_source2test = source_kernel(Z_S, Zu_test)  # (num_S, numu_test)
    p_target2test = target_kernel(Z_T, Zu_test)  # (num_T, numu_test)
    p_source2test = p_source2test / np.sum(p_source2test, axis=1, keepdims=True)
    p_target2test = p_target2test / np.sum(p_target2test, axis=1, keepdims=True)
    # source_probs[i, j] can be considered a transition probability
    # from source sample i to test sample j.


    # The probability of a test sample (according to source) is
    # proportional to the sum of all transition probabilities to
    # that given test sample. And similarly for target.
    p_test_source = np.sum(p_source2test, axis=0)  # (numu_test,)
    p_test_target = np.sum(p_target2test, axis=0)  # (numu_test,)
    # 
    p_test_source *= Zu_test_counts
    p_test_target *= Zu_test_counts
    # Normalize to probabilities:
    p_test_source = p_test_source / np.sum(p_test_source)  # (num_test,)
    p_test_target = p_test_target / np.sum(p_test_target)  # (num_test,)

    # Probability of a test sample is proportional to the root of product:
    p_test = np.sqrt(p_test_source * p_test_target)  # (num_test,)
    p_test = p_test / np.sum(p_test)  # (num_test,)
    np.testing.assert_almost_equal(np.sum(p_test), 1.0, decimal=4)
    assert p_test.shape == (Z_test.shape[0],)
    return p_test.reshape(-1, 1)

