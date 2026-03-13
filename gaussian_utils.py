import numpy as np
from numpy.linalg import slogdet, LinAlgError

# def mutual_information(S, s, t):
#     """
#     Given a joint covariance matrix S, calculate the mutual information
#     between variables indexed by the lists s and t.
#     """
#     Hx = np.linalg.slogdet(S[np.ix_(s, s)])[1]
#     Hy = np.linalg.slogdet(S[np.ix_(t, t)])[1]
#     Hxy = np.linalg.slogdet(S[np.ix_(s + t, s + t)])[1]
#     return 0.5 * (Hx + Hy - Hxy) / np.log(2)

def log_likelihood(data, cov):
    """
    Compute the log-likelihood of each sample in `data` under a multivariate Gaussian
    with covariance `cov` and zero mean.

    Parameters
    data : np.ndarray
        Array of shape (d, N) with d variables and N samples.
    cov : np.ndarray
        Covariance matrix of shape (d, d).

    Returns
    ll  : np.ndarray
        Array of shape (N,) with the log-likelihood of each sample.
    """
    d = data.shape[0]
    inv_cov = np.linalg.inv(cov)
    q = np.sum((inv_cov @ data) * data, axis=0) 
    sign, logdet = np.linalg.slogdet(cov)
    if sign <= 0:
        raise ValueError("Covariance matrix must be positive definite")
    ll = -0.5 * (d * np.log(2 * np.pi) + logdet + q)
    return ll

from scipy.linalg import cho_factor, cho_solve

def h(S):
    """
    Computes the differential entropy of a multivariate Gaussian.

    Parameters:
    S : ndarray
        Covariance matrix of the system (assumed to be positive definite).

    Returns:
    float: Half the log-determinant of the covariance matrix.
    """
    sign, logdet = slogdet(S)
    if sign <= 0:
        raise LinAlgError("Matrix is not positive definite.")
    return 0.5 * logdet

def demean(data):
    """
    Demean and standardise time series to mean 0 and unit variance.

    Parameters:
        data (numpy.ndarray):
            Multivariate time series, shape (n, T, m), where n is the number of channels,
            T is the number of time steps, m the number of trials.
    Returns:
        (numpy.ndarray):
            Standardised multivariate time series of shape (n, T, m).
    """
    n, T, m = data.shape if len(data.shape) == 3 else (*data.shape, 1)
    data2d = data.reshape(n, T * m, order="F")

    mean = np.mean(data2d, axis=1, keepdims=True)
    std = np.std(data2d, axis=1, keepdims=True, ddof=1)

    data_dem = (data2d - mean) / std
    return (
        data_dem.reshape(n, T, m, order="F")
        if len(data.shape) == 3
        else data_dem.reshape(n, T, order="F")
    )


def mutual_information(cov, idx1, idx2, pointwise=False, data=None):
    """
    Compute mutual information between two sets of variables.

    Parameters:
        cov (numpy.ndarray): Covariance matrix.
        idx1 (list): Indices of the first set of variables.
        idx2 (list): Indices of the second set of variables.
        pointwise (bool): Whether to compute pointwise mutual information.
        data (numpy.ndarray): Data for pointwise calculation.

    Returns:
        float: Mutual information value in BITS.
    """
    if isinstance(idx1, int) or np.isscalar(idx1):
        idx1 = [idx1]
    if isinstance(idx2, int) or np.isscalar(idx2):
        idx2 = [idx2]
    full_cov = cov[np.ix_(idx1 + idx2, idx1 + idx2)]
    cov1 = cov[np.ix_(idx1, idx1)]
    cov2 = cov[np.ix_(idx2, idx2)]
    mi = (h(cov1) + h(cov2) - h(full_cov)) / np.log(2)
    
    if not pointwise:
        return mi
    else:
        if data is None:
            raise ValueError("Data must be provided for pointwise mutual information.")

        log_px = log_likelihood(data[idx1, :], cov1)
        log_py = log_likelihood(data[idx2, :], cov2)
        log_pxy = log_likelihood(data[idx1 + idx2, :], full_cov)

        pmi = (log_pxy - log_px - log_py) / np.log(2)
        # assert np.isclose(mi, np.mean(pmi)), "Pointwise MI mean does not match non-pointwise MI."

        return (pmi, mi)

def compute_all_mi_terms(cov, nS1, nS2, nT1, nT2, pointwise=False, data=None):
    """
    Compute all mutual information terms needed for PhiID decomposition.
    
    Parameters:
        cov (numpy.ndarray): Covariance matrix of the system.
        nS1 (int): Number of variables in the first source.
        nS2 (int): Number of variables in the second source.
        nT1 (int): Number of variables in the first target.
        nT2 (int): Number of variables in the second target.
        pointwise (bool): Whether to compute pointwise mutual information.
        data (numpy.ndarray): Data for pointwise calculation.   
    Returns:
        dict: A dictionary containing all computed mutual information terms.
    """

    S1 = list(range(nS1))
    S2 = list(range(nS1, nS1 + nS2))
    T1 = list(range(nS1 + nS2, nS1 + nS2 + nT1))
    T2 = list(range(nS1 + nS2 + nT1, nS1 + nS2 + nT1 + nT2))

    def get_mi(idx_src, idx_tgt):
        return mutual_information(cov, idx_src, idx_tgt, pointwise=pointwise, data=data)

    Ixa  = get_mi(S1, T1)
    Ixb  = get_mi(S1, T2)
    Iya  = get_mi(S2, T1)
    Iyb  = get_mi(S2, T2)
    Ixya = get_mi(S1 + S2, T1)
    Ixyb = get_mi(S1 + S2, T2)
    Ixab = get_mi(S1, T1 + T2)
    Iyab = get_mi(S2, T1 + T2)
    Ixyab = get_mi(S1 + S2, T1 + T2)

    keys = ["Ixa","Ixb","Iya","Iyb","Ixya","Ixyb","Ixab","Iyab","Ixyab"]
    values = [Ixa,Ixb,Iya,Iyb,Ixya,Ixyb,Ixab,Iyab,Ixyab]

    if pointwise:
        return [dict(zip(keys, vals)) for vals in zip(*values)]
    else:
        return [dict(zip(keys, values))]


def get_projected_cov(cov, n1=None, n2=None):
    if n1 is None or n2 is None:
        # assume 4D covariance matrix, i.e. univariate sources/targets
        return get_projected_cov4D(cov)
    else:
        return get_projected_covND(cov, n1, n2)


def get_projected_covND(cov, n1, n2):
    """
    Project the covariance matrix onto the four manifolds
    where two sources are conditionally independent of the other two.
    Parameters
    ----------
    cov : ndarray, shape (n1+n2+n1+n2, n1+n2+n1+n2)
        Original covariance matrix
    n1 : int
        Dimension of the first source
    n2 : int
        Dimension of the second source

    Returns
    -------
    cov11, cov12, cov21, cov22 : ndarray, shape (n1+n2+n1+n2, n1+n2+n1+n2)
        Projected covariance matrices. covij is the projection where
        the conditioning is performed on Xi and Yj.
    """
    assert cov.shape == (
        n1 + n2 + n1 + n2,
        n1 + n2 + n1 + n2,
    ), "Covariance matrix shape does not match specified dimensions."
    cov11 = project_covariance(cov, n1, n2, [1], [3], [0, 2])
    cov12 = project_covariance(cov, n1, n2, [1], [2], [0, 3])
    cov21 = project_covariance(cov, n1, n2, [0], [3], [1, 2])
    cov22 = project_covariance(cov, n1, n2, [0], [2], [1, 3])

    return cov11, cov12, cov21, cov22


def project_covariance(Sigma, n1, n2, s1, s2, sc):
    """
    Project Sigma onto the manifold where the variables in s1 and s2
    are conditionally independent given sc.
    The four blocks are ordered as [0,1,2,3] with dimensions [n1,n2,n1,n2].
    """

    # block sizes and start indices
    block_sizes = [n1, n2, n1, n2]
    block_starts = np.cumsum([0] + block_sizes[:-1])  # [0, n1, n1+n2, n1+n2+n1]

    def block_indices(block_list):
        idx = []
        for b in block_list:
            start = block_starts[b]
            end = start + block_sizes[b]
            idx.extend(range(start, end))
        return idx

    ind1 = block_indices(s1)
    ind2 = block_indices(s2)
    indT = block_indices(sc)

    # Extract sub-blocks
    S12 = Sigma[np.ix_(ind1, ind2)]
    S1T = Sigma[np.ix_(ind1, indT)]
    ST2 = Sigma[np.ix_(indT, ind2)]
    STT = Sigma[np.ix_(indT, indT)]

    # Compute projected cross-covariance (conditional independence)
    S12_proj = S1T @ np.linalg.pinv(STT) @ ST2

    # Update Sigma
    Sigma_proj = Sigma.copy()
    Sigma_proj[np.ix_(ind1, ind2)] = S12_proj
    Sigma_proj[np.ix_(ind2, ind1)] = S12_proj.T

    return Sigma_proj


def get_projected_cov4D(cov):
    """
    Project the covariance matrix onto the four manifolds
    where two sources are conditionally independent of the other two.
    Parameters
    ----------
    cov : ndarray, shape (4, 4)
        Original covariance matrix

    Returns
    -------
    cov11, cov12, cov21, cov22 : ndarray, shape (4, 4)
        Projected covariance matrices. covij is the projection where
        the conditioning is performed on Xi and Yj.
    """

    p = cov[0, 1]
    q = cov[0, 2]
    s = cov[0, 3]
    r = cov[1, 2]
    t = cov[1, 3]
    u = cov[2, 3]

    def get_4matrix(p, q, r, s, t, u):
        cov = np.array([[1.0, p, q, s], [p, 1.0, r, t], [q, r, 1.0, u], [s, t, u, 1.0]])
        return cov

    constr = (p * q * u - r * u + q * r * s - p * s) / (q**2 - 1)
    cov11 = get_4matrix(p, q, r, s, constr, u)

    constr = (p * s * u - t * u + q * s * t - p * q) / (s**2 - 1)
    cov12 = get_4matrix(p, q, constr, s, t, u)

    constr = (p * r * u - q * u + q * r * t - p * t) / (r**2 - 1)
    cov21 = get_4matrix(p, q, r, constr, t, u)

    constr = (p * t * u - s * u + r * s * t - p * r) / (t**2 - 1)
    cov22 = get_4matrix(p, constr, r, s, t, u)

    return cov11, cov12, cov21, cov22


def KL_gaussian(Sigma_P, Sigma_Q):
    """
    KL divergence D(P || Q) between two zero-mean multivariate Gaussians.

    Parameters
    ----------
    Sigma_P : ndarray
        Covariance matrix of P
    Sigma_Q : ndarray
        Covariance matrix of Q

    Returns
    -------
    float
        KL divergence
    """
    n = Sigma_P.shape[0]
    inv_Q = np.linalg.inv(Sigma_Q)
    trace_term = np.trace(inv_Q @ Sigma_P)
    if np.linalg.det(Sigma_Q) < 0:
        raise ValueError("Covariance matrix Sigma_Q is not positive definite.")
    logdet_term = np.log(np.linalg.det(Sigma_Q) / np.linalg.det(Sigma_P))
    return 0.5 * (trace_term - n + logdet_term)


def KL_gaussian_inv(Sigma_P, Sigma_Q_inv):
    """
    KL divergence D(P || Q) between two zero-mean multivariate Gaussians.

    Parameters
    ----------
    Sigma_P : ndarray
        Covariance matrix of P
    Sigma_Q_inv : ndarray
        Inverse covariance matrix of Q

    Returns
    -------
    float
        KL divergence
    """
    n = Sigma_P.shape[0]
    inv_Q = Sigma_Q_inv
    trace_term = np.trace(inv_Q @ Sigma_P)
    if np.linalg.det(np.linalg.inv(inv_Q)) < 0:
        raise ValueError("Covariance matrix Sigma_Q is not positive definite.")
    logdet_term = np.log(np.linalg.det(Sigma_P @ inv_Q))
    return 0.5 * (trace_term - n - logdet_term)


def pointwise_pid_IG(cov, min_cov, data, nS1=1, nS2=1, nT=1, only_syn=False, as_dict=False):
    """
    Compute the pointwise PhiID using the IG measure for a Gaussian system.

    Parameters:
        cov (numpy.ndarray): Covariance matrix of the system.
        min_cov (numpy.ndarray): Minimum covariance matrix.
        data (numpy.ndarray): Data samples of shape (d, N).
        nS1 (int): Number of variables in the first source.
        nS2 (int): Number of variables in the second source.
        nT (int): Number of variables in the target.
        as_dict (bool): Whether to return results as a dictionary.
    Returns:
        np.array or dict: containing the pointwise PhiID values for each atom.
    """
    log_qs = log_likelihood(data, min_cov)
    log_ps = log_likelihood(data, cov)
    pt_syn = (log_ps - log_qs) / np.log(2)

    if only_syn:
        return pt_syn
    
    else:
        assert cov.shape[0] == nS1 + nS2 + nT, "Covariance matrix size does not match specified dimensions."
        S1 = list(range(nS1))
        S2 = list(range(nS1, nS1 + nS2))
        T = list(range(nS1 + nS2, nS1 + nS2 + nT))

        # calculate pointwise mutual information
        pmi0, _ = mutual_information(cov, S1, T, pointwise=True, data=data)
        pmi1, _ = mutual_information(cov, S2, T, pointwise=True, data=data)
        pmi01, _ = mutual_information(cov, S1 + S2, T, pointwise=True, data=data)

        pt_red = pt_syn + pmi0 + pmi1 - pmi01
        pt_unq0 = pmi0 - pt_red
        pt_unq1 = pmi1 - pt_red

        if as_dict:
            return {
                "red": pt_red,
                "unq0": pt_unq0,
                "unq1": pt_unq1,
                "syn": pt_syn,
            }
        else:
            return np.stack([pt_red, pt_unq0, pt_unq1, pt_syn], axis=0)
    

def pointwise_phiid_IG(cov, min_cov, data):
    """
    Compute the pointwise PhiID using the IG measure for a Gaussian system.

    Parameters:
        cov (numpy.ndarray): Covariance matrix of the system.
        min_cov (numpy.ndarray): Minimum covariance matrix.
        data (numpy.ndarray): Data samples of shape (d, N).
    Returns:
        np.array: containing the pointwise PhiID sts atom.
    """

    log_qs = log_likelihood(data, min_cov)
    log_ps = log_likelihood(data, cov)
    pt_syn = (log_ps - log_qs) / np.log(2)
    
    return pt_syn