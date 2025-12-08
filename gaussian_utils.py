import numpy as np


def mutual_information(S, s, t):
    """
    Given a joint covariance matrix S, calculate the mutual information
    between variables indexed by the lists s and t.
    """
    Hx = np.linalg.slogdet(S[np.ix_(s, s)])[1]
    Hy = np.linalg.slogdet(S[np.ix_(t, t)])[1]
    Hxy = np.linalg.slogdet(S[np.ix_(s + t, s + t)])[1]
    return 0.5 * (Hx + Hy - Hxy) / np.log(2)


def compute_all_mi_terms(cov):

    Ixa = mutual_information(cov, [0], [2])
    Ixb = mutual_information(cov, [0], [3])
    Iya = mutual_information(cov, [1], [2])
    Iyb = mutual_information(cov, [1], [3])
    Ixya = mutual_information(cov, [0, 1], [2])
    Ixyb = mutual_information(cov, [0, 1], [3])
    Ixab = mutual_information(cov, [0], [2, 3])
    Iyab = mutual_information(cov, [1], [2, 3])
    Ixyab = mutual_information(cov, [0, 1], [2, 3])

    return dict(
        Ixa=Ixa,
        Ixb=Ixb,
        Iya=Iya,
        Iyb=Iyb,
        Ixya=Ixya,
        Ixyb=Ixyb,
        Ixab=Ixab,
        Iyab=Iyab,
        Ixyab=Ixyab,
    )


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
