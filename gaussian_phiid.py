import numpy as np
from get_lattice import get_ig_lattice
from gaussian_utils import compute_all_mi_terms, get_projected_cov, KL_gaussian_inv, demean
from scipy.optimize import minimize

from scipy.linalg import cholesky, solve, block_diag
from scipy.optimize import root_scalar, minimize


def PID_IG(Sigma, S1, S2, T, pointwise=False, data=None, only_syn=False):
    """
    Compute PID IG for Gaussian data. 

    Parameters:
        Sigma (numpy.ndarray): Covariance matrix of the system.
        S1 (int): Dimension of the first source variable(s).
        S2 (int): Dimension of the second source variable(s).
        T (int): Dimension of the target variable(s).
        pointwise (bool): Whether to compute pointwise PID (default: False).
        data (numpy.ndarray): Data samples of shape (d, N) for pointwise calculation (required if pointwise=True).
        only_syn (bool): Whether to compute only the synergistic component for pointwise PID (default: False).
    Returns:
        np.array or dict: containing the PID values for each atom in BITS. 
                            If pointwise=False, returns a dict with keys 'Red', 'UnX', 'UnY', 'Syn'. 
                            If pointwise=True, returns a dict with the same keys but each value is an array of shape (N,).
    """
    assert Sigma.shape == ((S1 + S2 + T),(S1 + S2 + T)), "Covariance matrix has incorrect shape."
    # assert np.linalg.eigvals(Sigma).min() > 1e-8, "Covariance matrix is not positive definite."

    # Convert to correlation matrix
    d = np.diag(1 / np.sqrt(np.diag(Sigma)))
    Sigma_corr = d @ Sigma @ d

    n0, n1, n2 = S1, S2, T
    ind0 = np.arange(n0)
    ind1 = np.arange(n0, n0 + n1)
    ind2 = np.arange(n0 + n1, n0 + n1 + n2)

    I0 = np.eye(n0)
    I1 = np.eye(n1)
    I2 = np.eye(n2)

    S00 = Sigma_corr[np.ix_(ind0, ind0)]
    S01 = Sigma_corr[np.ix_(ind0, ind1)]
    S02 = Sigma_corr[np.ix_(ind0, ind2)]
    S11 = Sigma_corr[np.ix_(ind1, ind1)]
    S12 = Sigma_corr[np.ix_(ind1, ind2)]
    S22 = Sigma_corr[np.ix_(ind2, ind2)]

    InvSq00 = solve(cholesky(S00, lower=False), I0)
    InvSq11 = solve(cholesky(S11, lower=False), I1)
    InvSq22 = solve(cholesky(S22, lower=False), I2)

    P = InvSq00.T @ S01 @ InvSq11
    Q = InvSq00.T @ S02 @ InvSq22
    R = InvSq11.T @ S12 @ InvSq22

    P1 = P.T
    Q1 = Q.T
    R1 = R.T

    dP = np.linalg.det(I1 - P1 @ P)
    dQ = np.linalg.det(I2 - Q1 @ Q)
    dR = np.linalg.det(I2 - R1 @ R)
    dQR = np.linalg.det(I1 - R @ Q1 @ Q @ R1)

    # Full covariance matrix
    r1 = np.hstack([I0, P, Q])
    r2 = np.hstack([P1, I1, R])
    r3 = np.hstack([Q1, R1, I2])
    Sig = np.vstack([r1, r2, r3])

    # Check positive definiteness
    ev = np.linalg.eigvals(Sig)
    if not np.all(ev > 0):
        raise ValueError("Covariance matrix is not positive definite")

    PD = "yes"

    # Compose Sig5
    R5 = P.T @ Q
    r51 = np.hstack([I0, P, Q])
    r52 = np.hstack([P1, I1, R5])
    r53 = np.hstack([Q1, R5.T, I2])
    Sig5 = np.vstack([r51, r52, r53])

    # Compose Sig6
    Q6 = P @ R
    r61 = np.hstack([I0, P, Q6])
    r62 = np.hstack([P1, I1, R])
    r63 = np.hstack([Q6.T, R1, I2])
    Sig6 = np.vstack([r61, r62, r63])

    sig5_inv = np.linalg.inv(Sig5)
    sig6_inv = np.linalg.inv(Sig6)

    # Feasibility function for t
    def feas_test(t):
        m = (1 - t) * sig5_inv + t * sig6_inv
        return 2 * int(np.all(np.linalg.eigvals(m) > 0)) - 1

    extremes = [1e2, 1e3, 1e4, 1e5]
    root_hi = None
    root_lo = None
    for ext in extremes:
        try:
            root_hi = root_scalar(feas_test, bracket=[1, ext]).root
            root_lo = root_scalar(feas_test, bracket=[-ext, 0]).root
            break
        except ValueError:
            continue
    if root_hi is None or root_lo is None:
        raise ValueError("No feasible t found")

    feas = np.array([root_lo, root_hi])

    # KL divergence function
    def KLdiv(t):
        mm = np.linalg.inv((1 - t) * sig5_inv + t * sig6_inv)
        return 0.5 * (
            np.sum(np.log(np.linalg.eigvals(mm)))
            - np.sum(np.log(np.linalg.eigvals(Sig)))
        )

    # Initial t for optimization
    x0 = (root_lo + root_hi) / 2
    res = minimize(
        lambda t: KLdiv(t), x0, bounds=[(root_lo, root_hi)], method="L-BFGS-B"
    )

    tstar = res.x[0]
    syn = res.fun

    # minimising distribution
    distr = np.linalg.inv((1 - tstar) * sig5_inv + tstar * sig6_inv)
    # remap it to original space
    W = block_diag(InvSq00.T, InvSq11.T, InvSq22.T)
    TT = W @ d
    Tinv = np.linalg.inv(TT)
    distr = Tinv @ distr @ Tinv.T
    distr_og = Tinv @ Sig @ Tinv.T

    # Mutual informations
    i13 = 0.5 * np.log(1 / dQ)
    i23 = 0.5 * np.log(1 / dR)
    i13G2 = 0.5 * np.log(dP * dR / np.linalg.det(Sig))
    i23G1 = 0.5 * np.log(dP * dQ / np.linalg.det(Sig))
    jmi = 0.5 * np.log(dP / np.linalg.det(Sig))
    ii = jmi - i13 - i23

    inf = np.array([i13, i23, i13G2, i23G1, jmi, ii]) / np.log(2)

    unq1 = i13G2 - syn
    unq2 = i23G1 - syn
    red = i13 - unq1
    pid = np.array([red, unq1, unq2, syn]) / np.log(2)

    # NB: do not retrieve the atoms, which are still in NATS.
    out_dict = {"PD": PD, "feas": feas, "t_star": tstar, "distr": distr, "distr_original": distr_og, "inf": inf, "pid": pid}
    if pointwise:
        if data is None:
            raise ValueError("Data must be provided for pointwise PID calculation.")
        assert data.shape[0] == Sigma.shape[0], "Data dimensionality does not match covariance matrix."
        from gaussian_utils import pointwise_pid_IG
        pt_pid = pointwise_pid_IG(Sigma, distr, data, S1, S2, T, only_syn=only_syn)
        out_dict["pointwise_pid"] = pt_pid

    return out_dict


def get_sts_IG(cov, n1=1, n2=1, pointwise=False, data=None):

    cov = cov / np.sqrt(np.diag(cov)[:, None] * np.diag(cov)[None, :])

    cov11, cov12, cov21, cov22 = get_projected_cov(cov, n1, n2)
    inv_cov11 = np.linalg.inv(cov11)
    inv_cov12 = np.linalg.inv(cov12)
    inv_cov21 = np.linalg.inv(cov21)
    inv_cov22 = np.linalg.inv(cov22)

    def KLdiv(params):
        t, s, r = params
        m = (1 - t - s - r) * inv_cov11 + t * inv_cov12 + s * inv_cov21 + r * inv_cov22
        try:
            np.linalg.cholesky(m)
        except np.linalg.LinAlgError:
            eigs = np.linalg.eigvals(m)
            neg_sum = np.sum(np.abs(eigs[eigs <= 0]))
            return neg_sum * 1e6

        return KL_gaussian_inv(cov, m)

    bounds = None
    x0 = (0.3, 0.3, 0.3)

    res = minimize(
        KLdiv,
        x0=x0,
        bounds=bounds,
        method="L-BFGS-B",
        options={"ftol": 1e-5, "maxiter": 1000},
    )

    if not res.success:
        raise RuntimeError(f"KL minimization failed: {res.message}")

    syn = res.fun / np.log(2)
    if pointwise:
        if data is None:
            raise ValueError("Data must be provided for pointwise STS calculation.")
        distr = np.linalg.inv((1 - res.x[0] - res.x[1] - res.x[2]) * inv_cov11 + res.x[0] * inv_cov12 + res.x[1] * inv_cov21 + res.x[2] * inv_cov22)
        assert data.shape[0] == cov.shape[0], "Data dimensionality does not match covariance matrix."
        from gaussian_utils import pointwise_phiid_IG
        pt_syn = pointwise_phiid_IG(cov, distr, data)

        return pt_syn, syn
    else:
        return [syn]


def ig_phiid_gaussian(cov, n1=1, n2=1, as_dict=True, verbose=False, pointwise=False, data=None):
    if pointwise:
        if data is None:
            raise ValueError("Data must be provided for pointwise calculation.")
        assert data.shape[0] == cov.shape[0], "Data dimensionality does not match covariance matrix."
        # since the covariance will become a correlation matrix, we need to standardize the data
        data = demean(data)

    assert cov.shape == (
        2 * (n1 + n2),
        2 * (n1 + n2),
    ), "Covariance matrix has incorrect shape."
    assert (
        np.linalg.eigvals(cov).min() > 1e-8
    ), "Covariance matrix is not positive definite."

    x = list(range(n1))
    y = list(range(n1, n1 + n2))
    a = list(range(n1 + n2, n1 + n2 + n1))
    b = list(range(n1 + n2 + n1, 2 * (n1 + n2)))

    # make correlation matrix
    cov = cov / np.sqrt(np.diag(cov)[:, None] * np.diag(cov)[None, :])

    phiid = compute_all_mi_terms(cov, n1, n2, n1, n2, pointwise=pointwise, data=data)

    def get_PID(cov, S1, S2, T, pointwise=False, data=None):
        if pointwise:
            pid = PID_IG(cov, S1, S2, T, pointwise=True, data=data, only_syn=True)
            return pid["pointwise_pid"], pid["pid"][3]
        else:
            return [PID_IG(cov, S1, S2, T)["pid"][3]]

    # for phiid, do the 6 pids:
    # (X1,X2,Y1)
    I_stx_str = get_PID(cov[np.ix_(x + y + a, x + y + a)], n1, n2, n1, pointwise=pointwise, data=data[x+y+a, :] if data is not None else None)
    # (X1,X2,Y2)
    I_sty_str = get_PID(cov[np.ix_(x + y + b, x + y + b)], n1, n2, n2, pointwise=pointwise, data=data[x+y+b, :] if data is not None else None)
    # (X1,Y1,Y2)
    I_xts_rts = get_PID(cov[np.ix_(a + b + x, a + b + x)], n1, n2, n1, pointwise=pointwise, data=data[a+b+x, :] if data is not None else None)
    # (X2,Y1,Y2)
    I_yts_rts = get_PID(cov[np.ix_(a + b + y, a + b + y)], n1, n2, n2, pointwise=pointwise, data=data[a+b+y, :] if data is not None else None)
    # (X1,X2,(Y1,Y2))
    I_str_stx_sty_sts = get_PID(cov, n1, n2, n1+n2, pointwise=pointwise, data=data if data is not None else None)
    # ((X1,X2),Y1,Y2)
    I_rts_xts_yts_sts = get_PID(cov[np.ix_(a + b + x + y, a + b + x + y)], n1, n2, n1+n2, pointwise=pointwise, data=data[a+b+x+y, :] if data is not None else None)

    I_sts = get_sts_IG(cov, n1, n2, pointwise=pointwise, data=data)

    for n,d in enumerate(phiid):
        d.update(dict(
            I_stx_str=I_stx_str[n],
            I_sty_str=I_sty_str[n],
            I_xts_rts=I_xts_rts[n],
            I_yts_rts=I_yts_rts[n],
            I_str_stx_sty_sts=I_str_stx_sty_sts[n],
            I_rts_xts_yts_sts=I_rts_xts_yts_sts[n],
            I_sts=I_sts[n],
        ))
    # then construct the lattice and solve for the atoms
    phiid = [get_ig_lattice(d, verbose=verbose) for d in phiid]

    if as_dict:
        if pointwise:
            return phiid    
        else:
            return phiid[0]
    else:
        # convert list of dicts to array of values
        return np.array([list(d.values()) for d in phiid]).squeeze()
    