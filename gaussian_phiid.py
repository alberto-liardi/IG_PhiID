import numpy as np
from get_lattice import get_ig_lattice
from gaussian_utils import compute_all_mi_terms, get_projected_cov, KL_gaussian_inv
from scipy.optimize import minimize

from scipy.linalg import cholesky, solve
from scipy.optimize import root_scalar, minimize


def PID_IG(Sigma, S1, S2, T):

    # Convert to correlation matrix
    d = np.diag(1 / np.sqrt(np.diag(Sigma)))
    Sigma = d @ Sigma @ d

    n0, n1, n2 = S1, S2, T
    ind0 = np.arange(n0)
    ind1 = np.arange(n0, n0 + n1)
    ind2 = np.arange(n0 + n1, n0 + n1 + n2)

    I0 = np.eye(n0)
    I1 = np.eye(n1)
    I2 = np.eye(n2)

    S00 = Sigma[np.ix_(ind0, ind0)]
    S01 = Sigma[np.ix_(ind0, ind1)]
    S02 = Sigma[np.ix_(ind0, ind2)]
    S11 = Sigma[np.ix_(ind1, ind1)]
    S12 = Sigma[np.ix_(ind1, ind2)]
    S22 = Sigma[np.ix_(ind2, ind2)]

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

    return {"PD": PD, "feas": feas, "t_star": tstar, "inf": inf, "pid": pid}


def get_sts_IG(cov, n1=1, n2=1):

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

    syn = res.fun

    return syn


def ig_phiid_gaussian(cov, n1=1, n2=1, as_dict=True, verbose=False):

    assert cov.shape == (
        2 * (n1 + n2),
        2 * (n1 + n2),
    ), "Covariance matrix has incorrect shape."
    assert (
        np.linalg.eigvals(cov).min() > 1e-8
    ), "Covariance matrix is not positive definite."

    # make correlation matrix
    cov = cov / np.sqrt(np.diag(cov)[:, None] * np.diag(cov)[None, :])

    phiid = compute_all_mi_terms(cov)

    # for phiid, do the 6 pids:
    # (X1,X2,Y1)
    I_stx_str = PID_IG(cov[np.ix_([0, 1, 2], [0, 1, 2])], 1, 1, 1)["pid"][3]
    # (X1,X2,Y2)
    I_sty_str = PID_IG(cov[np.ix_([0, 1, 3], [0, 1, 3])], 1, 1, 1)["pid"][3]
    # (X1,Y1,Y2)
    I_xts_rts = PID_IG(cov[np.ix_([2, 3, 0], [2, 3, 0])], 1, 1, 1)["pid"][3]
    # (X2,Y1,Y2)
    I_yts_rts = PID_IG(cov[np.ix_([2, 3, 1], [2, 3, 1])], 1, 1, 1)["pid"][3]
    # (X1,X2,(Y1,Y2))
    I_str_stx_sty_sts = PID_IG(cov, 1, 1, 2)["pid"][3]
    # ((X1,X2),Y1,Y2)
    I_rts_xts_yts_sts = PID_IG(cov[np.ix_([2, 3, 0, 1], [2, 3, 0, 1])], 1, 1, 2)["pid"][
        3
    ]

    I_sts = get_sts_IG(cov, n1, n2) / np.log(2)

    phiid.update(
        dict(
            I_stx_str=I_stx_str,
            I_sty_str=I_sty_str,
            I_xts_rts=I_xts_rts,
            I_yts_rts=I_yts_rts,
            I_str_stx_sty_sts=I_str_stx_sty_sts,
            I_rts_xts_yts_sts=I_rts_xts_yts_sts,
            I_sts=I_sts,
        )
    )

    phiid = get_ig_lattice(phiid, verbose=verbose)

    if as_dict:
        return phiid
    else:
        return np.array(list(phiid.values()))
