import numpy as np
from scipy.optimize import minimize
from discrete_utils import compute_all_mi_terms, coalesc_distr
from get_lattice import get_ig_lattice


def ig_pid(d):
    """
    Compute the Information Geometric PID for two sources and one target.
    Implementation from https://github.com/dit/dit/ (see dit.pid.measures.iig.py)

    Parameters
    ----------
    d : np.ndarray
        The joint distribution P(X1, X2, Y).

    Returns
    -------
    float
        The Information Geometric Synergy value.
    """
    d += 1e-12  # avoid numerical zeros

    p_s0s1 = d.sum(axis=2, keepdims=True)
    p_s0 = d.sum(axis=(1, 2), keepdims=True)
    p_s1 = d.sum(axis=(0, 2), keepdims=True)

    p_t_s0 = d.sum(axis=1, keepdims=True) / p_s0
    p_t_s1 = d.sum(axis=0, keepdims=True) / p_s1

    def p_star(t):
        d = p_s0s1 * p_t_s0**t * p_t_s1 ** (1 - t)
        d /= d.sum()
        return d

    def objective(t):
        dkl = (d * np.log2(d / p_star(t))).sum().item()
        return dkl

    res = minimize(
        fun=objective,
        x0=0.5,
        method="L-BFGS-B",
        options={
            "maxiter": 1000,
            "ftol": 1e-6,
            "eps": 1.4901161193847656e-08,
        },
    )

    if not res.success:
        msg = f"Optimization failed: {res.message}"
        raise ValueError(msg)

    return objective(res.x)


def ig_synergy_4way(dist):
    """
    Compute the minimum D_KL(P || P*(t,s)) where P*(t,s) interpolates
    exponentially between four reference distributions:
    P11, P12, P21, P22, built from the input joint distribution P(x1,x2,y1,y2).

    Parameters
    ----------
    dist : np.ndarray
        4D normalized array P[x1, x2, y1, y2].
    fuzz : float
        Small value added to avoid numerical zeros.

    Returns
    -------
    min_dkl : float
        Minimum KL divergence.
    opt_t, opt_s : float
        Optimal interpolation parameters.
    """
    dist += 1e-12  # avoid numerical zeros

    # Copy and normalize input
    P = dist.copy().astype(float)
    assert P.ndim == 4, "Input distribution must be 4-dimensional."
    assert np.isclose(P.sum(), 1.0), "Input distribution must be normalized."

    # === Construct marginals ===
    # Axes: 0=X1, 1=X2, 2=Y1, 3=Y2

    Px2y1y2 = P.sum(axis=0, keepdims=True)
    Px1y1y2 = P.sum(axis=1, keepdims=True)
    Px1x2y2 = P.sum(axis=2, keepdims=True)
    Px1x2y1 = P.sum(axis=3, keepdims=True)

    Px1y1 = P.sum(axis=(1, 3), keepdims=True)
    Px1y2 = P.sum(axis=(1, 2), keepdims=True)
    Px2y1 = P.sum(axis=(0, 3), keepdims=True)
    Px2y2 = P.sum(axis=(0, 2), keepdims=True)

    # assert np.isclose(Px2y1y2.sum(), 1.0)
    # assert np.isclose(Px1y1y2.sum(), 1.0)
    # assert np.isclose(Px1x2y2.sum(), 1.0)
    # assert np.isclose(Px1x2y1.sum(), 1.0)

    # === Build 4 corner distributions ===
    P11 = Px1x2y1 * Px1y1y2 / Px1y1
    P12 = Px1x2y2 * Px1y1y2 / Px1y2
    P21 = Px1x2y1 * Px2y1y2 / Px2y1
    P22 = Px1x2y2 * Px2y1y2 / Px2y2

    def p_star(t, s, r):
        # 3D exponential interpolation
        p = P22 ** (1 - t - s - r) * P12**t * P21**s * P11**r
        p /= p.sum()
        return p

    def objective(params):
        t, s, r = params
        p_interp = p_star(t, s, r)
        return np.sum(P * np.log2((P + 1e-12) / (p_interp + 1e-12)))

    res = minimize(
        fun=objective,
        x0=np.random.rand(3),
        # bounds=[(0, 1), (0, 1), (0, 1)],
        method="L-BFGS-B",
        options={"maxiter": 2000, "ftol": 1e-10},
    )

    if not res.success:
        raise RuntimeError(f"Optimization failed: {res.message}")

    return objective(res.x)


def ig_phiid_discrete(prob, as_dict=True, verbose=False):
    """
    Compute the PhiID atoms using IG-PID for a discrete joint distribution
    P[X1, X2, Y1, Y2].

    Parameters
    ----------
    prob : np.ndarray
        4D normalized array P[x1, x2, y1, y2].
    as_dict : bool
        Whether to return the result as a dictionary (True) or np.array (False).
    verbose : bool
        Whether to print the resulting PhiID atoms.
    Returns
    -------
    dict or np.ndarray
        Dictionary or array containing the PhiID atoms.
    """

    # mutual information
    phiid = compute_all_mi_terms(prob)

    p_x1x2y1 = prob.sum(axis=3)
    p_x1x2y2 = prob.sum(axis=2)
    p_x1y1y2 = prob.sum(axis=1)
    p_x2y1y2 = prob.sum(axis=0)

    # PID on subsystems
    I_stx_str = ig_pid(p_x1x2y1)
    I_sty_str = ig_pid(p_x1x2y2)
    I_xts_rts = ig_pid(np.transpose(p_x1y1y2, (1, 2, 0)))
    I_yts_rts = ig_pid(np.transpose(p_x2y1y2, (1, 2, 0)))

    # PID on joint
    prob_coal = coalesc_distr(prob, [[0], [1], [2, 3]])
    I_str_stx_sty_sts = ig_pid(prob_coal)
    prob_coal = coalesc_distr(prob, [[2], [3], [0, 1]])
    I_rts_xts_yts_sts = ig_pid(prob_coal)

    # 4-way synergy
    I_sts = ig_synergy_4way(prob)

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
