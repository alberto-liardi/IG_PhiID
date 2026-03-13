import numpy as np
from scipy.optimize import minimize
from discrete_utils import compute_all_mi_terms, coalesc_distr, log_likelihood_discrete, coalesc_data
from get_lattice import get_ig_lattice


def pointwise_pid_IG(dist, min_dist, data, nS1=1, nS2=1, nT=1, only_syn=False):
    """
    Compute the pointwise PID using the IG measure for a discrete system.

    Parameters
    ----------
    dist : np.ndarray
        Joint probability distribution.
    min_dist : np.ndarray
        Minimum distribution from optimization.
    data : np.ndarray
        Data samples of shape (n_vars, n_samples).
    nS1 : int
        Number of variables in the first source.
    nS2 : int
        Number of variables in the second source.
    nT : int
        Number of variables in the target.
    only_syn : bool
        Whether to compute only the synergistic component.

    Returns
    -------
    np.ndarray or dict
        Pointwise PID values for each atom.
    """
    log_qs = log_likelihood_discrete(data, min_dist)
    log_ps = log_likelihood_discrete(data, dist)
    pt_syn = log_ps - log_qs

    if only_syn:
        return pt_syn
    else:
        from discrete_utils import pointwise_mutual_information
        
        S1 = list(range(nS1))
        S2 = list(range(nS1, nS1 + nS2))
        T = list(range(nS1 + nS2, nS1 + nS2 + nT))

        # Calculate pointwise mutual information
        pmi0, _ = pointwise_mutual_information(dist, data, S1, T)
        pmi1, _ = pointwise_mutual_information(dist, data, S2, T)
        pmi01, _ = pointwise_mutual_information(dist, data, S1 + S2, T)

        pt_red = pt_syn + pmi0 + pmi1 - pmi01
        pt_unq0 = pmi0 - pt_red
        pt_unq1 = pmi1 - pt_red

        return np.stack([pt_red, pt_unq0, pt_unq1, pt_syn], axis=0)


def pointwise_phiid_IG(dist, min_dist, data):
    """
    Compute the pointwise synergy using the IG measure for a discrete system.

    Parameters
    ----------
    dist : np.ndarray
        Joint probability distribution.
    min_dist : np.ndarray
        Minimum distribution from optimization.
    data : np.ndarray
        Data samples of shape (n_vars, n_samples).

    Returns
    -------
    np.ndarray
        Pointwise synergy values.
    """
    log_qs = log_likelihood_discrete(data, min_dist)
    log_ps = log_likelihood_discrete(data, dist)
    pt_syn = log_ps - log_qs
    
    return pt_syn


def ig_pid(d, pointwise=False, data=None, only_syn=False):
    """
    Compute the Information Geometric PID for two sources and one target.
    Implementation from https://github.com/dit/dit/ (see dit.pid.measures.iig.py)

    Parameters
    ----------
    d : np.ndarray
        The joint distribution P(X1, X2, Y).
    pointwise : bool
        Whether to compute pointwise PID.
    data : np.ndarray
        Data samples for pointwise calculation.
    only_syn : bool
        Whether to compute only synergy for pointwise PID.

    Returns
    -------
    dict or float
        Dictionary with 'pid' and optionally 'pointwise_pid', or just synergy value.
    """
    d = d.copy()
    d += 1e-12  # avoid numerical zeros
    d /= d.sum()  # renormalize

    p_s0s1 = d.sum(axis=2, keepdims=True)
    p_s0 = d.sum(axis=(1, 2), keepdims=True)
    p_s1 = d.sum(axis=(0, 2), keepdims=True)

    p_t_s0 = d.sum(axis=1, keepdims=True) / p_s0
    p_t_s1 = d.sum(axis=0, keepdims=True) / p_s1

    def p_star(t):
        dist = p_s0s1 * p_t_s0**t * p_t_s1 ** (1 - t)
        dist /= dist.sum()
        return dist

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

    syn = objective(res.x)
    min_dist = p_star(res.x)

    if not pointwise:
        return syn
    else:
        if data is None:
            raise ValueError("Data must be provided for pointwise PID calculation.")
        
        # Infer dimensions from distribution shape
        nS1 = d.shape[0]
        nS2 = d.shape[1]
        nT = d.shape[2]
        
        pt_pid = pointwise_pid_IG(d, min_dist, data, nS1, nS2, nT, only_syn=only_syn)
        
        if only_syn:
            return pt_pid, syn
        else:
            return {"pointwise_pid": pt_pid, "pid": syn}


def ig_synergy_4way(dist, pointwise=False, data=None):
    """
    Compute the minimum D_KL(P || P*(t,s,r)) where P*(t,s,r) interpolates
    exponentially between four reference distributions.

    Parameters
    ----------
    dist : np.ndarray
        Joint distribution P[x1, x2, y1, y2] (can be multivariate).
    pointwise : bool
        Whether to compute pointwise synergy.
    data : np.ndarray
        Data samples for pointwise calculation.

    Returns
    -------
    float or tuple
        Minimum KL divergence (synergy).
        If pointwise=True, returns (pt_syn, syn).
    """
    dist = dist.copy()
    dist += 1e-12  # avoid numerical zeros
    dist /= dist.sum()  # renormalize

    P = dist.astype(float)
    
    # Determine number of axes
    n_axes = P.ndim
    assert n_axes == 4, "Distribution must have 4 groups of variables (can be multivariate)"

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

    # === Build 4 corner distributions ===
    P11 = Px1x2y1 * Px1y1y2 / (Px1y1 + 1e-12)
    P12 = Px1x2y2 * Px1y1y2 / (Px1y2 + 1e-12)
    P21 = Px1x2y1 * Px2y1y2 / (Px2y1 + 1e-12)
    P22 = Px1x2y2 * Px2y1y2 / (Px2y2 + 1e-12)

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
        method="L-BFGS-B",
        options={"maxiter": 2000, "ftol": 1e-10},
    )

    if not res.success:
        raise RuntimeError(f"Optimization failed: {res.message}")

    syn = objective(res.x)
    min_dist = p_star(*res.x)

    if not pointwise:
        return [syn]
    else:
        if data is None:
            raise ValueError("Data must be provided for pointwise STS calculation.")
        
        pt_syn = pointwise_phiid_IG(P, min_dist, data)
        return pt_syn, syn


def ig_phiid_discrete(prob, as_dict=True, verbose=False, pointwise=False, data=None):
    """
    Compute the PhiID atoms using IG-PID for a discrete joint distribution.

    Parameters
    ----------
    prob : np.ndarray
        Joint distribution. Shape depends on n1, n2 and cardinality of each variable.
    as_dict : bool
        Whether to return the result as a dictionary (True) or np.array (False).
    verbose : bool
        Whether to print the resulting PhiID atoms.
    pointwise : bool
        Whether to compute pointwise PhiID.
    data : np.ndarray
        Data samples of shape (n_vars, n_samples) for pointwise calculation.

    Returns
    -------
    dict or np.ndarray
        Dictionary or array containing the PhiID atoms.
        If pointwise=True, returns list of dicts (one per sample).
    """
    if pointwise:
        if data is None:
            raise ValueError("Data must be provided for pointwise calculation.")
        assert data.shape[0] == 4, "Data dimensionality does not match distribution."
    assert prob.ndim == 4, f"Distribution must have 4 axes. Got {prob.ndim}."

    # Compute mutual information terms
    phiid = compute_all_mi_terms(prob, pointwise=pointwise, data=data)

    def get_PID(dist, pointwise=False, data_slice=None):
        
        if pointwise:
            return ig_pid(dist, pointwise=True, data=data_slice, only_syn=True)
        else:
            return [ig_pid(dist, pointwise=False)]

    p_x1x2y1 = prob.sum(axis=3)
    p_x1x2y2 = prob.sum(axis=2)
    p_x1y1y2 = prob.sum(axis=1)
    p_x2y1y2 = prob.sum(axis=0)

        # Compute PIDs on subsystems
    # (X1, X2, Y1)
    I_stx_str = get_PID(p_x1x2y1, pointwise=pointwise, 
                        data_slice=data[[0,1,2], :] if data is not None else None)
    
    # (X1, X2, Y2)
    I_sty_str = get_PID(p_x1x2y2, pointwise=pointwise,
                        data_slice=data[[0,1,3], :] if data is not None else None)
    
    # (Y1, Y2, X1) - note the reordering
    I_xts_rts = get_PID(np.transpose(p_x1y1y2, (1, 2, 0)), pointwise=pointwise,
                        data_slice=data[[2,3,0], :] if data is not None else None)
    
    # (Y1, Y2, X2)
    I_yts_rts = get_PID(np.transpose(p_x2y1y2, (1, 2, 0)), pointwise=pointwise,
                        data_slice=data[[2,3,1], :] if data is not None else None)
    
    # (X1, X2, (Y1,Y2))
    prob_coal = coalesc_distr(prob, [[0], [1], [2, 3]])
    data_coal = coalesc_data(data, [[0], [1], [2, 3]], [prob.shape[0], prob.shape[1], prob.shape[2], prob.shape[3]]) if data is not None else None
    I_str_stx_sty_sts = get_PID(prob_coal, pointwise=pointwise,
                                 data_slice=data_coal)
    
    # ((Y1,Y2), X1, X2) - reordered as (Y1, Y2, X1, X2)
    prob_coal = coalesc_distr(prob, [[2], [3], [0, 1]])
    data_coal = coalesc_data(data, [[2], [3], [0, 1]], [prob.shape[2], prob.shape[3], prob.shape[0], prob.shape[1]]) if data is not None else None
    I_rts_xts_yts_sts = get_PID(prob_coal, pointwise=pointwise,
                                 data_slice=data_coal)


    # 4-way synergy
    I_sts = ig_synergy_4way(prob, pointwise=pointwise, data=data)

    # Update phiid dictionaries
    for n, d in enumerate(phiid):
        d.update(dict(
            I_stx_str=I_stx_str[n],
            I_sty_str=I_sty_str[n],
            I_xts_rts=I_xts_rts[n],
            I_yts_rts=I_yts_rts[n],
            I_str_stx_sty_sts=I_str_stx_sty_sts[n],
            I_rts_xts_yts_sts=I_rts_xts_yts_sts[n],
            I_sts=I_sts[n],
        ))

    # Construct the lattice and solve for the atoms
    phiid = [get_ig_lattice(d, verbose=verbose) for d in phiid]

    if as_dict:
        if pointwise:
            return phiid
        else:
            return phiid[0]
    else:
        # Convert list of dicts to array of values
        return np.array([list(d.values()) for d in phiid]).squeeze()