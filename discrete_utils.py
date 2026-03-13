import numpy as np
import itertools

def pointwise_mutual_information(dist, data, src, tgt):
    """
    Vectorized pointwise mutual information per sample.

    PMI(s, t) = log_base p(s,t) - log_base p(s) - log_base p(t)
    with base=2 returning bits.

    Parameters
    ----------
    dist : np.ndarray
        The full joint probability distribution (should be normalized).
    data : np.ndarray
        Array of shape (n_vars, n_samples) with observed outcomes.
    src : list[int]
        Indices of source variables (axes of dist). Must be non-empty and disjoint from tgt.
    tgt : list[int]
        Indices of target variables (axes of dist). Must be non-empty and disjoint from src.
    base : {2, 'e', 10, float}, default 2
        Logarithm base for PMI.
    zero_policy : {'zero', 'allow-inf'}, default 'zero'
        - 'zero': set PMI to 0 where any of p(s), p(t), p(s,t)=0 (matches your loop behavior).
        - 'allow-inf': keep -inf/+inf as produced by logs.
    check : bool, default True
        Validate inputs and normalize dist if needed (tolerant normalization).

    Returns
    -------
    pmi : np.ndarray of shape (n_samples,)
        Pointwise mutual information for each sample (in chosen log base).
    mi : float
        Average PMI across samples.
    """
    d = np.asarray(dist, dtype=float)

    src = list(src)
    tgt = list(tgt)
    if len(src) == 0 or len(tgt) == 0:
        raise ValueError("src and tgt must both be non-empty.")
    if set(src) & set(tgt):
        raise ValueError("src and tgt must be disjoint.")

    # Build marginals; axes of the result are in sorted(keep_axes) order
    def marginalize(dist_full, keep_axes):
        sum_axes = tuple(ax for ax in range(dist_full.ndim) if ax not in keep_axes)
        marg = dist_full.sum(axis=sum_axes, keepdims=False)
        order_sorted = sorted(keep_axes)
        return marg, order_sorted

    p_src, src_sorted = marginalize(d, src)
    p_tgt, tgt_sorted = marginalize(d, tgt)
    p_joint, joint_sorted = marginalize(d, src + tgt)

    # Align data rows to each marginal's axis order
    data_src = data[src_sorted, :]
    data_tgt = data[tgt_sorted, :]
    data_joint = data[joint_sorted, :]

    # Vectorized log-likelihoods
    ll_src = log_likelihood_discrete(data_src, p_src)
    ll_tgt = log_likelihood_discrete(data_tgt, p_tgt)
    ll_joint = log_likelihood_discrete(data_joint, p_joint)

    # PMI per sample
    pmi = ll_joint - ll_src - ll_tgt

    # Where any prob is zero -> corresponding ll is -inf -> set PMI to 0
    finite_mask = np.isfinite(ll_src) & np.isfinite(ll_tgt) & np.isfinite(ll_joint)
    pmi = np.where(finite_mask, pmi, 0.0)

    mi = float(np.mean(pmi))
    return pmi, mi


def mutual_information(dist, src, tgt, pointwise=False, data=None):
    """
    Compute the mutual information I(src ; tgt) from a joint distribution.

    Parameters
    ----------
    dist : np.ndarray
        The full joint probability distribution (must be normalized).
    src : list of int
        Indices of source variables (axes of dist).
    tgt : list of int
        Indices of target variables (axes of dist).
    pointwise : bool
        Whether to compute pointwise mutual information.
    data : np.ndarray
        Data samples of shape (n_vars, n_samples) for pointwise calculation.

    Returns
    -------
    float or tuple
        Mutual information I(src ; tgt) in bits.
        If pointwise=True, returns (pmi, mi) where pmi is array of pointwise values.
    """

    # Ensure array and normalization
    d = np.asarray(dist, dtype=float)
    assert np.isclose(d.sum(), 1.0), "Input distribution must be normalized."

    if not pointwise:
        # Marginals
        p_src = d.sum(axis=tuple(i for i in range(d.ndim) if i not in src), keepdims=False)
        p_tgt = d.sum(axis=tuple(i for i in range(d.ndim) if i not in tgt), keepdims=False)
        p_joint = d.sum(axis=tuple(i for i in range(d.ndim) if i not in (src + tgt)), keepdims=False)

        # Broadcast shapes to match for division
        p_src_exp = np.expand_dims(p_src, tuple(range(len(src), len(src) + len(tgt))))
        p_tgt_exp = np.expand_dims(p_tgt, tuple(range(0, len(src))))

        # Avoid divisions by zero
        with np.errstate(divide='ignore', invalid='ignore'):
            ratio = np.where((p_joint > 0) & (p_src_exp * p_tgt_exp > 0),
                             p_joint / (p_src_exp * p_tgt_exp), 1)
            mi = (p_joint * np.log2(ratio)).sum()

        return mi
    else:
        if data is None:
            raise ValueError("Data must be provided for pointwise mutual information.")
        return pointwise_mutual_information(dist, data, src, tgt)


def compute_all_mi_terms(dist, pointwise=False, data=None):
    """
    Compute all relevant mutual information terms for the distribution
    P[x1, x2, y1, y2] with multivariate support.

    Parameters
    ----------
    dist : np.ndarray
        4D joint distribution.
    pointwise : bool
        Whether to compute pointwise mutual information.
    data : np.ndarray
        Data samples for pointwise calculation.

    Returns
    -------
    list of dict
        List containing dictionary/dictionaries with all mutual information terms.
    """
    
    # Define indices for each group
    S1 = [0]; S2 = [1]; T1 = [2]; T2 = [3]

    def get_mi(idx_src, idx_tgt):
        return mutual_information(dist, idx_src, idx_tgt, pointwise=pointwise, data=data)

    Ixa   = get_mi(S1, T1)
    Ixb   = get_mi(S1, T2)
    Iya   = get_mi(S2, T1)
    Iyb   = get_mi(S2, T2)
    Ixya  = get_mi(S1 + S2, T1)
    Ixyb  = get_mi(S1 + S2, T2)
    Ixab  = get_mi(S1, T1 + T2)
    Iyab  = get_mi(S2, T1 + T2)
    Ixyab = get_mi(S1 + S2, T1 + T2)

    keys = ["Ixa","Ixb","Iya","Iyb","Ixya","Ixyb","Ixab","Iyab","Ixyab"]
    values = [Ixa, Ixb, Iya, Iyb, Ixya, Ixyb, Ixab, Iyab, Ixyab]

    if pointwise:
        return [dict(zip(keys, vals)) for vals in zip(*values)]
    else:
        return [dict(zip(keys, values))]


def coalesc_distr(d, indx):
    """
    Coalesce a discrete probability distribution along given groups of indices.

    Parameters
    ----------
    d : np.ndarray
        The full joint distribution over several variables, e.g. with shape (2,2,2,2)
        for four binary variables (X1,X2,Y1,Y2).
    indx : list of lists
        Each element of indx specifies which original axes to merge into one new variable.
        Example: [[0], [1], [2,3]] merges the last two variables into a single composite one.

    Returns
    -------
    d_coalesced : np.ndarray
        The coalesced distribution, with shape given by the product of the sizes
        of each index group. For example, if d.shape=(2,2,2,2) and indx=[[0],[1],[2,3]],
        the output shape is (2,2,4), since 2*2=4 for the merged last variable.
    """

    d = np.asarray(d, dtype=float)
    d_size = d.shape

    # --- Compute the new shape ---
    coal_size = [int(np.prod([d_size[i] for i in group])) for group in indx]
    d_coalesced = np.zeros(coal_size)

    # --- Iterate through all original outcomes ---
    for outcome in itertools.product(*[range(n) for n in d_size]):
        prob = d[outcome]
        new_coords = []
        for group in indx:
            # Map each group of indices to a single integer index
            subshape = [d_size[i] for i in group]
            flat_index = np.ravel_multi_index([outcome[i] for i in group], subshape)
            new_coords.append(flat_index)
        d_coalesced[tuple(new_coords)] += prob

    return d_coalesced


def log_likelihood_discrete(data, dist):
    """
    Compute the log-likelihood of each sample under a discrete distribution.

    Parameters
    ----------
    data : np.ndarray
        Array of shape (n_vars, n_samples) containing the observed outcomes.
    dist : np.ndarray
        Joint probability distribution.

    Returns
    -------
    ll : np.ndarray
        Array of shape (n_samples,) with the log-likelihood of each sample.
    """
    data = np.asarray(data, dtype=np.int64)
    if data.ndim != 2:
        raise ValueError("data must be 2D: (n_vars, n_samples)")
    n_vars, n_samples = data.shape

    if dist.ndim != n_vars:
        raise ValueError(f"dist.ndim ({dist.ndim}) != n_vars ({n_vars})")
    
    # Advanced indexing: one 1D index array per axis
    idx = tuple(data[v] for v in range(n_vars))  # each has shape (n_samples,)
    probs = dist[idx]  # shape (n_samples,)

    ll = np.full(n_samples, -np.inf, dtype=float)
    mask = probs > 0
    ll[mask] = np.log2(probs[mask])

    return ll


def coalesc_data(data_slice, idx_groups, axis_sizes_in_order):
    """
    Vectorized coalescing of data rows into composite indices, using known axis sizes.

    Parameters
    ----------
    data_slice : np.ndarray of shape (n_axes, n_samples)
        Rows correspond to axes in the SAME order as dist_subset_in_order.
    idx_groups : list[list[int]]
        Groups based on positions (0..n_axes-1) in 'data_slice'.
    axis_sizes_in_order : list[int]
        The cardinality for each row/axis in 'data_slice' order.

    Returns
    -------
    data_coalesced : np.ndarray of shape (len(idx_groups), n_samples)
        Composite indices per group.
    """
    data_slice = np.asarray(data_slice, dtype=np.int64)
    n_axes, n_samples = data_slice.shape
    if len(axis_sizes_in_order) != n_axes:
        raise ValueError("axis_sizes_in_order length must match data_slice rows")

    out = np.zeros((len(idx_groups), n_samples), dtype=np.int64)

    # For each group, compute flat index via mixed radix representation
    for g_idx, g in enumerate(idx_groups):
        if len(g) == 1:
            out[g_idx, :] = data_slice[g[0], :]
        else:
            # radix sizes for axes in this group (in order)
            radices = [axis_sizes_in_order[i] for i in g]
            # strides: product of subsequent radices
            strides = np.array([int(np.prod(radices[j+1:])) for j in range(len(radices))], dtype=np.int64)
            # indices for these axes
            vals = data_slice[g, :]  # shape (len(g), n_samples)
            # flat = sum(vals[j] * strides[j])
            out[g_idx, :] = (vals * strides[:, None]).sum(axis=0)
    return out


def sample_from_discrete(prob, n_samples):
    """
    Sample from a discrete distribution.
    
    Parameters
    ----------
    prob : np.ndarray
        Joint probability distribution (can be multidimensional).
    n_samples : int
        Number of samples to generate.
    
    Returns
    -------
    data : np.ndarray
        Array of shape (n_vars, n_samples) with sampled outcomes.
    """
    # Flatten the distribution
    prob_flat = prob.flatten()
    prob_flat /= prob_flat.sum()  # ensure normalization
    
    # Get number of variables and their cardinalities
    n_vars = prob.ndim
    cardinalities = prob.shape
    
    # Sample indices from flattened distribution
    flat_indices = np.random.choice(len(prob_flat), size=n_samples, p=prob_flat)
    
    # Convert flat indices back to multi-dimensional indices
    multi_indices = np.array(np.unravel_index(flat_indices, cardinalities))
    
    return multi_indices


def estimate_discrete_distribution(x1, x2, y1, y2, alph_size=2):
    """
    Estimate the joint probability distribution of 4 discrete time series.
    
    Parameters:
    - x1, x2, y1, y2: Lists or 1D numpy arrays of the same length, containing discrete values.
    - alph_size: The size of the alphabet (number of discrete values each variable can take). Default is 2.
    
    Returns:
    - A 2x2x2x2 numpy array representing the joint probability distribution.
    """
    # Ensure the inputs are numpy arrays
    x1, x2, y1, y2 = map(np.asarray, (x1, x2, y1, y2))
    
    # Check that all series have the same length
    if not (len(x1) == len(x2) == len(y1) == len(y2)):
        raise ValueError("All input time series must have the same length.")
    
    # Initialize a 2x2x2x2 matrix to store joint counts
    joint_counts = np.zeros((alph_size, alph_size, alph_size, alph_size))
    
    # Count occurrences of each combination
    for i in range(len(x1)):
        joint_counts[x1[i], x2[i], y1[i], y2[i]] += 1
    
    # Normalize to obtain probabilities
    joint_probabilities = joint_counts / np.sum(joint_counts)
    
    return joint_probabilities