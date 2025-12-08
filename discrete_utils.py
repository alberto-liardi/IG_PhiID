import numpy as np
import itertools

def mutual_information(dist, src, tgt):
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

    Returns
    -------
    float
        Mutual information I(src ; tgt) in bits.
    """

    # Ensure array and normalization
    d = np.asarray(dist, dtype=float)
    assert np.isclose(d.sum(), 1.0), "Input distribution must be normalized."

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


def compute_all_mi_terms(dist):
    """
    Compute all relevant mutual information terms for the 4D distribution
    P[x1, x2, y1, y2].

    Parameters
    ----------
    dist : np.ndarray
        4D normalized array P[x1, x2, y1, y2].

    Returns
    -------
    dict
        Dictionary containing all mutual information terms.
    """

    Ixa   = mutual_information(dist, [0],   [2])
    Ixb   = mutual_information(dist, [0],   [3])
    Iya   = mutual_information(dist, [1],   [2])
    Iyb   = mutual_information(dist, [1],   [3])
    Ixya  = mutual_information(dist, [0,1], [2])
    Ixyb  = mutual_information(dist, [0,1], [3])
    Ixab  = mutual_information(dist, [0],   [2,3])
    Iyab  = mutual_information(dist, [1],   [2,3])
    Ixyab = mutual_information(dist, [0,1], [2,3])

    return dict(
        Ixa=Ixa, Ixb=Ixb, Iya=Iya, Iyb=Iyb,
        Ixya=Ixya, Ixyb=Ixyb, Ixab=Ixab, Iyab=Iyab, Ixyab=Ixyab
    )

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

    Notes
    -----
    The coalescing operation preserves normalization:
        d_coalesced.sum() == d.sum() == 1.
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
