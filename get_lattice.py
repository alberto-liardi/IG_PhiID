import numpy as np
import sympy as sp

def get_ig_lattice(pid, verbose=False):
    """
    Pointwise lattice solver.
    Expects pid to contain arrays (bits) for:
      Ixa, Ixb, Iya, Iyb, Ixya, Ixyb, Ixab, Iyab, Ixyab,
      I_stx_str, I_sty_str, I_xts_rts, I_yts_rts,
      I_str_stx_sty_sts, I_rts_xts_yts_sts, I_sts
    Returns dict of arrays for atoms:
      rtr,rta,rtb,rts,xtr,xta,xtb,xts,ytr,yta,ytb,yts,str,sta,stb,sts
    """
    # Unknowns in this fixed order
    names = ["rtr","rta","rtb","rts","xtr","xta","xtb","xts","ytr","yta","ytb","yts","str","sta","stb","sts"]

    # Coefficient matrix A (16x16), matches the equations in get_ig_lattice
    A = np.zeros((16, 16))
    # 1) rtr + rta + xtr + xta = Ixa
    A[0, [0, 1, 4, 5]] = 1
    # 2) rtr + rtb + ytr + ytb = Iyb
    A[1, [0, 2, 8, 10]] = 1
    # 3) rtr + rtb + xtr + xtb = Ixb
    A[2, [0, 2, 4, 6]] = 1
    # 4) rtr + rta + ytr + yta = Iya
    A[3, [0, 1, 8, 9]] = 1
    # 5) rtr + rta + xtr + xta + ytr + yta + str + sta = Ixya
    A[4, [0, 1, 4, 5, 8, 9, 12, 13]] = 1
    # 6) rtr + rtb + xtr + xtb + ytr + ytb + str + stb = Ixyb
    A[5, [0, 2, 4, 6, 8, 10, 12, 14]] = 1
    # 7) rtr + xtr + rta + xta + rtb + xtb + rts + xts = Ixab
    A[6, [0, 4, 1, 5, 2, 6, 3, 7]] = 1
    # 8) rtr + ytr + rta + yta + rtb + ytb + rts + yts = Iyab
    A[7, [0, 8, 1, 9, 2, 10, 3, 11]] = 1
    # 9) sum of all atoms = Ixyab
    A[8, :] = 1
    # 10) sta + str = I_stx_str
    A[9, [13, 12]] = 1
    # 11) stb + str = I_sty_str
    A[10, [14, 12]] = 1
    # 12) xts + rts = I_xts_rts
    A[11, [7, 3]] = 1
    # 13) yts + rts = I_yts_rts
    A[12, [11, 3]] = 1
    # 14) str + sta + stb + sts = I_str_stx_sty_sts
    A[13, [12, 13, 14, 15]] = 1
    # 15) rts + xts + yts + sts = I_rts_xts_yts_sts
    A[14, [3, 7, 11, 15]] = 1
    # 16) sts = I_sts
    A[15, [15]] = 1

    # Build RHS B (16 x N)
    # All entries must be arrays; broadcast scalars if needed
    keys = ["Ixa","Iyb","Ixb","Iya","Ixya","Ixyb","Ixab","Iyab","Ixyab",
            "I_stx_str","I_sty_str","I_xts_rts","I_yts_rts",
            "I_str_stx_sty_sts","I_rts_xts_yts_sts","I_sts"]

    B = np.vstack([pid[k] for k in keys])  # 16 x N

    # Solve A X = B (vectorized over columns)
    X = (np.linalg.inv(A) @ B).squeeze()  # X is 16 x N

    # Package result as dict of arrays
    res = {}
    for i, n in enumerate(names):
        res[n] = X[i]
        if verbose:
            print(f"{n}: mean={np.mean(res[n]):.6f}")

    return res