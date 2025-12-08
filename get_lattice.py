import numpy as np
import sympy as sp

def get_ig_lattice(pid, verbose = True):
    """
    Calculate PhiID lattice and return the PhiID atoms. 
    """

    rtr, rta, rtb, rts = sp.var('rtr rta rtb rts')
    xtr, xta, xtb, xts = sp.var('xtr xta xtb xts')
    ytr, yta, ytb, yts = sp.var('ytr yta ytb yts')
    str, sta, stb, sts = sp.var('str sta stb sts')

    # Extract MI and PID values from the provided dict (use 0.0 as safe default)
    Ixa   = pid.get('Ixa')
    Ixb   = pid.get('Ixb')
    Iya   = pid.get('Iya')
    Iyb   = pid.get('Iyb')
    Ixya  = pid.get('Ixya')
    Ixyb  = pid.get('Ixyb')
    Ixab  = pid.get('Ixab')
    Iyab  = pid.get('Iyab')
    Ixyab = pid.get('Ixyab')

    I_stx_str = pid.get('I_stx_str')
    I_sty_str = pid.get('I_sty_str')
    I_xts_rts = pid.get('I_xts_rts')
    I_yts_rts = pid.get('I_yts_rts')
    I_str_stx_sty_sts = pid.get('I_str_stx_sty_sts')
    I_rts_xts_yts_sts = pid.get('I_rts_xts_yts_sts')

    I_sts = pid.get('I_sts')
    
    ## Set up sympy system of equations
    eqs = [ \
        rtr + rta + xtr + xta - Ixa,
        rtr + rtb + ytr + ytb - Iyb,
        rtr + rtb + xtr + xtb - Ixb,
        rtr + rta + ytr + yta - Iya,
        rtr + rta + xtr + xta + ytr + yta + str + sta - Ixya,
        rtr + rtb + xtr + xtb + ytr + ytb + str + stb - Ixyb,
        rtr + xtr + rta + xta + rtb + xtb + rts + xts - Ixab,
        rtr + ytr + rta + yta + rtb + ytb + rts + yts - Iyab,
        rtr+xtr+ytr+str+ rta+xta+yta+sta+ rtb+xtb+ytb+stb+ rts+xts+yts+sts - Ixyab,
        sta + str - I_stx_str,
        stb + str - I_sty_str,
        xts + rts - I_xts_rts,
        yts + rts - I_yts_rts,
        str + sta + stb + sts - I_str_stx_sty_sts,
        rts + xts + yts +sts - I_rts_xts_yts_sts,
        sts - I_sts,
    ]

    ## Solve and print
    all_pid = [rtr, rta, rtb, rts, \
                xtr, xta, xtb, xts, \
                ytr, yta, ytb, yts, \
                str, sta, stb, sts]

    m, b = sp.linear_eq_to_matrix(eqs, all_pid)
    v = np.linalg.lstsq(np.matrix(m, dtype=float), np.matrix(b, dtype=float), rcond=None)[0]
    v = [val.item() for val in v]
    
    res = {}
    for n,a in zip(all_pid, np.round(v, 5)):
        res[n.name] = a
        if verbose:
            print('%s: %f' % (n, a))

    return res

