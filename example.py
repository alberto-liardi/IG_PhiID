import numpy as np
from discrete_phiid import ig_phiid_discrete
from gaussian_phiid import ig_phiid_gaussian
from synthetic_systems import get_PhiID_synthetic_discrete, get_synthetic_gaussian

if "__main__" == __name__:

    print("Analysing discrete systems...")
    prob_dicts = get_PhiID_synthetic_discrete(1e-20)[0]
    for k in prob_dicts.keys():
        print(f"\nAnalyzing system: {k}")
        prob = np.array(list(prob_dicts[k]))
        prob /= prob.sum()

        phiid = ig_phiid_discrete(prob, as_dict=True, verbose=False)

        print("\nprinting significant PhiID atoms (> 0.1 bits):")
        for k,v in phiid.items():
            if v > 0.1:
                print(f"{k}: {v}")
        print("\n")
    print("Done Discrete case.")


    print("Analysing Gaussian systems...")
    atoms = ["rtr", "rtx", "rty", "rts",
         "xtr", "xtx", "xty", "xts",
         "ytr", "ytx", "yty", "yts",
         "str", "stx", "sty", "sts"]
    for a in atoms:
        print(a)
        cov = get_synthetic_gaussian(a)
        phiid = ig_phiid_gaussian(cov, verbose=False)
        print(max(phiid, key=phiid.get), max(phiid.values()))
        print()
