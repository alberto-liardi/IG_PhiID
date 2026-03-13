import numpy as np
from collections import defaultdict


def get_PID_synthetic_discrete(sys):
    eps = 1e-32
    if sys == "two-bit":
        p = 1 / 4
        dist = {
            "00A": p,
            "00B": eps,
            "00C": eps,
            "00D": eps,
            "01A": eps,
            "01B": p,
            "01C": eps,
            "01D": eps,
            "10A": eps,
            "10B": eps,
            "10C": p,
            "10D": eps,
            "11A": eps,
            "11B": eps,
            "11C": eps,
            "11D": p,
        }
    elif sys == "copy":
        p = 1 / 2
        dist = {
            "00A": p,
            "00B": eps,
            "00C": eps,
            "00D": eps,
            "01A": eps,
            "01B": eps,
            "01C": eps,
            "01D": eps,
            "10A": eps,
            "10B": eps,
            "10C": eps,
            "10D": eps,
            "11A": eps,
            "11B": eps,
            "11C": eps,
            "11D": p,
        }
    elif sys == "copyx":
        p = 1 / 4
        dist = {
            "00A": p,
            "00B": eps,
            "00C": eps,
            "00D": eps,
            "01A": p,
            "01B": eps,
            "01C": eps,
            "01D": eps,
            "10A": eps,
            "10B": eps,
            "10C": p,
            "10D": eps,
            "11A": eps,
            "11B": eps,
            "11C": p,
            "11D": eps,
        }
    elif sys == "unx":
        p = 1 / 8
        dist = {
            "00A": p,
            "00B": p,
            "00C": eps,
            "00D": eps,
            "01A": p,
            "01B": p,
            "01C": eps,
            "01D": eps,
            "10A": eps,
            "10B": eps,
            "10C": p,
            "10D": p,
            "11A": eps,
            "11B": eps,
            "11C": p,
            "11D": p,
        }
    elif sys == "uny":
        p = 1 / 8
        dist = {
            "00A": p,
            "00B": eps,
            "00C": p,
            "00D": eps,
            "01A": eps,
            "01B": p,
            "01C": eps,
            "01D": p,
            "10A": p,
            "10B": eps,
            "10C": p,
            "10D": eps,
            "11A": eps,
            "11B": p,
            "11C": eps,
            "11D": p,
        }
    elif sys == "xor":
        p = 1 / 8
        dist = {
            "00A": p,
            "00B": eps,
            "00C": eps,
            "00D": p,
            "01A": eps,
            "01B": p,
            "01C": p,
            "01D": eps,
            "10A": eps,
            "10B": p,
            "10C": p,
            "10D": eps,
            "11A": p,
            "11B": eps,
            "11C": eps,
            "11D": p,
        }
    elif sys == "unif" or sys == "noise":
        p = 1 / 16
        dist = {
            "00A": p,
            "00B": p,
            "00C": p,
            "00D": p,
            "01A": p,
            "01B": p,
            "01C": p,
            "01D": p,
            "10A": p,
            "10B": p,
            "10C": p,
            "10D": p,
            "11A": p,
            "11B": p,
            "11C": p,
            "11D": p,
        }

    return dist


def get_PhiID_synthetic_discrete(noise=1e-32):
    """
    Returns a dictionary of predefined joint probability mass functions (PMFs)
    for four binary random variables (each taking values in {0, 1}).

    This function includes the following distributions:
    - "Independent"
    - "COPY_transfer"
    - "COPY"
    - "XOR"
    - "down_XOR"
    - "up_XOR"
    - "transfer"

    Returns:
        dict: A dictionary mapping scenario names (str) to 4D NumPy arrays
              of shape (2, 2, 2, 2), representing the joint PMFs over
              binary variables (X1, X2, X3, X4).
    """

    # sts
    p = 1 / 8
    xor = {
        (0, 0, 0, 0): p,
        (0, 0, 0, 1): noise,
        (0, 0, 1, 0): noise,
        (0, 0, 1, 1): p,
        (0, 1, 0, 0): noise,
        (0, 1, 0, 1): p,
        (0, 1, 1, 0): p,
        (0, 1, 1, 1): noise,
        (1, 0, 0, 0): noise,
        (1, 0, 0, 1): p,
        (1, 0, 1, 0): p,
        (1, 0, 1, 1): noise,
        (1, 1, 0, 0): p,
        (1, 1, 0, 1): noise,
        (1, 1, 1, 0): noise,
        (1, 1, 1, 1): p,
    }

    # rtr or xtb
    copy_transfer = {
        (0, 0, 0, 0): p,
        (0, 0, 0, 1): noise,
        (0, 0, 1, 0): p,
        (0, 0, 1, 1): noise,
        (0, 1, 0, 0): p,
        (0, 1, 0, 1): noise,
        (0, 1, 1, 0): p,
        (0, 1, 1, 1): noise,
        (1, 0, 0, 0): noise,
        (1, 0, 0, 1): p,
        (1, 0, 1, 0): noise,
        (1, 0, 1, 1): p,
        (1, 1, 0, 0): noise,
        (1, 1, 0, 1): p,
        (1, 1, 1, 0): noise,
        (1, 1, 1, 1): p,
    }

    # xta and ytb
    p = 1 / 4
    transfer = {
        (0, 0, 0, 0): p,
        (0, 0, 0, 1): noise,
        (0, 0, 1, 0): noise,
        (0, 0, 1, 1): noise,
        (0, 1, 0, 0): noise,
        (0, 1, 0, 1): noise,
        (0, 1, 1, 0): p,
        (0, 1, 1, 1): noise,
        (1, 0, 0, 0): noise,
        (1, 0, 0, 1): p,
        (1, 0, 1, 0): noise,
        (1, 0, 1, 1): noise,
        (1, 1, 0, 0): noise,
        (1, 1, 0, 1): noise,
        (1, 1, 1, 0): noise,
        (1, 1, 1, 1): p,
    }

    # TBC
    p = 1 / 4
    copy = {
        (0, 0, 0, 0): p,
        (0, 0, 0, 1): noise,
        (0, 0, 1, 0): noise,
        (0, 0, 1, 1): noise,
        (0, 1, 0, 0): noise,
        (0, 1, 0, 1): p,
        (0, 1, 1, 0): noise,
        (0, 1, 1, 1): noise,
        (1, 0, 0, 0): noise,
        (1, 0, 0, 1): noise,
        (1, 0, 1, 0): p,
        (1, 0, 1, 1): noise,
        (1, 1, 0, 0): noise,
        (1, 1, 0, 1): noise,
        (1, 1, 1, 0): noise,
        (1, 1, 1, 1): p,
    }

    # rtr
    p = 1 / 2
    giant_bit = {
        (0, 0, 0, 0): p,
        (0, 0, 0, 1): noise,
        (0, 0, 1, 0): noise,
        (0, 0, 1, 1): noise,
        (0, 1, 0, 0): noise,
        (0, 1, 0, 1): noise,
        (0, 1, 1, 0): noise,
        (0, 1, 1, 1): noise,
        (1, 0, 0, 0): noise,
        (1, 0, 0, 1): noise,
        (1, 0, 1, 0): noise,
        (1, 0, 1, 1): noise,
        (1, 1, 0, 0): noise,
        (1, 1, 0, 1): noise,
        (1, 1, 1, 0): noise,
        (1, 1, 1, 1): p,
    }

    # rtx
    p = 1 / 4
    rtx = {
        (0, 0, 0, 0): p,
        (0, 0, 0, 1): p,
        (0, 0, 1, 0): noise,
        (0, 0, 1, 1): noise,
        (0, 1, 0, 0): noise,
        (0, 1, 0, 1): noise,
        (0, 1, 1, 0): noise,
        (0, 1, 1, 1): noise,
        (1, 0, 0, 0): noise,
        (1, 0, 0, 1): noise,
        (1, 0, 1, 0): noise,
        (1, 0, 1, 1): noise,
        (1, 1, 0, 0): noise,
        (1, 1, 0, 1): noise,
        (1, 1, 1, 0): p,
        (1, 1, 1, 1): p,
    }

    # xtr
    p = 1 / 4
    xtr = {
        (0, 0, 0, 0): p,
        (0, 0, 0, 1): noise,
        (0, 0, 1, 0): noise,
        (0, 0, 1, 1): noise,
        (0, 1, 0, 0): p,
        (0, 1, 0, 1): noise,
        (0, 1, 1, 0): noise,
        (0, 1, 1, 1): noise,
        (1, 0, 0, 0): noise,
        (1, 0, 0, 1): noise,
        (1, 0, 1, 0): noise,
        (1, 0, 1, 1): p,
        (1, 1, 0, 0): noise,
        (1, 1, 0, 1): noise,
        (1, 1, 1, 0): noise,
        (1, 1, 1, 1): p,
    }

    p = 1 / 8
    # sta
    down_XOR = {
        (0, 0, 0, 0): p,
        (0, 0, 0, 1): p,
        (0, 0, 1, 0): noise,
        (0, 0, 1, 1): noise,
        (0, 1, 0, 0): noise,
        (0, 1, 0, 1): noise,
        (0, 1, 1, 0): p,
        (0, 1, 1, 1): p,
        (1, 0, 0, 0): noise,
        (1, 0, 0, 1): noise,
        (1, 0, 1, 0): p,
        (1, 0, 1, 1): p,
        (1, 1, 0, 0): p,
        (1, 1, 0, 1): p,
        (1, 1, 1, 0): noise,
        (1, 1, 1, 1): noise,
    }

    # yts
    up_XOR = {
        (0, 0, 0, 0): p,
        (0, 0, 0, 1): noise,
        (0, 0, 1, 0): noise,
        (0, 0, 1, 1): p,
        (0, 1, 0, 0): noise,
        (0, 1, 0, 1): p,
        (0, 1, 1, 0): p,
        (0, 1, 1, 1): noise,
        (1, 0, 0, 0): p,
        (1, 0, 0, 1): noise,
        (1, 0, 1, 0): noise,
        (1, 0, 1, 1): p,
        (1, 1, 0, 0): noise,
        (1, 1, 0, 1): p,
        (1, 1, 1, 0): p,
        (1, 1, 1, 1): noise,
    }

    # rts
    p = 1 / 4
    rts = {
        (0, 0, 0, 0): p,
        (0, 0, 0, 1): noise,
        (0, 0, 1, 0): noise,
        (0, 0, 1, 1): p,
        (0, 1, 0, 0): noise,
        (0, 1, 0, 1): noise,
        (0, 1, 1, 0): noise,
        (0, 1, 1, 1): noise,
        (1, 0, 0, 0): noise,
        (1, 0, 0, 1): noise,
        (1, 0, 1, 0): noise,
        (1, 0, 1, 1): noise,
        (1, 1, 0, 0): noise,
        (1, 1, 0, 1): p,
        (1, 1, 1, 0): p,
        (1, 1, 1, 1): noise,
    }

    # strr
    p = 1 / 4
    strr = {
        (0, 0, 0, 0): p,
        (0, 0, 0, 1): noise,
        (0, 0, 1, 0): noise,
        (0, 0, 1, 1): noise,
        (0, 1, 0, 0): noise,
        (0, 1, 0, 1): noise,
        (0, 1, 1, 0): noise,
        (0, 1, 1, 1): p,
        (1, 0, 0, 0): noise,
        (1, 0, 0, 1): noise,
        (1, 0, 1, 0): noise,
        (1, 0, 1, 1): p,
        (1, 1, 0, 0): p,
        (1, 1, 0, 1): noise,
        (1, 1, 1, 0): noise,
        (1, 1, 1, 1): noise,
    }

    dicts = {
        "xor": xor,
        "copy_transfer": copy_transfer,
        "copy": copy,
        "transfer": transfer,
        "down_XOR": down_XOR,
        "up_XOR": up_XOR,
        "rts": rts,
        "giant_bit": giant_bit,
        "rtx": rtx,
        "xtr": xtr,
        "str": strr,
    }

    probs = {
        "Independent": np.repeat(1 / 16, 16).reshape(2, 2, 2, 2),
        "COPY_transfer": np.array(list(copy_transfer.values())).reshape(2, 2, 2, 2),
        "COPY": np.array(list(copy.values())).reshape(2, 2, 2, 2),
        "XOR": np.array(list(xor.values())).reshape(2, 2, 2, 2),
        "down_XOR": np.array(list(down_XOR.values())).reshape(2, 2, 2, 2),
        "up_XOR": np.array(list(up_XOR.values())).reshape(2, 2, 2, 2),
        "str": np.array(list(strr.values())).reshape(2, 2, 2, 2),
        "rtx": np.array(list(rtx.values())).reshape(2, 2, 2, 2),
        "xtr": np.array(list(xtr.values())).reshape(2, 2, 2, 2),
        "rts": np.array(list(rts.values())).reshape(2, 2, 2, 2),
        "transfer": np.array(list(transfer.values())).reshape(2, 2, 2, 2),
        "giant_bit": np.array(list(giant_bit.values())).reshape(2, 2, 2, 2),
    }

    return probs, dicts


def get_synthetic_gaussian(sys, N=10000, tseries=False):
    def eps():
        return 0.1 * np.random.randn(N)

    c = np.random.randn(N)
    if sys == "copy" or sys == "rtr":
        # RTR
        X1 = c + eps()
        X2 = c + eps()
        Y1 = c + eps()
        Y2 = c + eps()
        cov = np.cov([X1, X2, Y1, Y2]) + np.eye(4) * 1e-10
    elif sys == "rtx":
        # RTX
        X1 = c + eps()
        X2 = c + eps()
        Y1 = c + eps()
        Y2 = eps()
        cov = np.cov([X1, X2, Y1, Y2])
    elif sys == "rty":
        # RTY
        X1 = c + eps()
        X2 = c + eps()
        Y1 = eps()
        Y2 = c + eps()
        cov = np.cov([X1, X2, Y1, Y2])
    elif sys == "rts":
        # RTS
        Y1 = 100 * eps()
        Y2 = 100 * eps()
        X1 = Y1 + Y2 + eps()
        X2 = Y1 + Y2 + eps()
        cov = np.cov([X1, X2, Y1, Y2])  # + 5*np.eye(4)
    elif sys == "xtr":
        # XTR
        X1 = c + eps()
        X2 = eps()
        Y1 = c + eps()
        Y2 = c + eps()
        cov = np.cov([X1, X2, Y1, Y2])
    elif sys == "unx" or sys == "xtx":
        # XTX
        X1 = c + eps()
        X2 = eps()
        Y1 = c + eps()
        Y2 = eps()
        cov = np.cov([X1, X2, Y1, Y2])
    elif sys == "xty":
        # XTY
        X1 = c + eps()
        X2 = eps()
        Y1 = eps()
        Y2 = c + eps()
        cov = np.cov([X1, X2, Y1, Y2])
    elif sys == "xts":
        # XTS
        Y1 = 100 * eps()
        Y2 = 100 * eps()
        X1 = Y1 + Y2 + eps()
        X2 = eps()
        cov = np.cov([X1, X2, Y1, Y2])
    elif sys == "ytr":
        # YTR
        X1 = eps()
        X2 = c + eps()
        Y1 = c + eps()
        Y2 = c + eps()
        cov = np.cov([X1, X2, Y1, Y2])
    elif sys == "ytx":
        # YTX
        X1 = eps()
        X2 = c + eps()
        Y1 = c + eps()
        Y2 = eps()
        cov = np.cov([X1, X2, Y1, Y2])
    elif sys == "uny" or sys == "yty":
        # YTY
        X1 = eps()
        X2 = c + eps()
        Y1 = eps()
        Y2 = c + eps()
        cov = np.cov([X1, X2, Y1, Y2])
    elif sys == "yts":
        # YTS
        Y1 = 100 * eps()
        Y2 = 100 * eps()
        X1 = eps()
        X2 = Y1 + Y2 + eps()
        cov = np.cov([X1, X2, Y1, Y2])  # + 5*np.random.randn(4,4)
    elif sys == "str":
        # STR
        X1 = 100 * eps()
        X2 = 100 * eps()
        Y1 = X1 + X2 + eps()
        Y2 = X1 + X2 + eps()
        cov = np.cov([X1, X2, Y1, Y2])  # + 5*np.random.randn(4,4)
    elif sys == "stx":
        # STX
        X1 = 100 * eps()
        X2 = 100 * eps()
        Y1 = X1 + X2 + eps()
        Y2 = eps()
        cov = np.cov([X1, X2, Y1, Y2])
    elif sys == "sty":
        # STY
        X1 = 100 * eps()
        X2 = 100 * eps()
        Y1 = eps()
        Y2 = X1 + X2 + eps()
        cov = np.cov([X1, X2, Y1, Y2])
    elif sys == "xor" or sys == "sts":
        # STS
        X1 = 100 * eps()
        X2 = 100 * eps()
        Y1 = 100 * eps()
        Y2 = X1 + X2 - Y1 + eps()
        cov = np.cov([X1, X2, Y1, Y2])

    if tseries:
        return cov, np.array([X1, X2, Y1, Y2])
    else:
        return cov


def parse_pid_dicts(pid_dicts):
    """
    Convert a PID-like dict or list of such dicts into configs and probability arrays.

    Args:
        pid_dicts: Either a single dict or a list of dicts.
            Each dict keys look like 'XYZ' where X,Y are '0' or '1' and Z in 'A','B','C','D'.
            Values are probabilities.

    Returns:
        configs: list of tuples (int, int, str) like (0, 0, 'A')
        probs: numpy array of probabilities (if list input, concatenated along axis=0)
    """

    def single_parse(d):
        configs = []
        probs = []
        for key, p in d.items():
            x = int(key[0])
            y = int(key[1])
            z = key[2]
            configs.append((x, y, z))
            probs.append(p)
        probs = np.array(probs)
        # normalize probabilities just in case (sum to 1)
        probs = probs / probs.sum()
        return configs, probs

    if isinstance(pid_dicts, dict):
        return single_parse(pid_dicts)
    elif isinstance(pid_dicts, list):
        all_configs = []
        all_probs = []
        for d in pid_dicts:
            c, p = single_parse(d)
            all_configs.append(c)
            all_probs.append(p)
        # Assuming all have same configs order:
        # flatten probs and keep configs from first dict
        return all_configs[0], np.concatenate(all_probs)
    else:
        raise ValueError("Input must be a dict or list of dicts")


import numpy as np
from collections import defaultdict


def build_transition_dict(gate):
    transition_dict = defaultdict(list)
    for (x, y, x_next, y_next), prob in gate.items():
        transition_dict[(x, y)].append(((x_next, y_next), prob))

    # Normalize probabilities for each (x, y)
    for key in transition_dict:
        outcomes, probs = zip(*transition_dict[key])
        probs = np.array(probs, dtype=float)
        probs /= probs.sum()
        transition_dict[key] = (outcomes, probs)
    return transition_dict


def sample_from_transition(outcomes, probs):
    idx = np.random.choice(len(outcomes), p=probs)
    return outcomes[idx]


def generate_markov_time_series(
    gate, T=1000, seed=None, init_state=(0, 0), num_resets=None
):
    """
    Generate a time series based on a Markov process defined by a transition dictionary.

    Args:
        gate (dict): Transition dictionary where keys are tuples of states (x, y)
                     and values are tuples of next states (x_next, y_next) and their probabilities.
        T (int): Length of the time series to generate.
        seed (int, optional): Random seed for reproducibility.
        init_state (tuple): Initial state of the Markov process, default is (0, 0).
        num_resets (int, optional): Number of random resets to inject into the time series.
                                    If None, defaults to 1% of T with at least 1 reset.

    Returns:
        np.ndarray: A 2D NumPy array of shape (2, T) representing the time series.
    """
    if seed is not None:
        np.random.seed(seed)

    transition_dict = build_transition_dict(gate)

    time_series = [init_state]
    current_state = init_state

    # Compute reset points (e.g., 10 evenly spaced times)
    if num_resets is None:
        num_resets = max(1, T // 100)  # Default to 1% of T
    reset_points = np.linspace(1, T - 1, num=num_resets, dtype=int)

    for t in range(1, T):
        # Inject random reset if current time is in reset_points
        if t in reset_points:
            current_state = (np.random.randint(0, 2), np.random.randint(0, 2))

        if current_state not in transition_dict:
            raise ValueError(f"No transition defined for state {current_state}")
        outcomes, probs = transition_dict[current_state]
        next_state = sample_from_transition(outcomes, probs)
        time_series.append(next_state)
        current_state = next_state

    return np.array(time_series, dtype=int).T
