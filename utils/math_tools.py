import numpy as np


def vec2freq(vec):
    if type(vec) is not np.ndarray:
        vec = np.array(vec)

    k = np.max(vec) + 1
    items = []
    freqs = []
    for i_k in range(k):
        items.append(i_k)
        freqs.append(np.sum(vec == i_k))

    return np.array(items), np.array(freqs)


def freq2vec(freqs):
    if type(freqs) is not np.ndarray:
        freqs = np.array(freqs)

    n = np.sum(freqs)
    k = len(freqs)

    vec = np.zeros((n,)).astype(int)
    from_idx = 0
    for i_k in range(k - 1):
        to_idx = from_idx + freqs[i_k]
        vec[from_idx:to_idx] = i_k
        from_idx = to_idx
    vec[from_idx:] = k - 1

    return vec


def one_hot(types):
    n = len(types)

    u_mat = np.zeros((n, np.max(types) + 1))

    u_mat[list(range(n)), types] = 1

    return u_mat
