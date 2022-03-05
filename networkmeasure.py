import numpy as np
import networkx as nx
from networkx import closeness_centrality as nx_closeness_centrality
from networkx import betweenness_centrality as nx_betweenness_centrality

from utils.math_tools import one_hot


def katz_bonacich_centrality(adj_mat, phi, **_kwargs):
    n = adj_mat.shape[0]

    c = np.linalg.inv(np.eye(n) - phi*adj_mat).dot(np.ones((n, 1)))

    return c[:, 0]


def eigvec_centrality(adj_mat, **_kwargs):
    w, v_mat = np.linalg.eig(adj_mat)

    return np.abs(v_mat[:, np.argmax(np.real(w))])


def first_lorenz_dominates(vec_1, vec_2, mu=0):
    cum_1 = np.cumsum(np.sort(vec_1))/np.sum(vec_1)
    cum_2 = np.cumsum(np.sort(vec_2))/np.sum(vec_2)

    return np.all(cum_1 >= cum_2 + mu)


def first_cumulative_mu(vec_1, vec_2):
    cum_1 = np.cumsum(np.sort(vec_1))/np.sum(vec_1)
    cum_2 = np.cumsum(np.sort(vec_2))/np.sum(vec_2)

    return np.min(cum_1 - cum_2)


def laplacian_matrix(adj_mat):
    d_mat = np.diag(np.sum(adj_mat, axis=0))

    return d_mat - adj_mat


def algebraic_connectivity(adj_mat, **kwargs):
    w, _ = np.linalg.eig(laplacian_matrix(adj_mat))

    return np.sort(w)[1]


def share_of_quantile(vec, q):
    cum = np.cumsum(np.sort(vec))/np.sum(vec)

    return cum[np.floor(len(vec)*q).astype(int)]


def share_of_quantile_of_eigenvector_centralities(adj_mat, q=0.1, **_kwargs):
    c = eigvec_centrality(adj_mat)

    return share_of_quantile(c, q)


def share_of_quantile_of_kb_centralities(adj_mat, phi, q=0.1, **_kwargs):
    c = katz_bonacich_centrality(adj_mat, phi)

    return share_of_quantile(c, q)


def adj2nx(adj_mat):
    g = nx.Graph()
    g.add_edges_from(np.argwhere(np.tril(adj_mat) > 0))
    return g


def closeness_centrality(adj_mat, **_kwargs):
    g = adj2nx(adj_mat)

    c_dic = nx_closeness_centrality(g)

    n = adj_mat.shape[0]
    c = np.zeros((n,))
    for v in range(n):
        if v in c_dic:
            c[v] = c_dic[v]
        else:
            c[v] = 0

    return c


def betweenness_centrality(adj_mat, **_kwargs):
    g = adj2nx(adj_mat)

    c_dic = nx_betweenness_centrality(g)

    n = adj_mat.shape[0]
    c = np.zeros((n,))
    for v in range(n):
        if v in c_dic:
            c[v] = c_dic[v]
        else:
            c[v] = 0

    return c


def share_of_quantile_of_closeness_centralities(adj_mat, q=0.1, **_kwargs):
    c = closeness_centrality(adj_mat)

    return share_of_quantile(c, q)


def share_of_quantile_of_betweenness_centralities(adj_mat, q=0.1, **_kwargs):
    c = betweenness_centrality(adj_mat)

    return share_of_quantile(c, q)


def integration(adj_mat, types, **_kwargs):
    n = adj_mat.shape[0]

    # Separate mono and bi wedges
    u_mat = one_hot(types)
    mo_mask_mat = u_mat.dot(u_mat.T) - np.eye(n)
    bi_mask_mat = 1 - u_mat.dot(u_mat.T)

    # Calc integration
    mo_sum = np.sum(adj_mat*mo_mask_mat)
    bi_sum = np.sum(adj_mat*bi_mask_mat)

    return bi_sum/(bi_sum + mo_sum)


if __name__ == '__main__':
    adj = np.array([[0, 1, 0],
                    [1, 0, 1],
                    [0, 1, 0]])

    gr = adj2nx(adj)

    cent = betweenness_centrality(adj)
