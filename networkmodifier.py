import abc
import warnings

import numpy as np
from abc import ABC

from utils.math_tools import one_hot


class NetworkModifier(ABC):
    @abc.abstractmethod
    def fit(self, adj_mat):
        pass

    @abc.abstractmethod
    def transform(self, **kwargs):
        pass

    def fit_transform(self, adj_mat, **kwargs):
        self.fit(adj_mat)
        return self.transform(**kwargs)


class ModificationType:
    RANDOM = 'random'
    RANDOM_MONO = 'random_mono'
    RANDOM_BI = 'random_bi'
    RANDOM_BIASED = 'random_biased'


class WedgeCloser(NetworkModifier):
    def __init__(self, modification_type):
        self.modification_type = modification_type

        self.adj_mat = None  # This is a copy version of the input matrix
        self.wedges_mat = None

    def fit(self, adj_mat):
        self.adj_mat = adj_mat.copy()

        n = self.adj_mat.shape[0]

        self.wedges_mat = np.zeros((n, n))

        adj_2_mat = self.adj_mat.dot(self.adj_mat)

        for v in range(n):
            e_v = np.zeros((n, 1))
            e_v[v, 0] = 1

            n_2_v = adj_2_mat.dot(e_v)[:, 0]
            n_2_v[v] = 0

            n_1_v = self.adj_mat.dot(e_v)[:, 0]

            filt = (n_2_v >= 1) & (n_1_v == 0)
            self.wedges_mat[v, filt] = n_2_v[filt]

    def transform(self, m=1, types=None, gamma=1, gammap=1, **_kwargs):
        # Attention: run only for m=1
        if m == 0:
            return self.adj_mat, self.wedges_mat

        if self.modification_type == ModificationType.RANDOM:
            self.adj_mat, self.wedges_mat = \
                self._close_random_biased_wedges(self.adj_mat, self.wedges_mat, m, types, gamma=1, gammap=1)
        elif self.modification_type == ModificationType.RANDOM_MONO:
            self.adj_mat, self.wedges_mat = \
                self._close_random_biased_wedges(self.adj_mat, self.wedges_mat, m, types, gamma=1, gammap=0)
        elif self.modification_type == ModificationType.RANDOM_BI:
            self.adj_mat, self.wedges_mat = \
                self._close_random_biased_wedges(self.adj_mat, self.wedges_mat, m, types, gamma=0, gammap=1)
        elif self.modification_type == ModificationType.RANDOM_BIASED:
            self.adj_mat, self.wedges_mat = \
                self._close_random_biased_wedges(self.adj_mat, self.wedges_mat, m, types, gamma, gammap)
        else:
            raise 'bad modification type!'

        # Important! Wedges will be different after a new edge is added to the network.
        # ToDo: maybe a more efficient way exists?
        self.fit(self.adj_mat)

        return self.adj_mat, self.wedges_mat

    @staticmethod
    def _close_random_biased_wedges(adj_mat, wedges_mat, m, types, gamma, gammap):
        assert (gamma > 0) or (gammap > 0)

        adj_mat_copy = adj_mat.copy()
        wedges_mat_copy = wedges_mat.copy()

        n = adj_mat_copy.shape[0]

        # Separate mono and bi wedges
        u_mat = one_hot(types)
        mo_mask_mat = u_mat.dot(u_mat.T) - np.eye(n)
        bi_mask_mat = 1 - u_mat.dot(u_mat.T)

        masked_mo_wedges = wedges_mat_copy*mo_mask_mat
        masked_bi_wedges = wedges_mat_copy*bi_mask_mat

        # Form a probability matrix
        sum_mo_wedges = np.sum(masked_mo_wedges)
        sum_bi_wedges = np.sum(masked_bi_wedges)

        if (sum_mo_wedges == 0) and (sum_bi_wedges == 0):
            warnings.warn('No wedge to be closed!')
            return adj_mat_copy, wedges_mat_copy

        if (sum_mo_wedges == 0) and (gammap == 0):
            warnings.warn('gammap is 0 but no mono wedge to be closed!')
            return adj_mat_copy, wedges_mat_copy

        if (sum_bi_wedges == 0) and (gamma == 0):
            warnings.warn('gamma is 0 but no bi wedge to be closed!')
            return adj_mat_copy, wedges_mat_copy

        p_mat = gamma*masked_mo_wedges + gammap*masked_bi_wedges
        p_mat = p_mat/np.sum(p_mat)*2

        # Find indices to close
        rows, cols = np.where(np.tril(p_mat) > 0)

        if m > len(rows):
            warnings.warn('There are less number of wedges than number of required wedges. Closing all wedges.')
            m = len(rows)

        p = p_mat[(rows, cols)]

        indices = np.random.choice(range(len(rows)), size=m, replace=False, p=p)

        # Update adj_mat and wedges_mat
        adj_mat_copy[(rows[indices], cols[indices])] = 1
        adj_mat_copy[(cols[indices], rows[indices])] = 1

        wedges_mat_copy[(rows[indices], cols[indices])] = 0
        wedges_mat_copy[(cols[indices], rows[indices])] = 0

        return adj_mat_copy, wedges_mat_copy


class EdgeDropper(NetworkModifier):
    def __init__(self, modification_type):
        self.modification_type = modification_type

        self.adj_mat = None  # This is a copy version of the input matrix

    def fit(self, adj_mat):
        self.adj_mat = adj_mat.copy()

    def transform(self, m=1, types=None, gamma=1, gammap=1, **_kwargs):
        if m == 0:
            return self.adj_mat

        if self.modification_type == ModificationType.RANDOM:
            self.adj_mat = self._drop_random_biased_edges(self.adj_mat, m, types, gamma=1, gammap=1)
        elif self.modification_type == ModificationType.RANDOM_MONO:
            self.adj_mat = self._drop_random_biased_edges(self.adj_mat, m, types, gamma=1, gammap=0)
        elif self.modification_type == ModificationType.RANDOM_BI:
            self.adj_mat = self._drop_random_biased_edges(self.adj_mat, m, types, gamma=0, gammap=1)
        elif self.modification_type == ModificationType.RANDOM_BIASED:
            self.adj_mat = self._drop_random_biased_edges(self.adj_mat, m, types, gamma, gammap)
        else:
            raise 'bad modification type!'

        return self.adj_mat

    @staticmethod
    def _drop_random_biased_edges(adj_mat, m, types, gamma, gammap):
        adj_mat_copy = adj_mat.copy()

        n = adj_mat_copy.shape[0]

        # Separate mono and bi edges
        u_mat = one_hot(types)
        mo_mask_mat = u_mat.dot(u_mat.T) - np.eye(n)
        bi_mask_mat = 1 - u_mat.dot(u_mat.T)

        masked_mo_adj = adj_mat_copy*mo_mask_mat
        masked_bi_adj = adj_mat_copy*bi_mask_mat

        # Form a probability matrix
        sum_mo_adj = np.sum(masked_mo_adj)
        sum_bi_adj = np.sum(masked_bi_adj)

        if (sum_mo_adj == 0) and (sum_bi_adj == 0):
            warnings.warn('No edge to be dropped!')
            return adj_mat_copy

        if (sum_mo_adj == 0) and (gammap == 0):
            warnings.warn('gammap is 0 but no mono edge to be dropped!')
            return adj_mat_copy

        if (sum_bi_adj == 0) and (gamma == 0):
            warnings.warn('gamma is 0 but no bi edge to be dropped!')
            return adj_mat_copy

        p_mat = gamma*masked_mo_adj + gammap*masked_bi_adj
        p_mat = p_mat/np.sum(p_mat)*2

        # Find indices to drop
        rows, cols = np.where(np.tril(p_mat) > 0)

        if m > len(rows):
            warnings.warn('There are less number of edges than number of required edges. Dropping all edges.')
            m = len(rows)

        p = p_mat[(rows, cols)]

        indices = np.random.choice(range(len(rows)), size=m, replace=False, p=p)

        adj_mat_copy[(rows[indices], cols[indices])] = 0
        adj_mat_copy[(cols[indices], rows[indices])] = 0

        return adj_mat_copy


class EdgeCloser(NetworkModifier):
    def __init__(self, modification_type):
        self.modification_type = modification_type

        self.adj_mat = None  # This is a copy version of the input matrix
        self.ed = EdgeDropper(modification_type=modification_type)

    def fit(self, adj_mat):
        self.adj_mat = adj_mat.copy()

        adj_p_mat = 1 - self.adj_mat - np.eye(self.adj_mat.shape[0])
        self.ed.fit(adj_p_mat)

    def transform(self, m=1, types=None, gamma=1, gammap=1, **kwargs):
        adj_p_mat = self.ed.transform(m, types, gamma=gamma, gammap=gammap, **kwargs)

        self.adj_mat = 1 - adj_p_mat - np.eye(self.adj_mat.shape[0])

        return self.adj_mat


class LocalBridgeCloser(NetworkModifier):
    def __init__(self, modification_type):
        self.modification_type = modification_type

        self.adj_mat = None
        self.wc = WedgeCloser(modification_type=modification_type)
        self.ec = EdgeCloser(modification_type=modification_type)

    def fit(self, adj_mat):
        self.adj_mat = adj_mat.copy()

        self.wc.fit(self.adj_mat)

        ec_adj_mat = self.adj_mat.copy()
        ec_adj_mat[self.wc.wedges_mat >= 1] = 1
        self.ec.fit(ec_adj_mat)

    def transform(self, m=1, types=None, p_s=0.5, **kwargs):
        # Attention: run it for m=1 only if you want a local bridge to really be closed

        # Close edges (except wedges)
        self.ec.transform(m=m, types=types, p_s=p_s, **kwargs)

        # Update adj_mat
        self.adj_mat = self.ec.adj_mat.copy()
        self.adj_mat[self.wc.wedges_mat >= 1] = 0

        # Refit #ToDo: Maybe more efficient ways exist?
        self.fit(self.adj_mat)

        return self.adj_mat


class Dummy(NetworkModifier):
    def __init__(self, modification_type):
        self.modification_type = modification_type

        self.adj_mat = None  # This is a copy version of the input matrix

    def fit(self, adj_mat):
        self.adj_mat = adj_mat.copy()

    def transform(self, **kwargs):
        return self.adj_mat
