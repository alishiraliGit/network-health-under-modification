import numpy as np
from tqdm import tqdm

from utils.math_tools import freq2vec, one_hot

rng = np.random.default_rng(1)


class StochasticBlockModel:
    def __init__(self, p, q, k, ns):
        self.p = p
        self.q = q
        self.k = k
        self.ns = ns
        self.types = freq2vec(ns)
        self.p_mat = self.init_probability_mat()

    def init_probability_mat(self):
        u_mat = one_hot(self.types)

        p_mat = (self.p - self.q)*u_mat.dot(u_mat.T) + self.q

        return p_mat

    def draw(self):
        adj_mat_tril = np.tril(np.random.random((np.sum(self.ns), np.sum(self.ns))) <= self.p_mat, k=-1)*1

        adj_mat = adj_mat_tril + adj_mat_tril.T

        return adj_mat


class JacksonRogersModifiedModel:
    def __init__(self, n_s, n_d, n_f, alpha, ps, init_integrated=True):
        assert np.sum(ps) == 1

        self.n_s = n_s
        self.n_d = n_d
        self.n_f = n_f
        self.alpha = alpha
        self.k = len(ps)
        self.ps = ps  # ps[i]: probability of having a new node from type i

        self.types = None
        self.edges = None  # edge=[a, b] means a <-- b
        self.adj_mat = None  # adj_mat[a, b] = 1 means a <-- b

        if init_integrated:
            self.init_complete_directed_graph()
        else:
            self.init_disintegrated_graph()

    def init_complete_directed_graph(self):
        sufficient_n_k = np.maximum(self.n_s, self.n_d) + self.n_f

        self.adj_mat = np.ones((sufficient_n_k*self.k,)*2)
        self.adj_mat[np.tril(self.adj_mat) == 1] = 0

        self.edges = np.argwhere(self.adj_mat == 1)

        self.types = freq2vec((sufficient_n_k,)*self.k)

    def init_disintegrated_graph(self):
        sufficient_n = np.maximum(self.n_s, self.n_d) + self.n_f
        sbm = StochasticBlockModel(p=1, q=0, k=self.k, ns=(sufficient_n,)*self.k)

        self.adj_mat = sbm.draw()
        self.adj_mat[np.tril(self.adj_mat) == 1] = 0

        self.edges = np.argwhere(self.adj_mat == 1)

        self.types = sbm.types

    def forward_one_step(self):
        n = self.types.shape[0]

        # Draw a random type for the new node v
        v_type = np.random.choice(range(self.k), size=1, p=self.ps).item()
        v_index = n

        # Find new friends (phase 1)
        sim_friends_1 = np.random.choice(np.where(self.types == v_type)[0], size=self.n_s, replace=False)
        dis_friends_1 = np.random.choice(np.where(self.types != v_type)[0], size=self.n_d, replace=False)

        # Find new friends (phase 2)
        n_f_s = np.round(self.alpha*self.n_f).astype(int)
        n_f_d = np.round((1 - self.alpha)*self.n_f).astype(int)

        u_sim = np.zeros((n, 1))
        u_sim[sim_friends_1, :] = 1
        u_dis = np.zeros((n, 1))
        u_dis[dis_friends_1, :] = 1

        friends_of_sim_friends = self.adj_mat.dot(u_sim)[:, 0]
        friends_of_dis_friends = self.adj_mat.dot(u_dis)[:, 0]

        friends_of_sim_friends[sim_friends_1] = 0
        friends_of_sim_friends[dis_friends_1] = 0
        friends_of_dis_friends[sim_friends_1] = 0
        friends_of_dis_friends[dis_friends_1] = 0

        n_f_s = np.minimum(n_f_s, np.sum(friends_of_sim_friends > 0))
        n_f_d = np.minimum(n_f_d, np.sum(friends_of_dis_friends > 0))
        f_sim_f_2 = np.random.choice(np.where(friends_of_sim_friends > 0)[0], size=n_f_s, replace=False)
        f_dis_f_2 = np.random.choice(np.where(friends_of_dis_friends > 0)[0], size=n_f_d, replace=False)

        # Concat. all new friends and make new edges
        new_friends = np.concatenate((sim_friends_1, dis_friends_1, f_sim_f_2, f_dis_f_2))

        edges_to_new_friends = np.append(new_friends.reshape((-1, 1)),
                                         v_index*np.ones((len(new_friends), 1)).astype(int),
                                         axis=1)

        # Update types and edges
        self.types = np.append(self.types, [v_type])
        self.edges = np.append(self.edges, edges_to_new_friends, axis=0)

        # Update adj_mat
        self.adj_mat = np.zeros((n + 1, n + 1))
        self.adj_mat[self.edges[:, 0], self.edges[:, 1]] = 1

    def forward(self, steps=1):
        for step in range(steps):
            self.forward_one_step()


class AsikainenModel:
    def __init__(self, n_a, n_b, c, s, sp, p_0=0.5, q_0=0.5):
        self.n_a, self.n_b = n_a, n_b  # number of nodes from type a and b
        self.types = [0]*self.n_a + [1]*self.n_b
        self.n = self.n_a + self.n_b
        self.c = c
        self.s = s
        self.sp = sp
        self.adj_mat = StochasticBlockModel(p=p_0, q=q_0, k=2, ns=[n_a, n_b]).draw()

    def forward_one_step(self):
        focal_node = rng.integers(0, self.n)

        do_triadic_closure = rng.random() <= self.c
        if do_triadic_closure:
            # Find a candidate node
            e_focal = np.zeros((self.n, 1))
            e_focal[focal_node, 0] = 1
            ff_focal = self.adj_mat.dot(self.adj_mat.dot(e_focal))[:, 0]
            ff_focal[focal_node] = 0
            ff_focal[self.adj_mat[focal_node] == 1] = 0

            if np.sum(ff_focal) == 0:
                return

            candidate_node = rng.choice(range(self.n), p=ff_focal/np.sum(ff_focal))

            # Decide whether to connect or not
            if self.types[focal_node] == self.types[candidate_node]:
                add_edge = rng.random() <= self.sp
            else:
                add_edge = rng.random() <= 1 - self.sp

        else:
            # Find a candidate node
            candidate_node = rng.integers(0, self.n)
            if self.adj_mat[focal_node, candidate_node] == 1 or candidate_node == focal_node:
                return

            # Decide whether to connect or not
            if self.types[focal_node] == self.types[candidate_node]:
                add_edge = rng.random() <= self.s
            else:
                add_edge = rng.random() <= 1 - self.s

        if add_edge:
            self.adj_mat[focal_node, candidate_node] = 1
            self.adj_mat[candidate_node, focal_node] = 1

            del_node = rng.choice(range(self.n), p=self.adj_mat[focal_node]/np.sum(self.adj_mat[focal_node]))
            self.adj_mat[focal_node, del_node] = 0
            self.adj_mat[del_node, focal_node] = 0

    def forward(self, steps=1, verbose=False):
        for _step in tqdm(range(steps), disable=not verbose):
            self.forward_one_step()


if __name__ == '__main__':
    mdl = JacksonRogersModifiedModel(
        n_s=4,
        n_d=3,
        n_f=8,
        alpha=0.5,
        ps=(0.5, 0.5)
    )
