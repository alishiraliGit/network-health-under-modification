import os
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA

from networksimulator import StochasticBlockModel
from networkmodifier import WedgeCloser, Dummy, EdgeCloser, LocalBridgeCloser, ModificationType
from networkmeasure import katz_bonacich_centrality, eigvec_centrality, \
    first_lorenz_dominates, first_cumulative_mu, closeness_centrality, betweenness_centrality


def plot_first_vs_second(vec_1, vec_2, color):
    plt.plot(vec_2, vec_1, 'o', alpha=0.03, color=color)

    plt.plot([np.min(vec_2), np.max(vec_2)], [np.min(vec_2), np.max(vec_2)], '-', color='k')

    reg = LinearRegression()
    x_mat = vec_2.reshape((-1, 1))
    y = vec_1
    reg.fit(x_mat, y)
    y_pr = reg.predict(np.array([[np.min(vec_2)], [np.max(vec_2)]]))

    plt.plot([np.min(vec_2), np.max(vec_2)], y_pr, '--', color='k')

    reg = LinearRegression()
    x_mat = vec_1.reshape((-1, 1))
    y = vec_2
    reg.fit(x_mat, y)
    y_pr = reg.predict(np.array([[np.min(vec_1)], [np.max(vec_1)]]))

    plt.plot(y_pr, [np.min(vec_1), np.max(vec_1)], '-.', color='k')


if __name__ == '__main__':
    # ----- Settings ------
    colors = ['r', 'b', 'k']
    plt.rcParams['axes.prop_cycle'] = plt.cycler(color=colors)

    # Path
    save_path = os.path.join('..', 'results')

    # Simulator settings
    p, q = 0.2, 0.05
    ns = (40, 20)
    k = len(ns)

    # Modifier settings
    modifier_sett = {
        'm': 40,
        # 'p_s': 1
    }

    # Measure settings
    measure_sett = {
        # 'phi': 0.2
    }

    # General
    n_rept = 10000

    # ----- Init. network simulator -----
    mdl = StochasticBlockModel(p, q, k, ns)
    modifier_sett['types'] = mdl.types

    # ----- Init. network modifiers -----
    modifier_1 = Dummy(modification_type=ModificationType.RANDOM)
    modifier_2 = WedgeCloser(modification_type=ModificationType.RANDOM)
    modifiers = [modifier_1, modifier_2]

    n_modifier = len(modifiers)

    # ----- Select measure function -----
    measure_func = eigvec_centrality

    # ----- Loop over samples -----
    measures = np.zeros((n_rept, n_modifier, len(mdl.types)))
    for i_rept in tqdm(range(n_rept)):
        adj_mat = mdl.draw()

        for i_m, modifier in enumerate(modifiers):
            modifier.fit_transform(adj_mat, **modifier_sett)

            measure = measure_func(modifier.adj_mat, **measure_sett)

            measures[i_rept, i_m, :] = measure

    # ----- Results -----
    # Compare dominance of average measures
    avg_measures = np.mean(measures, axis=0)

    first_dominates = np.zeros((n_modifier, n_modifier)).astype(bool)
    for m_1 in range(n_modifier):
        for m_2 in range(n_modifier):
            first_dominates[m_1, m_2] = first_lorenz_dominates(avg_measures[m_1], avg_measures[m_2])

    print('Pairwise dominance of average measure:')
    print(first_dominates)

    # ----- Compare dominance -----
    dominates_all = np.zeros((n_rept, n_modifier))
    for i_rept in range(n_rept):
        for m_1 in range(n_modifier):
            is_dominant = True
            for m_2 in range(n_modifier):
                is_dominant = is_dominant & \
                              first_lorenz_dominates(measures[i_rept, m_1], measures[i_rept, m_2])

            dominates_all[i_rept, m_1] = is_dominant

    print('out of %d cases:' % n_rept)
    for i_m in range(n_modifier):
        print('modifier %d is dominant in %d cases' % (i_m + 1, np.sum(dominates_all[:, i_m])))

    # Compare cumulative mu
    plt.figure(figsize=(4, 4))
    first_mu = np.zeros((n_rept, n_modifier, n_modifier))
    for m_1 in range(n_modifier):
        for m_2 in range(n_modifier):
            if m_1 == m_2:
                continue

            for i_rept in range(n_rept):
                first_mu[i_rept, m_1, m_2] = \
                    first_cumulative_mu(measures[i_rept, m_1], measures[i_rept, m_2])

            filt = first_mu[:, m_1, m_2] > -0.2
            plt.hist(first_mu[filt, m_1, m_2], bins=100, density=True, alpha=0.5)

    plt.ylim((0, 60))
    plt.xlabel('$\mu$')
    plt.ylabel('density')
    plt.title(r'Distribution of arg$\min_{\mu} \; LD(EV(A), EV(B))$')
    plt.legend(('$A$=mono, $B$=bi', '$A$=bi, $B$=mono'))

    plt.tight_layout()

    # plt.savefig(os.path.join(save_path, 'muLD_EV_localbridge-wedge_homophilic(unbalanced).png'))

    # Plot before-after centralities
    plt.figure(figsize=(9, 3))

    plt.subplot(1, 3, 1)
    plot_first_vs_second(measures[:, 1, :ns[0]].reshape((-1)),
                         measures[:, 0, :ns[0]].reshape((-1)),
                         color='b')
    plt.xlim((np.quantile(measures[:, 0, :], 0.1)/1.2, np.quantile(measures[:, 0, :], 0.9)*1.2))
    plt.ylim((np.quantile(measures[:, 1, :], 0.1)/1.2, np.quantile(measures[:, 1, :], 0.9)*1.2))
    plt.ylabel('centralities of B')
    plt.xlabel('centralities of A')
    plt.title('Majority')

    plt.subplot(1, 3, 2)
    plot_first_vs_second(measures[:, 1, ns[0]:].reshape((-1)),
                         measures[:, 0, ns[0]:].reshape((-1)),
                         color='r')
    plt.xlim((np.quantile(measures[:, 0, :], 0.1)/1.2, np.quantile(measures[:, 0, :], 0.9)*1.2))
    plt.ylim((np.quantile(measures[:, 1, :], 0.1)/1.2, np.quantile(measures[:, 1, :], 0.9)*1.2))
    plt.ylabel('centralities of B')
    plt.xlabel('centralities of A')
    plt.title('Minority')

    plt.subplot(1, 3, 3)
    plot_first_vs_second(measures[:, 1, :].reshape((-1)),
                         measures[:, 0, :].reshape((-1)),
                         color='k')
    plt.xlim((np.quantile(measures[:, 0, :], 0.1)/1.2, np.quantile(measures[:, 0, :], 0.9)*1.2))
    plt.ylim((np.quantile(measures[:, 1, :], 0.1)/1.2, np.quantile(measures[:, 1, :], 0.9)*1.2))
    plt.ylabel('centralities of B')
    plt.xlabel('centralities of A')
    plt.title('Overall')

    plt.gcf().suptitle('A=BC before, B=BC after closing random-wedges')

    plt.tight_layout()

    # plt.savefig(os.path.join(save_path, 'BC_dist_randomwedge(unbalanced).png'))

    #
    plt.figure(figsize=(4, 4))

    for m in range(n_modifier):
        plt.hist(measures[:, m, :ns[0]].reshape((-1)), bins=1000, density=True, alpha=0.5/(m + 1), color='b')
        plt.hist(measures[:, m, ns[0]:].reshape((-1)), bins=1000, density=True, alpha=0.5/(m + 1), color='r')
    
    plt.xlabel('centrality')
    plt.ylabel('distribution')

    plt.legend(['before (majority)', 'before (minority)',
                'after closing random wedges (majority)', 'after closing random wedges (minority)'])

    plt.tight_layout()