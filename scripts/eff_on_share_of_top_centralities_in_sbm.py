import os
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

from networksimulator import StochasticBlockModel
from networkmodifier import WedgeCloser, EdgeCloser, LocalBridgeCloser, Dummy, ModificationType
from networkmeasure import katz_bonacich_centrality, eigvec_centrality, closeness_centrality, betweenness_centrality


if __name__ == '__main__':
    # ----- Settings ------
    colors = ['r', 'b', 'k', 'g']
    plt.rcParams['axes.prop_cycle'] = plt.cycler(color=colors)

    # Path
    save_path = os.path.join('..', 'results')

    # Simulator settings
    p, q = 0.2, 0.05
    ns = (20, 10)
    k = len(ns)

    # Modifier settings
    modifier_sett = {
        'm': 5,
        # 'p_s': 0.70
    }

    # Measure settings
    measure_sett = {
        'phi': 0.1
    }

    # General
    n_rept = 500

    # ----- Init. network simulator -----
    mdl = StochasticBlockModel(p, q, k, ns)
    modifier_sett['types'] = mdl.types

    # ----- Init. network modifiers -----
    modifier_1 = WedgeCloser(modification_type=ModificationType.RANDOM_MONO)
    modifier_2 = WedgeCloser(modification_type=ModificationType.RANDOM_BI)
    modifier_3 = Dummy(modification_type=ModificationType.RANDOM)
    modifier_4 = WedgeCloser(modification_type=ModificationType.RANDOM)
    modifiers = [modifier_1, modifier_2, modifier_3, modifier_4]

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
    n = np.sum(ns)

    sorted_measures = np.argsort(-measures, axis=2)
    share_of_top = np.zeros((n_modifier, n))
    share_of_top_std = np.zeros((n_modifier, n))
    for v in range(n):
        for m in range(n_modifier):
            share_of_top[m, v] = np.sum(sorted_measures[:, m, ns[0]:] <= v)/((v + 1)*n_rept)
            share_of_top_std[m, v] = np.std(np.sum(sorted_measures[:, m, ns[0]:] <= v, axis=1)/(v + 1))

    plt.figure(figsize=(4, 4))
    for m in range(n_modifier):
        plt.plot(range(n), share_of_top[m], color=colors[m])
        plt.fill_between(range(n),
                         share_of_top[m] - share_of_top_std[m]/np.sqrt(n_rept),
                         share_of_top[m] + share_of_top_std[m]/np.sqrt(n_rept),
                         color=colors[m], alpha=0.2)

    plt.plot([0, n - 1], [ns[1]/n]*2, 'k--')

    plt.ylim((0, 1))

    plt.xlabel('k')
    plt.ylabel('share of minority in top k')
    plt.legend(('after closing mono', 'after closing bi', 'before'))

    # plt.savefig(os.path.join(save_path, 'share_of_top_k_BC_wedge(40-20).png'))

    plt.tight_layout()
