import os
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

from networksimulator import StochasticBlockModel
from networkmodifier import WedgeCloser, Dummy, EdgeCloser, LocalBridgeCloser, ModificationType
from networkmeasure import algebraic_connectivity, share_of_quantile_of_eigenvector_centralities,\
    share_of_quantile_of_kb_centralities, share_of_quantile_of_closeness_centralities,\
    share_of_quantile_of_betweenness_centralities, integration

if __name__ == '__main__':
    # ----- Settings ------
    plt.rcParams['axes.prop_cycle'] = plt.cycler(color=['r', 'b', 'k', 'g'])

    # Path
    save_path = os.path.join('..', 'results')

    # Simulator settings
    p, q = 0.8, 0.1
    ns = (20, 20)
    k = len(ns)

    # Modifier settings
    modifier_sett = {
        'm': 1,
        'p_s': 1/2
    }

    # Measure settings
    measure_sett = {
        # 'q': 0.2,
    }

    # General
    n_rept = 10000

    # ----- Init. network simulator -----
    mdl = StochasticBlockModel(p, q, k, ns)
    modifier_sett['types'] = mdl.types
    measure_sett['types'] = mdl.types

    # ----- Init. network modifiers -----
    modifier_1 = EdgeCloser(modification_type=ModificationType.RANDOM_BIASED)
    modifier_2 = WedgeCloser(modification_type=ModificationType.RANDOM)
    modifier_3 = Dummy(modification_type=ModificationType.RANDOM)
    modifiers = [modifier_1, modifier_2, modifier_3]

    n_modifier = len(modifiers)

    # ----- Select measure function -----
    measure_func = integration
    # measure_name = 'share of %.1f quantile' % measure_sett['q']
    # measure_name = '$\lambda_2$'
    measure_name = 'integration'

    # ----- Loop over samples -----
    measures = np.zeros((n_rept, n_modifier))
    for i_rept in tqdm(range(n_rept)):
        adj_mat = mdl.draw()

        for i_m, modifier in enumerate(modifiers):
            modifier.fit_transform(adj_mat, **modifier_sett)

            measure = measure_func(modifier.adj_mat, **measure_sett)

            measures[i_rept, i_m] = measure

    # ----- Results -----
    # Compare average measures
    avg_measures = np.mean(measures, axis=0)

    print('Average measure:')
    print(avg_measures)

    # Compare cumulative mu
    plt.figure(figsize=(4, 4))
    for m in range(n_modifier):
        plt.hist(measures[:, m], bins=50, density=True, alpha=0.5)

    # plt.xlim((0, measure_sett['q']))
    plt.xlabel(measure_name)
    plt.ylabel('density')
    # plt.title('Distribution of ' + measure_name)
    plt.legend(('after closing random edges', 'after closing random wedges', 'before'))

    plt.tight_layout()

    # plt.savefig(os.path.join(save_path, 'integration_wedge(40-20).png'))
