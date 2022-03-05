import os
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
from scipy.io import savemat

from networksimulator import StochasticBlockModel
from networkmodifier import WedgeCloser, EdgeCloser, ModificationType
from networkmeasure import integration

if __name__ == '__main__':
    # ----- Settings ------
    plt.rcParams['axes.prop_cycle'] = plt.cycler(color=['r', 'b', 'k', 'g'])

    # Path
    save_path = os.path.join('..', 'results', 'sim_rel_effect_on_integration_in_sbm')
    os.makedirs(save_path, exist_ok=True)

    # Simulator settings
    q = 0.2
    p_to_qs = np.linspace(0, 4, 30)

    lam = 1
    n_1 = 10
    k = 5
    ns = n_1*lam**np.arange(k)

    # Modifier settings
    modifier_sett = {
        'm': 1,
    }

    gammas = np.linspace(1, 3, 30)

    # Measure settings
    measure_sett = {}

    # General
    n_rept = 3000

    # ----- Init. network modifiers -----
    modifier_1 = WedgeCloser(modification_type=ModificationType.RANDOM)
    modifier_2 = EdgeCloser(modification_type=ModificationType.RANDOM_BIASED)
    modifiers = [modifier_1, modifier_2]

    n_modifier = len(modifiers)

    # ----- Select measure function -----
    measure_func = integration
    measure_name = 'integration'

    # -----
    plt.figure(figsize=(4, 4))

    measures = np.zeros((len(gammas), len(p_to_qs), n_rept, n_modifier))

    for i_gamma, gamma in enumerate(gammas):
        modifier_sett['gamma'] = gamma

        for i_ptq, p_to_q in enumerate(tqdm(p_to_qs)):
            p = p_to_q*q

            # Init. network simulator
            mdl = StochasticBlockModel(p, q, k, ns)
            modifier_sett['types'] = mdl.types
            measure_sett['types'] = mdl.types

            # Loop over samples
            for i_rept in range(n_rept):
                adj_mat = mdl.draw()

                for i_m, modifier in enumerate(modifiers):
                    modifier.fit_transform(adj_mat, **modifier_sett)

                    measure = measure_func(modifier.adj_mat, **measure_sett)

                    measures[i_gamma, i_ptq, i_rept, i_m] = measure

        # Plot results
        avg_measures = np.mean(measures, axis=2)

        filt_1 = avg_measures[i_gamma, :, 0] > avg_measures[i_gamma, :, 1]
        plt.plot(gamma*np.ones(np.sum(filt_1)), p_to_qs[filt_1], 'r*')
        plt.plot(gamma*np.ones(np.sum(~filt_1)), p_to_qs[~filt_1], 'bo')

    plt.tight_layout()

    savemat(os.path.join(save_path, 'lambda%.1f_k%d.mat' % (lam, k)), {
        'avg_integration': avg_measures,
        'q': q,
        'p_to_qs': p_to_qs,
        'n_1': n_1,
        'lambda': lam,
        'k': k,
        'ns': ns,
        'm': modifier_sett['m'],
        'gammas': gammas
    })

    # plt.savefig(os.path.join(save_path, 'integration_wedge(40-20).png'))
