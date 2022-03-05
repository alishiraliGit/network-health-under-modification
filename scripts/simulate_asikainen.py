import os
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
from scipy.io import savemat

from networksimulator import AsikainenModel
from networkmeasure import integration

if __name__ == '__main__':
    # ----- Settings -----
    # Model
    mdl_sett_ = {
        'n_a': 50,
        'n_b': 50,
        'c': 1,
        'sp': 0.5,
        'p_0': 0.05,
        'q_0': 0.05
    }

    # Simulation
    steps_ = 100000
    n_s_ = 5
    n_rept_ = 10

    # Path
    save_path = os.path.join('..', 'results', 'sim_asikainen')
    os.makedirs(save_path, exist_ok=True)

    # ----- Big loop! ------
    ss_ = np.linspace(0, 1, n_s_)

    integs_ = np.zeros((n_rept_, n_s_,))
    for i_s_, s_ in enumerate(tqdm(ss_)):
        for rept_ in range(n_rept_):
            mdl_ = AsikainenModel(
                s=s_, **mdl_sett_
            )

            mdl_.forward(steps=steps_, verbose=False)

            integs_[rept_, i_s_] = integration(mdl_.adj_mat, mdl_.types)

    print(np.mean(integs_, axis=0))

    # ----- Plot -----
    plt.plot(ss_, np.mean(integs_, axis=0), 'k')
    plt.xlim((0, 1))
    plt.ylim((0, 1))
    plt.xlabel('s')
    plt.ylabel('integration')

    # ----- Save results -----
    savemat(os.path.join(save_path, 'sim_integ_sp0.5_c%0.2f.mat' % mdl_sett_['c']),
            {'model_settings': mdl_sett_, 'sim_integ': np.mean(integs_, axis=0), 'sim_s': ss_})
