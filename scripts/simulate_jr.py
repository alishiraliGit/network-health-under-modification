import os
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
from scipy.io import savemat

from networksimulator import JacksonRogersModifiedModel
from networkmeasure import integration

if __name__ == '__main__':
    # ----- Settings -----
    # Model
    mdl_sett_ = {
        'n_s': 3,
        'n_d': 2,
        'n_f': 10,
        'alpha': 4/5,
        'ps': (0.7, 0.3),
        'init_integrated': True
    }

    idx_ = 166

    # Simulation
    n_step_ = 300
    n_rept_ = 10

    # Path
    save_path_ = os.path.join('..', 'results', 'sim_jr')
    os.makedirs(save_path_, exist_ok=True)

    # ----- Init. the model -----
    jr_ = JacksonRogersModifiedModel(**mdl_sett_)

    # ----- Big loop! ------
    measures_ = np.zeros((n_rept_, n_step_,))
    for i_rept_ in range(n_rept_):
        jr_ = JacksonRogersModifiedModel(**mdl_sett_)

        for i_step_ in tqdm(range(n_step_)):
            jr_.forward_one_step()

            measures_[i_rept_, i_step_] = integration(jr_.adj_mat, jr_.types)

        # Plot
        plt.plot(range(n_step_), measures_[i_rept_])

    # ----- Calc. equilibrium -----
    equil_ = (mdl_sett_['n_d'] + (1 - mdl_sett_['alpha'])*mdl_sett_['n_f']) \
        / (mdl_sett_['n_s'] + mdl_sett_['n_d'] + 2*(1 - mdl_sett_['alpha'])*mdl_sett_['n_f'])

    plt.plot(range(n_step_), (equil_,) * n_step_, 'k--')

    #
    savemat(os.path.join(save_path_, 'sim_%d.mat' % idx_), {
        'model_settings': mdl_sett_,
        'integration': measures_
    })



