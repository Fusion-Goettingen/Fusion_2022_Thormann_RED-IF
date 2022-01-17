from constants import *


def get_configs(init_state, ax):
    config_base = {
        'init_state': init_state,
        'init_cov': INIT_STATE_COV,
        'time_steps': TIME_STEPS,
        'ax': ax,
    }

    # MEM-EKF*
    config_memekfstar = {
        'name': 'MEM-EKF*',
        'color': 'blue',
        'Q': np.array([SIGMA_V1, SIGMA_V2]),
        'SH': np.array([SIGMA_OR, 0.001, 0.001]),
        'al_approx': False,  # true for ignoring alpha in state and using orientation of velocity vector instead
        'mmgw': False,
    }
    config_memekfstar.update(config_base)

    config_memekfstar_mmgw = {
        'name': 'MEM-EKF*-MMGW',
        'color': 'cyan',
        'Q': np.array([SIGMA_V1, SIGMA_V2]),
        'SH': np.array([SIGMA_OR, 0.001, 0.001]),
        'al_approx': False,  # true for ignoring alpha in state and using orientation of velocity vector instead
        'mmgw': True,
    }
    config_memekfstar_mmgw.update(config_base)

    return config_memekfstar, config_memekfstar_mmgw
