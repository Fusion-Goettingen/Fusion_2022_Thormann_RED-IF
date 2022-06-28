from constants import *


def get_configs(init_state, ax):
    config_base = {
        'init_state': init_state,
        'init_cov': INIT_STATE_COV,
        'init_rate': INIT_RATE,  # for random matrix
        'time_steps': TIME_STEPS,
        'ax': ax,
    }

    # MEM-EKF*
    config_memekfstar1 = {
        'name': 'MEM-EKF* X-noise',
        'color': 'blue',
        'Q': np.array([SIGMA_V1, SIGMA_V2]),
        'SH': np.array([SIGMA_OR, 0.001, 0.001]),
        'al_approx': False,  # true for ignoring alpha in state and using orientation of velocity vector instead
        'mmgw': False,
    }
    config_memekfstar1.update(config_base)

    config_memekfstar2 = {
        'name': 'MEM-EKF* Y-noise',
        'color': 'blue',
        'Q': np.array([SIGMA_V1, SIGMA_V2]),
        'SH': np.array([SIGMA_OR, 0.001, 0.001]),
        'al_approx': False,  # true for ignoring alpha in state and using orientation of velocity vector instead
        'mmgw': False,
    }
    config_memekfstar2.update(config_base)

    config_rm1 = {
        'name': 'RM X-noise',
        'color': 'lightblue',
        'Q': np.array([SIGMA_V1, SIGMA_V2]),
        'delta': RM_FORGET,
    }
    config_rm1.update(config_base)

    config_rm2 = {
        'name': 'RM Y-noise',
        'color': 'lightblue',
        'Q': np.array([SIGMA_V1, SIGMA_V2]),
        'delta': RM_FORGET,
    }
    config_rm2.update(config_base)

    config_direct1 = {
        'name': 'Direct L-noise',
        'color': 'darkblue',
        'Q': np.array([SIGMA_V1, SIGMA_V2]),
        'SH': np.array([SIGMA_OR, 0.001, 0.001]),
        'use_if': True,  # use information fusion
        'use_red': True,
    }
    config_direct1.update(config_base)

    config_direct2 = {
        'name': 'Direct W-noise',
        'color': 'darkblue',
        'Q': np.array([SIGMA_V1, SIGMA_V2]),
        'SH': np.array([SIGMA_OR, 0.001, 0.001]),
    }
    config_direct2.update(config_base)

    # fusion
    config_fusion = {
        'name': 'Normal fusion',
        'color': 'red',
        'Q': np.array([SIGMA_V1, SIGMA_V2]),
        'SH': np.array([SIGMA_OR, 0.001, 0.001]),
        'use_if': False,  # use information fusion
        'use_red': False,
    }
    config_fusion.update(config_base)

    config_fusion_red = {
        'name': 'RED normal fusion',
        'color': 'green',
        'Q': np.array([SIGMA_V1, SIGMA_V2]),
        'SH': np.array([SIGMA_OR, 0.001, 0.001]),
        'use_if': False,  # use information fusion
        'use_red': True,
    }
    config_fusion_red.update(config_base)

    config_fusion_if = {
        'name': 'Information form fusion',
        'color': 'orange',
        'Q': np.array([SIGMA_V1, SIGMA_V2]),
        'SH': np.array([SIGMA_OR, 0.001, 0.001]),
        'use_if': True,  # use information fusion
        'use_red': False,
    }
    config_fusion_if.update(config_base)

    config_fusion_if_red = {
        'name': 'RED information form fusion',
        'color': 'lightgreen',
        'Q': np.array([SIGMA_V1, SIGMA_V2]),
        'SH': np.array([SIGMA_OR, 0.001, 0.001]),
        'use_if': True,  # use information fusion
        'use_red': True,
    }
    config_fusion_if_red.update(config_base)

    config_fusion_rm = {
        'name': 'RM fusion',
        'color': 'magenta',
        'Q': np.array([SIGMA_V1, SIGMA_V2]),
        'delta': RM_FORGET,
    }
    config_fusion_rm.update(config_base)

    return config_memekfstar1, config_memekfstar2, config_fusion, config_fusion_red, config_fusion_if, \
           config_fusion_if_red, config_rm1, config_rm2, config_fusion_rm, config_direct1, config_direct2
