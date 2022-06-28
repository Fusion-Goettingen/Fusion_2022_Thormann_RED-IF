import numpy as np

from Filters.basefilters import ExtendedObjectFilter
from Filters.filtersupport import to_matrix, get_ellipse_params, get_proc_cov
from constants import *


class FusionCenterRM(ExtendedObjectFilter):
    """
    Simple fusion of two random matrices.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self._kin_state = self._est[[X1, X2, V1, V2]]
        self._kin_cov = kwargs.get('init_cov')[:4, :4]

        self._init_rate = kwargs.get('init_rate')
        self._shape_rate = self._init_rate
        self._shape_state = to_matrix(self._est[AL], self._est[L], self._est[W], False) * (self._shape_rate - 6.0)

        self._kin_error_cov = kwargs.get('Q')
        self._shape_forget = kwargs.get('delta')

    def get_est(self):
        return self._est

    def reset(self, init_state, init_cov):
        self._est = init_state.copy()

        self._kin_state = self._est[[X1, X2, V1, V2]]
        self._kin_cov = init_cov[:4, :4]

        self._shape_rate = self._init_rate
        self._shape_state = to_matrix(self._est[AL], self._est[L], self._est[W], False) * (self._shape_rate - 6.0)

    def predict(self, td):
        # kinematic prediction
        if not NO_KIN:
            proc_noise = get_proc_cov(self._kin_error_cov, np.zeros(3), td)
            self._kin_state = MEM_KIN_DYM @ self._kin_state
            self._kin_cov = MEM_KIN_DYM @ self._kin_cov @ MEM_KIN_DYM.T + proc_noise[:4, :4]

        # shape prediction
        self._shape_rate = 6.0 + np.exp(-td / self._shape_forget) * (self._shape_rate - 6.0)
        self._shape_state *= np.exp(-td / self._shape_forget)

    def correct(self, meas, meas_cov):
        # Note: meas consists of the kinematic state and the shape matrix, meas_cov consists of the kinematic covariance
        # and the rate
        # based on
        # Li, Wenling, et al. "Distributed tracking of extended targets using random matrices." 2015 54th IEEE
        # Conference on Decision and Control (CDC). IEEE, 2015.

        state_est = np.array([meas[n][0] for n in range(len(meas))])
        shape_est = np.array([meas[n][1] for n in range(len(meas))])
        state_est_cov = np.array([meas_cov[n][0] for n in range(len(meas))])
        shape_est_rate = np.array([meas_cov[n][1] for n in range(len(meas))])

        weights = np.array([1.0 / np.trace(state_est_cov[n]) for n in range(len(meas))])
        weights = np.hstack([weights, 1.0 / np.trace(self._kin_cov)])
        weights /= np.sum(weights)

        kin_cov_new = np.linalg.inv(np.sum(weights[:-1, None, None] * np.linalg.inv(state_est_cov), axis=0)
                                    + weights[-1] * np.linalg.inv(self._kin_cov))
        self._kin_state = kin_cov_new @ (np.einsum('x, xab, xb -> a', weights[:-1], np.linalg.inv(state_est_cov),
                                                   state_est)
                                         + weights[-1] * np.linalg.inv(self._kin_cov) @ self._kin_state)
        self._kin_cov = kin_cov_new

        self._shape_rate = np.sum(weights[:-1] * shape_est_rate) + weights[-1] * self._shape_rate
        self._shape_state = np.sum(weights[:-1, None, None] * shape_est, axis=0) + weights[-1] * self._shape_state

        self._est = np.hstack([self._kin_state, get_ellipse_params(self._shape_state / (self._shape_rate - 6.0))])
