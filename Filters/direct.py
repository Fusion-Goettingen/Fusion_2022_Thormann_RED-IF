import numpy as np
from scipy.linalg import block_diag

from Filters.basefilters import ExtendedObjectFilter
from Filters.filtersupport import get_proc_cov

from constants import *


class DirectTracker(ExtendedObjectFilter):
    """
    Tracker measures ground truth directly

    """
    def __init__(self, sensor_id, **kwargs):
        super().__init__(**kwargs)

        self._kin_state = self._est[[X1, X2, V1, V2]]
        self._kin_cov = kwargs.get('init_cov')[:4, :4].copy()
        self._shape_state = self._est[[AL, L, W]]
        self._shape_cov = kwargs.get('init_cov')[4:, 4:].copy()

        self._kin_error_cov = kwargs.get('Q')
        self._shape_error_cov = kwargs.get('SH')

        self._sensor_id = sensor_id

        # introduce ambiguity
        if SIMULATE_AMBIGUITY & (self._sensor_id == 1):
            self._est = self._est[[X1, X2, V1, V2, AL, W, L]]
            self._est[AL] += 0.5 * np.pi
            self._shape_state = self._est[4:].copy()

            self._shape_cov[0, 1:] = self._shape_cov[0, -1:0:-1]
            self._shape_cov[1:, 0] = self._shape_cov[-1:0:-1, 0]
            save_l = self._shape_cov[1, 1]
            self._shape_cov[1, 1] = self._shape_cov[2, 2]
            self._shape_cov[2, 2] = save_l

    def get_est(self):
        est_out = self._est
        est_cov_out = block_diag(self._kin_cov, self._shape_cov)

        return est_out, est_cov_out

    def reset(self, init_est, init_cov):
        self._est = init_est.copy()
        self._kin_state = self._est[[X1, X2, V1, V2]]
        self._kin_cov = init_cov[:4, :4].copy()
        self._shape_state = self._est[[AL, L, W]]
        self._shape_cov = init_cov[4:, 4:].copy()

        # introduce ambiguity
        if SIMULATE_AMBIGUITY & (self._sensor_id == 1):
            self._est = self._est[[X1, X2, V1, V2, AL, W, L]]
            self._est[AL] += 0.5 * np.pi
            self._shape_state = self._est[4:].copy()

            self._shape_cov[0, 1:] = self._shape_cov[0, -1:0:-1]
            self._shape_cov[1:, 0] = self._shape_cov[-1:0:-1, 0]
            save_l = self._shape_cov[1, 1]
            self._shape_cov[1, 1] = self._shape_cov[2, 2]
            self._shape_cov[2, 2] = save_l

    def predict(self, td):
        # get process noise
        proc_noise = get_proc_cov(self._kin_error_cov, self._shape_error_cov, td)

        if not NO_KIN:
            self._kin_state = MEM_KIN_DYM @ self._kin_state
            self._kin_cov = MEM_KIN_DYM @ self._kin_cov @ MEM_KIN_DYM.T + proc_noise[:4, :4]

        self._shape_cov = self._shape_cov + proc_noise[4:, 4:]

    def correct(self, est, est_cov):
        if len(est) == 0:
            return

        kin_est = est[[X1, X2, V1, V2]].copy()
        kin_est_cov = est_cov[:4, :4].copy()
        shape_est = est[[AL, L, W]].copy()
        shape_est_cov = est_cov[4:, 4:].copy()

        if SIMULATE_AMBIGUITY & (self._sensor_id == 1):
            shape_est = shape_est[[0, 2, 1]]
            shape_est[0] += 0.5 * np.pi

            shape_est_cov[0, 1:] = shape_est_cov[0, -1:0:-1]
            shape_est_cov[1:, 0] = shape_est_cov[-1:0:-1, 0]
            save_l = self._shape_cov[1, 1]
            shape_est_cov[1, 1] = shape_est_cov[2, 2]
            shape_est_cov[2, 2] = save_l

        # kinematic update
        kin_innov = kin_est - self._kin_state
        kin_innov_cov = kin_est_cov + self._kin_cov
        kin_gain = self._kin_cov @ np.linalg.inv(kin_innov_cov)
        self._kin_state += kin_gain @ kin_innov
        self._kin_cov = (np.eye(4) - kin_gain) @ self._kin_cov

        shape_innov = shape_est - self._shape_state
        shape_innov_cov = shape_est_cov + self._shape_cov
        shape_gain = self._shape_cov @ np.linalg.inv(shape_innov_cov)
        self._shape_state += shape_gain @ shape_innov
        self._shape_cov = (np.eye(3) - shape_gain) @ self._shape_cov

        self._est = np.hstack([self._kin_state, self._shape_state])
