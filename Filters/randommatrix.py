import numpy as np

from Filters.basefilters import ExtendedObjectFilter
from Filters.filtersupport import to_matrix, get_ellipse_params, get_proc_cov
from constants import *


class RandomMatrix(ExtendedObjectFilter):
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

        self._h_mat = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
        ])

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
        # based on
        # Feldmann, Michael, Dietrich Fränken, and Wolfgang Koch. "Tracking of extended objects and group targets using
        # random matrices." IEEE Transactions on Signal Processing 59.4 (2010): 1409-1420.
        meas_2d = np.atleast_2d(meas)
        num_m = len(meas_2d)

        if num_m == 0:
            return

        meas_mean = np.mean(meas_2d, axis=0)
        meas_spread = np.einsum('xa, xb -> ab', meas_2d - meas_mean[None, :], meas_2d - meas_mean[None, :])
        shape = self._shape_state / (self._shape_rate - 6.0)
        shape_hat = 0.25*shape + meas_cov

        # kinematic update
        innov = meas_mean - self._h_mat @ self._kin_state
        innov_cov = self._h_mat @ self._kin_cov @ self._h_mat.T + shape_hat / num_m
        gain = self._kin_cov @ self._h_mat.T @ np.linalg.inv(innov_cov)
        self._kin_state += gain @ innov
        self._kin_cov -= gain @ innov_cov @ gain.T
        self._kin_cov = 0.5 * (self._kin_cov + self._kin_cov.T)

        # shape update
        x_sqrt = np.linalg.cholesky(shape)
        if not np.allclose(x_sqrt, np.tril(x_sqrt)):
            x_sqrt = x_sqrt.T
            if not np.allclose(x_sqrt, np.tril(x_sqrt)):
                print('Error in Cholesky decomposition in PMBM correct!')
        innov_cov_sqrt = np.linalg.cholesky(innov_cov)  # sqrtm(innov_cov)
        if not np.allclose(innov_cov_sqrt, np.tril(innov_cov_sqrt)):
            innov_cov_sqrt = innov_cov_sqrt.T
            if not np.allclose(innov_cov_sqrt, np.tril(innov_cov_sqrt)):
                print('Error in Cholesky decomposition in PMBM correct!')
        innov_cov_sqrt_inv = np.linalg.inv(innov_cov_sqrt)
        shape_hat_sqrt = np.linalg.cholesky(shape_hat)  # sqrtm(shape_hat)
        if not np.allclose(shape_hat_sqrt, np.tril(shape_hat_sqrt)):
            shape_hat_sqrt = shape_hat_sqrt.T
            if not np.allclose(shape_hat_sqrt, np.tril(shape_hat_sqrt)):
                print('Error in Cholesky decomposition in PMBM correct!')
        shape_hat_sqrt_inv = np.linalg.inv(shape_hat_sqrt)

        self._shape_state = self._shape_state \
                            + x_sqrt @ innov_cov_sqrt_inv @ np.outer(innov, innov) @ innov_cov_sqrt_inv.T @ x_sqrt.T \
                            + x_sqrt @ shape_hat_sqrt_inv @ meas_spread @ shape_hat_sqrt_inv.T @ x_sqrt.T
        self._shape_state = 0.5 * (self._shape_state + self._shape_state.T)
        self._shape_rate = self._shape_rate + num_m

        self._est = np.hstack([self._kin_state, get_ellipse_params(self._shape_state / (self._shape_rate - 6.0))])

    def get_est(self):
        return np.array([self._kin_state, self._shape_state]), np.array([self._kin_cov, self._shape_rate])
