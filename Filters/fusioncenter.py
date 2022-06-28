import numpy as np
from numpy.random import multivariate_normal as mvn

from scipy.special import logsumexp as lse

from matplotlib.patches import Ellipse

from Filters.basefilters import ExtendedObjectFilter
from Filters.filtersupport import get_proc_cov, turn_mult, mmgw_estimate_from_particles, sample_mult,\
    reduce_mult_salmond

from constants import *


class FusionCenter(ExtendedObjectFilter):
    """
    Fusion center combining two MEM-EKF* estimates with kinematic and shape state. Can use ordinary
    fusion ignoring correlations or information fusion subtracting redundant information. Can fuse
    classically or use REDs to handle ambiguities in the elliptic shape representation.

    Basic information prediction and update based on
    Durrant-Whyte, Hugh. "Multi Sensor Data Fusion." (2001).

    Basic fusion based on
    Aeberhard, Michael, et al. "Track-to-track fusion with asynchronous sensors using information matrix fusion for
    surround environment perception." IEEE Transactions on Intelligent Transportation Systems 13.4 (2012): 1717-1726.

    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self._use_if = kwargs.get('use_if')  # use information fusion, otherwise, fuse with ordinary Kalman filter
        self._use_red = kwargs.get('use_red')  # use RED representation of ellipse state

        self._kin_state = self._est[[X1, X2, V1, V2]].copy()
        self._kin_cov = kwargs.get('init_cov')[:4, :4].copy()
        self._shape_state = np.array([self._est[[AL, L, W].copy()]])
        self._shape_cov = np.array([kwargs.get('init_cov')[4:, 4:].copy()])
        self._shape_weight = np.ones(1) * 1.0

        self._kin_error_cov = kwargs.get('Q')
        self._shape_error_cov = kwargs.get('SH')

        if self._use_if:
            self._kin_cov = np.linalg.inv(self._kin_cov)
            self._kin_state = self._kin_cov @ self._kin_state
            self._shape_cov = np.array([np.linalg.inv(self._shape_cov[i]) for i in range(len(self._shape_weight))])
            self._shape_state = np.array([self._shape_cov[i] @ self._shape_state[i]
                                         for i in range(len(self._shape_weight))])

        # keeping old estimate covariances for information filter
        self._kin_est = np.array([self._kin_state.copy(), self._kin_state.copy()])
        self._kin_est_cov = np.array([self._kin_cov.copy(), self._kin_cov.copy()])
        self._shape_est = np.array([self._shape_state[0].copy(), self._shape_state[0].copy()])
        self._shape_est_cov = np.array([self._shape_cov[0].copy(), self._shape_cov[0].copy()])

        if SIMULATE_AMBIGUITY:
            # the second Filters will have different initial states and covariances
            if len(self._shape_weight) > 1:
                self._shape_est_cov[1] = self._shape_cov[1].copy()
                self._shape_est[1] = self._shape_state[1].copy()
            else:
                init_est = self._est[[AL, W, L]].copy()
                init_est[0] += 0.5 * np.pi

                init_cov = kwargs.get('init_cov')[4:, 4:].copy()
                save_cov_l = init_cov[1].copy()
                init_cov[1] = init_cov[2].copy()
                init_cov[2] = save_cov_l.copy()
                save_cov_l = init_cov[:, 1].copy()
                init_cov[:, 1] = init_cov[:, 2].copy()
                init_cov[:, 2] = save_cov_l.copy()

                self._shape_est_cov[1] = np.linalg.inv(init_cov)
                self._shape_est[1] = self._shape_est_cov[1] @ init_est

    def get_est(self):
        return self._est

    def reset(self, init_est, init_cov):
        self._est = init_est.copy()
        self._kin_state = self._est[[X1, X2, V1, V2]].copy()
        self._kin_cov = init_cov[:4, :4].copy()
        if self._use_red:
            self._shape_state, self._shape_cov, self._shape_weight = turn_mult(self._est[[AL, L, W]].copy(),
                                                                               init_cov[4:, 4:].copy())
        else:
            self._shape_state = np.array([self._est[[AL, L, W].copy()]])
            self._shape_cov = np.array([init_cov[4:, 4:].copy()])
            self._shape_weight = np.ones(1) * 1.0

        if self._use_if:
            self._kin_cov = np.linalg.inv(self._kin_cov)
            self._kin_state = self._kin_cov @ self._kin_state
            self._shape_cov = np.array([np.linalg.inv(self._shape_cov[i]) for i in range(len(self._shape_weight))])
            self._shape_state = np.array([self._shape_cov[i] @ self._shape_state[i]
                                          for i in range(len(self._shape_weight))])
        self._kin_est = np.array([self._kin_state.copy(), self._kin_state.copy()])
        self._kin_est_cov = np.array([self._kin_cov.copy(), self._kin_cov.copy()])
        self._shape_est = np.array([self._shape_state[0].copy(), self._shape_state[0].copy()])
        self._shape_est_cov = np.array([self._shape_cov[0].copy(), self._shape_cov[0].copy()])

        if SIMULATE_AMBIGUITY:
            # the second Filters will have different initial states and covariances
            if len(self._shape_weight) > 1:
                self._shape_est_cov[1] = self._shape_cov[1].copy()
                self._shape_est[1] = self._shape_state[1].copy()
            else:
                init_est = self._est[[AL, W, L]].copy()
                init_est[0] += 0.5 * np.pi

                init_cov_shift = init_cov[4:, 4:].copy()
                save_cov_l = init_cov_shift[1].copy()
                init_cov_shift[1] = init_cov_shift[2].copy()
                init_cov_shift[2] = save_cov_l.copy()
                save_cov_l = init_cov_shift[:, 1].copy()
                init_cov_shift[:, 1] = init_cov_shift[:, 2].copy()
                init_cov_shift[:, 2] = save_cov_l.copy()

                self._shape_est_cov[1] = np.linalg.inv(init_cov_shift)
                self._shape_est[1] = self._shape_est_cov[1] @ init_est

    def predict(self, td):
        if self._use_if:
            self.predict_if(td)
        else:
            self.predict_normal(td)

    def correct(self, est, est_cov):
        for n in range(len(est)):
            if self._use_if:
                self.correct_if(est[n], est_cov[n], n)
            else:
                self.correct_normal(est[n], est_cov[n])

            # as calculation of information innovation would require calculation of covariance matrix anyway, standard
            # form is used for mixture reduction
            if self._use_if:
                self._shape_cov = np.array([np.linalg.inv(self._shape_cov[j]) for j in range(len(self._shape_weight))])
                self._shape_state = np.array([self._shape_cov[j] @ self._shape_state[j]
                                              for j in range(len(self._shape_weight))])
            self._shape_state, self._shape_cov, self._shape_weight \
                = reduce_mult_salmond(self._shape_state, self._shape_cov, self._shape_weight)

            # turn shape state back into information form (but keep as it is for last iteration as we need it in
            # Gaussian form for calculating the MMGW estimate
            if self._use_if & (n < (len(est)-1)):
                self._shape_cov = np.array([np.linalg.inv(self._shape_cov[j]) for j in range(len(self._shape_weight))])
                self._shape_state = np.array([self._shape_cov[j] @ self._shape_state[j]
                                              for j in range(len(self._shape_weight))])

        # calculate estimate
        if self._use_red:
            self._est = np.hstack([np.linalg.inv(self._kin_cov) @ self._kin_state if self._use_if else self._kin_state,
                                   mmgw_estimate_from_particles(sample_mult(self._shape_state, self._shape_cov,
                                                                            self._shape_weight, 1000))])
        else:
            self._est = np.hstack([np.linalg.inv(self._kin_cov) @ self._kin_state if self._use_if else self._kin_state,
                                   mmgw_estimate_from_particles(mvn(self._shape_state[0], self._shape_cov[0], 1000))])

        # turn shape state back into information form
        if self._use_if:
            self._shape_cov = np.array([np.linalg.inv(self._shape_cov[j]) for j in range(len(self._shape_weight))])
            self._shape_state = np.array([self._shape_cov[j] @ self._shape_state[j]
                                          for j in range(len(self._shape_weight))])

    def predict_if(self, td):
        error_mat_kin = np.array([
            [0.5 * td ** 2, 0.0],
            [0.0, 0.5 * td ** 2],
            [td, 0.0],
            [0.0, td],
        ])
        error_mat_shape = np.eye(3) * td
        # predict state and recorded last sensor estimates to current time step
        if not NO_KIN:
            self._kin_state, self._kin_cov = predict_est_if(self._kin_state, self._kin_cov, td,
                                                            self._kin_error_cov, error_mat_kin)
        for i in range(len(self._shape_weight)):
            self._shape_state[i], self._shape_cov[i] = predict_est_if(self._shape_state[i], self._shape_cov[i], td,
                                                                      self._shape_error_cov, error_mat_shape)
        for n in range(len(self._kin_est)):
            if not NO_KIN:
                self._kin_est[n], self._kin_est_cov[n] = predict_est_if(self._kin_est[n], self._kin_est_cov[n],
                                                                        td, self._kin_error_cov, error_mat_kin)
            self._shape_est[n], self._shape_est_cov[n] = predict_est_if(self._shape_est[n], self._shape_est_cov[n], td,
                                                                        self._shape_error_cov, error_mat_shape)

    def predict_normal(self, td):
        # get process noise
        proc_noise = get_proc_cov(self._kin_error_cov, self._shape_error_cov, td)

        if not NO_KIN:
            self._kin_state = MEM_KIN_DYM @ self._kin_state
            self._kin_cov = MEM_KIN_DYM @ self._kin_cov @ MEM_KIN_DYM.T + proc_noise[:4, :4]

        self._shape_cov = self._shape_cov + proc_noise[None, 4:, 4:]

    def correct_if(self, est, est_cov, sensor_id):
        # turn estimates into information form
        kin_est_cov = np.linalg.inv(est_cov[:4, :4])
        kin_est = kin_est_cov @ est[[X1, X2, V1, V2]]
        shape_est_cov = np.linalg.inv(est_cov[4:, 4:])
        shape_est = shape_est_cov @ est[[AL, L, W]]
        shape_est_al = est[AL]

        # remove redundant information
        kin_est_reduced = kin_est - self._kin_est[sensor_id]
        kin_est_cov_reduced = kin_est_cov - self._kin_est_cov[sensor_id]
        shape_est_reduced = shape_est - self._shape_est[sensor_id]
        shape_est_cov_reduced = shape_est_cov - self._shape_est_cov[sensor_id]

        # turn into RED
        if self._use_red:
            shape_est_normal, shape_est_cov_normal, _ = turn_mult(est[[AL, L, W]], est_cov[4:, 4:], information=False)
            shape_est, shape_est_cov, shape_est_weight = turn_mult(shape_est_reduced, shape_est_cov_reduced,
                                                                   information=True)
        else:
            shape_est_normal = np.array([est[[AL, L, W]].copy()])
            shape_est_cov_normal = np.array([est_cov[4:, 4:].copy()])
            shape_est = np.array([shape_est_reduced])
            shape_est_cov = np.array([shape_est_cov_reduced])
            shape_est_weight = np.ones(1) * 1.0

        # fuse kinematic state
        self._kin_state += kin_est_reduced
        self._kin_cov += kin_est_cov_reduced

        # fuse shape state
        new_shape_state = np.zeros((len(self._shape_weight) * len(shape_est_weight), 3))
        new_shape_cov = np.zeros((len(self._shape_weight) * len(shape_est_weight), 3, 3))
        new_shape_weight = np.zeros(len(self._shape_weight) * len(shape_est_weight))
        prior_log_weights = np.log(self._shape_weight)
        for i in range(len(self._shape_weight)):
            shape_cov_normal = np.linalg.inv(self._shape_cov[i])
            shape_state_normal = shape_cov_normal @ self._shape_state[i]
            for j in range(len(shape_est_weight)):
                wrap_param = np.floor((shape_est_al - shape_state_normal[0] + (j+2)*0.5*np.pi) / (2.0*np.pi))
                cur_shape_est = shape_est[j] - shape_est_cov[j] @ np.array([wrap_param * 2.0 * np.pi, 0.0, 0.0])

                new_shape_state[i*len(shape_est_weight) + j] = self._shape_state[i] + cur_shape_est  # shape_est[j]
                new_shape_cov[i*len(shape_est_weight) + j] = self._shape_cov[i] + shape_est_cov[j]

                # update weight
                shape_innov_cov = shape_cov_normal + shape_est_cov_normal[j]
                shape_innov = shape_est_normal[j] - shape_state_normal
                shape_innov[0] = ((shape_innov[0] + np.pi) % (2.0 * np.pi)) - np.pi
                new_shape_weight[i * len(shape_est_weight) + j] = \
                    -2.5 * np.log(2 * np.pi) - 0.5 * np.log(np.linalg.det(shape_innov_cov)) \
                    - 0.5 * np.dot(np.dot(shape_innov, np.linalg.inv(shape_innov_cov)), shape_innov)
                new_shape_weight[i*len(shape_est_weight) + j] += prior_log_weights[i]

        new_shape_weight -= lse(new_shape_weight)
        self._shape_weight = np.exp(new_shape_weight)
        self._shape_state = new_shape_state
        self._shape_cov = new_shape_cov

        # save estimates for next update
        self._kin_est[sensor_id] = kin_est
        self._kin_est_cov[sensor_id] = kin_est_cov
        self._shape_est[sensor_id] = np.linalg.inv(est_cov[4:, 4:]) @ est[[AL, L, W]]
        self._shape_est_cov[sensor_id] = np.linalg.inv(est_cov[4:, 4:].copy())

    def correct_normal(self, est, est_cov):
        # prepare estimates
        kin_est = est[[X1, X2, V1, V2]]
        kin_est_cov = est_cov[:4, :4]
        if self._use_red:
            shape_est, shape_est_cov, shape_est_weight = turn_mult(est[[AL, L, W]], est_cov[4:, 4:])
        else:
            shape_est = np.array([est[[AL, L, W]]])
            shape_est_cov = np.array([est_cov[4:, 4:]])
            shape_est_weight = np.ones(1) * 1.0

        # fuse kinematic state
        kin_gain = self._kin_cov @ np.linalg.inv(self._kin_cov + kin_est_cov)
        self._kin_state = self._kin_state + kin_gain @ (kin_est - self._kin_state)
        self._kin_cov = (np.eye(4) - kin_gain) @ self._kin_cov

        # fuse shape state (if RED not used, only one iteration)
        new_shape_state = np.zeros((len(self._shape_weight) * len(shape_est_weight), 3))
        new_shape_cov = np.zeros((len(self._shape_weight) * len(shape_est_weight), 3, 3))
        new_shape_weight = np.zeros(len(self._shape_weight) * len(shape_est_weight))
        prior_log_weights = np.log(self._shape_weight)
        for i in range(len(self._shape_weight)):
            for j in range(len(shape_est_weight)):
                # prepare values
                shape_innov_cov = self._shape_cov[i] + shape_est_cov[j]
                shape_gain = self._shape_cov[i] @ np.linalg.inv(shape_innov_cov)
                shape_innov = (shape_est[j] - self._shape_state[i])
                shape_innov[0] = ((shape_innov[0] + np.pi) % (2.0*np.pi)) - np.pi

                # update mean and covariance
                new_shape_state[i*len(shape_est_weight) + j] = self._shape_state[i] + shape_gain @ shape_innov
                new_shape_cov[i*len(shape_est_weight) + j] = (np.eye(3) - shape_gain) @ self._shape_cov[i]

                # update weight based on prior and predicted likelihood
                new_shape_weight[i*len(shape_est_weight) + j] = \
                    -2.5 * np.log(2 * np.pi) - 0.5 * np.log(np.linalg.det(shape_innov_cov)) \
                    - 0.5 * np.dot(np.dot(shape_innov, np.linalg.inv(shape_innov_cov)), shape_innov)
                new_shape_weight[i*len(shape_est_weight) + j] += prior_log_weights[i]

        new_shape_weight -= lse(new_shape_weight)
        self._shape_weight = np.exp(new_shape_weight)
        self._shape_state = new_shape_state
        self._shape_cov = new_shape_cov


def predict_est_if(est, cov, td, error_cov, error_mat):
    error_cov_inv = np.diag(1.0 / error_cov**2)
    tran_mat_inv = np.array([
        [1.0, 0.0, -td,  0.0],
        [0.0, 1.0, 0.0,  -td],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]) if len(est) == 4 else np.eye(3)
    tran_cov_inv = tran_mat_inv.T @ cov @ tran_mat_inv
    omega = tran_cov_inv @ error_mat @ np.linalg.inv(error_mat.T @ tran_cov_inv @ error_mat + error_cov_inv)

    cov_new = tran_cov_inv - omega @ error_mat.T @ tran_cov_inv
    est_new = (np.eye(len(est)) - omega @ error_mat.T) @ tran_mat_inv.T @ est

    return est_new, cov_new
