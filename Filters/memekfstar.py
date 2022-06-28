import numpy as np
from numpy.random import multivariate_normal as mvn
from scipy.linalg import block_diag

from matplotlib.patches import Ellipse

from Filters.basefilters import ExtendedObjectFilter
from Filters.filtersupport import rot, get_proc_cov
from constants import *


class MemEkfStarTracker(ExtendedObjectFilter):
    """
    MEM-EKF* filter.
    Modified for paper, original code author: Shishan Yang
    Based on:
    Tracking the Orientation and Axes Lengths of an Elliptical Extended Object
    S. Yang and M. Baum
    IEEE Transactions on Signal Processing, vol. 67, no. 18, Sep. 2019.
    """
    def __init__(self, h_matrix, dynamic_matrix_kin, dynamic_matrix_extent, measurement_noise_cov, **kwargs):
        super().__init__(**kwargs)
        init_cov = kwargs.get('init_cov').copy()
        self.error = []
        self.name = 'MEM-EKF*'
        # parameters
        self.h_matrix = h_matrix
        self.dynamic_matrix_kin = dynamic_matrix_kin
        self.dynamic_matrix_extent = dynamic_matrix_extent
        self.measurement_noise_cov = measurement_noise_cov
        self.process_noise_cov_kin = kwargs.get('Q')
        self.process_noise_cov_extent = kwargs.get('SH')
        self.multiplicative_noise_cov = np.diag([.25, .25])

        self._last_m = np.eye(3)

        # introduce ambiguity
        if SIMULATE_AMBIGUITY & (measurement_noise_cov[1, 1] > measurement_noise_cov[0, 0]):
            save_l = self._est[L]
            self._est[L] = self._est[W]
            self._est[W] = save_l
            self._est[AL] += 0.5 * np.pi
            save_cov_l = init_cov[L].copy()
            init_cov[L] = init_cov[W].copy()
            init_cov[W] = save_cov_l.copy()
            save_cov_l = init_cov[:, L].copy()
            init_cov[:, L] = init_cov[:, W].copy()
            init_cov[:, W] = save_cov_l.copy()

            self._ambiguity = True
        else:
            self._ambiguity = False

        self.mem_est = MemEstimate(self._est[[X1, X2]], self._est[AL], self._est[[L, W]],
                                   init_cov[:4, :4], init_cov[4:, 4:],
                                   velocity=self._est[[V1, V2]], color=self._color)
        self._est_cov = init_cov.copy()
        self._al_approx = kwargs.get('al_approx')  # approximate shape orientation via velocity vector

        self._mmgw = kwargs.get('mmgw')

    def __repr__(self):
        return "MemEkfStarTracker({},{},{},{},{})".format(self.h_matrix, self.dynamic_matrix_kin,
                                                          self.dynamic_matrix_extent, self.measurement_noise_cov,
                                                          self.process_noise_cov_kin, self.process_noise_cov_extent)

    def __str__(self):
        return "MEM-EKF* tracker, with H = {}, Ar = {}, Ap = {}, Cv = {}, Cw_r = {}, Cw_p = {}".format(self.h_matrix,
                                                                                                       self.dynamic_matrix_kin,
                                                                                                       self.dynamic_matrix_extent,
                                                                                                       self.measurement_noise_cov,
                                                                                                       self.process_noise_cov_kin,
                                                                                                       self.process_noise_cov_extent)

    def reset(self, init_est, init_cov):
        self._est = init_est.copy()
        self._est_cov = init_cov.copy()

        # introduce ambiguity
        if self._ambiguity:
            init_cov_mod = init_cov.copy()
            save_l = self._est[L]
            self._est[L] = self._est[W]
            self._est[W] = save_l
            self._est[AL] += 0.5 * np.pi
            save_cov_l = init_cov_mod[L].copy()
            init_cov_mod[L] = init_cov_mod[W].copy()
            init_cov_mod[W] = save_cov_l.copy()
            save_cov_l = init_cov_mod[:, L].copy()
            init_cov_mod[:, L] = init_cov_mod[:, W].copy()
            init_cov_mod[:, W] = save_cov_l.copy()
            self._est_cov = init_cov_mod

            self.mem_est = MemEstimate(init_est[[X1, X2]].copy(), init_est[AL] + 0.5*np.pi, init_est[[W, L]].copy(),
                                       init_cov_mod[:4, :4], init_cov_mod[4:, 4:], velocity=init_est[[V1, V2]].copy(),
                                       color=self.mem_est.color)
        else:
            self.mem_est = MemEstimate(init_est[[X1, X2]].copy(), init_est[AL], init_est[[L, W]].copy(),
                                       init_cov[:4, :4].copy(), init_cov[4:, 4:].copy(),
                                       velocity=init_est[[V1, V2]].copy(), color=self.mem_est.color)

    def get_est(self):
        return self._est, self._est_cov, self._last_m

    def get_aux_variables(self, esti):
        """
        Create variables for correct step.
        :param esti:    Current estimate
        :return:        Parts of noise covariance, f matrices, and M matrix
        """

        alpha, l1, l2 = esti.extent
        if self._al_approx:
            alpha = np.arctan2(esti.velocity[1], esti.velocity[0])
        sin = np.sin(alpha)
        cos = np.cos(alpha)

        s = np.array([[cos, -sin], [sin, cos]]) @ np.diag([l1, l2])
        s1 = np.array([s[0]])
        s2 = np.array([s[1]])

        j1 = np.array([[-l1 * sin, cos, 0], [-l2 * cos, 0, -sin]])
        j2 = np.array([[l1 * cos, sin, 0], [-l2 * sin, 0, cos]])
        if self._al_approx:
            j1[:, 0] *= 0.0
            j2[:, 0] *= 0.0

        c1 = s @ self.multiplicative_noise_cov @ s.T
        e11 = np.trace(esti.covariance_extent @ j1.T @ self.multiplicative_noise_cov @ j1)
        e12 = np.trace(esti.covariance_extent @ j2.T @ self.multiplicative_noise_cov @ j1)
        e22 = np.trace(esti.covariance_extent @ j2.T @ self.multiplicative_noise_cov @ j2)
        c2 = np.array([[e11, e12], [e12, e22]])

        f_matrix = np.array([[1, 0, 0, 0],
                             [0, 0, 0, 1],
                             [0, 1, 0, 0]])
        f_dash_matrix = np.array([[1, 0, 0, 0],
                                  [0, 0, 0, 1],
                                  [0, 0, 1, 0]])

        m_matrix = np.vstack((2 * s1 @ self.multiplicative_noise_cov @ j1,
                              2 * s2 @ self.multiplicative_noise_cov @ j2,
                              s1 @ self.multiplicative_noise_cov @ j2 + s2 @ self.multiplicative_noise_cov @ j1))

        return c1, c2, f_matrix, f_dash_matrix, m_matrix

    def correct(self, measurements, meas_cov):
        """
        Correct step, processing measurements sequentially
        :param measurements:    Batch of measurements
        :param meas_cov:        Measurement covariance
        """
        self.measurement_noise_cov = meas_cov
        esti = self.mem_est
        for i, y in enumerate(measurements):
            extent_esti = esti.extent.reshape((-1, 1))
            kin_esti = esti.kinematics.reshape((-1, 1))
            cov_r = esti.covariance_kin.T
            cov_p = esti.covariance_extent.T
            c1, c2, f_matrix, f_dash_matrix, m_matrix = self.get_aux_variables(esti)
            y = np.asarray(y)[:, np.newaxis]

            # calculate moments for the kinematic state update
            y_dash = self.h_matrix @ kin_esti
            cov_ry = cov_r @ self.h_matrix.T
            cov_y = self.h_matrix @ cov_r @ self.h_matrix.T + c1 + c2 + self.measurement_noise_cov

            # update kinematic state
            y_dif = y - y_dash
            cov_y_inv = np.linalg.inv(cov_y)
            kin_esti = kin_esti + cov_ry @ cov_y_inv @ y_dif
            cov_r = cov_r - cov_ry @ cov_y_inv @ cov_ry.T
            cov_r = (cov_r + cov_r.T) / 2  # enforces symmetry of the covariance

            # construct pseudo-measurement for the extent update
            pseudo_meas = f_matrix @ np.kron(y_dif, y_dif)
            # calculate moments for the extent update
            pseudo_meas_xpctn = f_matrix @ cov_y.flatten()[:, np.newaxis]
            cov_extent_pseudo_meas = cov_p @ m_matrix.T
            cov_pseudo_meas = f_matrix @ np.kron(cov_y, cov_y) @ (f_matrix + f_dash_matrix).T

            # update extent
            inv_cov_pseudo = np.linalg.inv(cov_pseudo_meas)

            extent_esti = extent_esti + cov_extent_pseudo_meas @ inv_cov_pseudo @ (pseudo_meas - pseudo_meas_xpctn)

            cov_p = cov_p - cov_extent_pseudo_meas @ inv_cov_pseudo @ cov_extent_pseudo_meas.T
            cov_p = (cov_p + cov_p.T) / 2  # enforces symmetry of the covariance

            # get current estimate
            center = ((self.h_matrix @ kin_esti)[:, 0]).tolist()

            orientation, *semi_axes_length = extent_esti[:, 0]
            if self._al_approx:
                orientation = np.arctan2(kin_esti[3, 0], kin_esti[2, 0])

            esti = MemEstimate(center, orientation, semi_axes_length, cov_r, cov_p,
                               velocity=kin_esti[2:4, 0].tolist(), color=self.mem_est.color)

            if i == (len(measurements)-1):
                self._last_m = m_matrix.copy()

        if len(measurements) > 0:
            center = ((self.h_matrix @ kin_esti)[:, 0]).tolist()

            orientation, *semi_axes_length = extent_esti[:, 0]
            if self._al_approx:
                orientation = np.arctan2(kin_esti[3, 0], kin_esti[2, 0])

            updated_estimate = MemEstimate(center, orientation, semi_axes_length, cov_r, cov_p,
                                           velocity=kin_esti[2:4, 0].tolist(), color=self.mem_est.color)

            self.mem_est = updated_estimate

        self._est = self.mem_est.get_estimate(self._mmgw)
        self._est_cov = self.mem_est.get_estimate_covariance()

    def predict(self, td):
        """
        Prediction step.
        :param td:  Time difference
        """
        esti = self.mem_est
        extent_esti = esti.extent[:, np.newaxis]
        kin_esti = esti.kinematics[:, np.newaxis]
        cov_r = esti.covariance_kin
        cov_p = esti.covariance_extent
        # constant velocity model for now

        # get process noise
        proc_noise = get_proc_cov(self.process_noise_cov_kin, self.process_noise_cov_extent, td)

        if not NO_KIN:
            kin_esti = self.dynamic_matrix_kin @ kin_esti
            cov_r = self.dynamic_matrix_kin @ cov_r @ self.dynamic_matrix_kin.T + proc_noise[:4, :4]

        extent_esti = self.dynamic_matrix_extent @ extent_esti
        cov_p = self.dynamic_matrix_extent @ cov_p @ self.dynamic_matrix_extent.T + proc_noise[4:, 4:]
        self.mem_est = MemEstimate(kin_esti[0:2, 0], extent_esti[0, 0], extent_esti[1:3, 0], cov_r, cov_p,
                                   velocity=kin_esti[2:4, 0].tolist(), color=self.mem_est.color)

    def plotting(self):
        """
        Plot the estimate.
        """
        self.mem_est.plotting(self._ax, self._name, self._mmgw)


class ExtendedObject:
    """
    Base class used for MEM-EKF* estimate.
    """
    def __init__(self, center, orientation, semi_axes_length, **kwargs):
        '''
        center could be a list or tuple
        semi_axes_length is a list
        orientation is in radian (float)
        '''
        self.center = center
        self.kinematics = self.center
        self.orientation = orientation
        self.semi_axes_length = semi_axes_length
        self.rotation_matrix = np.array(
            [[np.cos(orientation), -np.sin(orientation)],
             [np.sin(orientation), np.cos(orientation)]])
        self.velocity = kwargs.get('velocity')
        self.kinematics = np.hstack((self.kinematics, self.velocity))
        self.extent = np.array([self.orientation, self.semi_axes_length[0], self.semi_axes_length[1]])

    def __repr__(self):
        return 'EllipticalObject({}, {}, {})'.format(self.center, self.orientation, self.semi_axes_length)

    def __str__(self):
        return "Elliptical Object with center {}, orientation {}\N{DEGREE SIGN} and semi-axes length {}" \
            .format(self.center, np.degrees(self.orientation), self.semi_axes_length)

    def plotting(self, ax, mmgw=False, **kwargs):

        esti = self.get_estimate(mmgw)
        ellip = Ellipse(esti[:2], 2 * esti[5], 2 * esti[6], np.rad2deg(esti[4]), **kwargs)
        ax.add_patch(ellip)

    @property
    def extent_vector(self):
        return [self.center[0], self.center[1], self.orientation, self.semi_axes_length[0], self.semi_axes_length[1]]

    def get_estimate(self, mmgw=False):
        return np.array([self.center[0], self.center[1], self.velocity[0], self.velocity[1], self.orientation,
                         self.semi_axes_length[0], self.semi_axes_length[1]])


class MemEstimate(ExtendedObject):
    """
    Author: Shishan Yang
    Class for a MEM-EKF* estimate.
    """
    def __init__(self, center, orientation, semi_axes_length, covariance_kin, covariance_extent, **kwargs):
        super().__init__(center, orientation, semi_axes_length, **kwargs)
        self.covariance_kin = covariance_kin
        self.covariance_extent = covariance_extent
        self.color = kwargs.get('color')
        self.exist_prob = kwargs.get('exist_prob')

    def __repr__(self):
        return "MemEstimate({},{},{},{},{})".format(self.center, self.orientation,
                                                    self.semi_axes_length, self.covariance_kin,
                                                    self.covariance_extent)

    def __str__(self):
        pass

    def get_estimate_covariance(self):
        return block_diag(self.covariance_kin, self.covariance_extent)

    def get_estimate(self, mmgw=False):
        """
        Modifications for MMGW by Kolja Thormann
        :param mmgw:
        :return:
        """
        if mmgw:
            samples = mvn(np.array([self.orientation, self.semi_axes_length[0], self.semi_axes_length[1]]),
                          self.covariance_extent, 1000)
            sample_mats = np.array([rot(samples[i, 0]) @ np.diag(samples[i, 1:3]) @ rot(samples[i, 0]).T
                                    for i in range(1000)])
            shape_mat = np.mean(sample_mats, axis=0)
            ellipse_axis, v = np.linalg.eig(shape_mat)
            ax_l = ellipse_axis[0]
            ax_w = ellipse_axis[1]
            al = np.arctan2(v[1, 0], v[0, 0])
            return np.array([self.center[0], self.center[1], self.velocity[0], self.velocity[1], al, ax_l, ax_w])
        else:
            return np.array([self.center[0], self.center[1], self.velocity[0], self.velocity[1], self.orientation,
                             self.semi_axes_length[0], self.semi_axes_length[1]])

    def plotting(self, ax, name, mmgw=False, **kwargs):
        super().plotting(ax, mmgw=mmgw, facecolor='None', edgecolor=self.color, zorder=2)
