import numpy as np
from numpy.random import multivariate_normal as mvn

from Filters.filtersupport import rot

from constants import *


def get_meas_sources(ellipse, n_m):
    """
    Sample measurement sources uniformly from surface of an ellipse.
    :param ellipse:     array parameterized with x, y, alpha, a, b
    :param n_m:         number of measurement sources to be sampled
    :return:            n_mx2 array of measurement sources
    """
    meas_sources = np.zeros((n_m, 2))

    for i in range(n_m):
        meas_sources[i] = np.random.uniform(-1.0, 1.0, 2)
        while not (np.dot(meas_sources[i], meas_sources[i]) <= 1.0):
            meas_sources[i] = np.random.uniform(-1.0, 1.0, 2)

    return np.dot(np.dot(meas_sources, np.diag(ellipse[3:5])), rot(ellipse[2]).T) + ellipse[:2]


def simulate_data(init, meas_cov):
    """
    Simulate trajectory and measurements.
    :param init:        initial state parameterized as x, y, vx, vy, alpha, a, b
    :param meas_cov:    measurement noise covariances as nx2x2 array with n being number of sensors
    :return:            for each time step, yields ground truth in same shape as init as well as list of nx2 measurement
                        point clouds with n>=1 (or direct measurement if specified) for each sensor or no measurements
                        for the first time step
    """
    gt = init.copy()
    meas_cov_3d = np.atleast_3d(meas_cov)
    n_sensors = len(meas_cov_3d)

    for i in range(TIME_STEPS):
        # move along predetermined path
        if i > 0:
            gt[2] += 0.0 if NO_KIN else np.random.normal(0, SIGMA_V1**2) * TD
            gt[3] += 0.0 if NO_KIN else np.random.normal(0, SIGMA_V2**2) * TD
            gt[4] += np.random.normal(0, SIGMA_OR**2) * TD
            if not NO_KIN:
                gt[:2] = gt[:2] + gt[2:4] * TD

            # generate measurements
            meas = []
            for n in range(n_sensors):
                if DIRECT:
                    meas.append(mvn(gt, meas_cov[n]))
                else:
                    n_m = np.max([np.random.poisson(GT_POIS, 1)[0], 1])  # at least one measurement
                    meas_sources = get_meas_sources(gt[[0, 1, 4, 5, 6]], n_m)
                    meas.append(np.asarray([mvn(meas_sources[j], meas_cov_3d[n]) for j in range(n_m)]))
        else:
            if DIRECT:
                meas = [np.zeros(0), np.zeros(0)]
            else:
                meas = [np.zeros((0, 2)), np.zeros((0, 2))]

        yield gt, meas
