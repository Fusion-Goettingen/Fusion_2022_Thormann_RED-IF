import numpy as np
from numpy.random import multivariate_normal as mvn

from constants import *


def rot(al):
    return np.array([
        [np.cos(al), -np.sin(al)],
        [np.sin(al),  np.cos(al)],
    ])


def get_meas_sources(ellipse, n_m):
    meas_sources = np.zeros((n_m, 2))

    for i in range(n_m):
        meas_sources[i] = np.random.uniform(-1.0, 1.0, 2)
        while not (np.dot(meas_sources[i], meas_sources[i]) <= 1.0):
            meas_sources[i] = np.random.uniform(-1.0, 1.0, 2)

    return np.dot(np.dot(meas_sources, np.diag(ellipse[3:5])), rot(ellipse[2]).T) + ellipse[:2]


def simulate_data(init, meas_cov):
    gt = init.copy()

    for i in range(TIME_STEPS):
        # move along predetermined path
        if i > 0:
            gt[2] += np.random.normal(0, SIGMA_V1**2) * TD
            gt[3] += np.random.normal(0, SIGMA_V2**2) * TD
            gt[4] += np.random.normal(0, SIGMA_OR**2) * TD
            gt[:2] = gt[:2] + gt[2:4] * TD

        # generate measurements
        n_m = np.max([np.random.poisson(GT_POIS, 1)[0], 1])  # at least one measurement
        meas_sources = get_meas_sources(gt[[0, 1, 4, 5, 6]], n_m)
        meas = np.asarray([mvn(meas_sources[j], meas_cov) for j in range(n_m)])

        yield gt, meas
