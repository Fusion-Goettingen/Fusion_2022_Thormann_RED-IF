import numpy as np

from matplotlib.patches import Ellipse

from constants import *


def plot_ellipse(gt, meas, ax):
    """
    Plot an ellipse along with measurements.
    :param gt:      The ellipse to be plotted
    :param meas:    Measurement points or direct meaurements
    :param ax:      Axis on which to plot
    """
    ell = Ellipse((gt[0], gt[1]), gt[5]*2.0, gt[6]*2.0, np.rad2deg(gt[4]), color='grey', zorder=1)
    ax.add_artist(ell)

    for n in range(len(meas)):
        # avoid tikzplotlib bug
        # if len(meas[n]) == 1:
        #     meas[n] = np.vstack([meas[n][0], meas[n][0]])
        if DIRECT:
            if len(meas[n]) > 0:
                ell = Ellipse((meas[n][0], meas[n][1]), meas[n][5] * 2.0, meas[n][6] * 2.0, np.rad2deg(meas[n][4]),
                              color='black', fill=False, zorder=1)
                ax.add_artist(ell)
        else:
            if len(meas[n]) > 1:
                ax.scatter(meas[n][:, 0], meas[n][:, 1], s=0.1, color='black', zorder=3)
            elif len(meas[n]) > 0:  # avoid tikz bug
                ax.scatter(np.array([meas[n][0, 0], meas[n][0, 0]]), np.array([meas[n][0, 1], meas[n][0, 1]]), s=0.1,
                           color='black', zorder=3)
