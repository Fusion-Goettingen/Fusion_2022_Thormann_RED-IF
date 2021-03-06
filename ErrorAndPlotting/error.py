import numpy as np
from scipy.linalg import sqrtm
from shapely.geometry import Polygon

from Filters.filtersupport import rot
from constants import *


def to_matrix(alpha, ax_l, ax_w, sr):
    """
    Turn ellipse parameters into a matrix or square root matrix depending on sr parameter
    :param alpha:   Orientation of the ellipse
    :param ax_l:    Semi-axis length of the ellipse
    :param ax_w:    Semi-axis width of the ellipse
    :param sr:      If True, square root matrix is calculated instead of shape matrix
    :return:        Shape or square root matrix depending of sr
    """
    p = 1 if sr else 2
    rot_m = rot(alpha)
    return np.dot(np.dot(rot_m, np.diag([ax_l, ax_w]) ** p), rot_m.T)


def gw_error(x, gt):
    """
    Calculates the squared Gaussian Wasserstein metric for two ellipses.
    :param x:   first ellipse, must be parameterized with center, orientation, and semi-axes
    :param gt:  second ellipse, must be parameterized with center, orientation, and semi-axes
    :return:    the squared Gaussian Wasserstein distance between the two ellipses
    """
    gt_sigma = to_matrix(gt[AL], gt[L], gt[W], False)
    gt_sigma += gt_sigma.T
    gt_sigma /= 2.0

    track_sigma = to_matrix(x[AL], x[L], x[W], False)
    track_sigma += track_sigma.T
    track_sigma /= 2.0

    error = np.linalg.norm(gt[[X1, X2]] - x[[X1, X2]]) ** 2 \
            + np.trace(gt_sigma + track_sigma - 2 * sqrtm(np.einsum('ab, bc, cd -> ad', sqrtm(gt_sigma), track_sigma,
                                                                    sqrtm(gt_sigma))))

    return error


def iou_error(x, gt):
    """
    Calculates intersection-over-union between the two ellipses.
    :param x:   first ellipse, must be parameterized with center, orientation, and semi-axes
    :param gt:  second ellipse, must be parameterized with center, orientation, and semi-axes
    :return:    the iou value between 0 and 1
    """
    # get points on ellipses
    theta = np.linspace(0.0, 2.0*np.pi, 100)
    x_points = x[[X1, X2], None] + rot(x[AL]) @ np.diag([x[L], x[W]]) @ np.array([np.cos(theta), np.sin(theta)])
    gt_points = gt[[X1, X2], None] + rot(gt[AL]) @ np.diag([gt[L], gt[W]]) @ np.array([np.cos(theta), np.sin(theta)])

    # create polygon
    x_pol = Polygon(x_points.T)
    gt_pol = Polygon(gt_points.T)

    # calculate IoU
    intersec = gt_pol.intersection(x_pol).area
    return intersec / (gt_pol.area + x_pol.area - intersec)
