import numpy as np
from numpy.random import multivariate_normal as mvn

from constants import *


def get_proc_cov(sigma_q, sigma_sh, td):
    """
    Calculate the process covariance based on kinematic and shape process noise.
    :param sigma_q:     standard deviation of kinematic process noise
    :param sigma_sh:    standard deviation of shape process noise
    :param td:          time difference between time steps
    :return:            the 7x7 process noise covariance
    """
    proc_cov = np.zeros((7, 7))

    error_mat = np.array([
        [0.5*td**2, 0.0],
        [0.0, 0.5*td**2],
        [td,        0.0],
        [0.0,        td],
    ])
    proc_cov[:4, :4] = np.dot(np.dot(error_mat, np.diag(sigma_q)**2), error_mat.T)

    proc_cov[4:, 4:] = td**2*np.diag(sigma_sh)**2

    return proc_cov


def rot(al):
    """
    Calculate rotation matrix. Can handle multiple rotation inputs.
    :param al:  rotation in radian, either scalar or an array
    :return:    rotation matrix corresponding to al or an nx2x2 array providing a rotation matrix for each al
    """
    rot = np.array([
        [np.cos(al), -np.sin(al)],
        [np.sin(al), np.cos(al)]
    ])
    if len(rot.shape) == 3:
        return rot.transpose((2, 0, 1))
    else:
        return rot


def turn_mult(x, cov, information=False):
    """
    Turn the input density into an RED approximation with 4 components by splitting it in the four different
    orientations possible between -pi and pi describing the same ellipse.
    :param x:           the input ellipse shape mean, parameterized with orientation and semi-axes
    :param cov:         the input covariance
    :param information: true if information form is used
    :return:            multimodal Gaussian density with 4 components, including means, covariances, and weights
    """
    x_mult = np.zeros((4, len(x)))
    cov_mult = np.zeros((4, len(x), len(x)))

    x_mult[0] = np.copy(x)
    cov_mult[0] = np.copy(cov)
    for i in range(1, 4):
        cov_mult[i] = np.copy(cov)
        cov_mult[i, 1, 0] = cov_mult[i - 1, 2, 0]
        cov_mult[i, 2, 0] = cov_mult[i - 1, 1, 0]
        cov_mult[i, 0, 1] = cov_mult[i - 1, 0, 2]
        cov_mult[i, 0, 2] = cov_mult[i - 1, 0, 1]
        cov_mult[i, 1, 1] = cov_mult[i - 1, 2, 2]
        cov_mult[i, 2, 2] = cov_mult[i - 1, 1, 1]

        x_mult[i, 1] = x_mult[i - 1, 2]
        x_mult[i, 2] = x_mult[i - 1, 1]

        if information:
            x_mult[i, 0] = x[0]
            x_mult[i] += i * 0.5 * np.pi * cov_mult[i, :, 0]
        else:
            x_mult[i, 0] = x[0] + i * 0.5 * np.pi
            # x_mult[i, 0] = (x_mult[i, 0] + np.pi) % (2 * np.pi) - np.pi  # handled by restricting innovation in fusion

    w = 0.25 * np.ones(4)

    return x_mult, cov_mult, w


def to_matrix(alpha, l, w, sr):
    """
    Turn ellipse parameters into a matrix or square root matrix.
    :param alpha:   Orientation of the ellipse
    :param l:       Semi-axis length of the ellipse
    :param w:       Semi-axis width of the ellipse
    :param sr:      If True, square root matrix is calculated instead of shape matrix
    :return:        Shape or square root matrix
    """
    p = 1 if sr else 2
    rot_mat = rot(alpha)
    if len(rot_mat.shape) == 3:
        lw_diag = np.array([np.diag([l[i], w[i]]) for i in range(len(l))])
        return np.einsum('xab, xbc, xdc -> xad', rot_mat, lw_diag ** p, rot_mat)
    else:
        return np.dot(np.dot(rot_mat, np.diag([l, w]) ** p), rot_mat.T)


def to_matrix_params(alpha, l, w, sr):
    """
    Turn ellipse parameters into a 3D vector containing the matrix or square root matrix elements
    parameter.
    :param alpha:   Orientation of the ellipse
    :param l:       Semi-axis length of the ellipse
    :param w:       Semi-axis width of the ellipse
    :param sr:      If True, square root matrix is calculated instead of shape matrix
    :return:        3D vector containing diagonal and corner of shape or square root matrix
    """
    if len(np.atleast_1d(alpha)) > 1:
        mats = to_matrix(alpha, l, w, sr)
        return np.array([mats[:, 0, 0], mats[:, 0, 1], mats[:, 1, 1]]).T
    else:
        mat = to_matrix(alpha, l, w, sr)
        return np.array([mat[0, 0], mat[0, 1], mat[1, 1]])


def get_ellipse_params(ell):
    """
    Calculate the ellipse semi-axis length and width and orientation based on shape matrix.
    :param ell: Input ellipse as 2x2 shape matrix
    :return:    Semi-axis length, width and orientation of input ellipse
    """
    ellipse_axis, v = np.linalg.eig(ell)
    ellipse_axis = np.sqrt(ellipse_axis)
    ax_l = ellipse_axis[0]
    ax_w = ellipse_axis[1]
    al = np.arctan2(v[1, 0], v[0, 0])

    return np.array([al, ax_l, ax_w])


def get_ellipse_params_from_sr(sr):
    """
    Calculate ellipse semi-axis length and width and orientation based on the elements of the square root matrix.
    :param sr:  Elements of the square root matrix [top-left, corner, bottom-right]
    :return:    Semi-axis length, width and orientation of input square root matrix
    """
    # calculate shape matrix based on square root matrix elements
    ell_sr = np.array([
        [sr[0], sr[1]],
        [sr[1], sr[2]],
    ])
    ell = np.dot(ell_sr, ell_sr)

    return get_ellipse_params(ell)


def mmgw_estimate_from_particles(particles):
    """
    Calculate the MMGW estimate of a particle density in ellipse parameter space.
    :param particles:   particle density in ellipse parameter space
    :return:            MMGW estimate in ellipse parameter space
    """
    particles = to_matrix_params(particles[:, 0], particles[:, 1], particles[:, 2], True)
    return get_ellipse_params_from_sr(np.mean(particles, axis=0))


def sample_mult(x, cov, w, n_samples):
    """
    Sample from a multimodal Gaussian density.
    :param x:           the components' means
    :param cov:         the components' covariances
    :param w:           the components' weights
    :param n_samples:   the number of samples to be drawn
    :return:            the samples
    """
    # sample the components with repetition
    chosen = np.random.choice(len(x), n_samples, True, p=w)

    # sample from the respective components
    samples = np.zeros((n_samples, len(x[0])))
    for i in range(len(x)):
        if np.sum(chosen == i) > 0:
            samples[chosen == i] = mvn(x[i], cov[i], np.sum(chosen == i))

    return samples


def reduce_mult_salmond(means, covs, w):
    """
    Reduce mixture density by removing unlikely components and merging close components based on
    Salmond, D. J. "Mixture reduction algorithms for point and extended object tracking in clutter." IEEE Transactions
    on Aerospace and Electronic Systems 45.2 (2009): 667-686.
    :param means:   set of means
    :param covs:    set of covariances
    :param w:       weights of the components
    :return:        reduced set of component means, covariances, and weights
    """
    if len(means.shape) == 2:
        x_dim = len(means[0])
    else:
        x_dim = len(means)
    means_merged, covs_merged, w_merged = means.copy(), covs.copy(), w.copy()

    # remove unlikely components
    keep = np.atleast_1d(w_merged > WEIGHT_THRESH)
    if not any(keep):  # keep most likely component as we assume the target always exists
        keep[np.argmax(w_merged)] = True
        means_merged, covs_merged, w_merged = means_merged[keep], covs_merged[keep], w_merged[keep]
        w_merged /= np.sum(w_merged)
        return means_merged, covs_merged, w_merged
    means_merged = means_merged[keep]
    covs_merged = covs_merged[keep]
    w_merged = w_merged[keep]
    w_merged /= np.sum(w_merged)

    num = len(w_merged)

    # merging
    close_thresh = CLOSE_THRESH
    while num > MAX_COMP:
        # cluster
        clusters = np.ones(num) * -1  # -1 means unclustered
        cluster_id = 0
        while any(clusters == -1):
            central_id = np.arange(num)[clusters == -1][np.argmax(w_merged[clusters == -1])]
            clusters[central_id] = cluster_id
            for i in np.arange(num)[clusters == -1]:
                dist = (w_merged[central_id]*w_merged[i] / (w_merged[central_id] + w_merged[i])) \
                       * ((means_merged[i] - means_merged[central_id]) @ np.linalg.inv(covs_merged[central_id])
                          @ (means_merged[i] - means_merged[central_id]))
                if dist < close_thresh:
                    clusters[i] = cluster_id
            cluster_id += 1

        # merge
        labels = np.arange(cluster_id)
        new_means_merged = np.zeros((cluster_id, x_dim))
        new_covs_merged = np.zeros((cluster_id, x_dim, x_dim))
        new_w_merged = np.zeros(cluster_id)
        for i in range(len(labels)):
            in_cluster = (clusters == i)
            if np.sum(in_cluster) > 1:
                new_w_merged[i] = np.sum(w_merged[in_cluster])
                new_means_merged[i] = np.sum(w_merged[in_cluster, None] * means_merged[in_cluster], axis=0) \
                                      / new_w_merged[i]
                new_covs_merged[i] = np.sum(w_merged[in_cluster, None, None]
                                        * (np.einsum('xa, xb -> xab', means_merged[in_cluster] - new_means_merged[i],
                                                     means_merged[in_cluster] - new_means_merged[i])
                                           + covs_merged[in_cluster]), axis=0) / new_w_merged[i]
            else:
                new_w_merged[i] = w_merged[in_cluster]
                new_means_merged[i] = means_merged[in_cluster]
                new_covs_merged[i] = covs_merged[in_cluster]
        new_w_merged /= np.sum(new_w_merged)

        means_merged = new_means_merged.copy()
        covs_merged = new_covs_merged.copy()
        w_merged = new_w_merged.copy()
        num = len(w_merged)

        close_thresh += 0.05

    return means_merged, covs_merged, w_merged
