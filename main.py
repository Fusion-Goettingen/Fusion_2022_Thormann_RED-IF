import os

import matplotlib.pyplot as plt
import tikzplotlib

from numpy.random import multivariate_normal as mvn

import time

from configs import get_configs
from Filters.memekfstar import MemEkfStarTracker
from Filters.randommatrix import RandomMatrix
from Filters.direct import DirectTracker
from Filters.fusioncenter import FusionCenter
from Filters.fusioncenterrm import FusionCenterRM
from Filters.filtersupport import to_matrix
from Data.simulation import simulate_data
from ErrorAndPlotting.plotting import plot_ellipse
from constants import *

# setup
_, ax = plt.subplots(1, 1)
init_state = mvn(INIT_STATE, INIT_STATE_COV)
init_state[L] = np.max([init_state[L], AX_MIN])
init_state[W] = np.max([init_state[W], AX_MIN])

# create folders if necessary
if not os.path.isdir('./simData'):
    os.mkdir('./simData')
if not os.path.isdir('./plots'):
    os.mkdir('./plots')
if not os.path.isdir('./plots/animation0'):
    os.mkdir('./plots/animation0')
if not os.path.isdir('./plots/animation1'):
    os.mkdir('./plots/animation1')

# for reproducibility
if LOAD_DATA:
    init_state = np.load('./simData/initState0.npy')
else:
    np.save('./simData/initState0.npy', init_state)

# tracker setup
config_memekfstar1, config_memekfstar2, config_fusion, config_fusion_red, config_fusion_if, config_fusion_if_red, \
config_rm1, config_rm2, config_fusion_rm, config_direct1, config_direct2 = get_configs(INIT_STATE, ax)

memekfstar1 = MemEkfStarTracker(MEM_H, MEM_KIN_DYM, MEM_SHAPE_DYM, MEAS_COV1, **config_memekfstar1)
memekfstar2 = MemEkfStarTracker(MEM_H, MEM_KIN_DYM, MEM_SHAPE_DYM, MEAS_COV2, **config_memekfstar2)

rm1 = RandomMatrix(**config_rm1)
rm2 = RandomMatrix(**config_rm2)

direct1 = DirectTracker(0, **config_direct1)
direct2 = DirectTracker(1, **config_direct2)

fusion_normal = FusionCenter(**config_fusion)
fusion_red = FusionCenter(**config_fusion_red)
fusion_normal_if = FusionCenter(**config_fusion_if)
fusion_red_if = FusionCenter(**config_fusion_if_red)

fusion_rm = FusionCenterRM(**config_fusion_rm)

time_normal = 0.0
time_red = 0.0
time_normal_if = 0.0
time_red_if = 0.0
time_rm = 0.0

for r in range(RUNS):
    print('Starting run %i of %i' % (r+1, RUNS))
    plot_cond = (r == RUNS-1)

    # generate data
    if DIRECT:
        simulator = simulate_data(init_state, DIRECT_MEAS_COV)
    else:
        simulator = simulate_data(init_state, MEAS_COV)

    # tracking
    step_id = 0
    for gt, meas in simulator:
        # print('Starting time step %i' % step_id)
        if LOAD_DATA:
            gt = np.load('./simData/gt%i-%i.npy' % (step_id, r))
            meas = np.load('./simData/meas%i-%i.npy' % (step_id, r))
        if step_id == 0:
            td = 0
        else:
            td = TD

        # run filters
        if DIRECT:
            direct1.step(meas[0].copy(), DIRECT_MEAS_COV[0], td, step_id, gt, plot_cond)
            direct2.step(meas[1].copy(), DIRECT_MEAS_COV[1], td, step_id, gt, plot_cond)

            est1, est_cov1 = direct1.get_est()
            est2, est_cov2 = direct2.get_est()
            rm_rate = 20 if SCENARIO_ID == 0 else 10
            est_rm1 = np.array([est1[:4], to_matrix(est1[AL], est1[L], est1[W], False) * rm_rate])
            est_rm_cov1 = np.array([est_cov1[:4, :4], rm_rate])
            est_rm2 = np.array([est2[:4], to_matrix(est2[AL], est2[L], est2[W], False) * rm_rate])
            est_rm_cov2 = np.array([est_cov2[:4, :4], rm_rate])
        else:
            memekfstar1.step(meas[0].copy(), MEAS_COV[0].copy(), td, step_id, gt, plot_cond)
            memekfstar2.step(meas[1].copy(), MEAS_COV[1].copy(), td, step_id, gt, plot_cond)
            rm1.step(meas[0].copy(), MEAS_COV[0].copy(), td, step_id, gt, plot_cond)
            rm2.step(meas[1].copy(), MEAS_COV[1].copy(), td, step_id, gt, plot_cond)

            est1, est_cov1, m_mat1 = memekfstar1.get_est()
            est2, est_cov2, m_mat2 = memekfstar2.get_est()
            est_rm1, est_rm_cov1 = rm1.get_est()
            est_rm2, est_rm_cov2 = rm2.get_est()

        if not LOAD_DATA:
            np.save('./simData/meas%i-%i.npy' % (step_id, r), meas)
            np.save('./simData/gt%i-%i.npy' % (step_id, r), gt)

        # fusion
        tic = time.perf_counter()
        fusion_normal.step(np.vstack([est1, est2]), np.stack([est_cov1, est_cov2]), td, step_id, gt, plot_cond)
        toc = time.perf_counter()
        time_normal += toc - tic

        tic = time.perf_counter()
        fusion_red.step(np.vstack([est1, est2]), np.stack([est_cov1, est_cov2]), td, step_id, gt, plot_cond)
        toc = time.perf_counter()
        time_red += toc - tic

        tic = time.perf_counter()
        fusion_normal_if.step(np.vstack([est1, est2]), np.stack([est_cov1, est_cov2]), td, step_id, gt, plot_cond)
        toc = time.perf_counter()
        time_normal_if += toc - tic

        tic = time.perf_counter()
        fusion_red_if.step(np.vstack([est1, est2]), np.stack([est_cov1, est_cov2]), td, step_id, gt, plot_cond)
        toc = time.perf_counter()
        time_red_if += toc - tic

        tic = time.perf_counter()
        fusion_rm.step(np.array([est_rm1, est_rm2]), np.array([est_rm_cov1, est_rm_cov2]), td, step_id, gt, plot_cond)
        toc = time.perf_counter()
        time_rm += toc - tic

        if plot_cond:
            plot_ellipse(gt, meas, ax)
            plt.axis([gt[0] - 5, gt[0] + 5, gt[1] - 5, gt[1] + 5])
            if NO_KIN:
                plt.xlabel('x in m')
                plt.ylabel('y in m')
                plt.savefig(('./plots/animation%i/sample' % SCENARIO_ID) + f'{step_id:03d}' + '.png')
                tikzplotlib.save(('./plots/animation%i/sample' % SCENARIO_ID) + f'{step_id:03d}' + '.tex',
                                 add_axis_environment=False)
                plt.cla()
        step_id += 1
    if LOAD_DATA:
        init_state = np.load('./simData/initState%i.npy' % (r+1))
    else:
        init_state = mvn(INIT_STATE, INIT_STATE_COV)
        np.save('./simData/initState%i.npy' % (r+1), init_state)
    if DIRECT:
        direct1.reset(INIT_STATE, INIT_STATE_COV)
        direct2.reset(INIT_STATE, INIT_STATE_COV)
    else:
        memekfstar1.reset(INIT_STATE, INIT_STATE_COV)
        memekfstar2.reset(INIT_STATE, INIT_STATE_COV)
        rm1.reset(INIT_STATE, INIT_STATE_COV)
        rm2.reset(INIT_STATE, INIT_STATE_COV)
    fusion_normal.reset(INIT_STATE, INIT_STATE_COV)
    fusion_red.reset(INIT_STATE, INIT_STATE_COV)
    fusion_normal_if.reset(INIT_STATE, INIT_STATE_COV)
    fusion_red_if.reset(INIT_STATE, INIT_STATE_COV)
    fusion_rm.reset(INIT_STATE, INIT_STATE_COV)

# time wrap up
time_normal /= TIME_STEPS*RUNS
time_red /= TIME_STEPS*RUNS
time_normal_if /= TIME_STEPS*RUNS
time_red_if /= TIME_STEPS*RUNS
time_rm /= TIME_STEPS*RUNS
print('Normal average time per step: %f' % time_normal)
print('RED average time per step: %f' % time_red)
print('Normal information form average time per step: %f' % time_normal_if)
print('RED information form average time per step: %f' % time_red_if)
print('Random Matrix average time per step: %f' % time_rm)

# example trajectory plotting
if not NO_KIN:
    plt.axis(AX_LIMS)
    if DIRECT:
        plt.plot([0], [0], color=direct1.get_color(), label=direct1.get_name())
        plt.plot([0], [0], color=direct2.get_color(), label=direct2.get_name())
    else:
        plt.plot([0], [0], color=memekfstar1.get_color(), label=memekfstar1.get_name())
        plt.plot([0], [0], color=memekfstar2.get_color(), label=memekfstar2.get_name())
        plt.plot([0], [0], color=rm1.get_color(), label=rm1.get_name())
        plt.plot([0], [0], color=rm2.get_color(), label=rm2.get_name())
    plt.plot([0], [0], color=fusion_normal.get_color(), label=fusion_normal.get_name())
    plt.plot([0], [0], color=fusion_red.get_color(), label=fusion_red.get_name())
    plt.plot([0], [0], color=fusion_normal_if.get_color(), label=fusion_normal_if.get_name())
    plt.plot([0], [0], color=fusion_red_if.get_color(), label=fusion_red_if.get_name())
    plt.plot([0], [0], color=fusion_rm.get_color(), label=fusion_rm.get_name())
    plt.xlabel('x in m')
    plt.ylabel('y in m')
    plt.gca().set_aspect('equal')
    plt.legend()
    tikzplotlib.save('./plots/examplerun.tex', add_axis_environment=False)
    plt.show()

# error wrap up
_, ax = plt.subplots(1, 1)
if DIRECT:
    direct1.plot_gw_error(ax, RUNS)
    direct2.plot_gw_error(ax, RUNS)
else:
    memekfstar1.plot_gw_error(ax, RUNS)
    memekfstar2.plot_gw_error(ax, RUNS)
    rm1.plot_gw_error(ax, RUNS)
    rm2.plot_gw_error(ax, RUNS)
fusion_normal.plot_gw_error(ax, RUNS)
fusion_red.plot_gw_error(ax, RUNS)
fusion_normal_if.plot_gw_error(ax, RUNS)
fusion_red_if.plot_gw_error(ax, RUNS)
fusion_rm.plot_gw_error(ax, RUNS)
plt.legend()
tikzplotlib.save('./plots/gw_error.tex', add_axis_environment=False)
plt.show()

_, ax = plt.subplots(1, 1)
if DIRECT:
    direct1.plot_iou_error(ax, RUNS)
    direct2.plot_iou_error(ax, RUNS)
else:
    memekfstar1.plot_iou_error(ax, RUNS)
    memekfstar2.plot_iou_error(ax, RUNS)
    rm1.plot_iou_error(ax, RUNS)
    rm2.plot_iou_error(ax, RUNS)
fusion_normal.plot_iou_error(ax, RUNS)
fusion_red.plot_iou_error(ax, RUNS)
fusion_normal_if.plot_iou_error(ax, RUNS)
fusion_red_if.plot_iou_error(ax, RUNS)
fusion_rm.plot_iou_error(ax, RUNS)
plt.legend()
tikzplotlib.save('./plots/iou_error.tex', add_axis_environment=False)
plt.show()

if not NO_KIN:
    _, ax = plt.subplots(1, 1)
    if DIRECT:
        direct1.plot_vel_error(ax, RUNS)
        direct2.plot_vel_error(ax, RUNS)
    else:
        memekfstar1.plot_vel_error(ax, RUNS)
        memekfstar2.plot_vel_error(ax, RUNS)
        rm1.plot_vel_error(ax, RUNS)
        rm2.plot_vel_error(ax, RUNS)
    fusion_normal.plot_vel_error(ax, RUNS)
    fusion_red.plot_vel_error(ax, RUNS)
    fusion_normal_if.plot_vel_error(ax, RUNS)
    fusion_red_if.plot_vel_error(ax, RUNS)
    fusion_rm.plot_vel_error(ax, RUNS)
    plt.legend()
    tikzplotlib.save('./plots/vel_error.tex', add_axis_environment=False)
    plt.show()
