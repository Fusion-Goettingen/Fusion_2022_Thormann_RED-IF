import matplotlib.pyplot as plt
import tikzplotlib

from numpy.random import multivariate_normal as mvn

import time

from configs import get_configs
from Filters.memekfstar import MemEkfStarTracker
from Data.simulation import simulate_data
from ErrorAndPlotting.plotting import plot_ellipse
from constants import *

# setup
_, ax = plt.subplots(1, 1)
init_state = mvn(INIT_STATE, INIT_STATE_COV)
init_state[L] = np.max([init_state[L], AX_MIN])
init_state[W] = np.max([init_state[W], AX_MIN])

# for reproducibility
if LOAD_DATA:
    init_state = np.load('./simData/initState0.npy')
else:
    np.save('./simData/initState0.npy', init_state)

# tracker setup
config_memekfstar, config_memekfstar_mmgw = get_configs(INIT_STATE, ax)

memekfstar = MemEkfStarTracker(MEM_H, MEM_KIN_DYM, MEM_SHAPE_DYM, MEAS_COV, **config_memekfstar)
memekfstar_mmgw = MemEkfStarTracker(MEM_H, MEM_KIN_DYM, MEM_SHAPE_DYM, MEAS_COV, **config_memekfstar_mmgw)

# timing (note that not all algorithms have been optimized)
memekfstar_time = 0.0
memekfstar_mmgw_time = 0.0

for r in range(RUNS):
    print('Starting run %i of %i' % (r+1, RUNS))
    plot_cond = (r == RUNS-1)
    # generate data
    simulator = simulate_data(init_state, MEAS_COV)

    # run filters
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
        tic = time.perf_counter()
        memekfstar.step(meas.copy(), MEAS_COV.copy(), td, step_id, gt, plot_cond)
        toc = time.perf_counter()
        memekfstar_time += toc - tic

        tic = time.perf_counter()
        memekfstar_mmgw.step(meas.copy(), MEAS_COV.copy(), td, step_id, gt, plot_cond)
        toc = time.perf_counter()
        memekfstar_mmgw_time += toc - tic

        if not LOAD_DATA:
            np.save('./simData/meas%i-%i.npy' % (step_id, r), meas)
            np.save('./simData/gt%i-%i.npy' % (step_id, r), gt)

        if plot_cond:
            plot_ellipse(gt, meas, ax)
            plt.axis([gt[0] - 5, gt[0] + 5, gt[1] - 5, gt[1] + 5])
            plt.plot([0], [0], color=memekfstar.get_color(), label=memekfstar.get_name())
            plt.plot([0], [0], color=memekfstar_mmgw.get_color(), label=memekfstar_mmgw.get_name())
            plt.xlabel('x in m')
            plt.ylabel('y in m')
            # plt.legend()
            plt.savefig(('./plots/animation%i/sample' % SCENARIO_ID) + f'{step_id:03d}' + '.png')
            plt.cla()
        step_id += 1
    if LOAD_DATA:
        init_state = np.load('./simData/initState%i.npy' % (r+1))
    else:
        init_state = mvn(INIT_STATE, INIT_STATE_COV)
        np.save('./simData/initState%i.npy' % (r+1), init_state)
    memekfstar.reset(INIT_STATE, INIT_STATE_COV)
    memekfstar_mmgw.reset(INIT_STATE, INIT_STATE_COV)

# example trajectory plotting
plt.axis(AX_LIMS)
plt.plot([0], [0], color=memekfstar.get_color(), label=memekfstar.get_name())
plt.plot([0], [0], color=memekfstar_mmgw.get_color(), label=memekfstar_mmgw.get_name())
plt.legend()
tikzplotlib.save('./plots/examplerun.tex', add_axis_environment=False)
plt.show()

# time wrap up
memekfstar_time /= TIME_STEPS*RUNS
memekfstar_mmgw_time /= TIME_STEPS*RUNS

ticks = [memekfstar.get_name(), memekfstar_mmgw.get_name()]
colors = [memekfstar.get_color(), memekfstar_mmgw.get_color()]
runtimes = [memekfstar_time, memekfstar_mmgw_time]
bars = np.arange(1, len(ticks)+1, 1)
for i in range(len(ticks)):
    plt.bar(bars[i], runtimes[i], width=0.5, color=colors[i], label=ticks[i], align='center')
# plt.xticks(bars, ticks)
plt.legend()
tikzplotlib.save('./plots/runtimes.tex', add_axis_environment=False)
plt.savefig('./plots/runtimes.svg')
plt.show()

# error wrap up
_, ax = plt.subplots(1, 1)
memekfstar.plot_gw_error(ax, RUNS)
memekfstar_mmgw.plot_gw_error(ax, RUNS)
plt.legend()
tikzplotlib.save('./plots/gw_error.tex', add_axis_environment=False)
plt.show()

_, ax = plt.subplots(1, 1)
memekfstar.plot_vel_error(ax, RUNS)
memekfstar_mmgw.plot_vel_error(ax, RUNS)
plt.legend()
tikzplotlib.save('./plots/vel_error.tex', add_axis_environment=False)
plt.show()
