# code is adapted from https://brian2.readthedocs.io/en/stable/examples/frompapers.Brunel_Wang_2001.html

from utils.methods_utils import ibnn_utils, shared_utils, turbo_utils, baseline_bo_utils
from utils.sim_utils import set_params_utils, eqs_utils, plotting_utils, obj_func_utils, set_param_space
from brian2 import *
from botorch.utils.transforms import unnormalize
from botorch.utils.sampling import draw_sobol_samples
from botorch.test_functions import Ackley
from botorch.optim.optimize import optimize_acqf as optimize_acqf_fn
from botorch.optim import optimize_acqf
from botorch.models.transforms.outcome import Standardize
from botorch.models.kernels import InfiniteWidthBNNKernel
from botorch.models.gp_regression import SingleTaskGP as STGP
from botorch.models import SingleTaskGP
from botorch.generation import MaxPosteriorSampling
from botorch.fit import fit_gpytorch_mll
from botorch.exceptions import BadInitialCandidatesWarning
from botorch.acquisition import LogExpectedImprovement, qExpectedImprovement, qLogExpectedImprovement
from botorch import manual_seed
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.kernels import MaternKernel, RBFKernel, ScaleKernel
from gpytorch.constraints import Interval
import gpytorch
from torch.quasirandom import SobolEngine
from torch import nn
import torch
from skopt.plots import plot_gaussian_process, plot_convergence
from skopt.space import Real
from skopt import gp_minimize
import matplotlib.animation as animation
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
from functools import partial
from dataclasses import dataclass
import warnings
import time
import math
import os
import sys

# Verify the current directory
print("Current directory:", os.getcwd())

# --- Environment settings ---


np.random.seed(237)

warnings.filterwarnings("ignore", category=BadInitialCandidatesWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
torch.set_printoptions(sci_mode=False)

# --- run initial simulation ---
# populations
N = 700
N_E = int(N * 0.8)  # pyramidal neurons
N_I = int(N * 0.2)  # interneurons
f = 0.1
p = 3
C_ext = 800
DC_amp = 0

DC_input_ts = 1 * ms
sino_input_ts = 0.1 * ms

# external stimuli
# rate = 3 * Hz # in external noise
currents_to_track = ['I_syn', 'I_AMPA_ext', 'I_GABA_rec',
                     'I_AMPA_rec', 'I_NMDA_rec', 'I_DC1']
currents_to_plot = currents_to_track

# set direct currents
DC_input1 = set_params_utils.set_DC_input()
# DC_input2 = set_params_utils.set_DC_input()


# run initial simulation
start_scope()

N_sub = int(N_E * f)
N_non = int(N_E * (1. - f * p))


E_neuron_index = [0]  # index of the neuron in the population
# map the index in the monitor to population name
E_index_map = {0: 'nonselective'}
for i in range(p):
    E_neuron_index.append(N_non + i * N_sub)
    E_index_map[i+1] = f'selective {i}'


# voltage
V_L, V_thr, V_reset, V_E, V_I = set_params_utils.set_voltage()
# membrane capacitance and membrane leak
C_m_E, C_m_I, g_m_E, g_m_I = set_params_utils.set_membrane_params()

# AMPA (excitatory)
g_AMPA_ext_E, g_AMPA_rec_E, g_AMPA_ext_I, g_AMPA_rec_I, tau_AMPA = set_params_utils.set_AMPA_params(
    N_E)
# NMDA (excitatory)
g_NMDA_E, g_NMDA_I, tau_NMDA_rise, tau_NMDA_decay, alpha, Mg2 = set_params_utils.set_NMDA_params(
    N_E)
# GABAergic (inhibitory)
g_GABA_E, g_GABA_I, tau_GABA = set_params_utils.set_GABA_params(N_I)

# Write the equations for the target population (e.g., excitatory population P_E)
eqs_E = eqs_utils.write_eqs_E()
eqs_I = eqs_utils.write_eqs_I()
eqs_glut, eqs_pre_glut, eqs_pre_gaba = eqs_utils.write_other_eqs()

# neuron groups
P_E, P_I = set_params_utils.set_neuron_groups(N_E, N_I, eqs_E, eqs_I, V_L)
# synapses
external_noise_rate = 3 * Hz
C_E, C_I, C_E_E, C_E_I, C_I_I, C_I_E, C_P_E, C_P_I = set_params_utils.set_synapses(
    P_E, P_I, N_E, N_I, N_sub, N_non, p, f, C_ext, external_noise_rate, eqs_glut, eqs_pre_glut, eqs_pre_gaba)


N_activity_plot = 15
current_monitor_E, r_E_sels, r_E, r_I = set_params_utils.set_monitors_for_optimization_algorithm(
    N_activity_plot, N_non, N_sub, p, P_E, P_I, E_neuron_index=E_neuron_index, currents_to_track=currents_to_track)

# set external stimuli
# at 1s, select population 1
C_selection = int(f * C_ext)
rate_selection = 25 * Hz


stimuli1 = TimedArray(
    np.r_[np.zeros(40), np.ones(2), np.zeros(100)], dt=25 * ms)
input1 = PoissonInput(P_E[N_non:N_non + N_sub], 's_AMPA_ext',
                      C_selection, rate_selection, 'stimuli1(t)')

# # at 2s, select population 2
# stimuli2 = TimedArray(np.r_[np.zeros(80), np.ones(2), np.zeros(100)], dt=25 * ms)
# input2 = PoissonInput(P_E[N_non + N_sub:N_non + 2 * N_sub], 's_AMPA_ext', C_selection, rate_selection, 'stimuli2(t)')


# simulate, can be long >120s
net = Network(collect())
net.add(r_E_sels)
net.add(P_E, P_I, C_E_E, C_E_I, C_I_I, C_I_E, C_P_E, C_P_I)


net.store('initial')

net.run(8.1 * second, report='stdout')

plotting_utils.plot_firing_rate(r_E, r_I, r_E_sels)
plotting_utils.plot_currents(
    current_monitor_E, None, currents_to_plot, E_index_map)


# --- set objective function ---
space = set_param_space.space
space_norm = [Real(0, 1, name=dim.name) for dim in space]
stim_bounds = torch.tensor([[dim.low, dim.high]
                           for dim in space], dtype=torch.float).T
stim_bounds_norm = torch.tensor([[0], [1]]).repeat(
    1, stim_bounds.shape[1]).float()
namespace = {'V_L': V_L, 'V_thr': V_thr, 'V_reset': V_reset, 'V_E': V_E, 'V_I': V_I,
             'C_m_E': C_m_E, 'C_m_I': C_m_I, 'g_m_E': g_m_E, 'g_m_I': g_m_I,
             'g_AMPA_ext_E': g_AMPA_ext_E, 'g_AMPA_rec_E': g_AMPA_rec_E, 'g_AMPA_ext_I': g_AMPA_ext_I,
             'g_AMPA_rec_I': g_AMPA_rec_I, 'tau_AMPA': tau_AMPA,
             'g_NMDA_E': g_NMDA_E, 'g_NMDA_I': g_NMDA_I,
             'tau_NMDA_rise': tau_NMDA_rise, 'tau_NMDA_decay': tau_NMDA_decay,
             'alpha': alpha, 'Mg2': Mg2,
             'g_GABA_E': g_GABA_E, 'g_GABA_I': g_GABA_I, 'tau_GABA': tau_GABA,
             'stimuli1': stimuli1,
             # 'stimuli2': stimuli2,
             'DC_input1': DC_input1,
             }

objective_with_factor = partial(obj_func_utils.objective_function, net=net, namespace=namespace, stim_bounds=stim_bounds, current_monitor_E=current_monitor_E, r_E=r_E, r_I=r_I, r_E_sels=r_E_sels,
                                E_index_map=E_index_map, maximize=False)


objective_func_tensor = partial(obj_func_utils.objective_function, net=net, namespace=namespace, stim_bounds=stim_bounds, current_monitor_E=current_monitor_E, r_E=r_E, r_I=r_I, r_E_sels=r_E_sels,
                                process_input_func=obj_func_utils.process_input_tensor, process_output_func=obj_func_utils.process_output_tensor, E_index_map=E_index_map,
                                maximize=True)

objective_func_tensor_w_plot = partial(obj_func_utils.objective_function, net=net, namespace=namespace, stim_bounds=stim_bounds, current_monitor_E=current_monitor_E, r_E=r_E, r_I=r_I, r_E_sels=r_E_sels,
                                       process_input_func=obj_func_utils.process_input_tensor, process_output_func=obj_func_utils.process_output_tensor, E_index_map=E_index_map,
                                       maximize=True, plotting=True)
