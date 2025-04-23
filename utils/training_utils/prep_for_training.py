# --- System path setup ---
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


def get_init_x_and_y(result_dir='all_stored_results/init_points'):
    """
    Get the initial x and y values from the most recent result subfolder.
    """
    latest_folder = shared_utils.get_latest_result_subfolder_name(
        result_dir=result_dir, prefix='run_')
    init_x = torch.load(os.path.join(latest_folder, "updated_x.pt"))
    init_y = torch.load(os.path.join(latest_folder, "updated_y.pt"))
    return init_x, init_y
