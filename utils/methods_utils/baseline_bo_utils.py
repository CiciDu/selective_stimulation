import os
import math
import warnings
from dataclasses import dataclass
from utils.methods_utils import ibnn_utils, shared_utils
import torch


def save_bo_results(res, result_dir='all_stored_results/bo_results'):
    train_x = torch.tensor(res['x_iters'])
    y_vals = res['func_vals'].reshape(-1, 1)
    y_vals = - y_vals  # because the BO function minimizes the objective function
    train_y = torch.tensor(y_vals)

    shared_utils.save_results(train_x, train_y, result_dir=result_dir)
    return train_x, train_y
