import os
import math
import warnings
from dataclasses import dataclass
from utils.bo_utils import ibnn_utils, shared_utils
import torch



def save_bo_results(res, result_dir='all_stored_results/bo_results'):
    train_x = torch.tensor(res['x_iters'])
    y_vals = res['func_vals'].reshape(-1, 1)
    y_vals = - y_vals # because the BO function minimizes the objective function
    train_y = torch.tensor(y_vals)
    
    result_folder = shared_utils.get_new_result_subfolder(result_dir)
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)

    torch.save(train_x, os.path.join(result_folder, "updated_x.pt"))
    torch.save(train_y, os.path.join(result_folder, "updated_y.pt"))
    return train_x, train_y