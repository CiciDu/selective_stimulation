import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import torch
import math
from utils.methods_utils import shared_utils


def parse_cma_log(log_text):
    with open("cma_output_log.txt", "r") as file:
        log_text = file.read()
        log_text
        pattern = re.compile(
            # r"params:\s+\[(.*?)\]\s*"
            r"params:\s+\[([\d\s.eE+-]+)\]\s*"
            r"DC_amp1:\s+([-.\d.eE+-]+), DC_amp_slope1:\s+([-.\d.eE+-]+), "
            r"DC_start_time1:\s+([-.\d.eE+-]+), DC_duration1:\s+([-.\d.eE+-]+)\s*"
            r"Persistent activity of pop1 after 1100 ms\s*:\s*([-.\d.eE+-]+) ms\s*"
            r"([-.\d.eE+-]+) \(value to minimize\)\s*"
            r"([-.\d.eE+-]+) \(Normalized value\)",
            re.DOTALL
        )

        all_matches = pattern.findall(log_text)
        all_data = []
        all_params = []
        all_normalized_value = []

        for i in range(len(all_matches)):
            match = all_matches[i]
            # print('i = ', i)
            param_str = match[0]
            # Clean and split param values
            params = [float(p.strip()) for p in param_str.strip().split()]
            rest = list(map(float, match[1:]))
            all_data.append(params + rest)
            all_params.append(params)
            all_normalized_value.append(rest[-1])

        columns = [
            'param_1', 'param_2', 'param_3', 'param_4',
            'DC_amp1', 'DC_amp_slope1', 'DC_start_time1', 'DC_duration1',
            'Persistent_activity', 'Value_to_minimize', 'Normalized_value'
        ]

        # Create a DataFrame from the matches
        df = pd.DataFrame(all_data, columns=columns)

    return all_params, all_normalized_value, df


def separate_cma_results_by_pop(updated_x, updated_y, pop_size):

    num_iter = math.floor(len(updated_y) / pop_size)

    # Reshape the tensor to have shape [num_iter, pop_size]
    y_over_iter = updated_y[:num_iter * pop_size].reshape(-1, pop_size)

    # Convert params to a tensor: assuming all_params is a list of list-like 4D vectors
    params_over_iter = updated_x[:num_iter * pop_size].reshape(-1, pop_size, 4)

    # Get top index and corresponding values/params
    top_index_over_iter = torch.argmax(y_over_iter, dim=1)
    top_y_over_iter = y_over_iter[torch.arange(num_iter), top_index_over_iter]
    top_params_over_iter = params_over_iter[torch.arange(
        num_iter), top_index_over_iter]

    return top_y_over_iter, top_params_over_iter, top_index_over_iter
