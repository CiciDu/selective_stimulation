from brian2 import *
from utils.sim_utils import plotting_utils, set_params_utils
import torch
from skopt.space import Real

DC_amp1_range = [-0.3, 0.3]            # in nA
DC_amp_slope1_range = [-0.2, 0.2]    # in nA/ms
DC_start_time1_range = [0, 2000]     # in ms
DC_duration1_range = [20, 8000]      # in ms

# DC_amp1_range = [-0.3, 0]            # in nA
# DC_amp_slope1_range = [-0.2, 0.2]    # in nA/ms
# DC_start_time1_range = [0, 4000]     # in ms
# DC_duration1_range = [20, 6000]      # in ms

# DC_amp2_range = [0, 0.3]             # in nA
# DC_amp_slope2_range = [-0.2, 0.2]    # in nA/ms
# DC_start_time2_range = [0, 4000]     # in ms
# DC_duration2_range = [20, 6000]      # in ms

# sino_start_time1_range = [0, 4000]   # in ms
# sino_duration1_range = [20, 6000]    # in ms
# sino_amp1_range = [0, 0.3]           # in nA
# sino_freq1_range = [0.01, 10]        # in Hz

# sino_start_time2_range = [0, 4000]   # in ms
# sino_duration2_range = [20, 6000]    # in ms
# sino_amp2_range = [0, 0.3]           # in nA
# sino_freq2_range = [0.01, 10]        # in Hz


space = [
    Real(DC_start_time1_range[0],
         DC_start_time1_range[1], name='DC_start_time1'),
    Real(DC_duration1_range[0], DC_duration1_range[1], name='DC_duration1'),
    Real(DC_amp1_range[0], DC_amp1_range[1], name='DC_amp1'),
    Real(DC_amp_slope1_range[0], DC_amp_slope1_range[1], name='DC_amp_slope1'),
    #     Real(DC_start_time2_range[0],
    #          DC_start_time2_range[1], name='DC_start_time2'),
    #     Real(DC_duration2_range[0], DC_duration2_range[1], name='DC_duration2'),
    #     Real(DC_amp2_range[0], DC_amp2_range[1], name='DC_amp2'),
    #     Real(DC_amp_slope2_range[0], DC_amp_slope2_range[1], name='DC_amp_slope2'),
    # Real(sino_start_time1_range[0], sino_start_time1_range[1], name='sino_start_time1'),
    # Real(sino_duration1_range[0], sino_duration1_range[1], name='sino_duration1'),
    # Real(sino_amp1_range[0], sino_amp1_range[1], name='sino_amp1'),
    # Real(sino_freq1_range[0], sino_freq1_range[1], name='sino_freq1'),
    # Real(sino_start_time2_range[0], sino_start_time2_range[1], name='sino_start_time2'),
    # Real(sino_duration2_range[0], sino_duration2_range[1], name='sino_duration2'),
    # Real(sino_amp2_range[0], sino_amp2_range[1], name='sino_amp2'),
    # Real(sino_freq2_range[0], sino_freq2_range[1], name='sino_freq2')
]

space_norm = [Real(0, 1, name=dim.name) for dim in space]
stim_bounds = torch.tensor([[dim.low, dim.high]
                           for dim in space], dtype=torch.float).T