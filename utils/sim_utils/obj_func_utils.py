from brian2 import *
from utils.sim_utils import plotting_utils, set_params_utils
import torch


def calculate_fr_in_window(r_E_sel, fr_window=[2200, 3000], smoothing_width=50 * ms):
    ts_index = np.where(
        (r_E_sel.t / ms > fr_window[0]) & (r_E_sel.t / ms < fr_window[1]))[0]
    t_in_window = r_E_sel.t[ts_index]
    rate_in_window = r_E_sel.smooth_rate(width=smoothing_width)[ts_index] / Hz
    mean_fr = np.mean(rate_in_window)
    return t_in_window, rate_in_window, mean_fr


def process_input_tensor(params):
    # if the first dimension is 1, remove it
    if len(params.shape) == 2 and params.shape[0] == 1:
        params = params[0]
    params = np.round(np.array(params.tolist()), 5).reshape(-1)
    return params


def process_output_tensor(output):
    output = torch.tensor(output)
    return output


def update_name_space_with_params_1_current(params, namespace, DC_input_ts, sino_input_ts):

    DC_start_time1, DC_duration1, DC_amp1, DC_amp_slope1 = params

    set_params_utils.update_DC_input(namespace['DC_input1'],
                                     DC_amp=DC_amp1,  # in nA
                                     DC_amp_slope=DC_amp_slope1,  # in nA/s
                                     DC_duration=DC_duration1,  # in ms
                                     DC_start_time=DC_start_time1,  # in ms
                                     timestep=DC_input_ts
                                     )

    print(f'DC_amp1: {DC_amp1}, DC_amp_slope1: {DC_amp_slope1}, DC_start_time1: {DC_start_time1}, DC_duration1: {DC_duration1}')

    return namespace


def calculate_fr_diff_after_distractor(r_E_sels, fr_window=[2200, 3000], smoothing_width=25 * ms):
    t_in_window, rate_in_window, mean_fr_sel1 = calculate_fr_in_window(
        r_E_sels[0], fr_window=fr_window, smoothing_width=smoothing_width)

    t_in_window, rate_in_window, mean_fr_sel2 = calculate_fr_in_window(
        r_E_sels[1], fr_window=fr_window, smoothing_width=smoothing_width)
    fr_diff = mean_fr_sel1 - mean_fr_sel2

    print(f'Pop1 mean_fr in the window {fr_window} ms: {mean_fr_sel1}')
    print(f'Pop2 mean_fr in the window {fr_window} ms: {mean_fr_sel2}')
    print(f'Difference in mean_fr: {fr_diff}')
    return fr_diff


def calculate_duration_of_pop1_persistent_activity(r_E_sels,
                                                   r_E,
                                                   fr_window,
                                                   diff_cutout=10,
                                                   smoothing_width=50 * ms):
    # Compute firing rates
    t_in_window, rate_pop1, _ = calculate_fr_in_window(
        r_E_sels[0], fr_window=fr_window, smoothing_width=smoothing_width)
    _, rate_pop2, _ = calculate_fr_in_window(
        r_E_sels[1], fr_window=fr_window, smoothing_width=smoothing_width)
    _, rate_pop3, _ = calculate_fr_in_window(
        r_E_sels[2], fr_window=fr_window, smoothing_width=smoothing_width)
    _, rate_all, _ = calculate_fr_in_window(
        r_E, fr_window=fr_window, smoothing_width=smoothing_width)

    # Compute min difference
    min_diff = np.minimum.reduce([
        rate_pop1 - rate_pop2,
        rate_pop1 - rate_pop3,
        rate_pop1 - rate_all
    ])

    indices = np.where(min_diff < diff_cutout)[0]
    cut_off = indices[0] if len(indices) > 0 else len(t_in_window) - 1

    # calculate the persistent activity of pop1 in ms starting from 1100 (fr_window[0]) ms
    persistent_t = (t_in_window[cut_off] - t_in_window[0]) / ms

    print(
        f'Persistent activity of pop1 after {fr_window[0]} ms : {persistent_t} ms')
    return persistent_t


def process_params(params, stim_bounds, process_input_func=None):
    if process_input_func is not None:
        params = process_input_func(params)

    params = np.array(params)
    assert np.all((params >= 0) & (params <= 1)
                  ), "All elements in params must be in [0, 1]"
    # unnormalize the params
    stim_bounds = np.array(stim_bounds)
    params = stim_bounds[0, :] + params * \
        (stim_bounds[1, :] - stim_bounds[0, :])
    params = np.round(params, 5)
    return params


def objective_function(params, net, namespace, stim_bounds,
                       current_monitor_E, r_E, r_I, r_E_sels, E_index_map,
                       process_input_func=None,
                       process_output_func=None,
                       currents_to_plot=[
                           'I_DC1'],
                       DC_input_ts=1 * ms,
                       sino_input_ts=0.1 * ms,
                       maximize=True,
                       plotting=False,
                       fr_window=[1100, 8100],
                       smoothing_width=50 * ms,
                       results=None,
                       idx=None,
                       ):

    net.restore('initial')

    print('================================================================================================')
    print('================================================================================================')
    print('Objective function called with params: ', params)

    params = process_params(params, stim_bounds,
                            process_input_func=process_input_func)

    namespace = update_name_space_with_params_1_current(
        params, namespace, DC_input_ts, sino_input_ts)

    net.run(8.1 * second, namespace=namespace)

    # fr_diff = calculate_fr_diff_after_distractor(r_E_sels)
    # output = fr_diff
    # normed_value = output/20

    persistent_t = calculate_duration_of_pop1_persistent_activity(
        r_E_sels, r_E, fr_window=fr_window, smoothing_width=smoothing_width)
    output = persistent_t
    norm_factor = int(fr_window[1] - fr_window[0])
    normed_value = output/norm_factor

    if plotting:
        plotting_utils.plot_currents(
            current_monitor_E, None, currents_to_plot, E_index_map, title_prefix='')

        plotting_utils.plot_firing_rate(
            r_E, r_I, r_E_sels, title_prefix='', smoothing_width=smoothing_width)

    if maximize:
        value = output
        optimize = 'maximize'
    else:
        # if we want to minimize the value, we need to negate it
        value = - output
        optimize = 'minimize'

    normed_value = np.round(value/norm_factor, 5)
    value = np.round(value, 5)
    normed_value = np.round(normed_value, 5)

    if process_output_func is not None:
        normed_value = process_output_func(normed_value)

    print(f'{value} (value to {optimize})')
    print(f'{normed_value} (Normalized value)')

    if results is not None and idx is not None:
        results[idx] = normed_value
    else:
        return normed_value
