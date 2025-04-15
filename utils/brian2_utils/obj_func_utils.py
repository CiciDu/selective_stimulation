from brian2 import *
from utils.brian2_utils import plotting_utils, set_params_utils
import torch


def calculate_fr_in_window(r_E_sel, fr_window=[2200, 3000]):
    ts_index = np.where(
        (r_E_sel.t / ms > fr_window[0]) & (r_E_sel.t / ms < fr_window[1]))[0]
    t_in_window = r_E_sel.t[ts_index]
    rate_in_window = r_E_sel.smooth_rate(width=25 * ms)[ts_index] / Hz
    mean_fr = np.mean(rate_in_window)
    return t_in_window, rate_in_window, mean_fr


def process_input_ibnn(params):
    # if the first dimension is 1, remove it
    if len(params.shape) == 2 and params.shape[0] == 1:
        params = params[0]
    params = np.round(np.array(params.tolist()), 4).reshape(-1)
    return params


def process_output_ibnn(output):
    output = torch.tensor(output)
    return output


def update_name_space_with_params(params, namespace, DC_input_ts, sino_input_ts):

    DC_start_time1, DC_duration1, DC_amp1, DC_amp_slope1, \
        DC_start_time2, DC_duration2, DC_amp2, DC_amp_slope2, \
        sino_start_time1, sino_duration1, sino_amp1, sino_freq1, \
        sino_start_time2, sino_duration2, sino_amp2, sino_freq2 = params

    DC_input1 = set_params_utils.set_DC_input(DC_amp=DC_amp1,  # in nA
                                              DC_amp_slope=DC_amp_slope1,  # in nA/s
                                              DC_duration=DC_duration1,  # in ms
                                              DC_start_time=DC_start_time1,  # in ms
                                              timestep=DC_input_ts
                                              )

    DC_input2 = set_params_utils.set_DC_input(DC_amp=DC_amp2,  # in nA
                                              DC_amp_slope=DC_amp_slope2,  # in nA/s
                                              DC_duration=DC_duration2,  # in ms
                                              DC_start_time=DC_start_time2,  # in ms
                                              timestep=DC_input_ts
                                              )

    sino_input1 = set_params_utils.set_sino_input(sino_start_time=sino_start_time1,  # in ms
                                                  sino_duration=sino_duration1,  # in ms
                                                  sino_amp=sino_amp1,  # in nA
                                                  sino_freq=sino_freq1,  # in Hz
                                                  timestep=sino_input_ts
                                                  )

    sino_input2 = set_params_utils.set_sino_input(sino_start_time=sino_start_time2,  # in ms
                                                  sino_duration=sino_duration2,  # in ms
                                                  sino_amp=sino_amp2,  # in nA
                                                  sino_freq=sino_freq2,  # in Hz
                                                  timestep=sino_input_ts
                                                  )

    namespace['DC_input1'] = DC_input1
    namespace['DC_input2'] = DC_input2
    namespace['sino_input1'] = sino_input1
    namespace['sino_input2'] = sino_input2

    print(f'DC_amp1: {DC_amp1}, DC_amp_slope1: {DC_amp_slope1}, DC_start_time1: {DC_start_time1}, DC_duration1: {DC_duration1}')
    print(f'DC_amp2: {DC_amp2}, DC_amp_slope2: {DC_amp_slope2}, DC_start_time2: {DC_start_time2}, DC_duration2: {DC_duration2}')
    print(f'sino_amp1: {sino_amp1}, sino_freq1: {sino_freq1}, sino_start_time1: {sino_start_time1}, sino_duration1: {sino_duration1}')
    print(f'sino_amp2: {sino_amp2}, sino_freq2: {sino_freq2}, sino_start_time2: {sino_start_time2}, sino_duration2: {sino_duration2}')

    # print(f'DC_start_time1: {DC_start_time1}, DC_duration1: {DC_duration1}, DC_amp1: {DC_amp1}, DC_amp_slope1: {DC_amp_slope1}')
    # print(f'DC_start_time2: {DC_start_time2}, DC_duration2: {DC_duration2}, DC_amp2: {DC_amp2}, DC_amp_slope2: {DC_amp_slope2}')
    # print(f'sino_start_time1: {sino_start_time1}, sino_duration1: {sino_duration1}, sino_amp1: {sino_amp1}, sino_freq1: {sino_freq1}')
    # print(f'sino_start_time2: {sino_start_time2}, sino_duration2: {sino_duration2}, sino_amp2: {sino_amp2}, sino_freq2: {sino_freq2}')

    return namespace


def calculate_fr_diff_after_distractor(r_E_sels, fr_window=[2200, 3000]):
    t_in_window, rate_in_window, mean_fr_sel1 = calculate_fr_in_window(
        r_E_sels[0], fr_window=fr_window)

    t_in_window, rate_in_window, mean_fr_sel2 = calculate_fr_in_window(
        r_E_sels[1], fr_window=fr_window)
    fr_diff = mean_fr_sel1 - mean_fr_sel2

    print(f'Pop1 mean_fr in the window {fr_window} ms: {mean_fr_sel1}')
    print(f'Pop2 mean_fr in the window {fr_window} ms: {mean_fr_sel2}')
    print(f'Difference in mean_fr: {fr_diff}')
    return fr_diff


def calculate_duration_of_pop1_persistent_activity(r_E_sels,
                                                   fr_window=[1100, 3000],
                                                   diff_cutout=10):
    t_in_window, rate_in_window1, _ = calculate_fr_in_window(
        r_E_sels[0], fr_window=fr_window)
    _, rate_in_window2, _ = calculate_fr_in_window(
        r_E_sels[1], fr_window=fr_window)
    _, rate_in_window3, _ = calculate_fr_in_window(
        r_E_sels[1], fr_window=fr_window)

    # calculate the difference between firing rate of pop 1 and pop 2
    diff_12 = rate_in_window1 - rate_in_window2
    # calculate the difference between firing rate of pop 2 and pop 3
    diff_13 = rate_in_window1 - rate_in_window3
    min_diff = np.minimum(diff_12, diff_13)
    # now, we want to know the cut off point, which is when the difference drops below 10

    indices = np.where(min_diff < diff_cutout)[0]
    cut_off = indices[0] if len(indices) > 0 else len(t_in_window) - 1

    # calculate the persistent activity of pop1 in ms starting from 1100 ms
    persistent_t = (t_in_window[cut_off] - t_in_window[0]) / ms

    print(
        f'Persistent activity of pop1 after {fr_window[0]} ms : {persistent_t} ms')
    return persistent_t


def objective_function(params, net, namespace, DC_monitor_E, r_E, r_I, r_E_sels, E_index_map,
                       process_input_func=None,
                       process_output_func=None,
                       currents_to_plot=[
                           'I_DC1', 'I_DC2', 'I_sino1', 'I_sino2'],
                       DC_input_ts=1 * ms,
                       sino_input_ts=0.1 * ms,
                       maximize=True,
                       ):

    net.restore('initial')

    if process_input_func is not None:
        params = process_input_func(params)

    namespace = update_name_space_with_params(
        params, namespace, DC_input_ts, sino_input_ts)

    net.run(5 * second, namespace=namespace)

    # fr_diff = calculate_fr_diff_after_distractor(r_E_sels)
    # output = fr_diff

    persistent_t = calculate_duration_of_pop1_persistent_activity(r_E_sels)
    output = persistent_t

    if process_output_func is not None:
        output = process_output_func(output)

    plotting_utils.plot_currents(
        DC_monitor_E, None, currents_to_plot, E_index_map, title_prefix='')

    plotting_utils.plot_firing_rate(
        r_E, r_I, r_E_sels, title_prefix='')

    if maximize:
        value = output
        print(f'Value to maximize: {value}')
    else:
        # if we want to minimize the value, we need to negate it
        value = - output
        print(f'Value to minimize: {value}')

    return value
