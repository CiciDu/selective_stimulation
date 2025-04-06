from brian2 import *
from utils import *


def calculate_fr_in_window(r_E_sel, fr_window=[2000, 2500]):
    ts_index = np.where(
        (r_E_sel.t / ms > fr_window[0]) & (r_E_sel.t / ms < fr_window[1]))[0]
    t_in_window = r_E_sel.t[ts_index]
    rate_in_window = r_E_sel.smooth_rate(width=25 * ms)[ts_index] / Hz
    mean_fr = np.mean(rate_in_window)
    return t_in_window, rate_in_window, mean_fr


def objective_function(params, net, namespace, DC_monitor_E, r_E, r_I, r_E_sels,
             E_index_map, currents_to_plot=['I_DC1', 'I_DC2', 'I_sino1', 'I_sino2'],
             fr_window=[2300, 2500]):
    
    print('New Iteration')

    net.restore('initial')

    DC_input_ts = 25 * ms
    sino_input_ts = 0.1 * ms

    DC_amp1, DC_start_time1, DC_duration1, DC_amp2, DC_start_time2, DC_duration2, \
        sino_start_time1, sino_duration1, sino_amp1, sino_freq1, \
        sino_start_time2, sino_duration2, sino_amp2, sino_freq2 = params

    DC_input1 = set_params_utils.set_DC_input(DC_amp=DC_amp1,  # in nA
                                              DC_duration=DC_duration1,  # in ms
                                              DC_start_time=DC_start_time1,  # in ms
                                              timestep=DC_input_ts
                                              )

    DC_input2 = set_params_utils.set_DC_input(DC_amp=DC_amp2,  # in nA
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

    net.run(3 * second, namespace=namespace)

    t_in_window, rate_in_window, mean_fr_sel1 = calculate_fr_in_window(
        r_E_sels[0], fr_window=fr_window)
    
    t_in_window, rate_in_window, mean_fr_sel2 = calculate_fr_in_window(
        r_E_sels[1], fr_window=fr_window)
    
    print(f'DC_amp1: {DC_amp1}, DC_start_time1: {DC_start_time1}, DC_duration1: {DC_duration1}')
    print(f'DC_amp2: {DC_amp2}, DC_start_time2: {DC_start_time2}, DC_duration2: {DC_duration2}')
    print(f'sino_amp1: {sino_amp1}, sino_freq1: {sino_freq1}, sino_start_time1: {sino_start_time1}, sino_duration1: {sino_duration1}')
    print(f'sino_amp2: {sino_amp2}, sino_freq2: {sino_freq2}, sino_start_time2: {sino_start_time2}, sino_duration2: {sino_duration2}')

    print(f'Pop1 mean_fr in the window {fr_window} ms: {mean_fr_sel1}')
    print(f'Pop2 mean_fr in the window {fr_window} ms: {mean_fr_sel2}')

    fr_diff = mean_fr_sel1 - mean_fr_sel2
    print(f'Difference in mean_fr: {fr_diff}')

    value_to_minimize = -fr_diff

    plotting_utils.plot_firing_rate(
        r_E, r_I, r_E_sels, title_prefix='')
    plotting_utils.plot_currents(
        DC_monitor_E, None, currents_to_plot, E_index_map, title_prefix='')

    print('================================================================================================')
    print('================================================================================================')

    return value_to_minimize
