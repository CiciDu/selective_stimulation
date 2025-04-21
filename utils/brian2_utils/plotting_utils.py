from brian2 import *
import math


def plot_firing_rate(r_E, r_I, r_E_sels, title_prefix=''):
    """
    Plot the firing rates of the neurons in the network."
    """

    # plotting
    figure(figsize=(8, 4))
    title(title_prefix + 'Population rates')
    xlabel('ms')
    ylabel('Hz')

    plot(r_E.t / ms, r_E.smooth_rate(width=25 * ms) / Hz, label='nonselective')
    plot(r_I.t / ms, r_I.smooth_rate(width=25 * ms) / Hz, label='inhibitory')

    p = len(r_E_sels)
    for i, r_E_sel in enumerate(r_E_sels[::-1]):
        plot(r_E_sel.t / ms, r_E_sel.smooth_rate(width=25 * ms) / Hz,
             label=f"selective {p - i}")

    legend()
    show()
    return


def plot_raster(N_activity_plot, sp_E, sp_I, sp_E_sels, p, title_prefix=''):
    """
    Plot the firing rates of the neurons in the network."
    """
    figure(figsize=(8, 4))
    title(title_prefix +
          f'Population activities ({N_activity_plot} neurons/pop)')
    xlabel('ms')
    # yticks([])

    plot(sp_E.t / ms, sp_E.i + (p + 1) * N_activity_plot, '.', markersize=2,
         label="nonselective")
    plot(sp_I.t / ms, sp_I.i + p * N_activity_plot,
         '.', markersize=2, label="inhibitory")

    for i, sp_E_sel in enumerate(sp_E_sels[::-1]):
        plot(sp_E_sel.t / ms, sp_E_sel.i + (p - i - 1) * N_activity_plot, '.', markersize=2,
             label=f"selective {p - i}")

    legend()
    show()


def _plot_neuron_currents(monitor, E_index_map, currents_to_plot, is_inhibitory=False,
                          title_prefix='', pop_to_plot=[1]):

    if monitor is None:
        # raise a warning
        print("current_monitor_I is None, skipping plot.")
        return

    figure(figsize=(8, 4))

    # figure(figsize=(12, 8))
    num_pop = len(E_index_map)
    if pop_to_plot is None:
        pop_to_plot = range(num_pop)
    for i in pop_to_plot:
        if len(pop_to_plot) > 1:
            subplot(math.ceil((num_pop)/2), 2, i+1)
        if is_inhibitory:
            title(title_prefix + "Inhibitory Population Currents")
        else:
            title(title_prefix + E_index_map[i])
        xlabel('Time (ms)')
        ylabel('Current (A)')

        for _, current in enumerate(currents_to_plot):
            if current in ['I_DC1', 'I_DC2']:
                plot(monitor.t / ms, getattr(monitor, current)
                     [i], label=current, linewidth=2, alpha=0.8)
            else:
                plot(monitor.t / ms,
                     getattr(monitor, current)[i], label=current, linewidth=0.5, alpha=0.5)
        legend(loc='best')
    show()


def plot_E_sel_1_currents(current_monitor_E, E_index_map, currents_to_plot, title_prefix=''):
    _plot_neuron_currents(current_monitor_E, E_index_map, currents_to_plot,
                          is_inhibitory=False, title_prefix=title_prefix,
                          pop_to_plot=[1])


def plot_E_neuron_currents(current_monitor_E, E_index_map, currents_to_plot, title_prefix='', pop_to_plot=None):
    _plot_neuron_currents(current_monitor_E, E_index_map, currents_to_plot,
                          is_inhibitory=False, title_prefix=title_prefix,
                          pop_to_plot=pop_to_plot)


def plot_I_neuron_currents(current_monitor_I, currents_to_plot, title_prefix):
    _plot_neuron_currents(current_monitor_I, {}, currents_to_plot,
                          is_inhibitory=True, title_prefix=title_prefix)


def plot_currents(current_monitor_E, current_monitor_I, currents_to_plot, E_index_map, title_prefix=''):
    plot_E_sel_1_currents(current_monitor_E, E_index_map,
                          currents_to_plot, title_prefix)

    # plot_E_neuron_currents(current_monitor_E, E_index_map,
    #                        currents_to_plot, title_prefix)
    # plot_I_neuron_currents(current_monitor_I, currents_to_plot, title_prefix)
