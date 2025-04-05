from brian2 import *
import math


def plot_firing_rate(r_E, r_I, r_E_sels, p, title_prefix=''):
    """
    Plot the firing rates of the neurons in the network."
    """

    # plotting
    figure(figsize=(6, 4))
    title(title_prefix + 'Population rates')
    xlabel('ms')
    ylabel('Hz')

    plot(r_E.t / ms, r_E.smooth_rate(width=25 * ms) / Hz, label='nonselective')
    plot(r_I.t / ms, r_I.smooth_rate(width=25 * ms) / Hz, label='inhibitory')

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
    figure(figsize=(6, 4))
    title(title_prefix + f'Population activities ({N_activity_plot} neurons/pop)')
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



def plot_E_neuron_currents(DC_monitor_E, E_index_map, currents_to_plot, title_prefix='', pop_to_plot=[1]):
    figure(figsize=(12, 8))
    num_pop = len(E_index_map)
    if pop_to_plot is None:
        pop_to_plot = range(num_pop)
    for i in pop_to_plot:
        subplot(math.ceil((num_pop)/2), 2, i+1)
        title(title_prefix + E_index_map[i])
        xlabel('Time (ms)')
        ylabel('Current (A)')

        for _, current in enumerate(currents_to_plot):
            if current == 'I_DC':
                plot(DC_monitor_E.t / ms, getattr(DC_monitor_E, current)
                    [i], label=current, linewidth=3, alpha=0.7)
            else:
                plot(DC_monitor_E.t / ms,
                    getattr(DC_monitor_E, current)[i], label=current, linewidth=1, alpha=0.7)
        legend(loc='best')
    show()


def plot_I_neuron_currents(DC_monitor_I, currents_to_plot, title_prefix):
    figure(figsize=(6, 4))
    title(title_prefix + "Inhibitory Population Currents")
    xlabel('Time (ms)')
    ylabel('Current (A)')
    for i, current in enumerate(currents_to_plot):
        if current == 'I_DC':
            plot(DC_monitor_I.t / ms, getattr(DC_monitor_I, current)
                 [0], label=current, linewidth=3)
        else:
            plot(DC_monitor_I.t / ms,
                 getattr(DC_monitor_I, current)[0], label=current)
    legend(loc='best')
    show()



def plot_currents(DC_monitor_E, DC_monitor_I, currents_to_plot, E_index_map, title_prefix=''):
    plot_E_neuron_currents(DC_monitor_E, E_index_map, currents_to_plot, title_prefix)
    #plot_I_neuron_currents(DC_monitor_I, currents_to_plot, title_prefix)

