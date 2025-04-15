from brian2 import *


def set_DC_input(DC_amp=0.5,  # in nA
                 DC_amp_slope=0,  # in nA/s
                 DC_duration=800,  # in ms
                 DC_start_time=0,  # in ms
                 timestep=1 * ms
                 ):
    
    # get number of timestamps before the DC current starts
    DC_prior_start_ts = int(DC_start_time * ms / timestep)
    # get number of timestamps in the DC duration
    DC_ts = int(DC_duration * ms / timestep)

    # make DC input incorporate the slope
    # Time array in seconds
    time_array = np.arange(DC_ts) * timestep
    # Compute ramping DC current in nA
    DC_array = DC_amp + DC_amp_slope * (time_array / second)

    DC_input = TimedArray(
        np.r_[np.zeros(DC_prior_start_ts), DC_array, np.zeros(50)] * nA, dt=timestep)
    return DC_input


def set_sino_input(sino_start_time=0,  # in ms
                   sino_duration=3000,  # in ms
                   sino_amp=5,  # in nA
                   sino_freq=0.5,  # in Hz
                   timestep=0.1 * ms
                   ):

    sino_prior_start_ts = int(sino_start_time * ms / timestep)
    sino_ts = arange(int(sino_duration * ms / timestep)) * \
        timestep  # break the time into timestep
    sino_array = sino_amp * sin(2 * pi * sino_freq * Hz * sino_ts)
    sino_input = TimedArray(
        np.r_[np.zeros(sino_prior_start_ts), sino_array, np.zeros(50)] * nA, dt=timestep)
    return sino_input


# def set_sino_input(sino_start_time=0,  # in ms
#                    sino_duration=3000,  # in ms
#                    sino_amp1=5,  # in nA
#                    sino_amp2=10,  # in nA
#                    sino_freq1=0.5,  # in Hz
#                    sino_freq2=1,  # in Hz
#                    timestep=25 * ms
#                    ):

#     sino_prior_start_ts = int(sino_start_time * ms / timestep)
#     sino_ts = arange(int(sino_duration * ms / timestep)) * timestep  # break the time into timestep
#     sino_array = sino_amp1 * sin(2 * pi * sino_freq1 * Hz * sino_ts) + sino_amp2 * sin(2 * pi * sino_freq2 * Hz * sino_ts)
#     sino_input = TimedArray(
#         np.r_[np.zeros(sino_prior_start_ts), sino_array] * nA, dt=timestep)
#     return sino_input


def set_voltage():

    V_L = -70. * mV
    V_thr = -50. * mV
    V_reset = -55. * mV
    V_E = 0. * mV
    V_I = -70. * mV
    return V_L, V_thr, V_reset, V_E, V_I


def set_membrane_params():
    # membrane capacitance
    C_m_E = 0.5 * nF
    C_m_I = 0.2 * nF

    # membrane leak
    g_m_E = 25. * nS
    g_m_I = 20. * nS
    return C_m_E, C_m_I, g_m_E, g_m_I


def set_AMPA_params(N_E):
    g_AMPA_ext_E = 2.08 * nS
    g_AMPA_rec_E = 0.104 * nS * 800. / N_E
    g_AMPA_ext_I = 1.62 * nS
    g_AMPA_rec_I = 0.081 * nS * 800. / N_E
    tau_AMPA = 2. * ms
    return g_AMPA_ext_E, g_AMPA_rec_E, g_AMPA_ext_I, g_AMPA_rec_I, tau_AMPA


def set_NMDA_params(N_E):
    g_NMDA_E = 0.327 * nS * 800. / N_E
    g_NMDA_I = 0.258 * nS * 800. / N_E
    tau_NMDA_rise = 2. * ms
    tau_NMDA_decay = 100. * ms
    alpha = 0.5 / ms
    Mg2 = 1.
    return g_NMDA_E, g_NMDA_I, tau_NMDA_rise, tau_NMDA_decay, alpha, Mg2


def set_GABA_params(N_I):
    g_GABA_E = 1.25 * nS * 200. / N_I
    g_GABA_I = 0.973 * nS * 200. / N_I
    tau_GABA = 10. * ms
    return g_GABA_E, g_GABA_I, tau_GABA


def set_synapses(P_E, P_I, N_E, N_I, N_sub, N_non, p, f, C_ext, rate,
                 eqs_glut, eqs_pre_glut, eqs_pre_gaba):
    C_E = N_E
    C_I = N_I

    w_plus = 2.1
    w_minus = 1. - f * (w_plus - 1.) / (1. - f)

    # E to E
    C_E_E = Synapses(P_E, P_E, model=eqs_glut,
                     on_pre=eqs_pre_glut, method='euler')
    C_E_E.connect('i != j')
    C_E_E.w[:] = 1
    for pi in range(N_non, N_non + p * N_sub, N_sub):

        # internal other subpopulation to current nonselective
        C_E_E.w[C_E_E.indices[:, pi:pi + N_sub]] = w_minus

        # internal current subpopulation to current subpopulation
        C_E_E.w[C_E_E.indices[pi:pi + N_sub, pi:pi + N_sub]] = w_plus

    # E to I
    C_E_I = Synapses(P_E, P_I, model=eqs_glut,
                     on_pre=eqs_pre_glut, method='euler')
    C_E_I.connect()
    C_E_I.w[:] = 1

    # I to I
    C_I_I = Synapses(P_I, P_I, on_pre=eqs_pre_gaba, method='euler')
    C_I_I.connect('i != j')

    # I to E
    C_I_E = Synapses(P_I, P_E, on_pre=eqs_pre_gaba, method='euler')
    C_I_E.connect()

    # external noise
    C_P_E = PoissonInput(P_E, 's_AMPA_ext', C_ext, rate, '1')
    C_P_I = PoissonInput(P_I, 's_AMPA_ext', C_ext, rate, '1')

    return C_E, C_I, C_E_E, C_E_I, C_I_I, C_I_E, C_P_E, C_P_I


def set_neuron_groups(N_E, N_I, eqs_E, eqs_I, V_L):
    # refractory period
    tau_rp_E = 2. * ms
    tau_rp_I = 1. * ms

    P_E = NeuronGroup(N_E, eqs_E, threshold='v > V_thr',
                      reset='v = V_reset', refractory=tau_rp_E, method='euler')
    P_E.v = V_L
    P_I = NeuronGroup(N_I, eqs_I, threshold='v > V_thr',
                      reset='v = V_reset', refractory=tau_rp_I, method='euler')
    P_I.v = V_L
    return P_E, P_I


def set_monitors(N_activity_plot, N_non, N_sub, p, P_E, P_I,
                 E_neuron_index=[0],
                 currents_to_track=['I_syn', 'I_DC', 'I_sino', 'I_AMPA_ext',
                                    'I_AMPA_rec', 'I_NMDA_rec', 'I_GABA_rec']):

    # Add StateMonitors to track the DC current
    DC_monitor_E = StateMonitor(P_E, currents_to_track, record=E_neuron_index)
    DC_monitor_I = StateMonitor(P_I, currents_to_track, record=True)

    sp_E_sels = [SpikeMonitor(P_E[pi:pi + N_activity_plot])
                 for pi in range(N_non, N_non + p * N_sub, N_sub)]
    sp_E = SpikeMonitor(P_E[:N_activity_plot])
    sp_I = SpikeMonitor(P_I[:N_activity_plot])

    r_E_sels = [PopulationRateMonitor(P_E[pi:pi + N_sub])
                for pi in range(N_non, N_non + p * N_sub, N_sub)]
    r_E = PopulationRateMonitor(P_E[:N_non])
    r_I = PopulationRateMonitor(P_I)
    return DC_monitor_E, DC_monitor_I, sp_E_sels, sp_E, sp_I, r_E_sels, r_E, r_I


def set_monitors_for_optimization_algorithm(N_activity_plot, N_non, N_sub, p, P_E, P_I,
                 E_neuron_index=[0],
                 currents_to_track=['I_syn', 'I_DC', 'I_sino', 'I_AMPA_ext',
                                    'I_AMPA_rec', 'I_NMDA_rec', 'I_GABA_rec']):

    # Add StateMonitors to track the DC current
    DC_monitor_E = StateMonitor(P_E, currents_to_track, record=E_neuron_index)

    r_E_sels = [PopulationRateMonitor(P_E[pi:pi + N_sub])
                for pi in range(N_non, N_non + p * N_sub, N_sub)]
    r_E = PopulationRateMonitor(P_E[:N_non])
    r_I = PopulationRateMonitor(P_I)
    return DC_monitor_E, r_E_sels, r_E, r_I
