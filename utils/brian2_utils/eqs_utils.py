from brian2 import *


def write_eqs_E():
    eqs_E = '''
    dv / dt = (- g_m_E * (v - V_L) - I_syn - I_DC1 - I_DC2 - I_sino1 - I_sino2) / C_m_E : volt (unless refractory)

    I_syn = I_AMPA_ext + I_AMPA_rec + I_NMDA_rec + I_GABA_rec : amp

    I_AMPA_ext = g_AMPA_ext_E * (v - V_E) * s_AMPA_ext : amp
    I_AMPA_rec = g_AMPA_rec_E * (v - V_E) * 1 * s_AMPA : amp
    ds_AMPA_ext / dt = - s_AMPA_ext / tau_AMPA : 1
    ds_AMPA / dt = - s_AMPA / tau_AMPA : 1

    I_NMDA_rec = g_NMDA_E * (v - V_E) / (1 + Mg2 * exp(-0.062 * v / mV) / 3.57) * s_NMDA_tot : amp
    s_NMDA_tot : 1

    I_GABA_rec = g_GABA_E * (v - V_I) * s_GABA : amp
    ds_GABA / dt = - s_GABA / tau_GABA : 1

    # Define the stimulation currents
    I_DC1 = DC_input1(t) : amp
    I_DC2 = DC_input2(t) : amp
    I_sino1 = sino_input1(t) : amp
    I_sino2 = sino_input2(t) : amp
    '''
    return eqs_E


def write_eqs_I():
    eqs_I = '''
        dv / dt = (- g_m_I * (v - V_L) - I_syn - I_DC1 - I_DC2 - I_sino1 - I_sino2) / C_m_I : volt (unless refractory)

        I_syn = I_AMPA_ext + I_AMPA_rec + I_NMDA_rec + I_GABA_rec : amp

        I_AMPA_ext = g_AMPA_ext_I * (v - V_E) * s_AMPA_ext : amp
        I_AMPA_rec = g_AMPA_rec_I * (v - V_E) * 1 * s_AMPA : amp
        ds_AMPA_ext / dt = - s_AMPA_ext / tau_AMPA : 1
        ds_AMPA / dt = - s_AMPA / tau_AMPA : 1

        I_NMDA_rec = g_NMDA_I * (v - V_E) / (1 + Mg2 * exp(-0.062 * v / mV) / 3.57) * s_NMDA_tot : amp
        s_NMDA_tot : 1

        I_GABA_rec = g_GABA_I * (v - V_I) * s_GABA : amp
        ds_GABA / dt = - s_GABA / tau_GABA : 1

        # Define the stimulation currents
        I_DC1 = DC_input1(t) : amp
        I_DC2 = DC_input2(t) : amp
        I_sino1 = sino_input1(t) : amp
        I_sino2 = sino_input2(t) : amp
        '''
    return eqs_I


def write_other_eqs():
    eqs_glut = '''
    s_NMDA_tot_post = w * s_NMDA : 1 (summed)
    ds_NMDA / dt = - s_NMDA / tau_NMDA_decay + alpha * x * (1 - s_NMDA) : 1 (clock-driven)
    dx / dt = - x / tau_NMDA_rise : 1 (clock-driven)
    w : 1
    '''

    eqs_pre_glut = '''
    s_AMPA += w
    x += 1
    '''

    eqs_pre_gaba = '''
    s_GABA += 1
    '''

    return eqs_glut, eqs_pre_glut, eqs_pre_gaba
