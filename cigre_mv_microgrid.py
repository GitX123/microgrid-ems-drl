'''
Modified CIGRE Task Force C6.04.02 network

elements:
- 8 PVs
- 1 WT
- 2 Batteries
- 8 Loads
'''

import pandapower as pp
from pandapower.control import ConstControl
from setting import *

def create_cigre_mv_microgrid(pv_ds, wt_ds, load_ds):
    net = pp.create_empty_network(name='CIGRE MV Microgrid')

    # --- Buses --- 
    bus0 = pp.create_bus(net, vn_kv=110, name='Buse 0', type='b', zone='CIGRE_MV')
    buses = pp.create_buses(net, 11, vn_kv=20, name=[f'Bus {i}' for i in range(1, 12)], type='b', zone='CIGRE_MV')

    # --- Lines ---
    line_data = {'c_nf_per_km': 151.1749, 'r_ohm_per_km': 0.501,
                 'x_ohm_per_km': 0.716, 'max_i_ka': 0.145,
                 'type': 'cs'}
    pp.create_std_type(net, line_data, name='CABLE_CIGRE_MV', element='line')

    pp.create_line(net, buses[0], buses[1], length_km=2.82,
                   std_type='CABLE_CIGRE_MV', name='Line 1-2')
    pp.create_line(net, buses[1], buses[2], length_km=4.42,
                   std_type='CABLE_CIGRE_MV', name='Line 2-3')
    pp.create_line(net, buses[2], buses[3], length_km=0.61,
                   std_type='CABLE_CIGRE_MV', name='Line 3-4')
    pp.create_line(net, buses[3], buses[4], length_km=0.56,
                   std_type='CABLE_CIGRE_MV', name='Line 4-5')
    pp.create_line(net, buses[4], buses[5], length_km=1.54,
                   std_type='CABLE_CIGRE_MV', name='Line 5-6')
    pp.create_line(net, buses[6], buses[7], length_km=1.67,
                   std_type='CABLE_CIGRE_MV', name='Line 7-8')
    pp.create_line(net, buses[7], buses[8], length_km=0.32,
                   std_type='CABLE_CIGRE_MV', name='Line 8-9')
    pp.create_line(net, buses[8], buses[9], length_km=0.77,
                   std_type='CABLE_CIGRE_MV', name='Line 9-10')
    pp.create_line(net, buses[9], buses[10], length_km=0.33,
                   std_type='CABLE_CIGRE_MV', name='Line 10-11')
    pp.create_line(net, buses[2], buses[7], length_km=1.3,
                   std_type='CABLE_CIGRE_MV', name='Line 3-8')

    # --- External Grid ---
    pp.create_ext_grid(net, bus0, vm_pu=1.03, va_degree=0., s_sc_max_mva=5000, s_sc_min_mva=5000, rx_max=0.1, rx_min=0.1)

    # --- Trafos ---
    trafo0 = pp.create_transformer_from_parameters(net, bus0, buses[0], sn_mva=25,
                                                   vn_hv_kv=110, vn_lv_kv=20, vkr_percent=0.16,
                                                   vk_percent=12.00107, pfe_kw=0, i0_percent=0,
                                                   shift_degree=30.0, name='Trafo 0-1')
    pp.create_switch(net, bus0, trafo0, et='t', closed=True, type='CB')

    # --- RESs ---
    # PV
    pv3 = pp.create_sgen(net, buses[2], 0.0, q_mvar=0, name='PV 3', type='PV')
    pv4 = pp.create_sgen(net, buses[3], 0.0, q_mvar=0, name='PV 4', type='PV')
    pv5 = pp.create_sgen(net, buses[4], 0.0, q_mvar=0, name='PV 5', type='PV')
    pv6 = pp.create_sgen(net, buses[5], 0.0, q_mvar=0, name='PV 6', type='PV')
    pv8 = pp.create_sgen(net, buses[7], 0.0, q_mvar=0, name='PV 8', type='PV')
    pv9 = pp.create_sgen(net, buses[8], 0.0, q_mvar=0, name='PV 9', type='PV')
    pv10 = pp.create_sgen(net, buses[9], 0.0, q_mvar=0, name='PV 10', type='PV')
    pv11 = pp.create_sgen(net, buses[10], 0.0, q_mvar=0, name='PV 11', type='PV')
    ConstControl(net, element='sgen', variable='p_mw', element_index=pv3, profile_name='pv3', data_source=pv_ds)
    ConstControl(net, element='sgen', variable='p_mw', element_index=pv4, profile_name='pv4', data_source=pv_ds)
    ConstControl(net, element='sgen', variable='p_mw', element_index=pv5, profile_name='pv5', data_source=pv_ds)
    ConstControl(net, element='sgen', variable='p_mw', element_index=pv6, profile_name='pv6', data_source=pv_ds)
    ConstControl(net, element='sgen', variable='p_mw', element_index=pv8, profile_name='pv8', data_source=pv_ds)
    ConstControl(net, element='sgen', variable='p_mw', element_index=pv9, profile_name='pv9', data_source=pv_ds)
    ConstControl(net, element='sgen', variable='p_mw', element_index=pv10, profile_name='pv10', data_source=pv_ds)
    ConstControl(net, element='sgen', variable='p_mw', element_index=pv11, profile_name='pv11', data_source=pv_ds)

    # WT
    wt7 = pp.create_sgen(net, buses[6], 0.0, q_mvar=0, name='WKA 7',type='WP')
    ConstControl(net, element='sgen', variable='p_mw', element_index=wt7, profile_name='wt7', data_source=wt_ds)

    # --- Generators ---
    # mgt5 = pp.create_sgen(net, bus=buses[4], p_mw=0.0, name='MGT 5')
    # mgt9 = pp.create_sgen(net, bus=buses[8], p_mw=0.0, name='MGT 9')
    # mgt10 = pp.create_sgen(net, bus=buses[9], p_mw=0.0, name='MGT 10')

    # --- Batteries ---
    bat5 = pp.create_storage(net, bus=buses[4], p_mw=0.0, max_e_mwh=E_B5_MAX, name='Battery 5', type='Battery', max_p_mw=P_B5_MAX, min_p_mw=P_B5_MIN)
    bat10 = pp.create_storage(net, bus=buses[9], p_mw=0.0, max_e_mwh=E_B10_MAX, name='Battery 10', type='Battery', max_p_mw=P_B10_MAX, min_p_mw=P_B10_MIN)

    # --- Loads ---
    load_r1 = pp.create_load_from_cosphi(net, buses[0], 0.0, 0.98, "underexcited", name='Load R1')
    load_r3 = pp.create_load_from_cosphi(net, buses[2], 0.0, 0.97, "underexcited", name='Load R3')
    load_r4 = pp.create_load_from_cosphi(net, buses[3], 0.0, 0.97, "underexcited", name='Load R4')
    load_r5 = pp.create_load_from_cosphi(net, buses[4], 0.0, 0.97, "underexcited", name='Load R5')
    load_r6 = pp.create_load_from_cosphi(net, buses[5], 0.0, 0.97, "underexcited", name='Load R6')
    load_r8 = pp.create_load_from_cosphi(net, buses[7], 0.0, 0.97, "underexcited", name='Load R8')
    load_r10 = pp.create_load_from_cosphi(net, buses[9], 0.0, 0.97, "underexcited", name='Load R10')
    load_r11 = pp.create_load_from_cosphi(net, buses[10], 0.0, 0.97, "underexcited", name='Load R11')
    ConstControl(net, element='load', variable='p_mw', element_index=load_r1, profile_name='load_r1', data_source=load_ds)
    ConstControl(net, element='load', variable='p_mw', element_index=load_r3, profile_name='load_r3', data_source=load_ds)
    ConstControl(net, element='load', variable='p_mw', element_index=load_r4, profile_name='load_r4', data_source=load_ds)
    ConstControl(net, element='load', variable='p_mw', element_index=load_r5, profile_name='load_r5', data_source=load_ds)
    ConstControl(net, element='load', variable='p_mw', element_index=load_r6, profile_name='load_r6', data_source=load_ds)
    ConstControl(net, element='load', variable='p_mw', element_index=load_r8, profile_name='load_r8', data_source=load_ds)
    ConstControl(net, element='load', variable='p_mw', element_index=load_r10, profile_name='load_r10', data_source=load_ds)
    ConstControl(net, element='load', variable='p_mw', element_index=load_r11, profile_name='load_r11', data_source=load_ds)

    ids = {
        'trafo0': trafo0,
        'pv3': pv3, 'pv4': pv4, 'pv5': pv5, 'pv6': pv6, 'pv8': pv8, 'pv9': pv9, 'pv10': pv10, 'pv11': pv11,
        'wt7': wt7,
        # 'mgt5': mgt5, 'mgt9': mgt9, 'mgt10': mgt10,
        'bat5': bat5, 'bat10': bat10,
        'load_r1': load_r1, 'load_r3': load_r3, 'load_r4': load_r4, 'load_r5': load_r5, 'load_r6': load_r6, 'load_r8': load_r8, 'load_r10': load_r10, 'load_r11': load_r11
    }

    return net, ids