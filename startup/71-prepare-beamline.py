print(ttime.ctime() + ' >>>> ' + __file__)

import sys
import numpy as np
from xas.xray import energy2angle
from scipy import interpolate



# def _compute_hhmy_value(energy):
#     # tabulation is done using data collected on 2021-09-12 and 2021-09-13
#     # data is stored in two files:
#     # /nsls2/xf08id/Sandbox/Beamline_components/2021_09_09_beamline_tabulation/beamline_hhmy_hhrmy_tabulation.json
#     # /nsls2/xf08id/Sandbox/Beamline_components/2021_09_09_beamline_tabulation/beamline_hhmy_hhrmy_tabulation_high_energies.json
#     # energy_tab = np.array([ 4800,  5000,  6000,  7000,  8000,  9000, 10000, 11000, 12000, 13000, 15000, 17500, 20000, 22500, 25000, 27500, 30000])
#     # hhmy_tab = np.array([9.71275, 9.62515, 9.3645 , 9.202  , 9.09365, 9.03125, 8.99405, 8.9373 , 8.9378 , 8.86945, 8.8442 , 8.68795, 8.70085, 8.62675, 8.65015, 8.55135, 8.55645])
#
#     # tabulation is done using data collected on 2021-09-12 and 2021-09-13
#     # data is stored in two files:
#     # /nsls2/xf08id/Sandbox/Beamline_components/2022_02_10_beamline_tabulation/beamline_hhmy_tabulation_att2.json
#     # /nsls2/xf08id/Sandbox/Beamline_components/2022_02_10_beamline_tabulation/beamline_hhmy_tabulation_att2_high_energies.json
#     energy_tab = np.array([ 4900,  5100,  5500,  6000,  7000,  8000,  9000, 10000, 11000, 12000, 13000, 15000, 17500, 20000, 22500, 25000, 27500, 30000])
#     hhmy_tab = np.array([10.50305, 10.4034, 10.2787, 10.1834, 10.03375, 9.93225, 9.8564, 9.8066, 9.7828, 9.7352, 9.71385, 9.6686, 9.6497, 9.60675, 9.532, 9.4319, 9.457, 9.2823])
#
#     def get_matrix_from_energy(energy_in, offset=0):
#         theta_deg = energy2angle(energy_in)
#         V = 1 / np.cos(np.deg2rad(theta_deg - offset))
#         A = np.vstack((V, energy_in, np.ones(V.size))).T
#         return A
#
#     def fit_hhmy(offset=0):
#         A = get_matrix_from_energy(energy_tab, offset=offset)
#         c, _, _, _ = np.linalg.lstsq(A, hhmy_tab, rcond=-1)
#         return c
#
#     coefs = fit_hhmy()
#     a = get_matrix_from_energy(np.array(energy))
#     # the correction is added following the Toyama visit in April 2023
#     f = interpolate.interp1d([4900, 20000], [0.56, 0.79], kind='linear', fill_value='extrapolate') #old values 2024-08-04 [1.1, 0.95]
#
#     offset = f(energy)# + 0.2 # 0.2 was added in Oct 2023 to correct the offset after August 2023 power shutdown
#     return (a @ coefs - offset)

'''
'''
def _compute_hhmy_value(energy):
    # tabulation is done using data collected on 2021-09-12 and 2021-09-13
    # data is stored in two files:
    # /nsls2/xf08id/Sandbox/Beamline_components/2021_09_09_beamline_tabulation/beamline_hhmy_hhrmy_tabulation.json
    # /nsls2/xf08id/Sandbox/Beamline_components/2021_09_09_beamline_tabulation/beamline_hhmy_hhrmy_tabulation_high_energies.json
    # energy_tab = np.array([ 4800,  5000,  6000,  7000,  8000,  9000, 10000, 11000, 12000, 13000, 15000, 17500, 20000, 22500, 25000, 27500, 30000])
    # hhmy_tab = np.array([9.71275, 9.62515, 9.3645 , 9.202  , 9.09365, 9.03125, 8.99405, 8.9373 , 8.9378 , 8.86945, 8.8442 , 8.68795, 8.70085, 8.62675, 8.65015, 8.55135, 8.55645])

    # tabulation is done using data collected on 2021-09-12 and 2021-09-13
    # data is stored in two files:
    # /nsls2/xf08id/Sandbox/Beamline_components/2022_02_10_beamline_tabulation/beamline_hhmy_tabulation_att2.json
    # /nsls2/xf08id/Sandbox/Beamline_components/2022_02_10_beamline_tabulation/beamline_hhmy_tabulation_att2_high_energies.json
    energy_tab = np.array([ 4900,  5100,  5500,  6000,  7000,  8000,  9000, 10000, 11000, 12000, 13000, 15000, 17500, 20000])
    hhmy_tab = np.array([9.937850000000001,
                         9.57315,
                         9.467450000000001,
                         9.3162,
                         9.177100000000001,
                         9.08755,
                         9.025,
                         8.978050000000001,
                         8.9411,
                         8.9000,
                         8.88475,
                         8.84365,
                         8.85205,
                         8.816450000000001])

    f = interpolate.interp1d(energy_tab, hhmy_tab, kind='cubic', fill_value='extrapolate')

    return f(energy)
    #
    # def get_matrix_from_energy(energy_in, offset=0):
    #     theta_deg = energy2angle(energy_in)
    #     V = 1 / np.cos(np.deg2rad(theta_deg - offset))
    #     A = np.vstack((V, energy_in, np.ones(V.size))).T
    #     return A
    #
    # def fit_hhmy(offset=0):
    #     A = get_matrix_from_energy(energy_tab, offset=offset)
    #     c, _, _, _ = np.linalg.lstsq(A, hhmy_tab, rcond=-1)
    #     return c
    #
    # coefs = fit_hhmy()
    # a = get_matrix_from_energy(np.array(energy))
    # # the correction is added following the Toyama visit in April 2023
    # f = interpolate.interp1d([4900, 20000], [0.56, 0.79], kind='linear', fill_value='extrapolate') #old values 2024-08-04 [1.1, 0.95]
    #
    # offset = f(energy)# + 0.2 # 0.2 was added in Oct 2023 to correct the offset after August 2023 power shutdown
    # return (a @ coefs - offset)


'''
'''

bl_prepare_energy_ranges = [
        {
            'energy_start': 4500,
            'energy_end': 6000,
            'He_flow': 4,  #5 old value 2024-09-06
            'N2_flow': 1,
            'IC_voltage': 1000,
            'HHRM': 4,
            'CM1':0,# 0,
            'Filterbox': 1,
            'ES BPM exposure': 0.4,
            'i0_gain': 5,
            'it_gain': 5,
            'ir_gain': 5,
        },
        {
            'energy_start': 5700,
            'energy_end': 10000,
            'He_flow': 0, #5 old value 2024-09-06
            'N2_flow': 5,
            'IC_voltage': 1650,
            'HHRM': 5,
            'CM1':0,# 0,
            'Filterbox': -69,
            'ES BPM exposure': 0.5, #0.05, # Denis 2024-06-02: changed to a higher value because of the beamline state
            'i0_gain': 5,
            'it_gain': 5,
            'ir_gain': 5,
        },
        {
            'energy_start': 10000,
            'energy_end': 13000,
            'He_flow': 0, #5 old value 2024-09-06
            'N2_flow': 5,
            'IC_voltage': 1650,
            'HHRM': 75, # IS THIS SUPPOSED TO BE 80?
            'CM1':0,# 0,
            'Filterbox': -139,
            'ES BPM exposure': 0.5,
            'i0_gain': 5,
            'it_gain': 5,
            'ir_gain': 5,
        },
        {
            'energy_start': 13000,
            'energy_end': 17000,
            'He_flow': 0, #5 old value 2024-09-06
            'N2_flow': 5,
            'IC_voltage': 1650,
            'HHRM': 75,
            'CM1': 40,# 0,
            'Filterbox': -139,
            'ES BPM exposure': 0.7,
            'i0_gain': 5,
            'it_gain': 5,
            'ir_gain': 5,
        },
        {
            'energy_start': 17000,
            'energy_end': 25000,
            'He_flow': 0, #2 old value 2024-09-06
            'N2_flow': 5,
            'IC_voltage': 1650,
            'HHRM': 75,
            'CM1': 40,# 0,
            'Filterbox': -209,
            'ES BPM exposure': 1.0,
            'i0_gain': 5,
            'it_gain': 6,
            'ir_gain': 6,

        },
        {
            'energy_start': 25000,
            'energy_end': 35000,
            'He_flow': 2,
            'N2_flow': 5,
            'IC_voltage': 1650,
            'HHRM': 75,
            'CM1': 40,# 0,
            'Filterbox': -209,
            'ES BPM exposure': 1.5,
            'i0_gain': 5,
            'it_gain': 6,
            'ir_gain': 6,

        },
    ]

def prepare_beamline_plan(energy: int = -1, move_cm_mirror = False, move_hhm_y=True):
    energy_ranges = bl_prepare_energy_ranges

    BPM_exposure_setter = bpm_es.exp_time
    He_flow_setter = gas_he.flow
    N2_flow_setter = gas_n2.flow
    high_voltage_setters = [wps1.hv302, wps1.hv303, wps1.hv305]
    safe_high_voltage = 580
    filter_box_setter = filterbox.y
    cm_setter = cm1.x
    hhrm_setter = hhrm.hor_translation
    settling_time = 120

    if type(energy) == str:
        energy = int(energy)

    energy_range = [e_range for e_range in energy_ranges if
                  e_range['energy_end'] > energy >= e_range['energy_start']][0]
    if not energy_range:
        # print_to_gui('ERROR: Energy is outside of the beamline energy range')
        print('ERROR: Energy is outside of the beamline energy range')
        return


    current_filterbox_position = filterbox.y.read()[filterbox.y.name]['value']
    if (abs(energy_range['Filterbox'] - current_filterbox_position)) < 0.1:
        move_filter = False
    else:
        move_filter = True


    print_to_gui(f'[Prepare Beamline] Starting setting up the beamline to {energy} eV...')
    if move_cm_mirror == True:
        start_cm_position = cm_setter.position
        end_cm_position = energy_range['CM1']
        cm_motion_range = abs(end_cm_position-start_cm_position)
        moving_cm = cm_setter.set(end_cm_position)
    #
    start_hhrm_position = hhrm_setter.position
    end_hhrm_position = energy_range['HHRM']
    hhrm_motion_range = abs(start_hhrm_position-end_hhrm_position)
    moving_hhrm = hhrm_setter.set(end_hhrm_position)
    #
    print_to_gui('[Prepare Beamline] Setting high voltage supply to safe values...')

    hv_setter_values = []
    for high_voltage_setter in high_voltage_setters:
        hv_setter_values.append(high_voltage_setter)
        hv_setter_values.append(safe_high_voltage)
    yield from bps.mv(*hv_setter_values)
    print_to_gui('[Prepare Beamline] High voltage supply is set to safe values')


    start_time = ttime.time()
    print_to_gui('[Prepare Beamline] Setting ion chamber gas composition...')
    yield from bps.mv(
                        He_flow_setter, energy_range['He_flow'],
                        N2_flow_setter, energy_range['N2_flow'],
                      )
    print_to_gui('[Prepare Beamline] Ion chamber gas composition set')



    #close shutter before moving the filter


    if move_filter:
        print_to_gui('[Prepare Beamline] Closing frontend shutter before selecting filter')
        print('moving')
        # close shutter before moving the filter
        reopen_shutter = False
        if shutter_fe_2b.status.get() != 'Not Open':
            reopen_shutter = True
            try:
                yield from bps.mv(shutter_fe_2b, 'Close')
            except FailedStatus:
                raise CannotActuateShutter(f'Error: Photon shutter failed to close.')

        yield from bps.mv(filter_box_setter, energy_range['Filterbox'])
        print_to_gui('[Prepare Beamline] Filter set')
        print_to_gui('[Prepare Beamline] Closing frontend shutter before selecting filter')

        if reopen_shutter:
            try:
                yield from bps.mv(shutter_fe_2b, 'Open')
            except FailedStatus:
                print_to_gui(f'Error: Photon shutter failed to open.')

    if move_hhm_y:
        print_to_gui('[Prepare Beamline] Moving vertical position of the second monochromator crystal')
        hhmy_position = _compute_hhmy_value(energy)
        yield from bps.mv(hhm.y_precise, hhmy_position)
        if np.abs(hhm.y_precise.user_readback.get() - hhmy_position)>0.05:
            yield from bps.mv(hhm.y_precise, hhmy_position)


    while ttime.time() < (start_time + settling_time):
        print_to_gui(f'[Prepare Beamline] {int(settling_time - (ttime.time()-start_time))} s left to settle the ion chamber gas flow')
        yield from bps.sleep(10)
    print_to_gui('[Prepare Beamline] Setting high voltage values')
    hv_setter_values = []
    for high_voltage_setter in high_voltage_setters:
        hv_setter_values.append(high_voltage_setter)
        hv_setter_values.append(energy_range['IC_voltage'])
    yield from bps.mv(*hv_setter_values)
    print_to_gui('[Prepare Beamline] High voltage values set')

    while not moving_hhrm.done:
        motion_so_far = hhrm_setter.position
        percent_complete = int(abs(motion_so_far - start_hhrm_position) / hhrm_motion_range * 100)
        print_to_gui(f'[Prepare Beamline] HHRM motion is {percent_complete} % complete')
        yield from bps.sleep(10)

    print_to_gui('[Prepare Beamline] High harmonics rejection mirror position set')

#
    if move_cm_mirror == True:
        while not moving_cm.done:
            motion_so_far = cm_setter.position
            percent_complete = int(abs(motion_so_far - start_cm_position) / cm_motion_range * 100)
            print_to_gui(f'[Prepare Beamline] CM1 motion is {percent_complete} % set')
            yield from bps.sleep(10)
        print_to_gui('[Prepare Beamline] CM1 mirror position set')

    print_to_gui('[Prepare Beamline] Moving to the target energy')
    yield from bps.mv(hhm.energy, energy)

    print_to_gui('[Prepare Beamline] Adjusting IC amplifier gains')
    yield from i0_amp.set_gain_plan(energy_range['i0_gain'], True)
    yield from it_amp.set_gain_plan(energy_range['it_gain'], True)
    yield from ir_amp.set_gain_plan(energy_range['ir_gain'], True)
    # Additon to account for higher flows

    yield from bps.mv(
                        He_flow_setter, energy_range['He_flow']/10,
                        N2_flow_setter, energy_range['N2_flow']/10,
                      )

    print_to_gui('[Prepare Beamline] Ion chamber gas composition set')

    print_to_gui('[Prepare Beamline] Adjusting exposure on the monitor')
    yield from bps.mv(BPM_exposure_setter, energy_range['ES BPM exposure'])
    print_to_gui('[Prepare Beamline] Beamline preparation is complete')

    if move_hhm_y:
        if np.abs(hhm.y_precise.user_readback.get() - hhmy_position) > 0.05:
            print_to_gui(f'Error: vertical position of the second monochromator crystal is not set. Set manually to {hhmy_position}')




def simple_prepare_beamline_plan(energy: int = -1, move_cm_mirror = False, move_hhm_y=False):
    energy_ranges = bl_prepare_energy_ranges

    BPM_exposure_setter = bpm_es.exp_time
    # He_flow_setter = gas_he.flow
    # N2_flow_setter = gas_n2.flow
    # high_voltage_setters = [wps1.hv302, wps1.hv303, wps1.hv305]
    # safe_high_voltage = 580
    filter_box_setter = filterbox.y
    cm_setter = cm1.x
    # hhrm_setter = hhrm.hor_translation
    # settling_time = 120

    if type(energy) == str:
        energy = int(energy)

    energy_range = [e_range for e_range in energy_ranges if
                  e_range['energy_end'] > energy >= e_range['energy_start']][0]
    if not energy_range:
        # print_to_gui('ERROR: Energy is outside of the beamline energy range')
        print('ERROR: Energy is outside of the beamline energy range')
        return


    current_filterbox_position = filterbox.y.read()[filterbox.y.name]['value']
    if (abs(energy_range['Filterbox'] - current_filterbox_position)) < 0.1:
        move_filter = False
    else:
        move_filter = True


    print_to_gui(f'[Prepare Beamline] Starting setting up the beamline to {energy} eV...')
    if move_cm_mirror == True:
        start_cm_position = cm_setter.position
        end_cm_position = energy_range['CM1']
        cm_motion_range = abs(end_cm_position-start_cm_position)
        moving_cm = cm_setter.set(end_cm_position)
    #
    # start_hhrm_position = hhrm_setter.position
    # end_hhrm_position = energy_range['HHRM']
    # hhrm_motion_range = abs(start_hhrm_position-end_hhrm_position)
    # moving_hhrm = hhrm_setter.set(end_hhrm_position)
    #
    # print_to_gui('[Prepare Beamline] Setting high voltage supply to safe values...')
    #
    # hv_setter_values = []
    # for high_voltage_setter in high_voltage_setters:
    #     hv_setter_values.append(high_voltage_setter)
    #     hv_setter_values.append(safe_high_voltage)
    # yield from bps.mv(*hv_setter_values)
    # print_to_gui('[Prepare Beamline] High voltage supply is set to safe values')


    # start_time = ttime.time()
    # print_to_gui('[Prepare Beamline] Setting ion chamber gas composition...')
    # yield from bps.mv(
    #                     He_flow_setter, energy_range['He_flow'],
    #                     N2_flow_setter, energy_range['N2_flow'],
    #                   )
    # print_to_gui('[Prepare Beamline] Ion chamber gas composition set')



    #close shutter before moving the filter


    if move_filter:
        print_to_gui('[Prepare Beamline] Closing frontend shutter before selecting filter')
        print('moving')
        # close shutter before moving the filter
        try:
            yield from bps.mv(shutter_fe_2b, 'Close')
        except FailedStatus:
            raise CannotActuateShutter(f'Error: Photon shutter failed to close.')

        yield from bps.mv(filter_box_setter, energy_range['Filterbox'])
        print_to_gui('[Prepare Beamline] Filter set')
        print_to_gui('[Prepare Beamline] Closing frontend shutter before selecting filter')

        try:
            yield from bps.mv(shutter_fe_2b, 'Open')
        except FailedStatus:
            print_to_gui(f'Error: Photon shutter failed to open.')

    # if move_hhm_y:
    #     print_to_gui('[Prepare Beamline] Moving vertical position of the second monochromator crystal')
    #     hhmy_position = _compute_hhmy_value(energy)
    #     yield from bps.mv(hhm.y_precise, hhmy_position)
    #     if np.abs(hhm.y_precise.user_readback.get() - hhmy_position)>0.05:
    #         yield from bps.mv(hhm.y_precise, hhmy_position)


    # while ttime.time() < (start_time + settling_time):
    #     print_to_gui(f'[Prepare Beamline] {int(settling_time - (ttime.time()-start_time))} s left to settle the ion chamber gas flow')
    #     yield from bps.sleep(10)
    # print_to_gui('[Prepare Beamline] Setting high voltage values')
    # hv_setter_values = []
    # for high_voltage_setter in high_voltage_setters:
    #     hv_setter_values.append(high_voltage_setter)
    #     hv_setter_values.append(energy_range['IC_voltage'])
    # yield from bps.mv(*hv_setter_values)
    # print_to_gui('[Prepare Beamline] High voltage values set')

    # while not moving_hhrm.done:
    #     motion_so_far = hhrm_setter.position
    #     percent_complete = int(abs(motion_so_far - start_hhrm_position) / hhrm_motion_range * 100)
    #     print_to_gui(f'[Prepare Beamline] HHRM motion is {percent_complete} % complete')
    #     yield from bps.sleep(10)
    #
    # print_to_gui('[Prepare Beamline] High harmonics rejection mirror position set')

#
    if move_cm_mirror == True:
        while not moving_cm.done:
            motion_so_far = cm_setter.position
            percent_complete = int(abs(motion_so_far - start_cm_position) / cm_motion_range * 100)
            print_to_gui(f'[Prepare Beamline] CM1 motion is {percent_complete} % set')
            yield from bps.sleep(10)
        print_to_gui('[Prepare Beamline] CM1 mirror position set')

    print_to_gui('[Prepare Beamline] Moving to the target energy')
    yield from bps.mv(hhm.energy, energy)

    print_to_gui('[Prepare Beamline] Adjusting exposure on the monitor')
    yield from bps.mv(BPM_exposure_setter, energy_range['ES BPM exposure'])
    # bpm_es.adjust_camera_exposure_time()
    print_to_gui('[Prepare Beamline] Beamline preparation is complete')

    # if move_hhm_y:
    #     if np.abs(hhm.y_precise.user_readback.get() - hhmy_position) > 0.05:
    #         print_to_gui(f'Error: vertical position of the second monochromator crystal is not set. Set manually to {hhmy_position}')




# def adjust_filter_plan(energy: int = -1, energy_ranges=bl_prepare_energy_ranges, plan_description='HHM_Y tabulating'):
#     filter_box_setter = filterbox.y
#     current_filterbox_position = filterbox.y.read()[filterbox.y.name]['value']
#
#     energy_range = [e_range for e_range in energy_ranges if
#                     e_range['energy_end'] > energy >= e_range['energy_start']][0]
#     if not energy_range:
#         print_to_gui('ERROR: Energy is outside of the beamline energy range', stdout=stdout)
#         return
#
#     if (abs(energy_range['Filterbox'] - current_filterbox_position)) < 0.1:
#         move_filter = False
#     else:
#         move_filter = True
#
#     if move_filter:
#         print_to_gui(f'[{plan_description}] Closing frontend shutter before selecting filter', stdout=stdout)
#         print('moving')
#         # close shutter before moving the filter
#         try:
#             yield from bps.mv(shutter_fe_2b, 'Close')
#         except FailedStatus:
#             raise CannotActuateShutter(f'Error: Photon shutter failed to close.')
#
#         yield from bps.mv(filter_box_setter,energy_range['Filterbox'])
#         print_to_gui(f'[{plan_description}] Filter set',stdout=stdout)
#         print_to_gui(f'[{plan_description}] Closing frontend shutter before selecting filter',stdout=stdout)
#
#         try:
#             yield from bps.mv(shutter_fe_2b, 'Open')
#         except FailedStatus:
#             print_to_gui(f'Error: Photon shutter failed to open.',stdout=stdout)


# Old beamline prepare parameters

# bl_prepare_energy_ranges = [
#         {
#             'energy_start': 4500,
#             'energy_end': 6000,
#             'He_flow': 4,  #5 old value 2024-09-06
#             'N2_flow': 1,
#             'IC_voltage': 1000,
#             'HHRM': 4,
#             'CM1':0,# 0,
#             'Filterbox': 1,
#             'ES BPM exposure': 0.05,
#             'i0_gain': 5,
#             'it_gain': 5,
#             'ir_gain': 5,
#         },
#         {
#             'energy_start': 5700,
#             'energy_end': 10000,
#             'He_flow': 0, #5 old value 2024-09-06
#             'N2_flow': 3,
#             'IC_voltage': 1650,
#             'HHRM': 5,
#             'CM1':0,# 0,
#             'Filterbox': -69,
#             'ES BPM exposure': 0.1, #0.05, # Denis 2024-06-02: changed to a higher value because of the beamline state
#             'i0_gain': 4,
#             'it_gain': 4,
#             'ir_gain': 5,
#         },
#         {
#             'energy_start': 10000,
#             'energy_end': 13000,
#             'He_flow': 0, #5 old value 2024-09-06
#             'N2_flow': 5,
#             'IC_voltage': 1650,
#             'HHRM': 75, # IS THIS SUPPOSED TO BE 80?
#             'CM1':0,# 0,
#             'Filterbox': -139,
#             'ES BPM exposure': 0.1,
#             'i0_gain': 4,
#             'it_gain': 4,
#             'ir_gain': 5,
#         },
#         {
#             'energy_start': 13000,
#             'energy_end': 17000,
#             'He_flow': 0, #5 old value 2024-09-06
#             'N2_flow': 5,
#             'IC_voltage': 1650,
#             'HHRM': 75,
#             'CM1': 40,# 0,
#             'Filterbox': -139,
#             'ES BPM exposure': 0.2,
#             'i0_gain': 5,
#             'it_gain': 5,
#             'ir_gain': 5,
#         },
#         {
#             'energy_start': 17000,
#             'energy_end': 25000,
#             'He_flow': 0, #2 old value 2024-09-06
#             'N2_flow': 5,
#             'IC_voltage': 1650,
#             'HHRM': 75,
#             'CM1': 40,# 0,
#             'Filterbox': -209,
#             'ES BPM exposure': 0.5,
#             'i0_gain': 5,
#             'it_gain': 6,
#             'ir_gain': 6,
#
#         },
#         {
#             'energy_start': 25000,
#             'energy_end': 35000,
#             'He_flow': 2,
#             'N2_flow': 5,
#             'IC_voltage': 1650,
#             'HHRM': 75,
#             'CM1': 40,# 0,
#             'Filterbox': -209,
#             'ES BPM exposure': 0.8,
#             'i0_gain': 5,
#             'it_gain': 6,
#             'ir_gain': 6,
#
#         },
#     ]