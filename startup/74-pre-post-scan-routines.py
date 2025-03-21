print(ttime.ctime() + ' >>>> ' + __file__)

from datetime import datetime
import time

import numpy as np
from bluesky.plan_stubs import mv, mvr
from random import random
import json
from xas.ft_analysis import data_ft

from xas.energy_calibration import find_correct_foil


class CannotActuateShutter(Exception):
    pass

def actuate_photon_shutter_plan(state):
    try:
        yield from bps.mv(shutter_ph_2b, state)
    # except FailedStatus:
    #     raise CannotActuateShutter(f'Error: Photon shutter failed to {state.lower()}.')
    except Exception as e:
        print_to_gui(f'{str(e)}', tag='SHUTTER DEBUG')
        yield from bps.null()

def get_offsets_plan(time : float = 2):


    apb_ave.save_current_status()
    prepare_detectors_for_exposure_plan([apb_ave], n_exposures=1)
    set_detector_exposure_time_plan([apb_ave], time)

    yield from actuate_photon_shutter_plan('Close')

    yield from current_suppression_plan()  # added for Keithley Amplifier
    uid = (yield from bp.count([apb_ave], 1, md={"plan_name": "get_offsets_plan"}))
    yield from actuate_photon_shutter_plan('Open')

    yield from apb_ave.restore_to_saved_status()

    table = db[uid].table()

    for i in range(0,8):
        mean =  float(table[f'apb_ave_ch{i+1}_mean'])
        print_to_gui(f'Mean {(mean)}')
        ch_offset = getattr(apb_ave, f'ch{i+1}_offset')
        yield from bps.abs_set(ch_offset, mean)


def check_photon_shutter_plan():
    if shutter_ph_2b.status.get() == shutter_ph_2b.close_val:
        print_to_gui(f'Attempting to open PH shutter.', add_timestamp=True, tag='Beamline')
        yield from actuate_photon_shutter_plan('Open')


# def record_offsets_plan(suffix=''):
#     fpath = '/nsls2/xf08id/log/offsets/' + str(datetime.now()).replace(':', '-')[:-7] + suffix + '.dat'
#     uid = (yield from get_offsets())
#     table = db[uid].table()
#     table.to_csv(fpath)


# def record_offsets_for_all_gains_plan():
#     amps = [i0_amp, it_amp, ir_amp, iff_amp]
#     # set_gains_plan(*args)
#     for gain_value in range(3, 8):
#
#         for amp in amps:
#             yield from amp.set_gain_plan(gain_value, bool(0))
#         # ttime.sleep(0.5)
#
#         # output = str(gain_value)
#         # for amp in amps:
#         #     output += ' ' + amp.name + ' ' + str(amp.get_gain())
#         # print(output)
#         suffix = f' gain-{gain_value}'
#
#         yield from bps.sleep(60)
#         yield from record_offsets_plan(suffix=suffix)


def general_set_gains_plan(*args):
    """
    Parameters
    ----------
    Groups of three parameters: amplifier, gain, hs

    Example: set_gains_and_offsets(i0_amp, 5, False, it_amp, 4, False, iff_amp, 5, True)
    """

    mod = len(args) % 3
    if mod:
        args = args[:-mod]

    for ic_amp, val, hs in zip([ic for index, ic in enumerate(args) if index % 3 == 0],
                       [val for index, val in enumerate(args) if index % 3 == 1],
                       [hs for index, hs in enumerate(args) if index % 3 == 2]):
        yield from ic_amp.set_gain_plan(val, hs)

        # if type(ic_amp) != ICAmplifier:
        if type(ic_amp) != ICAmplifier_Keithley:
            raise Exception('Wrong type: {} - it should be ICAmplifier'.format(type(ic)))
        if type(val) != int:
            raise Exception('Wrong type: {} - it should be int'.format(type(val)))
        if type(hs) != bool:
            raise Exception('Wrong type: {} - it should be bool'.format(type(hs)))

        print_to_gui('set amplifier gain for {}: {}, {}'.format(ic_amp.name, val, hs))


def set_gains_plan(i0_gain: int = 5, it_gain: int = 5, ir_gain: int = 5, iff_gain: int = 5, hs: bool = False):
    i0_gain = int(i0_gain)
    it_gain = int(it_gain)
    iff_gain = int(iff_gain)
    ir_gain = int(ir_gain)
    if type(hs) == str:
        hs = hs == 'True'

    yield from general_set_gains_plan(i0_amp, i0_gain, hs, it_amp, it_gain, hs, iff_amp, iff_gain, hs, ir_amp, ir_gain, hs)
    # yield from general_set_gains_plan(i0_amp, i0_gain, hs, it_amp, it_gain, hs, iff_amp, iff_gain, hs, ir_amp, ir_gain, hs)
    yield from get_offsets_plan()


def prepare_trajectory_plan(trajectory_filename, offset=None):
    trajectory_stack.set_trajectory(trajectory_filename, offset=offset)
    yield from bps.null()

def prepare_scan_plan(scan_uid=None, aux_parameters=None):
    scan = scan_manager.scan_dict[scan_uid]
    if scan['scan_type'] == 'fly scan':
        mono_angle_offset = aux_parameters['offset']
        trajectory_filename = scan['scan_parameters']['filename']
        yield from prepare_trajectory_plan(trajectory_filename, offset=mono_angle_offset)


def optimize_gains_plan(n_tries=3, trajectory_filename=None, mono_angle_offset=None):
    # sys.stdout = kwargs.pop('stdout', sys.stdout)

    # if 'detector_names' not in kwargs:
    # detectors = [pba1.adc7, pba2.adc6, pba1.adc1, pba1.adc6]
    detectors = [apb_ave]
    channels = [ apb_ave.ch1,  apb_ave.ch2,  apb_ave.ch3,  apb_ave.ch4]
    # offsets = [apb.ch1_offset, apb.ch2_offset, apb.ch3_offset, apb.ch4_offset]

    if trajectory_filename is not None:
        yield from prepare_trajectory_plan(trajectory_filename, offset=mono_angle_offset)
        # trajectory_stack.set_trajectory(trajectory_filename, offset=mono_angle_offset)

    threshold_hi = 3.250
    threshold_lo = 0.250

    e_min, e_max = trajectory_manager.read_trajectory_limits()
    scan_positions = np.arange(e_max + 50, e_min - 50, -200).tolist()
    print_to_gui(f'Starting gain optimization', add_timestamp = True, tag = 'Gain optimization')
    yield from actuate_photon_shutter_plan('Open')
    print_to_gui(f'Photon shutter open', add_timestamp = True, tag = 'Gain optimization')
    yield from shutter.open_plan()

    for jj in range(n_tries):
        print_to_gui(f'Starting try {jj+1}', add_timestamp = True, tag = 'Gain optimization')
        plan = bp.list_scan(detectors, hhm.energy, scan_positions)
        yield from plan
        table = db[-1].table()

        all_gains_are_good = True

        for channel in channels:
            current_gain = channel.amp.get_gain()[0]
            if channel.polarity == 'neg':
                trace_extreme = table[channel.name].min()
            else:
                trace_extreme = table[channel.name].max()

            trace_extreme = trace_extreme / 1000

            print_to_gui(f'Extreme value {trace_extreme} for detector {channel.name}',add_timestamp = True, tag = 'Gain optimization')
            if abs(trace_extreme) > threshold_hi:
                print_to_gui(f'Decreasing gain for detector {channel.name}',add_timestamp = True, tag = 'Gain optimization')
                yield from channel.amp.set_gain_plan(current_gain - 1, False)
                all_gains_are_good = False
            elif abs(trace_extreme) <= threshold_hi and abs(trace_extreme) > threshold_lo:
                print_to_gui(f'Correct gain for detector {channel.name}', add_timestamp = True, tag = 'Gain optimization')
            elif abs(trace_extreme) <= threshold_lo:
                print_to_gui(f'Increasing gain for detector {channel.name}', add_timestamp = True, tag = 'Gain optimization')
                yield from channel.amp.set_gain_plan(current_gain + 1, False)
                all_gains_are_good = False

        if all_gains_are_good:
            print_to_gui(f'Gains are correct. Taking offsets..', add_timestamp = True, tag = 'Gain optimization')
            break

    yield from shutter.close_plan()
    yield from get_offsets_plan()


def quick_optimize_gains_plan(n_tries=3, trajectory_filename=None, mono_angle_offset=None):
    # sys.stdout = kwargs.pop('stdout', sys.stdout)

    # if 'detector_names' not in kwargs:
    # detectors = [pba1.adc7, pba2.adc6, pba1.adc1, pba1.adc6]
    # detectors = [apb_ave]
    # channels = [ apb_ave.ch1,  apb_ave.ch2,  apb_ave.ch3,  apb_ave.ch4]
    print_to_gui(f'Starting gain optimization')
    channels = [apb.ch1, apb.ch2, apb.ch3, apb.ch4]
    # offsets = [apb.ch1_offset, apb.ch2_offset, apb.ch3_offset, apb.ch4_offset]

    if trajectory_filename is not None:
        yield from prepare_trajectory_plan(trajectory_filename, offset=mono_angle_offset)
        # trajectory_stack.set_trajectory(trajectory_filename, offset=mono_angle_offset)

    threshold_hi = 3.250
    threshold_lo = 0.250

    e_min, e_max = trajectory_manager.read_trajectory_limits()
    e_min -= 50
    e_max += 50
    # scan_positions = np.arange(e_max + 50, e_min - 50, -200).tolist()

    yield from actuate_photon_shutter_plan('Open')
    yield from shutter.open_plan()

    for jj in range(n_tries):
        print_to_gui(f'Starting try {jj + 1}')
        # current_energy = hhm.energy.position
        # e1 = np.abs(current_energy - e_min), np.abs(current_energy - e_max)

        yield from bps.mv(hhm.energy, e_max)
        print_to_gui(f'Moved to max energy')
        @return_NullStatus_decorator
        def _move_energy_plan():
            yield from move_mono_energy(e_min, step=200, beampos_tol=10)

        ramp_plan = ramp_plan_with_multiple_monitors(_move_energy_plan(), [hhm.energy] + channels, bps.null)
        yield from ramp_plan

        # plan = bp.list_scan(detectors, hhm.energy, scan_positions)
        # yield from plan
        table = process_monitor_scan(db, -1)

        all_gains_are_good = True

        for channel in channels:

            # if channel.polarity == 'neg':
            trace_extreme = table[channel.name].min()
            # else:
            #     trace_extreme = table[channel.name].max()

            trace_extreme = trace_extreme / 1000

            print_to_gui(f'Extreme value {trace_extreme} for detector {channel.name}')

            current_gain = channel.amp.get_gain()[0]
            if abs(trace_extreme) > threshold_hi:
                if current_gain > 3:
                    print_to_gui(f'Decreasing gain for detector {channel.name}')
                    yield from channel.amp.set_gain_plan(current_gain - 1, False)
                    all_gains_are_good = False
            elif abs(trace_extreme) <= threshold_hi and abs(trace_extreme) > threshold_lo:
                print_to_gui(f'Correct gain for detector {channel.name}')
            elif abs(trace_extreme) <= threshold_lo:
                if current_gain < 7:
                    print(f'Increasing gain for detector {channel.name}')
                    yield from channel.amp.set_gain_plan(current_gain + 1, False)
                    all_gains_are_good = False

        if all_gains_are_good:
            print(f'Gains are correct. Taking offsets..')
            break

    yield from shutter.close_plan()
    yield from get_offsets_plan()




def set_reference_foil(element:str = 'Mn', edge:str = 'K' ):
    # Adding reference foil element list
    with open(f'{ROOT_PATH_SHARED}/settings/json/foil_wheel.json') as fp:
        reference_foils = json.load(fp)
    elems = [item['element'] for item in reference_foils]


    if element is None:
        yield from mv(foil_wheel.wheel1, 0)
        yield from mv(foil_wheel.wheel2, 0)
    else:
        if element in elems:
            indx = elems.index(element)
            yield from mv(foil_wheel.wheel2, reference_foils[indx]['fw2'])
            yield from mv(foil_wheel.wheel1, reference_foils[indx]['fw1'])
        else:
            new_element, edge, energy = find_correct_foil(element=element, edge=edge)
            if new_element in elems:
                indx = elems.index(new_element)
                yield from mv(foil_wheel.wheel2, reference_foils[indx]['fw2'])
                yield from mv(foil_wheel.wheel1, reference_foils[indx]['fw1'])
            else:
                yield from mv(foil_wheel.wheel1, 0)
                yield from mv(foil_wheel.wheel2, 0)

        #yield from mv(foil_wheel.wheel2, reference[element]['foilwheel2'])
        #yield from mv(foil_wheel.wheel1, reference[element]['foilwheel1'])





def set_attenuator(thickness:int  = 0, **kwargs):
    # Adding reference foil element list
    with open(f'{ROOT_PATH_SHARED}/settings/json/attenuator.json') as fp:
        attenuators_list = json.load(fp)
    thickness_str_list = [item['attenuator'] for item in attenuators_list]
    thickness_str = str(thickness)
    if thickness_str in thickness_str_list:
        indx = thickness_str_list.index(thickness_str)
        yield from mv(attenuator_motor.pos, attenuators_list[indx]['position'])
    else:
        yield from mv(attenuator_motor.pos, 0)



def scan_beam_center(camera=camera_sp1, emin=10000, emax=20000, nsteps=10, roll_range=5, roll_nsteps=6):

    original_roll = hhm.roll.position
    hhm_rolls = np.linspace(original_roll-roll_range/2, original_roll+roll_range/2, roll_nsteps+1)
    energies = np.linspace(emin, emax, nsteps+1).tolist()

    yield from bps.open_run()

    for hhm_roll in hhm_rolls:
        yield from bps.mv(hhm.roll, hhm_roll)
        for energy in energies:
            # yield from bps.mv(hhm.energy, energy)
            yield from move_mono_energy(energy, step=500, beampos_tol=10)
            camera.adjust_camera_exposure_time(percentile=100)
            yield from bps.trigger_and_read([hhm.energy, hhm.roll, camera.stats1.centroid, bpm_es.stats4.centroid])

    yield from bps.close_run()

    yield from bps.mv(hhm.roll, original_roll)



def plot_beam_center_scan(db, uid):
    t = db[uid].table()

    plt.figure(1, clear=True)

    roll = np.unique(t.hhm_roll_user_setpoint.values)
    energy = np.unique(t.hhm_energy_user_setpoint.values)
    x_cam_sp1 = np.zeros((energy.size, roll.size))
    x_bpm_es = np.zeros((energy.size, roll.size))

    for i, _roll in enumerate(roll):
        mask = t.hhm_roll_user_setpoint == _roll
        x_cam_sp1[:, i] = t.camera_sp1_stats1_centroid_x[mask].values
        x_bpm_es[:, i] = t.bpm_es_stats4_centroid_x[mask].values


    plt.subplot(221)
    for i in range(roll.size):
        plt.plot(energy, x_cam_sp1[:, i])
        plt.text(energy[-1]+500, x_cam_sp1[-1, i], f'{roll[i]:0.3f}')

    plt.subplot(222)
    for i in range(roll.size):
        plt.plot(energy, x_bpm_es[:, i])
        plt.text(energy[-1] + 500, x_bpm_es[-1, i], f'{roll[i]:0.3f}')

    plt.subplot(223)
    plt.plot(roll, np.ptp(x_cam_sp1, axis=0))

    plt.subplot(224)
    plt.plot(roll, np.ptp(x_bpm_es, axis=0))

# def quick_pitch_optimization(pitch_range=1, pitch_speed=0.2, n_tries=3):
#     yield from set_hhm_feedback_plan(0)
#     yield from bps.sleep(0.2)
#
#     for i in range(n_tries):
#         current_pitch_value = hhm.pitch.position
#         print_to_gui(f'Starting attempt #{i + 1}', tag='Pitch Scan')
#         yield from bps.mvr(hhm.pitch, -pitch_range / 2)
#         yield from bps.sleep(0.2)
#         @return_NullStatus_decorator
#         def _move_pitch_plan():
#             yield from bps.mvr(hhm.pitch, pitch_range)
#             yield from bps.sleep(0.2)
#
#         yield from bps.mv(hhm.pitch.velocity, pitch_speed)
#         ramp_plan = ramp_plan_with_multiple_monitors(_move_pitch_plan(), [hhm.pitch, apb.ch1], bps.null)
#         yield from ramp_plan
#         yield from bps.mv(hhm.pitch.velocity, 60)
#         new_pitch_pos = find_optimum_pitch_pos(db, -1)
#         print_to_gui(f'New pitch position: {new_pitch_pos}', tag='Pitch Scan')
#         yield from bps.mv(hhm.pitch, new_pitch_pos, wait=True)
#
#         min_threshold = current_pitch_value - pitch_range / 2 + pitch_range / 10
#         max_threshold = current_pitch_value + pitch_range / 2 - pitch_range / 10
#
#         # print(f'PITCH RESULTS:\ncurrent_pitch_value={current_pitch_value}\nmin_threshold={min_threshold}\nnew_pitch_pos={new_pitch_pos}\nmax_threshold={max_threshold}')
#
#         if (new_pitch_pos > min_threshold) and (new_pitch_pos < max_threshold):
#             break
#         else:
#             if (i+1) > n_tries:
#                 print_to_gui(f'Exceeded number of attempts. Adjust pitch manually.', tag='Pitch Scan')
#
#     yield from set_hhm_feedback_plan(1, update_center=True)

def quick_pitch_optimization(scan_range=1, velocity=0.2, n_tries=3):
    yield from set_hhm_feedback_plan(0)
    yield from bps.sleep(0.2)

    yield from quick_tuning_scan(motor=hhm.pitch.name, detector='I0 ion Chamber instantaneous', channel='apb_ch1',
                                 scan_range=scan_range, velocity=velocity, n_tries=n_tries)
    yield from bps.sleep(bpm_es.exp_time.get())
    #TODO remove comment from next line when the beamline is ok
    #yield from set_hhm_feedback_plan(1, update_center=True)


# plot_beam_center_scan(db, -1)
