print(ttime.ctime() + ' >>>> ' + __file__)
import copy
import uuid

import numpy as np
from PyQt5 import QtGui
from redis_json_dict import RedisJSONDict
from xas.trajectory import TrajectoryCreator
from xas.xray import generate_energy_grid_from_dict, generate_emission_energy_grid_from_dict, generate_emission_relative_trajectory_from_dict
import os
from xas.xray import e2k
import logging
import logging.handlers
from collections import Counter

# def emit_list_update_signal_decorator(method):
#     def wrapper(obj, *args, emit_signal=True, **kwargs):
#         result = method(obj, *args, **kwargs)
#         if emit_signal:
#             # print_to_gui('before update signal', add_timestamp=True, tag='Debug')
#             obj.emit_list_update_signal()
#             # print_to_gui('after update signal', add_timestamp=True, tag='Debug')
#         return result
#     return wrapper
#
#
# class PersistentListInteractingWithGUI:
#     list_update_signal = None
#
#     def __init__(self, json_file_path='', boot_fresh=False):
#         self.items = []
#         self.json_file_path = json_file_path
#         self.init_from_settings(boot_fresh=boot_fresh)
#
#     @property
#     def local_file_default_path(self):
#         return f"{ROOT_PATH}/{USER_PATH}/{RE.md['year']}/{RE.md['cycle']}/{RE.md['proposal']}/"
#
#     def init_from_settings(self, boot_fresh=False):
#         if not boot_fresh:
#             try:
#                 self.add_items_from_file(self.json_file_path)
#             except FileNotFoundError:
#                 self.save_to_settings()
#
#     @emit_list_update_signal_decorator
#     def init_from_new_file(self, new_json_file_path):
#         self.items = []
#         self.json_file_path = new_json_file_path
#         self.init_from_settings()
#
#     @emit_list_update_signal_decorator
#     def add_items_from_file(self, file):
#         self.items += self.item_list_from_file(file)
#
#     def item_list_from_file(self, file):
#         with open(file, 'r') as f:
#             item_list = json.loads(f.read())
#         return item_list
#
#     def save_to_settings(self):
#         self.save_to_file(self.json_file_path)
#
#     def save_to_file(self, file):
#         with open(file, 'w') as f:
#             json.dump(self.items, f, indent=4)
#
#     @emit_list_update_signal_decorator
#     def reset(self):
#         self.items = []
#
#     @emit_list_update_signal_decorator
#     def insert_item_at_index(self, index, item):
#         self.items.insert(index, item)
#
#     @emit_list_update_signal_decorator
#     def add_item(self, item):
#         self.items.append(item)
#
#     @emit_list_update_signal_decorator
#     def delete_item_at_index(self, index):
#         self.items.pop(index)
#
#     @emit_list_update_signal_decorator
#     def delete_multiple_items(self, index_list):
#         index_list.sort(reverse=True)
#         for index in index_list:
#             self.delete_item_at_index(index, emit_signal=False)
#
#     @emit_list_update_signal_decorator
#     def update_item_at_index(self, index, item):
#         self.items[index] = index
#
#     def item_at_index(self, index):
#         return self.items[index]
#
#     def append_list_update_signal(self, signal):
#         self.list_update_signal = signal
#
#     def emit_list_update_signal(self):
#         if self.list_update_signal is not None:
#             # print_to_gui('before emitting signal', add_timestamp=True, tag='Debug')
#             self.list_update_signal.emit()
#             # print_to_gui('after emitting signal', add_timestamp=True, tag='Debug')
#         # print_to_gui('before saving to settings', add_timestamp=True, tag='Debug')
#         self.save_to_settings()
#         # print_to_gui('after saving to settings', add_timestamp=True, tag='Debug')


class ScanManager():
    _global_json_path = f'{ROOT_PATH_SHARED}/settings/json/scan_manager.json'

    def __init__(self, redis_key='scan_manager'):
        self.init_global_manager(redis_key)
        _default_local_key = f"scan_manager_local_{RE.md.get('year', 'unknown')}_{RE.md.get('cycle', 'unknown')}_{RE.md.get('proposal', 'unknown')}"
        _default_local_json = f"{ROOT_PATH}/{USER_PATH}/{RE.md.get('year', 'unknown')}/{RE.md.get('cycle', 'unknown')}/{RE.md.get('proposal', 'unknown')}/scan_manager.json"
        self.load_local_manager(_default_local_key, _default_local_json)
        self.trajectory_path = trajectory_manager.trajectory_path
        self.trajectory_creator = TrajectoryCreator()

    def init_global_manager(self, redis_key):
        self.redis_key = redis_key
        self._global_store = RedisJSONDict(redis_settings_client, prefix=redis_key)
        try:
            self.scan_dict = self._global_store['scan_dict']
        except KeyError:
            # Redis is reachable but key is absent — try JSON fallback then start empty
            try:
                with open(self._global_json_path, 'r') as f:
                    self.scan_dict = json.loads(f.read())
                self._global_store['scan_dict'] = self.scan_dict
            except Exception:
                self.scan_dict = {}
                self._global_store['scan_dict'] = self.scan_dict
        except Exception:
            # Redis is unreachable — fall back to JSON file
            try:
                with open(self._global_json_path, 'r') as f:
                    self.scan_dict = json.loads(f.read())
                try:
                    self._global_store['scan_dict'] = self.scan_dict
                except Exception:
                    pass
            except Exception:
                self.scan_dict = {}

    def load_local_manager(self, redis_key_local, json_path_local=None):
        self.redis_key_local = redis_key_local
        self.json_file_path_local = json_path_local
        self._local_store = RedisJSONDict(redis_settings_client, prefix=redis_key_local)
        try:
            self.scan_list_local = self._local_store['scan_list']
        except KeyError:
            # Redis reachable but key absent — try JSON fallback then start empty
            if json_path_local:
                try:
                    with open(json_path_local, 'r') as f:
                        self.scan_list_local = json.loads(f.read())
                    self._local_store['scan_list'] = self.scan_list_local
                except Exception:
                    self.scan_list_local = []
            else:
                self.scan_list_local = []
        except Exception:
            # Redis unreachable — fall back to JSON file
            if json_path_local:
                try:
                    with open(json_path_local, 'r') as f:
                        self.scan_list_local = json.loads(f.read())
                    try:
                        self._local_store['scan_list'] = self.scan_list_local
                    except Exception:
                        pass
                except Exception:
                    self.scan_list_local = []
            else:
                self.scan_list_local = []

    def dump_local_scan_list(self):
        try:
            self._local_store['scan_list'] = self.scan_list_local
        except Exception:
            pass
        if self.json_file_path_local and os.environ.get('SAVE_CONFIG_TO_JSON'):
            try:
                os.makedirs(os.path.dirname(self.json_file_path_local), exist_ok=True)
                with open(self.json_file_path_local, 'w') as f:
                    json.dump(self.scan_list_local, f)
            except Exception:
                pass

    def reset(self):
        pass
        # self.load_local_manager()
        # self.init_local_manager()
        # self.dump_local_scan_list()

    def init_from_settings(self):
        self.load_local_manager()

    def add_scan(self, scan, aux_parameters, name):
        uid = self.check_if_brand_new(scan)
        scan_def = f"{name} ({aux_parameters['scan_description']})"
        scan_name = f"{name}"
        scan_local = {'uid' : uid, 'scan_def' : scan_def, 'scan_name' : scan_name, 'aux_parameters' : aux_parameters, 'archived': False}
        self.scan_list_local.append(scan_local)
        self.dump_local_scan_list()
        return uid

    def delete_local_scan(self, idx):
        self.scan_list_local.pop(idx)
        self.dump_local_scan_list()

    def update_local_scan_offset_to_current(self, indexes):
        for index in indexes:
            self.scan_list_local[index]['aux_parameters']['offset'] = hhm.angle_offset.get()
        self.dump_local_scan_list()

    def update_local_scans_offset_in_vicinity_of_e(self, energy):
        indexes = []
        for index, local_scan in enumerate(self.scan_list_local):
            uid = local_scan['uid']
            global_scan = self.scan_dict[uid]
            if global_scan['scan_type'] == 'constant energy':
                scan_energy = global_scan['scan_parameters']['energy']
            else:
                scan_energy = global_scan['scan_parameters']['e0']

            if (scan_energy >= (energy - 200)) and (scan_energy <= (energy + 500)):
                indexes.append(index)

        # print(indexes)
        self.update_local_scan_offset_to_current(indexes)

    def archive_scan_at_uid(self, uid):
        for indx, scan in enumerate(self.scan_list_local):
            if scan['uid'] == uid:
                break
        self.scan_list_local[indx]['archived'] = True


    def restore_scan_at_uid(self, uid):
        for indx, scan in enumerate(self.scan_list_local):
            if scan['uid'] == uid:
                break
        self.scan_list_local[indx]['archived'] = False





    def check_if_brand_new(self, new_scan):
        for uid, scan in self.scan_dict.items():
            if scan['scan_type'] == scan['scan_type']:
                scan_parameters = scan['scan_parameters']
                new_scan_parameters = new_scan['scan_parameters']
                keys =  scan_parameters.keys()
                new_keys = new_scan_parameters.keys()
                # if all([(new_k in keys) for new_k in new_keys]):
                if Counter(new_keys) == Counter(keys):
                    parameters_match = True
                    for k in keys:
                        if k != 'filename':
                            if scan_parameters[k] != new_scan_parameters[k]:
                                parameters_match = False
                                break
                    if parameters_match:
                        #print_to_gui('SCAN IS OLD')
                        return uid

        new_uid = self.make_scan_uid()
        #print_to_gui('SCAN IS NEW')
        self.scan_dict[new_uid] = new_scan
        self.create_trajectory_file(new_scan, new_uid)
        try:
            self._global_store['scan_dict'] = self.scan_dict
        except Exception:
            pass
        if os.environ.get('SAVE_CONFIG_TO_JSON'):
            try:
                with open(self._global_json_path, 'w') as f:
                    json.dump(self.scan_dict, f)
            except Exception:
                pass
        return new_uid

    def trajectory_filename_from_uid(self, scan_uid):
        return self.scan_dict[scan_uid]['scan_parameters']['filename']

    def make_scan_uid(self):
        return str(uuid.uuid4())[:13]

    def create_scan_preview(self, scan, aux_parameters, plot_func):
        self.determine_and_describe_type_of_scan(scan, aux_parameters)
        self.create_lightweight_trajectory(scan, plot_func)
        self.add_spectrometer_grids_if_needed(aux_parameters)

    def determine_and_describe_type_of_scan(self, scan, aux_parameters):
        mono_is_moving = (scan['scan_type'] != 'constant energy')
        spectrometer_is_used = ('spectrometer' in aux_parameters.keys())

        scan_parameters = scan['scan_parameters']

        if mono_is_moving:
            mono_description = f"mono {scan['scan_type']} at {scan_parameters['element']}-{scan_parameters['edge']} edge"
            if scan['scan_type'] == 'step scan':
                if 'grid_kind' in scan_parameters.keys():
                    if scan_parameters['grid_kind'] == 'linear':
                        mono_description = f"mono step scan between {scan_parameters['energy_min']} and {scan_parameters['energy_max']}"
        else:
            mono_description = f"mono at Ein={scan_parameters['energy']}"

        if spectrometer_is_used:
            spectrometer_is_moving = (aux_parameters['spectrometer']['scan_type'] != 'constant energy')
            spectrometer_is_vonhamos = (aux_parameters['spectrometer']['kind'] == 'von_hamos')
        else: # redundant but why not
            spectrometer_is_moving = False
            spectrometer_is_vonhamos = False

        if mono_is_moving:
            if (not spectrometer_is_used):
                scan_key = 'xas'
                scan_description = f"{mono_description}"
            else:
                if spectrometer_is_moving: # only johann can move together with mono
                    scan_key = 'johann_rixs'
                    element = aux_parameters['spectrometer']['scan_parameters']['element']
                    line = aux_parameters['spectrometer']['scan_parameters']['line']
                    scan_description = f"RIXS with {mono_description} and Johann at {element}-{line} line"

                else:
                    if spectrometer_is_vonhamos:
                        scan_key = 'von_hamos_rixs'
                        scan_description = f"RIXS with {mono_description} with Von Hamos"
                    else:
                        scan_key = 'johann_herfd'
                        spectrometer_energy = aux_parameters['spectrometer']['scan_parameters']['energy']
                        scan_description = f"{mono_description} with Johann at Eout={spectrometer_energy}"
        else:
            if (not spectrometer_is_used):
                scan_key = 'constant_e'
                scan_description = f"{mono_description} x{scan_parameters['n_exposures']} for {scan_parameters['dwell_time']} s"
            else:
                if spectrometer_is_moving:
                    scan_key = 'johann_xes'
                    element = aux_parameters['spectrometer']['scan_parameters']['element']
                    line = aux_parameters['spectrometer']['scan_parameters']['line']
                    scan_description = f"{mono_description} with Johann {aux_parameters['spectrometer']['scan_type']} at {element}-{line} line"
                else:
                    if spectrometer_is_vonhamos:
                        scan_key = 'von_hamos_xes'
                        scan_description = f"{mono_description} with Von Hamos x{scan_parameters['n_exposures']} for {scan_parameters['dwell_time']} s"
                    else:
                        scan_key = 'constant_e_johann'
                        spectrometer_energy = aux_parameters['spectrometer']['scan_parameters']['energy']
                        scan_description = f"{mono_description} with Johann at Eout={spectrometer_energy} x{scan_parameters['n_exposures']} for {scan_parameters['dwell_time']} s"

        aux_parameters['scan_key'] = scan_key
        aux_parameters['scan_description'] = scan_description

    def create_lightweight_trajectory(self, scan, plot_func):
        if scan['scan_type'] == 'fly scan':
            self.trajectory_creator.define_complete(scan['scan_parameters'], lightweight=True)
            energy = self.trajectory_creator.energy
            time = self.trajectory_creator.time
        elif scan['scan_type'] == 'step scan':
            energy, _, time = generate_energy_grid_from_dict(scan['scan_parameters'])
        elif scan['scan_type'] == 'constant energy':
            energy_value = scan['scan_parameters']['energy']
            energy = np.ones(101) * energy_value
            time = np.linspace(0, 1, energy.size)
        plot_func(time, energy)

    def add_spectrometer_grids_if_needed(self, aux_parameters):
        scan_key = aux_parameters['scan_key']
        if scan_key == 'johann_xes':
            if aux_parameters['spectrometer']['scan_type'] == 'step scan':
                energy_grid, time_grid = generate_emission_energy_grid_from_dict(aux_parameters['spectrometer']['scan_parameters'])
                aux_parameters['spectrometer']['scan_parameters']['time_grid'] = time_grid.tolist()
                aux_parameters['spectrometer']['scan_parameters']['energy_grid'] = energy_grid.tolist()
            elif aux_parameters['spectrometer']['scan_type'] == 'fly scan':
                aux_parameters['spectrometer']['scan_parameters']['relative_trajectory'] = \
                    generate_emission_relative_trajectory_from_dict(aux_parameters['spectrometer']['scan_parameters'])

        elif scan_key == 'johann_rixs':
            if aux_parameters['spectrometer']['scan_type'] == 'step scan':
                energy_grid, _ = generate_emission_energy_grid_from_dict(aux_parameters['spectrometer']['scan_parameters'])
                aux_parameters['spectrometer']['scan_parameters']['energy_grid'] = energy_grid.tolist()

            elif aux_parameters['spectrometer']['scan_type'] == 'fly scan':
                aux_parameters['spectrometer']['scan_parameters']['relative_trajectory'] = \
                    generate_emission_relative_trajectory_from_dict(aux_parameters['spectrometer']['scan_parameters'])

    def create_trajectory_file(self, scan, uid):
        filename = f'{uid}.txt'
        filepath = self.trajectory_path + filename
        if scan['scan_type'] == 'fly scan':
            self.trajectory_creator.define_complete(scan['scan_parameters'])
            self.trajectory_creator.save(filepath)
        elif scan['scan_type'] == 'step scan':
            energy, dwell_time, _ = generate_energy_grid_from_dict(scan['scan_parameters'])
            data = np.vstack((energy, dwell_time)).T
            header = self._make_mono_step_scan_header(scan['scan_parameters'])
            np.savetxt(filepath, data, header=header)
        elif scan['scan_type'] == 'constant energy':
            filename = ''

        self.scan_dict[uid]['scan_parameters']['filename'] = filename


    def standard_scan_dict(self, element, edge, short=False):
        if short:
            preedge_start = -150.0
            XANES_start = -30.0
            XANES_end = 30.0
            EXAFS_end = 8.0
            type = 'standard'
            preedge_duration = 2.0
            edge_duration = 3.0
            postedge_duration = 5.0
        else:
            preedge_start = -200.0
            XANES_start = -30.0
            XANES_end = 50.0
            EXAFS_end = 16.0
            type = 'standard'
            preedge_duration = 4.0
            edge_duration = 6.0
            postedge_duration = 20.0


        return {'scan_type': 'fly scan',
                'scan_parameters': {'element': element,
                                    'edge': edge,
                                    'e0': xraydb.xray_edge(element, edge).energy,
                                    'preedge_start': preedge_start,
                                    'XANES_start': XANES_start,
                                    'XANES_end': XANES_end,
                                    'EXAFS_end': EXAFS_end,
                                    'type': type,
                                    'preedge_duration': preedge_duration,
                                    'edge_duration': edge_duration,
                                    'postedge_duration': postedge_duration,
                                    'preedge_flex': 0.5,
                                    'postedge_flex': 0.3,
                                    'pad': 0.5,
                                    'repeat': 1,
                                    'single_direction': True,
                                    'revert': True,
                                    'filename': ''}}

    def quick_herfd_scan_dict(self, element, edge, duration, scan_range):
        range_base = 200
        range_scaler = scan_range / range_base
        duration_base = 10
        duration_scaler = duration / duration_base
        preedge_start = -50.0 * range_scaler
        XANES_start = -10.0 * range_scaler
        XANES_end = 20.0 * range_scaler
        EXAFS_end = e2k(150 * range_scaler, 0)
        type = 'standard'
        preedge_duration = 1.5 * duration_scaler
        edge_duration = 5.0 * duration_scaler
        postedge_duration = 3.0 * duration_scaler

        return {'scan_type': 'fly scan',
                'scan_parameters': {'element': element,
                                    'edge': edge,
                                    'e0': xraydb.xray_edge(element, edge).energy,
                                    'preedge_start': preedge_start,
                                    'XANES_start': XANES_start,
                                    'XANES_end': XANES_end,
                                    'EXAFS_end': EXAFS_end,
                                    'type': type,
                                    'preedge_duration': preedge_duration,
                                    'edge_duration': edge_duration,
                                    'postedge_duration': postedge_duration,
                                    'preedge_flex': 0.5,
                                    'postedge_flex': 0.3,
                                    'pad': 0.25,
                                    'repeat': 1,
                                    'single_direction': True,
                                    'revert': True,
                                    'filename': ''}}

    def standard_trajectory_filename(self, element, edge, short=False):
        standard_scan_dict = self.standard_scan_dict(element, edge, short=short)
        uid = self.check_if_brand_new(standard_scan_dict)
        return self.scan_dict[uid]['scan_parameters']['filename']

    def quick_herfd_trajectory_filename(self, element, edge, duration, scan_range):
        quick_herfd_scan_dict = self.quick_herfd_scan_dict(element, edge, duration, scan_range)
        uid = self.check_if_brand_new(quick_herfd_scan_dict)
        return self.scan_dict[uid]['scan_parameters']['filename']


    def quick_linear_scan_dict(self, e_cen, e_width, e_velocity):
        preedge_start = -e_width * 0.5
        XANES_start = -e_width * 0.25
        XANES_end = e_width * 0.25
        EXAFS_end = e2k(e_width * 0.5, 0)

        type = 'sine'
        duration = np.pi * e_width / (2 * e_velocity)

        return {'scan_type': 'fly scan',
                'scan_parameters': {'element': 'NA',
                                    'edge': 'NA',
                                    'e0': e_cen,
                                    'preedge_start': preedge_start,
                                    'XANES_start': XANES_start,
                                    'XANES_end': XANES_end,
                                    'EXAFS_end': EXAFS_end,
                                    'type': type,
                                    'duration': duration,
                                    'pad': 0.5,
                                    'repeat': 1,
                                    'single_direction': True,
                                    'revert': True,
                                    'filename': ''}}


        # type = 'standard'
        # _duration = e_width / e_velocity
        # preedge_duration = _duration / 2
        # edge_duration = _duration
        # postedge_duration = _duration / 2
        #
        # return {'scan_type': 'fly scan',
        #         'scan_parameters': {'element': 'NA',
        #                             'edge': 'NA',
        #                             'e0': e_cen,
        #                             'preedge_start': preedge_start,
        #                             'XANES_start': XANES_start,
        #                             'XANES_end': XANES_end,
        #                             'EXAFS_end': EXAFS_end,
        #                             'type': type,
        #                             'preedge_duration': preedge_duration,
        #                             'edge_duration': edge_duration,
        #                             'postedge_duration': postedge_duration,
        #                             'preedge_flex': 0.5,
        #                             'postedge_flex': 0.5,
        #                             'pad': 0.5,
        #                             'repeat': 1,
        #                             'single_direction': True,
        #                             'revert': True,
        #                             'filename': ''}}

    def quick_linear_trajectory_filename(self, e_cen, e_width, e_velocity):
        quick_linear_scan_dict = self.quick_linear_scan_dict(e_cen, e_width, e_velocity)
        uid = self.check_if_brand_new(quick_linear_scan_dict)
        return self.scan_dict[uid]['scan_parameters']['filename']

    def _make_mono_step_scan_header(self, scan_parameters):
        if 'grid_kind' in scan_parameters.keys():
            grid_kind = scan_parameters['grid_kind']
        else:
            grid_kind = 'xas'

        if scan_parameters['revert']:
            direction = 'backward'
        else:
            direction = 'forward'

        if grid_kind == 'xas':
            element = scan_parameters['element']
            edge = scan_parameters['edge']
            e0 = scan_parameters['e0']
            header = f'element: {element}, edge: {edge}, E0: {e0}'
        elif grid_kind == 'linear':
            energy_min = scan_parameters['energy_min'],
            energy_max = scan_parameters['energy_max'],
            energy_step = scan_parameters['energy_step'],
            dwell_time = scan_parameters['dwell_time']
            header = f'energy_min: {energy_min}, energy_max: {energy_max}, energy_step: {energy_step}'
        else:
            header = ''
        return f'{header}, direction: {direction}'

    def parse_scan_to_plan(self, name, comment, scan_idx, sample_coordinates=None, metadata=None, ):
        if metadata is None:
            metadata = {}
        scan_local = self.scan_list_local[scan_idx]
        scan_uid = scan_local['uid']
        metadata['monochromator_scan_uid'] = scan_uid
        scan = self.scan_dict[scan_uid]
        return self.parse_scan_to_plan_from_parameters(name,
                                                       comment,
                                                       scan,
                                                       scan_local['aux_parameters'],
                                                       sample_coordinates=sample_coordinates,
                                                       metadata=metadata)


    def parse_scan_to_plan_from_parameters(self, name, comment, scan, aux_parameters, sample_coordinates=None, metadata=None,
                                           rixs_kwargs=None):
        if rixs_kwargs is None:
            rixs_kwargs = {}
        if metadata is None:
            metadata = {}

        scan_type = scan['scan_type']
        scan_parameters = scan['scan_parameters']

        scan_key = aux_parameters['scan_key']
        common_kwargs = {'name': name,
                         'comment': comment,
                         'detectors': aux_parameters['detectors'],
                         'mono_angle_offset': aux_parameters['offset'],
                         'metadata': metadata}

        output = []

        if type(sample_coordinates) == dict:
            plan_name = 'move_sample_stage_plan'
            plan_kwargs = {'sample_coordinates': sample_coordinates}
            output.append({'plan_name': plan_name,
                           'plan_kwargs': plan_kwargs})

        if scan_key == 'xas':
            if scan_type == 'step scan':
                plan_name = 'step_scan_plan'
            elif scan_type == 'fly scan':
                plan_name = 'fly_scan_plan'
            plan_kwargs = {'trajectory_filename': scan_parameters['filename'],
                           'element': scan_parameters['element'],
                           'edge': scan_parameters['edge'],
                           'e0': scan_parameters['e0']}

        elif scan_key == 'constant_e':
            plan_name = 'collect_n_exposures_plan'
            plan_kwargs = {'n_exposures': scan_parameters['n_exposures'],
                           'dwell_time': scan_parameters['dwell_time'],
                           'mono_energy': scan_parameters['energy']}

        elif scan_key == 'von_hamos_xes':
            plan_name = 'collect_von_hamos_xes_plan'
            plan_kwargs = {'n_exposures': scan_parameters['n_exposures'],
                           'dwell_time': scan_parameters['dwell_time'],
                           'mono_energy': scan_parameters['energy']}

        elif scan_key == 'von_hamos_rixs':
            if scan_type == 'step scan':
                plan_name = 'step_scan_von_hamos_plan'
            elif scan_type == 'fly scan':
                plan_name = 'fly_scan_von_hamos_plan'
            plan_kwargs = {'trajectory_filename': scan_parameters['filename'],
                           'element': scan_parameters['element'],
                           'edge': scan_parameters['edge'],
                           'e0': scan_parameters['e0'],
                           'scan_for_calibration_purpose': aux_parameters['scan_for_calibration_purpose']}

        elif scan_key == 'constant_e_johann':
            plan_name = 'collect_n_exposures_johann_plan'
            spectrometer_energy = aux_parameters['spectrometer']['scan_parameters']['energy']
            spectrometer_config_uid = aux_parameters['spectrometer']['spectrometer_config_uid']
            plan_kwargs = {'n_exposures': scan_parameters['n_exposures'],
                           'dwell_time': scan_parameters['dwell_time'],
                           'mono_energy': scan_parameters['energy'],
                           'spectrometer_energy': spectrometer_energy,
                           'spectrometer_config_uid': spectrometer_config_uid}

        elif scan_key == 'johann_xes':
            if aux_parameters['spectrometer']['scan_type'] == 'step scan':
                plan_name = 'step_scan_johann_xes_plan'
                _scan_kwargs = {'emission_energy_list': aux_parameters['spectrometer']['scan_parameters']['energy_grid'],
                               'emission_time_list': aux_parameters['spectrometer']['scan_parameters']['time_grid']}
            elif aux_parameters['spectrometer']['scan_type'] == 'fly scan':
                plan_name = 'epics_fly_scan_johann_xes_plan'
                _scan_kwargs = {'spectrometer_central_energy': aux_parameters['spectrometer']['scan_parameters']['e0'],
                                'relative_trajectory': aux_parameters['spectrometer']['scan_parameters']['relative_trajectory'],
                                'trajectory_as_energy': True}
            element = aux_parameters['spectrometer']['scan_parameters']['element']
            line = aux_parameters['spectrometer']['scan_parameters']['line']
            e0 = aux_parameters['spectrometer']['scan_parameters']['e0']
            spectrometer_config_uid = aux_parameters['spectrometer']['spectrometer_config_uid']
            plan_kwargs = {'mono_energy': scan_parameters['energy'],
                           **_scan_kwargs,
                           'element': element,
                           'line': line,
                           'e0': e0,
                           'spectrometer_config_uid': spectrometer_config_uid}

        elif scan_key == 'johann_herfd':
            if scan_type == 'step scan':
                plan_name = 'step_scan_johann_herfd_plan'
            elif scan_type == 'fly scan':
                plan_name = 'fly_scan_johann_herfd_plan'
            spectrometer_energy = aux_parameters['spectrometer']['scan_parameters']['energy']
            spectrometer_config_uid = aux_parameters['spectrometer']['spectrometer_config_uid']
            plan_kwargs = {'trajectory_filename': scan_parameters['filename'],
                           'element': scan_parameters['element'],
                           'edge': scan_parameters['edge'],
                           'e0': scan_parameters['e0'],
                           'spectrometer_energy': spectrometer_energy,
                           'spectrometer_config_uid': spectrometer_config_uid}
            # if rixs_file_name is not None:
            #     plan_kwargs['rixs_file_name'] = rixs_file_name

        elif scan_key == 'johann_rixs':
            if (scan_type == 'step scan') and (aux_parameters['spectrometer']['scan_type'] == 'step scan'):
                plan_name = 'step_scan_johann_rixs_plan_bundle'
                _scan_kwargs = {'emission_energy_list': aux_parameters['spectrometer']['scan_parameters']['energy_grid']}
            elif (scan_type == 'step scan') and (aux_parameters['spectrometer']['scan_type'] == 'fly scan'):
                plan_name = 'fly_spectrometer_scan_johann_rixs_plan_bundle'
                _scan_kwargs = {'spectrometer_central_energy': aux_parameters['spectrometer']['scan_parameters']['e0'],
                                'relative_trajectory': aux_parameters['spectrometer']['scan_parameters']['relative_trajectory'],
                                'trajectory_as_energy': True,
                                'scan_for_calibration_purpose': aux_parameters['scan_for_calibration_purpose']}
            elif (scan_type == 'fly scan') and (aux_parameters['spectrometer']['scan_type'] == 'step scan'):
                plan_name = 'fly_scan_johann_rixs_plan_bundle'
                _scan_kwargs = {'emission_energy_list': aux_parameters['spectrometer']['scan_parameters']['energy_grid']}
            elif (scan_type == 'fly scan') and (aux_parameters['spectrometer']['scan_type'] == 'fly scan'):
                plan_name = ''
                _scan_kwargs = {}
                raise NotImplementedError('Such scan is not implemented yet!')

            element_line = aux_parameters['spectrometer']['scan_parameters']['element']
            line = aux_parameters['spectrometer']['scan_parameters']['line']
            e0_line = aux_parameters['spectrometer']['scan_parameters']['e0']

            spectrometer_config_uid = aux_parameters['spectrometer']['spectrometer_config_uid']
            plan_kwargs = {'trajectory_filename': scan_parameters['filename'],
                           'element': scan_parameters['element'],
                           'edge': scan_parameters['edge'],
                           'e0': scan_parameters['e0'],
                           'element_line' : element_line,
                           'line': line,
                           'e0_line': e0_line,
                           **_scan_kwargs,
                           'sample_coordinates' : sample_coordinates,
                           'rixs_kwargs' : rixs_kwargs,
                           'spectrometer_config_uid': spectrometer_config_uid}

            #
            # _local_rixs_file_name = create_interp_file_name(name, '.rixs')
            #
            # for emission_energy, _local_sample_coordinates in zip(spectrometer_energy_grid, sample_coordinates):
            #
            #     _local_name = f'{name} E={str(emission_energy)}'
            #     _local_comment = f'{comment} E={str(emission_energy)}'
            #     _local_spectrometer_parameters = {'kind' : 'johann',
            #                                       'scan_type': 'constant energy',
            #                                       'scan_parameters': {'energy': emission_energy}}
            #     _local_aux_parameters = copy.deepcopy(aux_parameters)
            #     _local_aux_parameters['spectrometer'] = _local_spectrometer_parameters
            #     _local_aux_parameters['scan_key'] = 'johann_herfd'
            #     _local_output = self.parse_scan_to_plan_from_parameters(_local_name,
            #                                                             _local_comment,
            #                                                             scan,
            #                                                             _local_aux_parameters,
            #                                                             sample_coordinates=_local_sample_coordinates,
            #                                                             metadata=metadata,
            #                                                             )
            #     output.extend(_local_output)

        common_kwargs['metadata'] = {**common_kwargs['metadata'],
                                     **{'plan_name': plan_name,
                                        'scan_kind': scan_key}}

        output.append({'plan_name': plan_name,
                       'plan_kwargs': {**plan_kwargs, **common_kwargs}})

        return output

    def generate_plan_list(self, name, comment, repeat, delay, scan_idx, sample_coordinates=None, metadata=None):
        if metadata is None:
            metadata = dict()
        plans = []

        # add scan group uid
        if 'scan_group_uid' not in metadata:
            metadata['scan_group_uid'] = get_scan_group_uid()

        scan_local = self.scan_list_local[scan_idx]
        scan_name = scan_local['scan_name']
        metadata['scan_name'] = scan_name
        metadata['radiation_damage_scan'] = scan_local['aux_parameters']['radiation_damage_scan']
        for indx in range(int(repeat)):
            if type(name) == list:
                name_n = [f'{n} {scan_name} {indx+1:04d}' for n in name]
            else:
                name_n = f'{name} {scan_name} {indx+1:04d}'
            plan_list_for_scan = self.parse_scan_to_plan(name_n, comment, scan_idx, sample_coordinates=sample_coordinates, metadata=metadata)
            plans.extend(plan_list_for_scan)
            if delay > 0:
                plans.append({'plan_name': 'sleep', 'plan_kwargs': {'delay': delay}})
        return plans

scan_manager = ScanManager()

def get_scan_group_uid():
    return str(uuid.uuid4())

