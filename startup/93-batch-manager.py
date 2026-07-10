print(ttime.ctime() + ' >>>> ' + __file__)
# from xas.trajectory import trajectory, trajectory_manager
import json
import os
import pandas as pd
from bluesky.plan_stubs import mv
from collections import defaultdict
from redis_json_dict import RedisJSONDict


def _redis_key_from_path(path):
    """Derive a stable Redis key from a JSON file path."""
    return os.path.splitext(os.path.basename(path))[0]

class Sample:
    position_data = pd.DataFrame(columns=['x', 'y', 'z', 'th', 'exposure_time', 'exposed', 'uids'])

    def __init__(self, name, comment='', coordinates=[], max_exposure=None, archived=False):
        self.name = name
        self.comment = comment
        self.max_exposure = max_exposure
        self.add_new_positions_from_coordinates(coordinates)
        self.uid = str(uuid.uuid4())[:20]
        self.archived = archived
        print("SAMPLE INIT")
        print("PS", self.position_data.shape)
        print("PI", self.position_data.index)
        print("PC", self.position_data.columns)
        print(self.position_data)
        # self.add_positions(positions)

    def validate_coordinates(self, coordinates_dict):
        ok_flag = (type(coordinates_dict) == dict) or (len(coordinates_dict) != 4)
        assert ok_flag, 'Unacceptable position format. Coordinates must be a dict with 4 elements corresponding to x, y, z, th coordinates.'

    def add_new_positions_from_coordinates(self, coordinates):
        for coordinates_dict in coordinates:
            self.add_new_position(coordinates_dict)

    def add_new_position(self, coordinates_dict):
        self.validate_coordinates(coordinates_dict)
        point = {**coordinates_dict, **{'exposure_time': 0, 'exposed': False, 'sample_point_uid': self.make_point_uid()}}
        self.add_one_position(point)

    def add_one_position(self, point):
        index_list = [int(i) for i in self.position_data.index]
        if len(index_list) == 0:
            last_index = -1
        else:
            last_index = max(index_list)
        self.position_data = pd.concat([self.position_data, pd.DataFrame(point, index=[last_index+1])])
        print("add_one_pos", self.position_data)

    def make_point_uid(self):
        return str(uuid.uuid4())[:13]

    def add_positions(self, positions):
        for position in positions:
            self.add_one_position(position)

    def remove_positions(self, index_list):
        self.position_data.drop(self.position_data.index[index_list], inplace=True)

    def to_dict(self):
        sample_dict = {}
        sample_dict['name'] = self.name
        sample_dict['comment'] = self.comment
        sample_dict['archived'] = self.archived
        sample_dict['position_data'] = self.position_data.to_dict()
        print("to_dict", sample_dict)
        print("PS", self.position_data.shape)
        print("PI", self.position_data.index)
        print("PC", self.position_data.columns)
        return sample_dict

    def update_position_coordinates(self, index, new_coordinates):
        for axis, coordinate in new_coordinates.items():
            self.position_data[axis][index] = coordinate

    @property
    def number_of_points(self):
        return len(self.position_data.index)

    @property
    def number_of_unexposed_points(self):
        # return 0
        return int(self.number_of_points - sum(self.position_data['exposed']))

    # def index_coordinate_dict(self, index):
    #     return self.position_data.iloc[index][['x', 'y', 'z', 'th']].to_dict()

    def index_coordinate_dict(self, index):
        return {k: v for k, v in self.position_data.iloc[index].to_dict().items() if k in ['x', 'y', 'z', 'th']}

    # def index_coordinate_str(self, index):
    #     coord_dict = self.index_coordinate_dict(index)
    #     return ' '.join([(f"{key}={value : 0.2f}") for key, value in coord_dict.items()])

    def index_coordinate_str(self, index):
        coord_dict = self.index_coordinate_dict(index)
        return ' '.join([(f"{key}={value : 0.2f}") for key, value in coord_dict.items()])

    def index_position_index(self, index):
        return int(self.position_data.index[index])

    def sort_positions_by(self, keys):
        self.position_data = self.position_data.sort_values(keys)
        print("sort_positions_by", self.position_data)

    # def index_exposed(self, index):
    #     return bool(self.position_data.iloc[index][['exposed']].item())

    def index_exposed(self, index):
        return self.position_data.iloc[index]['exposed']

    def set_exposed(self, index, exposed=True):
        # self.position_data['exposed'][index] = exposed
        self.position_data['exposed'].iloc[index] = exposed
        print("set_exposed", self.position_data)

    # def index_uid(self, index):
    #     return str(self.position_data.iloc[index][['sample_point_uid']].item())

    def index_uid(self, index):
        return self.position_data.iloc[index]['sample_point_uid']

    def index_point_data(self, index):
        return self.position_data.iloc[index]

    def index_point_info_for_qt_item(self, index):
        point_data = self.index_point_data(index)
        point_idx = int(point_data.name)
        coord_dict = {k: point_data[k] for k in ['x', 'y', 'z', 'th']}
        point_str = ' '.join([(f"{key}={value : 0.2f}") for key, value in coord_dict.items()])
        point_exposed = point_data['exposed']
        point_str = f'{point_idx + 1:3d} - {point_str}'
        return point_str, point_exposed

    @property
    def uids(self):
        return self.position_data['sample_point_uid'].tolist()

    @classmethod
    def from_dict(cls, sample_dict):
        name = sample_dict['name']
        comment = sample_dict['comment']
        archived =  sample_dict['archived']
        result = cls(name, comment=comment, archived=archived)
        result.position_data = pd.DataFrame.from_dict(sample_dict['position_data'], orient='index').T
        print("from_dict", result.position_data, sample_dict)
        print("PS", result.position_data.shape)
        print("PI", result.position_data.index)
        print("PC", result.position_data.columns)
        return result

FOIL_UID = 'foil'


def emit_list_update_signal_decorator(method):
    def wrapper(obj, *args, emit_signal=True, **kwargs):
        result = method(obj, *args, **kwargs)
        if emit_signal:
            # print_to_gui('before update signal', add_timestamp=True, tag='Debug')
            obj.emit_list_update_signal()
            # print_to_gui('after update signal', add_timestamp=True, tag='Debug')
        return result
    return wrapper


class PersistentListInteractingWithGUI:
    list_update_signal = None

    def __init__(self, json_file_path='', boot_fresh=False):
        self.items = []
        self.json_file_path = json_file_path
        self._redis_key = _redis_key_from_path(json_file_path) if json_file_path else ''
        self._redis_store = RedisJSONDict(redis_settings_client, prefix=self._redis_key) if self._redis_key else None
        self.init_from_settings(boot_fresh=boot_fresh)

    @property
    def local_file_default_path(self):
        return f"{ROOT_PATH}/{USER_PATH}/{RE.md['year']}/{RE.md['cycle']}/{RE.md['proposal']}/"

    def init_from_settings(self, boot_fresh=False):
        if not boot_fresh:
            try:
                self.add_items_from_file(self.json_file_path)
            except (KeyError, Exception):
                self.save_to_settings()

    @emit_list_update_signal_decorator
    def init_from_new_file(self, new_json_file_path):
        self.items = []
        self.json_file_path = new_json_file_path
        self._redis_key = _redis_key_from_path(new_json_file_path) if new_json_file_path else ''
        self._redis_store = RedisJSONDict(redis_settings_client, prefix=self._redis_key) if self._redis_key else None
        self.init_from_settings()

    @emit_list_update_signal_decorator
    def add_items_from_file(self, file):
        self.items += self.item_list_from_file(file)

    def item_list_from_file(self, file):
        _key = _redis_key_from_path(file)
        _store = RedisJSONDict(redis_settings_client, prefix=_key)
        print(f'Key: {_key}')
        print(f'Store: {_store}')
        try:
            item_list = _store['items']
            print(f"{item_list=}")
            if len(item_list) == 0:
                print("zero length item list")
                raise KeyError('items')
        except KeyError:
            # Redis reachable but key absent or empty — try JSON fallback
            try:
                with open(file, 'r') as f:
                    item_list = json.loads(f.read())
                _store['items'] = item_list
            except Exception:
                item_list = []
        except Exception:
            # Redis unreachable — fall back to JSON file
            try:
                with open(file, 'r') as f:
                    item_list = json.loads(f.read())
                try:
                    _store['items'] = item_list
                except Exception:
                    pass
            except Exception:
                item_list = []
        return item_list

    def save_to_settings(self):
        self.save_to_file(self.json_file_path)

    def save_to_file(self, file):
        _key = _redis_key_from_path(file)
        _store = RedisJSONDict(redis_settings_client, prefix=_key)
        try:
            _store['items'] = self.items
        except Exception:
            pass
        if os.environ.get('SAVE_CONFIG_TO_JSON'):
            try:
                with open(file, 'w') as f:
                    json.dump(self.items, f, indent=4)
            except Exception:
                pass

    @emit_list_update_signal_decorator
    def reset(self):
        self.items = []

    @emit_list_update_signal_decorator
    def insert_item_at_index(self, index, item):
        self.items.insert(index, item)

    @emit_list_update_signal_decorator
    def add_item(self, item):
        self.items.append(item)

    @emit_list_update_signal_decorator
    def delete_item_at_index(self, index):
        self.items.pop(index)

    @emit_list_update_signal_decorator
    def delete_multiple_items(self, index_list):
        index_list.sort(reverse=True)
        for index in index_list:
            self.delete_item_at_index(index, emit_signal=False)

    @emit_list_update_signal_decorator
    def update_item_at_index(self, index, item):
        self.items[index] = index

    def item_at_index(self, index):
        return self.items[index]

    def append_list_update_signal(self, signal):
        self.list_update_signal = signal

    def emit_list_update_signal(self):
        if self.list_update_signal is not None:
            # print_to_gui('before emitting signal', add_timestamp=True, tag='Debug')
            self.list_update_signal.emit()
            # print_to_gui('after emitting signal', add_timestamp=True, tag='Debug')
        # print_to_gui('before saving to settings', add_timestamp=True, tag='Debug')
        self.save_to_settings()
        # print_to_gui('after saving to settings', add_timestamp=True, tag='Debug')


class SampleManager(PersistentListInteractingWithGUI):

    def __init__(self, json_file_path = f'{ROOT_PATH_SHARED}/settings/json/sample_manager.json'):
        super().__init__(json_file_path)
        # self.samples = []
        # self.json_file_path = json_file_path
        # self.init_from_settings()

    #Class specific decorators
    @property
    def samples(self):
        non_archived_items = [item for item in self.items if not item.archived]
        return non_archived_items

    @property
    def archived_samples(self):
        archived_items = [item for item in self.items if item.archived]
        return archived_items

    @property
    def all_samples(self):
        return self.items

    @samples.setter
    def samples(self, value):
        self.items = value

    @samples.deleter
    def samples(self):
        del self.items

    # Dealing with file io
    @emit_list_update_signal_decorator
    def add_items_from_file(self, file):
        items = self.item_list_from_file(file)
        item_list = self.parse_sample_dict_list(items)
        self.items += item_list

    def init_from_settings(self, boot_fresh=False):
        super().init_from_settings(boot_fresh=boot_fresh)
        if (len(self.items) == 0) or (self.items[0].name.lower() != 'foil'):
            foil_sample = Sample('foil')
            foil_sample.uid = FOIL_UID
            self.items = [foil_sample] + self.items
            self.save_to_settings()
        else:
            if self.items[0].uid != FOIL_UID:
                self.items[0].uid = FOIL_UID
                self.save_to_settings()

    def parse_sample_dict_list(self, sample_dict_list):
        sample_list = []
        for sample_dict in sample_dict_list:
            sample_list.append(Sample.from_dict(sample_dict))

        return sample_list

    def save_to_file(self, file):
        _key = _redis_key_from_path(file)
        _store = RedisJSONDict(redis_settings_client, prefix=_key)
        try:
            _store['items'] = self.samples_as_dict_list
        except Exception:
            pass
        if os.environ.get('SAVE_CONFIG_TO_JSON'):
            try:
                with open(file, 'w') as f:
                    json.dump(self.samples_as_dict_list, f)
            except Exception:
                pass

    @property
    def samples_as_dict_list(self):
        sample_dict_list = []
        for sample in self.all_samples:
            sample_dict_list.append(sample.to_dict())
        return sample_dict_list

    # dealing with addition and deletion of samples
    def add_new_sample(self, name, comment='', coordinates=[], max_exposure=None):
        sample = Sample(name, comment=comment, coordinates=coordinates, max_exposure=max_exposure, archived=False)
        self.add_sample(sample)

    def insert_new_sample_at_index(self, index, name, comment='', coordinates=[], max_exopsure=None):
        sample = Sample(name, comment=comment, coordinates=coordinates, max_exposure=max_exopsure)
        self.insert_item_at_index(index, sample)

    def add_sample(self, sample):
        self.add_item(sample)

    # def delete_sample_at_index(self, index):
    #     self.delete_item_at_index(index)
    #
    # def delete_multiple_samples(self, index_list, emit_signal=True):
    #     self.delete_multiple_items(index_list, emit_signal=emit_signal)

    @emit_list_update_signal_decorator
    def delete_samples_with_index_dict(self, index_dict):
        # sample_idx_to_delete = []
        for sample_index, point_index_list in index_dict.items():
            if len(point_index_list) > 0:
                sample = self.samples[sample_index]
                point_index_set = set(point_index_list)
                sample.remove_positions(list(point_index_set))
            # else:
                # sample_idx_to_delete.append(sample_index)
        # if len(sample_idx_to_delete) > 0:
        #     self.delete_multiple_samples(sample_idx_to_delete, emit_signal=False)

    @emit_list_update_signal_decorator
    def update_sample_at_index(self, index, new_name, new_comment):
        self.samples[index].name = new_name
        self.samples[index].comment = new_comment

    @emit_list_update_signal_decorator
    def update_sample_coordinates_at_index(self, sample_index, sample_point_index, new_coordinate_dict):
        self.samples[sample_index].update_position_coordinates(sample_point_index, new_coordinate_dict)

    # misc
    @property
    def number_of_samples(self):
        return len(self.samples)

    def sample_at_index(self, index):
        # return self.item_at_index(index)
        return self.samples[index]

    def sample_name_at_index(self, index):
        # return self.sample_at_index(index).name
        return self.samples[index].name

    def sample_uid_at_index(self, index):
        # return self.sample_at_index(index).uid
        return self.samples[index].uid

    def sample_comment_at_index(self, index):
        # return self.sample_at_index(index).comment
        return self.samples[index].comment

    def sample_coord_str_at_index(self, sample_index, sample_point_index):
        # return self.sample_at_index(sample_index).index_coordinate_str(sample_point_index)
        return self.samples[sample_index].index_coordinate_str(sample_point_index)

    def sample_point_uid_at_index(self, sample_index, sample_point_index):
        return self.samples[sample_index].index_uid(sample_point_index)

    def sample_point_index_position_index(self, sample_index, sample_point_index):
        return self.samples[sample_index].index_position_index(sample_point_index)

    def sample_coordinate_dict_at_index(self, sample_index, sample_point_index):
        return self.samples[sample_index].index_coordinate_dict(sample_point_index)

    def sample_exposed_at_index(self, sample_index, sample_point_index):
        return self.samples[sample_index].index_exposed(sample_point_index)

    @emit_list_update_signal_decorator
    def add_points_to_sample_at_index(self, index, sample_coordinate_list):
        self.samples[index].add_new_positions_from_coordinates(sample_coordinate_list)

    @emit_list_update_signal_decorator
    def set_as_exposed_with_index_dict(self, index_dict, exposed=True):
        for sample_index, point_index_list in index_dict.items():
            for point_index in set(point_index_list):
                self.set_exposed_at_index(sample_index, point_index, exposed=exposed)

    @property
    def uids(self):
        _uids = []
        for sample in self.samples:
            _uids.append(sample.uids)
        return _uids

    def uid_to_sample_index(self, uid):
        for sample_index, sample in enumerate(self.all_samples):
             if uid == sample.uid:
                return sample_index

    def set_exposed_at_index(self, sample_index, sample_point_index, exposed=True):
        self.samples[sample_index].set_exposed(sample_point_index, exposed=exposed)

    @emit_list_update_signal_decorator
    def sort_sample_positions_by_at_index(self, sample_index, keys):
        self.samples[sample_index].sort_positions_by(keys)

    # def uid_to_sample_index(self, uid):
    #     for sample_index, sample in enumerate(self.samples):
    #         if uid in sample.uids:
    #             sample_point_index = sample.uids.index(uid)
    #             return sample_index, sample_point_index
    #     return None
    #
    #
    # def sample_exposed_at_uid(self, uid):
    #     index_tuple = self.sample_point_uid_to_sample_index(uid)
    #     if index_tuple is not None:
    #         self.sample_exposed_at_index(*index_tuple)
    #
    # def set_exposed_at_uid(self, uid):
    #     index_tuple = self.sample_point_uid_to_sample_index(uid)
    #     if index_tuple is not None:
    #         self.set_exposed_at_index(*index_tuple)


    @emit_list_update_signal_decorator
    def archive_at_index(self, index):
        if type(index) != list:
            index = [index]
        unarchived_samples = [self.samples[i] for i in index]
        for sample in unarchived_samples:
            if not ((sample.name == 'foil') and (sample.uid == FOIL_UID)):
                sample.archived = True

    @emit_list_update_signal_decorator
    def restore_at_index(self, index):
        if type(index) != list:
            index = [index]
        archived_samples = [self.archived_samples[i] for i in index]
        for sample in archived_samples:
            sample.archived = False




sample_manager = SampleManager()


class BatchScanManager(PersistentListInteractingWithGUI):
    def __init__(self, json_file_path = f'{ROOT_PATH_SHARED}/settings/json/scan_sequence_manager.json'):
        super().__init__(json_file_path)

    #Class specific decorators
    @property
    def scans(self):
        return self.items

    @scans.setter
    def scans(self, value):
        self.items = value

    @scans.deleter
    def scans(self):
        del self.items

    @property
    def required_keys(self):
        return ['name', 'repeat', 'delay', 'scan_idx', 'scan_local_dict']

    # validator
    def validate_element(self, element_dict):
        if element_dict['type'] == 'scan':
            required_keys = self.required_keys
        else:
            raise Exception(f'Type of scan element is unknown: {element_dict}')

        valid = all([(k in element_dict.keys()) for k in required_keys])
        if not valid: raise Exception(f'element contains missing keys.\n'
                                      f'Element: {element_dict}. Required keys: {required_keys}')

    def add_element(self, element_dict):
        self.validate_element(element_dict)
        self.add_item(element_dict)

    def delete_element(self, element_index):
        if type(element_index) == int:
            self.delete_item_at_index(element_index)

    def delete_many_elements(self, element_index_list):
        self.delete_multiple_items(element_index_list)

    def update_element_at_index(self, element_index, element_dict):
        self.validate_element(element_dict)
        if type(element_index) == int:
            self.update_item_at_index(element_index, element_dict)

    def check_if_scan_index_is_used(self, scan_index):
        bad_indexes = []
        for i, scan in enumerate(self.scans):
            if scan['type'] == 'scan':
                if scan['scan_idx'] == scan_index:
                    bad_indexes.append(i)
        return bad_indexes

    def scan_at_index(self, index):
        return self.item_at_index(index)

    def scan_str_at_index(self, index):
        return self.scan_at_index(index)['name']



scan_sequence_manager = BatchScanManager()
# batch_scan_manager = BatchScanManager()

# class ScanSequenceManager(PersistentListInteractingWithGUI):
#     def __init__(self, json_file_path = '/nsls2/xf08id/settings/json/scan_sequence_manager.json'):
#         super().__init__(json_file_path)
#         self.local_file_default_path = f"{ROOT_PATH}/{USER_PATH}/{RE.md['year']}/{RE.md['cycle']}/{RE.md['proposal']}/"
def chunker(seq, size):
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))



class BatchManager(PersistentListInteractingWithGUI):

    def __init__(self, sample_manager : SampleManager, scan_manager: ScanManager, scan_sequence_manager : BatchScanManager,
                 json_file_path=f'{ROOT_PATH_SHARED}/settings/json/batch_manager.json'):
        super().__init__(json_file_path)

        self.sample_manager = sample_manager
        self.scan_manager = scan_manager
        self.scan_sequence_manager = scan_sequence_manager

    @property
    def experiments(self):
        return self.items

    @experiments.setter
    def experiments(self, value):
        self.items = value

    @experiments.deleter
    def experiments(self):
        del self.items

    # validator
    def validate_element(self, element_dict):
        if element_dict['type'] == 'experiment':
            required_keys = ['name', 'repeat', 'element_list']
        elif element_dict['type'] == 'scan':
            required_keys = self.scan_sequence_manager.required_keys
        elif element_dict['type'] == 'sample':
            required_keys = ['name', 'comment', 'sample_name', 'sample_comment', 'sample_condition', 'sample_uid', 'sample_coordinates']
        elif element_dict['type'] == 'service':
            required_keys = ['plan_name', 'plan_kwargs']
        else:
            raise Exception(f'Type of scan element is unknown: {element_dict}')

        valid = all([(k in element_dict.keys()) for k in required_keys])
        if not valid: raise Exception(f'element contains missing keys.\n'
                                      f'Element: {element_dict}. Required keys: {required_keys}')

        if 'element_list' in element_dict.keys():
            for element in element_dict['element_list']:
                self.validate_element(element)

    def add_experiment_from_dict(self, experiment_dict):
        self.validate_element(experiment_dict)
        self.add_item(experiment_dict)
        # self.emit_batch_list_update_signal()

    def add_new_experiment(self, name, repeat):
        experiment_dict = {'type' : 'experiment', 'name' : name, 'repeat' : repeat, 'element_list' : []}
        self.add_experiment_from_dict(experiment_dict)

    @emit_list_update_signal_decorator
    def add_element_to_experiment(self, experiment_index, element_dict):
        self.validate_element(element_dict)
        self.experiments[experiment_index]['element_list'].append(element_dict)

    # def sample_index_iterator(self, sample_index_dict):
    #     for sample_index, point_index_list in sample_index_dict.items():
    #         for point_index in point_index_list:
    #             yield (sample_index, point_index)

    def sample_info_from_index(self, sample_index, sample_point_index, sample_condition=None):
        sample_name = self.sample_manager.sample_name_at_index(sample_index)
        actual_sample_point_index = self.sample_manager.sample_point_index_position_index(sample_index, sample_point_index)
        sample_point_str = f'(pos {(actual_sample_point_index+1):0>3d})'
        if sample_condition is not None:
            name = f'{sample_name} {sample_condition} {sample_point_str}'
        else:
            name = f'{sample_name} {sample_point_str}'
        sample_comment = self.sample_manager.sample_comment_at_index(sample_index)
        sample_uid = self.sample_manager.sample_uid_at_index(sample_index)
        sample_coordinates = self.sample_manager.sample_coordinate_dict_at_index(sample_index, sample_point_index)
        return name, sample_name, sample_comment, sample_uid, sample_coordinates

    def scan_iterator(self, scan_indexes):
        for scan_index in scan_indexes:
            scan_dict = self.scan_sequence_manager.scan_at_index(scan_index)
            yield {**{'type' : 'scan'}, **scan_dict}

    def sample_iterator(self, sample_index_dict, sample_condition=None, comment=''):
        for sample_index, point_index_list in sample_index_dict.items():
            for point_index in point_index_list:
                name, sample_name, sample_comment, sample_uid, sample_coordinates = self.sample_info_from_index(sample_index, point_index, sample_condition=sample_condition)
                yield {'type' : 'sample',
                       'name' : name, 'comment' : comment,
                       'sample_name': sample_name,
                       'sample_comment' : sample_comment,
                       'sample_condition': sample_condition,
                       'sample_uid' : sample_uid,
                       'sample_coordinates' : sample_coordinates}

    def sample_point_intertor(self, sample_index_dict, sample_condition=None, comment=''):
        # reshuffle the dictionary
        sample_point_index_dict = defaultdict(lambda : [])
        for sample_index, point_index_list in sample_index_dict.items():
            for point_index in point_index_list:
                sample_point_index_dict[point_index].append(sample_index)

        for point_index, sample_index_list in sample_point_index_dict.items():
            for sample_index in sample_index_list:
                name, sample_name, sample_comment, sample_uid, sample_coordinates = self.sample_info_from_index(sample_index, point_index, sample_condition=sample_condition)
                yield {'type' : 'sample',
                       'name' : name, 'comment' : comment,
                       'sample_name': sample_name,
                       'sample_comment' : sample_comment,
                       'sample_condition': sample_condition,
                       'sample_uid' : sample_uid,
                       'sample_coordinates' : sample_coordinates}

    @emit_list_update_signal_decorator
    def add_measurement_to_experiment(self, experiment_index, sample_index_dict, scan_indexes,
                                      priority='scan', sample_condition=None, comment=''):
        if priority == 'scan':
            element_iterator = self.scan_iterator(scan_indexes)
            element_list_iterator = self.sample_iterator(sample_index_dict, sample_condition=sample_condition, comment=comment)
        elif priority == 'sample':
            element_iterator = self.sample_iterator(sample_index_dict, sample_condition=sample_condition, comment=comment)
            element_list_iterator = self.scan_iterator(scan_indexes)
        elif priority == 'sample_point':
            element_iterator = self.sample_point_intertor(sample_index_dict, sample_condition=sample_condition, comment=comment)
            element_list_iterator = self.scan_iterator(scan_indexes)

        element_list = list(element_list_iterator)
        for element_dict in element_iterator:

            # special treatment of rixs scans - A BANDAID
            if ((priority == 'scan') and
                (element_dict['scan_local_dict']['aux_parameters']['scan_key'] == 'johann_rixs')):
                n_eff = len(element_dict['scan_local_dict']['aux_parameters']['spectrometer']['scan_parameters']['energy_grid'])
                chunked_element_list = list(chunker(element_list, n_eff))
                repeats = element_dict['repeat']
                chunked_element_list = chunked_element_list[:repeats]
                element_dict_one_repeat = copy.deepcopy(element_dict)
                element_dict_one_repeat['repeat'] = 1
                element_dict_one_repeat['name'] = element_dict_one_repeat['name'].replace(f' x{repeats} ', ' x1 ')
                for _element_list in chunked_element_list:
                    measurement = {**element_dict_one_repeat, **{'element_list': _element_list}}
                    self.add_element_to_experiment(experiment_index, measurement)
            else:
                measurement = {**element_dict, **{'element_list': element_list}}
                self.add_element_to_experiment(experiment_index, measurement)

    @emit_list_update_signal_decorator
    def add_service_to_element_list(self, index_tuple, service_dict):
        self.validate_element(service_dict)
        nidx = len(index_tuple)
        if nidx == 1:
            experiment_index = index_tuple[0]
            # self.experiments[experiment_index]['element_list'].insert(0, service_dict)
            element_to_service = self.experiments[experiment_index]
            service_dict['element_to_service'] = {k : v for k, v in element_to_service.items() if k!='element_list'}
            element_to_service['element_list'].append(service_dict)
        elif nidx == 2:
            experiment_index, element_index1 = index_tuple
            element_to_service = self.experiments[experiment_index]['element_list'][element_index1]
            service_dict['element_to_service'] = {k : v for k, v in element_to_service.items() if k!='element_list'}
            self.experiments[experiment_index]['element_list'].insert(element_index1, service_dict)
        elif nidx == 3:
            experiment_index, element_index1, element_index2 = index_tuple
            element_to_service = self.experiments[experiment_index]['element_list'][element_index1]['element_list'][element_index2]
            service_dict['element_to_service'] = {k : v for k, v in element_to_service.items() if k!='element_list'}
            self.experiments[experiment_index]['element_list'][element_index1]['element_list'].insert(element_index2, service_dict)

    # def sample_point_data_from_index(self, sample_index, sample_point_index):
    #     sample = self.sample_manager.samples[sample_index]
    #     return self.sample_manager.samples[sample_index].position_data.iloc[sample_point_index]

    def sample_str_from_element(self, sample_element):
        name = sample_element['name']
        point_coord_str = ' '.join([(f"{key}={value : 0.2f}") for key, value in sample_element['sample_coordinates'].items()])
        return f'{name} at {point_coord_str} '

    def sample_str_from_index(self, sample_index, sample_point_index):
        sample_str = self.sample_manager.sample_name_at_index(sample_index)
        point_str = self.sample_manager.sample_coord_str_at_index(sample_index, sample_point_index)
        return f'{sample_str} at {point_str} (point #{sample_point_index + 1})'

    def scan_str_from_element(self, scan_element):
        return scan_element['name']

        # return self.scan_str_from_index(scan_element['scan_sequence_index'])

    def scan_str_from_index(self, scan_index):
        return self.scan_sequence_manager.scan_str_at_index(scan_index)

    def service_str_from_element(self, service_element):
        return f"{service_element['plan_name']} ({service_element['plan_kwargs']})"

    @emit_list_update_signal_decorator
    def delete_element(self, index_tuple):
        nidx = len(index_tuple)
        if nidx == 1:
            experiment_index = index_tuple[0]
            self.delete_item_at_index(experiment_index, emit_signal=False)
        elif nidx == 2:
            experiment_index, element_index1 = index_tuple
            self.experiments[experiment_index]['element_list'].pop(element_index1)
        elif nidx == 3:
            experiment_index, element_index1, element_index2 = index_tuple
            element = self.experiments[experiment_index]['element_list'][element_index1]
            element['element_list'].pop(element_index2)
            if len(element['element_list']) == 0:
                self.delete_element(index_tuple[:-1], emit_signal=False)

    def get_booked_sample_points_list(self):
        pass

    # def parse_element_into_plan(self, element):
    #     plans = []
    #     if element['type'] == 'experiment':
    #         pass
    #     elif

    def get_sample_data_from_sample_element(self, sample_element):
        sample_metadata = {'sample_name': sample_element['sample_name'],
                           'sample_comment': sample_element['sample_comment'],
                           'sample_condition': sample_element['sample_condition'],
                           'sample_uid': sample_element['sample_uid']}

        return (sample_element['name'], sample_element['comment'],
                sample_element['sample_coordinates'],
                sample_metadata)

    def get_scan_name_from_scan_dict(self, scan_dict):
        if 'name' in scan_dict.keys():
            scan_name = scan_dict['name']
        else:
            try:
                scan_description = scan_dict['aux_parameters']['scan_description']
                scan_def = scan_dict['scan_def']
                _idx = scan_def.index('(' + scan_description)
                scan_name = scan_def[:_idx]
            except:
                scan_name = scan_def
        return scan_name

    def get_scan_data_from_scan_element(self, scan_element):
        actual_scan_dict = scan_element['scan_local_dict']
        scan_idx = scan_element['scan_idx']
        nominal_scan_dict = self.scan_manager.scan_list_local[scan_idx]
        if actual_scan_dict == nominal_scan_dict:
            scan_name = self.get_scan_name_from_scan_dict(actual_scan_dict)
            scan_key = actual_scan_dict['aux_parameters']['scan_key']
            return scan_element['repeat'], scan_element['delay'], scan_idx, scan_name, scan_key
        else:
            # raise Exception('Seems like the scan for batch measurement was deleted/cannot be found')
            print('Warning: Seems like the scan for batch measurement was deleted/cannot be found')

    def prepare_scan_plan_from_scan_element(self, scan_element):
        actual_scan_dict = scan_element['scan_local_dict']
        return [{'plan_name' : 'prepare_scan_plan',
                 'plan_kwargs' : {'scan_uid' : actual_scan_dict['uid'],
                                  'aux_parameters' : actual_scan_dict['aux_parameters']}}]

    def get_data_from_element(self, element):
        if element['type'] == 'scan':
            return self.get_scan_data_from_scan_element(element)
        elif element['type'] == 'sample':
            return self.get_sample_data_from_sample_element(element)

    def convert_service_to_plans(self, service_element):
        plans = []
        if 'element_to_service' in service_element.keys():
            element_to_service = service_element['element_to_service']
            if element_to_service['type'] == 'scan':
                # print('THIS IS SCAN')
                scan_uid = element_to_service['scan_local_dict']['uid']
                aux_parameters = element_to_service['scan_local_dict']['aux_parameters']
                plans.append({'plan_name': 'prepare_scan_plan',
                              'plan_kwargs': {'scan_uid' : scan_uid, 'aux_parameters' : aux_parameters}})
            elif element_to_service['type'] == 'sample':
                # print('THIS IS SAMPLE')
                sample_coordinates = element_to_service['sample_coordinates']
                plans.append({'plan_name': 'move_sample_stage_plan',
                              'plan_kwargs': {'sample_coordinates': sample_coordinates}})
        plans.append({'plan_name': service_element['plan_name'], 'plan_kwargs' : service_element['plan_kwargs']})
        return plans


    def generate_plan_list(self):
        plans = []
        for experiment in self.experiments:
            for i in range(experiment['repeat']):
                for element in experiment['element_list']:

                    if element['type'] == 'scan':
                        repeat, delay, scan_idx, scan_name, scan_key = self.get_data_from_element(element)
                        new_plans = self.prepare_scan_plan_from_scan_element(element)

                        if scan_key != 'johann_rixs':
                            for sub_element in element['element_list']:
                                if sub_element['type'] == 'sample':
                                    name, comment, sample_coordinates, sample_metadata  = self.get_data_from_element(sub_element)
                                    sample_plans = [{'plan_name': 'move_sample_stage_plan',
                                                     'plan_kwargs': {'sample_coordinates': sample_coordinates}}]
                                    scan_plans = self.scan_manager.generate_plan_list(name, comment,
                                                                                      repeat, delay, scan_idx,
                                                                                      metadata=sample_metadata)
                                    new_plans.extend(sample_plans + scan_plans)
                                elif sub_element['type'] == 'service':
                                    new_plans.extend(self.convert_service_to_plans(sub_element))
                        else:
                            sample_coordinates_list = []
                            sample_name_for_scan_list = []
                            for sub_element in element['element_list']:
                                if sub_element['type'] == 'sample':
                                    name, comment, sample_coordinates, sample_metadata = self.get_data_from_element(sub_element)
                                    sample_coordinates_list.append(sample_coordinates)
                                    sample_name_for_scan_list.append(name)
                                elif sub_element['type'] == 'service':
                                    new_plans.extend(self.convert_service_to_plans(sub_element))


                            print(len(sample_coordinates_list))
                            if len(sample_coordinates_list) == 1:
                                sample_coordinates_list = sample_coordinates_list[0]
                            if len(sample_name_for_scan_list) == 1:
                                sample_name_for_scan_list = sample_name_for_scan_list[0]

                            scan_plans = self.scan_manager.generate_plan_list(sample_name_for_scan_list,
                                                                              comment,
                                                                              repeat, delay, scan_idx,
                                                                              sample_coordinates=sample_coordinates_list,
                                                                              metadata=sample_metadata)
                            new_plans.extend(scan_plans)

                    elif element['type'] == 'sample':
                        name, comment, sample_coordinates, sample_metadata = self.get_data_from_element(element)
                        new_plans = [{'plan_name': 'move_sample_stage_plan',
                                      'plan_kwargs': {'sample_coordinates': sample_coordinates}}]
                        for sub_element in element['element_list']:
                            if sub_element['type'] == 'scan':
                                repeat, delay, scan_idx, scan_name, scan_key  = self.get_data_from_element(sub_element)
                                # sample_name_for_scan = f'{sample_name} {scan_name}'
                                new_plans.extend(self.scan_manager.generate_plan_list(name, comment,
                                                                                      repeat, delay, scan_idx,
                                                                                      sample_coordinates=sample_coordinates,
                                                                                      metadata=sample_metadata))

                            elif sub_element['type'] == 'service':
                                new_plans.extend(self.convert_service_to_plans(sub_element))
                    elif element['type'] == 'service':
                        new_plans = self.convert_service_to_plans(element)


                    plans.extend(new_plans)

        return plans


batch_manager = BatchManager(sample_manager, scan_manager, scan_sequence_manager)



# def batch_parse_and_run(hhm,sample_stage,batch,plans_dict ):
#     # tm = trajectory_manager(hhm)
#     for ii in range(batch.rowCount()):
#         experiment = batch.item(ii)
#         print(experiment.item_type)
#         repeat=experiment.repeat
#         print(repeat)
#         for jj in range(experiment.rowCount()):
#             child = experiment.child(jj)
#             print(child.item_type)
#             if child.item_type == 'sample':
#                 print('  ' + sample.name)
#                 print('  ' + str(sample.x))
#                 print('  ' + str(sample.y))
#                 yield from mv(sample_stage.x, sample.x, sample_stage.y, sample.y)
#                 for kk in range(sample.rowCount()):
#                     scan = sample.child(kk)
#                     traj_index= scan.trajectory
#                     print('      ' + scan.scan_type)
#                     plan = plans_dict[scan.scan_type]
#                     kwargs = {'name': sample.name,
#                               'comment': '',
#                               'delay': 0,
#                               'n_cycles': repeat}
#                     trajectory_manager.init(traj_index+1)
#                     yield from plan(**kwargs)
#             elif child.item_type == 'service':
#                     print(child.service_plan)

# summarize_plan(parse_and_execute())