print(f"Loading {__file__}...")

import os
import h5py
import sys
import numpy as np
import time as ttime
import itertools
from uuid import uuid4
from event_model import compose_resource

from ophyd.areadetector.plugins import PluginBase
from ophyd import Signal, EpicsSignal, DeviceStatus
from ophyd import Component as Cpt
from ophyd.areadetector.filestore_mixins import FileStorePluginBase
from ophyd.device import Staged
from ophyd.sim import NullStatus
from enum import Enum
from collections import deque, OrderedDict

from ophyd.areadetector import Xspress3Detector
from nslsii.areadetector.xspress3 import (
    build_xspress3_class,
    Xspress3HDF5Plugin,
    Xspress3Trigger,
    Xspress3FileStore,
)

# this is the community IOC package
from nslsii.areadetector.xspress3 import (
    build_xspress3_class
)

from databroker.assets.handlers import XS3_XRF_DATA_KEY as XRF_DATA_KEY


# def __init__(
#         self,
#         *args,
#         root_path,
#         path_template,
#         resource_kwargs,
#         **kwargs,

class ISSXspress3HDF5Plugin(Xspress3HDF5Plugin):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if kwargs["root_path"] is None:
            self.root_path.put(self.root_path_str)
        if kwargs["path_template"] is None:
            self.path_template.put(self.path_template_str)    

    def stage(self, *args, **kwargs):
        self.root_path.put(self.root_path_str)
        return super().stage()

    @property
    def root_path_str(self):
        # data_session = self._redis_dict["data_session"]
        # cycle = self._redis_dict["cycle"]
        data_session = RE.md["data_session"]
        cycle = RE.md["cycle"]
        # if "Commissioning" in get_proposal_type():
        #     root_path = f"/nsls2/data/iss/proposals/commissioning/{data_session}/assets/xspress3x/"
        # else:
        root_path = f"/nsls2/data/iss/proposals/{cycle}/{data_session}/assets/xspress3/"
        return root_path

    @property
    def path_template_str(self):
        path_template = "%Y/%m/%d"
        return path_template


# build a community IOC xspress3 class with 4 channels
CommunityXspress3_4Channel = build_xspress3_class(
    channel_numbers=(1, 2, 3, 4),
    mcaroi_numbers=(1, 2, 3, 4),
    image_data_key="fluor",  # TODO:
    xspress3_parent_classes=(Xspress3Detector, Xspress3Trigger),
    extra_class_members={
        "hdf5": Cpt(
            ISSXspress3HDF5Plugin,
            "HDF1:",
            name="hdf5",
            resource_kwargs={},
            path_template=None,
            root_path=None,
        )
    }
)

class ISSXspress3CDetector(CommunityXspress3_4Channel):
    def __init__(self, prefix, *, configuration_attrs=None, read_attrs=None, **kwargs):
        if configuration_attrs is None:
            configuration_attrs = ["external_trig", "total_points", "spectra_per_point", "cam", "rewindable"]

        # for chn in range(8):
        #     for roin in range(4):
        #         configuration_attrs.append(f'xsx_channel{chn:02d}_mcaroi{roin:02d}_min_x')
        #         configuration_attrs.append(f'xsx_channel{chn:02d}_mcaroi{roin:02d}_size_x')

        super().__init__(prefix, configuration_attrs=configuration_attrs, read_attrs=read_attrs, **kwargs)
        self.channels = {}
        for ic, channel in enumerate(self.iterate_channels()):
            self.channels[ic] = channel
            channel.kind = 7
            for mcaroi in channel.iterate_mcarois():
                mcaroi.kind = "Hinted"
                mcaroi.total_rbv.kind = "Hinted"
                mcaroi.min_x.kind = 'Config'
                mcaroi.size_x.kind = 'Config'
        self._asset_docs_cache = deque()
        self._datum_counter = None
        self._datum_ids = []

        self.cam.acquire.put(0)
        self.cam.trigger_mode.put(1)  # put the trigger mode to internal
        self.cam.num_images.put(1)
        self.cam.acquire_time.put(1)
        self.cam.erase.put(1)
        self._acquisition_signal = self.cam.acquire

    def stage(self, *args, **kwargs):
        self.cam.acquire.put(0, wait=True)
        self.cam.erase.put(1, wait=True)
        self.cam.trigger_mode.put(1)
        self.cam.num_images.put(1)
        self.cam.acquire_time.put(1, wait=True)
        super().stage(*args, **kwargs)

    def unstage(self, *args, **kwargs):
        self.cam.acquire.put(0, wait=True)
        self.cam.trigger_mode.put(1)  # put the trigger mode to internal
        self.cam.num_images.put(1)
        self.cam.acquire_time.put(1)
        self.cam.erase.put(1, wait=True)
        super().unstage(*args, **kwargs)

    def set_limits_for_roi(self, energy_nom, roi, window='auto'):
        # In [8]: xsc.channel01.mcaroi01.total_rbv.get()
        # Out[8]: 0.0

        for ch_index, channel in self.channels.items():
            if window == 'auto':
                w = _compute_window_for_xs_roi_energy(energy_nom)
            else:
                w = int(window)
            energy = _convert_xs_energy_nom2act(energy_nom, ch_index)
            ev_low_new = int(energy - w / 2) // 10
            ev_high_new = int(energy + w / 2) // 10
            size_new = ev_high_new - ev_low_new

            roi_obj = getattr(channel, f'mcaroi{roi:02d}')
            roi_obj.min_x.put(ev_low_new)
            roi_obj.size_x.put(size_new)


            # roi_obj = getattr(channel.rois, roi)
            # current_high = roi_obj.min_x.get() + roi_obj.size_x.get()
            # if ev_high_new < current_high:
            #     # roi_obj.ev_low.put(ev_low_new)
            #     # roi_obj.ev_high.put(ev_high_new)
            #     roi_obj.min_x.put(ev_low_new)
            #     roi_obj.size_x.put(size_new)
            # else:
            #     roi_obj.size_x.put(ev_high_new)
            #     roi_obj.min_x.put(ev_low_new)

    def set_exposure_time(self, new_exp_time):
        self.cam.acquire_time.set(new_exp_time).wait()

    def read_exposure_time(self):
        return self.cam.acquire_time.get()

    def test_exposure(self, acq_time=1, num_images=1):
        _old_acquire_time = self.cam.acquire_time.value
        _old_num_images = self.cam.num_images.value
        self.set_exposure_time(acq_time)
        self.cam.num_images.set(num_images).wait()
        self.cam.erase.put(1)
        self._acquisition_signal.put(1, wait=True)
        # self.settings.acquire_time.set(_old_acquire_time).wait()
        self.set_exposure_time(_old_acquire_time)
        self.cam.num_images.set(_old_num_images).wait()



class ISSXspress3CDetectorStream(ISSXspress3CDetector):
    hints = None

    def stage(self, acq_rate, traj_time, *args, **kwargs):

        self.hdf5.file_write_mode.put(2)  # put it to Stream |||| IS ALREADY STREAMING
        self.external_trig.put(True)
        self.set_expected_number_of_points(acq_rate, traj_time)
        self.spectra_per_point.put(1)
        super().stage(*args, **kwargs)

        self.cam.trigger_mode.put(3)  # put the trigger mode to TTL in
        # self.cam.erase.put(1)

        self.hdf5._resource['spec'] = "XSP3X"
        self._datum_counter = itertools.count()
        # note, hdf5 is already capturing at this point
        self.cam.num_images.put(self._num_points)
        self.cam.acquire.put(1)  # start recording data

    def unstage(self):
        self.hdf5.capture.put(0)
        # self.cam.acquire.put(0)
        # self.cam.trigger_mode.put(1)  # put the trigger mode to internal
        # self.cam.num_images.put(1)
        # self.cam.acquire_time.put(1)
        # self.cam.erase.put(1)
        super().unstage()
        self._datum_counter = None

    def set_expected_number_of_points(self, acq_rate, traj_time):
        self._num_points = int(acq_rate * (traj_time + 1))
        self.total_points.put(self._num_points)

    def describe_collect(self):
        return_dict = {self.name:
                           {f'{self.name}': {'source': 'XSX',
                                             'dtype': 'array',
                                             'shape': [self.cam.num_images.get(),
                                                       # self.settings.array_counter.get()
                                                       self.hdf5.array_size.height.get(),
                                                       self.hdf5.array_size.width.get()],
                                             'filename': f'{self.hdf5.full_file_name.get()}',
                                             'external': 'FILESTORE:'}}}
        return return_dict

    def collect(self):
        # num_frames = len(self._datum_ids)
        num_frames = self.hdf5.num_captured.get()
        # break num_frames up and yield in sections?

        for frame_num in range(num_frames):
            datum_id = self._datum_ids[frame_num]
            data = {self.name: datum_id}
            ts = ttime.time()

            yield {'data': data,
                   'timestamps': {key: ts for key in data},
                   'time': ts,  # TODO: use the proper timestamps from the mono start and stop times
                   'filled': {key: False for key in data}}
            # print(f"-------------------{ts}-------------------------------------")

    def collect_asset_docs(self):
        items = list(self._asset_docs_cache)
        self._asset_docs_cache.clear()
        for item in items:
            yield item

    def complete(self, *args, **kwargs):
        for resource in self.hdf5._asset_docs_cache:
            res_dict = resource[1]
            self._asset_docs_cache.append(('resource', res_dict))

        self._datum_ids = []

        num_frames = self.hdf5.num_captured.get()

        for frame_num in range(num_frames):
            # for channel in self.iterate_channels():
            datum_id = '{}/{}'.format(self.hdf5._resource['uid'], next(self._datum_counter))
            datum = {'resource': self.hdf5._resource['uid'],
                     'datum_kwargs': {'frame': frame_num}, # 'channel': channel.channel_number},
                     'datum_id': datum_id}
            self._asset_docs_cache.append(('datum', datum))
            self._datum_ids.append(datum_id)

        return NullStatus()


xsc = ISSXspress3CDetector('XF:08IDB-ES{Xsp:3}:', name='xsc')
xsc_stream = ISSXspress3CDetectorStream('XF:08IDB-ES{Xsp:3}:', name='xsc_stream')


xsc_stream.hints = {'fields': []}

from databroker.assets.handlers import HandlerBase, Xspress3HDF5Handler

# Not required for tiled writing
class ISSXspress3CHDF5Handler(Xspress3HDF5Handler):
    HANDLER_NAME = "XSP3X"
    XRF_DATA_KEY = "entry/instrument/detector/data"
    def __init__(self, *args, **kwargs):
        print("Handler init kwargs", kwargs)
        # kwargs.pop('join_method', 'concat')
        # kwargs.pop('chunk_shape', [1])
        # kwargs.pop('dataset', '')
        super().__init__(*args, **kwargs)
        # print("XSP3X _file", self._file)
        # self._filepath = filepath

    def _get_dataset(self):
        if hasattr(self, '_num_images') and self._num_images is not None:
            # print("No more data to return")
            return
        arr_data = np.asarray(self._file[self.XRF_DATA_KEY])
        shape = arr_data.shape
        self._num_images = shape[1]
        self._array_data = arr_data

    def __call__(self, **kwargs):
        self._get_dataset()
        frame_number = kwargs.get('frame')
        if frame_number is None:
            return
        # print(kwargs)
        # print(self._file)
        # with h5py.File(self._file, "r") as f:
        # arr_data = np.asarray(self._file[self.XRF_DATA_KEY])
        # print(arr_data.shape)
        return self._array_data[frame_number, :, :]

