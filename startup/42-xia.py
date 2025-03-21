print(ttime.ctime() + ' >>>> ' + __file__)

import logging
logger = logging.getLogger(__name__)
#from ophyd import EpicsSignal, Signal, SignalWithRBV, EpicsSignalRO, Kind
from ophyd.device import (BlueskyInterface, Staged)
from ophyd.device import (Device,
                          DynamicDeviceComponent as DDC,
                          Component as Cpt)
from ophyd.areadetector.plugins import PluginBase
from ophyd.areadetector.filestore_mixins import FileStorePluginBase
from ophyd.areadetector.plugins import HDF5Plugin_V33
from ophyd.areadetector import (DetectorBase, CamBase)
from ophyd.sim import NullStatus
import time
from collections import OrderedDict
import h5py
from databroker.assets.handlers import (Xspress3HDF5Handler, XS3_XRF_DATA_KEY as XRF_DATA_KEY)



#class Xspress3FileStore(FileStorePluginBase, HDF5Plugin):
class XIAXMAPFileStore(FileStorePluginBase, HDF5Plugin_V33):
    '''XIA XMAP acquisition -> filestore'''
    # num_capture_calc = Cpt(EpicsSignal, 'NumCapture_CALC')
    # num_capture_calc_disable = Cpt(EpicsSignal, 'NumCapture_CALC.DISA')
    filestore_spec = "XIA_XMAP_HDF5"
    #filestore_spec = "AD_HDF5"  #Xspress3HDF5Handler.HANDLER_NAME

    def __init__(self, basename, *, config_time=0.5,
                 mds_key_format='{self.settings.name}_ch{chan}', parent=None,
                 **kwargs):
        super().__init__(basename, parent=parent, **kwargs)
        det = parent
        self.settings = det.settings

        # Use the EpicsSignal file_template from the detector
        self.stage_sigs[self.blocking_callbacks] = 1
        self.stage_sigs[self.enable] = 1
        self.stage_sigs[self.compression] = 'zlib'
        self.stage_sigs[self.file_template] = '%s%s_%6.6d.h5'

        self._filestore_res = None
        self.create_directory.set(6).wait()
#        self.channels = list(range(1, len([_ for _ in det.component_names
#                                           if _.startswith('chan')]) + 1))
        self.channels = list(range(1, 33))  # TODO: REmove hardcoded values
        # self.channels = list(range(1, int(det.settings.num_channels)+1))
        # this was in original code, but I kinda-sorta nuked because
        # it was not needed for SRX and I could not guess what it did
        self._master = None

        self._config_time = config_time
        self.mds_keys = {chan: mds_key_format.format(self=self, chan=chan)
                         for chan in self.channels}

    def stop(self, success=False):
        ret = super().stop(success=success)
        self.capture.put(0)
        return ret

    def kickoff(self):
        # need implementation
        raise NotImplementedError()

    def collect(self):
        # (hxn-specific implementation elsewhere)
        raise NotImplementedError()

    def make_filename(self):
        fn, rp, write_path = super().make_filename()
        if self.parent.make_directories.get():
            makedirs(write_path)
        return fn, rp, write_path

    def unstage(self):
        try:
            i = 0
            # this needs a fail-safe, RE will now hang forever here
            # as we eat all SIGINT to ensure that cleanup happens in
            # orderly manner.
            # If we are here this is a sign that we have not configured the xs3
            # correctly and it is expecting to capture more points than it
            # was triggered to take.
            while self.capture.get() == 1:  # HDF5 plugin .capture
                i += 1
                if (i % 50) == 0:
                    logger.warning('Still capturing data .... waiting.')
                time.sleep(0.1)
                if i > 150:
                    logger.warning('Still capturing data .... giving up.')
                    logger.warning('Check that the XIA XMAP is configured to take the right '
                                   'number of frames '
                                   f'(it is trying to take {self.parent.settings.num_images.get()})')
                    self.capture.put(0)
                    break

        except KeyboardInterrupt:
            self.capture.put(0)
            logger.warning('Still capturing data .... interrupted.')

        return super().unstage()

    def generate_datum(self, key, timestamp, datum_kwargs):
        sn, n = next((f'channel{j}', j)  # TODO:
                     for j in self.channels
                     if getattr(self.parent.channels, f'mca{j:1d}').name == key)
        datum_kwargs.update({'frame': self.parent._abs_trigger_count,
                             'channel': int(sn[7:])})  # No idea what's happening here
        self.mds_keys[n] = key
        # print(f"{datum_kwargs=}")
        super().generate_datum(key, timestamp, datum_kwargs)

    def stage(self):
        # if should external trigger
        ext_trig = self.parent.external_trig.get()
        print("XIA External_trigger", ext_trig)

        logger.debug('Stopping XIA XMAP acquisition')
        # really force it to stop acquiring
        self.settings.stop_all.put(1)

        total_points = self.parent.total_points.get()
        if total_points < 1:
            raise RuntimeError("You must set the total points")
        # print(total_points)
#        spec_per_point = self.parent.spectra_per_point.get()
#        total_capture = total_points * spec_per_point

        # stop previous acquisition
        # self.stage_sigs[self.settings.acquire] = 0
        # self.stage_sigs[self.settings.stop_all] = 1
        # print(self.stage_sigs[self.settings.acquire])

        # re-order the stage signals and disable the calc record which is
        # interfering with the capture count
        self.stage_sigs.pop(self.num_capture, None)
        self.stage_sigs.pop(self.settings.num_images, None)
        # self.stage_sigs[self.num_capture_calc_disable] = 1
        # print(self.stage_sigs)

        if ext_trig:
            print('Setting up external triggering')
            logger.debug('Setting up external triggering')
            # self.stage_sigs[self.settings.collection_mode] = 2  # SCA Mapping
            # self.stage_sigs[self.settings.trigger_mode] = 0  # Gate
#            self.stage_sigs[self.settings.trigger_mode] = 'TTL Veto Only'
            # self.stage_sigs[self.settings.num_images] = total_points
        else:
            logger.debug('Setting up internal triggering')
            # self.settings.trigger_mode.put('Internal')
            # self.settings.num_images.put(1)
            # self.stage_sigs[self.settings.collection_mode] = 0  # MCA Spectra
            # self.stage_sigs[self.settings.preset_mode] = 1  # Real Time
#            self.stage_sigs[self.settings.trigger_mode] = 'Internal'
#            self.stage_sigs[self.settings.num_images] = spec_per_point

        self.stage_sigs[self.auto_save] = 'No'
        logger.debug('Configuring other filestore stuff')

        logger.debug('Making the filename')
        filename, read_path, write_path = self.make_filename()
        # print(filename, read_path, write_path)

        logger.debug('Setting up hdf5 plugin: ioc path: %s filename: %s',
                     write_path, filename)

        logger.debug('Erasing old spectra')
        self.settings.erase.put(1, wait=True)

        # this must be set after self.settings.num_images because at the Epics
        # layer  there is a helpful link that sets this equal to that (but
        # not the other way)
        self.stage_sigs[self.num_capture] = total_points

        # actually apply the stage_sigs

        # print("Preparing to stage parent")
        ret = super().stage()
        # print(ret)
        # Re-setting hdf5 in order to save proper data
        self.enable.put(0, wait=True)
        ttime.sleep(0.5)
        self.enable.put(1, wait=True)

        self._fn = self.file_template.get() % (self._fp,
                                               self.file_name.get(),
                                               self.file_number.get())
        # print(self._fn)
        if not self.file_path_exists.get():
            raise IOError("Path {} does not exits on IOC!! Please Check"
                          .format(self.file_path.get()))

        logger.debug('Inserting the filestore resource: %s', self._fn)
        # print("Generating resource")
        self._generate_resource({})
        self._filestore_res = self._asset_docs_cache[-1][-1]

        # print(self._filestore_res)

        # this gets auto turned off at the end
        self.capture.put(1)

        # Xspress3 needs a bit of time to configure itself...
        # this does not play nice with the event loop :/
        # time.sleep(self._config_time)

        return ret

    def configure(self, total_points=0, master=None, external_trig=False,
                  **kwargs):
        raise NotImplementedError()

    def describe(self):
        # should this use a better value?
        size = (self.width.get(), )

        spec_desc = {'external': 'FILESTORE:',
                     'dtype': 'array',
                     'shape': size,
                     'source': 'FileStore:'
                     }

        desc = OrderedDict()
        for chan in self.channels:
            key = self.mds_keys[chan]
            desc[key] = spec_desc

        return desc


class XIAXMAPDetectorSettings(CamBase):
    '''XIA XMAP'''

    def __init__(self, prefix, *, read_attrs=None, configuration_attrs=None,
                 **kwargs):
        self.num_channels = 32  # TODO: HARDCODED
        if read_attrs is None:
            read_attrs = []
        if configuration_attrs is None:
            configuration_attrs = [
                # 'config_path', 'config_save_path',
                                   ]
        super().__init__(prefix, read_attrs=read_attrs,
                         configuration_attrs=configuration_attrs, **kwargs)

    array_counter = None
    array_rate = None
    asyn_io = None

    nd_attributes_file = None
    pool_alloc_buffers = None
    pool_free_buffers = None
    pool_max_buffers = None
    pool_max_mem = None
    pool_used_buffers = None
    pool_used_mem = None
    port_name = None

    # Cam-specific
    # acquire = ADCpt(SignalWithRBV, "Acquire")
    # acquire_period = ADCpt(SignalWithRBV, "AcquirePeriod")
    # acquire_time = ADCpt(SignalWithRBV, "AcquireTime")

    array_callbacks = None
    array_size = None

    array_size_bytes = None
    bin_x = None
    bin_y = None
    color_mode = None
    data_type = None
    detector_state = None
    frame_type = None
    gain = None

    image_mode = None
    manufacturer = None

    max_size = None

    min_x = None
    min_y = None
    model = None

    num_exposures = None
    num_exposures_counter = None
    num_images = None
    num_images_counter = None

    read_status = None
    reverse = None

    shutter_close_delay = None
    shutter_close_epics = None
    shutter_control = None
    shutter_control_epics = None
    shutter_fanout = None
    shutter_mode = None
    shutter_open_delay = None
    shutter_open_epics = None
    shutter_status_epics = None
    shutter_status = None

    size = None

    status_message = None
    string_from_server = None
    string_to_server = None
    temperature = None
    temperature_actual = None
    time_remaining = None
    # trigger_mode = ADCpt(SignalWithRBV, "TriggerMode")

    start = Cpt(EpicsSignal,'EraseStart')
    acquire = Cpt(EpicsSignal,'EraseStart')
    erase = Cpt(EpicsSignal,'EraseAll')
    stop_all = Cpt(EpicsSignal,'StopAll')
    acquiring = Cpt(EpicsSignalRO,'Acquiring')
    preset_mode =  Cpt(EpicsSignal,'PresetMode')
    real_time = Cpt(EpicsSignal,'PresetReal')
    actual_time = Cpt(EpicsSignal,'ElapsedReal')
    acquire_time = Cpt(EpicsSignal,'PresetReal')
    acquire_period = Cpt(EpicsSignal,'PresetReal')
    # MCA Spectra=0, MCA Mapping=1, SCA Mapping=2, List Mapping=3
    collection_mode = Cpt(EpicsSignal,'CollectMode')
    num_images = Cpt(EpicsSignal, 'PixelsPerRun')
    trigger_mode = Cpt(EpicsSignal, 'PixelAdvanceMode')
#    xsp_name = Cpt(EpicsSignal, 'NAME')
#    trigger_signal = Cpt(EpicsSignal, 'TRIGGER')
    copy_ROI_SCA = Cpt(EpicsSignal, 'CopyROI_SCA')


class XmapMCA(Device):
    # DO WE REALLY NEED THEM ALL HINTED?
    val = Cpt(EpicsSignal, ".VAL", kind=Kind.hinted)
    R0low = Cpt(EpicsSignal, ".R0LO", kind=Kind.hinted)
    R0high = Cpt(EpicsSignal, ".R0HI", kind=Kind.hinted)
    R0 = Cpt(EpicsSignal, ".R0", kind=Kind.hinted)
    R0nm = Cpt(EpicsSignal, ".R0NM", kind=Kind.hinted)

    R1low = Cpt(EpicsSignal, ".R1LO", kind=Kind.hinted)
    R1high = Cpt(EpicsSignal, ".R1HI", kind=Kind.hinted)
    R1 = Cpt(EpicsSignal, ".R1", kind=Kind.hinted)
    R1nm = Cpt(EpicsSignal, ".R1NM", kind=Kind.hinted)

    R2low = Cpt(EpicsSignal, ".R2LO", kind=Kind.hinted)
    R2high = Cpt(EpicsSignal, ".R2HI", kind=Kind.hinted)
    R2 = Cpt(EpicsSignal, ".R2", kind=Kind.hinted)
    R2nm = Cpt(EpicsSignal, ".R2NM", kind=Kind.hinted)

    R3low = Cpt(EpicsSignal, ".R3LO", kind=Kind.hinted)
    R3high = Cpt(EpicsSignal, ".R3HI", kind=Kind.hinted)
    R3 = Cpt(EpicsSignal, ".R3", kind=Kind.hinted)
    R3nm = Cpt(EpicsSignal, ".R3NM", kind=Kind.hinted)        

class XmapSCA(Device):
    sca0counts = Cpt(EpicsSignal, ":SCA0Counts", kind=Kind.hinted)
    sca1counts = Cpt(EpicsSignal, ":SCA1Counts", kind=Kind.hinted)
    sca2counts = Cpt(EpicsSignal, ":SCA2Counts", kind=Kind.hinted)
    sca3counts = Cpt(EpicsSignal, ":SCA3Counts", kind=Kind.hinted)

def make_scas(channels):
    out_dict = OrderedDict()
    for channel in channels:  # [int]
        attr = f'dxp{channel:1d}'
        out_dict[attr] = (XmapSCA, attr, dict())
        # attr = f"preamp{channel:1d}_gain"
        # out_dict[attr] = (EpicsSignal, f"dxp{channel:1d}.PreampGain", dict())
    return out_dict

def make_channels(channels):
    out_dict = OrderedDict()
    for channel in channels:  # [int]
        attr = f'mca{channel:1d}'
        out_dict[attr] = (XmapMCA, attr, dict())
        # attr = f"preamp{channel:1d}_gain"
        # out_dict[attr] = (EpicsSignal, f"dxp{channel:1d}.PreampGain", dict())
    return out_dict


class XIAXMAPDetector(DetectorBase):
    settings = Cpt(XIAXMAPDetectorSettings, '')

    _channels = DDC(make_channels(range(1, 33)), kind=Kind.hinted)

    scas = DDC(make_scas(range(1, 33)), kind=Kind.hinted)

    external_trig = Cpt(Signal, value=False,
                        doc='Use external triggering')
    total_points = Cpt(Signal, value=-1,
                       doc='The total number of points to acquire overall')
    make_directories = Cpt(Signal, value=False,
                           doc='Make directories on the DAQ side')
    rewindable = Cpt(Signal, value=False,
                     doc='XIA XMAP cannot safely be rewound in bluesky')  # WTF

    data_key = XRF_DATA_KEY

    def __init__(self, prefix, *, read_attrs=None, configuration_attrs=None,
                 name=None, parent=None,
                 # to remove?
                 file_path='', ioc_file_path='', default_channels=None,
                 channel_prefix=None,
                 roi_sums=False,
                 # to remove?
                 **kwargs):

#        if read_attrs is None:
#            read_attrs = ['channel1', ]

        if configuration_attrs is None:
            configuration_attrs = ['settings']  # Do we need channel1.rois?
#            configuration_attrs = ['channel1.rois', 'settings']

        super().__init__(prefix, read_attrs=read_attrs,
                         configuration_attrs=configuration_attrs,
                         name=name, parent=parent, **kwargs)

        # get all sub-device instances
#        sub_devices = {attr: getattr(self, attr)
#                       for attr in self._sub_devices}

        # filter those sub-devices, just giving channels
#        channels = {dev.channel_num: dev
#                    for attr, dev in sub_devices.items()
#                    if isinstance(dev, Xspress3Channel)
#                    }
#        
        

        # make an ordered dictionary with the channels in order
#        self._channelsDict = OrderedDict(sorted(channels.items()))
        self._channelsDict = {chn: getattr(self._channels, f"mca{chn:1d}") for chn in range(1, 33)}

    @property
    def channelsDict(self):
        return self._channelsDict

    @property
    def all_rois(self):
        for ch_num, channel in self._channels.items():
            yield channel
            # for roi in channel.all_rois:
            #     yield roi

    @property
    def enabled_rois(self):
        for roi in self.all_rois:
            # if roi.enable.get():
            yield roi

    def read_hdf5(self, fn, *, rois=None, max_retries=2):  # TODO: ADAPT FOR XIA XMAP
        pass
        '''Read ROI data from an HDF5 file using the current ROI configuration

        Parameters
        ----------
        fn : str
            HDF5 filename to load
        rois : sequence of Xspress3ROI instances, optional

        '''
#        if rois is None:
#            rois = self.enabled_rois

        num_points = self.settings.num_images.get()
        if isinstance(fn, h5py.File):
            hdf = fn
        else:
            hdf = h5py.File(fn, 'r')

        RoiTuple = Xspress3ROI.get_device_tuple()

        handler = Xspress3HDF5Handler(hdf, key=self.data_key)
        for roi in self.enabled_rois:
            roi_data = handler.get_roi(chan=roi.channel_num,
                                       bin_low=roi.bin_low.get(),
                                       bin_high=roi.bin_high.get(),
                                       max_points=num_points)

            roi_info = RoiTuple(bin_low=roi.bin_low.get(),
                                bin_high=roi.bin_high.get(),
                                ev_low=roi.ev_low.get(),
                                ev_high=roi.ev_high.get(),
                                value=roi_data,
                                value_sum=None,
                                enable=None)

            yield roi.name, roi_info

class XIAXMAPTrigger(BlueskyInterface):  # See existing implementation
    """Base class for trigger mixin classes

    Subclasses must define a method with this signature:

    `acquire_changed(self, value=None, old_value=None, **kwargs)`
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._status = None
        self._acquisition_start = self.settings.start
        self._acquisition_signal = self.settings.acquiring
        self._abs_trigger_count = 0

    def stage(self):
        self._abs_trigger_count = 0
        self._acquisition_signal.subscribe(self._acquire_changed, run=False)
        staging = super().stage()
        return staging

    def unstage(self):
        ret = super().unstage()
        self._acquisition_signal.clear_sub(self._acquire_changed)
        self._status = None
        return ret

    def _acquire_changed(self, value=None, old_value=None, **kwargs):
        # "This is called when the 'acquire' signal changes."
        if self._status is None:
            return

        if (old_value == 1) and (value == 0):
            # Negative-going edge means an acquisition just finished.
            self._status._finished()

    def trigger(self):
        if self._staged != Staged.yes:
            raise RuntimeError("not staged")

        self._status = DeviceStatus(self)
        if self._status is None:
            self._status._finished()
            return
        self._acquisition_start.put(1, wait=False)
        trigger_time = ttime.time()

        self._abs_trigger_count += 1
        return self._status


class XIAXMAPFileStoreFlyable(XIAXMAPFileStore):
    def warmup(self):

        def run_mode(mode):
            self.parent.settings.collection_mode.put(mode, wait=True)
            ttime.sleep(0.5)
            self.parent.settings.acquire_time.put(1, wait=True)
            ttime.sleep(0.5)
            self.parent.settings.stop_all.put(1, wait=True)
            ttime.sleep(0.5)
            self.parent.settings.acquire.put(1, wait=False)
            ttime.sleep(2)  # wait for acquisition
            self.parent.settings.stop_all.put(1, wait=True)
        """
        A convenience method for 'priming' the plugin.
        The plugin has to 'see' one acquisition before it is ready to capture.
        This sets the array size, etc.
        NOTE : this comes from:
            https://github.com/NSLS-II/ophyd/blob/master/ophyd/areadetector/plugins.py
        We had to replace "cam" with "settings" here.
        Also modified the stage sigs.
        """
        print_to_gui(f'XIA HDF warmup starting...', add_timestamp=False)
        self.enable.set(1).wait()
        sigs = OrderedDict([  # (self.parent.settings.array_callbacks, 1),
                            (self.parent.settings.collection_mode, 2),
                            # just in case the acquisition time is set very long...
                            (self.parent.settings.acquire_time, 1)])  #,

        original_vals = {sig: sig.get() for sig in sigs}
        #print(original_vals)

        # Remove the hdf5.capture item here to avoid an error as it should reset back to 0 itself
        # del original_vals[self.capture]
        ttime.sleep(1)
        run_mode(2)
        ttime.sleep(1)
        # code below checks if warmup was performed correctly and re-runs the extended procedure if not
        if int(self.array_size.width.get()) != 24064:
            print("Bad warmup, repeating...")
            print("Warming up in MCA Mapping mode")
            run_mode(1)
            ttime.sleep(1)
            print("Now warming up in SCA Mapping mode")
            run_mode(2)


        for sig, val in reversed(list(original_vals.items())):
            ttime.sleep(0.1)
            sig.put(val, wait=True)

        print_to_gui(f'XIA XMAP warmup complete...', add_timestamp=False)

ROOT_PATH_WIN = "J:\\legacy\\Sandbox\\epics"

class ISSXIAXMAPDetector(XIAXMAPTrigger, XIAXMAPDetector):  # For step scans

    hdf5 = Cpt(XIAXMAPFileStoreFlyable, 'HDF1:',
               read_path_template='/nsls2/data/iss/legacy/Sandbox/epics/raw/dxp/%Y/%m/%d/',
               root='/nsls2/data/iss/legacy/Sandbox/epics/raw/dxp',
               write_path_template=f'{ROOT_PATH_WIN}\\{RAW_PATH}\\dxp\\%Y\\%m\\%d\\',
               )


    def __init__(self, prefix, *, configuration_attrs=None, read_attrs=None, **kwargs):
        if configuration_attrs is None:
            configuration_attrs = [
                                   'external_trig',
                                   'total_points',
#                                   'spectra_per_point',
                                   'settings',
                                   'rewindable'
                                   ]
        if read_attrs is None:
# #            read_attrs = ['channel1', 'channel2', 'channel3', 'channel4', 'hdf5', 'settings.acquire_time']
            read_attrs = ['settings.acquire_time', 'settings.actual_time']
#            read_attrs.extend([f"scas.dxp{n:1d}.counts" for n in range(1, 33)])
#             read_attrs.extend([f"_channels.mca{n:1d}.R0" for n in range(1, 33)])
#             # read_attrs = ['hdf5'] #, 'settings.acquire_time']


        super().__init__(prefix, configuration_attrs=configuration_attrs,
                         read_attrs=read_attrs, **kwargs)
        # self.set_channels_for_hdf5()
#        self.spectra_per_point.put(1)

        self._asset_docs_cache = deque()
        # self._datum_counter = None
        self.warmup()

    # Step-scan interface methods.
    def stage(self):
        staged_list = super().stage()
        return staged_list

    def unstage(self):
    
        return super().unstage()

    def set_exposure_time(self, new_exp_time):
        self.settings.acquire_time.set(new_exp_time).wait()

    def read_exposure_time(self):
        return self.settings.acquire_time.get()

    def test_exposure(self, acq_time=1, num_images=1):
        # THIS MUST WORK WITH STEP MODE
        _old_acquire_time = self.settings.acquire_time.value
#        _old_num_images = self.settings.num_images.value
        # self.settings.acquire_time.set(acq_time).wait()
        self.set_exposure_time(acq_time)
#        self.settings.num_images.set(num_images).wait()
        self.settings.erase.put(1)
        self._acquisition_signal.put(1, wait=True)
        # self.settings.acquire_time.set(_old_acquire_time).wait()
        self.set_exposure_time(_old_acquire_time)
#        self.settings.num_images.set(_old_num_images).wait()

    def set_channels_for_hdf5(self, channels=list(range(1, 33))):
        """
        Configure which channels' data should be saved in the resulted hdf5 file.
        Parameters
        ----------
        channels: tuple, optional
            the channels to save the data for
        """
        # The number of channel
#        for n in channels:
#            getattr(self, f'channel{n}').rois.read_attrs = ['roi{:02}'.format(j) for j in [1, 2, 3, 4]]

        for n in channels:
            getattr(self._channels, f'mca{n:1d}').read_attrs = ['R0']   # ['roi{:02}'.format(j) for j in [1, 2, 3, 4]]


        self.hdf5.num_extra_dims.put(0)
        # self.settings.num_channels.put(len(channels))  # TODO: HARDCODED

    def warmup(self, hdf5_warmup=False):
#        self.channel1.vis_enabled.put(1)
#        self.channel2.vis_enabled.put(1)
#        self.channel3.vis_enabled.put(1)
#        self.channel4.vis_enabled.put(1)
        self.total_points.put(1)
        emergency_warmup = False


        if int(self.hdf5.array_size.width.get()) in [0, 1048000]:
            print_to_gui(f'XIA HDF plugin warmup required...', add_timestamp=False)
            emergency_warmup = True

        if hdf5_warmup or emergency_warmup:
            self.hdf5.warmup()
        self.settings.stop_all.put(1)


        if self.settings.acquire_time.get() == 0:
            self.settings.acquire_time.put(1)

        if self.settings.collection_mode.get() == 0:
            self.settings.preset_mode.put(1)

        # Hints:
        # for n in range(1, 5):
        #     getattr(self, f'channel{n}').rois.roi01.value.kind = 'hinted'
        for n in range(1, 33):
            mca_ch = getattr(self._channels, f"mca{n:1d}")
            mca_ch.R0.kind = 'hinted'
            mca_ch.R1.kind = 'hinted'
            mca_ch.R2.kind = 'hinted'
            mca_ch.R3.kind = 'hinted'
            # getattr(self.scas, f"dxp{n:1d}").counts.kind = 'hinted'
            # print(getattr(self._channels, f"mca{n:1d}").R0.kind)

        self.settings.configuration_attrs = [
                                           'acquire_time',
                                           'num_images',
                                           'trigger_mode',
                                           ]

        if int(self._channels.mca1.R0low.get()) < 0 or int(self._channels.mca1.R0high.get()) < 0:
            print_to_gui(f'XIA Initializing MCA/SCA...', add_timestamp=False)
            self.set_limits_for_roi(5000, window='max')

        ## THIS IS TO CONFIGURE DATASOURCE KIND
        # for key, channel in self.channelsDict.items():
        #     roi_names = ['roi{:02}'.format(j) for j in [1, 2, 3, 4]]
        #     channel.rois.read_attrs = roi_names
        #     channel.rois.configuration_attrs = roi_names
        #     for roi_n in roi_names:
        #         getattr(channel.rois, roi_n).value_sum.kind = 'omitted'

    def set_limits_for_roi(self, energy_nom, roi=0, window='auto'):

        for ch_index in range(1, self.settings.num_channels+1):
            if window == 'auto':
                print("USING HARDCODED WINDOW OF 250EV AROUND THE PEAK FOR CHANNEL", ch_index)
                energy = energy_nom
                w = 125
            elif window == 'max':
                energy = 5001
                w = 10000
            elif isinstance(window, (int, float)):
                w = window
            #     w = _compute_window_for_xs_roi_energy(energy_nom)
            # else:
            #     w = int(window)
            # energy = _convert_xs_energy_nom2act(energy_nom, ch_index)
            ev_low_new = int((energy - w / 2) / 5)  # TODO: divide by bin size?
            ev_high_new = int((energy + w / 2) / 5)

#            roi_obj = getattr(channel.rois, roi)
#            roi_obj = getattr(channel, )
            channel = getattr(self._channels, f"mca{ch_index:1d}")
            roi_high = getattr(channel, f"R{roi:1d}high")
            roi_low = getattr(channel, f"R{roi:1d}low")

            if ev_high_new < roi_low.get():
                roi_low.put(ev_low_new)
                roi_high.put(ev_high_new)
            else:
                roi_high.put(ev_high_new)
                roi_low.put(ev_low_new)
        # THIS FUNCTION WILL COPY ROI TO SCA FOR ALL SCAs ON ALL CHANNELS: SCA0 to SCA16
        self.settings.copy_ROI_SCA.put(1, wait=True)

    def ensure_roi4_covering_total_mca(self, emin=600, emax=40960):  # TODO: Needs adjustment for XMAP DXP
        for channel in self.channelsDict.items():
            channel.R0high.put(emax)
            channel.R0low.put(emin)

    @property
    def roi_metadata(self):
        md = {}
        for ch_index, channel in self.channelsDict.items():
            v = {}

            # roi_idx = 0
            for roi_idx in range(4):
                roi_str = f'roi{roi_idx:1d}'
                roi_high = getattr(channel, f"R{roi_idx:1d}high")
                roi_low = getattr(channel, f"R{roi_idx:1d}low")
                
                v[roi_str] = [roi_low.get(), roi_high.get()]
            md[f"ch{ch_index:02d}"] = v
        return md

    def read_config_metadata(self):
        md = {'device_name': self.name,
              'roi': self.roi_metadata}
        return md


# def compose_bulk_datum_xs(*, resource_uid, counter, datum_kwargs, validate=True):
#     # print_message_now(datum_kwargs)
#     # any_column, *_ = datum_kwargs
#     # print_message_now(any_column)
#     N = len(datum_kwargs)
#     # print_message_now(N)
#     doc = {'resource': resource_uid,
#            'datum_ids': ['{}/{}'.format(resource_uid, next(counter)) for _ in range(N)],
#            'datum_kwarg_list': datum_kwargs}
#     # if validate:
#     #     schema_validators[DocumentNames.bulk_datum].validate(doc)
#     return doc

class ISSXIAXMAPDetectorStream(ISSXIAXMAPDetector):

    def __init__(self, *args, ext_trigger_device=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.ext_trigger_device = ext_trigger_device
        self._datum_counter = None
        self._infer_datum_keys()

    def _infer_datum_keys(self):
        self.datum_keys = []
        for i in range(1, 33):
            for j in range(4):
                # self.datum_keys.append({"name": f"{self.name}",
                #                         "channel": i + 1,
                #                         "type": "spectrum"})
                # self.datum_keys.append({"name": f"{self.name}",
                #                         "channel": i + 1,
                #                         "type": "roi"})
                self.datum_keys.append({"data_type": "roi",
                                        "channel": i,
                                        "roi_num": j})

    def format_datum_key(self, input_dict):
        output = f'ge_detector_channels_mca{input_dict["channel"]:1d}_R{input_dict["roi_num"]:1d}'
#        if input_dict["data_type"] == 'roi':
#            output += f'{input_dict["roi_num"]:02d}'
        return output

    def prepare_to_fly(self, traj_duration):
        acq_rate = self.ext_trigger_device.freq.get()
        self.num_points = int(acq_rate * (traj_duration + 2))
        self.ext_trigger_device.prepare_to_fly(traj_duration)

    def stage(self):
        self._infer_datum_keys()
        self._datum_counter = itertools.count()
        self.total_points.put(self.num_points)
        # self.hdf5.file_write_mode.put(2)
        self.external_trig.put(True)
        # self.settings.trigger_mode.put(3)
        current_mode = self.settings.collection_mode.get()
        if current_mode != 2:
            self.settings.collection_mode.put(2, wait=True)  # SCA Mapping
        # self.settings.trigger_mode.put(0, wait=True)  # Gate
        # Re-setting hdf5 in order to save proper data
        self.hdf5.enable.put(0, wait=True)
        ttime.sleep(1.)
        self.hdf5.enable.put(1, wait=True)
        # self.hdf5.warmup()  # TODO: TEST the array size
        self.hdf5.file_write_mode.put(1)  # Capture
        # self.hdf5.file_write_mode.put(2)  # Stream
        self.settings.num_images.put(self.num_points, wait=True)
        self.hdf5.num_capture.set(self.num_points)
        # self.hdf5.capture.set(1)

        staged_list = super().stage()
        staged_list += self.ext_trigger_device.stage()
        print("Moving XIA modules synchronization to staging...")
        self.settings.acquire.put(1, wait=False)
        print("Waiting for sync: 10s")
        ttime.sleep(10)

        return staged_list

    def unstage(self):
        unstaged_list = super().unstage()
        self._datum_counter = None
        self.hdf5.file_write_mode.put(0)
        self.external_trig.put(False)
        # self.settings.trigger_mode.put(1)
        # self.total_points.put(1)
        self.settings.stop_all.put(1, wait=True)  # STOP
        unstaged_list += self.ext_trigger_device.unstage()
        return unstaged_list


    def kickoff(self):
        # self.settings.acquire.put(1, wait=False)
        # print_to_gui(f'Waiting for XIA modules to start...', add_timestamp=True)
        # ttime.sleep(10)  # THIS IS MOVED TO STAGING TO PREVENT FLYER FROM FREEZING
        # ttime.sleep()
        # print_to_gui(f'Starting GATE signal...', add_timestamp=True)
        return self.ext_trigger_device.kickoff()

    def complete(self):
        print_to_gui(f'XIA XMAP complete is starting...', add_timestamp=True)

        acquire_status = self.settings.stop_all.put(1, wait=True)  # STOP
        capture_status = self.hdf5.capture.put(0, wait=True)      # STOP
        self.hdf5.write_file.put(1)  # ONLY REQUIRED FOR Capture FILE_WRITE mode
        # (acquire_status and capture_status).wait()


        ext_trigger_status = self.ext_trigger_device.complete()
        for resource in self.hdf5._asset_docs_cache:
            self._asset_docs_cache.append(('resource', resource[1]))

        _resource_uid = self.hdf5._resource_uid
        self._datum_ids = {}

        for datum_key_dict in self.datum_keys:
            # print(datum_key_dict)
            datum_key = self.format_datum_key(datum_key_dict)
            datum_id = f'{_resource_uid}/{datum_key}'
            self._datum_ids[datum_key] = datum_id
            doc = {'resource': _resource_uid,
                   'datum_id': datum_id,
                   'datum_kwargs': datum_key_dict}
            # print(doc)
            self._asset_docs_cache.append(('datum', doc))

        print_to_gui(f'XIA XMAP complete is done.', add_timestamp=True)
        complete_status = NullStatus() and ext_trigger_status
        print(f"{complete_status=}")
        return complete_status

    def collect(self):
        print_to_gui(f'XIA XMAP collect is starting...', add_timestamp=True)
        ts = ttime.time()
        yield {'data': self._datum_ids,
               'timestamps': {self.format_datum_key(key_dict): ts for key_dict in self.datum_keys},
               'time': ts,  # TODO: use the proper timestamps from the mono start and stop times
               'filled': {self.format_datum_key(key_dict): False for key_dict in self.datum_keys}}
        print_to_gui(f'XIA XMAP collect is done.', add_timestamp=True)
        yield from self.ext_trigger_device.collect()

    def describe_collect(self):  # TODO: NEEDS TESTING
        xia_spectra_dicts = {}
        for datum_key_dict in self.datum_keys:
            datum_key = self.format_datum_key(datum_key_dict)
            if datum_key_dict['data_type'] == 'spectrum':
                value = {'source': 'Ge detector',
                         'dtype': 'array',
                         'shape': [self.settings.num_images.get(),
                                   self.hdf5.array_size.width.get()],
                         'dims': ['frames', 'row'],
                         'external': 'FILESTORE:'}
            elif datum_key_dict['data_type'] == 'roi':
                value = {'source': 'Ge detector',
                         'dtype': 'array',
                         'shape': [self.settings.num_images.get()],
                         'dims': ['frames'],
                         'external': 'FILESTORE:'}
            else:
                raise KeyError(f'data_type={datum_key_dict["data_type"]} not supported')
            xia_spectra_dicts[datum_key] = value
        # print(xia_spectra_dicts)

        return_dict_xs = {self.name : xia_spectra_dicts}

        return_dict_trig = self.ext_trigger_device.describe_collect()
        return {**return_dict_xs, **return_dict_trig}

    def collect_asset_docs(self):
        items = list(self._asset_docs_cache)
        # print_to_gui(f"items = {items}", tag='XS DEBUG')
        self._asset_docs_cache.clear()
        for item in items:
            yield item
        yield from self.ext_trigger_device.collect_asset_docs()

    def read_config_metadata(self):
        md = super().read_config_metadata()
        freq = self.ext_trigger_device.freq.get()
        dc = self.ext_trigger_device.duty_cycle.get()
        md['frame_rate'] = freq
        md['duty_cycle'] = dc
        md['acquire_time'] = 1/freq
        md['exposure_time'] = 1/freq * dc/100
        return md
    

ge_detector = ISSXIAXMAPDetector('XF:08IDB-ES{GE-Det:1}', name='ge_detector')
ge_detector.scas.kind = Kind.hinted
ge_detector._channels.kind = Kind.hinted
# ttime.sleep(2)
ge_detector_stream = ISSXIAXMAPDetectorStream('XF:08IDB-ES{GE-Det:1}', name="ge_detector_stream", ext_trigger_device=apb_trigger_ge_detector)    