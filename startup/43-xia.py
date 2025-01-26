print(ttime.ctime() + ' >>>> ' + __file__)

#from nslsii.detectors.xspress3 import (XspressTrigger, Xspress3Detector,
#                                       Xspress3Channel, Xspress3FileStore, Xspress3ROI, logger)
from ophyd.areadetector.filestore_mixins import FileStorePluginBase
from ophyd.device import (
    Device,
    Component as Cpt,
    DynamicDeviceComponent as DDC,
    BlueskyInterface,
    FormattedComponent as FC)

from ophyd.areadetector import DetectorBase, CamBase
import bluesky.plans as bp
import bluesky.plan_stubs as bps
# bp.list_scan
import numpy as np
import itertools
import time as ttime
from collections import deque, OrderedDict
from itertools import product
import pandas as pd
import warnings

class XIATrigger(BlueskyInterface):
    """Base class for trigger mixin classes

    Subclasses must define a method with this signature:

    `acquire_changed(self, value=None, old_value=None, **kwargs)`
    """
    # TODO **
    # count_time = self.settings.acquire_period

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # settings
        self._status = None
        self._acquisition_signal = self.settings.acquire
        self._abs_trigger_count = 0

    # def stage(self):
    #     self._abs_trigger_count = 0
    #     self._acquisition_signal.subscribe(self._acquire_changed)
    #     return super().stage()
    #
    # def unstage(self):
    #     ret = super().unstage()
    #     self._acquisition_signal.clear_sub(self._acquire_changed)
    #     self._status = None
    #     return ret
    #
    # def _acquire_changed(self, value=None, old_value=None, **kwargs):
    #     "This is called when the 'acquire' signal changes."
    #     if self._status is None:
    #         return
    #     if (old_value == 1) and (value == 0):
    #         # Negative-going edge means an acquisition just finished.
    #         self._status._finished()

    # def trigger(self):
    #     if self._staged != Staged.yes:
    #         raise RuntimeError("not staged")
    #
    #     self._status = DeviceStatus(self)
    #     self._acquisition_signal.put(1, wait=False)
    #     trigger_time = ttime.time()
    #
    #     # for sn in self.read_attrs:
    #     #     if sn.startswith('channel') and '.' not in sn:
    #     #         ch = getattr(self, sn)
    #     #         self.dispatch(ch.name, trigger_time)
    #
    #     self._abs_trigger_count += 1
    #     return self._status

    def trigger(self):
        return self.get_mca()

    def get_mca(self):
        def is_done(value, old_value, **kwargs):
            if old_value == 1 and value ==0:
                return True
            return False

        status = SubscriptionStatus(self.settings.acquiring, run=False, callback=is_done)
        self._acquisition_signal.put(1, wait=False)
        self.settings.start.put(1)
        return status

class XIAXMAPFileStore(FileStorePluginBase, HDF5Plugin):
    '''Xspress3 acquisition -> filestore'''
    # num_capture_calc = Cpt(EpicsSignal, 'NumCapture_CALC')
    # num_capture_calc_disable = Cpt(EpicsSignal, 'NumCapture_CALC.DISA')
    # filestore_spec = Xspress3HDF5Handler.HANDLER_NAME

    def __init__(self, basename, *, config_time=0.5,
                 mds_key_format='{self.settings.name}_mca{chan}', parent=None,
                 **kwargs):
        super().__init__(basename, parent=parent, **kwargs)
        det = parent
        self.settings = det.settings

        # Use the EpicsSignal file_template from the detector
        self.stage_sigs[self.blocking_callbacks] = 1
        self.stage_sigs[self.enable] = 1
        self.stage_sigs[self.compression] = 'zlib'
        self.stage_sigs[self.file_template] = '%s%s_%6.6d.h5'
        # print(self.stage_sigs)

        self._filestore_res = None
        self.channels = list(range(1, len([_ for _ in det.channels.component_names
                                           if _.startswith('mca')]) + 1))
        # this was in original code, but I kinda-sorta nuked because
        # it was not needed for SRX and I could not guess what it did
        # self._master = None
        #
        # self._config_time = config_time
        self.mds_keys = {chan: mds_key_format.format(self=self, chan=chan)
                         for chan in self.channels}
        # print(self.mds_keys)

    def stop(self, success=False):
        ret = super().stop(success=success)
        self.capture.put(0)
        return ret

    def kickoff(self):
        # TODO
        raise NotImplementedError()

    def collect(self):
        # TODO (hxn-specific implementation elsewhere)
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
            while self.capture.get() == 1:
                i += 1
                if (i % 50) == 0:
                    logger.warning('Still capturing data .... waiting.')
                time.sleep(0.1)
                if i > 150:
                    logger.warning('Still capturing data .... giving up.')
                    logger.warning('Check that the xspress3 is configured to take the right '
                                   'number of frames '
                                   f'(it is trying to take {self.parent.settings.num_images.get()})')
                    self.capture.put(0)
                    break

        except KeyboardInterrupt:
            self.capture.put(0)
            logger.warning('Still capturing data .... interrupted.')

        return super().unstage()

    def generate_datum(self, key, timestamp, datum_kwargs):
        sn, n = next((f'channel{j}', j)
                     for j in self.channels
                     if getattr(self.parent, f'channels.mca{j}').name == key)
        datum_kwargs.update({'frame': self.parent._abs_trigger_count,
                             'channel': int(sn[7:])})
        self.mds_keys[n] = key
        super().generate_datum(key, timestamp, datum_kwargs)

    def stage(self):
        # if should external trigger
        ext_trig = self.parent.external_trig.get()

        logger.debug('Stopping XIA acquisition')
        # really force it to stop acquiring
        self.settings.erase.put(0, wait=True)

        total_points = self.parent.settings.total_points.get()
        if total_points < 1:
            raise RuntimeError("You must set the total points")
        # spec_per_point = self.parent.spectra_per_point.get()
        # total_capture = total_points * spec_per_point

        # stop previous acquisition
        self.stage_sigs[self.settings.acquire] = 0

        # re-order the stage signals and disable the calc record which is
        # interfering with the capture count
        self.stage_sigs.pop(self.num_capture, None)
        self.stage_sigs.pop(self.num_images, None)
        self.stage_sigs[self.num_capture_calc_disable] = 1

        if ext_trig:
            logger.debug('Setting up external triggering')
            self.stage_sigs[self.settings.trigger_mode] = 0
            # self.stage_sigs[self.settings.num_images] = total_points
        else:
            logger.debug('Setting up internal triggering')
            # TODO:
            # self.settings.trigger_mode.put('Internal')
            # self.settings.num_images.put(1)
            # self.stage_sigs[self.settings.trigger_mode] = 'Internal'
            # self.stage_sigs[self.settings.num_images] = spec_per_point

        self.stage_sigs[self.auto_save] = 'No'
        logger.debug('Configuring other filestore stuff')

        logger.debug('Making the filename')
        filename, read_path, write_path = self.make_filename()

        logger.debug('Setting up hdf5 plugin: ioc path: %s filename: %s',
                     write_path, filename)

        logger.debug('Erasing old spectra')
        self.settings.erase.put(1, wait=True)

        # this must be set after self.settings.num_images because at the Epics
        # layer  there is a helpful link that sets this equal to that (but
        # not the other way)
        self.stage_sigs[self.num_capture] = total_points

        # actually apply the stage_sigs
        ret = super().stage()

        self._fn = self.file_template.get() % (self._fp,
                                               self.file_name.get(),
                                               self.file_number.get())

        if not self.file_path_exists.get():
            raise IOError("Path {} does not exits on IOC!! Please Check"
                          .format(self.file_path.get()))

        logger.debug('Inserting the filestore resource: %s', self._fn)
        self._generate_resource({})
        self._filestore_res = self._asset_docs_cache[-1][-1]

        # this gets auto turned off at the end
        self.capture.put(1)

        # Xspress3 needs a bit of time to configure itself...
        # this does not play nice with the event loop :/
        time.sleep(self._config_time)

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

class XIADetectorSettings(CamBase):
   '''XIA XMAP detector'''

   def __init__(self, prefix, *, read_attrs=None, configuration_attrs=None,
                **kwargs):
       if read_attrs is None:
           read_attrs = []
       # if configuration_attrs is None:
       #     configuration_attrs = ['config_path', 'config_save_path',
       #                            ]
       super().__init__(prefix, read_attrs=read_attrs,
                        configuration_attrs=configuration_attrs, **kwargs)

   start = Cpt(EpicsSignal,'EraseStart')
   acquire = Cpt(EpicsSignal,'EraseStart')
   stop_all = Cpt(EpicsSignal,'StopAll')
   acquiring = Cpt(EpicsSignal,'Acquiring')
   preset_mode =  Cpt(EpicsSignal,'PresetMode')
   real_time = Cpt(EpicsSignal,'PresetReal')
   # MCA Spectra=0, MCA Mapping=1, SCA Mapping=2, List Mapping=3
   collection_mode = Cpt(EpicsSignal,'CollectMode')
   acquisition_time=Cpt(EpicsSignal,'PresetReal')
   total_points = Cpt(EpicsSignal, 'PixelsPerRun')
   trigger_mode = Cpt(EpicsSignal, 'PixelAdvanceMode')
   erase = Cpt(EpicsSignal,'EraseAll')
   copy_ROI_SCA = Cpt(EpicsSignal, 'CopyROI_SCA')
   acquire_time = Cpt(EpicsSignal,'PresetReal')
   acquire_period = Cpt(EpicsSignal, 'PresetReal')
   # image_mode = Cpt(EpicsSignal, 'PresetReal')
   # manufacturer = Cpt(EpicsSignal, 'PresetReal')
   # model = Cpt(EpicsSignal, 'PresetReal'
   # = Cpt(EpicsSignal, 'PixelsPerRun')
   # config_path = Cpt(SignalWithRBV, 'CONFIG_PATH', string=True)
   # config_save_path = Cpt(SignalWithRBV, 'CONFIG_SAVE_PATH', string=True)
#    connect = Cpt(EpicsSignal, 'CONNECT')
#    connected = Cpt(EpicsSignal, 'CONNECTED')
#    ctrl_dtc = Cpt(SignalWithRBV, 'CTRL_DTC')
#    ctrl_mca_roi = Cpt(SignalWithRBV, 'CTRL_MCA_ROI')
#    debounce = Cpt(SignalWithRBV, 'DEBOUNCE')
#    disconnect = Cpt(EpicsSignal, 'DISCONNECT')
#    erase = Cpt(EpicsSignal, 'ERASE')
#    # erase_array_counters = Cpt(EpicsSignal, 'ERASE_ArrayCounters')
#    # erase_attr_reset = Cpt(EpicsSignal, 'ERASE_AttrReset')
#    # erase_proc_reset_filter = Cpt(EpicsSignal, 'ERASE_PROC_ResetFilter')
#    frame_count = Cpt(EpicsSignalRO, 'FRAME_COUNT_RBV')
#    invert_f0 = Cpt(SignalWithRBV, 'INVERT_F0')
#    invert_veto = Cpt(SignalWithRBV, 'INVERT_VETO')
#    max_frames = Cpt(EpicsSignalRO, 'MAX_FRAMES_RBV')
#    max_frames_driver = Cpt(EpicsSignalRO, 'MAX_FRAMES_DRIVER_RBV')
#    max_num_channels = Cpt(EpicsSignalRO, 'MAX_NUM_CHANNELS_RBV')
#    max_spectra = Cpt(SignalWithRBV, 'MAX_SPECTRA')
#    xsp_name = Cpt(EpicsSignal, 'NAME')
#    num_cards = Cpt(EpicsSignalRO, 'NUM_CARDS_RBV')
#    num_channels = Cpt(SignalWithRBV, 'NUM_CHANNELS')
#    num_frames_config = Cpt(SignalWithRBV, 'NUM_FRAMES_CONFIG')
#    reset = Cpt(EpicsSignal, 'RESET')
#    restore_settings = Cpt(EpicsSignal, 'RESTORE_SETTINGS')
#    run_flags = Cpt(SignalWithRBV, 'RUN_FLAGS')
#    save_settings = Cpt(EpicsSignal, 'SAVE_SETTINGS')
#    trigger_signal = Cpt(EpicsSignal, 'TRIGGER')
#    # update = Cpt(EpicsSignal, 'UPDATE')
#    # update_attr = Cpt(EpicsSignal, 'UPDATE_AttrUpdate')


class XmapMCA(Device):
    val = Cpt(EpicsSignal, ".VAL", kind=Kind.hinted)
    R0low = Cpt(EpicsSignal, ".R0LO", kind=Kind.hinted)
    R0high = Cpt(EpicsSignal, ".R0HI", kind=Kind.hinted)
    R0 = Cpt(EpicsSignal, ".R0", kind=Kind.hinted)
    R0nm = Cpt(EpicsSignal, ".R0NM", kind=Kind.hinted)


def make_channels(channels):
    out_dict = OrderedDict()
    for channel in channels:  # [int]
        attr = f'mca{channel:1d}'
        out_dict[attr] = (XmapMCA, attr, dict())
        # attr = f"preamp{channel:1d}_gain"
        # out_dict[attr] = (EpicsSignal, f"dxp{channel:1d}.PreampGain", dict())
    return out_dict


class XMAPFileStoreFlyable(XIAXMAPFileStore):
    def warmup(self):
        """
        A convenience method for 'priming' the plugin.
        The plugin has to 'see' one acquisition before it is ready to capture.
        This sets the array size, etc.
        NOTE : this comes from:
            https://github.com/NSLS-II/ophyd/blob/master/ophyd/areadetector/plugins.py
        We had to replace "cam" with "settings" here.
        Also modified the stage sigs.
        """
        print_to_gui("warming up the hdf5 plugin...")
        self.enable.set(1).wait()

        self.parent.settings.collection_mode.put(2)
        while int(self.parent.settings.collection_mode.get()) != 2:
            print_to_gui("...")
            ttime.sleep(1)
        print_to_gui("Acquiring empty frame...")
        self.parent.settings.stop_all.put(1)
        ttime.sleep(0.5)
        self.parent.settings.start.put(1)
        while int(self.parent.settings.acquiring.get()) != 1:
            ttime.sleep(0.1)
        ttime.sleep(0.1)
        self.parent.settings.stop_all.put(1)

        print_to_gui("done")


class GeDetector(XIATrigger, DetectorBase):
    settings = Cpt(XIADetectorSettings, '')
    channels = DDC(make_channels(range(1, 33)))
    # start = Cpt(EpicsSignal,'EraseStart')
    # stop_all = Cpt(EpicsSignal,'StopAll')
    # acquiring = Cpt(EpicsSignal,'Acquiring')
    # preset_mode =  Cpt(EpicsSignal,'PresetMode')
    # real_time = Cpt(EpicsSignal,'PresetReal')
    # # MCA Spectra=0, MCA Mapping=1, SCA Mapping=2, List Mapping=3
    # collection_mode = Cpt(EpicsSignal,'CollectMode')
    # acquisition_time=Cpt(EpicsSignal,'PresetReal')
    # total_points = Cpt(EpicsSignal, 'PixelsPerRun')
    # trigger_mode = Cpt(EpicsSignal, 'PixelAdvanceMode')


    hdf5 = Cpt(XMAPFileStoreFlyable, 'HDF1:',
               read_path_template=f'{ROOT_PATH}/{RAW_PATH}/dxp/%Y/%m/%d/',
               root=f'{ROOT_PATH}/{RAW_PATH}/',
               write_path_template=f'{ROOT_PATH}/{RAW_PATH}/dxp/%Y/%m/%d/',
               )

    def __init__(self, prefix, ext_trigger_device=None, configuration_attrs=None, read_attrs=None, **kwargs):
        # super().__init__(prefix, *args, **kwargs)
        self.num_channels = 32  # TODO: Do we need it configurable?
        self.ext_trigger_device = ext_trigger_device
        if configuration_attrs is None:
            configuration_attrs = ['settings'] #, 'ext_trigger_device']
#            configuration_attrs = ['external_trig', 'total_points',
#                                   'spectra_per_point', 'settings',
#                                   'rewindable']
        if read_attrs is None:
            read_attrs = ['hdf5']
        super().__init__(prefix, configuration_attrs=configuration_attrs,
                         read_attrs=read_attrs, **kwargs)
        # self.set_channels_for_hdf5()  # TODO:
        # self.create_dir.put(-3)
#        self.spectra_per_point.put(1)
#        self.channel1.rois.roi01.configuration_attrs.append('bin_low')

#        self._asset_docs_cache = deque()
        # self._datum_counter = None
        self.warmup(hdf5_warmup=True)


    # def trigger(self):
    #     return self.get_mca()
    #
    # def get_mca(self):
    #     def is_done(value, old_value, **kwargs):
    #         if old_value == 1 and value ==0:
    #             return True
    #         return False
    #
    #     status = SubscriptionStatus(self.settings.acquiring, run=False, callback=is_done)
    #     self.settings.start.put(1)
    #     return status

    def set_limits_for_roi(self, energy_nom, roi, window='auto'):

        for ch_index in range(1, self.num_channels+1):
            if window == 'auto':
                print("USING HARDCODED WINDOW OF 250EV AROUND THE PEAK FOR CHANNEL", ch_index)
                energy = energy_nom
                w = 125
            #     w = _compute_window_for_xs_roi_energy(energy_nom)
            # else:
            #     w = int(window)
            # energy = _convert_xs_energy_nom2act(energy_nom, ch_index)
            ev_low_new = int((energy - w / 2) / 5)  # TODO: divide by bin size?
            ev_high_new = int((energy + w / 2) / 5)

#            roi_obj = getattr(channel.rois, roi)
#            roi_obj = getattr(channel, )
            channel = getattr(self.channels, f"mca{ch_index:1d}")
            if ev_high_new < channel.R0low.get():
                channel.R0low.put(ev_low_new)
                channel.R0high.put(ev_high_new)
            else:
                channel.R0high.put(ev_high_new)
                channel.R0low.put(ev_low_new)
        self.settings.copy_ROI_SCA.put(1)


    def warmup(self, hdf5_warmup=False):
        self.settings.total_points.put(1)
        if hdf5_warmup:
            self.hdf5.warmup()

    def prepare_to_fly(self, traj_duration):
        acq_rate = self.ext_trigger_device.freq.get()
        self.num_points = int(acq_rate * (traj_duration + 2))
        self.ext_trigger_device.prepare_to_fly(traj_duration)

    def stage(self):
        # self._infer_datum_keys()  # TODO: check parent class
        # self._datum_counter = itertools.count()  # TODO: check parent class
        self.settings.total_points.put(self.num_points)
        self.hdf5.file_write_mode.put(2)  # Stream.  Can be Capture (1)
        self.external_trig.put(True)
        self.settings.trigger_mode.put(0)  # Gate (0), Sync (1)
        staged_list = super().stage()
        staged_list += self.ext_trigger_device.stage()
        return staged_list

    def unstage(self):
        unstaged_list = super().unstage()
        # self._datum_counter = None
        self.hdf5.file_write_mode.put(0)
        self.external_trig.put(False)
        # self.settings.trigger_mode.put(1)  # TODO: Should we really change this?
        self.settings.total_points.put(1)
        unstaged_list += self.ext_trigger_device.unstage()
        return unstaged_list


ge_detector = GeDetector('XF:08IDB-ES{GE-Det:1}', name='ge_detector')

ttime.sleep(2)
ge_detector_stream = GeDetector('XF:08IDB-ES{GE-Det:1}', name="ge_detector_stream", ext_trigger_device=apb_trigger_ge_detector)