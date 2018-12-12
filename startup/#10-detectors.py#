import uuid
from collections import namedtuple
from concurrent.futures import ThreadPoolExecutor
import os
import shutil
import time as ttime
from ophyd import (ProsilicaDetector, SingleTrigger, Component as Cpt, Device,
                   EpicsSignal, EpicsSignalRO, ImagePlugin, StatsPlugin, ROIPlugin,
                   DeviceStatus)
from ophyd import DeviceStatus, set_and_wait
from bluesky.examples import NullStatus

from databroker.assets.handlers_base import HandlerBase


class BPM(SingleTrigger, ProsilicaDetector):
    image = Cpt(ImagePlugin, 'image1:')
    stats1 = Cpt(StatsPlugin, 'Stats1:')
    stats2 = Cpt(StatsPlugin, 'Stats2:')
    roi1 = Cpt(ROIPlugin, 'ROI1:')
    roi2 = Cpt(ROIPlugin, 'ROI2:')
    counts = Cpt(EpicsSignal, 'Pos:Counts')
    exp_time = Cpt(EpicsSignal, 'cam1:AcquireTime_RBV', write_pv='cam1:AcquireTime')
    # Dan Allan guessed about the nature of these signals. Fix them if you need them.
    ins = Cpt(EpicsSignal, 'Cmd:In-Cmd')
    ret = Cpt(EpicsSignal, 'Cmd:Out-Cmd')
    switch_insert = Cpt(EpicsSignalRO, 'Sw:InLim-Sts')
    switch_retract = Cpt(EpicsSignalRO, 'Sw:OutLim-Sts')
    polarity = 'pos'

    def insert(self):
        self.ins.put(1)

    def retract(self):
        self.ret.put(1)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.stage_sigs['cam.image_mode'] = 'Single'
        #self.stage_sigs.clear()  # default stage sigs do not apply

class CAMERA(SingleTrigger, ProsilicaDetector):
    image = Cpt(ImagePlugin, 'image1:')
    stats1 = Cpt(StatsPlugin, 'Stats1:')
    stats2 = Cpt(StatsPlugin, 'Stats2:')
    roi1 = Cpt(ROIPlugin, 'ROI1:')
    roi2 = Cpt(ROIPlugin, 'ROI2:')
    exp_time = Cpt(EpicsSignal, 'cam1:AcquireTime_RBV', write_pv='cam1:AcquireTime')
    polarity = 'pos'
    tiff_filepath = Cpt(EpicsSignal, 'TIFF1:FilePath_RBV', write_pv='TIFF1:FilePath')
    tiff_filename = Cpt(EpicsSignal, 'TIFF1:FileName_RBV', write_pv='TIFF1:FileName')
    tiff_filenumber = Cpt(EpicsSignal, 'TIFF1:FileNumber_RBV', write_pv='TIFF1:FileNumber')
    tiff_filefmt = Cpt(EpicsSignal, 'TIFF1:FileTemplate_RBV', write_pv='TIFF1:FileTemplate')

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.stage_sigs['cam.image_mode'] = 'Single'
        #self.stage_sigs.clear()  # default stage sigs do not apply

bpm_fm = BPM('XF:08IDA-BI{BPM:FM}', name='bpm_fm')
bpm_cm = BPM('XF:08IDA-BI{BPM:CM}', name='bpm_cm')
bpm_bt1 = BPM('XF:08IDA-BI{BPM:1-BT}', name='bpm_bt1')
bpm_bt2 = BPM('XF:08IDA-BI{BPM:2-BT}', name='bpm_bt2')
bpm_es = BPM('XF:08IDB-BI{BPM:ES}', name='bpm_es')
bpm_sp1 = CAMERA('XF:08IDB-BI{BPM:SP-1}', name='bpm_sp1')
bpm_ms1 = CAMERA('XF:08IDB-BI{BPM:MS-1}', name='bpm_ms1')

for bpm in [bpm_fm, bpm_cm, bpm_bt1, bpm_bt2, bpm_es, bpm_sp1]:
    bpm.read_attrs = ['stats1', 'stats2']
    bpm.image.read_attrs = ['array_data']
    bpm.stats1.read_attrs = ['total', 'centroid']
    bpm.stats2.read_attrs = ['total', 'centroid']

tc_mask2_4 = EpicsSignal('XF:08IDA-OP{Mir:2-CM}T:Msk2_4-I',
                         name='tc_mask2_4')
tc_mask2_3 = EpicsSignal('XF:08IDA-OP{Mir:2-CM}T:Msk2_3-I',
                         name='tc_mask2_3')


class Encoder(Device):
    """This class defines components but does not implement actual reading.

    See EncoderFS and EncoderParser"""
    pos_I = Cpt(EpicsSignal, '}Cnt:Pos-I')
    sec_array = Cpt(EpicsSignal, '}T:sec_Bin_')
    nsec_array = Cpt(EpicsSignal, '}T:nsec_Bin_')
    pos_array = Cpt(EpicsSignal, '}Cnt:Pos_Bin_')
    index_array = Cpt(EpicsSignal, '}Cnt:Index_Bin_')
    data_array = Cpt(EpicsSignal, '}Data_Bin_')
    # The '$' in the PV allows us to write 40 chars instead of 20.
    filepath = Cpt(EpicsSignal, '}ID:File.VAL', string=True)
    dev_name = Cpt(EpicsSignal, '}DevName')

    filter_dy = Cpt(EpicsSignal, '}Fltr:dY-SP')
    filter_dt = Cpt(EpicsSignal, '}Fltr:dT-SP')
    reset_counts = Cpt(EpicsSignal, '}Rst-Cmd')

    ignore_rb = Cpt(EpicsSignal, '}Ignore-RB')
    ignore_sel = Cpt(EpicsSignal, '}Ignore-Sel')

    def __init__(self, *args, reg, **kwargs):
        super().__init__(*args, **kwargs)
        self._ready_to_collect = False
        self._reg = reg
        if self.connected:
            self.ignore_sel.put(1)
            # self.filter_dt.put(10000)


class EncoderFS(Encoder):
    "Encoder Device, when read, returns references to data in filestore."
    chunk_size = 1024

    def stage(self):
        "Set the filename and record it in a 'resource' document in the filestore database."

        if(self.connected):
            print('Staging of {} starting'.format(self.name))
            DIRECTORY = '/GPFS/xf08id/'
            rpath = 'pizza_box_data'
            filename = 'en_' + str(uuid.uuid4())[:6]
            full_path = os.path.join(rpath, filename)
            self._full_path = os.path.join(DIRECTORY, full_path)  # stash for future reference
            if len(self._full_path) > 40:
                raise RuntimeError("Stupidly, EPICS limits the file path to 80 characters. "
                               "Choose a different DIRECTORY with a shorter path. (I know....)")
            self._full_path = os.path.join(DIRECTORY, full_path)  # stash for future reference
            # print(self._full_path)
            #
            # FIXME: Quick TEMPORARY fix for beamline disaster
            # we are writing the file to a temp directory in the ioc and
            # then moving it to the GPFS system.
            #
            ioc_file_root = '/home/softioc/tmp/'
            self._ioc_full_path = os.path.join(ioc_file_root, filename)
            self._filename = filename

            #self.filepath.put(self._full_path)   # commented out during disaster
            self.filepath.put(self._ioc_full_path)

            self.resource_uid = self._reg.register_resource(
                'PIZZABOX_ENC_FILE_TXT',
                DIRECTORY, full_path,
                {'chunk_size': self.chunk_size})

            super().stage()
            print('Staging of {} complete'.format(self.name))

    def unstage(self):
        if(self.connected):
            set_and_wait(self.ignore_sel, 1)
            return super().unstage()

    def kickoff(self):
        print('kickoff', self.name)
        self._ready_to_collect = True
        "Start writing data into the file."

        set_and_wait(self.ignore_sel, 0)

        # Return a 'status object' that immediately reports we are 'done' ---
        # ready to collect at any time.
        return NullStatus()

    def complete(self):
        print('storing', self.name, 'in', self._full_path)
        if not self._ready_to_collect:
            raise RuntimeError("must called kickoff() method before calling complete()")
        # Stop adding new data to the file.
        set_and_wait(self.ignore_sel, 1)
        #while not os.path.isfile(self._full_path):
        #    ttime.sleep(.1)
        return NullStatus()

    def collect(self):
        """
        Record a 'datum' document in the filestore database for each encoder.

        Return a dictionary with references to these documents.
        """
        print('Collect of {} starting'.format(self.name))
        self._ready_to_collect = False

        # Create an Event document and a datum record in filestore for each line
        # in the text file.
        now = ttime.time()
        ttime.sleep(1)  # wait for file to be written by pizza box

        workstation_file_root = '/mnt/xf08ida-ioc1/'
        workstation_full_path = os.path.join(workstation_file_root, self._filename)


        # FIXME: beam line disaster fix.
        # Let's move the file to the correct place
        print('Moving file from {} to {}'.format(workstation_full_path, self._full_path))
        cp_stat = shutil.copy(workstation_full_path, self._full_path)

        if os.path.isfile(self._full_path):
            with open(self._full_path, 'r') as f:
                linecount = len(list(f))
            chunk_count = linecount // self.chunk_size + int(linecount % self.chunk_size != 0)
            for chunk_num in range(chunk_count):
                datum_uid = self._reg.register_datum(
                    self.resource_uid, {'chunk_num': chunk_num})
                data = {self.name: datum_uid}
                yield {'data': data,
                       'timestamps': {key: now for key in data}, 'time': now}
            print('Collect of {} complete'.format(self.name))

        else:
            print('Collect {}: File was not created'.format(self.name))

    def describe_collect(self):
        # TODO Return correct shape (array dims)
        now = ttime.time()
        return {self.name: {self.name:
                     {'filename': self._full_path,
                      'devname': self.dev_name.value,
                      'source': 'pizzabox-enc-file',
                      'external': 'FILESTORE:',
                      'shape': [1024, 5],
                      'dtype': 'array'}}}


class DigitalOutput(Device):
    """ DigitalOutput """
    enable = Cpt(EpicsSignal, '}Ena-Cmd')
    period_sp = Cpt(EpicsSignal, '}Period-SP')
    unit_sel = Cpt(EpicsSignal, '}Unit-Sel')
    dutycycle_sp = Cpt(EpicsSignal, '}DutyCycle-SP')
    default_pol = Cpt(EpicsSignal, '}Dflt-Sel')

    def __init__(self, *args, reg, **kwargs):
        self._reg = reg
        super().__init__(*args, **kwargs)
        self._ready_to_collect = False
        if self.connected:
            self.enable.put(0)


class DigitalInput(Device):
    """This class defines components but does not implement actual reading.

    See DigitalInputFS """
    data_I = Cpt(EpicsSignal, '}Raw:Data-I_')
    sec_array = Cpt(EpicsSignal, '}T:sec_Bin_')
    nsec_array = Cpt(EpicsSignal, '}T:nsec_Bin_')
    index_array = Cpt(EpicsSignal, '}Cnt:Index_Bin_')
    data_array = Cpt(EpicsSignal, '}Data_Bin_')
    # The '$' in the PV allows us to write 40 chars instead of 20.
    filepath = Cpt(EpicsSignal, '}ID:File.VAL', string=True)
    dev_name = Cpt(EpicsSignal, '}DevName')

    ignore_rb = Cpt(EpicsSignal, '}Ignore-RB')
    ignore_sel = Cpt(EpicsSignal, '}Ignore-Sel')

    def __init__(self, *args, reg, **kwargs):
        self._reg = reg
        super().__init__(*args, **kwargs)
        self._ready_to_collect = False
        if self.connected:
            self.ignore_sel.put(1)



class DIFS(DigitalInput):
    "Encoder Device, when read, returns references to data in filestore."
    chunk_size = 1024

    def stage(self):
        "Set the filename and record it in a 'resource' document in the filestore database."

        print('Staging of {} starting'.format(self.name))
        DIRECTORY = '/GPFS/xf08id/'
        rpath = 'pizza_box_data'
        filename = 'di_' + str(uuid.uuid4())[:6]
        full_path = os.path.join(rpath, filename)
        self._full_path = os.path.join(DIRECTORY, full_path)  # stash for future reference
        if len(self._full_path) > 40:
            raise RuntimeError("EPICS filepath is limited to 80 characters. "
                           "Choose a DIRECTORY with a shorter path")
        self._full_path = os.path.join(DIRECTORY, full_path)  # stash for future reference

        #
        # FIXME: Quick TEMPORARY fix for beamline disaster
        # we are writing the file to a temp directory in the ioc and
        # then moving it to the GPFS system.
        #
        ioc_file_root = '/home/softioc/tmp/'
        self._ioc_full_path = os.path.join(ioc_file_root, filename)
        self._filename = filename

        # self.filepath.put(self._full_path)   # commented out during disaster
        self.filepath.put(self._ioc_full_path)

        self.resource_uid = self._reg.register_resource(
            'PIZZABOX_DI_FILE_TXT',
            DIRECTORY, full_path,
            {'chunk_size': self.chunk_size})

        super().stage()
        print('Staging of {} complete'.format(self.name))

    def unstage(self):
        set_and_wait(self.ignore_sel, 1)
        return super().unstage()

    def kickoff(self):
        print('kickoff', self.name)
        self._ready_to_collect = True
        "Start writing data into the file."

        set_and_wait(self.ignore_sel, 0)

        # Return a 'status object' that immediately reports we are 'done' ---
        # ready to collect at any time.
        return NullStatus()

    def complete(self):
        print('storing', self.name, 'in', self._full_path)
        if not self._ready_to_collect:
            raise RuntimeError("must called kickoff() method before calling complete()")
        # Stop adding new data to the file.
        set_and_wait(self.ignore_sel, 1)
        #while not os.path.isfile(self._full_path):
        #    ttime.sleep(.1)
        return NullStatus()

    def collect(self):
        """
        Record a 'datum' document in the filestore database for each encoder.

        Return a dictionary with references to these documents.
        """
        print('Collect of {} starting'.format(self.name))
        self._ready_to_collect = False

        # Create an Event document and a datum record in filestore for each line
        # in the text file.
        now = ttime.time()
        ttime.sleep(1)  # wait for file to be written by pizza box

        workstation_file_root = '/mnt/xf08idb-ioc1/'
        workstation_full_path = os.path.join(workstation_file_root, self._filename)


        # FIXME: beam line disaster fix.
        # Let's move the file to the correct place
        print('Moving file from {} to {}'.format(workstation_full_path, self._full_path))
        cp_stat = shutil.copy(workstation_full_path, self._full_path)

        if os.path.isfile(self._full_path):
            with open(self._full_path, 'r') as f:
                linecount = len(list(f))
            chunk_count = linecount // self.chunk_size + int(linecount % self.chunk_size != 0)
            for chunk_num in range(chunk_count):
                datum_uid = self._reg.register_datum(self.resource_uid,
                                                     {'chunk_num': chunk_num})
                data = {self.name: datum_uid}

                yield {'data': data,
                       'timestamps': {key: now for key in data}, 'time': now}
            print('Collect of {} complete'.format(self.name))
        else:
            print('collect {}: File was not created'.format(self.name))

    def describe_collect(self):
        # TODO Return correct shape (array dims)
        now = ttime.time()
        return {self.name: {self.name:
                     {'filename': self._full_path,
                      'devname': self.dev_name.value,
                      'source': 'pizzabox-di-file',
                      'external': 'FILESTORE:',
                      'shape': [1024, 5],
                      'dtype': 'array'}}}

class PizzaBoxFS(Device):
    ts_sec = Cpt(EpicsSignal, '}T:sec-I')
    #internal_ts_sel = Cpt(EpicsSignal, '}T:Internal-Sel')

    enc1 = Cpt(EncoderFS, ':1', reg=db.reg)
    enc2 = Cpt(EncoderFS, ':2', reg=db.reg)
    enc3 = Cpt(EncoderFS, ':3', reg=db.reg)
    enc4 = Cpt(EncoderFS, ':4', reg=db.reg)
    di = Cpt(DIFS, ':DI', reg=db.reg)
    do0 = Cpt(DigitalOutput, '-DO:0', reg=db.reg)
    do1 = Cpt(DigitalOutput, '-DO:1', reg=db.reg)
    do2 = Cpt(DigitalOutput, '-DO:2', reg=db.reg)
    do3 = Cpt(DigitalOutput, '-DO:3', reg=db.reg)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # must use internal timestamps or no bytes are written

    def kickoff(self):
        "Call encoder.kickoff() for every encoder."
        for attr_name in ['enc1', 'enc2', 'enc3', 'enc4']:
            status = getattr(self, attr_name).kickoff()
            print("Eli's test", self.attr_name)
        # it's fine to just return one of the status objects
        return status

    def collect(self):
        "Call encoder.collect() for every encoder."
        # Stop writing data to the file in all encoders.
        for attr_name in ['enc1', 'enc2', 'enc3', 'enc4']:
            set_and_wait(getattr(self, attr_name).ignore_sel, 1)
        # Now collect the data accumulated by all encoders.
        for attr_name in ['enc1', 'enc2', 'enc3', 'enc4']:
            yield from getattr(self, attr_name).collect()



pb1 = PizzaBoxFS('XF:08IDA-CT{Enc01', name = 'pb1')
pb2 = PizzaBoxFS('XF:08IDA-CT{Enc02', name = 'pb2')
pb4 = PizzaBoxFS('XF:08IDA-CT{Enc04', name = 'pb4') #PB inside hutch B (for now)
pb5 = PizzaBoxFS('XF:08IDA-CT{Enc05', name = 'pb5')
pb6 = PizzaBoxFS('XF:08IDA-CT{Enc06', name = 'pb6')
pb7 = PizzaBoxFS('XF:08IDA-CT{Enc07', name = 'pb7')
pb9 = PizzaBoxFS('XF:08IDA-CT{Enc09', name = 'pb9')
pb9.enc1.pulses_per_deg = 360000

class Adc(Device):
    file_size = Cpt(EpicsSignal, '}FileSize')
    reset = Cpt(EpicsSignal, '}Rst-Cmd')
    filepath = Cpt(EpicsSignal, '}ID:File.VAL', string=True)
    sec_array = Cpt(EpicsSignal, '}T:sec_Bin_')
    nsec_array = Cpt(EpicsSignal, '}T:nsec_Bin_')
    #pos_array = Cpt(EpicsSignal, '}Cnt:Pos_Bin_')
    index_array = Cpt(EpicsSignal, '}Cnt:Index_Bin_')
    data_array = Cpt(EpicsSignal, '}Data_Bin_')
    sample_rate = Cpt(EpicsSignal,'}F:Sample-I_', write_pv='}F:Sample-SP')
    enable_averaging = Cpt(EpicsSignal, '}Avrg-Sts', write_pv='}Avrg-Sel')
    averaging_points = Cpt(EpicsSignal, '}Avrg-SP')
    averaging_points_rbv = Cpt(EpicsSignal, '}GP-ADC:Reg0-RB_')
    volt_array = Cpt(EpicsSignal, '}V-I')
    volt = Cpt(EpicsSignal, '}E-I')
    offset = Cpt(EpicsSignal, '}Offset')
    dev_name = Cpt(EpicsSignal, '}DevName')
    dev_saturation = Cpt(EpicsSignal, '}DevSat')
    polarity = 'neg'

    enable_sel = Cpt(EpicsSignal, '}Ena-Sel')
    enable_rb = Cpt(EpicsSignal, '}Ena-Sts')

    def timeout_handler(self, signum, frame):
        print("{}.connected timeout".format(self.name))
        raise Exception("end of time")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._ready_to_collect = False

        #signal.signal(signal.SIGALRM, self.timeout_handler)
        #signal.setitimer(signal.ITIMER_REAL, 2)
        #try:
        #    while(self.connected == False):
        #        pass
        if self.connected:
            #self.enable_sel.put(1)
            #self.sample_rate.put(350)
            self.enable_averaging.put(1)
            if self.averaging_points.value == 0:
                self.averaging_points.put("1024")
        #except Exception as exc:
        #    pass
        #signal.alarm(0)


class AdcFS(Adc):
    "Adc Device, when read, returns references to data in filestore."
    chunk_size = 1024

    def __init__(self, *args, reg, **kwargs):
        self._reg = reg
        self.file_move_executor = ThreadPoolExecutor(max_workers=2)
        super().__init__(*args, **kwargs)

    def stage(self):
        "Set the filename and record it in a 'resource' document in the filestore database."


        if(self.connected):
            print( 'Staging of {} starting'.format(self.name))
            DIRECTORY = '/GPFS/xf08id/'
            rpath = 'pizza_box_data'
            filename = 'an_' + str(uuid.uuid4())[:6]
            full_path = os.path.join(rpath, filename)
            self._full_path = os.path.join(DIRECTORY, full_path)  # stash for future reference
            if len(self._full_path) > 40:
                raise RuntimeError("EPICS filepath is limited to 80 characters. "
                                   "Choose a DIRECTORY with a shorter path")
            self._full_path = os.path.join(DIRECTORY, full_path)  # stash for future reference
            #print(self._full_path) Eli 2018-01-30
            #
            # FIXME: Quick TEMPORARY fix for beamline disaster
            # we are writing the file to a temp directory in the ioc and
            # then moving it to the GPFS system.
            #
            ioc_file_root = '/home/softioc/tmp/'
            self._ioc_full_path = os.path.join(ioc_file_root, filename)
            self._filename = filename

            print("Temp IOC filename:" + self._ioc_full_path)

            #self.filepath.put(self._full_path)   # commented out during disaster
            self.filepath.put(self._ioc_full_path)

            self.resource_uid = self._reg.register_resource(
                'PIZZABOX_AN_FILE_TXT',
                DIRECTORY, full_path,
                {'chunk_size': self.chunk_size})
            print('Staging of {} complete'.format(self.name))
            super().stage()

    def unstage(self):
        if(self.connected):
            set_and_wait(self.enable_sel, 1)
            return super().unstage()

    def kickoff(self):
        print('kickoff', self.name)
        self._ready_to_collect = True
        "Start writing data into the file."
        set_and_wait(self.enable_sel, 0)
        # Return a 'status object' that immediately reports we are 'done' ---
        # ready to collect at any time.
        return NullStatus()

    def complete(self):
        print('storing', self.name, 'in', self._full_path)
        if not self._ready_to_collect:
            raise RuntimeError("must called kickoff() method before calling complete()")
        # Stop adding new data to the file.
        set_and_wait(self.enable_sel, 1)
        return NullStatus()

    def collect(self):
        """
        Record a 'datum' document in the filestore database for each encoder.

        Return a dictionary with references to these documents.
        """
        print('Collect of {} starting'.format(self.name))
        self._ready_to_collect = False

        # Create an Event document and a datum record in filestore for each line
        # in the text file.
        now = ttime.time()
        ttime.sleep(1)  # wait for file to be written by pizza box

        workstation_file_root = '/mnt/xf08idb-ioc1/'
        workstation_full_path = os.path.join(workstation_file_root, self._filename)

        # FIXME: beam line disaster fix.
        # Let's move the file to the correct place
        print('Moving file from {} to {}'.format(workstation_full_path, self._full_path))
        stat = shutil.copy(workstation_full_path, self._full_path)

        if os.path.isfile(self._full_path):
            with open(self._full_path, 'r') as f:
                linecount = 0
                for ln in f:
                    linecount += 1
            chunk_count = linecount // self.chunk_size + int(linecount % self.chunk_size != 0)
            for chunk_num in range(chunk_count):
                datum_uid = self._reg.register_datum(self.resource_uid,
                                                     {'chunk_num': chunk_num})
                data = {self.name: datum_uid}

                yield {'data': data,
                       'timestamps': {key: now for key in data}, 'time': now}
            print('Collect of {} complete'.format(self.name))
        else:
            print('collect {}: File was not created'.format(self.name))

    def describe_collect(self):
        # TODO Return correct shape (array dims)
        now = ttime.time()
        return {self.name: {self.name:
                     {'filename': self._full_path,
                      'devname': self.dev_name.value,
                      'source': 'pizzabox-adc-file',
                      'external': 'FILESTORE:',
                      'shape': [5,],
                      'dtype': 'array'}}}


class PizzaBoxAnalogFS(Device):
    #internal_ts_sel = Cpt(EpicsSignal, 'Gen}T:Internal-Sel')

    adc1 = Cpt(AdcFS, 'ADC:1', reg=db.reg)
    adc6 = Cpt(AdcFS, 'ADC:6', reg=db.reg)
    adc7 = Cpt(AdcFS, 'ADC:7', reg=db.reg)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # must use internal timestamps or no bytes are written
        # self.stage_sigs[self.internal_ts_sel] = 1

    def kickoff(self):
        "Call encoder.kickoff() for every encoder."
        for attr_name in ['adc1']: #, 'adc2', 'adc3', 'adc4']:
            status = getattr(self, attr_name).kickoff()
        # it's fine to just return one of the status objects
        return status

    def collect(self):
        "Call adc.collect() for every encoder."
        # Stop writing data to the file in all encoders.
        for attr_name in ['adc1']: #, 'adc2', 'adc3', 'adc4']:
            set_and_wait(getattr(self, attr_name).enable_sel, 0)
        # Now collect the data accumulated by all encoders.
        for attr_name in ['adc1']: #, 'adc2', 'adc3', 'adc4']:
            yield from getattr(self, attr_name).collect()

pba1 = PizzaBoxAnalogFS('XF:08IDB-CT{GP1-', name = 'pba1')
pba2 = PizzaBoxAnalogFS('XF:08IDB-CT{GP-', name = 'pba2')

class PizzaBoxEncHandlerTxt(HandlerBase):
    encoder_row = namedtuple('encoder_row',
                             ['ts_s', 'ts_ns', 'encoder', 'index', 'state'])
    "Read PizzaBox text files using info from filestore."
    def __init__(self, fpath, chunk_size):
        self.chunk_size = chunk_size
        with open(fpath, 'r') as f:
            self.lines = list(f)

    def __call__(self, chunk_num):
        cs = self.chunk_size
        return [self.encoder_row(*(int(v) for v in ln.split()))
                for ln in self.lines[chunk_num*cs:(chunk_num+1)*cs]]


class PizzaBoxDIHandlerTxt(HandlerBase):
    di_row = namedtuple('di_row', ['ts_s', 'ts_ns', 'encoder', 'index', 'di'])
    "Read PizzaBox text files using info from filestore."
    def __init__(self, fpath, chunk_size):
        self.chunk_size = chunk_size
        with open(fpath, 'r') as f:
            self.lines = list(f)

    def __call__(self, chunk_num):
        cs = self.chunk_size
        return [self.di_row(*(int(v) for v in ln.split()))
                for ln in self.lines[chunk_num*cs:(chunk_num+1)*cs]]


class PizzaBoxAnHandlerTxt(HandlerBase):
    encoder_row = namedtuple('encoder_row', ['ts_s', 'ts_ns', 'index', 'adc'])
    "Read PizzaBox text files using info from filestore."

    bases = (10, 10, 10, 16)
    def __init__(self, fpath, chunk_size):
        self.chunk_size = chunk_size
        with open(fpath, 'r') as f:
            self.lines = list(f)

    def __call__(self, chunk_num):

        cs = self.chunk_size
        return [self.encoder_row(*(int(v, base=b) for v, b in zip(ln.split(), self.bases)))
                for ln in self.lines[chunk_num*cs:(chunk_num+1)*cs]]



db.reg.register_handler('PIZZABOX_AN_FILE_TXT',
                        PizzaBoxAnHandlerTxt, overwrite=True)
db.reg.register_handler('PIZZABOX_ENC_FILE_TXT',
                        PizzaBoxEncHandlerTxt, overwrite=True)
db.reg.register_handler('PIZZABOX_DI_FILE_TXT',
                        PizzaBoxDIHandlerTxt, overwrite=True)
