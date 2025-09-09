print(ttime.ctime() + ' >>>> ' + __file__)

class Electrometer(Device):

    polarity = 'neg'

    ch1 = Cpt(EpicsSignal, 'SA:A:uA-I')
    ch2 = Cpt(EpicsSignal, 'SA:B:uA-I')
    ch3 = Cpt(EpicsSignal, 'SA:C:uA-I')
    ch4 = Cpt(EpicsSignal, 'SA:D:uA-I')
    ch5 = Cpt(EpicsSignal, 'SA:E:mV-I')
    ch6 = Cpt(EpicsSignal, 'SA:F:mV-I')
    ch7 = Cpt(EpicsSignal, 'SA:G:mV-I')
    ch8 = Cpt(EpicsSignal, 'SA:H:mV-I')

    ch1_range = Cpt(EpicsSignal, 'ADC:Range:A-SP')
    ch2_range = Cpt(EpicsSignal, 'ADC:Range:D-SP')
    ch3_range = Cpt(EpicsSignal, 'ADC:Range:C-SP')
    ch4_range = Cpt(EpicsSignal, 'ADC:Range:D-SP')
    ch5_range = Cpt(EpicsSignal, 'ADC:Range:E-SP')
    ch6_range = Cpt(EpicsSignal, 'ADC:Range:F-SP')
    ch7_range = Cpt(EpicsSignal, 'ADC:Range:G-SP')
    ch8_range = Cpt(EpicsSignal, 'ADC:Range:H-SP')


    # ch1_offset = Cpt(EpicsSignal, 'Ch1:User:Offset-SP', kind=Kind.config)
    # ch2_offset = Cpt(EpicsSignal, 'Ch2:User:Offset-SP', kind=Kind.config)
    # ch3_offset = Cpt(EpicsSignal, 'Ch3:User:Offset-SP', kind=Kind.config)
    # ch4_offset = Cpt(EpicsSignal, 'Ch4:User:Offset-SP', kind=Kind.config)
    # ch5_offset = Cpt(EpicsSignal, 'Ch5:User:Offset-SP', kind=Kind.config)
    # ch6_offset = Cpt(EpicsSignal, 'Ch6:User:Offset-SP', kind=Kind.config)
    # ch7_offset = Cpt(EpicsSignal, 'Ch7:User:Offset-SP', kind=Kind.config)
    # ch8_offset = Cpt(EpicsSignal, 'Ch8:User:Offset-SP', kind=Kind.config)

    acquire = Cpt(EpicsSignal, 'FA:SoftTrig-SP', kind=Kind.omitted)
    acquiring = Cpt(EpicsSignal, 'FA:Busy-I', kind=Kind.omitted)


    divide = Cpt(EpicsSignal, 'FA:Divide-SP')
    sample_len = Cpt(EpicsSignal, 'FA:Samples-SP')
    wf_len = Cpt(EpicsSignal, 'FA:Wfm:Length-SP')

    stream = Cpt(EpicsSignal, 'FA:Stream-SP', kind=Kind.omitted)
    streaming = Cpt(EpicsSignal, 'FA:Streaming-I', kind=Kind.omitted)

    acq_rate= Cpt(EpicsSignal,'FA:Rate-I', kind=Kind.omitted)
    stream_samples = Cpt(EpicsSignal, 'FA:Stream:Samples-SP')

    filename_bin = Cpt(EpicsSignal, 'FA:Stream:Bin:File-SP')
    filebin_status = Cpt(EpicsSignal, 'FA:Stream:Bin:File:Status-I')

    trig_source = Cpt(EpicsSignal, 'Machine:Clk-SP')


em = Electrometer('xf08id-em1:', name = 'em')

class ElectrometerAverage(Electrometer):

    ch1_mean = Cpt(EpicsSignal, 'FA:A:Mean-I', kind=Kind.hinted)
    ch2_mean = Cpt(EpicsSignal, 'FA:B:Mean-I', kind=Kind.hinted)
    ch3_mean = Cpt(EpicsSignal, 'FA:C:Mean-I', kind=Kind.hinted)
    ch4_mean = Cpt(EpicsSignal, 'FA:D:Mean-I', kind=Kind.hinted)
    ch5_mean = Cpt(EpicsSignal, 'FA:E:Mean-I', kind=Kind.hinted)
    ch6_mean = Cpt(EpicsSignal, 'FA:F:Mean-I', kind=Kind.hinted)
    ch7_mean = Cpt(EpicsSignal, 'FA:G:Mean-I', kind=Kind.hinted)
    ch8_mean = Cpt(EpicsSignal, 'FA:H:Mean-I', kind=Kind.hinted)

    time_wf = Cpt(EpicsSignal, 'FA:Time-Wfm', kind=Kind.hinted)
    ch1_wf = Cpt(EpicsSignal, 'FA:A-Wfm', kind=Kind.hinted)
    ch2_wf = Cpt(EpicsSignal, 'FA:B-Wfm', kind=Kind.hinted)
    ch3_wf = Cpt(EpicsSignal, 'FA:C-Wfm', kind=Kind.hinted)
    ch4_wf = Cpt(EpicsSignal, 'FA:D-Wfm', kind=Kind.hinted)
    ch5_wf = Cpt(EpicsSignal, 'FA:E-Wfm', kind=Kind.hinted)
    ch6_wf = Cpt(EpicsSignal, 'FA:F-Wfm', kind=Kind.hinted)
    ch7_wf = Cpt(EpicsSignal, 'FA:G-Wfm', kind=Kind.hinted)
    ch8_wf = Cpt(EpicsSignal, 'FA:H-Wfm', kind=Kind.hinted)



    saved_status = None
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._capturing = None
        self._ready_to_collect = False

    def trigger(self):
        def callback(value, old_value, **kwargs):
            if self._capturing and int(round(old_value)) == 1 and int(round(value)) == 0:
                self._capturing = False
                return True
            else:
                self._capturing = True
                return False

        status = SubscriptionStatus(self.acquiring, callback)
        # print_to_gui('TRIGGERING')
        self.acquire.set(1)
        return status

    def save_current_status(self):
        self.saved_status = {}
        self.saved_status['divide'] = self.divide.get()
        self.saved_status['sample_len'] = self.sample_len.get()
        self.saved_status['wf_len'] = self.wf_len.get()

    def restore_to_saved_status(self):
        yield from bps.abs_set(self.divide, self.saved_status['divide'])
        yield from bps.abs_set(self.sample_len, self.saved_status['sample_len'])
        yield from bps.abs_set(self.wf_len, self.saved_status['wf_len'])

    def read_exposure_time(self):
        # data_rate = self.data_rate.get()
        data_rate = self.acq_rate.get()
        sample_len = self.sample_len.get()
        return np.round((data_rate * sample_len / 1000), 3)

    def set_exposure_time(self, new_exp_time):
        data_rate = self.acq_rate.get()
        sample_len = 500 * (np.round(new_exp_time * data_rate * 1000 / 500))
        self.sample_len.set(sample_len).wait()
        self.wf_len.set(sample_len).wait()



em_ave = ElectrometerAverage('xf08id-em1:', name = 'em_ave')

class ElectrometerStream(ElectrometerAverage):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._acquiring = None

        self._asset_docs_cache = deque()
        self._resource_uid = None
        self._datum_counter = None
        self.num_points = None

    def stage(self):
        file_uid = new_uid()
        # self.calc_num_points(traj_duration)
        self.stream_samples.put(self.num_points)
        #self.filename_target = f'{ROOT_PATH}/data/apb/{dt.datetime.strftime(dt.datetime.now(), "%Y/%m/%d")}/{file_uid}'
        # Note: temporary static file name in GPFS, due to the limitation of 40 symbols in the filename field.
        #self.filename = f'{ROOT_PATH}/data/apb/{file_uid[:8]}'
        self.filename = f'{ROOT_PATH}/{RAW_PATH}/em/{dt.datetime.strftime(dt.datetime.now(), "%Y/%m/%d")}/{file_uid}'
        self.filename_bin.put(f'{self.filename}.bin')
        # self.filename_txt.put(f'{self.filename}.txt')

        self._resource_uid = new_uid()
        resource = {'spec': 'EM',
                    'root': ROOT_PATH,  # from 00-startup.py (added by mrakitin for future generations :D)
                    'resource_path': f'{self.filename}.bin',
                    'resource_kwargs': {},
                    'path_semantics': os.name,
                    'uid': self._resource_uid}
        self._asset_docs_cache.append(('resource', resource))
        self._datum_counter = itertools.count()

        status = self.trig_source.set(0)
        status.wait()
        return super().stage()


    def kickoff(self):
        return self.stream.set(1)



    def trigger(self):
        def callback(value, old_value, **kwargs):
            # print(f'{ttime.time()} {old_value} ---> {value}')
            if self._acquiring and int(round(old_value)) == 1 and int(round(value)) == 0:
                self._acquiring = False
                return True
            else:
                self._acquiring = True
                return False

        status = SubscriptionStatus(self.acquiring, callback)
        self.acquire.set(1)
        return status

    def unstage(self, *args, **kwargs):
        self._datum_counter = None
        return super().unstage(*args, **kwargs)
        # self.stream.set(0)


    # # Fly-able interface

    # Not sure if we need it here or in FlyerAPB (see 63-...)


    def complete(self, *args, **kwargs):
        # print(f'{ttime.ctime()} >>> {self.name} complete: begin')
        print_to_gui(f'{self.name} complete starting', add_timestamp=True)
        self.stream.set(0).wait()
        def callback_saving(value, old_value, **kwargs):
            if int(round(old_value)) == 1 and int(round(value)) == 0:
                return True
            else:
                return False
        filebin_st = SubscriptionStatus(self.filebin_status, callback_saving)
        filetxt_st = SubscriptionStatus(self.filetxt_status, callback_saving)
        # print_debug(f'filebin_st={filebin_st} filetxt_st={filetxt_st}')
        self._datum_ids = []
        datum_id = '{}/{}'.format(self._resource_uid, next(self._datum_counter))
        datum = {'resource': self._resource_uid,
                 'datum_kwargs': {},
                 'datum_id': datum_id}
        self._asset_docs_cache.append(('datum', datum))
        # print(f'{ttime.ctime()} >>> {self.name} complete: done')
        self._datum_ids.append(datum_id)
        print_to_gui(f'{self.name} complete done', add_timestamp=True)
        return filebin_st & filetxt_st

    def collect(self): # Copied from 30-detectors.py (class EncoderFS)
        # print(f'{ttime.ctime()} >>> {self.name} collect starting')
        print_to_gui(f'{self.name} collect starting', add_timestamp=True)
        now = ttime.time()
        for datum_id in self._datum_ids:
            data = {self.name: datum_id}
            yield {'data': data,
                   'timestamps': {key: now for key in data},
                   'time': now,
                   'filled': {key: False for key in data}}
        print_to_gui(f'{self.name} collect done', add_timestamp=True)
        # print(f'{ttime.ctime()} >>> {self.name} collect complete')

    def describe_collect(self):
        return_dict = {self.name:
                            {self.name: {'source': 'APB',
                                         'dtype': 'array',
                                         'shape': [-1, -1],
                                         'filename_bin': f'{self.filename}.bin',
                                         'filename_txt': f'{self.filename}.txt',
                                         'external': 'FILESTORE:'}}}
        return return_dict

    def collect_asset_docs(self):
        items = list(self._asset_docs_cache)
        self._asset_docs_cache.clear()
        for item in items:
            yield item

    def prepare_to_fly(self, traj_duration):
        # traj_duration = get_traj_duration()
        acq_num_points = traj_duration * self.acq_rate.get() * 1000 * 1.3
        self.num_points = int(round(acq_num_points, ndigits=-3))







    # def set_stream_points(self):
    #     trajectory_manager.current_trajectory_duration


em_stream = ElectrometerStream(prefix="xf08id-em1:", name="apb_stream")
