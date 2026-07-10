print(ttime.ctime() + ' >>>> ' + __file__)
class BPM(SingleTrigger, ProsilicaDetector):
    polarity = 'pos'
    image = Cpt(ImagePlugin, 'image1:')
    stats1 = Cpt(StatsPluginV33, 'Stats1:')
    stats2 = Cpt(StatsPluginV33, 'Stats2:')
    stats3 = Cpt(StatsPluginV33, 'Stats3:')
    stats4 = Cpt(StatsPluginV33, 'Stats4:')

    roi1 = Cpt(ROIPlugin, 'ROI1:')
    roi2 = Cpt(ROIPlugin, 'ROI2:')
    roi3 = Cpt(ROIPlugin, 'ROI3:')
    roi4 = Cpt(ROIPlugin, 'ROI4:')

    counts = Cpt(EpicsSignal, 'Pos:Counts')
    exp_time = Cpt(EpicsSignal, 'cam1:AcquireTime_RBV', write_pv='cam1:AcquireTime')
    image_mode = Cpt(EpicsSignal,'cam1:ImageMode')
    acquire = Cpt(EpicsSignal, 'cam1:Acquire')

    # Actuator
    insert = Cpt(EpicsSignal, 'Cmd:In-Cmd')
    inserted = Cpt(EpicsSignalRO, 'Sw:InLim-Sts')

    retract = Cpt(EpicsSignal, 'Cmd:Out-Cmd')
    retracted = Cpt(EpicsSignal, 'Sw:OutLim-Sts')


    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.stage_sigs['cam.image_mode'] = 'Single'
        self.polarity = 'pos'
        self.image_height = self.image.height.get()
        self.image_width = self.image.width.get()
        self.frame_rate = self.cam.ps_frame_rate
        self.stats1.total.polarity = 'pos'
        self.stats2.total.polarity = 'pos'
        self.stats3.total.polarity = 'pos'
        self.stats4.total.polarity = 'pos'
        # self._inserting = None
        # self._retracting = None

    def set(self, command):
        def callback(value, old_value, **kwargs):
            if value == 1:
                return True
            return False

        if command.lower() == 'insert':
            status = SubscriptionStatus(self.inserted, callback)
            self.insert.set('Insert')
            return status

        if command.lower() == 'retract':
            status = SubscriptionStatus(self.retracted, callback)
            self.retract.set('Retract')
            return status

    def read_exposure_time(self):
        return self.exp_time.get()

    def set_exposure_time(self, new_exp_time):
        self.exp_time.set(new_exp_time).wait()

    def adjust_camera_exposure_time(self, roi_index=1,
                                    target_max_counts=80, atol=10,
                                    max_exp_time_thresh=1,
                                    min_exp_time_thresh=0.00002, percentile=95):
        stats = getattr(self, f'stats{roi_index}')
        while True:
            # current_maximum = stats.max_value.get()
            current_maximum = np.percentile(self.image.array_data.get(), percentile)
            current_exp_time = self.exp_time.get()
            delta = np.abs(current_maximum - target_max_counts)
            ratio = target_max_counts / current_maximum
            new_exp_time = np.clip(current_exp_time * ratio, min_exp_time_thresh, max_exp_time_thresh)

            if new_exp_time != current_exp_time:
                if delta > atol:
                    # self.exp_time.set(new_exp_time).wait()
                    self.set_exposure_time(new_exp_time)
                    ttime.sleep(np.max((0.5, new_exp_time)))
                    continue
            break

    def adjust_camera_exposure_time_full_image(self, **kwargs):
        x = self.roi1.min_xyz.min_x.get()
        y = self.roi1.min_xyz.min_y.get()
        dx = self.roi1.size.x.get()
        dy = self.roi1.size.y.get()

        self.roi1.min_xyz.min_x.put(0)
        self.roi1.min_xyz.min_y.put(0)
        self.roi1.size.x.put(self.image_width)
        self.roi1.size.y.put(self.image_height)

        self.adjust_camera_exposure_time(**kwargs)

        self.roi1.min_xyz.min_x.put(x)
        self.roi1.min_xyz.min_y.put(y)
        self.roi1.size.x.put(dx)
        self.roi1.size.y.put(dy)

    def get_image_array_data_reshaped(self):
        return np.reshape(self.image.array_data.get(), (self.image_height, self.image_width))

    # @property
    # def image_height(self):
    #     return self.image.height.get()

    # @property
    # def image_width(self):
    #     return self.image.width.get()



bpm_fm = BPM('XF:08IDA-BI{BPM:FM}', name='bpm_fm')
bpm_cm = BPM('XF:08IDA-BI{BPM:CM}', name='bpm_cm')
bpm_bt1 = BPM('XF:08IDA-BI{BPM:1-BT}', name='bpm_bt1')
bpm_bt2 = BPM('XF:08IDA-BI{BPM:2-BT}', name='bpm_bt2')

class FeedbackBPM(BPM):
    ioc_reboot_pv = None

    def append_ioc_reboot_pv(self, ioc_reboot_pv):
        self.ioc_reboot_pv = ioc_reboot_pv

    def reboot_ioc(self):
        if self.ioc_reboot_pv is not None:
            self.ioc_reboot_pv.put(1)
            ttime.sleep(10)
            self.acquire.put(1)
        else:
            print_to_gui('ioc_reboot_pv is not appended. IOC reboot impossible.')

    @property
    def acquiring(self):
        return bool(self.acquire.get())

    @property
    def image_centroid_y(self):
        y = self.stats1.centroid.y.get()
        return self.image_height - y

    @property
    def image_centroid_x(self):
        return self.stats1.centroid.x.get()


from xas.image_analysis import CameraCalibrationFF
class SamplePositionerBPM(BPM, ObjectWithSettings):

    def __init__(self, *args, **kwargs):
        from redis_json_dict import RedisJSONDict
        BPM.__init__(self, *args, **kwargs)
        ObjectWithSettings.__init__(self, json_path=f'{ROOT_PATH_SHARED}/settings/json/{self.name}_settings.json', defaultdict_use=True)
        self._read_beam_pos_from_settings()

        self._calibration_redis_store = RedisJSONDict(redis_settings_client, prefix=f'{self.name}_calibration')
        self._calibration_json_path = f'{ROOT_PATH_SHARED}/settings/json/{self.name}_calibration.json'
        self.load_calibration()

        self.grid_lines = None

    def _read_beam_pos_from_settings(self):
        self.beam_pos_x = self.config['beam_pos_x']
        self.beam_pos_y = self.config['beam_pos_y']

    def read_current_config(self):
        return {'beam_pos_x': self.beam_pos_x,
                'beam_pos_y': self.beam_pos_y}

    def set_beam_coordinates(self, beam_pos_x, beam_pos_y):
        self.beam_pos_x = beam_pos_x
        self.beam_pos_y = beam_pos_y
        self.save_current_config_to_settings()

    def load_calibration(self):
        try:
            try:
                calibration_data_dict = self._calibration_redis_store['calibration']
            except Exception:
                with open(self._calibration_json_path) as fp:
                    calibration_data_dict = json.load(fp)
                try:
                    self._calibration_redis_store['calibration'] = calibration_data_dict
                except Exception:
                    pass
            self.calibration = CameraCalibrationFF(calibration_data_dict['pix_xy1'],
                                                   calibration_data_dict['pix_xy2'],
                                                   calibration_data_dict['stage_xy'],
                                                   npoly=calibration_data_dict['npoly'])
        except Exception as e:
            print(f'Calibration for {self.name} could not be loaded. Reason: {e}')
            self.calibration = None

    def set_calibration(self, calibration_data_dict):
        self.calibration = CameraCalibrationFF(calibration_data_dict['pix_xy1'],
                                               calibration_data_dict['pix_xy2'],
                                               calibration_data_dict['stage_xy'],
                                               npoly=calibration_data_dict['npoly'])
        self.save_calibration_to_settings(calibration_data_dict)

    def save_calibration_to_settings(self, calibration_data_dict):
        _dict = {**calibration_data_dict}
        for k, v in _dict.items():
            if type(v) == np.ndarray:
                _dict[k] = v.tolist()
        try:
            self._calibration_redis_store['calibration'] = _dict
        except Exception:
            pass
        if os.environ.get('SAVE_CONFIG_TO_JSON'):
            try:
                with open(self._calibration_json_path, 'w') as f:
                    json.dump(_dict, f)
            except Exception:
                pass

    def update_calibration_npoly(self, npoly):
        self.calibration.update_npoly(npoly)
        self.save_calibration_to_settings(self.calibration_data_dict)

    @property
    def calibration_data_dict(self):
        return self.calibration.calibration_data_dict

    @property
    def calibration_info(self):
        if self.calibration is not None:
            return f'{self.name}: {self.calibration.info}'
        return ''

    def compute_calibration_grid_lines(self, nlines=8, stage_step=-5):
        self.grid_lines = self.calibration.compute_grid_lines(  nlines=nlines,
                                                                xmax=self.image_width,
                                                                ymax=self.image_height,
                                                                stage_step=stage_step)

    def compute_stage_motion_to_beam(self, x, y):
        xy = np.vstack((x, y)).T
        return self.calibration.compute_stage_motion( xy, (self.beam_pos_x, self.beam_pos_y))

    def compute_point_from_stage(self, stage_x, stage_y):
        stage_xy = np.vstack([stage_x, stage_y]).T
        xy = np.array([[self.beam_pos_x, self.beam_pos_y]] * stage_x.size)
        return self.calibration.compute_new_pixel(xy, stage_xy)


camera_sp1 = SamplePositionerBPM('XF:08IDB-BI{BPM:SP-1}', name='camera_sp1')
# camera_sp1.calibration.update_npoly(1)

camera_sp2 = SamplePositionerBPM('XF:08IDB-BI{BPM:SP-2}', name='camera_sp2')
# camera_sp2.calibration.update_npoly(1)
# camera_sp2.grid_lines = compute_calibration_grid_lines(camera_sp2, stage_step=10)

class CAMERA(SingleTrigger, ProsilicaDetector):
    image = Cpt(ImagePlugin, 'image1:')

    stats1 = Cpt(StatsPluginV33, 'Stats1:')
    stats2 = Cpt(StatsPluginV33, 'Stats2:')
    roi1 = Cpt(ROIPlugin, 'ROI1:')
    roi2 = Cpt(ROIPlugin, 'ROI2:')

    exp_time = Cpt(EpicsSignal, 'cam1:AcquireTime_RBV', write_pv='cam1:AcquireTime')
    polarity = 'pos'
    tiff_filepath = Cpt(EpicsSignal, 'TIFF1:FilePath_RBV', write_pv='TIFF1:FilePath')
    tiff_filename = Cpt(EpicsSignal, 'TIFF1:FileName_RBV', write_pv='TIFF1:FileName')
    tiff_filenumber = Cpt(EpicsSignal, 'TIFF1:FileNumber_RBV', write_pv='TIFF1:FileNumber')
    tiff_filefmt = Cpt(EpicsSignal, 'TIFF1:FileTemplate_RBV', write_pv='TIFF1:FileTemplate')

    bar1 = Cpt(EpicsSignal, 'Bar1:BarcodeMessage1_RBV')
    bar2 = Cpt(EpicsSignal, 'Bar1:BarcodeMessage2_RBV')
    bar3 = Cpt(EpicsSignal, 'Bar1:BarcodeMessage3_RBV')
    bar4 = Cpt(EpicsSignal, 'Bar1:BarcodeMessage4_RBV')
    bar5 = Cpt(EpicsSignal, 'Bar1:BarcodeMessage5_RBV')

    bar1Corner1X = Cpt(EpicsSignal, 'Bar1:UpperLeftX_RBV')
    bar1Corner2X = Cpt(EpicsSignal, 'Bar1:UpperRightX_RBV')
    bar1Corner3X = Cpt(EpicsSignal, 'Bar1:LowerLeftX_RBV')
    bar1Corner4X = Cpt(EpicsSignal, 'Bar1:LowerRightX_RBV')
    bar1Corner1Y = Cpt(EpicsSignal, 'Bar1:UpperLeftY_RBV')
    bar1Corner2Y = Cpt(EpicsSignal, 'Bar1:UpperRightY_RBV')
    bar1Corner3Y = Cpt(EpicsSignal, 'Bar1:LowerLeftY_RBV')
    bar1Corner4Y = Cpt(EpicsSignal, 'Bar1:LowerRightY_RBV')

    bar2Corner1X = Cpt(EpicsSignal, 'Bar2:UpperLeftX_RBV')
    bar2Corner2X = Cpt(EpicsSignal, 'Bar2:UpperRightX_RBV')
    bar2Corner3X = Cpt(EpicsSignal, 'Bar2:LowerLeftX_RBV')
    bar2Corner4X = Cpt(EpicsSignal, 'Bar2:LowerRightX_RBV')
    bar2Corner1Y = Cpt(EpicsSignal, 'Bar2:UpperLeftY_RBV')
    bar2Corner2Y = Cpt(EpicsSignal, 'Bar2:UpperRightY_RBV')
    bar2Corner3Y = Cpt(EpicsSignal, 'Bar2:LowerLeftY_RBV')
    bar2Corner4Y = Cpt(EpicsSignal, 'Bar2:LowerRightY_RBV')

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        #self.stage_sigs.clear()  # default stage sigs do not apply

    def stage(self, acquire_time, image_mode, *args, **kwargs):
        self.stage_sigs['cam.acquire_time'] =  acquire_time
        self.stage_sigs['cam.image_mode'] = image_mode
        super().stage(*args, **kwargs)


class FoilCAMERA(CAMERA):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.read_foil_data()

    def _read_foil_info(self):
        from redis_json_dict import RedisJSONDict
        try:
            foil_wheel_store = RedisJSONDict(redis_settings_client, prefix='foil_wheel')
            foil_info = foil_wheel_store['foil_wheel']
        except Exception:
            with open(f'{ROOT_PATH_SHARED}/settings/json/foil_wheel.json') as fp:
                foil_info = json.load(fp)
            try:
                foil_wheel_store = RedisJSONDict(redis_settings_client, prefix='foil_wheel')
                foil_wheel_store['foil_wheel'] = foil_info
            except Exception:
                pass
        return foil_info

    def read_foil_data(self):
        foil_info = self._read_foil_info()
        reference_foils = [item['element'] for item in foil_info]
        edges = [item['edge'] for item in foil_info]
        self.edge_dict = {}
        for foil, edge in zip(reference_foils, edges):
            self.edge_dict[foil] = edge

    @property
    def barcode1(self):
        return str(self.bar1.get()[:-1], encoding='UTF-8').strip()

    @property
    def barcode2(self):
        return str(self.bar2.get()[:-1], encoding='UTF-8').strip()

    def read_current_foil(self, error_message_func=None):
        #WIP test that it works with single foil
        if self.barcode1 != '':
            if self.barcode1 != 'empty':
                return self.barcode1
        if self.barcode2 != '':
            if self.barcode2 != 'empty':
                return self.barcode2

        msg = f'Reference foil not found'
        if error_message_func is not None:
            error_message_func(msg)
        print_to_gui(msg)
        raise Exception(msg)

    def check_if_foil_is_in(self, error_message_func=None):
        self.read_current_foil(error_message_func=error_message_func)

    def read_current_foil_and_edge(self, error_message_func=None):
        element = self.read_current_foil(error_message_func=error_message_func)
        # element = 'Fe'
        edge = self.edge_dict[element]
        return element, edge

    @property
    def current_foil_and_edge(self):
        return self.read_current_foil_and_edge()

    def validate_barcode(self, input, error_message_func):
        if input in (self.barcode1, self.barcode2):
            return
        msg = f'String {input} not found in {self.name} barcodes'
        if error_message_func is not None:
            error_message_func(msg)
        print_to_gui(msg)
        raise Exception(msg)




bpm_es = FeedbackBPM('XF:08IDB-BI{BPM:ES}', name='bpm_es')
bpm_es_ioc_reset = EpicsSignal('XF:08IDB-CT{IOC:BPM:ES}:SysReset', name='bpm_es_ioc_reset')
bpm_es.append_ioc_reboot_pv(bpm_es_ioc_reset)



# camera_sp3 = BPM('XF:08IDB-BI{BPM:SP-3}', name='camera_sp3')

# camera_sp1 = BPM('XF:08IDB-BI{BPM:SP-1}', name='camera_sp1')
# camera_sp2 = CAMERA('XF:08IDB-BI{BPM:SP-2}', name='camera_sp2')
# camera_sp2 = BPM('XF:08IDB-BI{BPM:SP-2}', name='camera_sp2')



camera_sp3 = BPM('XF:08IDB-BI{BPM:SP-3}', name='camera_sp3')

camera_sp4 = CAMERA('XF:08IDB-BI{BPM:SP-4}', name='camera_sp4')
camera_sp5 = FoilCAMERA('XF:08IDB-BI{BPM:SP-5}', name='camera_sp5')
camera_sp6 = CAMERA('XF:08IDB-BI{BPM:SP-6}', name='camera_sp6')


foil_camera = camera_sp5
# foil_camera_b2 = FoilCAMERA('XF:08ID1-ES{USB-Cam:2}', name='foil_camera_b2')
#bpm_ms1 = CAMERA('XF:08IDB-BI{BPM:MS-1}', name='bpm_ms1')

for bpm in [bpm_fm, bpm_cm, bpm_bt1, bpm_bt2, bpm_es,]: #camera_sp1, camera_sp2, camera_sp3, camera_sp4]:

    bpm.read_attrs = ['stats1', 'stats2']
    bpm.image.read_attrs = ['array_data']
    bpm.stats1.read_attrs = ['total', 'centroid']
    bpm.stats2.read_attrs = ['total', 'centroid']

bpm_fm.read_attrs = ['stats1', 'stats2', 'stats3', 'stats4']
bpm_fm.stats1.read_attrs = ['total', 'centroid']
bpm_fm.stats2.read_attrs = ['total', 'centroid']
bpm_fm.stats3.read_attrs = ['total', 'centroid']
bpm_fm.stats4.read_attrs = ['total', 'centroid']

for camera in [ camera_sp1, camera_sp2, camera_sp4]:   #camera_sp3,
    bpm.read_attrs = ['stats1', 'stats2']
    bpm.image.read_attrs = ['array_data']
    #bpm.stats1.read_attrs = ['total', 'centroid']
    #bpm.stats2.read_attrs = ['total', 'centroid']



tc_mask2_4 = EpicsSignal('XF:08IDA-OP{Mir:2-CM}T:Msk2_4-I',
                         name='tc_mask2_4')
tc_mask2_3 = EpicsSignal('XF:08IDA-OP{Mir:2-CM}T:Msk2_3-I',
                         name='tc_mask2_3')


bpm_fm.stats1.kind = 'hinted'
bpm_fm.stats1.total.kind = 'hinted'

bpm_es.stats1.kind = 'hinted'
bpm_es.stats1.total.kind = 'hinted'


camera_sp1.stats1.kind = 'hinted'
camera_sp1.stats1.total.kind = 'hinted'
camera_sp1.stats1.net.kind = 'hinted'
# camera_sp1.image.kind = 'hinted'
# camera_sp1.image.array_data.kind = 'hinted'

camera_sp1.stats2.kind = 'hinted'
camera_sp1.stats2.total.kind = 'hinted'
camera_sp1.stats2.net.kind = 'hinted'

camera_sp2.stats1.kind = 'hinted'
camera_sp2.stats1.total.kind = 'hinted'
camera_sp2.stats1.net.kind = 'hinted'

camera_sp2.stats2.kind = 'hinted'
camera_sp2.stats2.total.kind = 'hinted'
camera_sp2.stats2.net.kind = 'hinted'

camera_sp3.stats1.kind = 'hinted'
camera_sp3.stats1.total.kind = 'hinted'

camera_sp3.stats2.kind = 'hinted'
camera_sp3.stats2.total.kind = 'hinted'