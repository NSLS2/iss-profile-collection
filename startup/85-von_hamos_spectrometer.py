print(ttime.ctime() + ' >>>> ' + __file__)



_default_von_hamos_config = {'crystal': 'Si',
                             'hkl': [4, 4, 0],
                             'R': 500,
                             'parking': {'motor_det_x':     -10.0,  # high limit of this motor should be at 529 mm
                                         'motor_det_th1':    0.0,
                                         'motor_det_th2':    0.0},
                             'det_offsets': {'motor_det_th1':  69.0,
                                             'motor_det_th2': -69.0},

                             'energy_calibration_uid': '',}

_short_von_hamos_config_keys = ['crystal', 'hkl', 'R', 'energy_calibration_uid']

class VonHamosGeometry(ObjectWithSettings):

    def __init__(self):
        json_path = f'{ROOT_PATH_SHARED}/settings/json/von_hamos_config.json'
        super().__init__(json_path=json_path, default_config=_default_von_hamos_config)
        self.det_L1 = _BIG_DETECTOR_ARM_LENGTH  # length of the big arm
        self.det_L2 = _SMALL_DETECTOR_ARM_LENGTH

    @property
    def det_dx(self):
        return self.det_L1 * np.cos(np.deg2rad(self.config['det_offsets']['motor_det_th1'])) - self.det_L2

    @property
    def det_h(self):
        return self.det_L1 * np.sin(np.deg2rad(self.config['det_offsets']['motor_det_th1']))

    @property
    def crystal(self):
        return self.config['crystal']

    @crystal.setter
    def crystal(self, value):
        self.config['crystal'] = value

    @property
    def hkl(self):
        return self.config['hkl']

    @hkl.setter
    def hkl(self, value):
        self.config['hkl'] = value

    @property
    def R(self):
        return self.config['R']

    @R.setter
    def R(self, value):
        self.config['R'] = value

    @property
    def energy_calibration_uid(self):
        return self.config['energy_calibration_uid']

    @energy_calibration_uid.setter
    def energy_calibration_uid(self, value):
        self.config['energy_calibration_uid'] = value

    @property
    def short_config(self):
        cur_config = self.config
        output = {}
        for k in _short_von_hamos_config_keys:
            output[k] = cur_config[k]
        return output

    # def compute_geometry_for_bragg(self, bragg_deg):
    #     bragg = np.deg2rad(bragg_deg)
    #     cr_x = self.R
    #     cr_y = self.R / np.tan(bragg)
    #     det_x = 0
    #     det_y = 2 * cr_y
    #     return cr_x, cr_y, det_x, det_y

    def compute_geometry_for_bragg(self, bragg_deg):
        bragg = np.deg2rad(bragg_deg)
        cr_x = self.R/np.sin(bragg)
        det_x = 2*self.R*(np.cos(bragg)**2)/np.sin(bragg)
        det_y = 2*self.R* np.cos(bragg)
        theta = 90-bragg_deg
        return cr_x, det_x, det_y, theta

    def e2bragg(self, energy):
        return e2bragg(energy, self.crystal, self.hkl)

    def bragg2e(self, bragg):
        return bragg2e(bragg, self.crystal, self.hkl)

    def compute_geometry_for_energy(self, energy):
        bragg = self.e2bragg(energy)
        return self.compute_geometry_for_bragg(bragg)

    def set_spectrometer_calibration(self, uid):
        self.energy_calibration_uid = uid
        self.save_current_config_to_settings()

    def reset_spectrometer_calibration(self):
        self.energy_calibration_uid = ''
        self.save_current_config_to_settings()

von_hamos_geometry = VonHamosGeometry()

class VonHamosPseudoPositioner(ISSPseudoPositioner):
    von_hamos_geometry = von_hamos_geometry



class VonHamosDetectorArm(VonHamosPseudoPositioner):
    motor_det_x = Cpt(EpicsMotor, 'XF:08IDB-OP{Stage:Aux1-Ax:Y}Mtr')
    motor_det_th1 = Cpt(EpicsMotor, 'XF:08IDB-OP{HRS:1-Det:Gon:Theta1}Mtr')  # give better names
    motor_det_th2 = Cpt(EpicsMotor, 'XF:08IDB-OP{HRS:1-Det:Gon:Theta2}Mtr')

    det_pitch = Cpt(PseudoSingle, name='det_pitch')
    det_x = Cpt(PseudoSingle, name='det_x')
    det_y = Cpt(PseudoSingle, name='det_y')

    _real = ['motor_det_x', 'motor_det_th1', 'motor_det_th2']
    _pseudo = ['det_pitch', 'det_x', 'det_y']

    @property
    def det_h(self):
        return self.von_hamos_geometry.det_h

    @property
    def det_L1(self):
        return self.von_hamos_geometry.det_L1

    @property
    def det_L2(self):
        return self.von_hamos_geometry.det_L2

    @property
    def det_dx(self):
        return self.von_hamos_geometry.det_dx

    def _forward(self, pseudo_dict):
        det_pitch, det_x, det_y = pseudo_dict['det_pitch'], pseudo_dict['det_x'], pseudo_dict['det_y']

        det_pitch_rad = np.deg2rad(det_pitch)
        _phi = np.pi - 2 * det_pitch_rad
        _sin_th1 = (self.det_h - self.det_L2 * np.sin(_phi) - det_y) / self.det_L1
        motor_det_th1 = np.arcsin(_sin_th1)
        motor_det_th2 = _phi + motor_det_th1
        motor_det_x = -self.det_dx + self.det_L1 * np.cos(motor_det_th1) - self.det_L2 * np.cos(_phi) - det_x
        motor_det_th1 = np.rad2deg(motor_det_th1)
        motor_det_th2 = -np.rad2deg(motor_det_th2)
        return {'motor_det_x':      motor_det_x   + self.von_hamos_geometry.config['parking']['motor_det_x'],
                'motor_det_th1':    motor_det_th1 + self.von_hamos_geometry.config['parking']['motor_det_th1'],
                'motor_det_th2':    motor_det_th2 + self.von_hamos_geometry.config['parking']['motor_det_th2']}

    def _inverse(self, real_dict):
        motor_det_x, motor_det_th1, motor_det_th2 = real_dict['motor_det_x'], real_dict['motor_det_th1'], real_dict['motor_det_th2']
        motor_det_x -= self.von_hamos_geometry.config['parking']['motor_det_x']
        motor_det_th1 -= self.von_hamos_geometry.config['parking']['motor_det_th1']
        motor_det_th2 -= self.von_hamos_geometry.config['parking']['motor_det_th2']

        motor_det_th2 *= -1
        det_pitch = (180 - (motor_det_th2 - motor_det_th1)) / 2
        det_x = -self.det_dx + self.det_L1 * np.cos(np.deg2rad(motor_det_th1)) - self.det_L2 * np.cos(np.deg2rad(motor_det_th2 - motor_det_th1)) - motor_det_x
        det_y = self.det_h - self.det_L1 * np.sin(np.deg2rad(motor_det_th1)) - self.det_L2 * np.sin(np.deg2rad(motor_det_th2 - motor_det_th1))
        return {'det_pitch' :   det_pitch,
                'det_x' :       det_x,
                'det_y' :       det_y}



von_hamos_det_arm = VonHamosDetectorArm(name='von_hamos_det_arm')


_von_hamos_det_arm_dict = {'motor_det_x':   {'name': von_hamos_det_arm.motor_det_x.name,   'description': 'Detector x',   'keyword': 'Det x',     'object': von_hamos_det_arm.motor_det_x,   'group': 'spectrometer'},
                           'motor_det_th1': {'name': von_hamos_det_arm.motor_det_th1.name, 'description': 'Detector th1', 'keyword': 'Det th1',   'object': von_hamos_det_arm.motor_det_th1, 'group': 'spectrometer'},
                           'motor_det_th2': {'name': von_hamos_det_arm.motor_det_th2.name, 'description': 'Detector th2', 'keyword': 'Det th2',   'object': von_hamos_det_arm.motor_det_th2, 'group': 'spectrometer'}}

motor_dictionary.update(_von_hamos_det_arm_dict)
#
#
# von_hamos_det_arm._forward({'det_pitch' : 76.82020215599975,
#                             'det_x': -50.618941215615074,
#                             'det_y': 216.1531741058653})
# von_hamos_det_arm._inverse({ 'motor_det_x': 339.2871930945466,
#                              'motor_det_th1': 27.84701046545642,
#                              'motor_det_th2': -54.2066061534569})