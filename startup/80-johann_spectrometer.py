import copy

print(ttime.ctime() + ' >>>> ' + __file__)


import numpy as np
import pandas as pd

from ophyd import (PseudoPositioner, PseudoSingle)
from ophyd.pseudopos import (pseudo_position_argument,
                             real_position_argument)

from xas.fitting import Nominal2ActualConverter
from xas.xray import bragg2e, e2bragg, crystal_reflectivity
from xas.spectrometer import compute_rowland_circle_geometry, _compute_rotated_rowland_circle_geometry

from scipy import interpolate
def extrapolate_linearly(x_in, x, y):
    if type(x_in) != np.ndarray:
        x_in = np.array(x_in)
    if len(x) == 0:
        return np.zeros(x_in.shape)
    elif len(x) == 1:
        return y * np.ones(x_in.shape)
    else:
        f = interpolate.interp1d(x, y, kind='linear', fill_value='extrapolate')
        return f(x_in)

# spectrometer detector arms
_BIG_DETECTOR_ARM_LENGTH = 550 # length of the big arm
_SMALL_DETECTOR_ARM_LENGTH = 91 # distance between the second gon and the sensitive surface of the detector

# organizing motor keys for convenience
_johann_det_arm_motor_keys = ['motor_det_x', 'motor_det_th1', 'motor_det_th2']
_johann_cr_assy_motor_keys = ['motor_cr_assy_x', 'motor_cr_assy_y']
_johann_cr_main_motor_keys = ['motor_cr_main_roll', 'motor_cr_main_yaw']
_johann_cr_aux2_motor_keys = ['motor_cr_aux2_x', 'motor_cr_aux2_y', 'motor_cr_aux2_roll', 'motor_cr_aux2_yaw']
_johann_cr_aux3_motor_keys = ['motor_cr_aux3_x', 'motor_cr_aux3_y', 'motor_cr_aux3_roll', 'motor_cr_aux3_yaw']
_johann_cr_aux4_motor_keys = ['motor_cr_aux4_x', 'motor_cr_aux4_y', 'motor_cr_aux4_roll', 'motor_cr_aux4_yaw']
_johann_cr_aux5_motor_keys = ['motor_cr_aux5_x', 'motor_cr_aux5_y', 'motor_cr_aux5_roll', 'motor_cr_aux5_yaw']

_johann_cr_all_motor_keys = (_johann_cr_assy_motor_keys +
                             _johann_cr_main_motor_keys +
                             _johann_cr_aux2_motor_keys +
                             _johann_cr_aux3_motor_keys +
                             _johann_cr_aux4_motor_keys +
                             _johann_cr_aux5_motor_keys)
_johann_spectrometer_motor_keys = (_johann_det_arm_motor_keys +
                                   _johann_cr_all_motor_keys)

# roll offsets curretnly available for the spectrometer
_allowed_roll_offsets = [2.5, 11.5, 20.5]

class RowlandCircle:
    """
    Rowland Circle (RC) class to compute motor positions for Johann geometry and manage spectrometer configuration.

    The geometry calculations are done in several steps:
    1 - set up the geometry parameters (RC radius R, side crystals' Z-offsets, and detector arm parameters)
    2 - compute theoretical positions of spectrometer elements (crystals, detectors) WRT to source (sample)
    3 - convert the element positions to motor positions
    4 - offset the crystal roll angles by roll_offset
    5 - offset the computed motor positions according to the parking position. Parking positions for motors
    correspond to nominal motor positions at specific nominal R. They are obtained via crystal prealignment.
    6 - correct the motor positions according to the Bragg registration (configured for the setup)
    7 - correct the motor positions according to the energy calibration (sample specific) # this is not actually implemented at the moment

    main attributes:
    config - main dictionary that defines the Rowland Circle parameters. Saved to settings on every update for
    persistence purposes.
    R - Rowland circle radius
    roll_offset - manually set angular offset on the crystals
    det_focus - detector shift inside (positive) or outside (negative) of the Rowland circle
    cr_aux2_z - z-offset between the main crystal and inner side crystals (aux2, aux3)
    cr_aux4_z - z-offset between the main crystal and outer side crystals (aux4, aux5)

    status attributes:
    det_focus_status - checks whether the detector position corresponds to the nominal det_focus

    auxiliary attributes:
    json_path - path for config file
    x_src - x-coordinate of the source
    y_src - y-coordinate of the source
    det_L1 - length of the big detector arm
    det_L2 - length of the small detector arm
    bragg_min - min limit for geometry calculations (does not reflect the motor limits)
    bragg_max - max limit for geometry calculations (does not reflect the motor limits)
    allowed_roll_offsets - permitted roll offsets defined by the ones available on the beamline
    gui_update_signal - PyQt signal to interact with GUIs
    """

    def __init__(self):
        """
        Boot rowland circle from config saved in settings
        """
        self.json_path = f'{ROOT_PATH_SHARED}/settings/json/johann_config_upd.json'
        self.energy_converter = None
        self.gui_update_signal = None

        self.x_src = 0
        self.y_src = 0
        self.bragg_min = 59
        self.bragg_max = 95

        self.allowed_roll_offsets = _allowed_roll_offsets

        self.det_L1 = _BIG_DETECTOR_ARM_LENGTH
        self.det_L2 = _SMALL_DETECTOR_ARM_LENGTH

        # 1000 mm values - old
        # self.cr_aux2_z = 140  # z-distance between the main and the auxiliary crystal (stack #2)
        # self.cr_aux4_z = 125 + self.cr_aux2_z  # z-distance between the main and the auxiliary crystal (stack #4)

        # 1000 mm values - new adapter plates
        # self.cr_aux2_z = 144.738  # z-distance between the main and the auxiliary crystal (stack #2)
        # self.cr_aux4_z = 131.014 + self.cr_aux2_z  # z-distance between the main and the auxiliary crystal (stack #4)

        # 500 mm values - new adapter plates
        # self.cr_aux2_z = 132.752  # z-distance between the main and the auxiliary crystal (stack #2)
        # self.cr_aux4_z = 120.921 + self.cr_aux2_z  # z-distance between the main and the auxiliary crystal (stack #4)

        self.det_focus_status = 'good'
        self.init_from_settings()

    def append_gui_update_signal(self, signal):
        """
        Set the PyQt signal that drives gui updates.

        Parameters
        ----------
        signal - PyQt signal defined in the widget.
        """
        self.gui_update_signal = signal

    # config management functions
    def load_config(self, file):
        with open(file, 'r') as f:
            config = json.loads(f.read())
        return config

    def save_current_spectrometer_config(self, file):
        with open(file, 'w') as f:
            json.dump(self.config, f)

    def save_current_spectrometer_config_to_settings(self):
        self.save_current_spectrometer_config(self.json_path)

    def init_from_settings(self):
        try:
            config = self.load_config(self.json_path)
        except Exception as e:
            print(f'Spectrometer config could not be loaded from settings. Loading the default configuration.\n\n{e}')
            config = {'R': 1000.0,
                      'crystal': 'Si',
                      'hkl': [5, 3, 1],
                      'parking': {'motor_det_x': -77.0,
                      'motor_det_th1': 0.0,
                      'motor_det_th2': 0.0,
                      'motor_cr_assy_x': -3.0001261410000097,
                      'motor_cr_assy_y': 7.181946,
                      'motor_cr_main_roll': 577.4580000000001,
                      'motor_cr_main_yaw': -185.0,
                      'motor_cr_aux2_x': 0.0,
                      'motor_cr_aux2_y': -8200.0,
                      'motor_cr_aux2_roll': 419.5590000000002,
                      'motor_cr_aux2_yaw': -1160.0,
                      'motor_cr_aux3_x': -2500.0,
                      'motor_cr_aux3_y': -8900.0,
                      'motor_cr_aux3_roll': 1495.426,
                      'motor_cr_aux3_yaw': -506.468},
                      'roll_offset': 2.5,
                      'det_offsets': {'motor_det_th1': 69.0,
                                      'motor_det_th2': -69.0},
                      'det_focus': -5,
                      'bragg_registration': {'bragg': {k : [] for k in _johann_spectrometer_motor_keys},
                                             'pos_nom' : {k : [] for k in _johann_spectrometer_motor_keys},
                                             'pos_act' : {k : [] for k in _johann_spectrometer_motor_keys}},
                       'energy_calibration': {'x_nom': [], 'x_act': [], 'n_poly': 2},
                       'enabled_crystals': {'main': True, 'aux2': True, 'aux3': True},
                       'initialized': False,
                       'energy_limits': [0, 0]}
        self.set_spectrometer_config(config)

    def set_spectrometer_config(self, config):
        # config needs a validation may be
        self.config = config
        self.update_crystal_aux_dz()
        self.compute_trajectories()
        if 'energy_calibration' in config.keys():
            config_ec = config['energy_calibration']
            if (len(config_ec['x_nom']) > 0) and (len(config_ec['x_act']) > 0) and (config_ec['n_poly'] > 0):
                self.energy_converter = Nominal2ActualConverter(config_ec['x_nom'], config_ec['x_act'], config_ec['n_poly'])

        if self.gui_update_signal is not None:
            self.gui_update_signal.emit()

    # parking functions
    def set_det_arm_parking(self, pos_dict: dict):
        """
        Set detector parking using position dictionary. Must be done when the detector is placed at nominal 90-degree
        Bragg angle (i.e. backscattering).

        Notes
        -----
        Note that parking purpose is to offset the motors during the manual alignment. Since motor_det_th1 and
        motor_det_th2 are assumed to be correctly prealigned, the correspoding fields in PARKING must be set to 0.

        At the same time, the goniometer positions at 90-degree Bragg angle are used to compute the detector
        trajectory (see det_dx and det_h) and so the input values are used to update ['det_offsets'] fields in the
        config.

        See also
        --------
        det_dx
        det_h
        """
        self.config['parking']['motor_det_x'] = pos_dict['motor_det_x']
        self.config['det_offsets']['motor_det_th1'] = pos_dict['motor_det_th1']
        self.config['det_offsets']['motor_det_th2'] = pos_dict['motor_det_th2']
        self.save_current_spectrometer_config_to_settings()

    def det_arm_parking(self):
        return (self.config['parking']['motor_det_x'],
                self.config['det_offsets']['motor_det_th1'],
                self.config['det_offsets']['motor_det_th2'])

    def set_main_crystal_parking(self, pos_dict: dict):
        """
        Set main crystal parking.

        Notes
        -----
        Due to the way parking offset is applied to the motor_cr_assy_x, the stored value equals the
        motor position minus the current R.

        Similarly, the roll offset must be subtracted from the roll motor readback. Since parking is done at
        90-degree bragg angle, the offset must be set to 2500 degrees.
        """
        self.config['parking']['motor_cr_assy_x'] = pos_dict['motor_cr_assy_x'] - self.R
        self.config['parking']['motor_cr_assy_y'] = pos_dict['motor_cr_assy_y']
        self.config['parking']['motor_cr_main_roll'] = pos_dict['motor_cr_main_roll'] - 2500
        self.config['parking']['motor_cr_main_yaw'] = pos_dict['motor_cr_main_yaw']
        self.save_current_spectrometer_config_to_settings()

    def main_crystal_parking(self, human_readable=True):
        """
        Get main crystal parking.

        Parameters
        ----------
        human_readable - offsets the motor positions so that they correspond to the nominal motor positions in the
        parking configuration (90-degree Bragg angle).
        """
        x, y, roll, yaw = (self.config['parking']['motor_cr_assy_x'],
                           self.config['parking']['motor_cr_assy_y'],
                           self.config['parking']['motor_cr_main_roll'],
                           self.config['parking']['motor_cr_main_yaw'])
        if human_readable:
            x = x + self.R
            roll = (roll + 2500) / 1000
            yaw = yaw / 1000
        return x, y, roll, yaw

    def set_aux2_crystal_parking(self, pos_dict):
        """
        Set aux2 crystal parking.

        See also
        --------
        set_main_crystal_parking
        """
        self.config['parking']['motor_cr_aux2_x'] = pos_dict['motor_cr_aux2_x']
        self.config['parking']['motor_cr_aux2_y'] = pos_dict['motor_cr_aux2_y']
        self.config['parking']['motor_cr_aux2_roll'] = pos_dict['motor_cr_aux2_roll'] - 2500
        self.config['parking']['motor_cr_aux2_yaw'] = pos_dict['motor_cr_aux2_yaw']
        self.save_current_spectrometer_config_to_settings()

    def aux2_crystal_parking(self, human_readable=True):
        """
        Get aux2 crystal parking.

        See also
        --------
        main_crystal_parking
        """
        x, y, roll, yaw = (self.config['parking']['motor_cr_aux2_x'],
                           self.config['parking']['motor_cr_aux2_y'],
                           self.config['parking']['motor_cr_aux2_roll'],
                           self.config['parking']['motor_cr_aux2_yaw'])
        if human_readable:
            x /= 1000
            y /= 1000
            roll = (roll + 2500) / 1000
            yaw /= 1000
        return x, y, roll, yaw

    def set_aux3_crystal_parking(self, pos_dict):
        """
        Set aux3 crystal parking.

        See also
        --------
        set_main_crystal_parking
        """
        self.config['parking']['motor_cr_aux3_x'] = pos_dict['motor_cr_aux3_x']
        self.config['parking']['motor_cr_aux3_y'] = pos_dict['motor_cr_aux3_y']
        self.config['parking']['motor_cr_aux3_roll'] = pos_dict['motor_cr_aux3_roll'] - 2500
        self.config['parking']['motor_cr_aux3_yaw'] = pos_dict['motor_cr_aux3_yaw']
        self.save_current_spectrometer_config_to_settings()

    def aux3_crystal_parking(self, human_readable=True):
        """
        Get aux3 crystal parking.

        See also
        --------
        main_crystal_parking
        """
        x, y, roll, yaw = (self.config['parking']['motor_cr_aux3_x'],
                           self.config['parking']['motor_cr_aux3_y'],
                           self.config['parking']['motor_cr_aux3_roll'],
                           self.config['parking']['motor_cr_aux3_yaw'])
        if human_readable:
            x /= 1000
            y /= 1000
            roll = (roll + 2500) / 1000
            yaw /= 1000
        return x, y, roll, yaw

    def set_aux4_crystal_parking(self, pos_dict):
        """
        Set aux4 crystal parking.

        See also
        --------
        set_main_crystal_parking
        """
        self.config['parking']['motor_cr_aux4_x'] = pos_dict['motor_cr_aux4_x']
        self.config['parking']['motor_cr_aux4_y'] = pos_dict['motor_cr_aux4_y']
        self.config['parking']['motor_cr_aux4_roll'] = pos_dict['motor_cr_aux4_roll'] - 2500
        self.config['parking']['motor_cr_aux4_yaw'] = pos_dict['motor_cr_aux4_yaw']
        self.save_current_spectrometer_config_to_settings()

    def aux4_crystal_parking(self, human_readable=True):
        """
        Get aux4 crystal parking.

        See also
        --------
        main_crystal_parking
        """
        x, y, roll, yaw = (self.config['parking']['motor_cr_aux4_x'],
                           self.config['parking']['motor_cr_aux4_y'],
                           self.config['parking']['motor_cr_aux4_roll'],
                           self.config['parking']['motor_cr_aux4_yaw'])
        if human_readable:
            x /= 1000
            y /= 1000
            roll = (roll + 2500) / 1000
            yaw /= 1000
        return x, y, roll, yaw

    def set_aux5_crystal_parking(self, pos_dict):
        """
        Set aux5 crystal parking.

        See also
        --------
        set_main_crystal_parking
        """
        self.config['parking']['motor_cr_aux5_x'] = pos_dict['motor_cr_aux5_x']
        self.config['parking']['motor_cr_aux5_y'] = pos_dict['motor_cr_aux5_y']
        self.config['parking']['motor_cr_aux5_roll'] = pos_dict['motor_cr_aux5_roll'] - 2500
        self.config['parking']['motor_cr_aux5_yaw'] = pos_dict['motor_cr_aux5_yaw']
        self.save_current_spectrometer_config_to_settings()

    def aux5_crystal_parking(self, human_readable=True):
        """
        Get aux5 crystal parking.

        See also
        --------
        main_crystal_parking
        """
        x, y, roll, yaw = (self.config['parking']['motor_cr_aux5_x'],
                           self.config['parking']['motor_cr_aux5_y'],
                           self.config['parking']['motor_cr_aux5_roll'],
                           self.config['parking']['motor_cr_aux5_yaw'])
        if human_readable:
            x /= 1000
            y /= 1000
            roll = (roll + 2500) / 1000
            yaw /= 1000
        return x, y, roll, yaw

    # geometry-related methods and attributes
    @property
    def det_dx(self):
        """
        X-projection of the vector between the pivot point of the big detector goniometer (motor_det_th1) and the
        center of the detector face.
        """
        return self.det_L1 * np.cos(np.deg2rad(self.config['det_offsets']['motor_det_th1'])) - self.det_L2

    @property
    def det_h(self):
        """
        Y-projection of the vector between the pivot point of the big detector goniometer (motor_det_th1) and the
        center of the detector face.
        """
        return self.det_L1 * np.sin(np.deg2rad(self.config['det_offsets']['motor_det_th1']))

    @property
    def det_focus(self):
        """
        Detector shift in or out of the Rowland circle.
        When computed, also updates self.det_focus_status to check if the config value represents the reality.
        """
        det_focus_est = self.compute_motor_position('motor_det_x', 90, nom2act=False) - self.config['parking']['motor_det_x']
        if not np.isclose(det_focus_est, self.config['det_focus'], atol=0.1):
            self.det_focus_status = 'bad'
        else:
            self.det_focus_status = 'good'
            # print_to_gui('WARNING: estimated det_focus differs from the config value by >0.1 mm')
        return self.config['det_focus']

    @det_focus.setter
    def det_focus(self, value):
        """
        Changing the detector focus updates the config and the detetor trajectory.
        """
        self.update_nominal_trajectory_for_detector(value)
        self.compute_trajectory_correction()
        self.config['det_focus'] = value
        self.save_current_spectrometer_config_to_settings()

    @property
    def R_parking(self):
        return self.config['parking']['R']

    @R_parking.setter
    def R_parking(self, value):
        self.config['parking']['R'] = value
        self.update_crystal_aux_dz()
        self.save_current_spectrometer_config_to_settings()

    @property
    def cr_aux20_dz(self):
        return self.config['cr_aux20_dz']

    @property
    def cr_aux42_dz(self):
        return self.config['cr_aux42_dz']

    @property
    def cr_aux2_z(self):
        return self.cr_aux20_dz

    @property
    def cr_aux4_z(self):
        return self.cr_aux20_dz + self.cr_aux42_dz

    def update_crystal_aux_dz(self):
        if self.R_parking == 500:  # 500 mm values - new adapter plates
            self.config['cr_aux20_dz'] = 132.752  # z-distance between the main and the aux2 (stack #2)
            self.config['cr_aux42_dz'] = 120.921  # z-distance between the aux2 and the aux4 (stack #4)

        elif self.R_parking == 1000:  # 1000 mm values - new adapter plates
            self.config['cr_aux20_dz'] = 144.738  # z-distance between the main and the aux2 (stack #2)
            self.config['cr_aux42_dz'] = 131.014  # z-distance between the aux2 and the aux4 (stack #4)
        else:
            print('Nominal Rowland cirle radius should be either 1000 or 500 mm.')

    @property
    def R(self):
        """
        Rowland circle radius R stored in the config.
        """
        return self.config['R']

    @R.setter
    def R(self, value):
        """
        Changing Rowland circle radius R also updates the spectrometer trajectories and the config.
        """
        self.config['R'] = value
        self.compute_trajectories()
        self.save_current_spectrometer_config_to_settings()

    @property
    def crystal(self):
        """
        Current crystal material stored in the config.
        """
        return self.config['crystal']

    @crystal.setter
    def crystal(self, value):
        self.config['crystal'] = value

    @property
    def hkl(self):
        """
        Current crystal miller indices stored in the config.
        """
        return self.config['hkl']

    @hkl.setter
    def hkl(self, value):
        self.config['hkl'] = value

    @property
    def roll_offset(self):
        """
        Current manual angular roll offset.
        """
        return self.config['roll_offset']

    @roll_offset.setter
    def roll_offset(self, value):
        """
        Changing the angular offset updates the config and trajectories.
        The offset can only take specific values defined in the allowed_roll_offsets
        """
        assert value in self.allowed_roll_offsets, f'roll_offset value must be equal to one of {self.allowed_roll_offsets}'
        self.config['roll_offset'] = value
        self.compute_trajectories()
        self.save_current_spectrometer_config_to_settings()

    @property
    def enabled_crystals(self):
        """
        Currently enabled crystals. Enabled crystals are active during scans, i.e. move during scans. Disabled
        crystals do not move at all.
        """
        return self.config['enabled_crystals']

    @property
    def enabled_crystals_list(self):
        """
        List keys for currently enabled crystals. Enabled crystals are active during scans, i.e. move during scans. Disabled
        crystals do not move at all.
        """
        return [k for k, v in self.config['enabled_crystals'].items() if v]

    def enable_crystal(self, crystal_key: str, enable: bool):
        """
        Enable or disable one of the spectrometer crystals.

        Parameters
        ----------
        crystal_key - spectrometer crystal (main, aux2, aux3, aux4, aux5)
        enable - enable flag. True - enabled, False - disabled.
        """
        self.enabled_crystals[crystal_key] = enable
        self.save_current_spectrometer_config_to_settings()

    @property
    def initialized(self):
        """
        True/False flag indicating whether the spectrometer was initialized for safety. At the beginning of a new
        user experiment the initialized flag is set to False. Once the instrument is aligned, the flag is set to True.
        """
        return self.config['initialized']

    @initialized.setter
    def initialized(self, value):
        """
        Updates the initialized flag and sets it to settings.
        """
        self.config['initialized'] = value
        self.save_current_spectrometer_config_to_settings()

    @property
    def energy_limits(self):
        """
        Set energy limits on the spectrometer.
        """
        return self.config['energy_limits']

    @energy_limits.setter
    def energy_limits(self, value):
        """
        Update the spectrometer energy limits.
        """
        self.config['energy_limits'] = value
        self.save_current_spectrometer_config_to_settings()

    @property
    def alignment_data(self):
        """
        Data with spectrometer alignment scan uids and analysis results. Stored for metadata tracking purposes.
        """
        if 'alignment_data' not in self.config.keys():
            self.config['alignment_data'] = []
        return self.config['alignment_data']

    @property
    def fly_calibration_dict(self):
        """
        dictionary with data/conversion coefficients for converting crystal roll positions into energy during fly scanning
        """
        if 'fly_calibration_dict' not in self.config.keys():
            self.config['fly_calibration_dict'] = {'data': [], 'LUT': None}
        return self.config['fly_calibration_dict']

    def reset_fly_calibration_dict(self):
        self.fly_calibration_dict['data'] = []
        self.fly_calibration_dict['LUT'] = None
        self.save_current_spectrometer_config_to_settings()

    def _compute_nominal_trajectory(self, npt=1000):
        """
        Compute nominal trajectory for spectrometer motors as a function of Bragg angle.
        This method takes care of steps 2, 3, and 4a of the geometry calculation workflow (see the class docstring).

        Parameters
        ----------
        npt - number of points for Bragg angle array.
        """
        braggs = np.linspace(self.bragg_min, self.bragg_max, npt-1)
        braggs = np.sort(np.hstack((braggs, 90)))

        # pseudos
        det_x = np.zeros(npt)
        det_y = np.zeros(npt)

        cr_main_x = np.zeros(npt)
        cr_main_y = np.zeros(npt)
        cr_main_roll = braggs.copy()
        cr_main_yaw = np.zeros(npt)

        cr_aux2_x = np.zeros(npt)
        cr_aux2_y = np.zeros(npt)
        cr_aux2_roll = np.zeros(npt)
        cr_aux2_yaw = np.zeros(npt)

        cr_aux4_x = np.zeros(npt)
        cr_aux4_y = np.zeros(npt)
        cr_aux4_roll = np.zeros(npt)
        cr_aux4_yaw = np.zeros(npt)

        for i, bragg in enumerate(braggs):
            cr_main_x[i], cr_main_y[i], det_x[i], det_y[i] = \
                compute_rowland_circle_geometry(self.x_src, self.y_src, self.R, bragg, 0)
            cr_aux2_x[i], cr_aux2_y[i], cr_aux2_roll[i], cr_aux2_yaw[i] = \
                _compute_rotated_rowland_circle_geometry(cr_main_x[i], cr_main_y[i], det_x[i], det_y[i], bragg, self.cr_aux2_z)
            cr_aux4_x[i], cr_aux4_y[i], cr_aux4_roll[i], cr_aux4_yaw[i] = \
                _compute_rotated_rowland_circle_geometry(cr_main_x[i], cr_main_y[i], det_x[i], det_y[i], bragg,
                                                         self.cr_aux4_z)

        # reals
        # detector
        motor_det_x, motor_det_th1, motor_det_th2 = self._compute_trajectory_for_detector(braggs, det_x, det_y)

        # main crystal
        motor_cr_assy_x = -cr_main_x
        motor_cr_assy_y = cr_main_y
        motor_cr_main_roll = (cr_main_roll - 90 + self.config['roll_offset']) * 1000
        motor_cr_main_yaw = cr_main_yaw * 1000

        # aux crystal2
        _cr_aux2_yaw_0 = np.arcsin(self.cr_aux2_z / self.R)
        _cr_aux2_dx_0 = (-self.R * np.cos(_cr_aux2_yaw_0)) + self.R
        _cr_aux2_dx = cr_aux2_x - cr_main_x
        motor_cr_aux2_x = -(_cr_aux2_dx - _cr_aux2_dx_0) * 1000
        motor_cr_aux2_y = cr_aux2_y * 1000
        motor_cr_aux2_roll = (cr_aux2_roll - 90 + self.config['roll_offset']) * 1000
        motor_cr_aux2_yaw = (cr_aux2_yaw - np.rad2deg(_cr_aux2_yaw_0)) * 1000

        # aux crystal3
        motor_cr_aux3_x = motor_cr_aux2_x.copy()
        motor_cr_aux3_y = motor_cr_aux2_y.copy()
        motor_cr_aux3_roll = motor_cr_aux2_roll.copy()
        motor_cr_aux3_yaw = -motor_cr_aux2_yaw.copy()

        # aux crystal4
        _cr_aux4_yaw_0 = np.arcsin(self.cr_aux4_z / self.R)
        _cr_aux4_dx_0 = (-self.R * np.cos(_cr_aux4_yaw_0)) + self.R
        _cr_aux4_dx = cr_aux4_x - cr_main_x
        motor_cr_aux4_x = -(_cr_aux4_dx - _cr_aux4_dx_0) * 1000
        motor_cr_aux4_y = cr_aux4_y * 1000
        motor_cr_aux4_roll = (cr_aux4_roll - 90 + self.config['roll_offset']) * 1000
        motor_cr_aux4_yaw = (cr_aux4_yaw - np.rad2deg(_cr_aux4_yaw_0)) * 1000

        # aux crystal5
        motor_cr_aux5_x = motor_cr_aux4_x.copy()
        motor_cr_aux5_y = motor_cr_aux4_y.copy()
        motor_cr_aux5_roll = motor_cr_aux4_roll.copy()
        motor_cr_aux5_yaw = -motor_cr_aux4_yaw.copy()

        return pd.DataFrame(
               {'bragg' :               braggs,
                'det_x' :               det_x,
                'det_y' :               det_y,
                'cr_main_x' :           cr_main_x,
                'cr_main_y' :           cr_main_y,
                'cr_main_roll' :        cr_main_roll,
                'cr_main_yaw' :         cr_main_yaw,
                'cr_aux2_x':            cr_aux2_x,
                'cr_aux2_y':            cr_aux2_y,
                'cr_aux2_roll':         cr_aux2_roll,
                'cr_aux2_yaw':          cr_aux2_yaw,
                'motor_det_x':          motor_det_x,
                'motor_det_th1':        motor_det_th1,
                'motor_det_th2':        motor_det_th2,
                'motor_cr_assy_x' :     motor_cr_assy_x,
                'motor_cr_assy_y':      motor_cr_assy_y,
                'motor_cr_main_roll':   motor_cr_main_roll,
                'motor_cr_main_yaw':    motor_cr_main_yaw,
                'motor_cr_aux2_x' :     motor_cr_aux2_x,
                'motor_cr_aux2_y':      motor_cr_aux2_y,
                'motor_cr_aux2_roll':   motor_cr_aux2_roll,
                'motor_cr_aux2_yaw':    motor_cr_aux2_yaw,
                'motor_cr_aux3_x':      motor_cr_aux3_x,
                'motor_cr_aux3_y':      motor_cr_aux3_y,
                'motor_cr_aux3_roll':   motor_cr_aux3_roll,
                'motor_cr_aux3_yaw':    motor_cr_aux3_yaw,
                'motor_cr_aux4_x':      motor_cr_aux4_x,
                'motor_cr_aux4_y':      motor_cr_aux4_y,
                'motor_cr_aux4_roll':   motor_cr_aux4_roll,
                'motor_cr_aux4_yaw':    motor_cr_aux4_yaw,
                'motor_cr_aux5_x':      motor_cr_aux5_x,
                'motor_cr_aux5_y':      motor_cr_aux5_y,
                'motor_cr_aux5_roll':   motor_cr_aux5_roll,
                'motor_cr_aux5_yaw':    motor_cr_aux5_yaw})

    def _compute_trajectory_for_detector(self, braggs, det_x, det_y, det_focus=0):
        """
        Compute trajectory for the detector.

        Parameters
        ----------
        braggs - array of Bragg angles in degrees
        det_x - x coordinate of the detector sensor center
        det_y - y coordinate of the detector sensor center
        det_focus - detector shift in/out of the Rowland circle
        """

        _det_bragg_rad = np.deg2rad(braggs)
        det_x = det_x.copy() - det_focus * np.cos(np.pi - 2 * _det_bragg_rad)
        det_y = det_y.copy() - det_focus * np.sin(np.pi - 2 * _det_bragg_rad)

        _phi = np.pi - 2 * _det_bragg_rad
        _sin_th1 = (self.det_h - self.det_L2 * np.sin(_phi) - det_y) / self.det_L1
        motor_det_th1 = np.arcsin(_sin_th1)
        motor_det_th2 = _phi + motor_det_th1
        motor_det_x = -self.det_dx + self.det_L1 * np.cos(motor_det_th1) - self.det_L2 * np.cos(_phi) - det_x
        motor_det_th1 = np.rad2deg(motor_det_th1)
        motor_det_th2 = -np.rad2deg(motor_det_th2)
        return motor_det_x, motor_det_th1, motor_det_th2

    def update_nominal_trajectory_for_detector(self, det_focus, force_update=False):
        """
        Update detector trajectory given det_focus.

        Parameters:
            det_focus - detector shift in/out of the Rowland circle.
            force_update - new trajectory is computed if True. If False, then the method checks if the new det_focus
            is sufficiently different from the current one.
        """
        if force_update or (not np.isclose(det_focus, self.det_focus, atol=1e-4)):
            # self.det_focus = det_focus
            motor_det_x, motor_det_th1, motor_det_th2 = self._compute_trajectory_for_detector(self.traj_nom['bragg'],
                                                                                              self.traj_nom['det_x'],
                                                                                              self.traj_nom['det_y'],
                                                                                              det_focus=det_focus)
            self.traj_nom['motor_det_x'] = motor_det_x
            self.traj_nom['motor_det_th1'] = motor_det_th1
            self.traj_nom['motor_det_th2'] = motor_det_th2


    def compute_nominal_trajectory(self, npt=250):
        """
        Wrapper method for _compute_nominal_trajectory which was retained for historical reasons.
        """
        self.traj_nom = self._compute_nominal_trajectory(npt=npt)

    # methods for computing parking offsets
    def get_motor_parking_offset(self, motor_key):
        """
        Compute parking offset for a motor.

        Parameters:
            motor_key - spectrometer motor key.

        Returns:
            parking offset value.

        Notes:
            This method performs step 5 of the geometry calculation workflow.
        """
        if motor_key in self.config['parking'].keys():
            pos0 = self.config['parking'][motor_key]
            delta = self.get_motor_parking_offset_correction(motor_key)  # additional correction due to change in R
            pos0 += delta
        else:
            pos0 = 0
        return pos0

    def get_motor_parking_offset_correction(self, motor_key):
        """
        Compute motor parking position correction based on current Rowland circle radius.

        Parameters:
            motor_key - spectrometer motor key.

        Returns:
            parking offset correction factor.

        Notes:
            Parking is performed for a specific nominal Rowland circle radius R, usually 1000 or 500 mm. During
            alignment R is varied, which results in the incorrect parking positions of the crystals. This method
            computes correction factors for parking offsets using current R.
        """
        if not motor_key.startswith('motor_cr_'):
            return 0

        if motor_key in ['motor_cr_assy_x', 'motor_cr_assy_y', 'motor_cr_main_yaw', 'motor_cr_main_roll',
                         'motor_cr_aux2_roll', 'motor_cr_aux2_y',
                         'motor_cr_aux3_y', 'motor_cr_aux3_roll',
                         'motor_cr_aux4_roll', 'motor_cr_aux4_y',
                         'motor_cr_aux5_roll', 'motor_cr_aux5_y']:
            return 0

        R1 = self.config['parking']['R']
        R2 = self.R

        cr_main_x1, cr_main_y1, det_x1, det_y1 = compute_rowland_circle_geometry(self.x_src, self.y_src, R1, 90, 0)
        cr_main_x2, cr_main_y2, det_x2, det_y2 = compute_rowland_circle_geometry(self.x_src, self.y_src, R2, 90, 0)

        # if motor_key == 'motor_cr_assy_x':
        #     return cr_main_x2 - cr_main_x1

        cr_aux2_x1, cr_aux2_y1, cr_aux2_roll1, cr_aux2_yaw1 = \
            _compute_rotated_rowland_circle_geometry(cr_main_x1, cr_main_y1, det_x1, det_y1, 90, self.cr_aux2_z)
        cr_aux2_x2, cr_aux2_y2, cr_aux2_roll2, cr_aux2_yaw2 = \
            _compute_rotated_rowland_circle_geometry(cr_main_x2, cr_main_y2, det_x2, det_y2, 90, self.cr_aux2_z)

        if motor_key in ['motor_cr_aux2_x', 'motor_cr_aux3_x']:
            return -((cr_aux2_x2 - cr_aux2_x1) - (cr_main_x2 - cr_main_x1)) * 1000

        if motor_key == 'motor_cr_aux2_yaw':
            return (cr_aux2_yaw2 - cr_aux2_yaw1) * 1000
        elif motor_key == 'motor_cr_aux3_yaw':
            return (cr_aux2_yaw1 - cr_aux2_yaw2) * 1000

        cr_aux4_x1, cr_aux4_y1, cr_aux4_roll1, cr_aux4_yaw1 = \
            _compute_rotated_rowland_circle_geometry(cr_main_x1, cr_main_y1, det_x1, det_y1, 90, self.cr_aux4_z)
        cr_aux4_x2, cr_aux4_y2, cr_aux4_roll2, cr_aux4_yaw2 = \
            _compute_rotated_rowland_circle_geometry(cr_main_x2, cr_main_y2, det_x2, det_y2, 90, self.cr_aux4_z)

        if motor_key in ['motor_cr_aux4_x', 'motor_cr_aux5_x']:
            return -((cr_aux4_x2 - cr_aux4_x1) - (cr_main_x2 - cr_main_x1)) * 1000

        if motor_key == 'motor_cr_aux4_yaw':
            return (cr_aux4_yaw2 - cr_aux4_yaw1) * 1000
        elif motor_key == 'motor_cr_aux5_yaw':
            return (cr_aux4_yaw1 - cr_aux4_yaw2) * 1000

    # bragg registration methods
    def register_bragg(self, bragg_act, motor_pos_dict):
        '''
        Register observed (actual) Bragg angle for given real motor positions.
        The method updates the spectrometer motor trajectories.

        Parameters:
             bragg_act - observed (actual) Bragg angle obtained from calibration
             motor_pos_dict - real motor position dictionary

        Notes:
            The registration is done on the individual motor basis and stored in config field 'bragg_registration'.
        '''
        pos_nom = {}
        for motor_key in motor_pos_dict.keys():
            print(motor_key)
            pos_nom[motor_key] = self.compute_motor_position(motor_key, bragg_act, nom2act=False)
        pos_act = {**motor_pos_dict}
        for motor_key in pos_act.keys():
            # self.converter_nom2act[motor_key].append_point(pos_nom[motor_key], pos_act[motor_key])
            self.config['bragg_registration']['bragg'][motor_key].append(bragg_act)
            self.config['bragg_registration']['pos_nom'][motor_key].append(pos_nom[motor_key])
            self.config['bragg_registration']['pos_act'][motor_key].append(pos_act[motor_key])

        self.compute_trajectory_correction()
        self.save_current_spectrometer_config_to_settings()

    def register_energy(self, energy_act, motor_pos_dict):
        """
        Register observed (actual) energy for given real motor positions.

        See also:
            register_bragg
        """
        bragg_act = self.e2bragg(energy_act)
        self.register_bragg(bragg_act, motor_pos_dict)

    def reset_bragg_registration(self):
        """
        Reset the bragg registration.
        """
        self.config['bragg_registration'] = {'bragg' :  {k: [] for k in _johann_spectrometer_motor_keys},
                                             'pos_nom': {k: [] for k in _johann_spectrometer_motor_keys},
                                             'pos_act': {k: [] for k in _johann_spectrometer_motor_keys}}
        self.compute_trajectory_correction()

    def compute_trajectory_correction(self):
        """
        Spectrometer motor trajectory correction based on the bragg registration.

        Notes:
            This method performs step 6 of the geometry calculation workflow.
        """
        self.traj_delta = {}
        for motor_key in _johann_spectrometer_motor_keys:
            pos_act = np.array(self.config['bragg_registration']['pos_act'][motor_key])
            pos_nom = np.array(self.config['bragg_registration']['pos_nom'][motor_key])
            delta =  pos_act - pos_nom
            bragg = np.array(self.config['bragg_registration']['bragg'][motor_key])
            bragg_in = self.traj_nom['bragg']
            self.traj_delta[motor_key] = extrapolate_linearly(bragg_in, bragg, delta)
            # self.traj[motor_key] = self.traj[motor_key] + self.traj_delta[motor_key]
        self.traj_delta = pd.DataFrame(self.traj_delta)
        self.traj = self.traj_nom.copy()
        self.traj[self.traj_delta.columns] = self.traj[self.traj_delta.columns] + self.traj_delta

    # methods for computing motor positions
    def compute_trajectories(self):
        """
        Compute motor trajectories for current spectrometer configuration.
        """
        self.compute_nominal_trajectory()
        self.update_nominal_trajectory_for_detector(self.det_focus, force_update=True)
        self.compute_trajectory_correction()

    # spectrometer calibration functions
    def set_spectrometer_calibration(self, energy_nom, energy_act, n_poly=2):
        """
        Set energy calibration for the spectrometer.

        Parameters:
            energy_nom - nominal energy array
            energy_act - actual (observed) energy array
            n_poly - polynom order to fit the energy conversion
        """
        self.config['energy_calibration'] = {'x_nom' : energy_nom.tolist(), 'x_act' : energy_act.tolist(), 'n_poly' : n_poly}
        self.energy_converter = Nominal2ActualConverter(energy_nom, energy_act, n_poly=n_poly)
        self.save_current_spectrometer_config_to_settings()

    def reset_spectrometer_calibration(self):
        """
        Reset the energy calibration.
        """
        self.config['energy_calibration'] = {'x_nom': [], 'x_act': [], 'n_poly': 2}
        self.energy_converter = None
        self.save_current_spectrometer_config_to_settings()

    def reset_alignment_data(self):
        """
        Reset alignment data dictionary
        """
        self.config['alignment_data'] = []
        self.save_current_spectrometer_config_to_settings()

    def save_alignment_data_to_settings(self):
        self.save_current_spectrometer_config_to_settings()

    # motor position calculation functions
    def plot_motor_pos_vs_bragg(self, motor_key, fignum=1):
        """
        Plot motor position as a function of bragg angle.

        Parameters:
            motor_key - spectrometer motor key;
            fignum - figure number for the plot.
        """
        bragg = self.traj['bragg']
        pos = self.compute_motor_position(motor_key, bragg)
        plt.figure(fignum, clear=True)
        plt.plot(bragg, pos)

    def _compute_motor_position(self, motor_key: str, bragg: float, nom2act=True):
        """
        Compute motor position for a given Bragg angle.

        Parameters:
            motor_key - spectrometer motor key;
            bragg - bragg angle in degrees;
            nom2act - accounts for Bragg registration if True; returns nominal values if False.

        Returns:
            motor position
        """
        if nom2act:
            pos = np.interp(bragg, self.traj['bragg'], self.traj[motor_key])
        else:
            pos = np.interp(bragg, self.traj_nom['bragg'], self.traj_nom[motor_key])
        pos0 = self.get_motor_parking_offset(motor_key)
        pos += pos0
        return pos

    def compute_motor_position(self, motor_keys, bragg, nom2act=True):
        """
        Wrapper method to compute positions of multiple motors.

        Parameters:
            motor_keys - list of spectrometer motor keys;
            bragg - bragg angle in degrees;
            nom2act - accounts for Bragg registration if True; returns nominal values if False.

        Returns:
            dictionary with motor_key/position key/value pairs.
        """
        if type(motor_keys) == str:
            return self._compute_motor_position(motor_keys, bragg, nom2act=nom2act)
        else:
            output = {}
            for motor_key in motor_keys:
                output[motor_key] = self._compute_motor_position(motor_key, bragg, nom2act=nom2act)
            return output

    def compute_motor_position_from_energy(self, motor_keys, energy, nom2act=True, use_energy_calibration=True):
        """
        Compute motor position for given energy.

        Parameters:
            motor_keys - list of spectrometer motor keys;
            energy - energy in eV;
            nom2act - accounts for Bragg registration if True; returns nominal values if False;
            use_energy_calibration - accounts for energy calibration if True AND if energy calibration was set. Skips
            the energy calibration if false.
        """
        if use_energy_calibration and (self.energy_converter is not None):
            energy = self.energy_converter.act2nom(energy)
        bragg = e2bragg(energy, self.crystal, self.hkl)
        return self.compute_motor_position(motor_keys, bragg, nom2act=nom2act)

    def compute_bragg_from_motor(self, motor_key: str, pos: float, nom2act=True):
        """
        Compute Bragg angle from the motor position.

        Parameters:
            motor_key - spectrometer motor key;
            pos - motor position;
            nom2act - accounts for Bragg registration if True; use nominal values if False.

        Returns:
            Derived Bragg angle value in degrees.

        Notes:
            This method should be used for motors that directly correlate with Bragg angle, i.e. crystal rolls and
            detector goniometers.
        """
        pos0 = self.get_motor_parking_offset(motor_key)

        if nom2act:
            bragg = np.interp(pos - pos0, self.traj[motor_key], self.traj['bragg'])
        else:
            bragg = np.interp(pos - pos0, self.traj_nom[motor_key], self.traj_nom['bragg'])

        if bragg > 90:
            bragg = 180 - bragg
        return bragg

    def compute_energy_from_motor(self, motor_key, pos, nom2act=True, use_energy_calibration=True):
        """
        Compute energy from motor position.

        Parameters:
            motor_key - spectrometer motor key;
            pos - motor position;
            nom2act - accounts for Bragg registration if True; use nominal values if False;
            use_energy_calibration - accounts for energy calibration if True AND if energy calibration was set. Skips
            the energy calibration if False.

        Returns:
            Derived energy in eV.

        Notes:
            This method should be used for motors that directly correlate with Bragg angle, i.e. crystal rolls and
            detector goniometers.
        """
        bragg = self.compute_bragg_from_motor(motor_key, pos, nom2act=nom2act)
        energy = bragg2e(bragg, self.crystal, self.hkl)
        if use_energy_calibration and (self.energy_converter is not None):
            energy = self.energy_converter.nom2act(energy)
        return energy

    def compute_bragg_from_motor_dict(self, motor_pos_dict, nom2act=True):
        """
        Wrapper method to compute Bragg angles for a set of motors.

        Parameters:
            motor_pos_dict - a dictionary with motor_key/position as key/value pairs.
            nom2act - accounts for Bragg registration if True; use nominal values if False;

        Returns:
            a dictionary with motor_key/Bragg angle as key/value pairs.

        See also:
            compute_bragg_from_motor
        """
        output = {}
        for motor_key, motor_pos in motor_pos_dict.items():
            output[motor_key] = self._compute_motor_position(motor_key, motor_pos, nom2act=nom2act)

    # auxilliary methods
    def e2bragg(self, energy):
        """
        Compute energy for a given Bragg angle.

        Parameters:
            energy - energy in eV;

        Returns:
            Bragg angle in degrees.
        """
        return e2bragg(energy, self.crystal, self.hkl)

    def bragg2e(self, bragg):
        """
        Compute Bragg angle for a given energy.

        Parameters:
            bragg - Bragg angle in degrees;

        Returns:
            energy - energy in eV.
        """
        return bragg2e(bragg, self.crystal, self.hkl)

    def e2reflectivity(self, energy):
        """
        Compute theoretical crystal reflectivity for a given energy.

        Parameters:
            energy - energy in eV;

        Returns:
            Integrated reflectivity.
        """
        bragg = self.e2bragg(energy)
        return crystal_reflectivity(self.crystal, self.hkl, bragg, energy)

    def suggest_roll_offset(self, bragg_target, roll_range=5):
        """
        Suggest a roll angle offset value for a given Bragg angle. Favors the smaller offsets.

        Parameters:
            bragg_target - Bragg angle in degrees;
            roll_range - roll motor motion range (+/- 5 degrees)

        Returns:
            Roll offset value to use for a given bragg agnle

        """
        offsets = np.array(self.allowed_roll_offsets)
        offset_braggs = 90 - offsets
        options = np.isclose(bragg_target, offset_braggs, atol=roll_range)
        return offsets[options][0]

    def convert_energy_trajectory_to_bragg(self, central_energy, relative_trajectory, crystals):
        relative_trajectory_roll = {}
        if self.fly_calibration_dict['LUT'] is None:
            print_to_gui('Fly scanning calibration is not defined. Will convert energy to motor positions based off the crystal kind.', tag='Spectrometer', add_timestamp=True)
            _rel_traj = {'positions': [], 'durations': relative_trajectory['durations']}
            central_bragg = self.e2bragg(central_energy)
            for delta_energy in relative_trajectory['positions']:
                position_bragg = self.e2bragg(central_energy + delta_energy)
                delta_roll = (position_bragg - central_bragg) * 1000
                _rel_traj['positions'].append(delta_roll)

            for crystal in crystals:
                relative_trajectory_roll[crystal] = copy.deepcopy(_rel_traj)

        else:
            LUT = self.fly_calibration_dict['LUT']
            for crystal in crystals:
                _roll = LUT[crystal]
                _energy = LUT['energy']
                e2roll_converter = Nominal2ActualConverter(_energy, _roll)
                central_roll = e2roll_converter.nom2act(central_energy)
                _rel_traj = {'positions': [], 'durations': relative_trajectory['durations']}
                for delta_energy in relative_trajectory['positions']:
                    position_roll = e2roll_converter.nom2act(central_energy + delta_energy)
                    delta_roll = position_roll - central_roll
                    _rel_traj['positions'].append(delta_roll)
                relative_trajectory_roll[crystal] = _rel_traj

        return relative_trajectory_roll



    # def append_energy_converter(self, ec):
    #     self.energy_converter = ec

rowland_circle = RowlandCircle()


# R1 = 1000
# R2 = 990
#
# cr_main_x1, cr_main_y1, det_x1, det_y1 = compute_rowland_circle_geometry(0, 0, R1, 90, 0)
# cr_main_x2, cr_main_y2, det_x2, det_y2 = compute_rowland_circle_geometry(0, 0, R2, 90, 0)
#
# cr_aux2_x1, cr_aux2_y1, cr_aux2_roll1, cr_aux2_yaw1 = \
#     _compute_rotated_rowland_circle_geometry(cr_main_x1, cr_main_y1, det_x1, det_y1, 90, 140)
# cr_aux2_x2, cr_aux2_y2, cr_aux2_roll2, cr_aux2_yaw2 = \
#     _compute_rotated_rowland_circle_geometry(cr_main_x2, cr_main_y2, det_x2, det_y2, 90, 140)
#
# print((cr_aux2_x2 - cr_aux2_x1) - (cr_main_x2 - cr_main_x1))
# print(cr_aux2_yaw2 - cr_aux2_yaw1)


# def suggest_roll_offsets(bragg_target, roll_range=5):

# bragg_target = 85
# allowed_roll_offsets = [2.5, 11.5, 20.5]
# offsets = np.array(allowed_roll_offsets)
# offset_braggs = 90 - offsets
#
# print(offsets[options][0])

# rowland_circle.det_focus =
# rowland_circle.update_nominal_trajectory_for_detector(-8)
# johann_det_arm.position_dict
# johann_main_crystal.position_dict

# rowland_circle.plot_motor_pos_vs_bragg('motor_det_th1')
# ___nom = rc.compute_motor_position('motor_cr_assy_x', 85.0, nom2act=False)
# rc.register_bragg(85.1, {'motor_cr_assy_x' : ___nom})
#
# _bragg = np.linspace(65, 90, 101)
# _x_nom = rc.compute_motor_position('motor_cr_assy_x', _bragg, nom2act=False)
# _x_act = rc.compute_motor_position('motor_cr_assy_x', _bragg, nom2act=True)
#
# plt.figure(1, clear=True)
# plt.subplot(211)
# plt.plot(_bragg, _x_nom)
# plt.plot(_bragg, _x_act)
#
# plt.hlines(___nom, 65, 90, colors='k')
# plt.vlines(85.0, _x_nom.min(), _x_nom.max(), colors='k')
# plt.vlines(85.1, _x_nom.min(), _x_nom.max(), colors='k')
#
# plt.subplot(212)
# plt.plot(_bragg, _x_act - _x_nom)
#
#
# rc.compute_motor_position('motor_cr_assy_x', 85.0, nom2act=False) - rc.compute_motor_position('motor_cr_assy_x', 85.1, nom2act=False)
# rc.compute_motor_position('motor_cr_assy_x', 85.2, nom2act=False) - rc.compute_motor_position('motor_cr_assy_x', 85.1, nom2act=False)
#
# rc.compute_motor_position('motor_cr_assy_x', 85.0, nom2act=True) - rc.compute_motor_position('motor_cr_assy_x', 85.1, nom2act=True)
# rc.compute_motor_position('motor_cr_assy_x', 85.1, nom2act=True) - rc.compute_motor_position('motor_cr_assy_x', 85.2, nom2act=True)

from collections import namedtuple
class ISSPseudoPositioner(PseudoPositioner):

    def __init__(self, *args, **kwargs):
        self.pseudo_keys = [k for k, _ in self._get_pseudo_positioners()]
        self.real_keys = [k for k, _ in self._get_real_positioners()]
        self.motor_keys = self.pseudo_keys + self.real_keys
        super().__init__(*args, **kwargs)

    def pseudo_pos2dict(self, pseudo_pos):
        ret = {k: getattr(pseudo_pos, k) for k in self.pseudo_keys}
        for k in ret.keys():
            if ret[k] is None:
                try:
                    ret[k] = getattr(self, k).position
                except:
                    ttime.sleep(1)
                    ret[k] = getattr(self, k).position
        return ret

    def real_pos2dict(self, real_pos):
        ret = {k: getattr(real_pos, k) for k in self.real_keys}
        for k in ret.keys():
            if ret[k] is None:
                try:
                    ret[k] = getattr(self, k).position
                except:
                    ttime.sleep(1)
                    ret[k] = getattr(self, k).position
        return ret

    @pseudo_position_argument
    def forward(self, pseudo_pos):
        pseudo_dict = self.pseudo_pos2dict(pseudo_pos)
        real_dict = self._forward(pseudo_dict)
        return self.RealPosition(**real_dict)

    @real_position_argument
    def inverse(self, real_pos):
        real_dict = self.real_pos2dict(real_pos)
        pseudo_dict = self._inverse(real_dict)
        return self.PseudoPosition(**pseudo_dict)

    @property
    def position_dict(self):
        return self.pseudo_pos2dict(self.position)

    @property
    def real_position_dict(self):
        return self.real_pos2dict(self.real_position)


class JohannPseudoPositioner(ISSPseudoPositioner):
    rowland_circle = rowland_circle

    def __init__(self,*args, **kwargs):
        super().__init__(*args, **kwargs)

# inclinometer1 = 12222 - th1 = 0.0
# inclinometer1 = 12877 - th1 = -7.366
# inclinometer1 = 11795 - th1 = 4.866




# class EpicsMotorWithExternalSensor(EpicsMotor):
#     def __init__(self, *args, sensor=None, conversion_json_path='', polynom_order=1, atol=0.1, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.sensor = sensor
#         self._conversion_df = pd.read_json(conversion_json_path)
#         self._get_converter_parameters(polynom_order)
#         self._atol = atol
#
#     #     def callback(value, old_value, **kwargs):
#     #         if (int(value == 1)) and (int(old_value) == 0):
#     #             self.check_with_sensor()
#     #     self._sensor_subscription_cid = self.motor_is_moving.subscribe(callback)
#     #
#     # def _unsubscribe_sensor(self):
#     #     self.motor_is_moving.unsubscribe(self._sensor_subscription_cid)
#
#     def _get_converter_parameters(self, polynom_order):
#         d = self._conversion_df.values
#         self._p_sen2pos = np.polyfit(d[:, 1], d[:, 0], polynom_order)
#     def _sen2pos(self, sensor_value):
#         return np.polyval(self._p_sen2pos, sensor_value)
#
#     @property
#     def position_from_sensor(self):
#         value = self.sensor.get()
#         return self._sen2pos(value)
#
#     def set_current_position(self, new_position):
#         limits = self.limits
#         old_offset = self.user_offset.get()
#         new_offset = old_offset + new_position - self.position
#         self.user_offset.set(new_offset)
#         self.set_lim(*limits)
#
#     def check_with_sensor(self):
#         if abs(self.position - self.position_from_sensor) > self._atol:
#             ttime.sleep(0.1)
#             print_to_gui(f'Detector Goniometer 1 position disagrees with inclinometer by >{self._atol} degrees. Correcting.', tag='Spectrometer', add_timestamp=True)
#             self.set_current_position(self.position_from_sensor)

    # def move(self, *args, **kwargs):
    #     st = super().move(*args, **kwargs)




# det_inclinometer1 = EpicsSignal('XF:08IDB-CT{DIODE-Box_B2:4}InCh0:Data-I', name='det_inclinometer1')
# bla = EpicsMotorWithExternalSensor('XF:08IDB-OP{HRS:1-Det:Gon:Theta1}Mtr', name='bla',
#                               sensor=det_inclinometer1,
#                               conversion_json_path=f'{ROOT_PATH_SHARED}/settings/json/inclinometer_data.json')

    # def move(self, *args, **kwargs):

#f'{ROOT_PATH_SHARED}/settings/json/inclinometer_data.json'

class JohannDetectorArm(JohannPseudoPositioner):
    motor_det_x = Cpt(EpicsMotor, 'XF:08IDB-OP{Stage:Aux1-Ax:Y}Mtr')
    motor_det_th1 = Cpt(EpicsMotor, 'XF:08IDB-OP{HRS:1-Det:Gon:Theta1}Mtr')  # give better names
    # motor_det_th1 = Cpt(EpicsMotorWithExternalSensor, 'XF:08IDB-OP{HRS:1-Det:Gon:Theta1}Mtr',
    #                     sensor=det_inclinometer1,
    #                     conversion_json_path=f'{ROOT_PATH_SHARED}/settings/json/inclinometer_data.json')  # give better names
    motor_det_th2 = Cpt(EpicsMotor, 'XF:08IDB-OP{HRS:1-Det:Gon:Theta2}Mtr')

    bragg = Cpt(PseudoSingle, name='bragg')
    det_focus = Cpt(PseudoSingle, name='det_focus')

    _real = ['motor_det_x', 'motor_det_th1', 'motor_det_th2']
    _pseudo = ['bragg', 'det_focus']

    def _forward(self, pseudo_dict):
        bragg, det_focus = pseudo_dict['bragg'], pseudo_dict['det_focus']
        self.rowland_circle.det_focus = det_focus
        return self.rowland_circle.compute_motor_position(self.real_keys, bragg)

    def _inverse(self, real_dict):
        motor_det_th1 = real_dict['motor_det_th1']
        bragg = self.rowland_circle.compute_bragg_from_motor('motor_det_th1', motor_det_th1)
        det_focus = self.rowland_circle.det_focus
        return {'bragg': bragg, 'det_focus' : det_focus}


johann_det_arm = JohannDetectorArm(name='johann_detector_arm')



class JohannMainCrystal(JohannPseudoPositioner):

    motor_cr_assy_x = Cpt(EpicsMotor, 'XF:08IDB-OP{Stage:Aux1-Ax:X}Mtr')
    motor_cr_assy_y = Cpt(EpicsMotor, 'XF:08IDB-OP{HRS:1-Ana:Assy:Y}Mtr')
    motor_cr_main_roll = Cpt(EpicsMotor, 'XF:08IDB-OP{HRS:1-Stk:1:Roll}Mtr')
    motor_cr_main_yaw = Cpt(EpicsMotor, 'XF:08IDB-OP{HRS:1-Stk:1:Yaw}Mtr')

    bragg = Cpt(PseudoSingle, name='bragg')

    _real = ['motor_cr_assy_x', 'motor_cr_assy_y', 'motor_cr_main_roll', 'motor_cr_main_yaw']
    _pseudo = ['bragg']

    def _forward(self, pseudo_dict):
        bragg = pseudo_dict['bragg']
        return self.rowland_circle.compute_motor_position(self.real_keys, bragg)

    def _inverse(self, real_dict):
        motor_cr_main_roll = real_dict['motor_cr_main_roll']
        bragg = self.rowland_circle.compute_bragg_from_motor('motor_cr_main_roll', motor_cr_main_roll)
        return {'bragg': bragg}

johann_main_crystal = JohannMainCrystal(name='johann_main_crystal')


class JohannAux2Crystal(JohannPseudoPositioner):
    motor_cr_assy_x = Cpt(EpicsMotor, 'XF:08IDB-OP{Stage:Aux1-Ax:X}Mtr')
    motor_cr_assy_y = Cpt(EpicsMotor, 'XF:08IDB-OP{HRS:1-Ana:Assy:Y}Mtr')
    motor_cr_aux2_x = Cpt(EpicsMotor, 'XF:08IDB-OP{HRS:1-Stk:2:X}Mtr')
    motor_cr_aux2_y = Cpt(EpicsMotor, 'XF:08IDB-OP{HRS:1-Stk:2:Y}Mtr')
    motor_cr_aux2_roll = Cpt(EpicsMotor, 'XF:08IDB-OP{HRS:1-Stk:2:Roll}Mtr')
    motor_cr_aux2_yaw = Cpt(EpicsMotor, 'XF:08IDB-OP{HRS:1-Stk:2:Yaw}Mtr')

    bragg = Cpt(PseudoSingle, name='bragg')
    _real = ['motor_cr_assy_x', 'motor_cr_assy_y',
             'motor_cr_aux2_x', 'motor_cr_aux2_y', 'motor_cr_aux2_roll', 'motor_cr_aux2_yaw']
    _pseudo = ['bragg']

    def _forward(self, pseudo_dict):
        bragg = pseudo_dict['bragg']
        return self.rowland_circle.compute_motor_position(self.real_keys, bragg)

    def _inverse(self, real_dict):
        motor_cr_aux2_roll = real_dict['motor_cr_aux2_roll']
        bragg = self.rowland_circle.compute_bragg_from_motor('motor_cr_aux2_roll', motor_cr_aux2_roll)
        return {'bragg': bragg}

johann_aux2_crystal = JohannAux2Crystal(name='johann_aux2_crystal')

class JohannAux3Crystal(JohannPseudoPositioner):
    motor_cr_assy_x = Cpt(EpicsMotor, 'XF:08IDB-OP{Stage:Aux1-Ax:X}Mtr')
    motor_cr_assy_y = Cpt(EpicsMotor, 'XF:08IDB-OP{HRS:1-Ana:Assy:Y}Mtr')
    motor_cr_aux3_x = Cpt(EpicsMotor, 'XF:08IDB-OP{HRS:1-Stk:3:X}Mtr')
    motor_cr_aux3_y = Cpt(EpicsMotor, 'XF:08IDB-OP{HRS:1-Stk:3:Y}Mtr')
    motor_cr_aux3_roll = Cpt(EpicsMotor, 'XF:08IDB-OP{HRS:1-Stk:3:Roll}Mtr')
    motor_cr_aux3_yaw = Cpt(EpicsMotor, 'XF:08IDB-OP{HRS:1-Stk:3:Yaw}Mtr')

    bragg = Cpt(PseudoSingle, name='bragg')
    _real = ['motor_cr_assy_x', 'motor_cr_assy_y',
             'motor_cr_aux3_x', 'motor_cr_aux3_y', 'motor_cr_aux3_roll', 'motor_cr_aux3_yaw']
    _pseudo = ['bragg']

    def _forward(self, pseudo_dict):
        bragg = pseudo_dict['bragg']
        return self.rowland_circle.compute_motor_position(self.real_keys, bragg)

    def _inverse(self, real_dict):
        motor_cr_aux3_roll = real_dict['motor_cr_aux3_roll']
        bragg = self.rowland_circle.compute_bragg_from_motor('motor_cr_aux3_roll', motor_cr_aux3_roll)
        return {'bragg': bragg}

johann_aux3_crystal = JohannAux3Crystal(name='johann_aux3_crystal')


class JohannAux4Crystal(JohannPseudoPositioner):
    motor_cr_assy_x = Cpt(EpicsMotor, 'XF:08IDB-OP{Stage:Aux1-Ax:X}Mtr')
    motor_cr_assy_y = Cpt(EpicsMotor, 'XF:08IDB-OP{HRS:1-Ana:Assy:Y}Mtr')
    motor_cr_aux4_x = Cpt(EpicsMotor, 'XF:08IDB-OP{HRS:1-Stk:4:X}Mtr')
    motor_cr_aux4_y = Cpt(EpicsMotor, 'XF:08IDB-OP{HRS:1-Stk:4:Y}Mtr')
    motor_cr_aux4_roll = Cpt(EpicsMotor, 'XF:08IDB-OP{HRS:1-Stk:4:Roll}Mtr')
    motor_cr_aux4_yaw = Cpt(EpicsMotor, 'XF:08IDB-OP{HRS:1-Stk:4:Yaw}Mtr')

    bragg = Cpt(PseudoSingle, name='bragg')
    _real = ['motor_cr_assy_x', 'motor_cr_assy_y',
             'motor_cr_aux4_x', 'motor_cr_aux4_y', 'motor_cr_aux4_roll', 'motor_cr_aux4_yaw']
    _pseudo = ['bragg']

    def _forward(self, pseudo_dict):
        bragg = pseudo_dict['bragg']
        return self.rowland_circle.compute_motor_position(self.real_keys, bragg)

    def _inverse(self, real_dict):
        motor_cr_aux3_roll = real_dict['motor_cr_aux4_roll']
        bragg = self.rowland_circle.compute_bragg_from_motor('motor_cr_aux4_roll', motor_cr_aux3_roll)
        return {'bragg': bragg}

johann_aux4_crystal = JohannAux4Crystal(name='johann_aux4_crystal')


class JohannAux5Crystal(JohannPseudoPositioner):
    motor_cr_assy_x = Cpt(EpicsMotor, 'XF:08IDB-OP{Stage:Aux1-Ax:X}Mtr')
    motor_cr_assy_y = Cpt(EpicsMotor, 'XF:08IDB-OP{HRS:1-Ana:Assy:Y}Mtr')
    motor_cr_aux5_x = Cpt(EpicsMotor, 'XF:08IDB-OP{HRS:1-Stk:5:X}Mtr')
    motor_cr_aux5_y = Cpt(EpicsMotor, 'XF:08IDB-OP{HRS:1-Stk:5:Y}Mtr')
    motor_cr_aux5_roll = Cpt(EpicsMotor, 'XF:08IDB-OP{HRS:1-Stk:5:Roll}Mtr')
    motor_cr_aux5_yaw = Cpt(EpicsMotor, 'XF:08IDB-OP{HRS:1-Stk:5:Yaw}Mtr')

    bragg = Cpt(PseudoSingle, name='bragg')
    _real = ['motor_cr_assy_x', 'motor_cr_assy_y',
             'motor_cr_aux5_x', 'motor_cr_aux5_y', 'motor_cr_aux5_roll', 'motor_cr_aux5_yaw']
    _pseudo = ['bragg']

    def _forward(self, pseudo_dict):
        bragg = pseudo_dict['bragg']
        return self.rowland_circle.compute_motor_position(self.real_keys, bragg)

    def _inverse(self, real_dict):
        motor_cr_aux3_roll = real_dict['motor_cr_aux5_roll']
        bragg = self.rowland_circle.compute_bragg_from_motor('motor_cr_aux5_roll', motor_cr_aux3_roll)
        return {'bragg': bragg}

johann_aux5_crystal = JohannAux5Crystal(name='johann_aux5_crystal')


class JohannSpectrometerX(JohannPseudoPositioner):
    motor_det_x = Cpt(EpicsMotor, 'XF:08IDB-OP{Stage:Aux1-Ax:Y}Mtr')
    motor_cr_assy_x = Cpt(EpicsMotor, 'XF:08IDB-OP{Stage:Aux1-Ax:X}Mtr')

    x = Cpt(PseudoSingle, name='x')

    _real = ['motor_det_x', 'motor_cr_assy_x']
    _pseudo = ['x']

    aligned = True

    def _compute_pos_nom(self):
        # instead of real keys we use hard-coded keys for the motors !
        bragg_det_arm = johann_det_arm.bragg.position
        pos_nom_det_arm = self.rowland_circle.compute_motor_position(['motor_det_x'], bragg_det_arm, nom2act=True)

        bragg_cr_main = johann_main_crystal.bragg.position
        pos_nom_cr_main = self.rowland_circle.compute_motor_position(['motor_cr_assy_x'], bragg_cr_main, nom2act=True)

        return {**pos_nom_det_arm, **pos_nom_cr_main}


    def _forward(self, pseudo_dict):
        x = pseudo_dict['x']
        # bragg = johann_spectrometer.bragg.position
        # pos_nom = self.rowland_circle.compute_motor_position(self.real_keys, bragg, nom2act=False)
        pos_nom = self._compute_pos_nom()
        pos_act = {}
        for key in pos_nom.keys():
            pos_act[key] = pos_nom[key] + x
        return pos_act

    def _inverse(self, pos_act):
        # bragg = johann_spectrometer.bragg.position
        # pos_nom = self.rowland_circle.compute_motor_position(self.real_keys, bragg, nom2act=False)
        pos_nom = self._compute_pos_nom()
        x = [(pos_act[k] - pos_nom[k]) for k in self.real_keys]
        self.aligned = np.all(np.isclose(x, x[0], atol=5e-3))
        return {'x' : np.mean(x)}

johann_spectrometer_x = JohannSpectrometerX(name='johann_spectrometer_x')


class JohannCrystalHoming(Device):
    cr_main_roll_home = Cpt(EpicsSignal, '1:Roll}Mtr.HOMF')
    cr_main_yaw_home = Cpt(EpicsSignal, '1:Yaw}Mtr.HOMF')

    cr_aux2_roll_home = Cpt(EpicsSignal, '2:Roll}Mtr.HOMF')
    cr_aux2_yaw_home = Cpt(EpicsSignal, '2:Yaw}Mtr.HOMF')
    cr_aux2_x_home = Cpt(EpicsSignal, '2:X}Mtr.HOMF')
    cr_aux2_y_home = Cpt(EpicsSignal, '2:Y}Mtr.HOMF')

    cr_aux3_roll_home = Cpt(EpicsSignal, '3:Roll}Mtr.HOMF')
    cr_aux3_yaw_home = Cpt(EpicsSignal, '3:Yaw}Mtr.HOMF')
    cr_aux3_x_home = Cpt(EpicsSignal, '3:X}Mtr.HOMF')
    cr_aux3_y_home = Cpt(EpicsSignal, '3:Y}Mtr.HOMF')

    cr_aux4_roll_home = Cpt(EpicsSignal, '4:Roll}Mtr.HOMF')
    cr_aux4_yaw_home = Cpt(EpicsSignal, '4:Yaw}Mtr.HOMF')
    cr_aux4_x_home = Cpt(EpicsSignal, '4:X}Mtr.HOMF')
    cr_aux4_y_home = Cpt(EpicsSignal, '4:Y}Mtr.HOMF')

    cr_aux5_roll_home = Cpt(EpicsSignal, '5:Roll}Mtr.HOMF')
    cr_aux5_yaw_home = Cpt(EpicsSignal, '5:Yaw}Mtr.HOMF')
    cr_aux5_x_home = Cpt(EpicsSignal, '5:X}Mtr.HOMF')
    cr_aux5_y_home = Cpt(EpicsSignal, '5:Y}Mtr.HOMF')

    def home_all_axes(self):
        for component in self.component_names:
            cpt = getattr(self, component)
            cpt.put(1)


class JohannMultiCrystalPseudoPositioner(JohannPseudoPositioner):
    _real = _johann_spectrometer_motor_keys
        # ['motor_det_x', 'motor_det_th1', 'motor_det_th2',
        #      'motor_cr_assy_x', 'motor_cr_assy_y', 'motor_cr_main_roll', 'motor_cr_main_yaw',  # ]
        #      'motor_cr_aux2_x', 'motor_cr_aux2_y', 'motor_cr_aux2_roll', 'motor_cr_aux2_yaw',
        #      'motor_cr_aux3_x', 'motor_cr_aux3_y', 'motor_cr_aux3_roll', 'motor_cr_aux3_yaw']

    aligned = True
    piezo_homing = Cpt(JohannCrystalHoming, 'XF:08IDB-OP{HRS:1-Stk:')

    @property
    def initialized(self):
        return self.rowland_circle.initialized

    @initialized.setter
    def initialized(self, value):
        self.rowland_circle.initialized = value

    # def __init__(self, *args, **kwargs):
    #     self.enabled_crystals = self.rowland_circle.enabled_crystals
    #     super().__init__(*args, **kwargs)

    @property
    def enabled_crystals(self):
        return self.rowland_circle.enabled_crystals

    @property
    def enabled_crystals_list(self):
        enabled_crystals_list = [_c for _c, _e in johann_emission.enabled_crystals.items() if _e]

        # enforce that the main crystal comes first
        if 'main' in enabled_crystals_list:
            enabled_crystals_list.pop(enabled_crystals_list.index('main'))
            enabled_crystals_list = ['main'] + enabled_crystals_list
        return enabled_crystals_list

    def enable_crystal(self, crystal_key, enable):
        self.rowland_circle.enable_crystal(crystal_key, enable)

    @property
    def motor_to_bragg_keys(self):
        keys = ['motor_det_th1']
        if self.enabled_crystals['main']: keys.append('motor_cr_main_roll')
        if self.enabled_crystals['aux2']: keys.append('motor_cr_aux2_roll')
        if self.enabled_crystals['aux3']: keys.append('motor_cr_aux3_roll')
        if self.enabled_crystals['aux4']: keys.append('motor_cr_aux4_roll')
        if self.enabled_crystals['aux5']: keys.append('motor_cr_aux5_roll')
        return keys

    def _compute_new_real_positions_for_enabled_crystals(self, bragg):
        new_real_position = self.rowland_circle.compute_motor_position(self.real_keys, bragg)
        for crystal_key, enabled in self.enabled_crystals.items():
            if not enabled: # then find keys that must be static during the motion of enabled crystals
                if crystal_key == 'main':
                    _motor_keys = _johann_cr_main_motor_keys
                elif crystal_key == 'aux2':
                    _motor_keys = _johann_cr_aux2_motor_keys
                elif crystal_key == 'aux3':
                    _motor_keys = _johann_cr_aux3_motor_keys
                elif crystal_key == 'aux4':
                    _motor_keys = _johann_cr_aux4_motor_keys
                elif crystal_key == 'aux5':
                    _motor_keys = _johann_cr_aux5_motor_keys
                else:
                    raise ValueError('this crystal key is not implemented yet')
                for k in _motor_keys:
                    new_real_position[k] = getattr(self, k).position
        return new_real_position

    def _compute_braggs_from_motors(self, real_dict):
        braggs = []
        for key in self.motor_to_bragg_keys:
            pos = real_dict[key]
            bragg = self.rowland_circle.compute_bragg_from_motor(key, pos)
            braggs.append(bragg)

        self.aligned = np.all(np.isclose(braggs, braggs[0], atol=1e-3))
        return braggs

    def home_crystal_piezos(self):
        self.piezo_homing.home_all_axes()


class JohannAllCrystals(JohannMultiCrystalPseudoPositioner):
    _real = _johann_cr_all_motor_keys
    motor_cr_assy_x = Cpt(EpicsMotor, 'XF:08IDB-OP{Stage:Aux1-Ax:X}Mtr')
    motor_cr_assy_y = Cpt(EpicsMotor, 'XF:08IDB-OP{HRS:1-Ana:Assy:Y}Mtr')
    motor_cr_main_roll = Cpt(EpicsMotor, 'XF:08IDB-OP{HRS:1-Stk:1:Roll}Mtr')
    motor_cr_main_yaw = Cpt(EpicsMotor, 'XF:08IDB-OP{HRS:1-Stk:1:Yaw}Mtr')

    motor_cr_aux2_x = Cpt(EpicsMotor, 'XF:08IDB-OP{HRS:1-Stk:2:X}Mtr')
    motor_cr_aux2_y = Cpt(EpicsMotor, 'XF:08IDB-OP{HRS:1-Stk:2:Y}Mtr')
    motor_cr_aux2_roll = Cpt(EpicsMotor, 'XF:08IDB-OP{HRS:1-Stk:2:Roll}Mtr')
    motor_cr_aux2_yaw = Cpt(EpicsMotor, 'XF:08IDB-OP{HRS:1-Stk:2:Yaw}Mtr')

    motor_cr_aux3_x = Cpt(EpicsMotor, 'XF:08IDB-OP{HRS:1-Stk:3:X}Mtr')
    motor_cr_aux3_y = Cpt(EpicsMotor, 'XF:08IDB-OP{HRS:1-Stk:3:Y}Mtr')
    motor_cr_aux3_roll = Cpt(EpicsMotor, 'XF:08IDB-OP{HRS:1-Stk:3:Roll}Mtr')
    motor_cr_aux3_yaw = Cpt(EpicsMotor, 'XF:08IDB-OP{HRS:1-Stk:3:Yaw}Mtr')

    motor_cr_aux4_x = Cpt(EpicsMotor, 'XF:08IDB-OP{HRS:1-Stk:4:X}Mtr')
    motor_cr_aux4_y = Cpt(EpicsMotor, 'XF:08IDB-OP{HRS:1-Stk:4:Y}Mtr')
    motor_cr_aux4_roll = Cpt(EpicsMotor, 'XF:08IDB-OP{HRS:1-Stk:4:Roll}Mtr')
    motor_cr_aux4_yaw = Cpt(EpicsMotor, 'XF:08IDB-OP{HRS:1-Stk:4:Yaw}Mtr')

    motor_cr_aux5_x = Cpt(EpicsMotor, 'XF:08IDB-OP{HRS:1-Stk:5:X}Mtr')
    motor_cr_aux5_y = Cpt(EpicsMotor, 'XF:08IDB-OP{HRS:1-Stk:5:Y}Mtr')
    motor_cr_aux5_roll = Cpt(EpicsMotor, 'XF:08IDB-OP{HRS:1-Stk:5:Roll}Mtr')
    motor_cr_aux5_yaw = Cpt(EpicsMotor, 'XF:08IDB-OP{HRS:1-Stk:5:Yaw}Mtr')

    bragg = Cpt(PseudoSingle, name='bragg')

    _pseudo = ['bragg']

    @property
    def motor_to_bragg_keys(self):
        keys = []
        if self.enabled_crystals['main']: keys.append('motor_cr_main_roll')
        if self.enabled_crystals['aux2']: keys.append('motor_cr_aux2_roll')
        if self.enabled_crystals['aux3']: keys.append('motor_cr_aux3_roll')
        if self.enabled_crystals['aux4']: keys.append('motor_cr_aux4_roll')
        if self.enabled_crystals['aux5']: keys.append('motor_cr_aux5_roll')
        return keys

    def _forward(self, pseudo_dict):
        bragg = pseudo_dict['bragg']
        new_real_positions = self._compute_new_real_positions_for_enabled_crystals(bragg)
        return new_real_positions

    def _inverse(self, real_dict):
        braggs = self._compute_braggs_from_motors(real_dict)
        return {'bragg': np.mean(braggs)}

johann_all_crystals = JohannAllCrystals(name='johann_all_crystals')

class JohannSpectrometer(JohannMultiCrystalPseudoPositioner):
    motor_det_x = Cpt(EpicsMotor, 'XF:08IDB-OP{Stage:Aux1-Ax:Y}Mtr')
    motor_det_th1 = Cpt(EpicsMotor, 'XF:08IDB-OP{HRS:1-Det:Gon:Theta1}Mtr')  # give better names
    motor_det_th2 = Cpt(EpicsMotor, 'XF:08IDB-OP{HRS:1-Det:Gon:Theta2}Mtr')

    motor_cr_assy_x = Cpt(EpicsMotor, 'XF:08IDB-OP{Stage:Aux1-Ax:X}Mtr')
    motor_cr_assy_y = Cpt(EpicsMotor, 'XF:08IDB-OP{HRS:1-Ana:Assy:Y}Mtr')
    motor_cr_main_roll = Cpt(EpicsMotor, 'XF:08IDB-OP{HRS:1-Stk:1:Roll}Mtr')
    motor_cr_main_yaw = Cpt(EpicsMotor, 'XF:08IDB-OP{HRS:1-Stk:1:Yaw}Mtr')

    motor_cr_aux2_x = Cpt(EpicsMotor, 'XF:08IDB-OP{HRS:1-Stk:2:X}Mtr')
    motor_cr_aux2_y = Cpt(EpicsMotor, 'XF:08IDB-OP{HRS:1-Stk:2:Y}Mtr')
    motor_cr_aux2_roll = Cpt(EpicsMotor, 'XF:08IDB-OP{HRS:1-Stk:2:Roll}Mtr')
    motor_cr_aux2_yaw = Cpt(EpicsMotor, 'XF:08IDB-OP{HRS:1-Stk:2:Yaw}Mtr')

    motor_cr_aux3_x = Cpt(EpicsMotor, 'XF:08IDB-OP{HRS:1-Stk:3:X}Mtr')
    motor_cr_aux3_y = Cpt(EpicsMotor, 'XF:08IDB-OP{HRS:1-Stk:3:Y}Mtr')
    motor_cr_aux3_roll = Cpt(EpicsMotor, 'XF:08IDB-OP{HRS:1-Stk:3:Roll}Mtr')
    motor_cr_aux3_yaw = Cpt(EpicsMotor, 'XF:08IDB-OP{HRS:1-Stk:3:Yaw}Mtr')

    motor_cr_aux4_x = Cpt(EpicsMotor, 'XF:08IDB-OP{HRS:1-Stk:4:X}Mtr')
    motor_cr_aux4_y = Cpt(EpicsMotor, 'XF:08IDB-OP{HRS:1-Stk:4:Y}Mtr')
    motor_cr_aux4_roll = Cpt(EpicsMotor, 'XF:08IDB-OP{HRS:1-Stk:4:Roll}Mtr')
    motor_cr_aux4_yaw = Cpt(EpicsMotor, 'XF:08IDB-OP{HRS:1-Stk:4:Yaw}Mtr')

    motor_cr_aux5_x = Cpt(EpicsMotor, 'XF:08IDB-OP{HRS:1-Stk:5:X}Mtr')
    motor_cr_aux5_y = Cpt(EpicsMotor, 'XF:08IDB-OP{HRS:1-Stk:5:Y}Mtr')
    motor_cr_aux5_roll = Cpt(EpicsMotor, 'XF:08IDB-OP{HRS:1-Stk:5:Roll}Mtr')
    motor_cr_aux5_yaw = Cpt(EpicsMotor, 'XF:08IDB-OP{HRS:1-Stk:5:Yaw}Mtr')

    bragg = Cpt(PseudoSingle, name='bragg')

    _pseudo = ['bragg']

    def _forward(self, pseudo_dict):
        bragg = pseudo_dict['bragg']
        new_real_positions = self._compute_new_real_positions_for_enabled_crystals(bragg)
        return new_real_positions

    def _inverse(self, real_dict):
        braggs = self._compute_braggs_from_motors(real_dict)
        return {'bragg': np.mean(braggs)}

johann_spectrometer = JohannSpectrometer(name='johann_spectrometer')

class JohannEmission(JohannMultiCrystalPseudoPositioner):
    motor_det_x = Cpt(EpicsMotor, 'XF:08IDB-OP{Stage:Aux1-Ax:Y}Mtr')
    motor_det_th1 = Cpt(EpicsMotor, 'XF:08IDB-OP{HRS:1-Det:Gon:Theta1}Mtr')  # give better names
    motor_det_th2 = Cpt(EpicsMotor, 'XF:08IDB-OP{HRS:1-Det:Gon:Theta2}Mtr')

    motor_cr_assy_x = Cpt(EpicsMotor, 'XF:08IDB-OP{Stage:Aux1-Ax:X}Mtr')
    motor_cr_assy_y = Cpt(EpicsMotor, 'XF:08IDB-OP{HRS:1-Ana:Assy:Y}Mtr')
    motor_cr_main_roll = Cpt(EpicsMotor, 'XF:08IDB-OP{HRS:1-Stk:1:Roll}Mtr')
    motor_cr_main_yaw = Cpt(EpicsMotor, 'XF:08IDB-OP{HRS:1-Stk:1:Yaw}Mtr')

    motor_cr_aux2_x = Cpt(EpicsMotor, 'XF:08IDB-OP{HRS:1-Stk:2:X}Mtr')
    motor_cr_aux2_y = Cpt(EpicsMotor, 'XF:08IDB-OP{HRS:1-Stk:2:Y}Mtr')
    motor_cr_aux2_roll = Cpt(EpicsMotor, 'XF:08IDB-OP{HRS:1-Stk:2:Roll}Mtr')
    motor_cr_aux2_yaw = Cpt(EpicsMotor, 'XF:08IDB-OP{HRS:1-Stk:2:Yaw}Mtr')

    motor_cr_aux3_x = Cpt(EpicsMotor, 'XF:08IDB-OP{HRS:1-Stk:3:X}Mtr')
    motor_cr_aux3_y = Cpt(EpicsMotor, 'XF:08IDB-OP{HRS:1-Stk:3:Y}Mtr')
    motor_cr_aux3_roll = Cpt(EpicsMotor, 'XF:08IDB-OP{HRS:1-Stk:3:Roll}Mtr')
    motor_cr_aux3_yaw = Cpt(EpicsMotor, 'XF:08IDB-OP{HRS:1-Stk:3:Yaw}Mtr')

    motor_cr_aux4_x = Cpt(EpicsMotor, 'XF:08IDB-OP{HRS:1-Stk:4:X}Mtr')
    motor_cr_aux4_y = Cpt(EpicsMotor, 'XF:08IDB-OP{HRS:1-Stk:4:Y}Mtr')
    motor_cr_aux4_roll = Cpt(EpicsMotor, 'XF:08IDB-OP{HRS:1-Stk:4:Roll}Mtr')
    motor_cr_aux4_yaw = Cpt(EpicsMotor, 'XF:08IDB-OP{HRS:1-Stk:4:Yaw}Mtr')

    motor_cr_aux5_x = Cpt(EpicsMotor, 'XF:08IDB-OP{HRS:1-Stk:5:X}Mtr')
    motor_cr_aux5_y = Cpt(EpicsMotor, 'XF:08IDB-OP{HRS:1-Stk:5:Y}Mtr')
    motor_cr_aux5_roll = Cpt(EpicsMotor, 'XF:08IDB-OP{HRS:1-Stk:5:Roll}Mtr')
    motor_cr_aux5_yaw = Cpt(EpicsMotor, 'XF:08IDB-OP{HRS:1-Stk:5:Yaw}Mtr')

    energy = Cpt(PseudoSingle, name='energy')
    _pseudo = ['energy']

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        e_lo, e_hi = self.rowland_circle.config['energy_limits']
        self._set_energy_limits(e_lo, e_hi)

    def _forward(self, pseudo_dict):
        energy = pseudo_dict['energy']
        bragg = self.rowland_circle.e2bragg(energy)
        new_real_positions = self._compute_new_real_positions_for_enabled_crystals(bragg)
        return new_real_positions

    def _inverse(self, real_dict):
        braggs = self._compute_braggs_from_motors(real_dict)
        bragg = np.mean(braggs)
        energy = self.rowland_circle.bragg2e(bragg)
        return {'energy': np.mean(energy)}

    # ops functions

    def move_crystals_to_90_deg(self):
        self.set_roll_offset(2.5)
        johann_main_crystal.move(bragg=90)
        johann_aux2_crystal.move(bragg=90)
        johann_aux3_crystal.move(bragg=90)
        johann_aux4_crystal.move(bragg=90)
        johann_aux5_crystal.move(bragg=90)

    def set_det_arm_parking(self):
        self.rowland_circle.set_det_arm_parking(self.real_position_dict)

    def read_det_arm_parking(self):
        return self.rowland_circle.det_arm_parking()

    def set_main_crystal_parking(self):
        self.rowland_circle.set_main_crystal_parking(self.real_position_dict)

    def read_main_crystal_parking(self):
        return self.rowland_circle.main_crystal_parking(human_readable=True)

    def set_aux2_crystal_parking(self):
        self.rowland_circle.set_aux2_crystal_parking(self.real_position_dict)

    def read_aux2_crystal_parking(self):
        return self.rowland_circle.aux2_crystal_parking(human_readable=True)

    def set_aux3_crystal_parking(self):
        self.rowland_circle.set_aux3_crystal_parking(self.real_position_dict)

    def read_aux3_crystal_parking(self):
        return self.rowland_circle.aux3_crystal_parking(human_readable=True)

    def set_aux4_crystal_parking(self):
        self.rowland_circle.set_aux4_crystal_parking(self.real_position_dict)

    def read_aux4_crystal_parking(self):
        return self.rowland_circle.aux4_crystal_parking(human_readable=True)

    def set_aux5_crystal_parking(self):
        self.rowland_circle.set_aux5_crystal_parking(self.real_position_dict)

    def read_aux5_crystal_parking(self):
        return self.rowland_circle.aux5_crystal_parking(human_readable=True)

    def update_R_parking(self, value):
        self.rowland_circle.R_parking = value

    def read_R_parking(self):
        return self.rowland_circle.R_parking

    def read_crystal_aux_dz(self):
        return self.rowland_circle.cr_aux20_dz, self.rowland_circle.cr_aux42_dz

    def register_energy(self, energy):
        self.rowland_circle.register_energy(energy, self.real_position_dict)

    def reset_config(self):
        self.rowland_circle.reset_bragg_registration()
        self.rowland_circle.reset_spectrometer_calibration()

    def put_detector_to_safe_position(self):
        self.motor_det_x.move(335)
        self.motor_det_th1.move(0)
        self.motor_det_th2.move(0)

    def set_spectrometer_calibration(self, x_nom, x_act, n_poly=2):
        self.rowland_circle.set_spectrometer_calibration(x_nom, x_act, n_poly=n_poly)

    def _set_energy_limits(self, e_lo, e_hi):
        self.energy._limits = (e_lo, e_hi)

    def _match_energy_limits_to_rowland_circle(self):
        e_lo, e_hi = self.rowland_circle.energy_limits
        self._set_energy_limits(e_lo, e_hi)

    def set_energy_limits(self, e_lo, e_hi):
        self.rowland_circle.energy_limits = [e_lo, e_hi]
        self._set_energy_limits(e_lo, e_hi)

    def reset_energy_limits(self):
        self.rowland_circle.energy_limits = [0, 0]
        self.energy._limits = (0, 0)

    def read_energy_limits(self):
        return self.energy._limits

    def set_crystal(self, value):
        self.rowland_circle.crystal = value

    def set_hkl(self, value):
        self.rowland_circle.hkl = value

    def set_R(self, value):
        self.rowland_circle.R = value

    def read_R(self):
        return self.rowland_circle.R

    def set_roll_offset(self, value):
        self.rowland_circle.roll_offset = value

    @property
    def allowed_roll_offsets(self):
        return self.rowland_circle.allowed_roll_offsets

    def get_current_roll_offset(self):
        return self.rowland_circle.roll_offset

    def suggest_roll_offset(self, target_bragg):
        return self.rowland_circle.suggest_roll_offset(target_bragg)

    def bragg2e(self, bragg):
        return self.rowland_circle.bragg2e(bragg)

    def e2bragg(self, energy):
        return self.rowland_circle.e2bragg(energy)

    def e2reflectivity(self, energy):
        return self.rowland_circle.e2reflectivity(energy)

    def read_basic_crystal_config(self):
        return self.rowland_circle.crystal, self.rowland_circle.R, self.rowland_circle.hkl, self.rowland_circle.roll_offset

    def append_gui_update_signal(self, signal):
        self.rowland_circle.append_gui_update_signal(signal)

    @property
    def alignment_data(self):
        return self.rowland_circle.alignment_data

    def reset_alignment_data(self):
        self.rowland_circle.reset_alignment_data()

    def save_alignment_data_to_settings(self):
        self.rowland_circle.save_alignment_data_to_settings()

    @property
    def alignment_df(self):
        return pd.DataFrame(self.alignment_data)



    # def move(self, position, step_size=0.5, **kwargs):
    #     old_position = self.energy.position
    #     n_steps = int(np.abs(position - old_position) / step_size + 1)
    #     _positions = np.linspace(old_position, position, n_steps)[1:]
    #     if _positions.size == 0:
    #         ret = NullStatus()
    #     else:
    #         for _pos in _positions:
    #             print_to_gui(f'Spectrometer moving to {_pos}')
    #             ret = super().move(_pos, **kwargs)
    #     return ret
    #
    # def set(self, *args, wait=True, **kwargs):
    #     return super().set(*args, wait=wait, **kwargs)

johann_emission = JohannEmission(name='johann_emission')


class EpicsSignalAsEncoderForMotor(EpicsSignal):
    """
    A hook class to connect an inclinometer to a goniometer motor and correct its position.
    Requires a table (pandas df saved as json) for conversion between the inclinometer readback values and the motor
    positions. Loosly, this allows to use the EpicsSignal sensor as an absolute encoder for the motor.
    """

    def __init__(self, *args, conversion_json_path: str = '', motor: EpicsMotor = None, polynom_order: int = 1,
                 atol: float = 0.1, **kwargs):
        """
        args, kwargs as those for EpicsSignal;
        conversion_json_path - conversion table saved as pandas DataFrame json. Format: 1st column - motor position, 2nd
        column - sensor values;
        motor - EpicsMotor instance to connect with sensor;
        polynom_order - order for polynomial fitting of the conversion table;
        atol - absolute tolerance to the differences between the nominal motor position and sensor-inferred motor position
        """
        super().__init__(*args, **kwargs)
        self._conversion_df = pd.read_json(conversion_json_path)
        self._get_converter_parameters(polynom_order)
        self._atol = atol
        self.motor = motor
        self._callback_cid = None
        self._subscribe_to_moving_status()

    def _subscribe_to_moving_status(self):
        def callback(value, old_value, timestamp, **kwargs):
            if (int(value) == 0) and (int(old_value) == 1):

                # check if the inclinometer value read was done ~0.3 seconds (empirically derived) AFTER the motion has stopped
                for i in range(10):
                    sensor_timestamp = self.read()[self.name]['timestamp']
                    if sensor_timestamp - timestamp >= 0.3:
                        break
                    ttime.sleep(0.2)

                # if a new motion has started, then skip the checking
                if not self.motor.moving:
                    self.check_with_sensor()

        self._callback_cid = self.motor.motor_is_moving.subscribe(callback, run=False)

    def _unsubscribe_from_moving_status(self):
        if self._callback_cid is not None:
            self.motor.motor_is_moving.unsubscribe(self._callback_cid)

    def _get_converter_parameters(self, polynom_order):
        d = self._conversion_df.values
        self._p_sen2pos = np.polyfit(d[:, 1], d[:, 0], polynom_order)

    def _sen2pos(self, sensor_value):
        return np.polyval(self._p_sen2pos, sensor_value)

    @property
    def position_from_sensor(self):
        value = self.get()
        return self._sen2pos(value)

    def update_motor_position(self, new_position):
        limits = self.motor.limits
        old_offset = self.motor.user_offset.get()
        new_offset = old_offset + new_position - self.motor.position
        self.motor.user_offset.set(new_offset)
        self.motor.set_lim(*limits)

    def check_with_sensor(self):
        if (abs(self.motor.position - self.position_from_sensor) > self._atol):
            print_to_gui(
                f'Detector Goniometer 1 position disagrees with inclinometer by >{self._atol} degrees. Correcting.',
                tag='Spectrometer', add_timestamp=True)
            self.update_motor_position(self.position_from_sensor)


det_inclinometer1 = EpicsSignalAsEncoderForMotor('XF:08IDB-CT{DIODE-Box_B2:4}InCh0:Data-I', name='det_inclinometer1',
                                conversion_json_path=f'{ROOT_PATH_SHARED}/settings/json/inclinometer_data_20250113.json',
                                motor=johann_emission.motor_det_th1)


# johann_emission.energy._limits=(8004, 8068)


_johann_motor_dictionary = {
'auxxy_x':                  {'name': auxxy.x.name,                                     'description': 'Johann Crystal Assy X',        'keyword': 'Crystal Assy X',            'object': auxxy.x,                                  'group': 'spectrometer',  'user': False, 'spectrometer_kind': 'johann', 'typical_step': 2.5},
'auxxy_y':                  {'name': auxxy.y.name,                                     'description': 'Johann Detector X',            'keyword': 'Detector X',                'object': auxxy.y,                                  'group': 'spectrometer',  'user': False, 'spectrometer_kind': 'johann', 'typical_step': 2.5},
'johann_cr_main_roll':      {'name': johann_main_crystal.motor_cr_main_roll.name,      'description': 'Johann Main Crystal Roll',     'keyword': 'Main Roll',                 'object': johann_main_crystal.motor_cr_main_roll,           'group': 'spectrometer',  'user': False, 'spectrometer_kind': 'johann', 'typical_step': 100},
'johann_cr_main_yaw':       {'name': johann_main_crystal.motor_cr_main_yaw.name,       'description': 'Johann Main Crystal Yaw',      'keyword': 'Main Yaw',                  'object': johann_main_crystal.motor_cr_main_yaw,            'group': 'spectrometer',  'user': False, 'spectrometer_kind': 'johann', 'typical_step': 50},
'johann_cr_aux2_roll':      {'name': johann_aux2_crystal.motor_cr_aux2_roll.name,      'description': 'Johann Aux2 Crystal Roll',     'keyword': 'Aux2 Roll',                 'object': johann_aux2_crystal.motor_cr_aux2_roll,           'group': 'spectrometer',  'user': False, 'spectrometer_kind': 'johann', 'typical_step': 100},
'johann_cr_aux2_yaw':       {'name': johann_aux2_crystal.motor_cr_aux2_yaw.name,       'description': 'Johann Aux2 Crystal Yaw',      'keyword': 'Aux2 Yaw',                  'object': johann_aux2_crystal.motor_cr_aux2_yaw,            'group': 'spectrometer',  'user': False, 'spectrometer_kind': 'johann', 'typical_step': 50},
'johann_cr_aux2_x':         {'name': johann_aux2_crystal.motor_cr_aux2_x.name,         'description': 'Johann Aux2 Crystal X',        'keyword': 'Aux2 X',                    'object': johann_aux2_crystal.motor_cr_aux2_x,              'group': 'spectrometer',  'user': False, 'spectrometer_kind': 'johann', 'typical_step': 2500},
'johann_cr_aux2_y':         {'name': johann_aux2_crystal.motor_cr_aux2_y.name,         'description': 'Johann Aux2 Crystal Y',        'keyword': 'Aux2 Y',                    'object': johann_aux2_crystal.motor_cr_aux2_y,              'group': 'spectrometer',  'user': False, 'spectrometer_kind': 'johann', 'typical_step': 1000},
'johann_cr_aux3_roll':      {'name': johann_aux3_crystal.motor_cr_aux3_roll.name,      'description': 'Johann Aux3 Crystal Roll',     'keyword': 'Aux3 roll',                 'object': johann_aux3_crystal.motor_cr_aux3_roll,           'group': 'spectrometer',  'user': False, 'spectrometer_kind': 'johann', 'typical_step': 100},
'johann_cr_aux3_yaw':       {'name': johann_aux3_crystal.motor_cr_aux3_yaw.name,       'description': 'Johann Aux3 Crystal Yaw',      'keyword': 'Aux3 Yaw',                  'object': johann_aux3_crystal.motor_cr_aux3_yaw,            'group': 'spectrometer',  'user': False, 'spectrometer_kind': 'johann', 'typical_step': 50},
'johann_cr_aux3_x':         {'name': johann_aux3_crystal.motor_cr_aux3_x.name,         'description': 'Johann Aux3 Crystal X',        'keyword': 'Aux3 X',                    'object': johann_aux3_crystal.motor_cr_aux3_x,              'group': 'spectrometer',  'user': False, 'spectrometer_kind': 'johann', 'typical_step': 2500},
'johann_cr_aux3_y':         {'name': johann_aux3_crystal.motor_cr_aux3_y.name,         'description': 'Johann Aux3 Crystal Y',        'keyword': 'Aux3 Y',                    'object': johann_aux3_crystal.motor_cr_aux3_y,              'group': 'spectrometer',  'user': False, 'spectrometer_kind': 'johann', 'typical_step': 1000},
'johann_cr_aux4_roll':      {'name': johann_aux4_crystal.motor_cr_aux4_roll.name,      'description': 'Johann Aux4 Crystal Roll',     'keyword': 'Aux4 roll',                 'object': johann_aux4_crystal.motor_cr_aux4_roll,           'group': 'spectrometer',  'user': False, 'spectrometer_kind': 'johann', 'typical_step': 100},
'johann_cr_aux4_yaw':       {'name': johann_aux4_crystal.motor_cr_aux4_yaw.name,       'description': 'Johann Aux4 Crystal Yaw',      'keyword': 'Aux4 Yaw',                  'object': johann_aux4_crystal.motor_cr_aux4_yaw,            'group': 'spectrometer',  'user': False, 'spectrometer_kind': 'johann', 'typical_step': 50},
'johann_cr_aux4_x':         {'name': johann_aux4_crystal.motor_cr_aux4_x.name,         'description': 'Johann Aux4 Crystal X',        'keyword': 'Aux4 X',                    'object': johann_aux4_crystal.motor_cr_aux4_x,              'group': 'spectrometer',  'user': False, 'spectrometer_kind': 'johann', 'typical_step': 2500},
'johann_cr_aux4_y':         {'name': johann_aux4_crystal.motor_cr_aux4_y.name,         'description': 'Johann Aux4 Crystal Y',        'keyword': 'Aux4 Y',                    'object': johann_aux4_crystal.motor_cr_aux4_y,              'group': 'spectrometer',  'user': False, 'spectrometer_kind': 'johann', 'typical_step': 1000},
'johann_cr_aux5_roll':      {'name': johann_aux5_crystal.motor_cr_aux5_roll.name,      'description': 'Johann Aux5 Crystal Roll',     'keyword': 'Aux5 roll',                 'object': johann_aux5_crystal.motor_cr_aux5_roll,           'group': 'spectrometer',  'user': False, 'spectrometer_kind': 'johann', 'typical_step': 100},
'johann_cr_aux5_yaw':       {'name': johann_aux5_crystal.motor_cr_aux5_yaw.name,       'description': 'Johann Aux5 Crystal Yaw',      'keyword': 'Aux5 Yaw',                  'object': johann_aux5_crystal.motor_cr_aux5_yaw,            'group': 'spectrometer',  'user': False, 'spectrometer_kind': 'johann', 'typical_step': 50},
'johann_cr_aux5_x':         {'name': johann_aux5_crystal.motor_cr_aux5_x.name,         'description': 'Johann Aux5 Crystal X',        'keyword': 'Aux5 X',                    'object': johann_aux5_crystal.motor_cr_aux5_x,              'group': 'spectrometer',  'user': False, 'spectrometer_kind': 'johann', 'typical_step': 2500},
'johann_cr_aux5_y':         {'name': johann_aux5_crystal.motor_cr_aux5_y.name,         'description': 'Johann Aux5 Crystal Y',        'keyword': 'Aux5 Y',                    'object': johann_aux5_crystal.motor_cr_aux5_y,              'group': 'spectrometer',  'user': False, 'spectrometer_kind': 'johann', 'typical_step': 1000},
'johann_cr_main_bragg':     {'name': johann_main_crystal.bragg.name,                   'description': 'Johann Main Crystal Bragg',    'keyword': 'Main Bragg',                'object': johann_main_crystal.bragg,                        'group': 'spectrometer',  'user': False, 'spectrometer_kind': 'johann', 'typical_step': 0.05},
'johann_cr_aux2_bragg':     {'name': johann_aux2_crystal.bragg.name,                   'description': 'Johann Aux2 Crystal Bragg',    'keyword': 'Aux2 Bragg',                'object': johann_aux2_crystal.bragg,                        'group': 'spectrometer',  'user': False, 'spectrometer_kind': 'johann', 'typical_step': 0.05},
'johann_cr_aux3_bragg':     {'name': johann_aux3_crystal.bragg.name,                   'description': 'Johann Aux3 Crystal Bragg',    'keyword': 'Aux3 Bragg',                'object': johann_aux3_crystal.bragg,                        'group': 'spectrometer',  'user': False, 'spectrometer_kind': 'johann', 'typical_step': 0.05},
'johann_cr_aux4_bragg':     {'name': johann_aux4_crystal.bragg.name,                   'description': 'Johann Aux4 Crystal Bragg',    'keyword': 'Aux4 Bragg',                'object': johann_aux4_crystal.bragg,                        'group': 'spectrometer',  'user': False, 'spectrometer_kind': 'johann', 'typical_step': 0.05},
'johann_cr_aux5_bragg':     {'name': johann_aux5_crystal.bragg.name,                   'description': 'Johann Aux5 Crystal Bragg',    'keyword': 'Aux5 Bragg',                'object': johann_aux5_crystal.bragg,                        'group': 'spectrometer',  'user': False, 'spectrometer_kind': 'johann', 'typical_step': 0.05},
'johann_det_focus':         {'name': johann_det_arm.det_focus.name,                    'description': 'Johann Detector Focus',        'keyword': 'Detector Focus',            'object': johann_det_arm.det_focus,                 'group': 'spectrometer',  'user': False, 'spectrometer_kind': 'johann', 'typical_step': 5},
'johann_det_bragg':         {'name': johann_det_arm.bragg.name,                        'description': 'Johann Detector Bragg',        'keyword': 'Detector Bragg',            'object': johann_det_arm.bragg,                        'group': 'spectrometer',  'user': False, 'spectrometer_kind': 'johann', 'typical_step': 0.1},
'johann_x':                 {'name': johann_spectrometer_x.x.name,                     'description': 'Johann Spectrometer X',        'keyword': 'Spectrometer X',            'object': johann_spectrometer_x.x,                  'group': 'spectrometer',  'user': False, 'spectrometer_kind': 'johann', 'typical_step': 2.5},
'johann_bragg_angle':       {'name': johann_spectrometer.bragg.name,                   'description': 'Johann Global Bragg Angle',    'keyword': 'Global Bragg Angle',        'object': johann_spectrometer.bragg,                'group': 'spectrometer',  'user': True,  'spectrometer_kind': 'johann', 'typical_step': 0.05},
'johann_energy':            {'name': johann_emission.energy.name,                      'description': 'Johann Emission Energy',       'keyword': 'Emission Energy',           'object': johann_emission.energy,                   'group': 'spectrometer',  'user': True,  'spectrometer_kind': 'johann', 'typical_step': 1},
}

motor_dictionary = {**motor_dictionary, **_johann_motor_dictionary}
