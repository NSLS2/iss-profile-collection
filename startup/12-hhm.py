print(ttime.ctime() + ' >>>> ' + __file__)
from xas.pid import PID
from xas.image_analysis import determine_beam_position_from_fb_image
#

#



class HHMTrajDesc(Device):
    filename = Cpt(EpicsSignal, '-Name')
    elem = Cpt(EpicsSignal, '-Elem')
    edge = Cpt(EpicsSignal, '-Edge')
    e0 = Cpt(EpicsSignal, '-E0')




class HHM(Device):
    _default_configuration_attrs = ('pitch', 'roll', 'theta', 'y', 'energy')
    _default_read_attrs = ('pitch', 'roll', 'theta', 'y', 'energy')
    "High Heat Load Monochromator"
    ip = '10.66.58.106'

    pitch = Cpt(EpicsMotor, 'Mono:HHM-Ax:P}Mtr', kind='hinted')
    roll = Cpt(EpicsMotor, 'Mono:HHM-Ax:R}Mtr', kind='hinted')
    y = Cpt(StuckingEpicsMotor, 'Mono:HHM-Ax:Y}Mtr', kind='hinted')
    theta = Cpt(EpicsMotor, 'Mono:HHM-Ax:Th}Mtr', kind='hinted')
    # theta_speed = Cpt(EpicsSignal, 'Mono:HHM-Ax:Th}Mtr.VMAX', kind='hinted')
    # theta_speed_max = Cpt(EpicsSignal, 'Mono:HHM-Ax:Th}Mtr.VELO', kind='hinted')
    energy = Cpt(StuckingEpicsMotorThatFlies, 'Mono:HHM-Ax:E}Mtr', kind=Kind.hinted)

    main_motor_res = Cpt(EpicsSignal, 'Mono:HHM-Ax:Th}Mtr.MRES')

    # The following are related to trajectory motion
    lut_number = Cpt(EpicsSignal, 'MC:06}LUT-Set')
    lut_number_rbv = Cpt(EpicsSignal, 'MC:06}LUT-Read')
    lut_start_transfer = Cpt(EpicsSignal, 'MC:06}TransferLUT')
    lut_transfering = Cpt(EpicsSignal, 'MC:06}TransferLUT-Read')
    trajectory_loading = Cpt(EpicsSignal, 'MC:06}TrajLoading')
    traj_mode = Cpt(EpicsSignal, 'MC:06}TrajFlag1-Set')
    traj_mode_rbv = Cpt(EpicsSignal, 'MC:06}TrajFlag1-Read')
    enable_ty = Cpt(EpicsSignal, 'MC:06}TrajFlag2-Set')
    enable_ty_rbv = Cpt(EpicsSignal, 'MC:06}TrajFlag2-Read')
    cycle_limit = Cpt(EpicsSignal, 'MC:06}TrajRows-Set')
    cycle_limit_rbv = Cpt(EpicsSignal, 'MC:06}TrajRows-Read')
    enable_loop = Cpt(EpicsSignal, 'MC:06}TrajLoopFlag-Set')
    enable_loop_rbv = Cpt(EpicsSignal, 'MC:06}TrajLoopFlag')

    prepare_trajectory = Cpt(EpicsSignal, 'MC:06}PrepareTraj')
    trajectory_ready = Cpt(EpicsSignal, 'MC:06}TrajInitPlc-Read')
    start_trajectory = Cpt(EpicsSignal, 'MC:06}StartTraj')
    stop_trajectory = Cpt(EpicsSignal, 'MC:06}StopTraj')
    trajectory_running = Cpt(EpicsSignal,'MC:06}TrajRunning', write_pv='MC:06}TrajRunning-Set')
    trajectory_progress = Cpt(EpicsSignal,'MC:06}TrajProgress')
    trajectory_name = Cpt(EpicsSignal, 'MC:06}TrajFilename')

    traj1 = Cpt(HHMTrajDesc, 'MC:06}Traj:1')
    traj2 = Cpt(HHMTrajDesc, 'MC:06}Traj:2')
    traj3 = Cpt(HHMTrajDesc, 'MC:06}Traj:3')
    traj4 = Cpt(HHMTrajDesc, 'MC:06}Traj:4')
    traj5 = Cpt(HHMTrajDesc, 'MC:06}Traj:5')
    traj6 = Cpt(HHMTrajDesc, 'MC:06}Traj:6')
    traj7 = Cpt(HHMTrajDesc, 'MC:06}Traj:7')
    traj8 = Cpt(HHMTrajDesc, 'MC:06}Traj:8')
    traj9 = Cpt(HHMTrajDesc, 'MC:06}Traj:9')



    if shutdown:
        # use this when softioc-piE712-02.service is down:
        fb_status = Signal(name='fb_status')
        fb_center = Signal(name='fb_center')
        fb_line = Signal(name='fb_line')
        fb_nlines = Signal(name='fb_nlines')
        fb_nmeasures = Signal(name='fb_nmeasures')
        fb_pcoeff = Signal(name='fb_pcoeff')
        fb_hostname = Signal(name='fb_hostname')
        fb_heartbeat = Signal(name='fb_heartbeat')
        fb_status_err = Signal(name='fb_status_err')
        fb_status_msg = Signal(name='fb_status_msg')
    else:
        # use this when softioc-piE712-02.service is up:
        fb_status = Cpt(EpicsSignal, 'Mono:HHM-Ax:P}FB-Sts')
        fb_center = Cpt(EpicsSignal, 'Mono:HHM-Ax:P}FB-Center')
        fb_line = Cpt(EpicsSignal, 'Mono:HHM-Ax:P}FB-Line')
        fb_nlines = Cpt(EpicsSignal, 'Mono:HHM-Ax:P}FB-NLines')
        fb_nmeasures = Cpt(EpicsSignal, 'Mono:HHM-Ax:P}FB-NMeasures')
        fb_pcoeff = Cpt(EpicsSignal, 'Mono:HHM-Ax:P}FB-PCoeff')
        fb_hostname = Cpt(EpicsSignal, 'Mono:HHM-Ax:P}FB-Hostname')
        fb_heartbeat = Cpt(EpicsSignal, 'Mono:HHM-Ax:P}FB-Heartbeat')
        fb_status_err = Cpt(EpicsSignal, 'Mono:HHM-Ax:P}FB-Err')
        fb_status_msg = Cpt(EpicsSignal, 'Mono:HHM-Ax:P}FB-StsMsg', string=True)



    angle_offset = Cpt(EpicsSignal, 'Mono:HHM-Ax:E}Offset', limits=True)
    home_y = Cpt(EpicsSignal, 'MC:06}Home-HHMY')
    y_precise = Cpt(InfirmStuckingEpicsMotor, 'Mono:HHM-Ax:Y}Mtr', kind='hinted')

    servocycle = 16000

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        try:
            self.pulses_per_deg = 1 / self.main_motor_res.get()
        except ZeroDivisionError:
            self.pulses_per_deg = -1

        # self.enc = enc
        self._preparing = None
        self._starting = None
        self.y_precise.append_homing_pv(self.home_y)
        # self.y_precise.set_low_lim(low_lim=8.5)

        self.energy.append_flying_status_pv(self.trajectory_running)

        self.flying_status = None


    # def stage(self):
    #     print(f'{ttime.ctime()} >>>>> HHM STAGED')
    #     return super().stage()

    def _ensure_mono_faces_down(self):
        _, emax = trajectory_manager.read_trajectory_limits()
        hhm.energy.move(emax + 50)

    def prepare(self):
        def callback(value, old_value, **kwargs):
            if int(round(old_value)) == 1 and int(round(value)) == 0:
                if self._preparing or self._preparing is None:
                    self._preparing = False
                    return True
                else:
                    self._preparing = True
            return False

        status = SubscriptionStatus(self.trajectory_ready, callback)

        # print_to_gui(f'Mono trajectory prepare starting...', add_timestamp=True, ntabs=2)

        self._ensure_mono_faces_down()
        print_to_gui(f'Prepare trajectory starting...', add_timestamp=True, tag='Monochromator')
        self.prepare_trajectory.set('1')  # Yes, the IOC requires a string.
        print_to_gui(f'Perpare trajectory complete.', add_timestamp=True, tag='Monochromator')
        # status.wait()
        # print_to_gui(f'Ensuring mono faces down (starting)', add_timestamp=True, ntabs=2)
        # self._ensure_mono_faces_down()
        # print_to_gui(f'Mono trajectory prepare done', add_timestamp=True, ntabs=2)
        self.flying_status = None
        return status

    def kickoff(self):
        def callback(value, old_value, **kwargs):

            if int(round(old_value)) == 1 and int(round(value)) == 0:
                if self._starting or self._starting is None:
                    self._starting = False
                    return True
                else:
                    self._starting = True
                return False

        self.flying_status = SubscriptionStatus(self.trajectory_running, callback)
        self.start_trajectory.set('1')
        return self.flying_status

    def complete(self):
        self.flying_status = None

    def abort_trajectory(self):
        is_flying = (self.flying_status is not None) and (not self.flying_status.done)
        self.stop_trajectory.put('1')
        if is_flying:
            print_to_gui('Stopping trajectory ... ', tag='HHM')
            if not self.flying_status.done:
                self.flying_status.set_finished()
            print_to_gui('Stopped trajectory', tag='HHM')
        return is_flying


    def home_y_pos(self):
        self.home_y.put('1')


    def set_new_angle_offset(self, value, error_message_func=None):
        try:
            self.angle_offset.put(float(value))
            return True
        except Exception as exc:
            if type(exc) == ophyd_utils.errors.LimitError:
                msg = 'HHM limit error'
                print_to_gui(f'[Energy calibration] {msg}.'.format(exc))
                if error_message_func is not None:
                    error_message_func(msg)
            else:
                msg = f'HHM error. Something went wrong, not the limit: {exc}'
                print_to_gui(f'[Energy calibration] {msg}')
                if error_message_func is not None:
                    error_message_func(msg)
            return False

    def calibrate(self, energy_nominal, energy_actual, error_message_func=None):
        offset_actual = xray.energy2encoder(energy_actual, hhm.pulses_per_deg) / hhm.pulses_per_deg
        offset_nominal = xray.energy2encoder(energy_nominal, hhm.pulses_per_deg) / hhm.pulses_per_deg
        angular_offset_shift = offset_actual - offset_nominal
        new_angular_offset = self.angle_offset.value - angular_offset_shift
        return self.set_new_angle_offset(new_angular_offset, error_message_func=error_message_func)

    def get_angle_offset_deg_str(self):
        return f'{np.round(hhm.angle_offset.get() * 180 / np.pi, 3)} deg'

    def get_mono_encoder_resolution_str(self):
        return f'{(np.round(hhm.main_motor_res.get() * np.pi / 180 * 1e9))} nrad'

    @property
    def current_trajectory_duration(self):
        # calls function from trajectory manager defined in the next file :'(
        return trajectory_manager.current_trajectory_duration

hhm = HHM('XF:08IDA-OP{', name='hhm')
# TODO: move to the HHM class definition.
hhm_z_home = Cpt(EpicsSignal,'XF:08IDA-OP{MC:06}Home-HHMY')

# Try to read it first time to avoid the generic 'object' to be returned
# as an old value from hhm.trajectory_running._readback.
try:
    hhm.wait_for_connection()
    _ = hhm.trajectory_ready.read()
    _ = hhm.trajectory_running.read()
except:
    pass


# hhm.hints = {'fields': ['hhm_energy', 'hhm_pitch', 'hhm_roll', 'hhm_theta', 'hhm_y']}
# hinted also is automatically set as read so no need to set read_attrs
hhm.energy.kind = 'hinted'
hhm.pitch.kind = 'hinted'
hhm.roll.kind = 'hinted'
hhm.theta.kind = 'hinted'
hhm.y.kind = 'hinted'

hhm.read_attrs = ['pitch', 'roll', 'theta', 'y', 'energy']

