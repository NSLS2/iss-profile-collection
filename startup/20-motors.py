from ophyd import (EpicsMotor, Device, Component as Cpt,
                   EpicsSignal)
import numpy as np


class Mirror(Device):
    pitch = Cpt(EpicsMotor, '-Ax:P}Mtr')
    roll = Cpt(EpicsMotor, '-Ax:R}Mtr')
    xd = Cpt(EpicsMotor, '-Ax:XD}Mtr')  # downstream
    xu = Cpt(EpicsMotor, '-Ax:XU}Mtr')  # upstream
    x = Cpt(EpicsMotor, '-Ax:X}Mtr')
    ydi = Cpt(EpicsMotor, '-Ax:YDI}Mtr')  # downstream inboard
    ydo = Cpt(EpicsMotor, '-Ax:YDO}Mtr')  # downstream outboard
    yu = Cpt(EpicsMotor, '-Ax:YU}Mtr')
    yaw = Cpt(EpicsMotor, '-Ax:Yaw}Mtr')
    y = Cpt(EpicsMotor, '-Ax:Y}Mtr')
    

cm1 = Mirror('XF:08IDA-OP{Mir:1-CM', name='cm1')
cm2 = Mirror('XF:08IDA-OP{Mir:2-CM', name='cm2')
fm = Mirror('XF:08IDA-OP{Mir:FM', name='fm')

h = 20 # mm


def fix_exit_trig_formula(theta):
    return h/(2*np.cos(np.deg2rad(theta)))


def energy2theta(energy):
    return np.rad2deg(np.arcsin(12400/energy/2/3.1356))


class HHMTrajDesc(Device):
    filename = Cpt(EpicsSignal, '-Name')
    elem = Cpt(EpicsSignal, '-Elem')
    edge = Cpt(EpicsSignal, '-Edge')
    e0 = Cpt(EpicsSignal, '-E0')


class HHM(Device):
    _default_configuration_attrs = ('pitch', 'roll', 'theta', 'y', 'energy')
    _default_read_attrs = ('pitch', 'roll', 'theta', 'y', 'energy')
    "High Heat Load Monochromator"
    ip = '10.8.2.86'
    traj_filepath = '/GPFS/xf08id/trajectory/'

    pitch = Cpt(EpicsMotor, 'Mono:HHM-Ax:P}Mtr', kind='hinted')
    roll = Cpt(EpicsMotor, 'Mono:HHM-Ax:R}Mtr', kind='hinted')
    y = Cpt(EpicsMotor, 'Mono:HHM-Ax:Y}Mtr', kind='hinted')
    theta = Cpt(EpicsMotor, 'Mono:HHM-Ax:Th}Mtr', kind='hinted')
    energy = Cpt(EpicsMotor, 'Mono:HHM-Ax:E}Mtr', kind='hinted')

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

    fb_status = Cpt(EpicsSignal, 'Mono:HHM-Ax:P}FB-Sts')
    fb_center = Cpt(EpicsSignal, 'Mono:HHM-Ax:P}FB-Center')
    fb_line = Cpt(EpicsSignal, 'Mono:HHM-Ax:P}FB-Line')
    fb_nlines = Cpt(EpicsSignal, 'Mono:HHM-Ax:P}FB-NLines')
    fb_nmeasures = Cpt(EpicsSignal, 'Mono:HHM-Ax:P}FB-NMeasures')
    fb_pcoeff = Cpt(EpicsSignal, 'Mono:HHM-Ax:P}FB-PCoeff')

    angle_offset = Cpt(EpicsSignal, 'Mono:HHM-Ax:E}Offset', limits=True)

    def __init__(self, *args, enc = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.pulses_per_deg = 1/self.main_motor_res.value
        self.enc = enc


hhm = HHM('XF:08IDA-OP{', enc = pb9.enc1, name='hhm')
#hhm.hints = {'fields': ['hhm_energy', 'hhm_pitch', 'hhm_roll', 'hhm_theta', 'hhm_y']}
# hinted also is automatically set as read so no need to set read_attrs
#hhm.energy.kind = 'hinted'
#hhm.pitch.kind = 'hinted'
#hhm.roll.kind = 'hinted'
#hhm.theta.kind = 'hinted'
#hhm.y.kind = 'hinted'

hhm.read_attrs = ['pitch', 'roll', 'theta', 'y', 'energy']


class HRM(Device):
    """High Resolution Monochromator"""
    theta = Cpt(EpicsMotor, '-Ax:Th}Mtr')
    y = Cpt(EpicsMotor, '-Ax:Y}Mtr')
    pitch = Cpt(EpicsMotor, '-Ax:P}Mtr')

hrm = HRM('XF:08IDA-OP{Mono:HRM', name='hrm')


class HHRM(Device):
    """High Harmonics Rejection Mirror"""
    yu = Cpt(EpicsMotor, 'Mir:HRM:YU}Mtr')
    yd1 = Cpt(EpicsMotor, 'Mir:HRM:YD1}Mtr')
    yd2 = Cpt(EpicsMotor, 'Mir:HRM:YD2}Mtr')
    mir_pitch = Cpt(EpicsMotor, 'Mir:HRM:P}Mtr')
    hor_translation = Cpt(EpicsMotor, 'Mir:HRM:H}Mtr')

    table_pitch = Cpt(EpicsMotor, 'Mir:HRM:TP}Mtr')
    y = Cpt(EpicsMotor, 'Mir:HRM:TY}Mtr')


hhrm = HHRM('XF:08IDB-OP{', name='hhrm')


class SampleXY(Device):
    x = Cpt(EpicsMotor, '-Ax:X}Mtr')
    y = Cpt(EpicsMotor, '-Ax:Y}Mtr')

samplexy = SampleXY('XF:08IDB-OP{SampleXY', name='samplexy')
giantxy = SampleXY('XF:08IDB-OP{PPMAC:Stage', name='giantxy')


class FilterBox(Device):
    y = Cpt(EpicsMotor, '-Ax:Y}Mtr')
    pos1 = Cpt(EpicsSignal, '}Fltr1:In-Sts')
    pos2 = Cpt(EpicsSignal, '}Fltr2:In-Sts')
    pos3 = Cpt(EpicsSignal, '}Fltr3:In-Sts')
    pos4 = Cpt(EpicsSignal, '}Fltr4:In-Sts')
    pos5 = Cpt(EpicsSignal, '}Fltr5:In-Sts')

    pos_limit = Cpt(EpicsSignal, '-Ax:Y}OvrP-Sts')
    neg_limit = Cpt(EpicsSignal, '-Ax:Y}OvrN-Sts')

filterbox = FilterBox('XF:08IDA-OP{Fltr:FB', name='filterbox')


class Slits(Device):
    v_gap = Cpt(EpicsMotor, '-Ax:V-GAP}Mtr')
    v_pos = Cpt(EpicsMotor, '-Ax:V-POS}Mtr')
    hor_in = Cpt(EpicsMotor, '-Ax:XI}Mtr')
    hor_out = Cpt(EpicsMotor, '-Ax:XO}Mtr')

slits = Slits('XF:08IDB-OP{Slt', name='slits')


class HuberStage(Device):
    y = Cpt(EpicsMotor, '-Ax:Y}Mtr')
    pitch = Cpt(EpicsMotor, '-Ax:P}Mtr')
    z = Cpt(EpicsMotor, '-Ax:Z}Mtr')

huber_stage = HuberStage('XF:08IDB-OP{Analyzer', name='huber_stage')


class XBIC(Device):
    dac1 = Cpt(EpicsSignal, 'MC:XBIC}DAC1-I', write_pv='MC:XBIC}DAC1R-SP')
    dac2 = Cpt(EpicsSignal, 'MC:XBIC}DAC2-I', write_pv='MC:XBIC}DAC2R-SP')

#xbic = XBIC('XF:08IDB-OP{', name='xbic')

class SixAxesStage(Device):
    x = Cpt(EpicsMotor, '-Ax:X}Mtr')
    y = Cpt(EpicsMotor, '-Ax:Y}Mtr')
    z = Cpt(EpicsMotor, '-Ax:Z}Mtr')
    pitch = Cpt(EpicsMotor, '-Ax:P}Mtr')
    yaw = Cpt(EpicsMotor, '-Ax:Yaw}Mtr')
    roll = Cpt(EpicsMotor, '-Ax:R}Mtr')

six_axes_stage = SixAxesStage('XF:08IDB-OP{PCL', name='six_axes_stage')

class FoilWheel(Device):
    wheel1 = Cpt(EpicsMotor, '1:Rot}Mtr')
    wheel2 = Cpt(EpicsMotor, '2:Rot}Mtr')

foil_wheel = FoilWheel('XF:08IDB-OP{FoilWheel', name='foil_wheel')

class GonioMeter(Device):
    th1 = Cpt(EpicsMotor, ':1}Mtr')
    th2 = Cpt(EpicsMotor, ':2}Mtr')

gonio_meter = GonioMeter('XF:08IDB-OP{Gon:Th', name='gonio_meter')

class CathodeChamberZeta(Device):
    smedley_zeta = Cpt(EpicsMotor, ':Rot}Mtr')

cathode_zeta = CathodeChamberZeta('XF:08IDB-OP{JS-Stg-Ax', name='cathode_zeta')

class CathodeChamberTrans(Device):
    smedley_x = Cpt(EpicsMotor, ':X}Mtr')
    smedley_y = Cpt(EpicsMotor, ':Y}Mtr')

cathode_trans = CathodeChamberTrans('XF:08IDB-OP{PPMAC:StageJS-Ax', name='cathode_trans')


