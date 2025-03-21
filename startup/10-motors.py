print(ttime.ctime() + ' >>>> ' + __file__)



from ophyd import utils as ophyd_utils
from xas import xray

# _EpicsMotor = EpicsMotor
# _EpicsMotor = EpicsMotorWithTweaking

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

    def __init__(self, *args, pos_dict=None, **kwargs):
        super().__init__(*args, **kwargs)
        if pos_dict is None:
            pos_dict = {}
        self.pos_dict = pos_dict

    def get_current_stripe(self):
        pos = self.x.position
        for k, value in self.pos_dict.items():
            if np.isclose(pos, value, atol=15):
                return k
        return 'undefined'


cm1 = Mirror('XF:08IDA-OP{Mir:1-CM', name='cm1', pos_dict={'Rh' : -40, 'Si' : 0, 'Pt': 40})
cm2 = Mirror('XF:08IDA-OP{Mir:2-CM', name='cm2', pos_dict={'Pt' : -20, 'Rh' : 20})
fm = Mirror('XF:08IDA-OP{Mir:FM', name='fm', pos_dict={'Pt' : -60, 'Rh': 60})

# h = 20 # mm
#
#
# def fix_exit_trig_formula(theta):
#     return h/(2*np.cos(np.deg2rad(theta)))




class HRM(Device):
    """High Resolution Monochromator"""
    theta = Cpt(EpicsMotor, '-Ax:Th}Mtr')
    y = Cpt(StuckingEpicsMotor, '-Ax:Y}Mtr')
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
    y = Cpt(StuckingEpicsMotor, 'Mir:HRM:TY}Mtr')


    #     # print_to_gui('WARNING HHRM STRIPE IS NOT DEFINED')
    #     # return 'undefined'
    #
    # def get_current_stripe(self):
    #     pos = self.hor_translation.user_readback.get()
    #     if np.isclose(pos, 0, atol=15):
    #         stripe = 'Si'
    #     elif np.isclose(pos, -80, atol=15):
    #         stripe = 'Pt'
    #     elif np.isclose(pos, 80, atol=15):
    #         stripe = 'Rh'
    #     else:
    #         stripe = 'undefined'
    #     return stripe

    def __init__(self, *args, pos_dict=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.x = self.hor_translation
        if pos_dict is None:
            pos_dict = {}
        self.pos_dict = pos_dict

    def get_current_stripe(self):
        pos = self.x.position
        for k, value in self.pos_dict.items():
            if np.isclose(pos, value, atol=15):
                return k
        return 'undefined'

    @property
    def current_stripe(self):
        return self.get_current_stripe()

hhrm = HHRM('XF:08IDB-OP{', name='hhrm', pos_dict={'Pt' : -80, 'Si' : 0, 'Rh': 80})


class SampleXY(Device):
    x = Cpt(EpicsMotor, '-Ax:X}Mtr')
    y = Cpt(EpicsMotor, '-Ax:Y}Mtr')

samplexy = SampleXY('XF:08IDB-OP{SampleXY', name='samplexy') # this it the sample stage in hutch B2/C
giantxy = SampleXY('XF:08IDB-OP{Stage:Sample', name='giantxy') # this is the important motor

auxxy = SampleXY('XF:08IDB-OP{Stage:Aux1', name='auxxy')


class DetStageXYZ(Device):
    x = Cpt(EpicsMotor, '-Ax:X}Mtr')
    y = Cpt(EpicsMotor, '-Ax:Y}Mtr')
    z = Cpt(EpicsMotor, '-Ax:Z}Mtr')

detstage = DetStageXYZ('XF:08IDB-OP{Stage:Det', name='detstage')


class Usermotor(Device):
    pos = Cpt(EpicsMotor, '}Mtr')


usermotor1 = Usermotor('XF:08IDB-OP{Misc-Ax:1', name='usermotor1')
usermotor2 = Usermotor('XF:08IDB-OP{Misc-Ax:2', name='usermotor2')
usermotor3 = Usermotor('XF:08IDB-OP{Misc-Ax:3', name='usermotor3')

attenuator_motor = Usermotor('XF:08IDB-OP{Misc-Ax:4', name='attenuator_motor')

det1_stage_x = Usermotor('XF:08IDB-OP{Analyzer-Ax:Y', name='det1_stage_x')
det1_stage_y = Usermotor('XF:08IDB-OP{Analyzer-Ax:Z', name='det1_stage_y')


det2_stage_x = Usermotor('XF:08IDB-OP{DetStage:2-Ax:X', name='det2_stage_x')
det2_stage_y = Usermotor('XF:08IDB-OP{DetStage:2-Ax:Y', name='det2_stage_y')



usermotor2.wait_for_connection()
usermotor3.wait_for_connection()

class IonChamberMotor(Device):
    pos = Cpt(EpicsMotor, '}Mtr')

i0_y = IonChamberMotor('XF:08IDB-OP{IC-Ax:Y:1', name='i0_y')
it_y = IonChamberMotor('XF:08IDB-OP{IC-Ax:Y:2', name='it_y')
ir_y = IonChamberMotor('XF:08IDB-OP{IC-Ax:Y:3', name='ir_y')



class Bender(Device):
    pos = Cpt(EpicsMotor, '}Mtr')
    load_cell = EpicsSignalRO('XF:08IDA-OP{Mir:CM-Ax:Bender}W-I', name='bender_loading')
# TODO: change the name of the class/object to BenderCM2/bender_cm2
bender = Bender('XF:08IDA-OP{Mir:CM-Bender', name='bender')
# fm_bender = Bender('XF:08IDA-OP{Mir:CM-Bender', name='bender')

class BenderFM(Device):
    pos = Cpt(EpicsMotor, '}Mtr')
    load_cell = EpicsSignalRO('XF:08IDA-OP{Mir:FM-Ax:Bender}W-I', name='bender_fm_loading')
bender_fm = BenderFM('XF:08IDA-OP{Mir:FM-Bender', name='bender_fm')

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


class Hutch_Slits(Device):
    top = Cpt(EpicsMotor, '1}Mtr')
    bottom = Cpt(EpicsMotor, '2}Mtr')
    inboard = Cpt(EpicsMotor, '3}Mtr')
    outboard = Cpt(EpicsMotor, '4}Mtr')

hutch_b_slits = Hutch_Slits('XF:08IDB-OP{Misc:2-Ax:', name='hutch_b_slits')

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


class SampleStage(Device):
    x = Cpt(EpicsMotor, 'XF:08IDB-OP{Stage:Sample-Ax:X}Mtr')
    y = Cpt(EpicsMotor, 'XF:08IDB-OP{Stage:Sample-Ax:Y}Mtr')
    z = Cpt(EpicsMotor, 'XF:08IDB-OP{Misc-Ax:2}Mtr')
    # z = Cpt(EpicsMotorThatCannotReachTheTargetProperly, 'XF:08IDB-OP{Misc-Ax:2}Mtr')
    th = Cpt(EpicsMotor, 'XF:08IDB-OP{Gon:Th:1}Mtr')
    def mv(self, pos_dict, wait=True):
        '''
        pos_dict = {'x' : -300,
                      'y' : -60}
        or
        pos_dict = {'th' : -800}
        '''
        st_list = []
        for axis, pos in pos_dict.items():
            motor = getattr(self, axis)
            # print(f'{motor.user_setpoint.value=}')
            if (not wait) or (not motor.moving):

                st = motor.move(pos, wait=False)
                st_list.append(st)
        if len(st_list) > 0:
            return combine_status_list(st_list)
        else:
            return NullStatus()

    def mvr(self, delta_dict, wait=True):
        pos_dict = {}
        for axis, delta in delta_dict.items():
            motor = getattr(self, axis)
            # if not motor.moving:
                # pos_dict[axis] = motor.position + delta
            pos_dict[axis] = motor.user_setpoint.value + delta
        return self.mv(pos_dict, wait=wait)

    def positions(self, *axes, prefix=''):
        if len(axes) == 0:
            axes = ['x', 'y', 'z', 'th']
        pos_dict = {}
        for axis in axes:
            motor = getattr(self, axis)
            if len(prefix) > 0:
                key = f'{prefix}_{axis}'
            else:
                key = axis
            pos_dict[key] = motor.position
        return pos_dict

sample_stage = SampleStage(name='sample_stage')
# samplexy = SampleXY('XF:08IDB-OP{SampleXY', name='samplexy')
# giantxy = SampleXY('XF:08IDB-OP{Stage:Sample', name='giantxy') # this is the important motor


class FIPSpectrometerMotor(Device):
    x = Cpt(EpicsMotor, ':X}Mtr')
    y = Cpt(EpicsMotor, ':Y}Mtr')

fip_spectrometer_crystal = FIPSpectrometerMotor('XF:08IDB-OP{FIP-VHS:Stage1-Ax', name='fip_spectrometer_crystal')
fip_spectrometer_detector = FIPSpectrometerMotor('XF:08IDB-OP{FIP-VHS:Stage2-Ax', name='fip_spectrometer_detector')


class VonHamosSpectrometerMotors(Device):
    assm_y = Cpt(EpicsMotor, ':1}Mtr')
    assm_x = Cpt(EpicsMotor, ':2}Mtr')
    arc = Cpt(EpicsMotor, ':3}Mtr')

    crystal_x = Cpt(EpicsMotor, ':6}Mtr')
    crystal_yaw = Cpt(EpicsMotor, ':7}Mtr')
    crystal_pitch = Cpt(EpicsMotor, ':8}Mtr')

vonhamos_motors = VonHamosSpectrometerMotors('XF:08IDB-OP{MC:3-Ax', name='vonhamos_motors')
