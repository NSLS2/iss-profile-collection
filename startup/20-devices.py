print(ttime.ctime() + ' >>>> ' + __file__)


from ophyd import (ProsilicaDetector, SingleTrigger, Component as Cpt, Device,
                   EpicsSignal, EpicsSignalRO, ImagePlugin, StatsPlugin, ROIPlugin,
                   DeviceStatus, Signal)
from nslsii.devices import TwoButtonShutter
import bluesky.plans as bp
from ophyd.status import SubscriptionStatus

import time
import datetime

class MFC(Device):
    flow = Cpt(EpicsSignal, '-RB', write_pv='-SP')


gas_he = MFC('XF:08IDB-OP{IC}FLW:He', name='gas_he')
gas_he.flow.tolerance = 0.05
gas_n2 = MFC('XF:08IDB-OP{IC}FLW:N2', name='gas_n2')
gas_n2.flow.tolerance = 0.05





class DeviceWithNegativeReadBack(Device):
    read_pv = Cpt(EpicsSignal, 'V-Sense')
    write_pv =  Cpt(EpicsSignal, 'V-Set')
    switch_pv = Cpt(EpicsSignal, 'Switch')
    def __init__(self,*args, **kwargs):
        super().__init__(*args, **kwargs)
        self._moving = None


    def set(self, value, switch=False):

        def callback(*args,**kwargs):
            if self._moving and  abs(abs(self.read_pv.get())-abs(self.write_pv.get())) < 2:
                self._moving = False
                return True
            else:
                self._moving = True
                return False

        status = SubscriptionStatus(self.read_pv, callback)

        if switch:
            self.switch_pv.set(value)
        else:
            self.write_pv.set(value)

        return status


class WPS(Device):
    hv300 = Cpt(DeviceWithNegativeReadBack, 'HV:u300}')
    hv301 = Cpt(DeviceWithNegativeReadBack, 'HV:u301}')
    hv302 = Cpt(DeviceWithNegativeReadBack, 'HV:u302}')
    hv303 = Cpt(DeviceWithNegativeReadBack, 'HV:u303}')
    hv304 = Cpt(DeviceWithNegativeReadBack, 'HV:u304}')
    hv305 = Cpt(DeviceWithNegativeReadBack, 'HV:u305}')
    hv306 = Cpt(DeviceWithNegativeReadBack, 'HV:u306}')
    hv307 = Cpt(DeviceWithNegativeReadBack, 'HV:u307}')

    '''
    lv0 = Cpt(EpicsSignal, '-LV:u0}V-Sense', write_pv='-LV:u0}V-Set')
    lv1 = Cpt(EpicsSignal, '-LV:u1}V-Sense', write_pv='-LV:u1}V-Set')
    lv2 = Cpt(EpicsSignal, '-LV:u2}V-Sense', write_pv='-LV:u2}V-Set')
    lv3 = Cpt(EpicsSignal, '-LV:u3}V-Sense', write_pv='-LV:u3}V-Set')
    lv4 = Cpt(EpicsSignal, '-LV:u4}V-Sense', write_pv='-LV:u4}V-Set')
    lv5 = Cpt(EpicsSignal, '-LV:u5}V-Sense', write_pv='-LV:u5}V-Set')
    lv6 = Cpt(EpicsSignal, '-LV:u6}V-Sense', write_pv='-LV:u6}V-Set')
    lv7 = Cpt(EpicsSignal, '-LV:u7}V-Sense', write_pv='-LV:u7}V-Set')
    '''

wps1 = WPS('XF:08IDB-OP{WPS:01-', name='wps1')


class ShutterMotor(Device):
    exposure_time = Cpt(Signal)
    def __init__(self, shutter_type, motor, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # self.name = name
        self.shutter_type = shutter_type
        self.motor = motor
        self.function_call = None
        self._start_time = None


        if motor.connected:
            self.output = motor.pos
            self.open_pos = 0.00
            self.open_range = 1
            self.closed_pos = 1.1
            # self.update_state(self.current_pos)
            self.output.subscribe(self.update_state) # , event_type='done_moving'
        else:
            self.state = 'unknown'

    @property
    def current_pos(self):
        return self.output.position

    def update_state(self, pvname=None, value=None, char_value=None, **kwargs):
        if self.current_pos-self.open_pos > self.open_range:
            self.state = 'closed'
        else:
            self.state = 'open'

        if self.function_call is not None:
            self.function_call(pvname=pvname, value=value, char_value=char_value, **kwargs)

    def subscribe(self, function):
        self.function_call = function

    def unsubscribe(self):
        self.function_call = None

    def open(self, printing=True, time_opening=False):
        if printing: print_to_gui(f'{ttime.ctime()} >>> {self.name} opening...')
        if time_opening: self._start_time = ttime.time()
        self.output.move(self.open_pos, wait=True)

    def close(self, printing=True):
        if printing: print_to_gui(f'{ttime.ctime()} >>> {self.name} closing')
        self.output.move(self.closed_pos, wait=True)
        if self._start_time is not None:
            self.exposure_time.put(ttime.time() - self._start_time)
            self._start_time = None
            print_to_gui(f'. Total exposure time: {self.exposure_time.get():0.2f} s')
        print()


    def open_plan(self, printing=True):
        if printing: print_to_gui('Opening {}'.format(self.name))
        yield from bps.mv(self.output, self.open_pos, wait=True)

    def close_plan(self, printing=True):
        if printing: print('Closing {}'.format(self.name))
        yield from bps.mv(self.output, self.closed_pos, wait=True)




shutter = ShutterMotor(prefix='',
                       name='User Shutter',
                       shutter_type='SP',
                       motor=usermotor3)



class TwoButtonShutterISS(TwoButtonShutter):
    RETRY_PERIOD = 1

    def stop(self, success=False):
        pass

    def set(self, val):
        if self._set_st is not None:
            raise RuntimeError(f'trying to set {self.name}'
                               ' while a set is in progress')

        cmd_map = {self.open_str: self.open_cmd,
                   self.close_str: self.close_cmd}
        target_map = {self.open_str: self.open_val,
                      self.close_str: self.close_val}

        cmd_sig = cmd_map[val]
        target_val = target_map[val]

        st = DeviceStatus(self)
        if self.status.get() == target_val:
            st._finished()
            return st

        self._set_st = st
        # print(self.name, val, id(st))
        # enums = self.status.enum_strs

        def shutter_cb(value, timestamp, **kwargs):
            # At some point ophyd/pyepics started to do this
            # remapping before passing to the callbacks so `int` call
            # here was failing. this means that we were never getting to
            # the next check which means we were never flipping that status
            # object to done.

            # value = enums[int(value)]
            if value == target_val:
                self._set_st = None
                self.status.clear_sub(shutter_cb)
                st._finished()

        # cmd_enums = cmd_sig.enum_strs
        count = 0
        _time_fmtstr = '%Y-%m-%d %H:%M:%S'

        def cmd_retry_cb(value, timestamp, **kwargs):
            nonlocal count
            # At some point ophyd/pyepics started to do this
            # remapping before passing to the callbacks so `int` call
            # here was failing
            # value = cmd_enums[int(value)]
            count += 1
            if count > self.MAX_ATTEMPTS:
                cmd_sig.clear_sub(cmd_retry_cb)
                self._set_st = None
                self.status.clear_sub(shutter_cb)
                st._finished(success=False)
            if value == 'None':
                if not st.done:
                    time.sleep(self.RETRY_PERIOD)
                    cmd_sig.set(1)

                    # ts = datetime.datetime.fromtimestamp(timestamp) \
                    #     .strftime(_time_fmtstr)
                    if count > 2:
                        print(f'** ({ttime.ctime()}) Had to reactuate {self.name} while {val[:4]}ing. Attempt {count}.')
                else:
                    cmd_sig.clear_sub(cmd_retry_cb)

        cmd_sig.subscribe(cmd_retry_cb, run=False)
        self.status.subscribe(shutter_cb)
        cmd_sig.set(1)
        return st


shutter_ph_2b = TwoButtonShutterISS('XF:08IDA-PPS{PSh}', name='shutter_ph_2b')
shutter_fe_2b = TwoButtonShutterISS('XF:08ID-PPS{Sh:FE}', name='shutter_fe_2b')

class EPS_Shutter(Device):
    state = Cpt(EpicsSignal, 'Pos-Sts')
    cls = Cpt(EpicsSignal, 'Cmd:Cls-Cmd')
    opn = Cpt(EpicsSignal, 'Cmd:Opn-Cmd')
    error = Cpt(EpicsSignal,'Err-Sts')
    permit = Cpt(EpicsSignal, 'Permit:Enbl-Sts')
    enabled = Cpt(EpicsSignal, 'Enbl-Sts')


    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.color = 'red'

    def open_plan(self):
        yield from bps.mv(self.opn, 1)

    def close_plan(self):
        yield from bps.mv(self.cls, 1)

    def open(self):
        print('Opening {}'.format(self.name))
        self.opn.put(1)

    def close(self):
        print('Closing {}'.format(self.name))
        self.cls.put(1)

shutter_fe = EPS_Shutter('XF:08ID-PPS{Sh:FE}', name = 'FE Shutter')
shutter_fe.shutter_type = 'FE'
shutter_ph = EPS_Shutter('XF:08IDA-PPS{PSh}', name = 'PH Shutter')
shutter_ph.shutter_type = 'PH'


class ICAmplifier(Device):
    #low_noise_gain = Cpt(EpicsSignal, 'LN}I0')

    def __init__(self, *args, gain_0, gain_1, gain_2, hspeed_bit, bw_10mhz_bit, bw_1mhz_bit, lnoise, hspeed, bwidth, par = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.gain_0 = EpicsSignal(self.prefix + gain_0, name=self.name + '_gain_0')
        self.gain_1 = EpicsSignal(self.prefix + gain_1, name=self.name + '_gain_1')
        self.gain_2 = EpicsSignal(self.prefix + gain_2, name=self.name + '_gain_2')
        self.hspeed_bit = EpicsSignal(self.prefix + hspeed_bit, name=self.name + '_hspeed_bit')
        self.bw_10mhz_bit = EpicsSignal(self.prefix + bw_10mhz_bit, name=self.name + '_bw_10mhz_bit')
        self.bw_1mhz_bit = EpicsSignal(self.prefix + bw_1mhz_bit, name=self.name + '_bw_1mhz_bit')
        self.low_noise_gain = EpicsSignal(self.prefix + lnoise, name=self.name + '_lnoise')
        self.high_speed_gain = EpicsSignal(self.prefix + hspeed, name=self.name + '_hspeed')
        self.band_width = EpicsSignal(self.prefix + bwidth, name=self.name + '_bwidth')
        self.par = par

    def set_gain(self, value: int, high_speed: bool):

        val = int(value) - 2
        if not ((high_speed and (1 <= val < 7)) or (not high_speed and (0 <= val < 6))):
            print('{} invalid value. Ignored...'.format(self.name))
            return 'Aborted'

        if high_speed:
            val -= 1
            self.low_noise_gain.put(0)
            self.high_speed_gain.put(val + 1)
            self.hspeed_bit.put(1)
        else:
            self.low_noise_gain.put(val + 1)
            self.high_speed_gain.put(0)
            self.hspeed_bit.put(0)

        self.gain_0.put((val >> 0) & 1)
        self.gain_1.put((val >> 1) & 1)
        self.gain_2.put((val >> 2) & 1)

    def set_gain_plan(self, value: int, high_speed: bool):

        val = int(value) - 2
        if not ((high_speed and (1 <= val < 7)) or (not high_speed and (0 <= val < 6))):
            print('{} invalid value. Ignored...'.format(self.name))
            return 'Aborted'

        if high_speed:
            val -= 1
            yield from bps.abs_set(self.low_noise_gain, 0)
            yield from bps.abs_set(self.high_speed_gain, val + 1)
            yield from bps.abs_set(self.hspeed_bit, 1)
        else:
            yield from bps.abs_set(self.low_noise_gain, val + 1)
            yield from bps.abs_set(self.high_speed_gain, 0)
            yield from bps.abs_set(self.hspeed_bit, 0)

        yield from bps.abs_set(self.gain_0, (val >> 0) & 1)
        yield from bps.abs_set(self.gain_1, (val >> 1) & 1)
        yield from bps.abs_set(self.gain_2, (val >> 2) & 1)

    def get_gain(self):
        if self.low_noise_gain.get() == 0:
            return [int(self.high_speed_gain.enum_strs[self.high_speed_gain.get()][-1]),1]
        elif self.high_speed_gain.get() == 0:
            return [int(self.low_noise_gain.enum_strs[self.low_noise_gain.get()][-1]),0]

    def get_gain_value(self):
        return self.get_gain()[0]

        '''
        if self.low_noise_gain.get() == 0:
            return [self.high_speed_gain.enum_strs[self.high_speed_gain.get()], 1]
        elif self.high_speed_gain.get() == 0:
            return [self.low_noise_gain.enum_strs[self.low_noise_gain.get()], 0]
        else:
            return ['0', 0]
        '''


i0_amp = ICAmplifier('XF:08IDB-CT{', gain_0='Amp-I0}GainBit:0-Sel', gain_1='Amp-I0}GainBit:1-Sel',
                     gain_2='Amp-I0}GainBit:2-Sel', hspeed_bit='Amp-I0}GainMode-Sel', bw_10mhz_bit='Amp-I0}10MHzMode-Sel',
                     bw_1mhz_bit='Amp-I0}1MHzMode-Sel',
                     lnoise='Amp-I0}LowNoise-Sel', hspeed='Amp-I0}HighSpeed-Sel', bwidth='Amp-I0}Bandwidth-Sel',
                     name='i0_amp')

it_amp = ICAmplifier('XF:08IDB-CT{', gain_0='Amp-It}GainBit:0-Sel', gain_1='Amp-It}GainBit:1-Sel',
                     gain_2='Amp-It}GainBit:2-Sel', hspeed_bit='Amp-It}GainMode-Sel', bw_10mhz_bit='Amp-It}10MHzMode-Sel',
                     bw_1mhz_bit='Amp-It}1MHzMode-Sel',
                     lnoise='Amp-It}LowNoise-Sel', hspeed='Amp-It}HighSpeed-Sel', bwidth='Amp-It}Bandwidth-Sel',
                     name='it_amp')

ir_amp = ICAmplifier('XF:08IDB-CT{', gain_0='Amp-Ir}GainBit:0-Sel', gain_1='Amp-Ir}GainBit:1-Sel',
                     gain_2='Amp-Ir}GainBit:2-Sel', hspeed_bit='Amp-Ir}GainMode-Sel', bw_10mhz_bit='Amp-Ir}10MHzMode-Sel',
                     bw_1mhz_bit='Amp-Ir}1MHzMode-Sel',
                     lnoise='Amp-Ir}LowNoise-Sel', hspeed='Amp-Ir}HighSpeed-Sel', bwidth='Amp-Ir}Bandwidth-Sel',
                     name='ir_amp')

iff_amp = ICAmplifier('XF:08IDB-CT{', gain_0='Amp-If}GainBit:0-Sel', gain_1='Amp-If}GainBit:1-Sel',
                     gain_2='Amp-If}GainBit:2-Sel', hspeed_bit='Amp-If}GainMode-Sel', bw_10mhz_bit='Amp-If}10MHzMode-Sel',
                     bw_1mhz_bit='Amp-If}1MHzMode-Sel',
                     lnoise='Amp-If}LowNoise-Sel', hspeed='Amp-If}HighSpeed-Sel', bwidth='Amp-If}Bandwidth-Sel',
                     name='iff_amp')




class ICAmplifier_Keithley(Device):

    gain = Cpt(EpicsSignal, 'Gain')
    risetime = Cpt(EpicsSignal, 'RiseTime')
    supr_mode = Cpt(EpicsSignal, 'SuppressionMode')

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.polarity = 'pos'

    def get_gain(self):
        return [self.gain.get() + 3, 0]

    def set_gain(self, gain, *args, **kwargs):
        self.gain.set(gain - 3)

    def set_gain_plan(self, gain, *args, **kwargs):
        yield from bps.abs_set(self.gain, gain - 3)

    def get_gain_value(self):
        return self.get_gain()[0]


k1_amp = ICAmplifier_Keithley('XF:08ID-ES:{K428}:A:',name='k1_amp')
k2_amp = ICAmplifier_Keithley('XF:08ID-ES:{K428}:B:',name='k2_amp')
k3_amp = ICAmplifier_Keithley('XF:08ID-ES:{K428}:C:',name='k3_amp')
k4_amp = ICAmplifier_Keithley('XF:08ID-ES:{K428}:D:',name='k4_amp')



#Updating the metadata dict to incorporate the gain values from the keithley's amplifier

i0_amp = k1_amp
it_amp = k2_amp
ir_amp = k3_amp
iff_amp = k4_amp

#Updating the metadata dict to incorporate the gain values from the keithley's amplifier

def current_suppression_plan(*args, **kwargs):

    #abp_c must be activated to perfrom current suppression on hutch C ionization chambers

    for i in range(1,5):
        print(f'Performing current suppression on Keithley{i}')
        yield from bps.mv(getattr(apb_ave, 'ch' + str(i)).amp_keithley.supr_mode, 2)
        yield from sleep(2)
        print(f'Current suppression on ch{i} is done')


#
# i0_amp = ICAmplifier('XF:08IDB-CT{', gain_0='ES-DO}2_8_0', gain_1='ES-DO}2_8_1',
#                      gain_2='ES-DO}2_8_2', hspeed_bit='ES-DO}2_8_3', bw_10mhz_bit='ES-DO}2_8_4', bw_1mhz_bit='ES-DO}2_8_5',
#                      lnoise='Amp-LN}I0', hspeed='Amp-HS}I0', bwidth='Amp-BW}I0', name='i0_amp')
#
# it_amp = ICAmplifier('XF:08IDB-CT{', gain_0='ES-DO}2_9_0', gain_1='ES-DO}2_9_1',
#                      gain_2='ES-DO}2_9_2', hspeed_bit='ES-DO}2_9_3', bw_10mhz_bit='ES-DO}2_9_4', bw_1mhz_bit='ES-DO}2_9_5',
#                      lnoise='Amp-LN}It', hspeed='Amp-HS}It', bwidth='Amp-BW}It', name='it_amp')
#
# ir_amp = ICAmplifier('XF:08IDB-CT{', gain_0='ES-DO}2_10_0', gain_1='ES-DO}2_10_1',
#                      gain_2='ES-DO}2_10_2', hspeed_bit='ES-DO}2_10_3', bw_10mhz_bit='ES-DO}2_10_4', bw_1mhz_bit='ES-DO}2_10_5',
#                      lnoise='Amp-LN}Ir', hspeed='Amp-HS}Ir', bwidth='Amp-BW}Ir', name='ir_amp')
#
# iff_amp = ICAmplifier('XF:08IDB-CT{', gain_0='ES-DO}2_11_0', gain_1='ES-DO}2_11_1',
#                      gain_2='ES-DO}2_11_2', hspeed_bit='ES-DO}2_11_3', bw_10mhz_bit='ES-DO}2_11_4', bw_1mhz_bit='ES-DO}2_11_5',
#                      lnoise='Amp-LN}If', hspeed='Amp-HS}If', bwidth='Amp-BW}If', name='iff_amp')




#old pizzabox
# pba1.adc7.amp = i0_amp
# pba1.adc1.amp = it_amp
# pba1.adc6.amp = iff_amp
# pba2.adc6.amp = i0_amp


class Lakeshore331Setpoint(Device):
    readback = Cpt(EpicsSignalRO, 'SAMPLE_A')
    setpoint = Cpt(EpicsSignal, 'SP')

lakeshore = Lakeshore331Setpoint('XF:08ID-ES{LS:331-1}:', name = 'lakeshore')

def set_lakeshore_temp(temperature: float= 5, *args, **kwargs):
    yield from bps.mv(lakeshore.setpoint, float(temperature))
    yield from bps.sleep(0.1)


