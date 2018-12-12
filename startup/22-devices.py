from ophyd import (ProsilicaDetector, SingleTrigger, Component as Cpt, Device,
                   EpicsSignal, EpicsSignalRO, ImagePlugin, StatsPlugin, ROIPlugin,
                   DeviceStatus)
import bluesky.plans as bp


class MFC(Device):
    flow = Cpt(EpicsSignal, '-RB', write_pv='-SP')

gas_he = MFC('XF:08IDB-OP{IC}FLW:He', name='gas_he')
gas_n2 = MFC('XF:08IDB-OP{IC}FLW:N2', name='gas_n2')

gas_he.flow.tolerance = 0.01
gas_n2.flow.tolerance = 0.01

class WPS(Device):
    hv300 = Cpt(EpicsSignal, '-HV:u300}V-Sense', write_pv='-HV:u300}V-Set')
    hv301 = Cpt(EpicsSignal, '-HV:u301}V-Sense', write_pv='-HV:u301}V-Set')
    hv302 = Cpt(EpicsSignal, '-HV:u302}V-Sense', write_pv='-HV:u302}V-Set')
    hv303 = Cpt(EpicsSignal, '-HV:u303}V-Sense', write_pv='-HV:u303}V-Set')
    hv304 = Cpt(EpicsSignal, '-HV:u304}V-Sense', write_pv='-HV:u304}V-Set')
    hv305 = Cpt(EpicsSignal, '-HV:u305}V-Sense', write_pv='-HV:u305}V-Set')
    hv306 = Cpt(EpicsSignal, '-HV:u306}V-Sense', write_pv='-HV:u306}V-Set')
    hv307 = Cpt(EpicsSignal, '-HV:u307}V-Sense', write_pv='-HV:u307}V-Set')

    lv0 = Cpt(EpicsSignal, '-LV:u0}V-Sense', write_pv='-LV:u0}V-Set')
    lv1 = Cpt(EpicsSignal, '-LV:u1}V-Sense', write_pv='-LV:u1}V-Set')
    lv2 = Cpt(EpicsSignal, '-LV:u2}V-Sense', write_pv='-LV:u2}V-Set')
    lv3 = Cpt(EpicsSignal, '-LV:u3}V-Sense', write_pv='-LV:u3}V-Set')
    lv4 = Cpt(EpicsSignal, '-LV:u4}V-Sense', write_pv='-LV:u4}V-Set')
    lv5 = Cpt(EpicsSignal, '-LV:u5}V-Sense', write_pv='-LV:u5}V-Set')
    lv6 = Cpt(EpicsSignal, '-LV:u6}V-Sense', write_pv='-LV:u6}V-Set')
    lv7 = Cpt(EpicsSignal, '-LV:u7}V-Sense', write_pv='-LV:u7}V-Set')

wps1 = WPS('XF:08IDB-OP{WPS:01', name='wps1')


class Shutter(Device):

    def __init__(self, name):
        self.name = name
        if pb4.connected:
            self.output = pb4.do3.default_pol
            if self.output.value == 1:
                self.state = 'closed'
            elif self.output.value == 0:
                self.state = 'open'
            self.function_call = None
            self.output.subscribe(self.update_state)
        else:
            self.state = 'unknown'

    def subscribe(self, function):
        self.function_call = function

    def unsubscribe(self):
        self.function_call = None
        
    def update_state(self, pvname=None, value=None, char_value=None, **kwargs):
        if value == 1:
            self.state = 'closed'
        elif value == 0:
            self.state = 'open'
        if self.function_call is not None:
            self.function_call(pvname=pvname, value=value, char_value=char_value, **kwargs)
        
    def open(self):
        print('Opening {}'.format(self.name))
        self.output.put(0)
        self.state = 'open'
        
    def close(self):
        print('Closing {}'.format(self.name))
        self.output.put(1)
        self.state = 'closed'

    def open_plan(self):
        print('Opening {}'.format(self.name))
        yield from bps.abs_set(self.output, 0, wait=True)
        self.state = 'open'

    def close_plan(self):
        print('Closing {}'.format(self.name))
        yield from bps.abs_set(self.output, 1, wait=True)
        self.state = 'closed'

shutter = Shutter(name = 'SP Shutter')
shutter.shutter_type = 'SP'


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
        if self.low_noise_gain.value == 0:
            return [self.high_speed_gain.enum_strs[self.high_speed_gain.value], 1]
        elif self.high_speed_gain.value == 0:
            return [self.low_noise_gain.enum_strs[self.low_noise_gain.value], 0]
        else:
            return ['0', 0]


i0_amp = ICAmplifier('XF:08IDB-CT{', gain_0='ES-DO}2_8_0', gain_1='ES-DO}2_8_1',
                     gain_2='ES-DO}2_8_2', hspeed_bit='ES-DO}2_8_3', bw_10mhz_bit='ES-DO}2_8_4', bw_1mhz_bit='ES-DO}2_8_5',
                     lnoise='Amp-LN}I0', hspeed='Amp-HS}I0', bwidth='Amp-BW}I0', name='i0_amp')

it_amp = ICAmplifier('XF:08IDB-CT{', gain_0='ES-DO}2_9_0', gain_1='ES-DO}2_9_1',
                     gain_2='ES-DO}2_9_2', hspeed_bit='ES-DO}2_9_3', bw_10mhz_bit='ES-DO}2_9_4', bw_1mhz_bit='ES-DO}2_9_5',
                     lnoise='Amp-LN}It', hspeed='Amp-HS}It', bwidth='Amp-BW}It', name='it_amp')

ir_amp = ICAmplifier('XF:08IDB-CT{', gain_0='ES-DO}2_10_0', gain_1='ES-DO}2_10_1',
                     gain_2='ES-DO}2_10_2', hspeed_bit='ES-DO}2_10_3', bw_10mhz_bit='ES-DO}2_10_4', bw_1mhz_bit='ES-DO}2_10_5',
                     lnoise='Amp-LN}Ir', hspeed='Amp-HS}Ir', bwidth='Amp-BW}Ir', name='ir_amp')

iff_amp = ICAmplifier('XF:08IDB-CT{', gain_0='ES-DO}2_11_0', gain_1='ES-DO}2_11_1',
                     gain_2='ES-DO}2_11_2', hspeed_bit='ES-DO}2_11_3', bw_10mhz_bit='ES-DO}2_11_4', bw_1mhz_bit='ES-DO}2_11_5',
                     lnoise='Amp-LN}If', hspeed='Amp-HS}If', bwidth='Amp-BW}If', name='iff_amp')
