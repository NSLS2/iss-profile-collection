print(ttime.ctime() + ' >>>> ' + __file__)
import sys
import collections


from ophyd.sim import motor
motor.move = motor.set



detector_dictionary =   {

                    'I0 ion Chamber': {'device': apb_ave, 'channels': ['apb_ave_ch1_mean']},
                    'It ion Chamber': {'device': apb_ave, 'channels': ['apb_ave_ch2_mean']},
                    'Ir ion Chamber': {'device': apb_ave, 'channels': ['apb_ave_ch3_mean']},
                    'PIPS detector': {'device': apb_ave, 'channels': ['apb_ave_ch4_mean']},
                    'I0 ion Chamber instantaneous': {'device': apb, 'channels': ['apb_ch1']},
                    'It ion Chamber instantaneous': {'device': apb, 'channels': ['apb_ch2']},
                    'Ir ion Chamber instantaneous': {'device': apb, 'channels': ['apb_ch3']},
                    'PIPS detector instantaneous': {'device': apb, 'channels': ['apb_ch4']},
                    'Collimating mirror BPM': {'device': bpm_cm, 'channels': ['bpm_cm_stats1_total', 'bpm_cm_stats2_total']},
                    'Focusing mirror BPM': {'device': bpm_fm, 'channels': ['bpm_fm_stats1_total', 'bpm_fm_stats2_total']},
                    'Endstation BPM': {'device': bpm_es, 'channels': ['bpm_es_stats1_total','bpm_es_stats2_total']},
                    'Camera SP1': {'device': camera_sp1, 'channels': ['camera_sp1_stats1_total', 'camera_sp1_stats1_net', 'camera_sp1_stats2_total', 'camera_sp1_stats2_net']},
                    'Camera SP2': {'device': camera_sp2, 'channels': ['camera_sp2_stats1_total', 'camera_sp2_stats1_net', 'camera_sp2_stats2_total', 'camera_sp2_stats2_net']},
                    'Camera SP3': {'device': camera_sp3, 'channels': ['camera_sp3_stats1_total', 'camera_sp3_stats2_total']},
                    'Camera SP5': {'device': camera_sp5, 'channels': ['camera_sp5_stats1_total', 'camera_sp5_stats2_total']},
                    'Camera SP6': {'device': camera_sp6, 'channels': ['camera_sp5_stats1_total', 'camera_sp5_stats2_total']},
                    'Pilatus 100k': {'device': pil100k, 'flying_device' : pil100k_stream,
                                     'channels': ['pil100k_stats1_total','pil100k_stats2_total',
                                                                    'pil100k_stats3_total','pil100k_stats4_total',
                                                                     'pil100k_stats1_max_value']},
                    'Pilatus 100k New': {'device': pil100k2, 'flying_device': pil100k2_stream,
                                     'channels': ['pil100k2_stats1_total', 'pil100k2_stats2_total',
                                                  'pil100k2_stats3_total', 'pil100k2_stats4_total',
                                                  'pil100k2_stats1_max_value']},
                    'Ge detector': {'device': ge_detector, 'flying_device' : ge_detector_stream,
                                'channels': [f'ge_detector__channels_mca{ch:1d}_R{rnum:1d}'
                                               for ch, rnum in product(range(1, 33), range(4))]+
                                               ['ge_detector_settings_acquire_time']},
                                # 'channels': [f'xia_channel{ch:1d}_roi01_value'
                                #                for ch in range(1, 33)]+
                                #                ['xia_settings_acquire_time']},                              
                    # 'Ge detector':{'device':ge_detector, 'flying_device': ge_detector_stream, 'channels':['ge_detector_channels_mca1_R0',
                    #                                                                                        'ge_detector_channels_mca2_R0',
                    #                                                                                        'ge_detector_channels_mca3_R0',
                    #                                                                                        'ge_detector_channels_mca5_R0',
                    #                                                                                        'ge_detector_channels_mca7_R0',
                    #                                                                                        'ge_detector_channels_mca8_R0',
                    #                                                                                        'ge_detector_channels_mca9_R0',
                    #                                                                                        'ge_detector_channels_mca10_R0',
                    #                                                                                        'ge_detector_channels_mca11_R0',
                    #                                                                                        'ge_detector_channels_mca12_R0',
                    #                                                                                        'ge_detector_channels_mca13_R0',
                    #                                                                                        'ge_detector_channels_mca14_R0',
                    #                                                                                        'ge_detector_channels_mca15_R0',
                    #                                                                                        'ge_detector_channels_mca16_R0',
                    #                                                                                        'ge_detector_channels_mca17_R0',
                    #                                                                                        'ge_detector_channels_mca18_R0',
                    #                                                                                        'ge_detector_channels_mca19_R0',
                    #                                                                                        'ge_detector_channels_mca20_R0',
                    #                                                                                        'ge_detector_channels_mca21_R0',
                    #                                                                                        'ge_detector_channels_mca22_R0',
                    #                                                                                        'ge_detector_channels_mca23_R0',
                    #                                                                                        'ge_detector_channels_mca24_R0',
                    #                                                                                        'ge_detector_channels_mca25_R0',
                    #                                                                                        'ge_detector_channels_mca26_R0',
                    #                                                                                        'ge_detector_channels_mca27_R0',
                    #                                                                                        'ge_detector_channels_mca28_R0',
                    #                                                                                        'ge_detector_channels_mca29_R0',
                    #                                                                                        'ge_detector_channels_mca31_R0',] },
                    # 'Ge detector': {'device': ge_detector, 'flying_device': ge_detector_stream},
                    # 'PI-MTE3': {'device': picam,
                    #                  'channels': ['picam_stats1_total','picam_stats2_total',
                    #                                                 'picam_stats3_total','picam_stats4_total']},
                    # 'XIAXMAP': {'device': xia, 'flying_device' : xia_stream,
                    #             'channels': [f'xia_channel{ch:1d}_roi01_value'
                    #                            for ch in range(1, 33)]+
                    #                            ['xia_settings_acquire_time']},
                    'Xspress3': {'device' : xs, 'flying_device' : xs_stream,
                                 'channels' : [     'xs_channel1_rois_roi01_value',
                                                    'xs_channel1_rois_roi02_value',
                                                    'xs_channel1_rois_roi03_value',
                                                    'xs_channel1_rois_roi04_value',
                                                    'xs_channel2_rois_roi01_value',
                                                    'xs_channel2_rois_roi02_value',
                                                    'xs_channel2_rois_roi03_value',
                                                    'xs_channel2_rois_roi04_value',
                                                    'xs_channel3_rois_roi01_value',
                                                    'xs_channel3_rois_roi02_value',
                                                    'xs_channel3_rois_roi03_value',
                                                    'xs_channel3_rois_roi04_value',
                                                    'xs_channel4_rois_roi01_value',
                                                    'xs_channel4_rois_roi02_value',
                                                    'xs_channel4_rois_roi03_value',
                                                    'xs_channel4_rois_roi04_value',
                                                    'xs_settings_acquire_time'
                                                    ]},
                    'EM I0 I monitor' : {'device' : em_ave, 'channels': ['em_ave_ch1_mean']},
                    'EM I1 I monitor' : {'device' : em_ave, 'channels' : ['em_ave_ch2_mean']},
                    'EM I2 I monitor' : {'device' : em_ave, 'channels' : ['em_ave_ch3_mean']},
                    'EM Fluo I monitor' : {'device' : em_ave, 'channels' : ['em_ave_ch4_mean']},
                    'EM I0 V monitor' : {'device' : em_ave, 'channels' : ['em_ave_ch5_mean']},
                    'EM I1 V monitor' : {'device' : em_ave, 'channels' : ['em_ave_ch6_mean']},
                    'EM I2 V monitor' : {'device' : em_ave, 'channels' : ['em_ave_ch7_mean']},
                    'EM Fluo V monitor' : {'device' : em_ave, 'channels' : ['em_ave_ch8_mean']},
                }
# comment
def get_detector_device_list(key_list, flying=True):
    dets = []
    for key in key_list:
        if flying:
            device = detector_dictionary[key]['flying_device']
        else:
            device = detector_dictionary[key]['device']
        dets.append(device)
    return dets

def _get_object_component_based_on_full_name(obj, full_component_name):
    for _, signal in obj.get_instantiated_signals():
        if signal.name == full_component_name:
            return signal
# _get_object_component_based_on_full_name(bpm_fm, 'bpm_fm_stats1_total')

def get_detector_channel(detector_key, channel_key):
    det = get_detector_device_list([detector_key], flying=False)[0]
    channels = []
    # for key, det in zip(key_list, dets):
        # channel = getattr(det, detector_dictionary[key]['channels'][0])
    channel = _get_object_component_based_on_full_name(det, channel_key)
    return channel


class KeyDetectorNotIncluded(Exception):
    pass

def ensure_pilatus_is_in_detector_list(detectors):
    if (not ('Pilatus 100k' in detectors)) and (not (('Pilatus 100k New' in detectors))):
        raise KeyDetectorNotIncluded(f'Error: Pilatus 100k not found in the detector list')



motor_dictionary = {
    'cm1_pitch':                {'name' : cm1.pitch.name,           'description': 'A CM1 Pitch',                           'keyword': 'A CM1 Pitch',                         'object': cm1.pitch},
    'cm1_y':                    {'name' : cm1.y.name,               'description': 'A CM1 Y',                               'keyword': 'A CM1 Y',                             'object': cm1.y},
    'cm1_x':                    {'name' : cm1.x.name,               'description': 'A CM1 X',                               'keyword': 'A CM1 X',                             'object': cm1.x},
    'cm2_pitch':                {'name' : cm2.pitch.name,           'description': 'A CM2 Pitch',                           'keyword': 'A CM2 Pitch',                         'object': cm2.pitch},
    'cm2_y':                    {'name' : cm2.y.name,               'description': 'A CM2 Y',                               'keyword': 'A CM2 Y',                             'object': cm2.y},
    'bender':                   {'name' : bender.name,              'description': 'A CM2 bender',                          'keyword': 'A CM2 bender',                        'object': bender.pos,           'group': 'spectrometer'},
    'hhm_theta':                {'name': hhm.theta.name,            'description': 'A Monochromator Theta',                 'keyword': 'A Mono Theta',               'object': hhm.theta},
    'hhm_energy':               {'name': hhm.energy.name,           'description': 'A Monochromator Energy',                'keyword': 'A Mono Energy',              'object': hhm.energy,           'group': 'spectrometer','user':True},
    'hhm_y':                    {'name': hhm.y.name,                'description': 'A Monochromator Y',                     'keyword': 'A Mono Y',                   'object': hhm.y,                'user': True},
    'hhm_y_precise':            {'name': hhm.y_precise.name,        'description': 'A Monochromator Y precise',             'keyword': 'A Mono Y precise',           'object': hhm.y_precise,        'user': True},
    'hhm_pitch':                {'name': hhm.pitch.name,            'description': 'A Monochromator Pitch',                 'keyword': 'A Mono Pitch',               'object': hhm.pitch,            'user': True},
    'hhm_roll':                 {'name': hhm.roll.name,             'description': 'A Monochromator Roll',                  'keyword': 'A Mono Roll',                'object': hhm.roll, },
    'fm_pitch':                 {'name': fm.pitch.name,             'description': 'A Focusing Mirror Pitch',               'keyword': 'A Focusing Mirror Pitch',             'object': fm.pitch},
    'fm_y':                     {'name': fm.y.name,                 'description': 'A Focusing Mirror Y',                   'keyword': 'A Focusing Mirror Y',                 'object': fm.y},
    'bender_fm':                {'name': bender_fm.name,            'description': 'A Focusing Mirror bender',              'keyword': 'A Focusing Mirror bender',            'object': bender_fm.pos},
    'hhrm_mir_pitch':           {'name': hhrm.mir_pitch.name,       'description': 'B1 HHR Mirror Pitch',                   'keyword': 'B1 HHR Mirror Pitch',                 'object': hhrm.mir_pitch},
    'hhrm_table_pitch':         {'name': hhrm.table_pitch.name,     'description': 'B1 HHR Mirror Table Pitch',             'keyword': 'B1 HHR Mirror Table Pitch',           'object': hhrm.table_pitch},
    'hhrm_y':                   {'name': hhrm.y.name,               'description': 'B1 HHR Mirror Table Height',            'keyword': 'B1 HHR Mirror Table Height',          'object': hhrm.y,               'user': True},
    'hhrm_yu':                  {'name': hhrm.yu.name,              'description': 'B1 HHR Mirror Y Upstream',              'keyword': 'B1 HHR Mirror Y Upstream',            'object': hhrm.yu},
    'hhrm_yd':                  {'name': hhrm.yd1.name,             'description': 'B1 HHR Mirror Y Downstream',            'keyword': 'B1 HHR Mirror Y Downstream',          'object': hhrm.yd1},
    'slits_v_gap':              {'name': slits.v_gap.name,          'description': 'B1 Slit Vertical Gap',                  'keyword': 'B1 Slit Vertical Gap',                'object': slits.v_gap},
    'slits_v_pos':              {'name': slits.v_pos.name,          'description': 'B1 Slit Vertical Position',             'keyword': 'B1 Slit Vertical Position',           'object': slits.v_pos},
    'slits_hor_in':             {'name': slits.hor_in.name,         'description': 'B1 Slit Horisontal Inboard Position',   'keyword': 'B1 Slit Horisontal Inboard Position', 'object': slits.hor_in},
    'slits_hor_out':            {'name': slits.hor_out.name,        'description': 'B1 Slit Horisontal Outboard Position',  'keyword': 'B1 Slit Horisontal Outboard Position','object': slits.hor_out},
    'sample_shutter_angle':     {'name': usermotor3.name,           'description': 'B1 User shutter angle ',                'keyword': 'B1 User shutter angle ',              'object': usermotor3.pos},
    # 'usermotor3' :              {'name' : usermotor3.name,          'description' : 'User shutter angle',                  keywordon' : 'User shutter angle',                 'object' : usermotor3.pos},
    'i0_y_pos':                 {'name': i0_y.pos.name,             'description': 'B1 I0 Chamber height',                  'keyword': 'B1 I0 Chamber height',                'object': i0_y.pos},
    'it_y_pos':                 {'name': it_y.pos.name,             'description': 'B1 It Chamber height',                  'keyword': 'B1 It Chamber height',                'object': it_y.pos},
    'ir_y_pos':                 {'name': ir_y.pos.name,             'description': 'B1 Ir Chamber height',                  'keyword': 'B1 Ir Chamber height',                'object': ir_y.pos},
    'sample_stage_x':           {'name': sample_stage.x.name,       'description': 'Sample stage X',                        'keyword': 'Sample X',                      'object': sample_stage.x,       'group': 'spectrometer',    'user':True},
    'sample_stage_y':           {'name': sample_stage.y.name,       'description': 'Sample stage Y',                        'keyword': 'Sample Y',                      'object': sample_stage.y,       'group': 'spectrometer',    'user':True},
    'sample_stage_z':           {'name': sample_stage.z.name,       'description': 'Sample stage Z',                        'keyword': 'Sample Z',                      'object': sample_stage.z,       'group': 'spectrometer',    'user':True},
    'sample_stage_th':          {'name': sample_stage.th.name,      'description': 'Sample stage Th',                       'keyword': 'Sample Th',                     'object': sample_stage.th,      'group': 'spectrometer',    'user':True},
    'foil_wheel_wheel1' :       {'name': foil_wheel.wheel1.name,    'description': 'B1 Reference foil wheel 1',             'keyword': 'B1 Reference foil wheel 1',           'object' : foil_wheel.wheel1},
    'foil_wheel_wheel2' :       {'name': foil_wheel.wheel2.name,    'description': 'B1 Reference foil wheel 2',             'keyword': 'B1 Reference foil wheel 2',           'object' : foil_wheel.wheel2},

    'attenuator_wheel' :        {'name': attenuator_motor.name,      'description': 'B1 Attenuator wheel',                     'keyword': 'B1 Attenuator wheel',           'object' : attenuator_motor, 'user':False},


    'det1_stage_x':             {'name': det1_stage_x.pos.name,     'description': 'B1 Detector stage 1 X',                 'keyword': 'B1 Detector stage 1 X',               'object': det1_stage_x.pos,     'user':True},
    'det1_stage_y':             {'name': det1_stage_y.pos.name,     'description': 'B1 Detector stage 1 Y',                 'keyword': 'B1 Detector stage 1 Y',               'object': det1_stage_y.pos,     'user':True},
    'det2_stage_x':             {'name': det2_stage_x.pos.name,     'description': 'B1 Detector stage 2 X',                 'keyword': 'B1 Detector stage 2 X',               'object': det2_stage_x.pos,     'user':True},
    'det2_stage_y':             {'name': det2_stage_y.pos.name,     'description': 'B1 Detector stage 2 Y',                 'keyword': 'B1 Detector stage 2 Y',               'object': det2_stage_y.pos,     'user':True},
    'huber_stage_y':            {'name': huber_stage.y.name,        'description':'Pilatus motion X',                       'keyword':'Pilatus motion X',                     'object': huber_stage.y,        'group': 'spectrometer'},
    # 'huber_stage_pitch':        {'name': huber_stage.pitch.name,    'description':'B2 Huber Stage Pitch',                  keywordon':'B2 Huber Stage Pitch',                 'object': huber_stage.pitch},
    'huber_stage_z':            {'name': huber_stage.z.name,        'description':'Pilatus motion Y',                       'keyword':'Pilatus motion Y',                     'object': huber_stage.z,        'group': 'spectrometer'},
    # 'detstage_x':               {'name': detstage.x.name,           'description': 'PIPS Stage X',                         keywordon': 'PIPS Stage X',                        'object': detstage.x},
    # 'detstage_y':               {'name': detstage.y.name,           'description': 'PIPS Stage Y',                         keywordon': 'PIPS Stage Y',                        'object': detstage.y},
    # 'detstage_z':               {'name': detstage.z.name,           'description': 'PIPS Stage Z',                         keywordon': 'PIPS Stage Z',                        'object': detstage.z},
    'six_axes_stage_x':         {'name': six_axes_stage.x.name,     'description': 'PCL Six Axes Stage X',                  'keyword': 'PCL Six Axes Stage X',                'object': six_axes_stage.x,     'group': 'spectrometer'},
    'six_axes_stage_y':         {'name': six_axes_stage.y.name,     'description': 'PCL Six Axes Stage Y',                  'keyword': 'PCL Six Axes Stage Y',                'object': six_axes_stage.y,     'group': 'spectrometer'},
    'six_axes_stage_z':         {'name': six_axes_stage.z.name,     'description': 'PCL Six Axes Stage Z',                  'keyword': 'PCL Six Axes Stage Z',                'object': six_axes_stage.z,     'group': 'spectrometer'},
    'six_axes_stage_pitch':     {'name': six_axes_stage.pitch.name, 'description': 'PCL Six Axes Stage Pitch',              'keyword': 'PCL Six Axes Stage Pitch',            'object': six_axes_stage.pitch, 'group': 'spectrometer'},
    'six_axes_stage_yaw':       {'name': six_axes_stage.yaw.name,   'description': 'PCL Six Axes Stage Yaw',                'keyword': 'PCL Six Axes Stage Yaw',              'object': six_axes_stage.yaw,   'group': 'spectrometer'},
    'six_axes_stage_roll':      {'name': six_axes_stage.roll.name,  'description': 'PCL Six Axes Stage Roll',               'keyword': 'PCL Six Axes Stage Roll',             'object': six_axes_stage.roll,  'group': 'spectrometer'},
    # 'motor_det_x':              {'name': von_hamos_det_arm.motor_det_x.name,  'description': 'Detector x',                ekeyword'description': 'Detector x',                'object': von_hamos_det_arm.motor_det_x,  'group': 'spectrometer'},
    # 'motor_det_th1':            {'name': von_hamos_det_arm.motor_det_th1.name,  'description': 'Detector th1',            et_th1.name,  'description': 'Detector th1',            'object': von_hamos_det_arm.motor_det_th1,  'group': 'spectrometer'},
    # 'motor_det_th2':            {'name': von_hamos_det_arm.motor_det_th2.name,  'description': 'Detector th2',            et_th2.name,  'description': 'Detector th2',            'object': von_hamos_det_arm.motor_det_th2,  'group': 'spectrometer'},
    'fip_spectrometer_crystal_x': {'name': fip_spectrometer_crystal.x.name,  'description': 'B1 FIP crystal X',              'keyword': 'B1 FIP crystal X',             'object': fip_spectrometer_crystal.x},
    'fip_spectrometer_crystal_y': {'name': fip_spectrometer_crystal.y.name,  'description': 'B1 FIP crystal Y',              'keyword': 'B1 FIP crystal Y',             'object': fip_spectrometer_crystal.y},
    'fip_spectrometer_detector_x':{'name': fip_spectrometer_detector.x.name, 'description': 'B1 FIP detector X',             'keyword': 'B1 FIP detector X',            'object': fip_spectrometer_detector.x},
    'fip_spectrometer_detector_x':{'name': fip_spectrometer_detector.y.name, 'description': 'B1 FIP detector Y',             'keyword': 'B1 FIP detector Y',            'object': fip_spectrometer_detector.y},
    'sample_stage_B2_x':            {'name': samplexy.x.name,       'description': 'B2 Sample stage X',                        'keyword': 'B2 Sample X',                      'object': samplexy.x,    'user':True},
    'sample_stage_B2_y':            {'name': samplexy.y.name,       'description': 'B2 Sample stage Y',                        'keyword': 'B2 Sample Y',                      'object': samplexy.y,    'user':True},
    'hutch_b_slits_top':            {'name': hutch_b_slits.top.name, 'description': 'Hutch B1 Slits top',                        'keyword': 'B1 slits top',                      'object': hutch_b_slits.top,    'user':False },
    'hutch_b_slits_bottom':         {'name': hutch_b_slits.bottom.name, 'description': 'Hutch B1 Slits bottom',                        'keyword': 'B1 slits bottom',                      'object': hutch_b_slits.bottom,    'user':False },
    'hutch_b_slits_inboard':        {'name': hutch_b_slits.inboard.name, 'description': 'Hutch B1 Slits inboard',                        'keyword': 'B1 slits inboard',                      'object': hutch_b_slits.inboard,    'user':False },
    'hutch_b_slits_outboard':        {'name': hutch_b_slits.outboard.name, 'description': 'Hutch B1 Slits outboard',                        'keyword': 'B1 slits outboard',                      'object': hutch_b_slits.outboard,    'user':False },
}

def get_motor_device(motor_attr, based_on='description'):
    for key, motor_dict in motor_dictionary.items():
        if based_on == 'description':
            if motor_attr == motor_dict['description']:
                return motor_dict['object']
        elif based_on == 'object_name':
            if motor_attr == motor_dict['name']:
                return motor_dict['object']


shutter_dictionary = collections.OrderedDict([(shutter_fe.name, shutter_fe),
                                         (shutter_ph.name, shutter_ph),
                                         (shutter.name, shutter)])

# ic_amplifiers = {'i0_amp': i0_amp,
#                  'it_amp': it_amp,
#                  'ir_amp': ir_amp,
#                  'iff_amp': iff_amp}

ic_amplifiers = {'i0_amp': k1_amp,
                 'it_amp': k2_amp,
                 'ir_amp': k3_amp,
                 'iff_amp': k4_amp}


camera_dictionary = {'camera_sample1': camera_sp1,
                     'camera_sample2': camera_sp2,
                     'camera_sample4': camera_sp4}
