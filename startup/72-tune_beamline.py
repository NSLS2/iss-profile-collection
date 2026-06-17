print(ttime.ctime() + ' >>>> ' + __file__)

tune_elements =  [{'motor': hhm.pitch.name,
                   'detector': 'Focusing mirror BPM',
                   'range': 10,
                   'step': 0.1,
                   'retries': 10,
                   'comment': 'rough monochromator pitch tune'},
                  {'motor': hhm.pitch.name,
                   'detector': 'Focusing mirror BPM',
                   'range': 1,
                   'step': 0.02,
                   'retries': 3,
                   'comment': 'fine monochromator pitch tune'},
                  # {'motor': hhm.y.name,
                  #  # 'detector':'Focusing mirror BPM',
                  #  'detector':'Endstation BPM',
                  #  'range': 1,
                  #  'step': 0.025,
                  #  'retries': 3,
                  #  'comment': 'monochromator crystal distance tune'},
                  {'motor': hhrm.y.name, #'motor': [hhrm.y.name, i0_y.pos.name],
                   'detector': 'I0 ion Chamber',
                   'range': 1,
                   'step': 0.025,
                   'retries': 3,
                   'comment': 'Harmonic rejection mirror tune'},
                  {'motor': hhm.pitch.name,
                   'detector': 'I0 ion Chamber',
                   'range': 0.3,
                   'step': 0.02,
                   'retries': 3,
                   'comment': 'fine monochromator pitch tune'},
                ]

tune_elements_alt = [
                  {'motor': hhm.pitch.name,
                   'detector': 'Endstation BPM',
                   'range': 1,
                   'step': 0.02,
                   'retries': 3,
                   'comment': 'fine monochromator pitch tune'},
                  {'motor': hhm.y.name,
                   'detector':'Endstation BPM',
                   'range': 1,
                   'step': 0.025,
                   'retries': 3,
                   'comment': 'monochromator crystal distance tune'},
                  ]



tune_elements_ext =  [{'motor': hhm.pitch.name,
                   'detector': 'Focusing mirror BPM',
                   'range': 10,
                   'step': 0.1,
                   'retries': 10,
                   'comment': 'rough monochromator pitch tune'},
                  {'motor': hhm.pitch.name,
                   'detector': 'Focusing mirror BPM',
                   'range': 1,
                   'step': 0.02,
                   'retries': 3,
                   'comment': 'fine monochromator pitch tune'},
                  {'motor': hhm.y_precise.name,
                   'detector':'Focusing mirror BPM',
                   'range': 1,
                   'step': 0.025,
                   'retries': 3,
                   'comment': 'monochromator crystal distance tune'},
                  {'motor': hhrm.y.name, #'motor': [hhrm.y.name, i0_y.pos.name],
                   'detector': 'I0 ion Chamber',
                   'range': 1,
                   'step': 0.025,
                   'retries': 3,
                   'comment': 'Harmonic rejection mirror tune'},
                  {'motor': hhm.pitch.name,
                   'detector': 'I0 ion Chamber',
                   'range': 0.3,
                   'step': 0.02,
                   'retries': 3,
                   'comment': 'fine monochromator pitch tune'},
                  {'motor': hhm.y_precise.name,
                   'detector':'I0 ion Chamber',
                   'range': 0.6,
                   'step': 0.025,
                   'retries': 3,
                   'comment': 'fine monochromator crystal distance tune'},
                  {'motor': hhm.pitch.name,
                   'detector': 'I0 ion Chamber',
                   'range': 0.3,
                   'step': 0.02,
                   'retries': 3,
                   'comment': 'fine monochromator pitch tune'},
                  {'motor': hhrm.y.name, #'motor': [hhrm.y.name, i0_y.pos.name],
                   'detector': 'I0 ion Chamber',
                   'range': 1,
                   'step': 0.025,
                   'retries': 3,
                   'comment': 'Fine harmonic rejection mirror tune'},
                  {'motor': hhm.pitch.name,
                   'detector': 'I0 ion Chamber',
                   'range': 0.3,
                   'step': 0.02,
                   'retries': 3,
                   'comment': 'fine monochromator pitch tune'},
                  {'motor': hhm.y_precise.name,
                   'detector':'I0 ion Chamber',
                   'range': 0.6,
                   'step': 0.025,
                   'retries': 3,
                   'comment': 'fine monochromator crystal distance tune',
                   'fb_enable': True},
                  {'motor': slits.v_pos.name,
                   'detector':'I0 ion Chamber',
                   'range': 3,
                   'step': 0.1,
                   'retries': 3,
                   'comment': 'fine B1 slits vertical position tune',
                   'fb_enable': True},
                ]


def tune_beamline_plan_bundle(extended_tuning : bool = False, enable_fb_in_the_end : bool = True, do_liveplot=False):

    if extended_tuning:
        # tune_elements_list = tune_elements_ext
        tune_elements_list = tune_elements_alt
    else:
        tune_elements_list = tune_elements

    plans = [{'plan_name': 'print_message_plan',
              'plan_kwargs': {'msg': 'Starting...', 'tag': 'Beamline tuning'}}]

    plans.append({'plan_name' : 'set_hhm_feedback_plan',
                   'plan_kwargs' : {'state' : 0}})

    if detector_dictionary[tune_elements_list[0]['detector']]['device'] != bpm_fm:
        if bpm_fm.inserted.get():
            plans.append({'plan_name': 'move_bpm_fm_plan',
                          'plan_kwargs': {'action': 'retract'}})

    for i, element in enumerate(tune_elements_list):
        detector_device = detector_dictionary[element['detector']]['device']
        if detector_device == bpm_fm:
            if bpm_fm.retracted.get():
                plans.append({'plan_name': 'move_bpm_fm_plan',
                              'plan_kwargs': {'action': 'insert'}})

        if 'fb_enable' in element.keys():
            if element['fb_enable']:
                plans.append({'plan_name' : 'set_hhm_feedback_plan',
                              'plan_kwargs' : {'state' : 1, 'update_center' : True}})
        if do_liveplot:
            channel = detector_dictionary[element['detector']]['device'].hints['fields'][0]
            liveplot_kwargs = {'channel' : channel,
                               'channel_den' : '1',
                               'result_name' : channel,
                               'curr_mot_name' : element['motor']}
        else:
            liveplot_kwargs = {}

        plans.append({'plan_name' : 'tuning_scan',
                     'plan_kwargs' : {'motor' : element['motor'],
                                      'detector' : element['detector'],
                                      'scan_range' : element['range'],
                                      'scan_step' : element['step'],
                                      'n_tries' : element['retries'],
                                      'liveplot_kwargs' : liveplot_kwargs}})

        if detector_device == bpm_fm:
            if ((i + 1) < len(tune_elements_list)) and (detector_dictionary[tune_elements_list[i + 1]['detector']]['device'] != bpm_fm):
                plans.append({'plan_name': 'move_bpm_fm_plan',
                              'plan_kwargs': {'action': 'retract'}})
                plans.append({'plan_name': 'put_bpm_fm_to_continuous_mode',
                              'plan_kwargs': {}})

    if enable_fb_in_the_end:
        plans.append({'plan_name': 'set_hhm_feedback_plan',
                      'plan_kwargs': {'state': 1, 'update_center' : True}})

    plans.append({'plan_name': 'print_message_plan',
                  'plan_kwargs': {'msg': 'Beamline tuning complete', 'tag' : 'Beamline tuning'}})

    return plans


quick_tune_elements =  [
    {  'motor': hhm.pitch.name,
       'detector': 'Focusing mirror BPM',
       'channel': 'bpm_fm_stats1_total',
       'range': 5,
       'velocity': 0.2,
       'n_tries': 10,
       'comment': 'rough monochromator pitch tune'},
      {'motor': hhm.pitch.name,
       'detector': 'Focusing mirror BPM',
       'channel': 'bpm_fm_stats1_total',
       'range': 1,
       'velocity': 0.1,
       'n_tries': 3,
       'comment': 'fine monochromator pitch tune'},
      {'motor': hhrm.y.name, #'motor': [hhrm.y.name, i0_y.pos.name],
       'detector': 'I0 ion Chamber instantaneous',
       'channel': 'apb_ch1',
       'range': 1,
       'velocity': 0.07,
       'n_tries': 3,
       'comment': 'Harmonic rejection mirror tune'},
      {'motor': hhm.pitch.name,
       'detector': 'I0 ion Chamber instantaneous',
       'channel': 'apb_ch1',
       'range': 0.3,
       'velocity': 0.03,
       'n_tries': 3,
       'comment': 'fine monochromator pitch tune'},
    ]


def quick_tune_beamline_plan_bundle(enable_fb_in_the_end : bool = True, plan_gui_services=None):


    tune_elements_list = quick_tune_elements

    plans = [{'plan_name': 'print_message_plan',
              'plan_kwargs': {'msg': 'Starting...', 'tag': 'Beamline tuning'}}]

    plans.append({'plan_name' : 'set_hhm_feedback_plan',
                   'plan_kwargs' : {'state' : 0}})

    if detector_dictionary[tune_elements_list[0]['detector']]['device'] != bpm_fm:
        if bpm_fm.inserted.get():
            plans.append({'plan_name': 'move_bpm_fm_plan',
                          'plan_kwargs': {'action': 'retract'}})

    for i, element in enumerate(tune_elements_list):
        detector_device = detector_dictionary[tune_elements_list[i]['detector']]['device']
        if detector_device == bpm_fm:
            if bpm_fm.retracted.get():
                plans.append({'plan_name': 'move_bpm_fm_plan',
                              'plan_kwargs': {'action': 'insert'}})

        if 'fb_enable' in element.keys():
            if element['fb_enable']:
                plans.append({'plan_name' : 'set_hhm_feedback_plan',
                              'plan_kwargs' : {'state' : 1, 'update_center' : True}})

        plans.append({'plan_name' : 'quick_tuning_scan',
                     'plan_kwargs' : {'motor' : element['motor'],
                                      'detector' : element['detector'],
                                      'channel': element['channel'],
                                      'scan_range' : element['range'],
                                      'velocity' : element['velocity'],
                                      'n_tries' : element['n_tries'],
                                      'liveplot_kwargs': {}},
                      'plan_gui_services' : plan_gui_services})

        if detector_device == bpm_fm:
            if ((i + 1) < len(tune_elements_list)) and (detector_dictionary[tune_elements_list[i+1]['detector']]['device'] != bpm_fm):
                plans.append({'plan_name': 'move_bpm_fm_plan',
                              'plan_kwargs': {'action': 'retract'}})
                plans.append({'plan_name': 'put_bpm_fm_to_continuous_mode',
                              'plan_kwargs': {}})

    if enable_fb_in_the_end:
        plans.append({'plan_name': 'set_hhm_feedback_plan',
                      'plan_kwargs': {'state': 1, 'update_center' : True}})

    plans.append({'plan_name': 'print_message_plan',
                  'plan_kwargs': {'msg': 'Beamline tuning complete', 'tag' : 'Beamline tuning'}})

    return plans



#
# def optimize_beamline_plan(energy: int = -1, extended_tuning: bool = False, force_prepare = False, enable_fb_in_the_end=True):
#     old_energy = hhm.energy.position
#     if force_prepare or ((np.abs((energy-old_energy)/old_energy)> 0.1) or (np.sign(old_energy-13000)) != (np.sign(energy-13000))):
#         yield from shutter.close_plan()
#         yield from prepare_beamline_plan(energy, move_cm_mirror = True)
#         yield from tune_beamline_plan(extended_tuning=extended_tuning, enable_fb_in_the_end=enable_fb_in_the_end)
#     else:
#         # print_to_gui(f'Beamline is already prepared for {energy} eV', stdout=stdout)
#         yield from bps.mv(hhm.energy, energy)


def optimize_beamline_plan_bundle(energy: int = -1, extended_tuning: bool = False, force_prepare = False, enable_fb_in_the_end=True, do_liveplot=False):
    old_energy = hhm.energy.position
    plans = []
    if force_prepare or ((np.abs((energy-old_energy)/old_energy)> 0.1) or (np.sign(old_energy-13000)) != (np.sign(energy-13000))):
        plans.append({'plan_name' : 'shutter_close_plan', 'plan_kwargs' : {}})
        plans.append({'plan_name' : 'prepare_beamline_plan',
                      'plan_kwargs' : {'energy': energy, 'move_cm_mirror': True}})
        tuning_plans = tune_beamline_plan_bundle(extended_tuning=extended_tuning, enable_fb_in_the_end=enable_fb_in_the_end, do_liveplot=do_liveplot)
        plans.extend(tuning_plans)

    else:
        # print_to_gui(f'Beamline is already prepared for {energy} eV', stdout=stdout)
        plans.append({'plan_name': 'move_mono_energy',
                      'plan_kwargs': {'energy': energy}})
        plans.append({'plan_name': 'print_message_plan',
                      'plan_kwargs': {'msg': 'Beamline is already prepared for {energy} eV', 'tag': 'Beamline tuning'}})
    return plans




