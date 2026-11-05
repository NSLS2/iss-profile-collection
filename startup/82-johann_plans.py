print(ttime.ctime() + ' >>>> ' + __file__)

_crystal_alignment_dict = {'main': {'roll': 'Johann Main Crystal Roll',
                                    'yaw':  'Johann Main Crystal Yaw',
                                    'x':    'Johann Crystal Assy X'},
                           'aux2': {'roll': 'Johann Aux2 Crystal Roll',
                                    'yaw':  'Johann Aux2 Crystal Yaw',
                                    'x':    'Johann Aux2 Crystal X',
                                    'y':    'Johann Aux2 Crystal Y'},
                           'aux3': {'roll': 'Johann Aux3 Crystal Roll',
                                    'yaw':  'Johann Aux3 Crystal Yaw',
                                    'x':    'Johann Aux3 Crystal X',
                                    'y':    'Johann Aux3 Crystal Y'},
                           'aux4': {'roll': 'Johann Aux4 Crystal Roll',
                                    'yaw':  'Johann Aux4 Crystal Yaw',
                                    'x':    'Johann Aux4 Crystal X',
                                    'y':    'Johann Aux4 Crystal Y'},
                           'aux5': {'roll': 'Johann Aux5 Crystal Roll',
                                    'yaw':  'Johann Aux5 Crystal Yaw',
                                    'x':    'Johann Aux5 Crystal X',
                                    'y':    'Johann Aux5 Crystal Y'}
                           }


def johann_focus_on_one_crystal_plan(crystal, yaw_shift=1200, print_msg=True):
    if print_msg:
        print_to_gui(f'Setting the focus on the {crystal} crystal. Moving other crystals from the field of view.',
                     add_timestamp=True, tag='Spectrometer')

    enabled_crystals = johann_emission.enabled_crystals_list
    unwanted_crystals = [c for c in enabled_crystals if c != crystal]
    for cr in unwanted_crystals:
        yaw_direction = 1 if cr in ['aux3', 'aux5'] else -1
        yield from move_relative_motor_plan(motor_attr=_crystal_alignment_dict[cr]['yaw'],
                                            based_on='description',
                                            rel_position=yaw_shift * yaw_direction)


def undo_johann_focus_on_one_crystal_plan(crystal, yaw_shift=1200):
    print_to_gui(f'Focus was on the {crystal} crystal. Moving other crystals back into the field of view.',
                 add_timestamp=True, tag='Spectrometer')
    yield from johann_focus_on_one_crystal_plan(crystal, yaw_shift=-yaw_shift, print_msg=False)

def move_rowland_circle_R_plan(new_R=None, energy=None, translations_only=True):
    motors_to_move = copy.deepcopy(johann_emission.real_keys)
    if translations_only:
        _motors_to_move = []
        for motor in motors_to_move:
            if not (motor.endswith('yaw') or motor.endswith('roll')):
                _motors_to_move.append(motor)
        motors_to_move = _motors_to_move

    old_pos_dict = johann_emission._forward({'energy': energy})
    rowland_circle.R = new_R

    if energy is None:
        energy = johann_emission.energy.position
    new_pos_dict = johann_emission._forward({'energy': energy})
    for motor in motors_to_move:
        motor_obj = getattr(johann_emission, motor)
        # print(motor, new_pos_dict[motor], f'delta={new_pos_dict[motor] - old_pos_dict[motor]}')
        # motor_obj.move(new_pos_dict[motor])
        yield from bps.mv(motor_obj, new_pos_dict[motor])

def move_johann_spectrometer_energy(energy : float=-1):
    # dealing with crystal assembly
    bragg = rowland_circle.e2bragg(float(energy))
    print_to_gui(f'Moving crystal analyzer assembly to {energy}')
    yield from bps.mv(johann_all_crystals, bragg, wait=False)

    # dealing with detector arm
    current_det_arm_bragg = johann_det_arm.bragg.position
    bragg_arr = np.linspace(current_det_arm_bragg, bragg, int(np.abs(bragg - current_det_arm_bragg)/0.25) + 2)
    energy_arr = rowland_circle.bragg2e(bragg_arr)
    for _bragg, _energy in zip(bragg_arr, energy_arr):
        print_to_gui(f'Moving detector arm to {_energy}')
        yield from bps.mv(johann_det_arm, _bragg, wait=True)


def prepare_johann_scan_plan(detectors, spectrometer_energy, spectrometer_config_uid):
    ensure_pilatus_is_in_detector_list(detectors)
    if spectrometer_config_uid is not None:
        johann_spectrometer_manager.set_config_by_uid(spectrometer_config_uid)
    if spectrometer_energy is not None:
        yield from move_johann_spectrometer_energy(spectrometer_energy)
    # yield from bps.mv(johann_emission, spectrometer_energy)

def prepare_johann_metadata_and_kwargs(**kwargs):
    metadata = kwargs.pop('metadata')
    j_metadata = {'spectrometer': 'johann',
                  'spectrometer_config': rowland_circle.config,}
    if 'spectrometer_energy' in kwargs.keys():
        spectrometer_energy = kwargs.pop('spectrometer_energy')
        j_metadata['spectrometer_energy'] = spectrometer_energy
    if 'spectrometer_config_uid' in kwargs.keys():
        j_metadata['spectrometer_config_uid'] = kwargs.pop('spectrometer_config_uid')
    return {**j_metadata, **metadata}, kwargs


def collect_n_exposures_johann_plan(**kwargs):
    yield from prepare_johann_scan_plan(kwargs['detectors'], kwargs['spectrometer_energy'], kwargs['spectrometer_config_uid'])
    metadata, kwargs = prepare_johann_metadata_and_kwargs(**kwargs)
    metadata['spectrometer_config']['scan_type'] = 'constant energy'
    yield from collect_n_exposures_plan(metadata=metadata, **kwargs)


def step_scan_johann_herfd_plan(**kwargs):
    yield from prepare_johann_scan_plan(kwargs['detectors'], kwargs['spectrometer_energy'], kwargs['spectrometer_config_uid'])
    metadata, kwargs = prepare_johann_metadata_and_kwargs(**kwargs)
    metadata['spectrometer_config']['scan_type'] = 'constant energy'
    yield from step_scan_plan(metadata=metadata, **kwargs)

def fly_scan_johann_herfd_plan(**kwargs):
    # rixs_file_name = kwargs.pop('rixs_file_name')
    yield from prepare_johann_scan_plan(kwargs['detectors'], kwargs['spectrometer_energy'], kwargs['spectrometer_config_uid'])
    metadata, kwargs = prepare_johann_metadata_and_kwargs(**kwargs)
    metadata['spectrometer_config']['scan_type'] = 'constant energy'
    yield from fly_scan_plan(metadata=metadata, **kwargs)


def get_johann_xes_step_scan_md(name, comment, detectors_dict, mono_energy, emission_energy_list, emission_time_list, element, e0, line, spectrometer_config_uid, metadata):
    try:
        full_element_name = getattr(elements, element).name.capitalize()
    except:
        full_element_name = element
    md_general = get_scan_md(name, comment, detectors_dict, '.dat')

    md_scan = {'experiment': 'step_scan',
               'spectrometer': 'johann',
               'hhm_energy': mono_energy,
               'spectrometer_config': rowland_circle.config,
               'spectrometer_config_uid': spectrometer_config_uid,
               'spectrometer_energy_steps': emission_energy_list,
               'spectrometer_time_steps': emission_time_list,
               'element': element,
               'element_full': full_element_name,
               'line': line,
               'e0': e0,}
    return {**md_scan, **md_general, **metadata}

def step_scan_johann_xes_plan(name=None, comment=None, detectors=[],
                              mono_energy=None, mono_angle_offset=None,
                              emission_energy_list=None, emission_time_list=None,
                              element='', line='', e0=None,
                              spectrometer_config_uid=None,
                              metadata=None):
    if metadata is None: metadata = {}

    default_detectors = [apb_ave, hhm_encoder]
    # default_detectors = []
    aux_detectors = get_detector_device_list(detectors, flying=False)
    all_detectors = default_detectors + aux_detectors
    detectors_dict = {k: {'device': v} for k, v in zip(detectors, aux_detectors)}

    if mono_angle_offset is not None: hhm.set_new_angle_offset(mono_angle_offset)
    yield from bps.mv(hhm.energy, mono_energy)
    yield from prepare_johann_scan_plan(detectors, emission_energy_list[0], spectrometer_config_uid)

    md = get_johann_xes_step_scan_md(name, comment, detectors_dict, mono_energy, emission_energy_list, emission_time_list, element,
                                     e0, line, spectrometer_config_uid, metadata)
    yield from general_energy_step_scan(all_detectors, johann_emission, emission_energy_list, emission_time_list, md=md)


def get_johann_xes_fly_scan_md(name, comment, detectors_dict, mono_energy, spectrometer_central_energy, relative_trajectory, element, e0, line, spectrometer_config_uid, metadata):
    try:
        full_element_name = getattr(elements, element).name.capitalize()
    except:
        full_element_name = element
    md_general = get_scan_md(name, comment, detectors_dict, '.dat')

    md_scan = {'experiment': 'epics_fly_scan',
               'hhm_energy': mono_energy,
               'spectrometer': 'johann',
               'spectrometer_config': rowland_circle.config,
               'spectrometer_config_uid': spectrometer_config_uid,
               'spectrometer_central_energy': spectrometer_central_energy,
               'spectrometer_relative_trajectory': relative_trajectory,
               'element': element,
               'element_full': full_element_name,
               'line': line,
               'e0': e0,}
    return {**md_scan, **md_general, **metadata}

def epics_fly_scan_custom_johann_piezo_plan(crystals=None, axis=None, detectors=[], relative_trajectory: dict=None, md: dict=None):
    motor_dict = {}
    trajectory_dict = {}

    if not all([(cr in relative_trajectory) for cr in crystals]):
        raise KeyError('The crystals in relative_trajectory must match the requested crystals!\n' +
                       f'{crystals=}\n' +
                       f'{relative_trajectory.keys()}')

    for crystal in crystals:
        motor_description = _crystal_alignment_dict[crystal][axis]
        motor_device = get_motor_device(motor_description, based_on='description')
        motor_pos = motor_device.position
        motor_dict[crystal] = motor_device
        trajectory_dict[crystal] = {
            'positions': [(motor_pos + delta) for delta in relative_trajectory[crystal]['positions']],
            'durations': copy.deepcopy(relative_trajectory[crystal]['durations'])}

    yield from general_epics_motor_fly_scan(detectors, motor_dict, trajectory_dict, md=md)


def epics_fly_scan_johann_xes_plan(name=None, comment=None, detectors=None,
                                   mono_energy=None, mono_angle_offset=None,
                                   spectrometer_central_energy=None,
                                   relative_trajectory=None, trajectory_as_energy=False,
                                   crystal_selection=None,
                                   element='', e0=0, line='',
                                   spectrometer_config_uid=None,
                                   metadata=None):
    if detectors is None: detectors = []
    if metadata is None: metadata = {}
    default_detectors = [apb_stream]
    aux_detectors = get_detector_device_list(detectors, flying=True)
    all_detectors = default_detectors + aux_detectors
    detectors_dict = {k: {'device': v} for k, v in zip(detectors, aux_detectors)}
    if mono_angle_offset is not None: hhm.set_new_angle_offset(mono_angle_offset)
    yield from bps.mv(hhm.energy, mono_energy)
    yield from prepare_johann_scan_plan(detectors, spectrometer_central_energy, spectrometer_config_uid)

    if crystal_selection is None:
        crystal_selection = [cr for cr, enabled in johann_emission.enabled_crystals.items() if enabled]
    if type(crystal_selection) == str:
        crystal_selection = [crystal_selection]

    if trajectory_as_energy:
        relative_trajectory = rowland_circle.convert_energy_trajectory_to_bragg(spectrometer_central_energy,
                                                                                relative_trajectory,
                                                                                crystal_selection)

    if not all((cr in relative_trajectory) for cr in crystal_selection):
        print_to_gui('Reformatting the trajectory for each individual crystal.', tag='Spectrometer', add_timestamp=True)
        _relative_trajectory = copy.deepcopy(relative_trajectory)
        relative_trajectory = {}
        for crystal in crystal_selection:
            relative_trajectory[crystal] = copy.deepcopy(_relative_trajectory)

    md = get_johann_xes_fly_scan_md(name, comment, detectors_dict, mono_energy, spectrometer_central_energy, relative_trajectory, element,
                                     e0, line, spectrometer_config_uid, metadata)

    yield from epics_fly_scan_custom_johann_piezo_plan(crystals=crystal_selection, axis='roll', detectors=all_detectors,
                                                relative_trajectory=relative_trajectory, md=md)

'''
for _energy in range(8015, 8060, 5):
# for _energy in [8020, 8035, 8050]:
# # for _energy in [8035]:
    for i in range(1):
        plan = epics_fly_scan_johann_xes_plan(name=f'fly_Cu2O calibration {i+1} {_energy}', comment='', detectors=['Pilatus 100k New'],
                                           mono_energy=_energy, mono_angle_offset=None,
                                           spectrometer_central_energy=None, relative_trajectory={'positions': [-900, 900],
                                                                             'durations': [30]},
                                           crystal_selection=None,
                                           element='Cu', e0=8046, line='Ka1',
                                           spectrometer_config_uid=None,
                                           metadata={})
        RE(plan)


# for i in range(5):
#     relative_trajectory = {'positions': [-900, 900],
#                            'durations': [60]}
#     plan = epics_fly_scan_johann_xes_plan(name=f'fly_Cu_foil 9000 30s att2 {i+1:04d}', comment='', detectors=['Pilatus 100k New'],
#                                        mono_energy=9000, mono_angle_offset=None,
#                                        spectrometer_central_energy=None, relative_trajectory=relative_trajectory,
#                                        crystal_selection=None,
#                                        element='Cu', e0=8046, line='Ka1',
#                                        spectrometer_config_uid=None,
#                                        metadata={'_relative_trajectory_info': relative_trajectory})
#     RE(plan)
plans = []

for i in range(3):
    relative_trajectory = {'positions': [-900, -600, -300,  210,  510,  900],
                           'durations': [5/1, 20/1, 8.5/1, 20/1, 6.5/1]}
    plans.append({'plan_name': 'epics_fly_scan_johann_xes_plan',
                  'plan_kwargs': {'name': f'fly Cu2O 9000 60s traj {i+1:04d}', 'comment': '', 'detectors': ['Pilatus 100k New'],
                                       'mono_energy': 9000, 'mono_angle_offset': None,
                                       'spectrometer_central_energy': None, 'relative_trajectory': relative_trajectory,
                                       'crystal_selection': None,
                                       'element': 'Cu', 'e0': 8046, 'line': 'Ka1',
                                       'spectrometer_config_uid': None,
                                       'metadata': {'_relative_trajectory_info': relative_trajectory}}})

for i in range(5):
    relative_trajectory = {'positions': [-900, -600, -300,  210,  510,  900],
                           'durations': [5/2, 20/2, 8.5/2, 20/2, 6.5/2]}
    plans.append({'plan_name': 'epics_fly_scan_johann_xes_plan',
                  'plan_kwargs': {'name': f'fly_Cu_foil 9000 30s traj {i+1:04d}', 'comment': '', 'detectors': ['Pilatus 100k New'],
                                       'mono_energy': 9000, 'mono_angle_offset': None,
                                       'spectrometer_central_energy': None, 'relative_trajectory': relative_trajectory,
                                       'crystal_selection': None,
                                       'element': 'Cu', 'e0': 8046, 'line': 'Ka1',
                                       'spectrometer_config_uid': None,
                                       'metadata': {'_relative_trajectory_info': relative_trajectory}}})

plan_processor.add_plans(plans)
'''

def deal_with_sample_coordinates_for_rixs(sample_coordinates, energy_list, name):
    if type(sample_coordinates) == list:
        assert len(sample_coordinates) == len(energy_list), 'number of positions on the sample must match the number of energy points on emission/mono energy grid'
    else:
        sample_coordinates = [sample_coordinates] * len(energy_list)

    if type(name) == list:
        assert len(name) == len(energy_list), 'number of positions on the sample must match the number of energy points on emission/mono energy grid'
    else:
        name = [name] * len(energy_list)

    return sample_coordinates, name


def get_johann_rixs_md(name, element_line, line, e0_line, metadata):
    # metadata['rixs_file_name'] = create_interp_file_name(name, '.rixs')
    metadata['element_line'] = element_line
    metadata['line'] = line
    metadata['e0_line'] = e0_line
    return metadata


def johann_rixs_plan_bundle(plan_name, name=None, comment=None, detectors=[],
                            trajectory_filename=None, mono_angle_offset=None,
                            emission_energy_list=None, sample_coordinates=None,
                            element='', edge='', e0=None, element_line='', line='', e0_line=None,
                            rixs_kwargs={}, spectrometer_config_uid=None, metadata={}):
    sample_coordinates, names = deal_with_sample_coordinates_for_rixs(sample_coordinates, emission_energy_list, name)
    metadata = get_johann_rixs_md(name, element_line, line, e0_line, metadata)
    plans = []
    for emission_energy, sample_position, name in zip(emission_energy_list, sample_coordinates, names):

        if sample_position is not None:
            plans.append({'plan_name': 'move_sample_stage_plan',
                          'plan_kwargs': {'sample_coordinates': sample_position}})

        plans.append({'plan_name': plan_name,
                      'plan_kwargs': {'name': f'{name} {emission_energy:0.2f}',
                                      'comment': comment,
                                      'detectors': detectors,
                                      'trajectory_filename': trajectory_filename,
                                      'element': element,
                                      'edge': edge,
                                      'e0': e0,
                                      'spectrometer_energy': emission_energy,
                                      'spectrometer_config_uid': spectrometer_config_uid,
                                      'mono_angle_offset': mono_angle_offset,
                                      'metadata': metadata}})
        # deal with rixs_kwargs
    return plans

def fly_scan_johann_rixs_plan_bundle(**kwargs):
    return johann_rixs_plan_bundle('fly_scan_johann_herfd_plan', **kwargs)

def step_scan_johann_rixs_plan_bundle(**kwargs):
    return johann_rixs_plan_bundle('step_scan_johann_herfd_plan', **kwargs)


def johann_reset_fly_caibration_data_plan():
    rowland_circle.reset_fly_calibration_dict()
    yield from bps.null()

def johann_add_scan_to_fly_calibration_data_plan(_uid=-1):
    start = dict(db[_uid].start)  # snapshot once — avoids repeated HTTP round-trips on Tiled
    uid = start['uid']
    rowland_circle.fly_calibration_dict['data'].append({
        'mono_energy': start['hhm_energy'],
        'scan_scope': start['scan_scope'],
        'alignment_plan': 'epics_fly_scan_johann_emission_alignment_plan_bundle', # this is to recycle the code for analyzing similar scan stacks
        'uid': uid,
        'rois': rowland_circle.enabled_crystals_list})
    rowland_circle.save_current_spectrometer_config_to_settings()
    yield from bps.null()

def johann_process_fly_calibration_data_plan(spectrometer_config_uid=None):
    johann_analyze_alignment_data(alignment_data=rowland_circle.fly_calibration_dict['data'],
                                  scan_scope='fly_scan_calibration') # need to do something about ROIs in the analysis

    data = rowland_circle.fly_calibration_dict['data']
    LUT = {'energy': [entry['mono_energy'] for entry in data]}
    for crystal in rowland_circle.enabled_crystals_list: # may be use crystals in the data
        LUT[crystal] = [entry[crystal]['com_loc'] for entry in data]

    rowland_circle.fly_calibration_dict['LUT'] = LUT
    rowland_circle.save_current_spectrometer_config_to_settings()
    if spectrometer_config_uid is not None:
        johann_spectrometer_manager.update_config_at_uid(spectrometer_config_uid, rowland_circle.config)
    yield from bps.null()

def fly_spectrometer_scan_johann_rixs_plan_bundle(name=None, comment=None, detectors=[],
                                                  mono_angle_offset=None,
                                                  trajectory_filename = None,
                                                  element='', edge=None, e0=0,
                                                  element_line=None, line='', e0_line=0,
                                                  spectrometer_central_energy=None, relative_trajectory=None,
                                                  trajectory_as_energy=False, scan_for_calibration_purpose=None,
                                                  crystal_selection=None,
                                                  sample_coordinates=None,
                                                  rixs_kwargs=None,
                                                  spectrometer_config_uid=None,
                                                  metadata=None):

    if metadata is None: metadata = {}
    if rixs_kwargs is not None: metadata = {**metadata, **rixs_kwargs}

    mono_energy_list, _ = read_step_scan_file(trajectory_filename)

    metadata = get_johann_rixs_md(name, element_line, line, e0_line, metadata)

    sample_coordinates, names = deal_with_sample_coordinates_for_rixs(sample_coordinates, mono_energy_list, name)

    plans = []

    if scan_for_calibration_purpose:  # reset previous calibration
        plans.append({'plan_name': 'johann_reset_fly_caibration_data_plan',
                      'plan_kwargs': {}})
        metadata['scan_scope'] = 'fly_scan_calibration'

    for mono_energy, sample_position, name in zip(mono_energy_list, sample_coordinates, names):

        if sample_position is not None:
            plans.append({'plan_name': 'move_sample_stage_plan',
                          'plan_kwargs': {'sample_coordinates': sample_position}})
        energy_str = f'{mono_energy:0.2f}'.replace('.', '_')
        plans.append({'plan_name': 'epics_fly_scan_johann_xes_plan',
                      'plan_kwargs': {'name': f'{name} {energy_str}',
                                      'comment': comment,
                                      'detectors': detectors,
                                      'mono_energy': mono_energy,
                                      'mono_angle_offset': mono_angle_offset,
                                      'spectrometer_central_energy': spectrometer_central_energy,
                                      'relative_trajectory': relative_trajectory,
                                      'trajectory_as_energy': trajectory_as_energy,
                                      'element': element_line,
                                      'line': line,
                                      'e0': e0_line,
                                      'spectrometer_config_uid': spectrometer_config_uid,
                                      'metadata': metadata}})

        if scan_for_calibration_purpose: # add a plan to include the last scan into the fly_calibration dict
            plans.append({'plan_name': 'johann_add_scan_to_fly_calibration_data_plan',
                          'plan_kwargs': {'_uid': -1}})

    if scan_for_calibration_purpose: # add a plan to process the calibration
        plans.append({'plan_name': 'johann_process_fly_calibration_data_plan',
                      'plan_kwargs': {'spectrometer_config_uid': spectrometer_config_uid}})

    return plans
