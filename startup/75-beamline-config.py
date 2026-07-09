print(ttime.ctime() + ' >>>> ' + __file__)

from xas.trajectory import TrajectoryCreator



def tune_beamline_plan(extended_tuning : bool = False, enable_fb_in_the_end : bool = True):

    if extended_tuning:
        # tune_elements_list = tune_elements_ext
        tune_elements_list = tune_elements_alt
    else:
        tune_elements_list = tune_elements

    print_to_gui(f'[Beamline tuning] Starting...')
    yield from bps.mv(hhm.fb_status, 0)
    # print('bla')

    # yield from bps.mv(bpm_fm, 'insert')

    for element in tune_elements_list:
        # detector = detector_dictionary[element['detector']]['device']
        detector = element['detector']

        if 'fb_enable' in element.keys():
            if element['fb_enable']:
                hhm_feedback.update_center()
                hhm.fb_status.put(1)


        # if detector == bpm_fm:
        if detector == 'Focusing mirror BPM':
            yield from bps.mv(bpm_fm, 'insert')
        else:
            yield from bps.mv(bpm_fm, 'retract')


        # motor = motor_dictionary[element['motor']]['object']
        motor = element['motor']
        yield from tuning_scan(motor, detector,
                              element['range'],
                              element['step'],
                              n_tries=element['retries'])
        # turn camera into continuous mode
        if hasattr(detector, 'image_mode'):
            yield from bps.mv(getattr(detector, 'image_mode'), 2)
            yield from bps.mv(getattr(detector, 'acquire'), 1)

    # yield from bps.mv(bpm_fm, 'retract')
    if enable_fb_in_the_end:
        hhm_feedback.update_center()
        # yield from update_hhm_fb_center(truncate_data=truncate_data)
        hhm.fb_status.put(1)
    print('[Beamline tuning] Beamline tuning complete')

def optimize_beamline_plan(energy: int = -1, extended_tuning: bool = False, force_prepare = False, enable_fb_in_the_end=False):
    old_energy = hhm.energy.position
    if force_prepare or ((np.abs((energy-old_energy)/old_energy)> 0.1) or (np.sign(old_energy-13000)) != (np.sign(energy-13000))):
        yield from shutter.close_plan()
        yield from simple_prepare_beamline_plan(energy, move_cm_mirror = True)
        yield from tune_beamline_plan(extended_tuning=extended_tuning, enable_fb_in_the_end=enable_fb_in_the_end)
    else:
        # print_to_gui(f'Beamline is already prepared for {energy} eV', stdout=stdout)
        yield from bps.mv(hhm.energy, energy)

def tabulate_hhmy_position_plan(stdout=sys.stdout):
    # _energies = [13000, 15000, 17500, 20000, 22500, 25000, 27500, 30000]  # np.arange(5000, 11000, 1000)
    # _energies = [4900, 5100, 5500, 6000, 7000, 8000, 9000, 10000, 11000, 12000]  # np.arange(5000, 11000, 1000)
    _energies = [25000, 27500, 30000]  # np.arange(5000, 11000, 1000)
    # _energies = [23000]
    data_df = pd.DataFrame(columns=['energy', 'hhmy', 'uid'])

    for energy in _energies:
        # enable_fb_in_the_end = energy>13000

        yield from optimize_beamline_plan(energy, extended_tuning=True, force_prepare=True, enable_fb_in_the_end=False)
        uid = db[-1].start['uid']
        data_df = data_df.append({'energy' : energy,
                                  'hhmy' : hhm.y.user_readback.get(),
                                  #'hhrmy' : hhrm.y.user_readback.get(),
                                  'uid' : uid},
                                   ignore_index=True)
        # data_df.to_json('/nsls2/data/iss/legacy/xf08id/calibration/beamline_hhmy_tabulation_2022_05_19.json')
        data_df.to_json('/nsls2/data/iss/legacy/xf08id/calibration/beamline_hhmy_tabulation_2022_05_19_high_energies_2.json')
        # data_df.to_json('/nsls2/xf08id/Sandbox/Beamline_components/2022_05_19_beamline_tabulation/beamline_hhmy_tabulation_high_energies.json')

def append_datapoint_plan(data_df):
    uid = db[-1].start['uid']
    data_df = data_df.append({'energy': energy,
                              'hhmy': hhm.y.user_readback.get(),
                              'uid': uid},
                             ignore_index=True)
    data_df.to_json(
        '/nsls2/data/iss/legacy/xf08id/calibration/beamline_hhmy_tabulation_2023_05_04.json')


def tabulate_hhmy_position_plan_bundle():
    _energies = [7000, 8000]
    try:
        data_df = pd.read_json('/nsls2/data/iss/legacy/xf08id/calibration/beamline_hhmy_tabulation_2023_05_04.json')
    except:
        data_df = pd.DataFrame(columns=['energy', 'hhmy', 'uid'])

    plans = []
    for energy in _energies:
        plans.append({'plan_name': 'move_motor_plan',
                      'plan_kwargs' : {'motor_attr' : hhm.energy.name,
                                       'based_on' : 'object_name',
                                       'position' : energy}})
        plans.append({'plan_name': 'tune_beamline_plan_bundle',
                      'plan_kwargs': {}})




# def gen_trajectory(element, edge):
#     trajectory_creator = TrajectoryCreator(servocycle=hhm.servocycle, pulses_per_deg=hhm.pulses_per_deg)
#     E0 = xraydb.xray_edge(element, edge).energy
#
#     trajectory_creator.define(edge_energy=E0,
#                               dsine_preedge_duration=4,
#                               dsine_edge_duration=6,
#                               dsine_postedge_duration=20,
#                               dsine_preedge_frac=0.5,
#                               dsine_postedge_frac=0.3,
#                               trajectory_type='Double Sine/Constant Edge',
#                               pad_time=0.5)
#     trajectory_creator.elem = f'{element}'
#     trajectory_creator.edge = f'{edge}'
#     trajectory_creator.e0 = E0
#     trajectory_creator.interpolate()
#     trajectory_creator.revert()
#     trajectory_creator.tile(reps=1,
#                             single_direction=True)
#     trajectory_creator.e2encoder(hhm.angle_offset.value)
#     filename = f'{element}-{edge}.txt'
#     filepath = trajectory_manager.trajectory_path + filename
#     np.savetxt(filepath,
#                trajectory_creator.energy_grid, fmt='%.6f',
#                header=f'element: {trajectory_creator.elem}, edge: {trajectory_creator.edge}, E0: {trajectory_creator.e0}')  # , scan_direction: {self.traj_creator.direction}')
#     trajectory_manager.load(filename, 1, is_energy=True, offset=hhm.angle_offset.value)
#     trajectory_manager.init(1)



class BeamlineConfig:

    def __init__(self):
        self.filepath = '/nsls2/xf08id/Sandbox/Beamline_components/beamline_config/config.json'
        try:
            self.data = pd.read_json(self.filepath)
        except:
            self.data = pd.DataFrame(columns=['timestamp', 'energy', 'hhm_pitch', 'hhm_y', 'hhrm_y',
                                              'hhm_angle_offset', 'hhm_fb_center', 'hhm_fb_line'])


    def save_current_config(self):

        timestamp = ttime.ctime()
        energy = hhm.energy.position
        hhm_pitch = hhm.pitch.position
        hhm_y =hhm.y.position
        hhrm_y = hhrm.y.position
        hhm_angle_offset = hhm.angle_offset.value
        hhm_fb_center = hhm.fb_center.value
        hhm_fb_line = hhm.fb_line.value


        self.data = self.data.append({'timestamp' : timestamp,
                                      'energy' : energy,
                                      'hhm_pitch' : hhm_pitch,
                                      'hhm_y' : hhm_y,
                                      'hhrm_y' : hhrm_y,
                                      'hhm_angle_offset' : hhm_angle_offset,
                                      'hhm_fb_center' : hhm_fb_center,
                                      'hhm_fb_line' : hhm_fb_line},
                                      ignore_index=True)

        if os.environ.get('SAVE_CONFIG_TO_JSON'):
            self.data.to_json(self.filepath)

beamline_config = BeamlineConfig()



def prepare_tune_calibrate_plan(element, edge):
    if validate_element_edge_in_db_proc(element):
        energy = xraydb.xray_edge(element, edge).energy

        yield from optimize_beamline_plan(energy, tune_elements=tune_elements, force_prepare=True,
                                          enable_fb_in_the_end=True)

        gen_trajectory(element, edge)

        # yield from adjust_ic_gains()
        # yield from adjust_ic_gains()
        yield from calibrate_energy_plan(element, edge, dE=35, plot_fun=None)
        yield from bps.mv(hhm.energy, energy)
        beamline_config.save_current_config()
    else:
        pass

# foil_list = [('Ti', 'K'),
#              ('V' , 'K'),
#              ('Cr', 'K'),
#              ('Mn', 'K'),
#              ('Fe', 'K'),
#              ('Co', 'K'),
#              ('Ni', 'K'),
#              ('Cu', 'K'),
#              ('Zn', 'K'),
#              ('Ta', 'L3'),
#              ('Re', 'L3'),
#              ('Ir', 'L3'),
#              ('Pt', 'L3'),
#              ('Au', 'L3'),
#              ('Pb', 'L3'),
#              ('Se', 'K'),
#              ('Zr', 'K'),
#              ('Nb', 'K'),
#              ('Mo', 'K'),
#              ('Ru', 'K'),
#              ('Rh', 'K'),
#              ('Pd', 'K'),
#              ('Ag', 'K'),
#              ('Cd', 'K'),
#              ('In', 'K'),
#              ('Sn', 'K')]


def go_over_foils(_foil_list):
    for foil in _foil_list:
        element, edge = foil
        yield from prepare_tune_calibrate_plan(element, edge)

foil_list = [('Cr', 'K'),
             ('Mn', 'K'),
             ('Fe', 'K'),
             ('Co', 'K'),
             ('Ni', 'K'),
             ('Cu', 'K'),
             ('Zn', 'K'),
             ('Ta', 'L3'),
             ('Pt', 'L3'),
             ('Au', 'L3'),
             ('Zr', 'K'),
             ('Nb', 'K'),
             ('Mo', 'K'),
             # ('Rh', 'K'),
             # ('Pd', 'K'),
             # ('Ag', 'K'),
             # ('Cd', 'K'),
             # ('Sn', 'K'),
            ]

# foil_list = [('Ni', 'K')]
# RE(go_over_foils(foil_list))

