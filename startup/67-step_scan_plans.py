print(ttime.ctime() + ' >>>> ' + __file__)


from ophyd import Staged
import bluesky.plan_stubs as bps
import bluesky.plans as bp


def step_scan_action_factory(energy_steps, time_steps):
    energy_to_time_step = dict(zip(energy_steps, time_steps))
    apb_ave_acquire_rate = apb_ave.acq_rate.get() * 1000
    def per_step_action(detectors, motor, step):
        time_step=energy_to_time_step[step]
        # yield from set_exposure_time_plan(detectors, time_step)
        for det in detectors:
            if det.name == 'apb_ave':
                # samples = 250 * (np.ceil(time_step*1005/250)) # sample length can only be changed in steps of 250
                samples = 250 * (np.round(time_step * 1005 / 250))  # sample length can only be changed in steps of 250
                # samples = np.ceil(time_step * apb_ave_acquire_rate ) # this is under investigation, but it seems that this can cause problems since the step is not enforced to be 250, hence the "yield from bps.abs_set(det.sample_len, samples, wait=True)" executed below can hang
                # det.sample_len.put(samples, wait=True)
                # det.wf_len.put(samples, wait=True)
                # yield from bps.null()
                yield from bps.abs_set(det.sample_len, samples, wait=True)
                yield from bps.abs_set(det.wf_len, samples, wait=True)
            elif (det.name == 'pil100k') or (det.name == 'pil100k2'):
                yield from bps.mv(det.cam.acquire_time, time_step)
            elif det.name == 'xs':
                yield from bps.mv(det.settings.acquire_time, time_step)
            elif det.name == 'ge_detector':
                yield from bps.mv(det.settings.acquire_time, time_step)

        yield from bps.mv(motor, step)
        devices = [*detectors, motor]
        start = ttime.time()
        # yield from shutter.open_plan(printing=False)
        yield from bps.trigger_and_read(devices=devices)
        # yield from shutter.close_plan(printing=False)
        print_to_gui(f'Total exposure time for this point is {ttime.time() - start} s')

    return per_step_action


def read_step_scan_file(filename):
    data = np.genfromtxt(trajectory_manager.trajectory_path + filename)
    energy, dwell_time = data[:, 0], data[:, 1]
    return energy.tolist(), dwell_time.tolist()


def get_step_scan_md(name, comment, trajectory_filename, detectors_dict, element, e0, edge, metadata):
    md_general = get_hhm_scan_md(name, comment, trajectory_filename, detectors_dict, element, e0, edge, metadata, fn_ext='.dat')
    md_scan = {'experiment': 'step_scan'}
    return {**md_general, **md_scan}


def general_energy_step_scan(detectors, motor, motor_positions, dwell_times, md):
    n_exposures = len(motor_positions)
    yield from prepare_detectors_for_exposure_plan(detectors, n_exposures=n_exposures)
    yield from shutter.open_plan()
    yield from bp.list_scan(detectors,
                            motor,
                            motor_positions,
                            per_step=step_scan_action_factory(motor_positions, dwell_times),
                            md=md)
    yield from shutter.close_plan()


def step_scan_plan(name=None, comment=None, trajectory_filename=None, mono_angle_offset=None, detectors=[], element='', e0=0, edge='', metadata={}):

    energy_list, time_list = read_step_scan_file(trajectory_filename)
    if mono_angle_offset is not None: hhm.set_new_angle_offset(mono_angle_offset)
    default_detectors = [apb_ave, hhm_encoder]
    aux_detectors = get_detector_device_list(detectors, flying=False)
    all_detectors = default_detectors + aux_detectors
    for det in all_detectors:
        if hasattr(det,'total_points'):
            det.total_points.put(len(energy_list))
    detectors_dict = {k: {'device': v} for k, v in zip(detectors, aux_detectors)}
    md = get_step_scan_md(name, comment, trajectory_filename, detectors_dict, element, e0, edge, metadata)

    yield from general_energy_step_scan(all_detectors, hhm.energy, energy_list, time_list, md=md)