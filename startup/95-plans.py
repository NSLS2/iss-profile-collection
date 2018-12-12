import bluesky as bs
import bluesky.plans as bp
import time as ttime
from subprocess import call
import os
from isstools.conversions import xray
import signal
import sys

from ophyd.device import Kind



def energy_scan(start, stop, num, flyers=None, name='', **metadata):
    """
    Example
    -------
    >>> RE(energy_scan(11350, 11450, 2))
    """
    if flyers is None:
        flyers = [pb9.enc1, pba2.adc6, pba1.adc7]
    def inner():
        md = {'plan_args': {}, 'plan_name': 'step scan', 'name': name}
        md.update(**metadata)
        yield from bps.open_run(md=md)

    # Start with a step scan.
    plan = bp.scan([hhm_en.energy], hhm_en.energy, start, stop, num, md={'name': name})
    # Wrap it in a fly scan with the Pizza Box.
    plan = bpp.fly_during_wrapper(plan, flyers)
    # Working around a bug in fly_during_wrapper, stage and unstage the pizza box manually.

    for flyer in flyers:
        yield from bps.stage(flyer)
    yield from bps.stage(hhm)

    plan = bpp.pchain(plan)

    yield from plan


def energy_multiple_scans(start, stop, repeats, name='', **metadata):
    """
    Example
    -------
    >>> RE(energy_scan(11350, 11450, 2))
    """
    flyers = [pb9.enc1, pba2.adc6, pba1.adc7]
    def inner():
        md = {'plan_args': {}, 'plan_name': 'energy_multiple_scans', 'name': name}
        md.update(**metadata)
        yield from bps.open_run(md=md)

        for i in range(0, repeats):
            print('Run:', i+1)
            hhm_en.energy.move(start)
            ttime.sleep(2)
            while (hhm_en.energy.moving == True):
                ttime.sleep(.1)
            hhm_en.energy.move(stop)
            ttime.sleep(2)
            while (hhm_en.energy.moving == True):
                ttime.sleep(.1)

        yield from bps.close_run()


    for flyer in flyers:
        yield from bps.stage(flyer)
    yield from bps.stage(hhm)

    yield from bpp.fly_during_wrapper(inner(), flyers)

    yield from bps.unstage(hhm)
    for flyer in flyers:
        yield from bps.unstage(flyer)



def get_offsets_plan(detectors, num = 1, name = '', **metadata):

    flyers = detectors 

    plan = bp.count(flyers, num, md={'plan_name': 'get_offset', 'name': name}, delay = 0.5)

    def set_offsets():
        for flyer in flyers:
            ret = flyer.volt.value
            yield from bps.abs_set(flyer.offset, ret, wait=True)

    yield from bpp.fly_during_wrapper(bpp.finalize_wrapper(plan, set_offsets()), flyers)



def tune(detectors, motor, start, stop, num, name='', **metadata):
    """
    Example
    -------
    >>> RE(tune([pba1.adc7], hhm.pitch,-2, 2, 5, ''), LivePlot('pba1.adc7_volt', 'hhm_pitch'))
    """

    flyers = detectors 

    plan = bp.relative_scan(flyers, motor, start, stop, num, md={'plan_name': 'tune ' + motor.name, 'name': name})
    
    if hasattr(flyers[0], 'kickoff'):
        plan = bpp.fly_during_wrapper(plan, flyers)
        plan = bpp.pchain(plan)

    yield from plan

def get_xia_energy_grid(e0, preedge_start, xanes_start, xanes_end, exafs_end, preedge_spacing, xanes_spacing, exafsk_spacing, int_time_preedge = 1, int_time_xanes = 1, int_time_exafs = 1, k_power = 0):
    preedge = np.arange(e0 + preedge_start, e0 + xanes_start, preedge_spacing)
    preedge_int = np.ones(len(preedge)) * int_time_preedge

    edge = np.arange(e0 + xanes_start, e0 + xanes_end, xanes_spacing)
    edge_int = np.ones(len(edge)) * int_time_xanes

    iterator = exafsk_spacing
    kenergy = 0
    postedge = np.array([])

    energy_end = xray.k2e(exafs_end, e0)
    exafs_int = []
    while(kenergy + e0 + xanes_end < energy_end):
        kenergy = xray.k2e(iterator, e0) - e0
        postedge = np.append(postedge, e0 + xanes_end + kenergy)
        exafs_int.append(int_time_exafs * (iterator ** k_power))
        iterator += exafsk_spacing

    integration_times = np.append(np.append(preedge_int, edge_int), np.array(exafs_int))
    grid = np.append(np.append(preedge, edge), postedge)
    return grid[::-1], integration_times
    #return np.append(np.append(preedge, edge), postedge)

def step_list_plan(detectors, motor, positions_grid, name = ''):
    """
    Example
    -------
    >>> Ni_energy_grid, time_grid = get_xia_energy_grid(8333, -200, -50, 30, 16, 10, 0.2, 0.04)
    >>> Ni_positions_grid = xray.energy2encoder(Ni_energy_grid) / 360000
    >>> RE(step_list_plan([xia1, pba1.adc7], hhm.theta, Ni_positions_grid), LivePlot('xia1_mca1_roi0_sum', 'hhm_theta'))
    """
    
    plan = bp.list_scan(detectors, motor, list(positions_grid), md={'name': name, 'plan_name': 'step_list_plan'})
    
    flyers = []
    for det in detectors:
        if hasattr(det, 'kickoff'):
            flyers.append(det)
            
    if len(flyers) > 0:
        plan = bpp.fly_during_wrapper(plan, flyers)
        
    yield from plan
    

    
def step_xia_scan(motor, filename, energy_grid, integration_times = np.array([])):
    """
    Example
    -------
    >>> TODO
    """
    xia1_chan1_array = []
    xia1_chan2_array = []
    xia1_chan3_array = []
    xia1_chan4_array = []
    i0_array = []

    if(len(integration_times)) == 0:
        integration_times = np.ones(len(energy_grid)) * xia1.real_time.value

    #energy_grid = np.arange(7112 - 50, 7112 + 50, 1)#get_xia_energy_grid(energy_start, e0, edge_start, edge_end, energy_end, preedge_spacing, xanes, exafsk)

    pba1.adc7.filepath.put('')
    pba1.adc7.enable_sel.put(0)
    xia1.collect_mode.put(0)
    while(xia1.collect_mode.value != 0):
        ttime.sleep(.01)
    for i in range(len(energy_grid)):
        print("[{}/{}]".format(i + 1, len(energy_grid)))
        if(xia1.real_time.value != integration_times[i]):
            xia1.real_time.put(integration_times[i])
        ttime.sleep(.005)
        motor.move(xray.energy2encoder(energy_grid[i])/360000)
        while(np.abs(motor.read()['hhm_theta']['value'] - motor.read()['hhm_theta_user_setpoint']['value']) > 0.00001 or motor.moving == True):
            ttime.sleep(.005)

        xia1.erase_start.put(1)
        ttime.sleep(.1)
        while(xia1.acquiring.value):
            ttime.sleep(.005)
        
        ttime.sleep(.1)

        i0_array.append([energy_grid[i], pba1.adc7.volt.value])
        xia1_chan1_array.append(xia1.mca_array1.value)
        xia1_chan2_array.append(xia1.mca_array2.value)
        xia1_chan3_array.append(xia1.mca_array3.value)
        xia1_chan4_array.append(xia1.mca_array4.value)
        
    pba1.adc7.enable_sel.put(1)
    
    np.savetxt('/GPFS/xf08id/xia_files/' + filename + '-i0', np.array(i0_array))
    np.savetxt('/GPFS/xf08id/xia_files/' + filename + '-1', np.array(xia1_chan1_array))
    np.savetxt('/GPFS/xf08id/xia_files/' + filename + '-2', np.array(xia1_chan2_array))
    np.savetxt('/GPFS/xf08id/xia_files/' + filename + '-3', np.array(xia1_chan3_array))
    np.savetxt('/GPFS/xf08id/xia_files/' + filename + '-4', np.array(xia1_chan4_array))

def general_scan_plan(detectors, motor, rel_start, rel_stop, num):
    
    plan = bp.relative_scan(detectors, motor, rel_start, rel_stop, num)
    
    if hasattr(detectors[0], 'kickoff'):
        plan = bpp.fly_during_wrapper(plan, detectors)

    yield from plan


def sampleXY_plan(detectors, motor, start, stop, num):
    """
    Example
    -------
    >>> RE(sampleXY_plan([pba1.adc7], samplexy.x, -2, 2, 5, ''), LivePlot('pba1.adc7_volt', 'samplexy_x'))
    """

    flyers = detectors 

    plan = bp.relative_scan(flyers, motor, start, stop, num)
    
    if hasattr(flyers[0], 'kickoff'):
        plan = bpp.fly_during_wrapper(plan, flyers)
        # Check if I can remove bpp.pchain

    yield from plan


def pb_scan_plan(detectors, motor, scan_center, scan_range, name = ''):

    flyers = detectors
    def inner():
        md = {'plan_args': {}, 'plan_name': 'pb_scan','experiment': 'pb_scan', 'name': name}
        #md.update(**metadata)
        yield from bps.open_run(md=md)
        yield from bps.sleep(.4)
        yield from bps.clear_checkpoint()
        yield from bps.abs_set(motor, scan_center + (scan_range / 2), wait=True)
        yield from bps.sleep(.4)
        yield from bps.close_run()
        yield from shutter.close_plan()
        yield from bps.abs_set(motor, scan_center, wait=True)

    def final_plan():
        for flyer in flyers:
            yield from bps.unstage(flyer)
        yield from bps.unstage(motor)

    yield from bps.abs_set(motor, scan_center - (scan_range / 2), wait=True)

    yield from shutter.open_plan()
    for flyer in flyers:
        yield from bps.stage(flyer)

    yield from bps.stage(motor)

    return (yield from bpp.fly_during_wrapper(bpp.finalize_wrapper(inner(), final_plan()),
                                              flyers))


def prep_trajectory(delay = 1):
    hhm.prepare_trajectory.put("1")
    while (hhm.trajectory_ready.value == 0):
        ttime.sleep(.1)
    while (hhm.trajectory_ready.value == 1):
        ttime.sleep(.1)
    ttime.sleep(delay)

def prep_traj_plan(delay = 0.1):
    yield from bps.abs_set(hhm.prepare_trajectory, '1', wait=True)

    # Poll the trajectory ready pv
    while True:
        ret = (yield from bps.read(hhm.trajectory_ready))
        if ret is None:
            break
        is_running = ret['hhm_trajectory_ready']['value']

        if is_running:
            break
        else:
            yield from bps.sleep(.1)

    while True:
        ret = (yield from bps.read(hhm.trajectory_ready))
        if ret is None:
            break
        is_running = ret['hhm_trajectory_ready']['value']

        if is_running:
            yield from bps.sleep(.05)
        else:
            break

    yield from bps.sleep(delay)

    curr_energy = (yield from bps.read(hhm.energy))

    if curr_energy is None:
        return
        raise Exception('Could not read current energy')

    curr_energy = curr_energy['hhm_energy']['value']
    print('Curr Energy: {}'.format(curr_energy))
    if curr_energy >= 12000:
        print('>12000')
        yield from bps.mv(hhm.energy, curr_energy + 100)
        yield from bps.sleep(1)
        print('1')
        yield from bps.mv(hhm.energy, curr_energy)


def execute_trajectory(name, **metadata):
    ''' Execute a trajectory on the flyers given:
            flyers : list of flyers to fly on
        scans on 'mono1' by default
        ignore_shutter : bool, optional
            If True, ignore the shutter
            (suspenders on shutter and ring current will be installed if not)
        NOTE: Not added yet
            (need to update isstools)
        ex:
            execute_trajectory(**md)
    '''
    #flyers = [pb4.di, pba2.adc7, pba1.adc6, pb9.enc1, pba1.adc1, pba2.adc6, pba1.adc7]
    #flyers = [pba2.adc7, pba1.adc6, pb9.enc1, pba1.adc1, pba2.adc6, pba1.adc7]
    flyers = [pba2.adc7, pba1.adc6, pba1.adc1, pba2.adc6, pba1.adc7, pb9.enc1]
    def inner():
        interp_fn = f"{ROOT_PATH}/{USER_FILEPATH}/{RE.md['year']}.{RE.md['cycle']}.{RE.md['PROPOSAL']}/{name}.txt"
        curr_traj = getattr(hhm, 'traj{:.0f}'.format(hhm.lut_number_rbv.value))
        md = {'plan_args': {},
              'plan_name': 'execute_trajectory',
              'experiment': 'transmission',
              'name': name,
              'interp_filename': interp_fn,
              'angle_offset': str(hhm.angle_offset.value),
              'trajectory_name': hhm.trajectory_name.value,
              'element': curr_traj.elem.value,
              'edge': curr_traj.edge.value,
              'e0': curr_traj.e0.value,
              'pulses_per_degree': hhm.pulses_per_deg,
}
        for flyer in flyers:
            if hasattr(flyer, 'offset'):
                md['{} offset'.format(flyer.name)] = flyer.offset.value
        md.update(**metadata)
        yield from bps.open_run(md=md)

        # TODO Replace this with actual status object logic.
        yield from bps.clear_checkpoint()
        yield from shutter.open_plan()
        #yield from xia1.start_trigger()
        # this must be a float
        yield from bps.abs_set(hhm.enable_loop, 0, wait=True)
        # this must be a string
        yield from bps.abs_set(hhm.start_trajectory, '1', wait=True)

        # this should be replaced by a status object
        def poll_the_traj_plan():
            while True:
                ret = (yield from bps.read(hhm.trajectory_running))
                if ret is None:
                    break
                is_running = ret['hhm_trajectory_running']['value']

                if is_running:
                    break
                else:
                    yield from bps.sleep(.1)

            while True:
                ret = (yield from bps.read(hhm.trajectory_running))
                if ret is None:
                    break
                is_running = ret['hhm_trajectory_running']['value']

                if is_running:
                    yield from bps.sleep(.05)
                else:
                    break

        yield from bpp.finalize_wrapper(poll_the_traj_plan(), 
                                       bpp.pchain(shutter.close_plan(), 
                                                 bps.abs_set(hhm.stop_trajectory, 
                                                            '1', wait=True)))
        #print('moving back')
        #hhm.prepare_trajectory.put('1')
        #print('should be done')
        yield from bps.close_run()

    def final_plan():
        yield from bps.abs_set(hhm.trajectory_running, 0, wait=True)
        #yield from xia1.stop_trigger()
        for flyer in flyers:
            yield from bps.unstage(flyer)
        yield from bps.unstage(hhm)


    for flyer in flyers:
        yield from bps.stage(flyer)

    yield from bps.stage(hhm)
    fly_plan = bpp.fly_during_wrapper(bpp.finalize_wrapper(inner(), final_plan()),
                                              flyers)
    # TODO : Add in when suspend_wrapper is avaialable
    #if not ignore_shutter:
        # this will use suspenders defined in 23-suspenders.py
        #fly_plan = bpp.suspend_wrapper(fly_plan, suspenders)

    yield from fly_plan


def execute_camera_trajectory(name, **metadata):
    #flyers = [pb4.di, pba2.adc7, pba1.adc6, pb9.enc1, pba1.adc1, pba2.adc6, pba1.adc7]
    flyers = [pb4.di, pba2.adc7, pba1.adc6, pb9.enc1, pba1.adc1, pba1.adc7]
    def inner():
        curr_traj = getattr(hhm, 'traj{:.0f}'.format(hhm.lut_number_rbv.value))

        ret = (yield from bps.read(bpm_ms1.tiff_filenumber))
        if ret is None:
            raise Exception("Cannot read bpm_ms1 settings")
        else:
            tiff_filenumber = ret[bpm_ms1.tiff_filenumber.name]['value']

        new_folder = '/GPFS/xf08id/bpm_cameras_images/{}-{}/'.format(name, tiff_filenumber)
        if not os.path.exists(new_folder):
            os.makedirs(new_folder)
            call(['setfacl', '-m', 'g:iss-staff:rwx', new_folder])
            call(['chmod', '777', new_folder])

        yield from bps.sleep(0.25)
        #yield from bps.abs_set(bpm_ms1.tiff_filepath, new_folder)
        bpm_ms1.tiff_filepath.put(new_folder)
        yield from bps.sleep(0.5)

        ret = (yield from bps.read(bpm_ms1.tiff_filepath))
        if ret is None:
            raise Exception("Cannot read bpm_ms1 settings")
        else:
            tiff_filepath = ret[bpm_ms1.tiff_filepath.name]['value']

        ret = (yield from bps.read(bpm_ms1.tiff_filename))
        if ret is None:
            raise Exception("Cannot read bpm_ms1 settings")
        else:
            tiff_filename = ret[bpm_ms1.tiff_filename.name]['value']

        ret = (yield from bps.read(bpm_ms1.tiff_filefmt))
        if ret is None:
            raise Exception("Cannot read bpm_ms1 settings")
        else:
            tiff_filefmt = ret[bpm_ms1.tiff_filefmt.name]['value']

        interp_fn = f"{ROOT_PATH}/{filepath}/{RE.md['year']}.{RE.md['cycle']}.{RE.md['PROPOSAL']}/{name}.txt"
        md = {'plan_args': {},
              'plan_name': 'execute_trajectory',
              'experiment': 'transmission',
              'name': name,
              'interp_filename': interp_fn,
              'images_dir': ''.join(chr(i) for i in tiff_filepath)[:-1],
              'images_name': ''.join(chr(i) for i in tiff_filename)[:-1],
              'images_name_fmt': ''.join(chr(i) for i in tiff_filefmt)[:-1],
              'first_image_number': tiff_filenumber,
              'angle_offset': str(hhm.angle_offset.value),
              'trajectory_name': hhm.trajectory_name.value,
              'element': curr_traj.elem.value,
              'edge': curr_traj.edge.value,
              'e0': curr_traj.e0.value}
        for flyer in flyers:
            if hasattr(flyer, 'offset'):
                md['{} offset'.format(flyer.name)] = flyer.offset.value
        md.update(**metadata)
        yield from bps.open_run(md=md)

        # TODO Replace this with actual status object logic.
        yield from bps.clear_checkpoint()
        yield from shutter.open_plan()
        yield from xia1.start_trigger()
        # this must be a float
        yield from bps.abs_set(hhm.enable_loop, 0, wait=True)
        # this must be a string
        yield from bps.abs_set(hhm.start_trajectory, '1', wait=True)

        # this should be replaced by a status object
        def poll_the_traj_plan():
            while True:
                ret = (yield from bps.read(hhm.trajectory_running))
                if ret is None:
                    break
                is_running = ret['hhm_trajectory_running']['value']

                if is_running:
                    break
                else:
                    yield from bps.sleep(.1)

            while True:
                ret = (yield from bps.read(hhm.trajectory_running))
                if ret is None:
                    break
                is_running = ret['hhm_trajectory_running']['value']

                if is_running:
                    yield from bps.sleep(.05)
                else:
                    break


        yield from bpp.finalize_wrapper(poll_the_traj_plan(), 
                                       bpp.pchain(shutter.close_plan(), 
                                                 bps.abs_set(hhm.stop_trajectory, 
                                                            '1', wait=True)))

        yield from bps.close_run()

    def final_plan():
        yield from bps.abs_set(hhm.trajectory_running, 0, wait=True)
        yield from xia1.stop_trigger()
        for flyer in flyers:
            yield from bps.unstage(flyer)
        yield from bps.unstage(hhm)

    for flyer in flyers:
        yield from bps.stage(flyer)

    yield from bps.stage(hhm)

    return (yield from bpp.fly_during_wrapper(bpp.finalize_wrapper(inner(), final_plan()),
                                              flyers))

def execute_xia_trajectory(name, **metadata):
    #flyers = [pba2.adc7, pba1.adc6, pb9.enc1, pba1.adc1, pba2.adc6, pba1.adc7, pb4.di]
    flyers = [pba2.adc7, pba1.adc6, pb9.enc1, pba1.adc1, pba1.adc7, pb4.di]
    def inner():
        # Setting the name of the file
        xia1.netcdf_filename.put(name)
        next_file_number = xia1.netcdf_filenumber_rb.value

        xia_rois = {}
        max_energy = xia1.mca_max_energy.value
        for mca in xia1.mcas:
            if mca.kind & Kind.normal:
                for roi_number in range(12):
                    #if not (eval('xia1.{}.roi{}.low.value'.format(mca, roi)) < 0 or eval('xia1.{}.roi{}.high.value'.format(mca, roi)) < 0):
                    roi = getattr(mca, f'roi{roi_number}')
                    if not roi.low.value < 0 or roi.high.value < 0:
                        #xia_rois[eval('xia1.{}.roi{}.high.name'.format(mca, roi))] = eval('xia1.{}.roi{}.high.value'.format(mca, roi)) * max_energy / 2048
                        #xia_rois[eval('xia1.{}.roi{}.low.name'.format(mca, roi))] = eval('xia1.{}.roi{}.low.value'.format(mca, roi)) * max_energy / 2048
                        xia_rois[roi.high.name] = roi.high.value * max_energy / 2048
                        xia_rois[roi.low.name] = roi.low.value * max_energy / 2048

        interp_fn = f"{ROOT_PATH}/{USER_FILEPATH}/{RE.md['year']}.{RE.md['cycle']}.{RE.md['PROPOSAL']}/{name}.txt"
        curr_traj = getattr(hhm, 'traj{:.0f}'.format(hhm.lut_number_rbv.value))
        md = {'plan_args': {}, 
              'plan_name': 'execute_xia_trajectory',
              'experiment': 'fluorescence_sdd', 
              'name': name,
              'interp_filename': interp_fn,
              'xia_max_energy': xia1.mca_max_energy.value,
              'xia_filename': '{}_{:03}.nc'.format(name, next_file_number), 
              'xia_rois':xia_rois, 
              'angle_offset': str(hhm.angle_offset.value),
              'trajectory_name': hhm.trajectory_name.value,
              'element': curr_traj.elem.value,
              'edge': curr_traj.edge.value,
              'e0': curr_traj.e0.value}
        for flyer in flyers:
            if hasattr(flyer, 'offset'):
                md['{} offset'.format(flyer.name)] = flyer.offset.value
        md.update(**metadata)
        yield from bps.open_run(md=md)

        # TODO Replace this with actual status object logic.
        yield from bps.clear_checkpoint()

        fname = ''.join(chr(i) for i in list(xia1.netcdf_filename_rb.value))
        fname = fname[0:len(fname) - 1]
        while(fname != name):
            fname = ''.join(chr(i) for i in list(xia1.netcdf_filename_rb.value))
            fname = name[0:len(fname) - 1]
            yield from bps.sleep(.05)
       
        yield from shutter.open_plan()
        yield from bps.sleep(.5)
        yield from xia1.start_mapping_scan()
        yield from bps.sleep(.5)

        # this must be a float
        yield from bps.abs_set(hhm.enable_loop, 0, wait=True)
        yield from bps.sleep(.5)
        # this must be a string
        yield from bps.abs_set(hhm.start_trajectory, "1", wait=True)
        yield from bps.sleep(.5)

        def poll_the_traj_plan():
            while True:
                ret = (yield from bps.read(hhm.trajectory_running))
                if ret is None:
                    break
                is_running = ret['hhm_trajectory_running']['value']

                if is_running:
                    break
                else:
                    yield from bps.sleep(.1)

            while True:
                ret = (yield from bps.read(hhm.trajectory_running))
                if ret is None:
                    break
                is_running = ret['hhm_trajectory_running']['value']

                if is_running:
                    yield from bps.sleep(.05)
                else:
                    break


        yield from bpp.finalize_wrapper(poll_the_traj_plan(), 
                                       bpp.pchain(xia1.stop_scan(),
                                                 shutter.close_plan(), 
                                                 bps.abs_set(hhm.stop_trajectory, 
                                                            '1', wait=True)))

        yield from bps.close_run()

    def final_plan():
        yield from bps.abs_set(hhm.trajectory_running, 0, wait=True)
#        yield from xia1.stop_scan()
        while xia1.capt_start_stop.value:
            pass
        print('Stopped XIA')
        for flyer in flyers:
            yield from bps.unstage(flyer)
        yield from bps.unstage(hhm)

    for flyer in flyers:
        yield from bps.stage(flyer)

    yield from bps.stage(hhm)

    return (yield from bpp.fly_during_wrapper(bpp.finalize_wrapper(inner(), final_plan()), flyers))


def execute_loop_trajectory(name, **metadata):

    flyers = [pba1.adc6, pb9.enc1, pba1.adc1, pba2.adc6, pba1.adc7]
    #flyers = [pba1.adc6, pb9.enc1, pba1.adc1, pba1.adc7]
    def inner():
        md = {'plan_args': {}, 'plan_name': 'execute_loop_trajectory','experiment': 'transmission', 'name': name, pba1.adc1.name + ' offset': pba1.adc1.offset.value, pba1.adc6.name + ' offset': pba1.adc6.offset.value, pba2.adc6.name + ' offset': pba2.adc6.offset.value, pba1.adc7.name + ' offset': pba1.adc7.offset.value, 'trajectory_name': hhm.trajectory_name.value}
        md.update(**metadata)
        yield from bps.open_run(md=md)

        # TODO Replace this with actual status object logic.
        yield from shutter.open_plan()
        yield from bps.abs_set(hhm.enable_loop, 1, wait=True)#hhm.enable_loop.put("1")
        yield from bps.abs_set(hhm.start_trajectory, "1", wait=True) # NOT SURE IF THIS LINE SHOULD BE HERE

        def poll_the_traj_plan():
            while True:
                ret = (yield from bps.read(hhm.trajectory_running))
                if ret is None:
                    break
                is_running = ret['hhm_trajectory_running']['value']

                if is_running:
                    break
                else:
                    yield from bps.sleep(.1)

            while True:
                ret = (yield from bps.read(hhm.trajectory_running))
                retloop = (yield from bps.read(hhm.enable_loop_rbv))
                if ret is None or retloop is None:
                    break
                loop_is_running = retloop['hhm_enable_loop_rbv']['value']

                if loop_is_running:
                    yield from bps.sleep(.05)
                else:
                    break

        yield from bpp.finalize_wrapper(poll_the_traj_plan(), 
                                       bpp.pchain(shutter.close_plan(), 
                                                 bps.abs_set(hhm.stop_trajectory, 
                                                            '1', wait=True), 
                                                 bps.abs_set(hhm.enable_loop, 
                                                            0, wait=True)))

        yield from bps.close_run()

    def final_plan():
        yield from bps.abs_set(hhm.trajectory_running, 0, wait=True)
        for flyer in flyers:
            yield from bps.unstage(flyer)
        yield from bps.unstage(hhm)

    for flyer in flyers:
        yield from bps.stage(flyer)

    yield from bps.stage(hhm)

    return (yield from bpp.fly_during_wrapper(bpp.finalize_wrapper(inner(), final_plan()),
                                              flyers))


def wait_filter_in_place(status_pv):
    #for j in range(5):
    while True:
        ret = yield from bps.read(status_pv)
        if ret is None:
            break
        if ret[status_pv.name]['value'] == 1:
            break
        else:
            yield from bps.sleep(.1)




def prepare_bl_plan(energy: int = -1, print_messages=True, debug=False,**kwargs):
    sys.stdout = kwargs.pop('stdout', sys.stdout)
    print('hello')
    if debug:
        print('[Prepare BL] Running Prepare Beamline in Debug Mode! (Not moving anything)')

    energy = int(energy)
    curr_energy = energy
    if energy < 0:
        ret = (yield from bps.read(hhm.energy))
        if ret is not None:
            curr_energy = ret[hhm.energy.name]['value']

    if print_messages:
        print('[Prepare BL] Setting up the beamline to {} eV'.format(curr_energy))



    curr_range = [ran for ran in prepare_bl_def[0] if
                  ran['energy_end'] > curr_energy >= ran['energy_start']]
    if not len(curr_range):
        print('Current energy is not valid. :( Aborted.')
        return


    curr_range = curr_range[0]
    pv_he = curr_range['pvs']['IC Gas He']['object']
    print(pv_he.tolerance)

    if print_messages:
        print('[Prepare BL] Setting HE to {}'.format(curr_range['pvs']['IC Gas He']['value']))

    if not debug:
        yield from bps.mv(pv_he, curr_range['pvs']['IC Gas He']['value'])


    pv_n2 = curr_range['pvs']['IC Gas N2']['object']
    if print_messages:
        print('[Prepare BL] Setting N2 to {}'.format(curr_range['pvs']['IC Gas N2']['value']))
    if not debug:
        yield from bps.mv(pv_n2, curr_range['pvs']['IC Gas N2']['value'])


    # If you go from less than 1000 V to more than 1400 V, you need a delay. 2 minutes
    # For now if you increase the voltage (any values), we will have the delay. 2 minutes

    pv_i0_volt = curr_range['pvs']['I0 Voltage']['object']
    ret = (yield from bps.read(pv_i0_volt))
    if ret is not None:
        old_i0 = ret[pv_i0_volt.name]['value']
    else:
        old_i0 = 0
    if print_messages:
        print('[Prepare BL] Old I0 Voltage: {} | New I0 Voltage: {}'.format(old_i0,
                                                                        curr_range['pvs']['I0 Voltage']['value']))

    pv_it_volt = curr_range['pvs']['It Voltage']['object']
    ret = (yield from bps.read(pv_it_volt))
    if ret is not None:
        old_it = ret[pv_it_volt.name]['value']
    else:
        old_it = 0
    if print_messages:
        print('[Prepare BL] Old It Voltage: {} | New It Voltage: {}'.format(old_it,
                                                                        curr_range['pvs']['It Voltage']['value']))

    pv_ir_volt = curr_range['pvs']['Ir Voltage']['object']
    ret = (yield from bps.read(pv_ir_volt))
    if ret is not None:
        old_ir = ret[pv_ir_volt.name]['value']
    else:
        old_ir = 0
    if print_messages:
        print('[Prepare BL] Old Ir Voltage: {} | New Ir Voltage: {}'.format(old_ir,
                                                                        curr_range['pvs']['Ir Voltage']['value']))

    # check if bpm_cm will move
    close_shutter = 0
    cm = [bpm for bpm in curr_range['pvs']['BPMs'] if bpm['name'] == bpm_cm.name][0]
    new_cm_value = cm['value']
    if new_cm_value == 'OUT':
        pv = cm['object'].switch_retract
    elif new_cm_value == 'IN':
        pv = cm['object'].switch_insert
    yield from bps.sleep(0.1)
    ret = (yield from bps.read(pv))
    if ret is not None:
        if ret[pv.name]['value'] == 0:
            close_shutter = 1

    # check if filtebox will move
    mv_fb = 0
    fb_value = prepare_bl_def[1]['FB Positions'][curr_range['pvs']['Filterbox Pos']['value'] - 1]
    pv_fb_motor = curr_range['pvs']['Filterbox Pos']['object']
    yield from bps.sleep(0.1)
    ret = (yield from bps.read(pv_fb_motor))
    if ret is not None:
        curr_fb_value = ret[pv_fb_motor.name]['value']
    else:
        curr_fb_value = -1

    if abs(fb_value - curr_fb_value) > 20 * (10 ** (-pv_fb_motor.precision)):
        close_shutter = 1
        mv_fb = 1

    def handler(signum, frame):
        print("[Prepare BL] Could not activate FE Shutter")
        raise Exception("Timeout")

    if close_shutter:
        if print_messages:
            print('[Prepare BL] Closing FE Shutter...')
        if not debug:
            signal.signal(signal.SIGALRM, handler)
            signal.alarm(12)
            try:
                yield from shutter_fe.close_plan()
            except Exception as exc:
                print('[Prepare BL] Timeout! Could not close FE Shutter. Aborting! (Try once again, maybe?)')
                return

            tries = 10
            ret = (yield from bps.read(shutter_fe.state))
            if ret is not None:
                while ret[shutter_fe.state.name]['value'] != 1:
                    yield from bps.sleep(0.1)
                    if tries:
                        yield from shutter_fe.close_plan()
                        tries -= 1
                    ret = (yield from bps.read(shutter_fe.state))

            signal.alarm(0)
        if print_messages:
            print('[Prepare BL] FE Shutter closed')

    yield from bps.sleep(0.1)
    fb_sts_pv = curr_range['pvs']['Filterbox Pos']['STS PVS'][curr_range['pvs']['Filterbox Pos']['value'] - 1]
    if mv_fb:
        if print_messages:
            print('[Prepare BL] Moving Filterbox to {}'.format(fb_value))
        if not debug:
            yield from bps.abs_set(pv_fb_motor, fb_value, group='prepare_bl')

    pv_hhrm_hor = curr_range['pvs']['HHRM Hor Trans']['object']
    yield from bps.sleep(0.1)
    if print_messages:
        print('[Prepare BL] Moving HHRM Horizontal to {}'.format(curr_range['pvs']['HHRM Hor Trans']['value']))
    if not debug:
        yield from bps.abs_set(pv_hhrm_hor, curr_range['pvs']['HHRM Hor Trans']['value'], group='prepare_bl')

    bpm_pvs = []
    for bpm in curr_range['pvs']['BPMs']:
        exposure_time = bpm['exposure_time']
        if bpm['value'] == 'IN':
            pv_set = bpm['object'].ins
            pv_read = bpm['object'].switch_insert
        elif bpm['value'] == 'OUT':
            pv_set = bpm['object'].ret
            pv_read = bpm['object'].switch_retract
        try:
            if pv:
                if print_messages:
                    print('[Prepare BL] Moving {} {}'.format(bpm['name'], bpm['value']))
                for i in range(3):
                    if not debug:
                        if exposure_time > 0:
                            yield from bps.abs_set(bpm['object'].exp_time, exposure_time)
                        yield from bps.abs_set(pv_set, 1)
                    yield from bps.sleep(0.1)
                bpm_pvs.append([pv_set, pv_read])
        except Exception as exp:
            print(exp)

    if close_shutter:
        yield from wait_filter_in_place(fb_sts_pv)
        #while fb_sts_pv.value != 1:
        #    pass
        if print_messages:
            print('[Prepare BL] Opening shutter...')
        if not debug:
            signal.signal(signal.SIGALRM, handler)
            signal.alarm(12)
            try:
                yield from shutter_fe.open_plan()
            except Exception as exc:
                print('[Prepare BL] Timeout! Could not open FE Shutter. Aborting! (Try once again, maybe?)')
                return

            tries = 10
            ret = (yield from bps.read(shutter_fe.state))
            if ret is not None:
                while ret[shutter_fe.state.name]['value'] != 0:
                    yield from bps.sleep(0.1)
                    if tries:
                        yield from shutter_fe.open_plan()
                        tries -= 1
                    ret = (yield from bps.read(shutter_fe.state))

            signal.alarm(0)
        if print_messages:
            print('[Prepare BL] FE Shutter open')

    if curr_range['pvs']['I0 Voltage']['value'] - old_i0 > 2 or \
            curr_range['pvs']['It Voltage']['value'] - old_it > 2 or \
            curr_range['pvs']['Ir Voltage']['value'] - old_ir > 2:
        old_time = ttime.time()
        wait_time = 120
        if print_messages:
            print('[Prepare BL] Waiting for gas ({}s)...'.format(wait_time))
        percentage = 0
        if not debug:
            while ttime.time() - old_time < wait_time:  # 120 seconds
                if ttime.time() - old_time >= percentage * wait_time:
                    print(
                        '[Prepare BL] {:3}% ({:.1f}s)'.format(int(np.round(percentage * 100)), percentage * wait_time))
                    percentage += 0.1
                yield from bps.sleep(0.1)
        print('[Prepare BL] 100% ({:.1f}s)'.format(wait_time))
        if print_messages:
            print('[Prepare BL] Done waiting for gas...')

    if print_messages:
        print('[Prepare BL] Setting i0 {}'.format(curr_range['pvs']['I0 Voltage']['value']))
        print('[Prepare BL] Setting it {}'.format(curr_range['pvs']['It Voltage']['value']))
        print('[Prepare BL] Setting ir {}'.format(curr_range['pvs']['Ir Voltage']['value']))
    if not debug:
        pv_i0_volt._put_complete = True
        pv_it_volt._put_complete = True
        pv_ir_volt._put_complete = True
        yield from bps.abs_set(pv_i0_volt, curr_range['pvs']['I0 Voltage']['value'])#, group='prepare_bl')
        yield from bps.abs_set(pv_it_volt, curr_range['pvs']['It Voltage']['value'])#, group='prepare_bl')
        yield from bps.abs_set(pv_ir_volt, curr_range['pvs']['Ir Voltage']['value'])#, group='prepare_bl')

    yield from bps.sleep(0.1)

    if print_messages:
        print('[Prepare BL] Waiting for everything to be in position...')
    if not debug:
        while abs(abs(pv_i0_volt.value) - abs(
                curr_range['pvs']['I0 Voltage']['value'])) > (10 ** -pv_i0_volt.precision) * 2000 or abs(
                        abs(pv_it_volt.value) - abs(
                        curr_range['pvs']['It Voltage']['value'])) > (10 ** -pv_it_volt.precision) * 2000 or abs(
                        abs(pv_ir_volt.value) - abs(
                        curr_range['pvs']['Ir Voltage']['value'])) > (10 ** -pv_ir_volt.precision) * 2000:
            yield from bps.sleep(0.1)
        yield from bps.wait(group='prepare_bl')
    if print_messages:
        print('[Prepare BL] Everything seems to be in position')
        print('[Prepare BL] Setting energy to {}'.format(curr_energy))

    if not debug:
        yield from bps.abs_set(hhm.energy, curr_energy, wait=True)

    if print_messages:
        print('[Prepare BL] Beamline preparation complete!')

def sleep_plan(sleep_time, **metadata):
    return (yield from bps.sleep(float(sleep_time)))


lut_offsets = {
               'i0':{
                     'ln':{
                           '2': -0.06645569368421053,
                           '3': -0.06640107894736844,
                           '4': -0.06637296842105264,
                           '5': -0.06638662210526317,
                           '6': -0.06640549631578947,
                           '7': -0.06617458842105263
                          },
                     'hs':{
                           '3': -0.04026150210526316,
                           '4': -0.03990208894736842,
                           '5': -0.03964066105263159,
                           '6': -0.03969447263157895,
                           '7': -0.03963704684210526,
                           '8': -0.035940111052631576
                          }
                    },
               'it':{
                     'ln':{
                           '2': -0.08072941578947367,
                           '3': -0.0821305247368421,
                           '4': -0.08204257894736844,
                           '5': -0.08158638526315788,
                           '6': -0.07702966894736843,
                           '7': -0.031005107368421055
                          },
                     'hs':{
                           '3': -0.03974386684210526,
                           '4': -0.052516085263157895,
                           '5': -0.051755896315789474,
                           '6': -0.047226487368421055,
                           '7': -0.0021817784210526312,
                           '8': 0.4464437489473684
                          }
                    },
               'iff':{
                     'ln':{
                           '2': -0.09427186263157893,
                           '3': -0.09435097368421054,
                           '4': -0.09431081578947369,
                           '5': -0.09429394947368422,
                           '6': -0.09429595736842106,
                           '7': -0.09408191578947321
                          },
                     'hs':{
                           '3': -0.06404622,
                           '4': -0.0648802994736842,
                           '5': -0.0645983910526316,
                           '6': -0.06455662684210525,
                           '7': -0.06442410578947368,
                           '8': -0.06138334999999999
                          }
                    },
               'ir':{
                     'ln':{
                           '2': -0.03500884947368421,
                           '3': -0.032811811052631576,
                           '4': -0.03258491894736841,
                           '5': -0.03255158789473684,
                           '6': -0.03523774947368421,
                           '7': -0.13698379947368422
                          },
                     'hs':{
                           '3': -0.03492451789473684,
                           '4': -0.03301139578947368,
                           '5': -0.032817433157894725,
                           '6': -0.03279735421052631,
                           '7': -0.035501988421052635,
                           '8': -0.13801103842105264
                          }
                    }
              }


def set_gains_and_offsets_plan(*args):

    mod = len(args) % 3
    if mod:
        args = args[:-mod]

    for ic, val, hs in zip([ic for index, ic in enumerate(args) if index % 3 == 0], 
                       [val for index, val in enumerate(args) if index % 3 == 1], 
                       [hs for index, hs in enumerate(args) if index % 3 == 2]):
        yield from ic.set_gain_plan(val, hs)

        if type(ic) != ICAmplifier:
            raise Exception('Wrong type: {} - it should be ICAmplifier'.format(type(ic)))
        if type(val) != int:
            raise Exception('Wrong type: {} - it should be int'.format(type(val)))
        if type(hs) != bool:
            raise Exception('Wrong type: {} - it should be bool'.format(type(hs)))

        if hs:
           hs_str = 'hs'
        else:
           hs_str = 'ln'
        yield from bps.mv(ic.par.offset, lut_offsets[ic.par.dev_name.value][hs_str][str(val)])
