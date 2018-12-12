import inspect
import bluesky.plans as bp
import os, sys

def tscan(name:str, comment:str, n_cycles:int=1, delay:float=0, **kwargs):
    """
    Trajectory Scan - Runs the monochromator along the trajectory that is previously loaded in the controller N times

    Parameters
    ----------
    name : str
        Name of the scan - it will be stored in the metadata

    n_cycles : int (default = 1)
        Number of times to run the scan automatically

    delay : float (default = 0)
        Delay in seconds between scans


    Returns
    -------
    uid : list(str)
        Lists containing the unique ids of the scans


    See Also
    --------
    :func:`tscanxia`
    """
    sys.stdout = kwargs.pop('stdout', sys.stdout)

    #uids = []
    RE.is_aborted = False
    for indx in range(int(n_cycles)): 
        if RE.is_aborted:
            return 'Aborted'
        if n_cycles == 1:
            name_n = name
        else:
            name_n = name + ' ' + str(indx + 1)
        print('Current step: {} / {}'.format(indx + 1, n_cycles))
        RE(prep_traj_plan())
        uid = RE(execute_trajectory(name_n, comment=comment))
        yield uid
        #hhm.prepare_trajectory.put('1')
        #uids.append(uid)
        print(f'Waiting for {delay} s ')
        RE(bps.sleep(float(delay)))
    print('Scan is complete!')
    #return uids
    

def tscan_plan(name:str, comment:str="", prepare_traj:bool=True, n_cycles:int=1, delay:float=0, **kwargs):
    sys.stdout = kwargs.pop('stdout', sys.stdout)
    uids = []
    for indx in range(int(n_cycles)): 
        name_n = name + ' ' + str(indx + 1)
        print(name_n) 
        if prepare_traj:
            yield from prep_traj_plan()
        #uid = (yield from execute_trajectory(name_n))
        uid = (yield from execute_trajectory(name_n, comment=comment))
        # uid = db[-1]['start']['uid']
        uids.append(uid)
			
        yield from bps.sleep(float(delay))
    print('Scan is complete!')
    return uids


def tscanxia(name:str, comment:str, n_cycles:int=1, delay:float=0, **kwargs):
    """
    Trajectory Scan XIA - Runs the monochromator along the trajectory that is previously loaded in the controller and get data from the XIA N times

    Parameters
    ----------
    name : str
        Name of the scan - it will be stored in the metadata

    n_cycles : int (default = 1)
        Number of times to run the scan automatically

    delay : float (default = 0)
        Delay in seconds between scans


    Returns
    -------
    uid : list(str)
        Lists containing the unique ids of the scans


    See Also
    --------
    :func:`tscan`
    """
    sys.stdout = kwargs.pop('stdout', sys.stdout)

    #uids = []
    RE.is_aborted = False
    for i in range(int(n_cycles)):
        if RE.is_aborted:
            return 'Aborted'
        if n_cycles == 1:
            name_n = name
        else:
            name_n = name + ' ' + str(i + 1)
        print('Current step: {} / {}'.format(i + 1, n_cycles))
        RE(prep_traj_plan())
        uid, = RE(execute_xia_trajectory(name_n, comment=comment))
        yield uid
        #uids.append(uid)
        time.sleep(float(delay))
    print('Done!')
    #return uids


def tscancam(name:str, comment:str, n_cycles:int=1, delay:float=0, **kwargs):
    """
    Trajectory Scan - Runs the monochromator along the trajectory that is previously loaded in the controller N times

    Parameters
    ----------
    name : str
        Name of the scan - it will be stored in the metadata

    n_cycles : int (default = 1)
        Number of times to run the scan automatically

    delay : float (default = 0)
        Delay in seconds between scans


    Returns
    -------
    uid : list(str)
        Lists containing the unique ids of the scans


    See Also
    --------
    :func:`tscanxia`
    """
    sys.stdout = kwargs.pop('stdout', sys.stdout)

    #uids = []
    RE.is_aborted = False
    for indx in range(int(n_cycles)): 
        if RE.is_aborted:
            return 'Aborted'
        if n_cycles == 1:
            name_n = name
        else:
            name_n = name + ' ' + str(indx + 1)
        print('Current step: {} / {}'.format(indx + 1, n_cycles))
        RE(prep_traj_plan())
        uid, = RE(execute_camera_trajectory(name_n, comment=comment))
        yield uid
        #uids.append(uid)
        time.sleep(float(delay))
    print('Done!')
    #return uids


def get_offsets(num:int = 20, *args, **kwargs):
    """
    Get Ion Chambers Offsets - Gets the offsets from the ion chambers and automatically subtracts from the acquired data in the next scans

    Parameters
    ----------
    num : int
        Number of points to acquire and average for each ion chamber


    Returns
    -------
    uid : list(str)
        List containing the unique id of the scan


    See Also
    --------
    :func:`tscan`
    """
    sys.stdout = kwargs.pop('stdout', sys.stdout)

    adcs = list(args)
    if not len(adcs):
        adcs = [pba2.adc7, pba1.adc7, pba2.adc6, pba1.adc1, pba1.adc6]

    old_avers = []
    for adc in adcs:
        old_avers.append(adc.averaging_points.get())
        adc.averaging_points.put(15)
    
    uid, = RE(get_offsets_plan(adcs, num = int(num)))

    if 'dummy_read' not in kwargs:
        print('Updating values...')

    arrays = []
    offsets = []
    df = db.get_table(db[-1])
    for index, adc in enumerate(adcs):
        key = '{}_volt'.format(adc.name)
        array = df[key]
        offset = np.mean(df[key][2:int(num)])

        arrays.append(array)
        offsets.append(offset)
        if 'dummy_read' not in kwargs:
            adc.offset.put(offset)
            print('{}\nMean ({}) = {}'.format(array, adc.dev_name.value, offset))
        adc.averaging_points.put(old_avers[index])
    
    run = db[uid]
    for i in run['descriptors']:
        if i['name'] != 'primary':
            os.remove(i['data_keys'][i['name']]['filename'])

    stdout = kwargs.pop('stdout', sys.stdout)

    if 'stdout' not in kwargs:
        print("Warning, stdout not passed. GUI should be passing this")

    if 'dummy_read' in kwargs:
        print_message = ''
        for index, adc in enumerate(adcs):
            print('Mean ({}) = {}'.format(adc.dev_name.value, offsets[index]))

            saturation = adc.dev_saturation.value

            if adc.polarity == 'neg':
                if offsets[index] > saturation/100:
                    print_message += 'Increase {} gain by 10^2\n'.format(adc.dev_name.value)
                elif offsets[index] <= saturation/100 and offsets[index] > saturation/10:
                    print_message += 'Increase {} gain by 10^1\n'.format(adc.dev_name.value)
        print('-' * 30, file=stdout)
        print(print_message[:-1], file=stdout)
        print('-' * 30, file=stdout)

    print(uid)
    print('Done!')
    yield uid

def general_scan(detectors, num_name, den_name, result_name, motor, rel_start, rel_stop, num, find_min_max, retries, **kwargs):
    sys.stdout = kwargs.pop('stdout', sys.stdout)
    for index, detector in enumerate(detectors):
        if type(detector) == str:
            detectors[index] = eval(detector)

    if type(motor) == str:
        motor = eval(motor)

    print('[General Scan] Starting scan...')
    ax = kwargs.get('ax')

    if find_min_max:
        over = 0
        while(not over):
            uid, = RE(general_scan_plan(detectors, motor, rel_start, rel_stop, int(num)), NormPlot(num_name, den_name, result_name, result_name, motor.name, ax=ax))
            yield uid
            last_table = db.get_table(db[-1])
            if detectors[0].polarity == 'pos':
                index = np.argmax(last_table[num_name])
            else:
                index = np.argmin(last_table[num_name])
            motor.move(last_table[motor.name][index])
            print('[General Scan] New {} position: {}'.format(motor.name, motor.position))
            if (num >= 10):
                if (((index > 0.2 * num) and (index < 0.8 * num)) or retries == 1):
                    over = 1
                if retries > 1:
                    retries -= 1
            else:
                over = 1
        print('[General Scan] {} tuning complete!'.format(motor.name))
    else:
        uid, = RE(general_scan_plan(detectors, motor, rel_start, rel_stop, int(num)), NormPlot(num_name, den_name, result_name, result_name, motor.name, ax=ax))
        yield uid
        print('[General Scan] Done!')


def xia_step_scan(name:str, comment:str, e0:int=8333, preedge_start:int=-200, xanes_start:int=-50, xanes_end:int=30, exafs_end:int=16, preedge_spacing:float=10, xanes_spacing:float=0.2, exafs_spacing:float=0.04, **kwargs):
    '''
    xia_step_scan - Runs the monochromator along the trajectory defined in the parameters. Gets data from the XIA and the ion chambers after each step. 

    Parameters
    ----------
    name : str
        Name of the scan - it will be stored in the metadata
        Other parameters: TODO


    Returns
    -------
    uid : str
        Unique id of the scan

    interp_filename : str
    Filename where the interpolated data was stored


    See Also
    --------
    :func:`tscan`

    '''
    sys.stdout = kwargs.pop('stdout', sys.stdout)

    energy_grid, time_grid = get_xia_energy_grid(e0, preedge_start, xanes_start, xanes_end, exafs_end, preedge_spacing, xanes_spacing, exafs_spacing)
    positions_grid = xray.energy2encoder(energy_grid) / 360000

    ax = kwargs.get('ax')
    if ax is not None:
        uid, = RE(step_list_plan([xia1, i0, it, iff, ir], hhm.theta, positions_grid, name), LivePlot(xia1.mca1.roi0.sum.name, hhm.theta.name, ax=ax))
    else:
        uid, = RE(step_list_plan([xia1, i0, it, iff, ir], hhm.theta, positions_grid, name))

    path = '/GPFS/xf08id/User Data/{}.{}.{}/'.format(db[uid]['start']['year'], db[uid]['start']['cycle'], db[uid]['start']['PROPOSAL'])
    filename = parse_xia_step_scan(uid, name, path)

    ax.cla()
    plot_xia_step_scan(uid, ax=ax) 

    print('Done!')
    return uid

    

def samplexy_scan(detectors, motor, rel_start, rel_stop, num, **kwargs):
    sys.stdout = kwargs.pop('stdout', sys.stdout)
    if type(detectors) is not list:
        detectors = [detectors]
    return RE(sampleXY_plan(detectors, motor, rel_start, rel_stop, int(num)), LivePlot(detectors[0].volt.name, motor.name))


def sleep_seconds(secs:float=1, **kwargs):
    sys.stdout = kwargs.pop('stdout', sys.stdout)
    RE(sleep_plan(secs))
    yield None


def set_gains_and_offsets(i0_gain:int=5, it_gain:int=5, iff_gain:int=6,
                          ir_gain:int=5, hs:bool=False):
    sys.stdout = kwargs.pop('stdout', sys.stdout)
    i0_gain = int(i0_gain)
    it_gain = int(it_gain)
    iff_gain = int(iff_gain)
    ir_gain = int(ir_gain)
    if type(hs) == str:
        hs = hs == 'True'

    RE(set_gains_and_offsets_plan(i0_amp, i0_gain, hs, it_amp, it_gain, hs, iff_amp, iff_gain, hs, ir_amp, ir_gain, hs))


def xymove_repeat(numrepeat=1, xyposlist=[], samplelist=[], sleeptime = 2, testing = False, simulation = True, runnum_start = 0, usexia = True, **kwargs):

    '''
    collect EXAFS scans on given sample locations repeatedly.

    variables:
    numrepeat (int): number of repeated runs
    xyposlist (list of of [x,y] positions in pairs, x, y are floats): \
                     list of [x, y] positions, e.g. [[2.3, 23], [34.5, 23.2]]
    samplelist (list of strings) sample/file names for each scans
    sleeptime (int): how long (in sec.) to wait between samples
    testing (bool): if True, no sample motion, no EXAFS scans but just print the process
    simulation (bool): if True, only sample motions, no EXAFS scans

    each scan file name will be: [samplename]_[current-runnum+runnum_start].txt

    '''
    sys.stdout = kwargs.pop('stdout', sys.stdout)
    if len(xyposlist) < 1:
        print('xyposlist is empty')
        raise
    if len(samplelist) < 1:
        print('samplelist is empty')
        raise

    if len(xyposlist) is not len(samplelist):
        print('xypolist and samplelist must have the same length')
        raise

    gen_parser = xasdata.XASdataGeneric(db)
    uids = []
    for runnum in range(numrepeat):
        print('current run', runnum)
        print('current run + run start', runnum+runnum_start)
        for i in range(len(xyposlist)):
            print('moving sample xy to', xyposlist[i])
            if testing is not True:
                samplexy.x.move(xyposlist[i][0])
                samplexy.y.move(xyposlist[i][1])

            print('done moving, taking a nap with wait time (s)', sleeptime)
            time.sleep(sleeptime)

            print('done napping, starting taking the scan')
            if testing is not True:
                if simulation is not True:
                    tscan_name = samplelist[i]+'_'+str(runnum+runnum_start).zfill(3)
                    if usexia is False:
                        uid = tscan(tscan_name)[0]
                    else:
                        uid = tscanxia(tscan_name)[0]

            print('done taking the current scan')

            print('parsing the current scan')
            current_filepath = '/GPFS/xf08id/User Data/{}.{}.{}/' \
                               '{}.txt'.format(db[uid]['start']['year'],
                                               db[uid]['start']['cycle'],
                                               db[uid]['start']['PROPOSAL'],
                                               db[uid]['start']['name'])
    
            gen_parser.load(uid)
            key_base = 'i0'
            if 'xia_filename' in db[uid]['start']:
                 key_base = 'xia_trigger'
            gen_parser.interpolate(key_base = key_base)
    
            if 'xia_filename' in db[uid]['start']:
                # Parse xia
                xia_filename = db[uid]['start']['xia_filename']
                xia_filepath = 'smb://elistavitski-ni/epics/{}'.format(xia_filename)
                xia_destfilepath = '/GPFS/xf08id/xia_files/{}'.format(xia_filename)
                smbclient = xiaparser.smbclient(xia_filepath, xia_destfilepath)
                smbclient.copy()
                xia_parser.parse(xia_filename, '/GPFS/xf08id/xia_files/')
                xia_parsed_filepath = current_filepath[0 : current_filepath.rfind('/') + 1]
                xia_parser.export_files(dest_filepath = xia_parsed_filepath, all_in_one = True)
    
                length = min(len(xia_parser.exporting_array1), len(gen_parser.interp_arrays['energy']))
    
                mcas = []
                if 'xia_rois' in db[uid]['start']:
                    xia_rois = db[uid]['start']['xia_rois']
                    for mca_number in range(1, 5):
                        mcas.append(xia_parser.parse_roi(range(0, length), mca_number, xia_rois['xia1_mca{}_roi0_low'.format(mca_number)], xia_rois['xia1_mca{}_roi0_high'.format(mca_number)]))
                    mca_sum = sum(mcas)
                else:
                    for mca_number in range(1, 5):
                        mcas.append(xia_parser.parse_roi(range(0, length), mca_number, 6.7, 6.9))
                    mca_sum = sum(mcas)
    
                gen_parser.interp_arrays['XIA_SUM'] = np.array([gen_parser.interp_arrays['energy'][:, 0], mca_sum]).transpose()
    
                gen_parser.export_trace(current_filepath[:-4], '')

        print('done with the current run')

    print('done with all the runs! congratulations!')

