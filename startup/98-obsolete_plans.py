print(ttime.ctime() + ' >>>> ' + __file__)
'''from xas.image_analysis import analyze_spiral_scan


def optimize_sample_plan(*args, **kwargs):
    # sys.stdout = kwargs.pop('stdout', sys.stdout)
    sample_x_nominal = kwargs['sample_x']
    sample_y_nominal = kwargs['sample_y']
    edge_energy = kwargs['energy']
    sample_name = kwargs['name']
    # print('moving giantxy to the nominal position')
    # yield from bps.mv(giantxy.x, sample_x_nominal) # move to nominal position
    # yield from bps.mv(giantxy.y, sample_y_nominal) # move to nominal position
    # print('adjusting gains')
    # yield from adjust_ic_gains(**kwargs)
    # yield from bps.sleep(0.2)
    # print('measuring offsets')
    # yield from get_offsets(*args, **kwargs)
    # yield from bps.sleep(0.2)
    # print('moving energy above the edge')
    # yield from bps.mv(hhm.energy, edge_energy + 100)  # move energy above the edge
    # # prelim spiral scan:
    # spiral_plan = general_spiral_scan([apb_ave],
    #                                   motor1=giantxy.x, motor2=giantxy.y,
    #                                   motor1_range=15, motor2_range=15,
    #                                   motor1_nsteps=15, motor2_nsteps=15,
    #                                   time_step=0.1)
    # print('performing spiral scan to find optimal position on the sample')
    # uid = (yield from spiral_plan)
    #
    conc = kwargs['concentration']
    # image_path = f"{ROOT_PATH}/{USER_FILEPATH}/{RE.md['year']}/{RE.md['cycle']}/{RE.md['proposal']}/{sample_name}_raster_scan.png"
    # print('analyzing spiral scan data and saving the image for the reference')
    # x, y = analyze_spiral_scan(db, uid, conc, None, image_path)
    x, y = analyze_spiral_scan(db, 'bbd9f23f-011e-40eb-b798-eb2a2ad5cfa8', conc, None, None)
    print(f'moving giantxy to the optimal postion ({x}, {y})')
    yield from bps.mv(giantxy.x, x)
    yield from bps.mv(giantxy.y, y)
    print('adjusting gains (final)')
    yield from adjust_ic_gains(**kwargs)
    print('measuring offsets (final)')
    yield from get_offsets(*args, **kwargs)



'''


'''from xas.file_io import validate_file_exists
import time as ttime
from datetime import datetime
from ophyd.status import SubscriptionStatus



class FlyerHHMwithTrigger(FlyerHHM):

    def __init__(self, det, pbs, motor, trigger): #, xs_det):
        super().__init__( det, pbs, motor)
        self.trigger = trigger

    def kickoff(self, traj_duration=None):
        self.trigger.stage()
        st_super = super().kickoff(traj_duration=traj_duration)
        return st_super

    def complete(self):
        st_super = super().complete()
        def callback_motor():
            self.trigger.complete()

        self._motor_status.add_callback(callback_motor)
        return st_super & self._motor_status #& st_xs

    def describe_collect(self):
        dict_super = super().describe_collect()
        dict_trig = self.trigger.describe_collect()
        return {**dict_super, **dict_trig}#, **dict_xs}

    def collect_asset_docs(self):
        yield from super().collect_asset_docs()
        yield from self.trigger.collect_asset_docs()

    def collect(self):
        yield from super().collect()
        yield from self.trigger.collect()
        self.trigger.unstage()


# flyer_apb_trigger = FlyerAPBwithTrigger(det=apb_stream, pbs=[pb9.enc1], motor=hhm, trigger=apb_trigger)



class FlyerXS(FlyerHHMwithTrigger):

    def __init__(self, det, pbs, motor, trigger, xs_det):
        super().__init__( det, pbs, motor, trigger)
        self.xs_det = xs_det

    def kickoff(self, traj_duration=None):
        traj_duration = trajectory_manager.current_trajectory_duration
        acq_rate = self.trigger.freq.get()
        self.xs_det.stage(acq_rate, traj_duration)
        st_super = super().kickoff(traj_duration=traj_duration)
        return st_super

    def complete(self):
        st_super = super().complete()
        def callback_xs(value, old_value, **kwargs):
            if int(round(old_value)) == 1 and int(round(value)) == 0:
                self.xs_det.complete()
                return True
            else:
                return False

        saving_st = SubscriptionStatus(self.xs_det.hdf5.capture, callback_xs)
        return st_super & saving_st

    def describe_collect(self):
        print(f'{ttime.ctime()} Plan describe_collect is starting...')
        dict_super = super().describe_collect()
        dict_xs = self.xs_det.describe_collect()
        print(f'{ttime.ctime()} Plan describe_collect is complete')
        return {**dict_super, **dict_xs}

    def collect_asset_docs(self):
        print(f'{ttime.ctime()} Plan collect_asset_docs is starting...')
        yield from super().collect_asset_docs()
        yield from self.xs_det.collect_asset_docs()
        print(f'{ttime.ctime()} Plan collect_asset_docs is complete')

    def collect(self):
        print(f'{ttime.ctime()} Plan collect is starting...')
        yield from super().collect()
        yield from self.xs_det.collect()
        self.xs_det.unstage()
        print(f'{ttime.ctime()} Plan collect is complete')


# flyer_xs = FlyerXS(det=apb_stream, pbs=[pb9.enc1], motor=hhm, trigger=apb_trigger, xs_det=xs_stream)


class FlyerPilatus(FlyerHHMwithTrigger):

    def __init__(self, det, pbs, motor, trigger, pil_det):
        super().__init__( det, pbs, motor, trigger)
        self.pil_det = pil_det

    def kickoff(self, traj_duration=None):
        print(f'     !!!!! {datetime.now()} PIL100K KICKOFF')
        traj_duration = trajectory_manager.current_trajectory_duration
        acq_rate = self.trigger.freq.get()
        self.pil_det.stage(acq_rate, traj_duration)

        st_pil = self.pil_det.trigger()
        st_super = super().kickoff(traj_duration=traj_duration)
        ttime.sleep(0.1)
        return st_super & st_pil

    def complete(self):
        print(f'     !!!!! {datetime.now()} PIL100K COMPLETE')
        shutter._close_direct()
        st_super = super().complete()
        # def callback_pil(value, old_value, **kwargs):
        #     print(f'     !!!!! {datetime.now()} callback_pil100k_capture {value} --> {old_value}')
        #     if int(round(old_value)) == 1 and int(round(value)) == 0:
        #         print(f'     !!!!! {datetime.now()} callback_pil100k_capture')
        #         self.pil_det.complete()
        #         return True
        #     else:
        #         return False
        # saving_st = SubscriptionStatus(self.pil_det.tiff.capture, callback_pil)
        self.pil_det.complete()
        return st_super# & saving_st

    def describe_collect(self):
        print(f'     !!!!! {datetime.now()} PIL100K DESCRIBE COLLECT')
        dict_super = super().describe_collect()
        dict_pil = self.pil_det.describe_collect()
        return {**dict_super, **dict_pil}

    def collect_asset_docs(self):
        print(f'     !!!!! {datetime.now()} PIL100K COLLECT ASSET DOCS')
        yield from super().collect_asset_docs()
        yield from self.pil_det.collect_asset_docs()

    def collect(self):
        print(f'     !!!!! {datetime.now()} PIL100K COLLECT')
        self.pil_det.unstage()
        yield from super().collect()
        yield from self.pil_det.collect()


# flyer_pil = FlyerPilatus(det=apb_stream, pbs=[pb9.enc1], motor=hhm, trigger=apb_trigger_pil100k, pil_det=pil100k_stream)
# flyer_pil = FlyerPilatus(det=apb_stream, pbs=[pb9.enc1], motor=hhm, trigger=apb_trigger_pil100k, pil_det=pil100k_stream)

### general flyer in development

class FlyerPilatus_:
    def __init__(self, det, pil_det, pbs, motor, trigger, shutter):
        self.name = f'{det.name}-{"-".join([pb.name for pb in pbs])}-flyer'
        self.parent = None
        self.det = det
        self.pil_det = pil_det
        self.pbs = pbs  # a list of passed pizza-boxes
        self.motor = motor
        self._motor_status = None
        self.trigger = trigger
        self.shutter = shutter

    def kickoff(self, traj_duration=None, *args, **kwargs):
        # self.det.trig_source.set(1).wait()
        # TODO: handle it on the plan level
        # self.motor.set('prepare').wait()

        def callback(value, old_value, **kwargs):

            if int(round(old_value)) == 0 and int(round(value)) == 1:
                # Now start mono move
                self._motor_status = self.motor.set('start')
                return True
            else:
                return False

        # print(f'     !!!!! {datetime.now()} Flyer kickoff is complete at')

        streaming_st = SubscriptionStatus(self.det.streaming, callback)

        if traj_duration is None:
            traj_duration = get_traj_duration()
        acq_rate = self.trigger.freq.get()

        self.det.stage(traj_duration)
        self.pil_det.stage(acq_rate, traj_duration)
        st_pil = self.pil_det.trigger()

        # Start apb after encoder pizza-boxes, which will trigger the motor.
        self.det.stream.set(1)

        ttime.sleep(1)
        self.trigger.stage()

        for pb in self.pbs:
            pb.stage()
            pb.kickoff()

        return streaming_st & st_pil

    def complete(self):
        def callback_motor():
            # When motor arrives to the position, it should stop streaming on
            # the detector. That will run 'callback_det' defined below, which
            # will perform the 'complete' step for all involved detectors.
            self.det.stream.put(0)
        self._motor_status.add_callback(callback_motor)

        def callback_det(value, old_value, **kwargs):
            if int(round(old_value)) == 1 and int(round(value)) == 0:
                if self.shutter.state == 'open':
                    yield from self.shutter.close_plan()

                self.det.complete()
                for pb in self.pbs:
                    pb.complete()
                return True
            else:
                return False

        def callback_trigger():
            self.trigger.complete()

        self._motor_status.add_callback(callback_trigger)

        self.pil_det.complete()
        streaming_st = SubscriptionStatus(self.det.streaming, callback_det)

        return self._motor_status & streaming_st

    def describe_collect(self):
        return_dict = self.det.describe_collect()
        # Also do it for all pizza-boxes
        for pb in self.pbs:
            return_dict[pb.name] = pb.describe_collect()[pb.name]

        dict_trig = self.trigger.describe_collect()
        dict_pil = self.pil_det.describe_collect()

        return {**return_dict, **dict_trig, **dict_pil}

    def collect(self):
        def collect_and_unstage_all():
            for pb in self.pbs:
                yield from pb.collect()
            yield from self.det.collect()

            yield from self.trigger.collect()
            yield from self.pil_det.collect()

            # The .unstage() method resets self._datum_counter, which is needed
            # by .collect(), so calling .unstage() afteer .collect().
            self.det.unstage()
            for pb in self.pbs:
                pb.unstage()
            self.trigger.unstage()
            self.pil_det.unstage()

        return (yield from collect_and_unstage_all())

    def collect_asset_docs(self):
        yield from self.det.collect_asset_docs()
        for pb in self.pbs:
            yield from pb.collect_asset_docs()
        yield from self.trigger.collect_asset_docs()
        yield from self.pil_det.collect_asset_docs()


    # def stop(self,*args, **kwargs):
    #     print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>.. AT STOP ')

# flyer_pil = FlyerPilatus_(det=apb_stream, pbs=[pb9.enc1], motor=hhm, trigger=apb_trigger_pil100k, pil_det=pil100k_hdf5_stream, shutter=shutter)



###




def execute_trajectory_apb_trigger(name, **metadata):
    md = get_md_for_scan(name,
                         'fly_scan',
                         'execute_trajectory_apb_trigger',
                         'fly_energy_scan_apb_trigger',
                         **metadata)
    yield from bp.fly([flyer_apb_trigger], md=md)


import bluesky.plan_stubs as bps
def fly_local(flyers, *, md=None):
    """
    Perform a fly scan with one or more 'flyers'.
    Parameters
    ----------
    flyers : collection
        objects that support the flyer interface
    md : dict, optional
        metadata
    Yields
    ------
    msg : Msg
        'kickoff', 'wait', 'complete, 'wait', 'collect' messages
    See Also
    --------
    :func:`bluesky.preprocessors.fly_during_wrapper`
    :func:`bluesky.preprocessors.fly_during_decorator`
    """
    uid = yield from bps.open_run(md)


    for flyer in flyers:
        print(f'({ttime.ctime()}) {flyer.name} kickoff start')
        yield from bps.kickoff(flyer, wait=True)
        print(f'({ttime.ctime()}) {flyer.name} kickoff end')
    for flyer in flyers:
        print(f'({ttime.ctime()}) {flyer.name} complete start')
        yield from bps.complete(flyer, wait=True)
        print(f'({ttime.ctime()}) {flyer.name} complete end')
    for flyer in flyers:
        print(f'({ttime.ctime()}) {flyer.name} collect start')
        yield from bps.collect(flyer)
        print(f'({ttime.ctime()}) {flyer.name} collect end')
    print(f'({ttime.ctime()}) close_run start')
    yield from bps.close_run()
    print(f'({ttime.ctime()}) close_run end')
    return uid



def execute_trajectory_xs(name, **metadata):
    md = get_md_for_scan(name,
                         'fly_scan',
                         'execute_trajectory_xs',
                         'fly_energy_scan_xs3',
                         **metadata)
    md['aux_detector'] = 'XSpress3'
    yield from fly_local([flyer_xs], md=md)



def execute_trajectory_pil100k(name, **metadata):
    md = get_md_for_scan(name,
                         'fly_scan',
                         'execute_trajectory_pil100k',
                         'fly_energy_scan_pil100k',
                         **metadata)
    md['aux_detector'] = 'Pilatus100k'

    roi_data = [[pil100k.roi1.min_xyz.min_x.get(), pil100k.roi1.min_xyz.min_y.get(),
                 pil100k.roi1.size.x.get(),        pil100k.roi1.size.y.get()        ],
                [pil100k.roi2.min_xyz.min_x.get(), pil100k.roi2.min_xyz.min_y.get(),
                 pil100k.roi2.size.x.get(),        pil100k.roi2.size.y.get()        ],
                [pil100k.roi3.min_xyz.min_x.get(), pil100k.roi3.min_xyz.min_y.get(),
                 pil100k.roi3.size.x.get(),        pil100k.roi3.size.y.get()],
                [pil100k.roi4.min_xyz.min_x.get(), pil100k.roi4.min_xyz.min_y.get(),
                 pil100k.roi4.size.x.get(),        pil100k.roi4.size.y.get()]]
    md['roi'] = roi_data
    yield from bp.fly([flyer_pil], md=md)
'''

'''from bluesky.plan_patterns import spiral_square_pattern
import time as ttime
import numpy as np
import bluesky.plans as bp
from bluesky.plans import rel_spiral_square
from ophyd.sim import NullStatus


# def sample_spiral_scan():
#     detectors = [apb_ave]
#
#     return general_spiral_scan(detectors, giantxy.x, giantxy.y, 15, 15, 15, 15, time_step=0.1)

    # channels = [apb_ave.ch1, apb_ave.ch2, apb_ave.ch3, apb_ave.ch4]
    # offsets = [apb.ch1_offset, apb.ch2_offset, apb.ch3_offset, apb.ch4_offset, ]

    # plan = rel_spiral_square(detectors, giantxy.x, giantxy.y, 15, 15, 15, 15)

    # time_step = 0.1
    # samples = 250 * (np.ceil(time_step * 10443 / 250))  # hn I forget what that does... let's look into the new PB OPI
    # yield from bps.abs_set(apb_ave.sample_len, time_step*1e3, wait=True)
    # yield from bps.abs_set(apb_ave.wf_len, time_step*1e3, wait=True)
    # yield from bps.abs_set(apb_ave.divide, 374, wait=True)

    # if hasattr(detector, 'kickoff'):
    # plan_with_flyers = bpp.fly_during_wrapper(plan, [detectors])
    # uid = (yield from plan)
    # table = db[uid].table()
    # row_num = table[detector.volt.name].idxmin()
    # x_pos = table['giantxy_x'][row_num]
    # y_pos = table['giantxy_y'][row_num]

def general_spiral_scan(detectors_list, *, motor1=giantxy.x, motor2=giantxy.y, motor1_range=15, motor2_range=15, motor1_nsteps=15, motor2_nsteps=15, time_step=0.1, **kwargs):

    sys.stdout = kwargs.pop('stdout', sys.stdout)
    print(f'Dets {detectors_list}')
    print(f'Motors {motor1}, {motor2}')


    plan = rel_spiral_square(detectors_list, motor1, motor2,
                                        motor1_range, motor2_range, motor1_nsteps, motor2_nsteps,
                                        md={"plan_name": "spiral scan"})

    if apb_ave in detectors_list:
        print('Preparing pizzabox')
        cur_divide_value = apb_ave.divide.value
        cur_sample_len = apb_ave.sample_len.value
        cur_wf_len = apb_ave.wf_len.value

    print('[General Spiral Scan] Starting scan...')
    yield from bps.abs_set(apb_ave.divide, 374, wait=True)
    yield from bps.abs_set(apb_ave.sample_len, int(time_step * 1e3), wait=True)
    yield from bps.abs_set(apb_ave.wf_len, int(time_step * 1e3), wait=True)

    uid = (yield from plan)

    if apb_ave in detectors_list:
        print('Returning the pizzabox to its original state')
        yield from bps.abs_set(apb_ave.divide, cur_divide_value, wait=True)
        yield from bps.abs_set(apb_ave.sample_len, cur_sample_len, wait=True)
        yield from bps.abs_set(apb_ave.wf_len, cur_wf_len, wait=True)
    return uid






from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def get_mus():
    data = db[-1].table()
    x = data['giantxy_x']
    y = data['giantxy_y']

    mut = np.log(data['apb_ave_ch1_mean']/data['apb_ave_ch2_mean'])
    muf = data['apb_ave_ch4_mean']/data['apb_ave_ch1_mean']

    return x,y, mut, muf

def analyze_surface():
    x, y, mut, muf = get_mus()
    plot_xyz(x, y, mut)
    plot_xyz(x, y, muf)


def com(a_orig, w_orig, mask=None):
    a = a_orig.copy()
    w = w_orig.copy()
    if mask is not None:
        a = a[mask]
        w = w[mask]
    return np.sum(a * w)/np.sum(w)




def plot_xyz(x, y, z, r1=5, r2=(13.4/2-1)):
    fig = plt.figure()
    # ax = fig.gca(projection='3d')
    # ax.plot_trisurf(x, y, z, linewidth=0.2, antialiased=True, cmap=plt.cm.Spectral)
    ax = fig.gca()

    x_im_center = x.iloc[0]
    y_im_center = y.iloc[0]
    # R = r1 #13.4/2-1
    xy_mask = (np.sqrt(np.abs(x - x_im_center)**2 +
                       np.abs(y - y_im_center)**2) < r1)

    x_ho_com = com(x, z.max() - z, ~xy_mask)
    y_ho_com = com(y, z.max() - z, ~xy_mask)

    xy_mask_recen = (np.sqrt(np.abs(x - x_ho_com) ** 2 +
                             np.abs(y - y_ho_com) ** 2) < r2)

    # x_max = x[xy_mask_recen][np.argmax(z[xy_mask_recen])]
    # y_max = y[xy_mask_recen][np.argmax(z[xy_mask_recen])]

    x_max = com(x, (z - z.min())**2, xy_mask_recen)
    y_max = com(y, (z - z.min())**2, xy_mask_recen)

    ax.tricontourf(x, y, z, 50)
    ax.plot(x_im_center, y_im_center, 'ro', ms=25)
    ax.plot(x_ho_com, y_ho_com, 'bx', ms=25, markeredgewidth=5)
    ax.plot(x_max, y_max, 'm+', ms=25, markeredgewidth=5)

    # plt.plot(x[xy_mask], y[xy_mask], 'g.', alpha=0.5)
    # plt.plot(x[~xy_mask], y[~xy_mask], 'r.', alpha=0.5)
    # plt.show()


# class SnakeFlyer():
#     def __init__(self, det, pbs, motor_stage):
#         self.name = 'snake_flyer'
#         self.parent = None
#         self.det = det
#         self.pbs = pbs  # a list of passed pizza-boxes
#         self.motor_stage = motor_stage
#         self._motor_status = None
#         self.traj = None
#
#     def _motor_snaker(self, motor_x=None, range_x=None, motor_y=None, range_y=None):
#         """Snake tragectory for flyer.
#
#         :param motor_x: ophyd object for motor
#         :param range_x: range in motor units
#         :param motor_y: ophyd object for motor
#         :param range_y: range in motor units
#         :return: None
#         """
#
#         # Read start positions.
#         start_pos_x = motor_x.user_readback.get()
#         start_pos_y = motor_y.user_readback.get()
#
#         step = 1
#
#         # We need the grid scan here to get the tragectory.
#         plan = bp.rel_grid_scan([], motor_y, -range_y / 2, range_y / 2, (range_y / step + 1),
#                                 motor_x, -range_x / 2, range_x / 2, 2,
#                                 True  # snake=True
#                                 )
#
#         # This is adapted from plot_raster_scan in bluesky.
#         cur_x = cur_y = None
#         self.traj = []
#         for msg in plan:
#             cmd = msg.command
#             if cmd == 'set':
#                 if msg.obj.name == motor_x.name:
#                     cur_x = msg.args[0]
#                 if msg.obj.name == motor_y.name:
#                     cur_y = msg.args[0]
#             elif cmd == 'save':
#                 self.traj.append((cur_x, cur_y))
#
#         # Move motors along the trajectory.
#         for (x, y) in self.traj:
#             print(x, y)
#             if abs(motor_x.user_readback.get() - x) > 5e-3:
#                 print(f"Moving {motor_x.name}")
#                 # .move blocks the operation, and waits until the motor arrives to the target position.
#                 motor_x.move(x)
#             if abs(motor_y.user_readback.get() - y) > 5e-3:
#                 print(f"Moving {motor_y.name}")
#                 # .move blocks the operation, and waits until the motor arrives to the target position.
#                 motor_y.move(y)
#
#         # Move back to the original position both motors simultaneously.
#         self._motor_status = motor_x.set(start_pos_x)
#         self._motor_status &= motor_y.set(start_pos_y)
#
#     def kickoff(self, *args, **kwargs):
#         for pb in self.pbs:
#             pb.stage()
#             pb.kickoff()
#
#         self.det.stage()
#         # Start apb after encoder pizza-boxes, which will trigger the motor.
#         self.det.stream.set(1)
#
#         self._motor_snaker(motor_x=self.motor_stage.x, range_x=10, motor_y=self.motor_stage.y, range_y=4)
#
#         print(f"Motor status in kickoff: {self._motor_status}")
#
#         return NullStatus()
#
#     def complete(self):
#         print(f"Motor status in complete: {self._motor_status}")
#
#         def callback_det(value, old_value, **kwargs):
#             if int(round(old_value)) == 1 and int(round(value)) == 0:
#                 print(f'callback_det {ttime.ctime()}')
#                 return True
#             else:
#                 return False
#         streaming_st = SubscriptionStatus(self.det.streaming, callback_det)
#
#         def callback_motor():
#             print(f'callback_motor {ttime.ctime()}')
#
#             for pb in self.pbs:
#                 pb.complete()
#
#             # TODO: see if this set is still needed (also called in self.det.unstage())
#             self.det.stream.put(0)
#             self.det.complete()
#
#         self._motor_status.add_callback(callback_motor)
#
#         # Jdun!
#         return streaming_st & self._motor_status
#
#     def describe_collect(self):
#         return_dict = {self.det.name:
#                         {f'{self.det.name}': {'source': 'APB',
#                                               'dtype': 'array',
#                                               'shape': [-1, -1],
#                                               'filename_bin': self.det.filename_bin,
#                                               'filename_txt': self.det.filename_txt,
#                                               'external': 'FILESTORE:'}}}
#         # Also do it for all pizza-boxes
#         for pb in self.pbs:
#             return_dict[pb.name] = pb.describe_collect()[pb.name]
#
#         # Add a stream for the motor positions.
#         return_dict[self.motor_stage.name] = {f'{self.motor_stage.x.name}': {'source': 'SNAKE',
#                                                                              'dtype': 'number',
#                                                                              'shape': []},
#                                               f'{self.motor_stage.y.name}': {'source': 'SNAKE',
#                                                                              'dtype': 'number',
#                                                                              'shape': []}
#                                               }
#         return return_dict
#
#     def collect_asset_docs(self):
#         yield from self.det.collect_asset_docs()
#         for pb in self.pbs:
#             yield from pb.collect_asset_docs()
#
#     def collect(self):
#         print(f"Motor status in collect: {self._motor_status}")
#
#         self.det.unstage()
#         for pb in self.pbs:
#             pb.unstage()
#
#         def collect_all():
#             for pb in self.pbs:
#                 yield from pb.collect()
#             yield from self.det.collect()
#
#             # Collect docs for motor positions.
#             now = ttime.time()
#             for (x, y) in self.traj:
#                 data = {f"{self.motor_stage.x.name}": x,
#                         f"{self.motor_stage.y.name}": y}
#
#                 yield {'data': data,
#                        'timestamps': {key: now for key in data}, 'time': now,
#                        'filled': {key: False for key in data}}
#
#         return collect_all()
#
#
#
# snake_flyer = SnakeFlyer(det=apb_stream, pbs=[pb4.enc3, pb4.enc4], motor_stage=giantxy)'''



# def show_timing_results_from_previous_scan(idx):
#     hdr = db[idx]
#     # hdr.stream_names
#     data_apb = list(hdr.data(field='apb_stream', stream_name='apb_stream'))
#     data_epb = list(hdr.data(field='pb9_enc1', stream_name='pb9_enc1'))
#     plt.figure(1)
#     plt.clf()
#     plt.subplot(221)
#     plt.plot(data_apb[0]['timestamp'])
#     plt.plot(data_epb[0]['ts_s'] + data_epb[0]['ts_ns'] * 1e-9)
#     # return data_apb, data_epb
#
#     t_epb = data_epb[0]['ts_s'] + data_epb[0]['ts_ns'] * 1e-9
#     plt.subplot(222)
#     plt.plot(t_epb - t_epb.min(), data_epb[0]['encoder'])
#     # return (t_epb - t_epb.min()), data_epb[0]['encoder']



'''

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

def set_gains(i0_gain:int=5, it_gain:int=5, iff_gain:int=5,
                          ir_gain:int=5, hs:bool=False, **kwargs):
    sys.stdout = kwargs.pop('stdout', sys.stdout)
    i0_gain = int(i0_gain)
    it_gain = int(it_gain)
    iff_gain = int(iff_gain)
    ir_gain = int(ir_gain)
    if type(hs) == str:
        hs = hs == 'True'

    yield from set_gains_plan(i0_amp, i0_gain, hs, it_amp, it_gain, hs, iff_amp, iff_gain, hs, ir_amp, ir_gain, hs)


def tune_mono_pitch(scan_range, step, retries=1, ax=None):
    aver = pba1.adc7.averaging_points.get()
    pba1.adc7.averaging_points.put(10)
    num_points = int(round(scan_range / step) + 1)
    over = 0

    while (not over):
        RE(tune([pba1.adc7], hhm.pitch, -scan_range / 2, scan_range / 2, num_points, ''),
           LivePlot('pba1_adc7_volt', 'hhm_pitch', ax=ax))
        last_table = db.get_table(db[-1])
        min_index = np.argmin(last_table['pba1_adc7_volt'])
        hhm.pitch.move(last_table['hhm_pitch'][min_index])
        print(hhm.pitch.position)

        run = db[-1]
        os.remove(run['descriptors'][0]['data_keys'][run['descriptors'][0]['name']]['filename'])
        # for i in run['descriptors']:
        #        if 'devname' in i['data_keys'][i['name']]

        # os.remove(db[-1]['descriptors'][0]['data_keys']['pba1_adc7']['filename'])
        if (num_points >= 10):
            if (((min_index > 0.2 * num_points) and (min_index < 0.8 * num_points)) or retries == 1):
                over = 1
            if retries > 1:
                retries -= 1
        else:
            over = 1

    pba1.adc7.averaging_points.put(aver)
    print('Pitch tuning complete!')


def tune_mono_pitch_encoder(scan_range, step, retries=1, ax=None):
    aver = pba1.adc7.averaging_points.get()
    pba1.adc7.averaging_points.put(10)
    num_points = int(round(scan_range / step) + 1)
    over = 0

    start_position = pb2.enc3.pos_I.value

    while (not over):
        RE(tune([pba1.adc7, pb2.enc3], hhm.pitch, -scan_range / 2, scan_range / 2, 2, ''))

        enc = xasdata.XASdataAbs.loadENCtrace('', '', db[-1]['descriptors'][0]['data_keys']['pb2_enc3']['filename'])
        i0 = xasdata.XASdataAbs.loadADCtrace('', '', db[-1]['descriptors'][1]['data_keys']['pba1_adc7']['filename'])

        min_timestamp = np.array([i0[0, 0], enc[0, 0]]).max()
        max_timestamp = np.array([i0[len(i0) - 1, 0], enc[len(enc) - 1, 0]]).min()
        interval = i0[1, 0] - i0[0, 0]
        timestamps = np.arange(min_timestamp, max_timestamp, interval)
        enc_interp = np.array([timestamps, np.interp(timestamps, enc[:, 0], enc[:, 1])]).transpose()
        i0_interp = np.array([timestamps, np.interp(timestamps, i0[:, 0], i0[:, 1])]).transpose()
        len_to_erase = int(np.round(0.015 * len(i0_interp)))
        enc_interp = enc_interp[len_to_erase:]
        i0_interp = i0_interp[len_to_erase:]

        xas_abs.data_manager.process_equal(i0_interp[:, 0],
                                           enc_interp[:, 1],
                                           i0_interp[:, 1],
                                           i0_interp[:, 1],
                                           i0_interp[:, 1],
                                           10)

        xas_abs.data_manager.en_grid = xas_abs.data_manager.en_grid[5:-5]
        xas_abs.data_manager.i0_interp = xas_abs.data_manager.i0_interp[5:-5]
        # plt.plot(enc_interp[:,1], i0_interp[:,1]) #not binned

        plt.plot(xas_abs.data_manager.en_grid, xas_abs.data_manager.i0_interp)  # binned
        minarg = np.argmin(xas_abs.data_manager.i0_interp)
        enc_diff = xas_abs.data_manager.en_grid[minarg] - start_position

        pitch_pos = enc_diff / 204  # Enc to pitch convertion
        print('Delta Pitch = {}'.format(pitch_pos))
        # convert enc_diff to position (need to know the relation)
        # then move to the new position

        print(hhm.pitch.position)
        # os.remove(db[-1]['descriptors'][0]['data_keys']['pba1_adc7']['filename'])
        over = 1

    pba1.adc7.averaging_points.put(aver)
    print('Pitch tuning complete!')


def tune_mono_y(scan_range, step, retries=1, ax=None):
    aver = pba1.adc7.averaging_points.get()
    pba1.adc7.averaging_points.put(10)
    num_points = int(round(scan_range / step) + 1)
    over = 0

    while (not over):
        RE(tune([pba1.adc7], hhm.y, -scan_range / 2, scan_range / 2, num_points, ''),
           LivePlot('pba1_adc7_volt', 'hhm_y', ax=ax))
        last_table = db.get_table(db[-1])
        min_index = np.argmin(last_table['pba1_adc7_volt'])
        hhm.y.move(last_table['hhm_y'][min_index])
        print('New position: {}'.format(hhm.y.position))
        run = db[-1]
        os.remove(run['descriptors'][0]['data_keys'][run['descriptors'][0]['name']]['filename'])
        # os.remove(db[-1]['descriptors'][0]['data_keys']['pba1_adc7']['filename'])
        if (num_points >= 10):
            if (((min_index > 0.2 * num_points) and (min_index < 0.8 * num_points)) or retries == 1):
                over = 1
            if retries > 1:
                retries -= 1
        else:
            over = 1

    pba1.adc7.averaging_points.put(aver)
    print('Y tuning complete!')


def tune_mono_y_bpm(scan_range, step, retries=1, ax=None):
    num_points = int(round(scan_range / step) + 1)
    over = 0

    while (not over):
        RE(tune([bpm_fm], hhm.y, -scan_range / 2, scan_range / 2, num_points, ''),
           LivePlot('bpm_fm_stats1_total', 'hhm_y', ax=ax))
        last_table = db.get_table(db[-1])
        max_index = np.argmax(last_table['bpm_fm_stats1_total'])
        hhm.y.move(last_table['hhm_y'][max_index])
        print('New position: {}'.format(hhm.y.position))
        if (num_points >= 10):
            if (((max_index > 0.2 * num_points) and (max_index < 0.8 * num_points)) or retries == 1):
                over = 1
            if retries > 1:
                retries -= 1
        else:
            over = 1

    print('Y tuning complete!')



    def prep_trajectory(delay = 1):
    hhm.prepare_trajectory.put("1")
    while (hhm.trajectory_ready.value == 0):
        ttime.sleep(.1)
    while (hhm.trajectory_ready.value == 1):
        ttime.sleep(.1)
    ttime.sleep(delay)




def tscan(name: str, comment: str, n_cycles: int = 1, delay: float = 0, **kwargs):
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
    --------f
    :func:`tscanxia`
    """

    # uids = []
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
        uid = RE(execute_trajectory(name_n,delay,comment=comment))
        yield uid
        # hhm.prepare_trajectory.put('1')
        # uids.append(uid)
        # return uids



def tscancam(name: str, comment: str, n_cycles: int = 1, delay: float = 0, **kwargs):
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

    # uids = []
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
        # uids.append(uid)
        time.sleep(float(delay))
    print('Done!')
    # return uids




def tscanxia(name: str, comment: str, n_cycles: int = 1, delay: float = 0, **kwargs):
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

    # uids = []
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
        # uids.append(uid)
        time.sleep(float(delay))
    print('Done!')
    # return uids






def xia_step_scan(name:str, comment:str, e0:int=8333, preedge_start:int=-200, xanes_start:int=-50, xanes_end:int=30, exafs_end:int=16, preedge_spacing:float=10, xanes_spacing:float=0.2, exafs_spacing:float=0.04, **kwargs):

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

record(ao,"$(P)Ch1:User:Offset-SP"){
  field(DTYP,"Soft Channel")
  field(VAL,0)
  field(UDF,1)
}

record(ao,"$(P)Ch2:User:Offset-SP"){
  field(DTYP,"Soft Channel")
  field(VAL,0)
  field(UDF,1)
}

record(ao,"$(P)Ch3:User:Offset-SP"){
  field(DTYP,"Soft Channel")
  field(VAL,0)
  field(UDF,1)
}

record(ao,"$(P)Ch4:User:Offset-SP"){
  field(DTYP,"Soft Channel")
  field(VAL,0)
  field(UDF,1)
}
"ADC.db" [readonly] 584L, 11404C


    sys.stdout = kwargs.pop('stdout', sys.stdout)

    energy_grid, time_grid = generate_energy_grid(e0, preedge_start, xanes_start, xanes_end, exafs_end, preedge_spacing, xanes_spacing, exafs_spacing)
    positions_grid = xray.energy2encoder(energy_grid) / 360000

    ax = kwargs.get('ax')
    if ax is not None:
        uid, = RE(step_list_plan([xia1, i0, it, iff, ir], hhm.theta, positions_grid, name), LivePlot(xia1.mca1.roi0.sum.name, hhm.theta.name, ax=ax))
    else:
        uid, = RE(step_list_plan([xia1, i0, it, iff, ir], hhm.theta, positions_grid, name))

    path = '/GPFS/xf08id/User Data/{}.{}.{}/'.format(db[uid]['start']['year'], db[uid]['start']['cycle'], db[uid]['start']['proposal'])
    filename = parse_xia_step_scan(uid, name, path)

    ax.cla()
    plot_xia_step_scan(uid, ax=ax)

    print('Done!')
    return uid


def parse(db, uid):
    dataset = pd.DataFrame()
    hdr = db[uid]

    detectors = [pba1.adc6, pba1.adc1, pba2.adc6, pba1.adc7]

    channels = ['iff', 'it', 'ir', 'i0']
    for detector, channel in zip(detectors, channels):
        indx = 0
        spectrum = [];
        print(f'Detector {detector.name}')
        data = list(hdr.data(detector.name, stream_name=detector.name))
        for point in data:

            indx +=1
            print(f'We are at {indx}')
            adc = point['adc']
            try:
                adc = adc.apply(lambda x: (int(x, 16) >> 8) - 0x40000 if (int(x, 16) >> 8) > 0x1FFFF else int(x,
                                                                                                          16) >> 8) * 7.62939453125e-05
                mean_val = np.mean(adc)
            except:
                mena_val = 1
            spectrum.append(mean_val)
        dataset[channel] = np.array(spectrum)

    energies = np.array(hdr.start['plan_pattern_args']['object'])
    dataset['energy']= energies
    return dataset

def save_dataset(dataset, name):
    dataset.to_csv()


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

def wait_filter_in_place(status_pv):
    # for j in range(5):
    while True:
        ret = yield from bps.read(status_pv)
        if ret is None:
            break
        if ret[status_pv.name]['value'] == 1:
            break
        else:
            yield from bps.sleep(.1)
'''
def write_html_log(uuid, figure, log_path='/GPFS/xf08id/User Data/'):
    # Get needed data from db
    uuid = db[uuid]['start']['uid']

    if 'name' in db[uuid]['start']:
        scan_name = db[uuid]['start']['name']
    else:
        scan_name = 'General Scan'

    year = db[uuid]['start']['year']
    cycle = db[uuid]['start']['cycle']
    proposal = db[uuid]['start']['proposal']

    # Create dirs if they are not there
    if log_path[-1] != '/':
        log_path += '/'
    log_path = '{}{}.{}.{}/'.format(log_path, year, cycle, proposal)
    if(not os.path.exists(log_path)):
        os.makedirs(log_path)
        call(['setfacl', '-m', 'g:iss-staff:rwx', log_path])
        call(['chmod', '770', log_path])

    log_path = log_path + 'log/'
    if(not os.path.exists(log_path)):
        os.makedirs(log_path)
        call(['setfacl', '-m', 'g:iss-staff:rwx', log_path])
        call(['chmod', '770', log_path])

    snapshots_path = log_path + 'snapshots/'
    if(not os.path.exists(snapshots_path)):
        os.makedirs(snapshots_path)
        call(['setfacl', '-m', 'g:iss-staff:rwx', snapshots_path])
        call(['chmod', '770', snapshots_path])

    file_path = 'snapshots/{}.png'.format(scan_name)
    fn = log_path + file_path
    repeat = 1
    while(os.path.isfile(fn)):
        repeat += 1
        file_path = 'snapshots/{}-{}.png'.format(scan_name, repeat)
        fn = log_path + file_path

    # Save figure
    figure.savefig(fn)
    call(['setfacl', '-m', 'g:iss-staff:rw', fn])
    call(['chmod', '660', fn])

    # Create or update the html file
    relative_path = './' + file_path

    comment = ''
    if 'comment' in db[uuid]['start']:
        comment = db[uuid]['start']['comment']
    comment = '<p><b> Comment: </b> {} </p>'.format(comment)
    start_timestamp = db[uuid]['start']['time']
    stop_timestamp = db[uuid]['stop']['time']
    time_stamp_start='<p><b> Scan start: </b> {} </p>\n'.format(datetime.fromtimestamp(start_timestamp).strftime('%m/%d/%Y    %H:%M:%S'))
    time_stamp='<p><b> Scan complete: </b> {} </p>\n'.format(datetime.fromtimestamp(stop_timestamp).strftime('%m/%d/%Y    %H:%M:%S'))
    time_total='<p><b> Total time: </b> {} </p>\n'.format(datetime.fromtimestamp(stop_timestamp - start_timestamp).strftime('%M:%S'))
    uuid_html='<p><b> Scan ID: </b> {} </p>\n'.format(uuid)

    filenames = {}
    for i in db[uuid]['descriptors']:
        if i['name'] in i['data_keys']:
            if 'filename' in i['data_keys'][i['name']]:
                name = i['name']
                if 'devname' in i['data_keys'][i['name']]:
                    name = i['data_keys'][i['name']]['devname']
                filenames[name] = i['data_keys'][i['name']]['filename']

    fn_html = '<p><b> Files: </b></p>\n<ul>\n'
    for key in filenames.keys():
        fn_html += '  <li><b>{}:</b> {}</ln>\n'.format(key, filenames[key])
    fn_html += '</ul>\n'

    image = '<img src="{}" alt="{}" height="447" width="610">\n'.format(fn, scan_name)

    if(not os.path.isfile(log_path + 'log.html')):
        create_file = open(log_path + 'log.html', "w")
        create_file.write('<html> <body>\n</body> </html>')
        create_file.close()
        call(['setfacl', '-m', 'g:iss-staff:rw', log_path + 'log.html'])
        call(['chmod', '660', log_path + 'log.html'])

    text_file = open(log_path + 'log.html', "r")
    lines = text_file.readlines()
    text_file.close()

    text_file = open(log_path + 'log.html', "w")

    for indx,line in enumerate(lines):
        if indx is 1:
            text_file.write('<header><h2> {} </h2></header>\n'.format(scan_name))
            text_file.write(comment)
            text_file.write(uuid_html)
            text_file.write(fn_html)
            text_file.write(time_stamp_start)
            text_file.write(time_stamp)
            text_file.write(time_total)
            text_file.write(image)
            text_file.write('<hr>\n\n')
        text_file.write(line)
    text_file.close()


'''


def gauss(x, *p):
    A, mu, sigma = p
    return A*np.exp(-(x-mu)**2/(2.*sigma**2))


def xia_gain_matching(center_energy, scan_range, channel_number):

    graph_x = xia1.mca_x.value
    graph_data = getattr(xia1, "mca_array" + "{}".format(channel_number) + ".value")

    condition = (graph_x <= (center_energy + scan_range)/1000) == (graph_x > (center_energy - scan_range)/1000)
    interval_x = np.extract(condition, graph_x)
    interval = np.extract(condition, graph_data)

    # p0 is the initial guess for fitting coefficients (A, mu and sigma)
    p0 = [.1, center_energy/1000, .1]
    coeff, var_matrix = curve_fit(gauss, interval_x, interval, p0=p0)
    print('Intensity = ', coeff[0])
    print('Fitted mean = ', coeff[1])
    print('Sigma = ', coeff[2])

    # For testing (following two lines)
    plt.plot(interval_x, interval)
    plt.plot(interval_x, gauss(interval_x, *coeff))

    #return gauss(interval_x, *coeff)



def generate_xia_file(uuid, name, log_path='/GPFS/xf08id/Sandbox/', graph='xia1_graph3'):
    arrays = db.get_table(db[uuid])[graph]
    np.savetxt('/GPFS/xf08id/Sandbox/' + name, [np.array(x) for x in arrays], fmt='%i',delimiter=' ')



'''

# print(ttime.ctime() + ' >>>> ' + __file__)
'''

class ROI(Device):
    low = Cpt(EpicsSignal, 'LO')
    high = Cpt(EpicsSignal, 'HI')
    sum = Cpt(EpicsSignal, '')
    net = Cpt(EpicsSignal, 'N')
    label = Cpt(EpicsSignal, 'NM')


class MCA(Device):
    array = Cpt(EpicsSignal, '')
    roi0 = Cpt(ROI, '.R0')
    roi1 = Cpt(ROI, '.R1')
    roi2 = Cpt(ROI, '.R2')
    roi3 = Cpt(ROI, '.R3')
    roi4 = Cpt(ROI, '.R4')
    roi5 = Cpt(ROI, '.R5')
    roi6 = Cpt(ROI, '.R6')
    roi7 = Cpt(ROI, '.R7')
    roi8 = Cpt(ROI, '.R8')
    roi9 = Cpt(ROI, '.R9')
    roi10 = Cpt(ROI, '.R10')
    roi11 = Cpt(ROI, '.R11')


class XIA(Device):
    graph1 = Cpt(EpicsSignal, 'mca1.VAL')
    graph2 = Cpt(EpicsSignal, 'mca2.VAL')
    graph3 = Cpt(EpicsSignal, 'mca3.VAL')
    graph4 = Cpt(EpicsSignal, 'mca4.VAL')
    mode = Cpt(EpicsSignal, 'PresetMode')
    collect_mode = Cpt(EpicsSignal, 'CollectMode')
    start_sig = Cpt(EpicsSignal, 'StartAll')
    stop_sig = Cpt(EpicsSignal, 'StopAll')
    erase_start = Cpt(EpicsSignal, 'EraseStart')
    erase = Cpt(EpicsSignal, 'EraseAll')
    acquiring = Cpt(EpicsSignalRO, 'Acquiring')
    polarity = 'pos'

    capt_start_stop = Cpt(EpicsSignal, 'netCDF1:Capture_RBV', write_pv='netCDF1:Capture')
    write_mode = Cpt(EpicsSignal, 'netCDF1:FileWriteMode')
    pixels_per_run = Cpt(EpicsSignal, 'PixelsPerRun')
    current_pixel = Cpt(EpicsSignal, 'dxp1:CurrentPixel')
    next_pixel = Cpt(EpicsSignal, 'NextPixel')
    pix_per_buf_auto = Cpt(EpicsSignal, 'AutoPixelsPerBuffer')
    pix_per_buf_set = Cpt(EpicsSignal, 'PixelsPerBuffer')
    pix_per_buf_rb = Cpt(EpicsSignal, 'PixelsPerBuffer_RBV')

    pre_amp_gain1 = Cpt(EpicsSignal, 'dxp1:PreampGain')
    pre_amp_gain2 = Cpt(EpicsSignal, 'dxp2:PreampGain')
    pre_amp_gain3 = Cpt(EpicsSignal, 'dxp3:PreampGain')
    pre_amp_gain4 = Cpt(EpicsSignal, 'dxp4:PreampGain')
    pre_amp_gain5 = Cpt(EpicsSignal, 'dxp5:PreampGain')
    pre_amp_gain6 = Cpt(EpicsSignal, 'dxp6:PreampGain')
    pre_amp_gain7 = Cpt(EpicsSignal, 'dxp7:PreampGain')
    pre_amp_gain8 = Cpt(EpicsSignal, 'dxp8:PreampGain')
    pre_amp_gain9 = Cpt(EpicsSignal, 'dxp9:PreampGain')
    pre_amp_gain10 = Cpt(EpicsSignal, 'dxp10:PreampGain')
    pre_amp_gain11 = Cpt(EpicsSignal, 'dxp11:PreampGain')
    pre_amp_gain12 = Cpt(EpicsSignal, 'dxp12:PreampGain')
    pre_amp_gain13 = Cpt(EpicsSignal, 'dxp13:PreampGain')
    pre_amp_gain14 = Cpt(EpicsSignal, 'dxp14:PreampGain')
    pre_amp_gain15 = Cpt(EpicsSignal, 'dxp15:PreampGain')
    pre_amp_gain16 = Cpt(EpicsSignal, 'dxp16:PreampGain')

    real_time = Cpt(EpicsSignal, 'PresetReal')
    real_time_rb = Cpt(EpicsSignal, 'ElapsedReal')
    live_time = Cpt(EpicsSignal, 'PresetLive')
    live_time_rb = Cpt(EpicsSignal, 'ElapsedLive')

    mca1 = Cpt(MCA, 'mca1')
    mca2 = Cpt(MCA, 'mca2')
    mca3 = Cpt(MCA, 'mca3')
    mca4 = Cpt(MCA, 'mca4')
    mca5 = Cpt(MCA, 'mca5')
    mca6 = Cpt(MCA, 'mca6')
    mca7 = Cpt(MCA, 'mca7')
    mca8 = Cpt(MCA, 'mca8')
    mca9 = Cpt(MCA, 'mca9')
    mca10 = Cpt(MCA, 'mca10')
    mca11 = Cpt(MCA, 'mca11')
    mca12 = Cpt(MCA, 'mca12')
    mca13 = Cpt(MCA, 'mca13')
    mca14 = Cpt(MCA, 'mca14')
    mca15 = Cpt(MCA, 'mca15')
    mca16 = Cpt(MCA, 'mca16')

    mca_array1 = Cpt(EpicsSignal, 'mca1')
    mca_array2 = Cpt(EpicsSignal, 'mca2')
    mca_array3 = Cpt(EpicsSignal, 'mca3')
    mca_array4 = Cpt(EpicsSignal, 'mca4')
    mca_array5 = Cpt(EpicsSignal, 'mca5')
    mca_array6 = Cpt(EpicsSignal, 'mca6')
    mca_array7 = Cpt(EpicsSignal, 'mca7')
    mca_array8 = Cpt(EpicsSignal, 'mca8')
    mca_array9 = Cpt(EpicsSignal, 'mca9')
    mca_array10 = Cpt(EpicsSignal, 'mca10')
    mca_array11 = Cpt(EpicsSignal, 'mca11')
    mca_array12 = Cpt(EpicsSignal, 'mca12')
    mca_array13 = Cpt(EpicsSignal, 'mca13')
    mca_array14 = Cpt(EpicsSignal, 'mca14')
    mca_array15 = Cpt(EpicsSignal, 'mca15')
    mca_array16 = Cpt(EpicsSignal, 'mca16')

    mca_x = Cpt(EpicsSignal, 'dxp1:Graph0X.AVAL')
    mca_max_energy = Cpt(EpicsSignal, 'dxp1:Graph0High')

    netcdf_filename = Cpt(EpicsSignal, 'netCDF1:FileName')
    netcdf_filename_rb = Cpt(EpicsSignal, 'netCDF1:FileName_RBV')
    netcdf_filenumber = Cpt(EpicsSignal, 'netCDF1:FileNumber')
    netcdf_filenumber_rb = Cpt(EpicsSignal, 'netCDF1:FileNumber_RBV')

    def start_trigger(self):
        yield from bps.abs_set(pb4.do0.enable, 1, wait=True)

    def stop_trigger(self):
        yield from bps.abs_set(pb4.do0.enable, 0, wait=True)

    def start_mapping_scan(self):
        yield from bps.abs_set(self.collect_mode, 'MCA mapping', wait=True)
        yield from bps.sleep(.25)
        # ttime.sleep(0.25)
        yield from bps.abs_set(pb4.do0.dutycycle_sp, 50, wait=True)
        print('<<<<<<After setting PB4 & before setting capture<<<<<')
        yield from bps.abs_set(self.write_mode, 'Capture', wait=True)
        print('<<<<<After setting capture & before starting capturing<<<<<')
        yield from bps.abs_set(self.capt_start_stop, 1, wait=True)
        print("<<<<<After capturing<<<<<")
        # self.capt_start_stop.put(1)
        yield from bps.abs_set(self.erase_start, 1, wait=True)
        print("<<<<<After start<<<<<")
        # self.erase_start.put(1)
        yield from bps.sleep(1)
        # ttime.sleep(1)
        yield from bps.abs_set(pb4.do0.enable, 1, wait=True)
        # pb4.do0.enable.put(1) # Workaround
        return self._status

    def stop_scan(self):
        yield from bps.abs_set(pb4.do0.enable, 0, wait=True)
        while (pb4.do0.enable.value):
            pass
        # pb4.do0.enable.put(0) # Workaround
        yield from bps.sleep(1.5)
        yield from bps.abs_set(self.stop_sig, 1, wait=True)
        # self.stop_sig.put(1)
        yield from bps.sleep(0.5)
        # ttime.sleep(0.5)
        yield from bps.abs_set(self.capt_start_stop, 0, wait=True)
        # self.capt_start_stop.put(0)

    def __init__(self, *args, **kwargs):
        # link trigger to xia object
        if 'input_trigger' in kwargs:
            self.input_trigger = kwargs['input_trigger']  # pb4.do0
            del kwargs['input_trigger']
        super().__init__(*args, **kwargs)
        self.stage_sigs[self.mode] = 'Real time'
        self._status = None

        self.mcas = [self.mca1, self.mca2, self.mca3, self.mca4,
                     self.mca5, self.mca6, self.mca7, self.mca8,
                     self.mca9, self.mca10, self.mca11, self.mca12,
                     self.mca13, self.mca14, self.mca15, self.mca16]

    def goto_next_pixel(self):
        self.next_pixel.put(1)

    def stage(self):
        self.collect_mode.put('MCA spectra')
        self.acquiring.subscribe(self._acquiring_changed)
        # pass

    def unstage(self):
        self.acquiring.clear_sub(self._acquiring_changed)
        # pass

    # def read(self):

    def trigger(self):
        self._status = DeviceStatus(self)
        ttime.sleep(0.1)
        self.erase_start.put(1)
        # pb4.do0.enable.put(1) # Workaround
        return self._status

    def _acquiring_changed(self, value=None, old_value=None, **kwargs):
        "This is run every time the value of 'acquiring' changes."
        if self._status is None:
            # We have not triggered anything; ignore this one.
            return
        if (old_value == 1) and (value == 0):
            # 'acquiring' has flipped from 'Acquiring' to 'Done'.
            # pb4.do0.enable.put(0) # Workaround
            self._status._finished()

# xia1 = XIA('XF:08IDB-OP{XMAP}', name='xia1', input_trigger=pb4.do0)
#
# xia1.read_attrs = []
#
# for mca in xia1.mcas:
#     if mca.connected:
#         xia1.read_attrs.append(mca.name.split('_')[1])
#
# list1 = [mca.name for mca in xia1.mcas]
# list2 = ['roi{}'.format(number) for number in range(12)]
#
# xia_list = ['{}_{}_sum'.format(x,y) for x in list1 for y in list2]
'''
# b= [{'uid': '703cd5cd-697a-4615-a682-18c9725a50f4',
#   'first_name': 'Denis',
#   'last_name': 'Leshchev',
#   'affiliation': 'NSLS II',
#   'email': 'istavitski.bnl.gov'},
#  {'uid': '68efa620-f2b3-44bb-960d-e18c9edbf6fb',
#   'first_name': 'Eli',
#   'last_name': 'Stavitski',
#   'affiliation': 'NSLS II',
#   'email': 'istavitski.bnl.gov'},
#  {'uid': '56f68a95-4c9a-4af8-aeab-2104876ded14',
#   'first_name': 'Akhil',
#   'last_name': 'Tayal',
#   'affiliation': 'DESY',
#   'email': 'istavitski.bnl.gov'},
#  {'uid': 'f2bfb046-e560-4fe1-abf7-2aa2c9c5494c',
#   'first_name': 'Randall',
#   'last_name': 'Meyer',
#   'affiliation': 'Exxon Mobil',
#   'email': 'istavitski@bnl.gov'},
#  {'uid': '2c3b36a2-146f-4a51-97bf-ab8bf09b5e64',
#   'first_name': 'Dominik',
#   'last_name': 'Wirz',
#   'affiliation': 'Exxon Mobil',
#   'email': 'istavitski@bnl.gov'},
#  {'uid': 'a7e89794-d318-47ac-a237-3fea5011dcf6',
#   'first_name': 'Akhik',
#   'last_name': 'Tayal',
#   'affiliation': 'DESY',
#   'email': 'istavitski@bnl.gov',
#   'runs': [{'start': 'Mon Apr 24 13:40:56 2023',
#     'timestamp': 1682358056.8729405,
#     'proposal': 312089,
#     'saf': 310653,
#     'experimenters': []},
#    {'start': 'Mon Apr 24 13:48:25 2023',
#     'timestamp': 1682358505.8591702,
#     'proposal': 312089,
#     'saf': 310653,
#     'experimenters': []}]},
#  {'uid': 'e4cd48a8-5587-4add-8e2d-e2da37851a25',
#   'first_name': 'Carl',
#   'last_name': 'Richardson',
#   'affiliation': 'Toyama',
#   'email': 'istavitski@bnl.gov',
#   'sample_manager': 'bc38fe9.json',
#   'scan_manager': '2ac9de4.json',
#   'runs': [{'start': 'Mon Apr 24 13:52:43 2023',
#     'timestamp': 1682358763.243945,
#     'proposal': 312089,
#     'saf': 310653,
#     'experimenters': []}]}]

