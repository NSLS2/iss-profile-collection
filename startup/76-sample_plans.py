print(ttime.ctime() + ' >>>> ' + __file__)



def move_sample_stage_plan(sample_coordinates : dict = {}):
    for motor_key, position in sample_coordinates.items():
        motor = getattr(sample_stage, motor_key)
        yield from bps.mv(motor, position)

def move_rel_sample_stage_plan(sample_coordinates : dict = {}):
    for motor_key, shift in sample_coordinates.items():
        motor = getattr(sample_stage, motor_key)
        yield from bps.mvr(motor, shift)


# OUTPUT = {}


from xas.image_analysis import find_points_moving_with_stage
def calibrate_sample_cameras_plan(cameras=[camera_sp1, camera_sp2], dx=5, dy=5, target_max_counts=200, npoly=1):
    dxs = [+dx/2, -dx/2, -dx/2, +dx/2]
    dys = [-dy/2, -dy/2, +dy/2, +dy/2]
    x0 = sample_stage.x.position
    y0 = sample_stage.y.position
    xs = x0 + np.array(dxs)
    ys = y0 + np.array(dys)

    cam_images = []

    current_exposure_times = [c.exp_time.get() for c in cameras]

    for camera in cameras:
        camera.adjust_camera_exposure_time_full_image(target_max_counts=target_max_counts)

    for _x, _y in zip(xs, ys):
        yield from bps.mv(sample_stage.x, _x, sample_stage.y, _y)
        yield from bps.sleep(0.01)
        for i, camera in enumerate(cameras):
            img = camera.get_image_array_data_reshaped()
            cam_images.append(img)

    yield from bps.mv(sample_stage.x, x0, sample_stage.y, y0)

    for camera, exp_time in zip(cameras, current_exposure_times):
        camera.exp_time.put(exp_time)

    ncams = len(cameras)
    for i, camera in enumerate(cameras):
        # OUTPUT[camera.name] = cam_images[i::ncams]
        calibration_data_dict = find_points_moving_with_stage(dxs, dys, cam_images[i::ncams])
        calibration_data_dict['npoly'] = npoly
        camera.set_calibration(calibration_data_dict)









def random_donut_shift(v):
    random_v = 2 * v * (random() - 0.5)
    random_v = random_v * 0.5 + 0.5 * np.sign(random_v)
    return random_v

def move_sample_by_random_xy_step(x: float = 0, y: float = 0):

    '''
    This plan will move the stage randomly by a random number between
    x/2 and x  and y/2 and y, sampling a donut around the original point
    '''
    # sys.stdout = kwargs.pop('stdout', sys.stdout)
    print_to_gui('Executing random move')


    random_x = random_donut_shift(x)
    random_y = random_donut_shift(y)
    yield from move_rel_sample_stage_plan({'x' : random_x, 'y' : random_y})
    # yield from mvr(motor_x,random_x,motor_y,random_y)
