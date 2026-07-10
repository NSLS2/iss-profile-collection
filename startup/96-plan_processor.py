print(ttime.ctime() + ' >>>> ' + __file__)


class PlanProcessor(PersistentListInteractingWithGUI):
    liveplot_maker = None
    gui_services_dict = None
    plan_list_update_signal = None
    status_update_signal = None
    add_plans_question_box=None

    def __init__(self, json_file_path = f'{ROOT_PATH_SHARED}/settings/json/plan_processor.json'):
        super().__init__(json_file_path, boot_fresh=True)

        self.logger = self.get_logger()
        self.RE = RE
        self.scan_manager = scan_manager
        # self.plan_list = []
        self.status = 'idle'

        # self.check_valves = False
        # self.check_shutters = False
        self.load_settings()


    def get_logger(self):
        # Setup beamline specifics:

        logger = logging.getLogger('xas_re_logger')
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

        # only add handlers if not added before
        if not len(logger.handlers):
            logger.setLevel(logging.DEBUG)
            # Write DEBUG and INFO messages to /var/log/data_processing_worker/debug.log.
            debug_file = logging.handlers.RotatingFileHandler(f'{ROOT_PATH_SHARED}/log/data_collection_debug.log',
                maxBytes=10000000, backupCount=9)
            debug_file.setLevel(logging.DEBUG)
            debug_file.setFormatter(formatter)
            logger.addHandler(debug_file)
        return logger

    def load_settings(self):
        from redis_json_dict import RedisJSONDict
        self._settings_store = RedisJSONDict(redis_settings_client, prefix='plan_processor_settings')
        try:
            self.settings_dict = self._settings_store['settings']
        except KeyError:
            self.settings_dict = {'beamline_readiness': False,
                                  'auto_foil_set': False}

    def save_settings(self):
        self._settings_store['settings'] = self.settings_dict

    @property
    def beamline_readiness(self):
        return self.settings_dict['beamline_readiness']

    @property
    def auto_foil_set(self):
        return self.settings_dict['auto_foil_set']

    @beamline_readiness.setter
    def beamline_readiness(self, value):
        self.settings_dict['beamline_readiness'] = value
        self.save_settings()

    @auto_foil_set.setter
    def auto_foil_set(self, value):
        self.settings_dict['auto_foil_set'] = value
        self.save_settings()

    #class specific property
    @property
    def plan_list(self):
        return self.items

    @plan_list.setter
    def plan_list(self, value):
        self.items = value

    @plan_list.deleter
    def plan_list(self):
        del self.items

    @emit_list_update_signal_decorator
    def reset(self, *args, **kwargs):
        super().reset(*args, **kwargs)
        self.update_status('idle')

    # dealing with RE
    @property
    def RE_state(self):
        return self.RE.state

    @property
    def RE_is_running(self):
        return self.RE_state == 'running'

    # def append_gui_plan_list_update_signal(self, signal):
    #     self.plan_list_update_signal = signal

    # delaing with GUI interactions
    def append_gui_status_update_signal(self, signal):
        self.status_update_signal = signal

    def append_liveplot_maker(self, liveplot_maker):
        self.liveplot_maker = liveplot_maker

    def make_liveplots(self, plan, plan_kwargs):
        if self.liveplot_maker is not None:
            return self.liveplot_maker(plan, plan_kwargs)
        else:
            return []

    def append_gui_services_dict(self, gui_services_dict):
        self.gui_services_dict = gui_services_dict

    def get_gui_services_kwargs(self, plan_info):
        gui_services_kwargs = {}
        if 'plan_gui_services' in plan_info.keys():
            for service_key in plan_info['plan_gui_services']:
                service_dict = self.gui_services_dict[service_key]
                kwarg_name = service_dict['kwarg_name']
                kwarg_value = service_dict['kwarg_value']
                gui_services_kwargs[kwarg_name] = kwarg_value
            # return gui_services_kwargs
        return gui_services_kwargs

    # plan misc
    def plan_status_at_index(self, idx):
        try:
            return self.plan_list[idx]['plan_status']
        except IndexError:
            return 'normal'

    @emit_list_update_signal_decorator
    def set_plan_status_at_index(self, index, status):
        self.plan_list[index]['plan_status'] = status

    @property
    def top_plan_status(self):
        return self.plan_status_at_index(0)

    @property
    def last_plan_status(self):
        return self.plan_status_at_index(-1)

    @property
    def top_plan_executing(self):
        return self.top_plan_status == 'executing'

    def describe_plan(self, plan_dict):
        if 'plan_description' not in plan_dict.keys():
            plan_dict['plan_description'] = generate_plan_description(plan_dict['plan_name'], plan_dict['plan_kwargs'])

    # dealing with plan adding and converting to RE-digestible pieces
    @emit_list_update_signal_decorator
    def _add_plans(self, plans, add_at='tail', idx=None):
        success = True
        if type(plans) != list:
            plans = [plans]

        # plans = self.deal_with_bundles_in_plan_list(plans)

        for plan_dict in plans:
            self.describe_plan(plan_dict)

        if add_at == 'tail':
            idx = len(self.plan_list) - 1
            _plan_status = self.last_plan_status
            if _plan_status == 'executing': _plan_status = 'normal'
            for plan_dict in plans:
                # self.plan_list.append({'plan_info' : plan_dict, 'plan_status' : _plan_status})
                self.add_item({'plan_info': plan_dict, 'plan_status': _plan_status}, emit_signal=False)
        elif add_at == 'head':
            _plan_status = self.top_plan_status
            if _plan_status == 'executing': _plan_status = 'normal'
            idx = self.smallest_index
            for plan_dict in plans[::-1]: # [::-1] is needed so insert doesn't revert the list order
                self.insert_item_at_index(idx, {'plan_info': plan_dict, 'plan_status': _plan_status}, emit_signal=False)
        elif add_at == 'index':
            _plan_status = self.plan_status_at_index(idx)
            if _plan_status == 'executing': _plan_status = 'normal'
            for plan_dict in plans[::-1]: # [::-1] is needed so insert doesn't revert the list order
                self.insert_item_at_index(idx, {'plan_info': plan_dict, 'plan_status': _plan_status}, emit_signal=False)
        if len(plans) == 0:
            success = False
        return (idx + len(plans)), success

    def append_add_plans_question_box_func(self, func):
        self.add_plans_question_box = func

    def add_plans(self, plans, add_at='tail', idx=None, pause_after=False):
        if (len(self.plan_list) > 0) and (self.add_plans_question_box is not None):
            plans, add_at, idx, pause_after = self.add_plans_question_box(plans, add_at, idx, pause_after)
        last_added_plan_index, success = self._add_plans(plans, add_at, idx)
        if pause_after:
            self.pause_after_index(last_added_plan_index)
        return success


    def plan_is_a_bundle(self, idx):
        plan_info = self.plan_list[idx]['plan_info']
        plan_name = plan_info['plan_name']
        plan_func = all_plan_funcs[plan_name]
        if type(plan_func) == dict:
            if plan_func['kind'] == 'bundle':
                return True
        return False

    def unwrap_bundle(self, idx):
        plan_info = self.plan_list[idx]['plan_info']
        plan_name = plan_info['plan_name']
        plan_func = all_plan_funcs[plan_name]
        bundle_kwargs = plan_info['plan_kwargs']
        bundle_gui_services = self.get_gui_services_kwargs(plan_info)
        bundle_all_kwargs = {**bundle_kwargs, **bundle_gui_services}
        plan_bundle = plan_func['func'](**bundle_all_kwargs)
        self.plan_list.pop(idx)
        self._add_plans(plan_bundle, add_at='index', idx=idx)


    def make_re_args(self, idx):
        plan_info = self.plan_list[idx]['plan_info']
        plan_name = plan_info['plan_name']
        plan_func = all_plan_funcs[plan_name]
        plan_kwargs = plan_info['plan_kwargs']
        liveplots = self.make_liveplots(plan_name, plan_kwargs)
        gui_services_kwargs = self.get_gui_services_kwargs(plan_info)
        plan_all_kwargs = {**plan_kwargs, **gui_services_kwargs}
        print_to_gui(f'Execution of plan {plan_name} starting...', add_timestamp=True, tag='Queue')
        if len(liveplots) == 0:
            return (plan_func(**plan_all_kwargs), )
        else:
            return (plan_func(**plan_all_kwargs), liveplots)

    # execution of the queue
    @emit_list_update_signal_decorator
    def execute_top_plan(self):
        if self.plan_is_a_bundle(0):
            self.unwrap_bundle(0)
            self.execute_top_plan()
        else:
            re_args = self.make_re_args(0)
            # self.RE(actuate_photon_shutter_plan('Open'))
            for i in range(15):
                if self.RE_state == 'idle':
                    self.set_plan_status_at_index(0, 'executing')
                    self.perform_pre_scan_routines()
                    self.RE(*re_args)
                    self.plan_list.pop(0)
                    break
                else:
                    print_to_gui(f'RE is busy. waiting...', add_timestamp=True, tag='Queue')
                    ttime.sleep(0.1)

    def run(self, unpause=True):
        if unpause:
            self.unpause_plan_list()

        while len(self.plan_list) > 0:

            if self.top_plan_status == 'normal':
                self.update_status('running')
                self.execute_top_plan()
                # try:
                #     self.execute_top_plan()
                # except Exception as e:
                #     print(e)
                #     print('Found and issue with the top plan. Stopping the queue')
                #     break

            elif self.top_plan_status == 'paused':
                break

        # self.unpause_plan_list()
        self.update_status('idle')

    def run_if_idle(self, unpause=False):
        if self.status == 'idle':
            self.run(unpause=unpause)

    def make_plan_dict_from_name_and_kwargs(self, plan_name, plan_kwargs, plan_gui_services=None):
        plan_dict = {'plan_name': plan_name, 'plan_kwargs': plan_kwargs}
        if plan_gui_services is not None:
            plan_dict['plan_gui_services'] = plan_gui_services
        return plan_dict

    def add_plan_and_run_if_idle(self, plan_name, plan_kwargs, plan_gui_services=None):
        plan_dict = self.make_plan_dict_from_name_and_kwargs(plan_name, plan_kwargs,
                                                             plan_gui_services=plan_gui_services)
        success = self.add_plans(plan_dict)
        if success:
            self.run_if_idle()

    # #### LEGACY FUNCTION
    # def add_execute_pause_plan_at_head(self, plan_name, plan_kwargs, plan_gui_services=None):
    #     plan_dict = self.make_plan_dict_from_name_and_kwargs(plan_name, plan_kwargs,
    #                                                          plan_gui_services=plan_gui_services)
    #     self.add_plans(plan_dict, add_at='head')
    #     idx = self.plan_list.index({'plan_info' : plan_dict, 'plan_status' : 'normal'})
    #     self.pause_after_index(idx+1)
    #     self.run_if_idle(unpause=False)
    # ####


    def update_status(self, status):
        self.status = status
        self._emit_status_update_signal()

    @property
    def smallest_index(self):
        if (self.status == 'running') or self.RE_is_running:
            return 1
        else:
            return 0

    def pause_plan_list(self):
        index = self.smallest_index
        self.pause_after_index(index)

    def pause_after_index(self, index):
        any_plan_status_changed = False # to reduce unnecessary emitting
        for i in range(index, len(self.plan_list)):
            if self.plan_status_at_index(i) == 'normal':
                self.plan_list[i]['plan_status'] = 'paused'
                any_plan_status_changed = True
        if any_plan_status_changed:
            self.emit_list_update_signal()

    def unpause_plan_list(self):
        self.unpause_before_index(len(self.plan_list))

    def unpause_before_index(self, index):
        any_plan_status_changed = False # to reduce unnecessary emitting
        for i in range(index):
            if self.plan_list[i]['plan_status'] == 'paused':
                self.plan_list[i]['plan_status'] = 'normal'
                any_plan_status_changed = True
        if any_plan_status_changed:
            self.emit_list_update_signal()

    @emit_list_update_signal_decorator
    def clear_plan_list(self):
        idx = self.smallest_index
        for i in range(idx, len(self.plan_list)):
            self.plan_list.pop(idx)

    def _emit_status_update_signal(self):
        if self.status_update_signal is not None:
            self.status_update_signal.emit()

    def append_liveplot_func_dictionary(self, liveplot_funcs):
        self.liveplot_funcs = liveplot_funcs

    def perform_pre_scan_routines(self, idx=0):
        if self.beamline_readiness: # this also checks FE shutter
            self.RE(check_hhm_roll_plan())
            self.RE(check_gate_valves_plan())
            self.RE(check_photon_shutter_plan())
            self.RE(check_ic_voltages())
        if self.auto_foil_set:
            plan_info = self.plan_list[idx]['plan_info']
            plan_kwargs = plan_info['plan_kwargs']
            if ('element' in plan_kwargs) and (plan_kwargs['element'] != '') and ('edge' in plan_kwargs) and (plan_kwargs['edge'] != '') :
                self.RE(set_reference_foil(plan_kwargs['element'],plan_kwargs['edge'] ))

plan_processor = PlanProcessor()


