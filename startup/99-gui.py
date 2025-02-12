print(ttime.ctime() + ' >>>> ' + __file__)
# from PyQt5.QtWidgets import QApplication
import atexit
import requests
import os
import sys
from requests.packages.urllib3.exceptions import InsecureRequestWarning
requests.packages.urllib3.disable_warnings(InsecureRequestWarning)
import time as ttime

# sample_stages = [{'x': giantxy.x.name, 'y': giantxy.y.name},
#                  {'x': samplexy.x.name, 'y': samplexy.y.name},
#                  {'x': huber_stage.z.name, 'y': huber_stage.y.name}]


#app = QApplication(sys.argv)

# processing_ioc_uid = EpicsSignal('XF:08IDB-Processing:pv_uid', name='process_ioc_uid')
processing_ioc_uid = None


if not os.environ.get('AZURE_TESTING'):
    from isstools import xlive

    xlive_gui = xlive.XliveGui(data_collection_plan_funcs=data_collection_plan_funcs,
                               service_plan_funcs=service_plan_funcs,
                               aux_plan_funcs=aux_plan_funcs,
                               scan_manager=scan_manager,
                               sample_manager=sample_manager,
                               scan_sequence_manager=scan_sequence_manager,
                               batch_manager=batch_manager,
                               user_manager=user_manager,
                               plan_processor=plan_processor,
                               RE=RE,
                               db=db,
                               processing_ioc_uid=processing_ioc_uid,
                               accelerator=nsls_ii,
                               front_end=front_end,
                               hhm=hhm,
                               hhm_encoder=hhm_encoder,
                               hhm_feedback=hhm_feedback,
                               trajectory_manager=trajectory_manager,
                               johann_emission=johann_emission,
                               johann_spectrometer_manager=johann_spectrometer_manager,
                               sdd=xs,
                               ge_detector=ge_detector,
                               inclinometers = [det_inclinometer1],
                               pil100k=pil100k,
                               apb=apb_ave,
                               apb_trigger_xs=apb_trigger,
                               apb_trigger_pil100k=apb_trigger_pil100k,
                               detector_dict=detector_dictionary,
                               shutter_dict=shutter_dictionary,
                               motor_dict=motor_dictionary,
                               camera_dict=camera_dictionary,
                               sample_env_dict=sample_env_dict,
                               # sample_stage=giantxy,
                               sample_stage=sample_stage,
                               tune_elements=tune_elements,
                               ic_amplifiers=ic_amplifiers,
                               print_to_gui=print_to_gui,
                               window_title="XLive @ISS/08-ID NSLS-II")


    def xlive():
        xlive_gui.show()

    xlive()

    print_to_gui(f'Startup complete at {ttime.ctime()}')
    #sys.exit(app.exec_())

    sys.stdout = xlive_gui.emitstream_out
    sys.stderr = xlive_gui.emitstream_err


    # def cleaning():
    #    if xlive_gui.widget_beamline_setup.piezo_thread.isRunning():
    #        xlive_gui.widget_beamline_setup.toggle_piezo_fb(0)
    #
    # atexit.register(cleaning)
