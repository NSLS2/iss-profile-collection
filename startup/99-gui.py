from isstools import xlive
import collections
import atexit
import requests
from requests.packages.urllib3.exceptions import InsecureRequestWarning
requests.packages.urllib3.disable_warnings(InsecureRequestWarning)

from bluesky.examples import motor
motor.move = motor.set

print("took {} sec".format(time.time()-t1))


detector_dictionary = {bpm_fm.name: {'obj': bpm_fm, 'elements': ['bpm_fm_stats1_total', 'bpm_fm_stats2_total']},
            bpm_cm.name: {'obj': bpm_cm, 'elements': ['bpm_cm_stats1_total','bpm_cm_stats2_total']},
            bpm_bt1.name: {'obj': bpm_bt1, 'elements': ['bpm_bt1_stats1_total','bpm_bt1_stats2_total']},
            bpm_bt2.name: {'obj': bpm_bt2, 'elements':['bpm_bt2_stats1_total','bpm_bt2_stats2_total']},
            bpm_es.name: {'obj': bpm_es, 'elements':['bpm_es_stats1_total','bpm_es_stats2_total']},
            bpm_sp3.name: {'obj': bpm_sp3, 'elements':['bpm_sp3_stats1_total','bpm_sp3_stats2_total']},
            pb9.enc1.name: {'obj': pb9.enc1, 'elements': ['pb9_enc1_pos_I']},
            it.name: {'obj': it, 'elements': ['pba1_adc1_volt']},
            iff.name: {'obj': iff, 'elements': ['pba1_adc6_volt']},
            i0.name: {'obj': i0, 'elements': ['pba1_adc7_volt']},
            ir.name: {'obj': ir, 'elements': ['pba2_adc6_volt']},
            pba2.adc7.name: {'obj': pba2.adc7, 'elements': ['pba2_adc7_volt']},
            xia1.name: {'obj': xia1, 'elements': xia_list}}

motors_dictionary = {'slits_v_gap': {'name': slits.v_gap.name, 'description':'B1 Slit Vertical Gap','object': slits.v_gap},
               'slits_v_pos': {'name': slits.v_pos.name, 'description':'B1 Slit Vertical Position','object': slits.v_pos},
               'slits_hor_in': {'name': slits.hor_in.name,'description':'B1 Slit Horisontal Inboard Position', 'object': slits.hor_in},
               'slits_hor_out': {'name': slits.hor_out.name,'description':'B1 Slit Horisontal Outboard Position', 'object': slits.hor_out},
               'samplexy_x': {'name': samplexy.x.name, 'description':'B2 Sample Stage X','object': samplexy.x},
               'samplexy_y': {'name': samplexy.y.name, 'description':'B2 Sample Stage Y','object': samplexy.y},
               'giantxy_x': {'name': giantxy.x.name, 'description':'B2 Giant Stage X','object': giantxy.x},
               'giantxy_y': {'name': giantxy.y.name, 'description':'B2 Giant Stage Y','object': giantxy.y},
               'hhm_theta': {'name': hhm.theta.name,'description':'A Monochromator Theta', 'object': hhm.theta},
               'hhm_energy': {'name': hhm.energy.name, 'description':'A Monochromator Energy','object': hhm.energy},
               'hhm_y': {'name': hhm.y.name,'description':'A Monochromator Y', 'object': hhm.y},
               'hhm_pitch': {'name': hhm.pitch.name, 'description':'A Monochromator Pitch','object': hhm.pitch},
               'hhm_roll': {'name': hhm.roll.name, 'description':'A Monochromator Roll', 'object': hhm.roll},
               #'hhrm_yu': {'name': hhrm.yu.name, 'object': hhrm.yu},
               #'hhrm_yd1': {'name': hhrm.yd1.name, 'object': hhrm.yd1},
               #'hhrm_yd2': {'name': hhrm.yd2.name, 'object': hhrm.yd2},
               'hhrm_mir_pitch': {'name': hhrm.mir_pitch.name, 'description':'B1 HHR Mirror Pitch','object': hhrm.mir_pitch},
               'hhrm_table_pitch': {'name': hhrm.table_pitch.name, 'description':'B1 HHR Mirror Table Pitch','object': hhrm.table_pitch},
               'hhrm_y': {'name': hhrm.y.name, 'description':'B1 HHR Mirror Table Height','object': hhrm.y},
               #'hrm_theta': {'name': hrm.theta.name, 'object': hrm.theta},
               #'hrm_pitch': {'name': hrm.pitch.name, 'object': hrm.pitch},
               #'hrm_y': {'name': hrm.y.name, 'object': hrm.y},
               'huber_stage_y': {'name': huber_stage.y.name,  'description':'B2 Huber Stage Y','object': huber_stage.y},
               'huber_stage_pitch': {'name': huber_stage.pitch.name, 'description':'B2 Huber Stage Pitch','object': huber_stage.pitch},
               'huber_stage_z': {'name': huber_stage.z.name, 'description':'B2 Huber Stage Z','object': huber_stage.z},
               'Dummy Motor': {'name': motor.name, 'description':'A dummy motor','object': motor},
#               'xbic_dac1': {'name': xbic.dac1.name, 'object': xbic.dac1},
#               'xbic_dac2': {'name': xbic.dac2.name, 'object': xbic.dac2}
               'six_axes_stage_x': {'name': six_axes_stage.x.name, 'description':'Six Axes Stage X', 'object': six_axes_stage.x},
               'six_axes_stage_y': {'name': six_axes_stage.y.name, 'description':'Six Axes Stage Y', 'object': six_axes_stage.y},
               'six_axes_stage_z': {'name': six_axes_stage.z.name, 'description':'Six Axes Stage Z', 'object': six_axes_stage.z},
               'six_axes_stage_pitch': {'name': six_axes_stage.pitch.name, 'description':'Six Axes Stage Pitch', 'object': six_axes_stage.pitch},
               'six_axes_stage_yaw': {'name': six_axes_stage.yaw.name, 'description':'Six Axes Stage Yaw', 'object': six_axes_stage.yaw},
               'six_axes_stage_roll': {'name': six_axes_stage.roll.name, 'description':'Six Axes Stage Roll', 'object': six_axes_stage.roll},
               'gonio_meter_th1': {'name': gonio_meter.th1.name, 'description':'Gonio Meter Theta1', 'object': gonio_meter.th1},
               'gonio_meter_th2': {'name': gonio_meter.th2.name, 'description':'Gonio Meter Theta2', 'object': gonio_meter.th2},
               'smedley_cathode_zeta': {'name': cathode_zeta.smedley_zeta.name, 'description':'Cathode Chamber Zeta', 'object': cathode_zeta.smedley_zeta},
               'smedley_cathode_trans_x': {'name': cathode_trans.smedley_x.name, 'description':'Cathode Chamber Trans X', 'object': cathode_trans.smedley_x},
               'smedley_cathode_trans_y': {'name': cathode_trans.smedley_y.name, 'description':'Cathode Chamber Trans Y', 'object': cathode_trans.smedley_y}
              }

sample_stages = [{'x': giantxy.x.name, 'y': giantxy.y.name},
                 {'x': samplexy.x.name, 'y': samplexy.y.name},
                 {'x': huber_stage.z.name, 'y': huber_stage.y.name}]

auto_tune = { 'pre_elements':[{'name' : bpm_fm.name,
                               'motor' : bpm_fm.ins,
                               'read_back' : bpm_fm.switch_insert,
                               'tries' : 3,
                               'value' : 1}
                             ],
              'post_elements':[{'name' : bpm_fm.name,
                                'motor' : bpm_fm.ret,
                                'read_back' : bpm_fm.switch_retract,
                                'tries' : 3,
                                'value' : 1}
                              ],
              'elements':[{'name' : hhm.pitch.name,
                           'object' : hhm.pitch,
                           'scan_range' : 5,
                           'step_size' : 0.025,#0.25,
                           'max_retries' : 3,#1,
                           'detector_name' : bpm_fm.name,
                           'detector_signame' : bpm_fm.stats1.total.name},
                           {'name' : hhm.y.name,
                           'object' : hhm.y,
                           'scan_range' : 1,
                           'step_size' : 0.025,#0.25,
                           'max_retries' : 3,#1,
                           'detector_name' : bpm_fm.name,
                           'detector_signame' : bpm_fm.stats1.total.name},
                           {'name' : hhrm.y.name,
                           'object' : hhrm.y,
                           'scan_range' : 3,
                           'step_size' : 0.1,#0.25,
                           'max_retries' : 3,#1,
                           'detector_name' : i0.dev_name.value,
                           'detector_signame' : i0.volt.name}
                          ]
           }

shutters_dictionary = collections.OrderedDict([(shutter_fe.name, shutter_fe),
                                         (shutter_ph.name, shutter_ph),
                                         (shutter.name, shutter)])

ic_amplifiers = {'i0_amp': i0_amp,
                 'it_amp': it_amp,
                 'ir_amp': ir_amp,
                 'iff_amp': iff_amp}

xlive_gui = xlive.XliveGui([tscan, tscanxia, tscancam, get_offsets, sleep_seconds],
                                 prep_traj_plan, 
                                 RE,
                                 db,
                                 nsls_ii,
                                 hhm,
                                 shutters_dictionary,
                                 detector_dictionary,
                                 motors_dictionary,
                                 general_scan,
                                 set_foil_reference,
                                 write_html_log = write_html_log,
                                 auto_tune_elements = auto_tune,
                                 ic_amplifiers = ic_amplifiers,
                                 set_gains_offsets = set_gains_and_offsets,
                                 prepare_bl = [prepare_bl_plan, prepare_bl_def],
                                 sample_stages = sample_stages,
                                 processing_sender = sender,
                                 job_submitter=job_submitter,
                                 bootstrap_servers=['cmb01:9092', 'cmb02:9092'],
                                 kafka_topic="iss-processing", 
                                 window_title="XLive @ISS/08-ID NSLS-II",
                                 )


def xlive():
    xlive_gui.show()




#xview_gui = xview.XviewGui(hhm.pulses_per_deg, db=db)

#def xview():
    #xview_gui.show()

xlive()
print('Startup complete')

sys.stdout = xlive_gui.emitstream_out
sys.stderr = xlive_gui.emitstream_err


#def cleaning():
#    if xlive_gui.piezo_thread.isRunning():
#        xlive_gui.toggle_piezo_fb(0)

#atexit.register(cleaning)
