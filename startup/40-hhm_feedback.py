print(ttime.ctime() + ' >>>> ' + __file__)

# hhm feedback

from piezo_feedback.piezo_fb import PiezoFeedback
from PyQt5.QtCore import QThread

machine_name = os.uname()[1]
if 'ws1' in machine_name:
    local_hostname = 'ws01'
elif 'ws2' in machine_name:
    local_hostname = 'ws02'
elif 'ws3' in machine_name:
    local_hostname = 'ws03'
else:
    raise ValueError('This machine does not support local feedback')

class MockPiezoFeedback:
    status = False
    shutters_open = False
    host = None
    fb_heartbeat = Signal(name='fb_heartbeat', value=0)

class _PiezoFeedback(PiezoFeedback):
    def __init__(self):
        super().__init__(hhm,
                         bpm_es,
                         {shutter_fe.name: shutter_fe,
                          shutter_ph.name: shutter_ph},
                         local_hostname=local_hostname)


class PiezoFeedbackThread(QThread, _PiezoFeedback):
    def __init__(self):
        super().__init__()


if shutdown:
    # use this when # use this when softioc-piE712-02.service is down:
    hhm_feedback = MockPiezoFeedback()
else:
    # use this when # use this when softioc-piE712-02.service is up:
    hhm_feedback = PiezoFeedbackThread()
    hhm_feedback.start()

