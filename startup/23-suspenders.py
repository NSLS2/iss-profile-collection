
from bluesky.suspenders import (SuspendBoolHigh,
                                SuspendBoolLow,
                                SuspendFloor,
                                SuspendCeil,
                                SuspendInBand, SuspendOutBand)


fe_shut_suspender = SuspendBoolHigh(shutter_fe.state, sleep=10*60)
ph_shut_suspender = SuspendBoolHigh(shutter_ph.state, sleep=10*60)


# suspender for beamline current is mA
beam_current_suspender = SuspendFloor(nsls_ii.beam_current,
                                      suspend_thresh = 300, sleep = 10*60)
suspenders = [fe_shut_suspender,
              ph_shut_suspender,
              beam_current_suspender,
              ]

''' Some help on suspenders /bluesky
# how to add a suspender:
# Method 1:
# RE.install_suspender(fe_shut_suspender)
# Method 2 (in the plan):
# RE(bpp.suspend_wrapper(myplan(), [suspenders]))




# general bluesky info
# blue sky plans (mostly) reside here:
# general plans
import bluesky.plans as bp
# (bp.count())
# components of plans
import bluesky.plan_stubs as bps
# (bps.mov())
# plan modifiers
import bluesky.preprocessors as bpp
# (bpp.stage_wrapper(myplan, det))
'''
