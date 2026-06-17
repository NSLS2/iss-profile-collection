from databroker import Broker
db = Broker.named("iss")
# Not critical, acceptable since this is only a test
db.v2.context.http_client.headers['tiled-qos'] = 'acquisition'

import sys
sys.path.insert(0, '/home/xf08id/Repos/workflows')
import interpolation

data = dict()
data['requester'] = "xf08id-ws02"
#data['uid'] = "55f14401-8c60-4474-a24e-62b7722c933c"
data['uid'] = db[-1].start['uid']

store =data.copy()
signal = None
context = None

interpolation.create_req_func(data, store, signal, context)
interpolation.process_run_func(data, store, signal, context)
